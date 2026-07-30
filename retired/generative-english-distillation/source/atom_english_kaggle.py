"""Train, resume, evaluate, and sample the generative Atom English model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from atom_english_core import (
    ATOM_ENGLISH_CORE_RUNTIME,
    AtomCausalLanguageModel,
    AtomEnglishConfig,
    atom_english_context_expansion_plan,
    atom_english_core_self_test,
    atom_english_profile,
    expand_atom_english_context,
)
from atom_english_context import (
    ATOM_ENGLISH_CONTEXT_RUNTIME,
    CONTEXT_TARGET_OPTIMIZER_STEPS,
    CONTEXT_TRAINING_LENGTHS,
    atom_english_context_self_test,
    train_atom_english_context_stage,
)
from atom_english_data import (
    DialogueEnglishStream,
    PackedEnglishStream,
    atom_english_data_self_test,
    broad_english_curriculum,
    chat_token_ids,
    load_aligned_tokenizer,
    stream_stage_sources,
    tokenizer_manifest,
    validate_tokenizer_alignment,
)
from atom_english_evaluation import (
    atom_english_evaluation_self_test,
    collect_long_context_distractor_tokens,
    evaluate_blimp,
    evaluate_hellaswag,
    evaluate_ifeval,
    evaluate_lambada,
    evaluate_long_context,
    write_language_evaluation,
)
from atom_english_knowledge import (
    ATOM_ENGLISH_RAG_RUNTIME,
    ATOM_ENGLISH_WIKI_RUNTIME,
    build_english_knowledge_graph,
    retrieve_english_knowledge,
)
from atom_english_side_view import (
    ATOM_ENGLISH_SIDE_VIEW_RUNTIME,
    render_english_generation_artifact,
    write_english_generation_side_view,
)
from atom_english_training import (
    AtomTrainingConfig,
    atom_english_training_self_test,
    evaluate_perplexity,
    load_atom_english_checkpoint,
    load_teacher,
    train_atom_english_stage,
)

ATOM_ENGLISH_KAGGLE_RUNTIME = "atom-english-kaggle-runner-v1"
FOUNDATION_TARGET_TOKENS = 4_915_200_000
SMOLTALK_TRAIN_ROWS = 1_043_917
DIALOGUE_EPOCHS = 2
DIALOGUE_TARGET_OPTIMIZER_STEPS = 130_490
BLIMP_DATASET_ID = "nyu-mll/blimp"
BLIMP_DATASET_REVISION = "877fba0801ffb7cbd8c39c1ff314a46f053f6036"
HELLASWAG_DATASET_ID = "Rowan/hellaswag"
HELLASWAG_DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
IFEVAL_DATASET_ID = "google/IFEval"
IFEVAL_DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit_runtime_event(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
                "event": event,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_runtime_knowledge_and_side_view(
    output_directory: Path,
    model: AtomCausalLanguageModel,
    curriculum: Any,
    summary: dict[str, Any],
) -> None:
    graph = build_english_knowledge_graph(
        model.config,
        curriculum,
        run_summary=summary,
    )
    contexts = retrieve_english_knowledge(
        graph,
        "English causal graph tokenizer corpus teacher evaluation run",
        limit=8,
    )
    summary["knowledge_runtime"] = {
        "wiki": ATOM_ENGLISH_WIKI_RUNTIME,
        "rag": ATOM_ENGLISH_RAG_RUNTIME,
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "retrieved_contexts": len(contexts),
    }
    summary["side_view_runtime"] = ATOM_ENGLISH_SIDE_VIEW_RUNTIME
    _write_json(output_directory / "english_knowledge_graph.json", graph)
    _write_json(output_directory / "english_rag_context.json", contexts)
    write_english_generation_side_view(
        output_directory / "atom_english_generation_side_view.html",
        summary,
        graph,
    )


def _device(require_accelerator: bool) -> torch.device:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        architecture = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if supported and architecture not in supported:
            raise RuntimeError(
                "installed PyTorch does not contain kernels for "
                f"{architecture}; supported architectures are "
                f"{sorted(supported)}"
            )
        return torch.device("cuda")
    if require_accelerator:
        raise RuntimeError("the requested accelerator is unavailable")
    return torch.device("cpu")


def _precision(device: torch.device, requested: str) -> str:
    if requested not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("precision must be auto, float32, float16, or bfloat16")
    if requested != "auto":
        if device.type == "cpu" and requested == "float16":
            raise ValueError("float16 execution is unsupported on CPU")
        return requested
    if device.type != "cuda":
        return "float32"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def resolve_training_resume_config(
    stage: str,
    checkpoint_stage: str,
    source_config: AtomEnglishConfig,
    target_config: AtomEnglishConfig,
) -> tuple[AtomEnglishConfig, dict[str, Any] | None]:
    """Keep foundation/dialogue resumes stable and defer long-context migration."""

    if stage not in {"foundation", "dialogue"}:
        raise ValueError("resume configuration applies to foundation or dialogue")
    if checkpoint_stage not in {"foundation", "dialogue", "context"}:
        raise ValueError("checkpoint training stage is invalid")
    if source_config == target_config:
        return source_config, None
    plan = atom_english_context_expansion_plan(source_config, target_config)
    plan.update(
        {
            "deferred_until_stage": "context",
            "optimizer_state_preserved": checkpoint_stage == stage,
        }
    )
    return source_config, plan


def discover_input_checkpoint(
    stage: str,
    *,
    input_root: Path = Path("/kaggle/input"),
) -> Path | None:
    """Find the strongest compatible checkpoint among attached inputs."""

    if stage not in {"foundation", "dialogue", "context", "evaluation"}:
        raise ValueError("unknown checkpoint discovery stage")
    if not input_root.is_dir():
        return None
    candidates: list[tuple[int, int, int, str, Path]] = []
    for manifest_path in input_root.rglob("checkpoint_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                continue
            if manifest.get("architecture") != ATOM_ENGLISH_CORE_RUNTIME:
                continue
            training_config = manifest["training_config"]
            training_state = manifest["training_state"]
            candidate_stage = str(training_config["stage"])
            if candidate_stage not in {"foundation", "dialogue", "context"}:
                continue
            if stage == "foundation" and candidate_stage != "foundation":
                continue
            if stage == "dialogue" and candidate_stage == "context":
                continue
            if stage == "context" and candidate_stage == "foundation":
                continue
            if stage == "evaluation" and candidate_stage != "context":
                continue
            same_stage = int(
                candidate_stage == stage
                or (stage == "evaluation" and candidate_stage == "context")
            )
            consumed_tokens = int(training_state["consumed_tokens"])
            optimizer_step = int(training_state["optimizer_step"])
            manifest_hash = str(manifest["manifest_hash"])
            if len(manifest_hash) != 64:
                continue
            candidates.append(
                (
                    same_stage,
                    consumed_tokens,
                    optimizer_step,
                    manifest_hash,
                    manifest_path.parent,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[0], item[1], item[2], item[3]),
        reverse=True,
    )
    return candidates[0][4]


def validate_training_stage_admission(
    stage: str,
    checkpoint: Mapping[str, Any] | None,
) -> None:
    """Prevent post-training from starting before broad pretraining is mature."""

    if stage not in {"foundation", "dialogue", "context"}:
        raise ValueError("unknown training stage")
    if stage == "foundation":
        return
    if checkpoint is None:
        raise ValueError(f"{stage} training requires an earlier compatible checkpoint")
    checkpoint_stage = str(checkpoint["training_config"]["stage"])
    if checkpoint_stage not in {"foundation", "dialogue", "context"}:
        raise ValueError("checkpoint training stage is invalid")
    if stage == "dialogue" and checkpoint_stage == "context":
        raise ValueError("dialogue training cannot resume from a context checkpoint")
    if stage == "context":
        if checkpoint_stage not in {"dialogue", "context"}:
            raise ValueError("context training requires dialogue maturity")
        if (
            checkpoint_stage == "dialogue"
            and int(checkpoint["training_state"]["optimizer_step"])
            < DIALOGUE_TARGET_OPTIMIZER_STEPS
        ):
            raise ValueError(
                "dialogue checkpoint has not reached its optimizer-step target"
            )
        return
    if (
        checkpoint_stage == "foundation"
        and int(checkpoint["training_state"]["consumed_tokens"])
        < FOUNDATION_TARGET_TOKENS
    ):
        raise ValueError("foundation checkpoint has not reached the full token target")


def _tokenizer_bundle(
    curriculum: Any,
    *,
    local_files_only: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    foundation = load_aligned_tokenizer(
        curriculum.foundation_tokenizer_id,
        revision=curriculum.foundation_tokenizer_revision,
        local_files_only=local_files_only,
    )
    dialogue = load_aligned_tokenizer(
        curriculum.tokenizer_id,
        revision=curriculum.tokenizer_revision,
        local_files_only=local_files_only,
    )
    alignment = validate_tokenizer_alignment(foundation, dialogue)
    manifest = {
        "schema_version": 1,
        "foundation": tokenizer_manifest(
            foundation,
            tokenizer_id=curriculum.foundation_tokenizer_id,
            revision=curriculum.foundation_tokenizer_revision,
        ),
        "dialogue": tokenizer_manifest(
            dialogue,
            tokenizer_id=curriculum.tokenizer_id,
            revision=curriculum.tokenizer_revision,
        ),
        "alignment": alignment,
    }
    return foundation, dialogue, manifest


def _new_model(profile: str, dialogue_tokenizer: Any) -> AtomCausalLanguageModel:
    config = atom_english_profile(
        profile,
        vocab_size=len(dialogue_tokenizer),
        bos_token_id=int(dialogue_tokenizer.bos_token_id),
        eos_token_id=int(dialogue_tokenizer.eos_token_id),
        pad_token_id=int(dialogue_tokenizer.pad_token_id),
    )
    return AtomCausalLanguageModel(config)


def _training_batches(
    stage: str,
    curriculum: Any,
    foundation_tokenizer: Any,
    dialogue_tokenizer: Any,
    *,
    sequence_length: int,
    maximum_steps: int,
    batch_size: int,
    gradient_accumulation: int,
    seed: int,
    shuffle_buffer: int,
) -> DataLoader:
    sources = curriculum.stage(stage)
    source = sources[0]
    rows = stream_stage_sources(
        sources,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
        epochs=1 if stage == "foundation" else DIALOGUE_EPOCHS,
    )
    if stage == "foundation":
        tokens = maximum_steps * batch_size * gradient_accumulation * sequence_length
        dataset = PackedEnglishStream(
            rows,
            foundation_tokenizer,
            text_field=source.content_field,
            sequence_length=sequence_length,
            maximum_tokens=tokens,
            minimum_characters=source.minimum_characters,
            seed=seed,
        )
    else:
        examples = maximum_steps * batch_size * gradient_accumulation
        dataset = DialogueEnglishStream(
            rows,
            dialogue_tokenizer,
            messages_field=source.content_field,
            sequence_length=sequence_length,
            maximum_examples=examples,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def generate_language_samples(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    *,
    device: torch.device,
    maximum_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = (
        "Explain in plain English why a glass can crack when boiling water is poured into it.",
        "Write a short story about a botanist who discovers a plant that remembers music.",
        "What is the difference between correlation and causation?",
        "Rewrite this clearly: The process failed because its inputs were inconsistent.",
        "Give a careful argument for and against exploring an unknown cave.",
        "Continue this paragraph naturally: The rain stopped just before dawn, and",
        "Describe how you would learn a new concept from three conflicting explanations.",
        "Summarize the purpose of memory in an intelligent system.",
    )
    model.eval()
    samples: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        token_ids = chat_token_ids(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        available = model.config.max_seq_len - maximum_new_tokens
        token_ids = token_ids[-available:]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(
            0
        )
        generated = model.generate(
            input_ids,
            max_new_tokens=maximum_new_tokens,
            temperature=0.75,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.08,
            seed=20260724 + index,
        )
        response_ids = generated[0, input_ids.shape[1] :].tolist()
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        samples.append(
            {
                "prompt": prompt,
                "response": response,
                "prompt_tokens": int(input_ids.shape[1]),
                "response_tokens": len(response_ids),
            }
        )
    return samples


def _external_benchmarks(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from datasets import get_dataset_config_names, load_dataset

    curriculum = broad_english_curriculum()
    evaluation_sources = {
        source.source_id: source for source in curriculum.stage("evaluation")
    }
    wiki = evaluation_sources["wikitext-103-heldout"]
    wiki_rows = load_dataset(
        wiki.dataset_id,
        wiki.dataset_config,
        split=wiki.split,
        revision=wiki.revision,
        streaming=True,
    )
    sequence_length = min(512, model.config.max_seq_len)
    wiki_stream = PackedEnglishStream(
        wiki_rows,
        tokenizer,
        text_field=wiki.content_field,
        sequence_length=sequence_length,
        maximum_tokens=(
            None if maximum_examples is None else maximum_examples * sequence_length
        ),
        minimum_characters=wiki.minimum_characters,
        seed=20260724,
    )
    wiki_batches = DataLoader(wiki_stream, batch_size=4, num_workers=0)
    wiki_limit = 100_000 if maximum_examples is None else max(1, maximum_examples // 4)
    wikitext = evaluate_perplexity(
        model,
        wiki_batches,
        device=device,
        maximum_batches=wiki_limit,
        precision=precision,
    )

    lambada_source = evaluation_sources["lambada-openai-heldout"]
    lambada_rows = load_dataset(
        lambada_source.dataset_id,
        lambada_source.dataset_config,
        split=lambada_source.split,
        revision=lambada_source.revision,
        streaming=True,
    )
    lambada = evaluate_lambada(
        model,
        tokenizer,
        lambada_rows,
        device=device,
        maximum_examples=maximum_examples,
    )

    blimp_id = BLIMP_DATASET_ID
    blimp_configs = get_dataset_config_names(
        blimp_id,
        revision=BLIMP_DATASET_REVISION,
    )
    blimp_rows = {
        name: load_dataset(
            blimp_id,
            name,
            split="train",
            revision=BLIMP_DATASET_REVISION,
            streaming=True,
        )
        for name in blimp_configs
    }
    per_config = None
    if maximum_examples is not None:
        per_config = max(1, maximum_examples // max(len(blimp_configs), 1))
    blimp = evaluate_blimp(
        model,
        tokenizer,
        blimp_rows,
        device=device,
        maximum_examples_per_config=per_config,
    )

    hellaswag_id = HELLASWAG_DATASET_ID
    hellaswag_rows = load_dataset(
        hellaswag_id,
        split="validation",
        revision=HELLASWAG_DATASET_REVISION,
        streaming=True,
    )
    hellaswag = evaluate_hellaswag(
        model,
        tokenizer,
        hellaswag_rows,
        device=device,
        maximum_examples=maximum_examples,
    )
    ifeval_rows = load_dataset(
        IFEVAL_DATASET_ID,
        split="train",
        revision=IFEVAL_DATASET_REVISION,
        streaming=True,
    )
    ifeval = evaluate_ifeval(
        model,
        tokenizer,
        ifeval_rows,
        device=device,
        maximum_examples=maximum_examples,
    )
    long_context_rows = load_dataset(
        wiki.dataset_id,
        wiki.dataset_config,
        split=wiki.split,
        revision=wiki.revision,
        streaming=True,
    )
    distractor_tokens = collect_long_context_distractor_tokens(
        tokenizer,
        long_context_rows,
    )
    long_context = evaluate_long_context(
        model,
        tokenizer,
        distractor_tokens,
        device=device,
    )
    benchmarks = {
        "wikitext": wikitext,
        "lambada": lambada,
        "blimp": blimp,
        "hellaswag": hellaswag,
        "ifeval": ifeval,
        "long_context": long_context,
    }
    sources = {
        "curriculum": [asdict(source) for source in curriculum.stage("evaluation")],
        "blimp": blimp_id,
        "blimp_revision": BLIMP_DATASET_REVISION,
        "blimp_configs": blimp_configs,
        "hellaswag": hellaswag_id,
        "hellaswag_revision": HELLASWAG_DATASET_REVISION,
        "ifeval": IFEVAL_DATASET_ID,
        "ifeval_revision": IFEVAL_DATASET_REVISION,
        "long_context_protocol": "atom-ruler-style-context-evaluation-v1",
        "long_context_distractor": wiki.dataset_id,
        "long_context_distractor_revision": wiki.revision,
    }
    return benchmarks, sources


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = _device(args.require_accelerator)
    precision = _precision(device, args.precision)
    curriculum = broad_english_curriculum()
    foundation_tokenizer, dialogue_tokenizer, tokenizers = _tokenizer_bundle(
        curriculum,
        local_files_only=args.local_files_only,
    )
    resume_directory = (
        Path(args.resume) if args.resume else discover_input_checkpoint(args.stage)
    )
    desired_config = atom_english_profile(
        args.profile,
        vocab_size=len(dialogue_tokenizer),
        bos_token_id=int(dialogue_tokenizer.bos_token_id),
        eos_token_id=int(dialogue_tokenizer.eos_token_id),
        pad_token_id=int(dialogue_tokenizer.pad_token_id),
    )
    deferred_context_expansion: dict[str, Any] | None = None
    if resume_directory is not None:
        model, checkpoint = load_atom_english_checkpoint(
            resume_directory,
            device=device,
        )
        if model.config.vocab_size != len(dialogue_tokenizer):
            raise ValueError("resume model and tokenizer vocabularies differ")
        model_config, deferred_context_expansion = resolve_training_resume_config(
            args.stage,
            checkpoint["training_config"]["stage"],
            model.config,
            desired_config,
        )
        if model.config != model_config:
            raise RuntimeError("resume configuration unexpectedly replaced the model")
    else:
        model = _new_model(args.profile, dialogue_tokenizer)
        checkpoint = None
    validate_training_stage_admission(args.stage, checkpoint)
    sequence_length = args.sequence_length
    if sequence_length is None:
        sequence_length = 1024 if args.stage == "foundation" else 2048
    if sequence_length > model.config.max_seq_len:
        raise ValueError("sequence length exceeds the model context")
    maximum_steps = args.maximum_steps
    if maximum_steps is None:
        maximum_steps = (
            300_000 if args.stage == "foundation" else DIALOGUE_TARGET_OPTIMIZER_STEPS
        )
    teacher_id = args.teacher_model_id or (
        curriculum.base_teacher_id
        if args.stage == "foundation"
        else curriculum.dialogue_teacher_id
    )
    teacher_revision = args.teacher_revision or (
        curriculum.base_teacher_revision
        if args.stage == "foundation"
        else curriculum.dialogue_teacher_revision
    )
    device_details: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        device_details.update(
            {
                "device_name": torch.cuda.get_device_name(),
                "device_capability": list(torch.cuda.get_device_capability()),
                "supported_architectures": torch.cuda.get_arch_list(),
            }
        )
    _emit_runtime_event(
        "run_initialized",
        stage=args.stage,
        profile=args.profile,
        model_parameters=model.parameter_count(),
        corpora=[
            {
                "dataset_id": source.dataset_id,
                "configuration": source.dataset_config,
                "weight": source.weight,
            }
            for source in curriculum.stage(args.stage)
        ],
        teacher_model_id=teacher_id,
        teacher_revision=teacher_revision,
        precision=precision,
        maximum_steps=maximum_steps,
        maximum_wall_seconds=args.maximum_wall_seconds,
        sequence_length=sequence_length,
        gradient_accumulation=args.gradient_accumulation,
        resume_directory=(
            str(resume_directory) if resume_directory is not None else None
        ),
        context_expansion=None,
        deferred_context_expansion=deferred_context_expansion,
        **device_details,
    )
    teacher = load_teacher(
        teacher_id,
        revision=teacher_revision,
        device=device,
        precision=precision,
        local_files_only=args.local_files_only,
    )
    _emit_runtime_event(
        "teacher_loaded",
        teacher_model_id=teacher_id,
        teacher_parameters=sum(parameter.numel() for parameter in teacher.parameters()),
    )
    batches = _training_batches(
        args.stage,
        curriculum,
        foundation_tokenizer,
        dialogue_tokenizer,
        sequence_length=sequence_length,
        maximum_steps=maximum_steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        seed=(
            args.seed
            + (
                int(checkpoint["training_state"]["optimizer_step"]) * 17
                if checkpoint is not None
                else 0
            )
        ),
        shuffle_buffer=args.shuffle_buffer,
    )
    training_config = AtomTrainingConfig(
        stage=args.stage,
        teacher_model_id=teacher_id,
        teacher_revision=teacher_revision,
        maximum_wall_seconds=args.maximum_wall_seconds,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        maximum_steps=maximum_steps,
        warmup_steps=min(args.warmup_steps, maximum_steps - 1),
        precision=precision,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    report = train_atom_english_stage(
        model,
        batches,
        teacher=teacher,
        config=training_config,
        output_directory=output_directory,
        tokenizer_manifest=tokenizers,
        curriculum_manifest=curriculum.to_dict(),
        device=device,
        resume_directory=(
            resume_directory
            if (
                resume_directory is not None
                and checkpoint is not None
                and checkpoint["training_config"]["stage"] == args.stage
            )
            else None
        ),
    )
    samples = generate_language_samples(
        model,
        dialogue_tokenizer,
        device=device,
        maximum_new_tokens=args.sample_tokens,
    )
    _write_json(output_directory / "language_samples.json", samples)
    summary = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
        "mode": "train",
        "stage": args.stage,
        "profile": args.profile,
        "parameter_count": model.parameter_count(),
        "precision": precision,
        "teacher_model_id": teacher_id,
        "teacher_revision": teacher_revision,
        "training_report": report,
        "resumed_manifest_hash": (checkpoint["manifest_hash"] if checkpoint else None),
        "context_expansion": None,
        "deferred_context_expansion": deferred_context_expansion,
        "resume_kind": (
            "same-stage"
            if (
                checkpoint is not None
                and checkpoint["training_config"]["stage"] == args.stage
            )
            else ("stage-transition" if checkpoint is not None else "new")
        ),
        "tokenizers": tokenizers,
        "samples": samples,
    }
    _write_runtime_knowledge_and_side_view(
        output_directory,
        model,
        curriculum,
        summary,
    )
    _write_json(output_directory / "run_summary.json", summary)
    _emit_runtime_event(
        "run_artifacts_written",
        output_directory=str(output_directory),
        checkpoint_manifest_hash=report["checkpoint_manifest_hash"],
        sample_count=len(samples),
    )
    return summary


def run_context_training(args: argparse.Namespace) -> dict[str, Any]:
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = _device(args.require_accelerator)
    precision = _precision(device, args.precision)
    curriculum = broad_english_curriculum()
    _, dialogue_tokenizer, tokenizers = _tokenizer_bundle(
        curriculum,
        local_files_only=args.local_files_only,
    )
    resume_directory = (
        Path(args.resume) if args.resume else discover_input_checkpoint("context")
    )
    if resume_directory is None:
        raise ValueError("context training requires a dialogue checkpoint")
    model, checkpoint = load_atom_english_checkpoint(
        resume_directory,
        device=device,
    )
    desired_config = atom_english_profile(
        args.profile,
        vocab_size=len(dialogue_tokenizer),
        bos_token_id=int(dialogue_tokenizer.bos_token_id),
        eos_token_id=int(dialogue_tokenizer.eos_token_id),
        pad_token_id=int(dialogue_tokenizer.pad_token_id),
    )
    context_expansion: dict[str, Any] | None = None
    if model.config != desired_config:
        source_context = model.config.max_seq_len
        source_offsets = model.config.graph_offsets
        model = expand_atom_english_context(model, desired_config)
        context_expansion = {
            "source_context_tokens": source_context,
            "target_context_tokens": desired_config.max_seq_len,
            "preserved_causal_distances": list(source_offsets),
            "initialized_causal_distances": sorted(
                set(desired_config.graph_offsets) - set(source_offsets)
            ),
            "optimizer_state_reset": True,
        }
    validate_training_stage_admission("context", checkpoint)

    from datasets import load_dataset

    evaluation_sources = {
        source.source_id: source for source in curriculum.stage("evaluation")
    }
    wiki = evaluation_sources["wikitext-103-heldout"]
    distractor_rows = load_dataset(
        wiki.dataset_id,
        wiki.dataset_config,
        split=wiki.split,
        revision=wiki.revision,
        streaming=True,
    )
    distractor_tokens = collect_long_context_distractor_tokens(
        dialogue_tokenizer,
        distractor_rows,
    )
    maximum_steps = (
        CONTEXT_TARGET_OPTIMIZER_STEPS
        if args.maximum_steps is None
        else args.maximum_steps
    )
    training_config = AtomTrainingConfig(
        stage="context",
        teacher_model_id="deterministic-long-context-curriculum",
        teacher_revision=ATOM_ENGLISH_CONTEXT_RUNTIME,
        maximum_wall_seconds=args.maximum_wall_seconds,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=args.context_learning_rate,
        maximum_steps=maximum_steps,
        warmup_steps=min(args.warmup_steps, maximum_steps - 1),
        precision=precision,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        hard_label_weight=1.0,
        distillation_weight=0.0,
        seed=args.seed,
    )
    _emit_runtime_event(
        "context_run_initialized",
        profile=args.profile,
        model_parameters=model.parameter_count(),
        context_ceiling=model.config.max_seq_len,
        training_lengths=list(CONTEXT_TRAINING_LENGTHS),
        maximum_steps=maximum_steps,
        precision=precision,
        resumed_manifest_hash=checkpoint["manifest_hash"],
        context_expansion=context_expansion,
        distractor_dataset=wiki.dataset_id,
        distractor_revision=wiki.revision,
    )
    report = train_atom_english_context_stage(
        model,
        dialogue_tokenizer,
        distractor_tokens,
        config=training_config,
        output_directory=output_directory,
        tokenizer_manifest=tokenizers,
        curriculum_manifest=curriculum.to_dict(),
        device=device,
        resume_directory=(
            resume_directory
            if (
                checkpoint["training_config"]["stage"] == "context"
                and context_expansion is None
            )
            else None
        ),
        trainable_tail_tokens=args.context_tail_tokens,
        differentiable_context_limit=args.context_differentiable_tokens,
    )
    samples = generate_language_samples(
        model,
        dialogue_tokenizer,
        device=device,
        maximum_new_tokens=args.sample_tokens,
    )
    _write_json(output_directory / "language_samples.json", samples)
    summary = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
        "mode": "train",
        "stage": "context",
        "profile": args.profile,
        "parameter_count": model.parameter_count(),
        "precision": precision,
        "training_report": report,
        "resumed_manifest_hash": checkpoint["manifest_hash"],
        "context_expansion": context_expansion,
        "context_training_lengths": list(CONTEXT_TRAINING_LENGTHS),
        "tokenizers": tokenizers,
        "samples": samples,
    }
    _write_runtime_knowledge_and_side_view(
        output_directory,
        model,
        curriculum,
        summary,
    )
    _write_json(output_directory / "run_summary.json", summary)
    return summary


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    resume_directory = (
        Path(args.resume) if args.resume else discover_input_checkpoint("evaluation")
    )
    if resume_directory is None:
        raise ValueError("evaluation requires --resume checkpoint directory")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = _device(args.require_accelerator)
    precision = _precision(device, args.precision)
    curriculum = broad_english_curriculum()
    _, dialogue_tokenizer, tokenizers = _tokenizer_bundle(
        curriculum,
        local_files_only=args.local_files_only,
    )
    model, checkpoint = load_atom_english_checkpoint(
        resume_directory,
        device=device,
    )
    if checkpoint["training_config"]["stage"] != "context":
        raise ValueError("evaluation requires a context-conditioned checkpoint")
    benchmarks, sources = _external_benchmarks(
        model,
        dialogue_tokenizer,
        device=device,
        precision=precision,
        maximum_examples=args.evaluation_examples,
    )
    report = write_language_evaluation(
        output_directory,
        benchmarks,
        model_parameter_count=model.parameter_count(),
        checkpoint_manifest_hash=checkpoint["manifest_hash"],
        sources=sources,
    )
    samples = generate_language_samples(
        model,
        dialogue_tokenizer,
        device=device,
        maximum_new_tokens=args.sample_tokens,
    )
    _write_json(output_directory / "language_samples.json", samples)
    summary = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
        "mode": "evaluate",
        "checkpoint_manifest_hash": checkpoint["manifest_hash"],
        "precision": precision,
        "evaluation": report,
        "tokenizers": tokenizers,
        "samples": samples,
    }
    _write_runtime_knowledge_and_side_view(
        output_directory,
        model,
        curriculum,
        summary,
    )
    _write_json(output_directory / "run_summary.json", summary)
    return summary


def run_self_test() -> dict[str, Any]:
    checks = {
        "core": atom_english_core_self_test(),
        "context": atom_english_context_self_test(),
        "data": atom_english_data_self_test(),
        "training": atom_english_training_self_test(),
        "evaluation": atom_english_evaluation_self_test(),
    }
    config = atom_english_profile(
        "verification",
        vocab_size=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AtomCausalLanguageModel(config)
    curriculum = broad_english_curriculum()
    summary = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
        "mode": "train",
        "parameter_count": model.parameter_count(),
        "training_report": {"training_state": {"consumed_tokens": 32}},
        "samples": [
            {
                "prompt": "Explain a causal graph.",
                "response": "A directed graph records what can affect what.",
                "prompt_tokens": 5,
                "response_tokens": 10,
            }
        ],
    }
    graph = build_english_knowledge_graph(config, curriculum, run_summary=summary)
    contexts = retrieve_english_knowledge(graph, "causal graph English corpus", limit=4)
    document = render_english_generation_artifact(summary, graph)
    checks["knowledge_and_side_view"] = {
        "wiki_nodes_present": len(graph["nodes"]) > 10,
        "rag_context_present": len(contexts) > 0,
        "artifact_output_visible": (
            "A directed graph records what can affect what." in document
        ),
        "side_view_binding_present": ("render_english_generation_artifact" in document),
    }
    passed = all(all(section.values()) for section in checks.values())
    return {
        "runtime": ATOM_ENGLISH_KAGGLE_RUNTIME,
        "checks": checks,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_mode = os.environ.get("ATOM_ENGLISH_DEFAULT_MODE", "train")
    default_stage = os.environ.get(
        "ATOM_ENGLISH_DEFAULT_STAGE",
        "foundation",
    )
    if default_mode not in {"train", "evaluate", "self-test"}:
        raise ValueError("ATOM_ENGLISH_DEFAULT_MODE is invalid")
    if default_stage not in {"foundation", "dialogue", "context"}:
        raise ValueError("ATOM_ENGLISH_DEFAULT_STAGE is invalid")
    default_sequence_text = os.environ.get("ATOM_ENGLISH_DEFAULT_SEQUENCE_LENGTH")
    default_sequence_length = (
        int(default_sequence_text) if default_sequence_text is not None else None
    )
    default_gradient_accumulation = int(
        os.environ.get("ATOM_ENGLISH_DEFAULT_GRADIENT_ACCUMULATION", "16")
    )
    if default_sequence_length is not None and default_sequence_length < 1:
        raise ValueError("ATOM_ENGLISH_DEFAULT_SEQUENCE_LENGTH is invalid")
    if default_gradient_accumulation < 1:
        raise ValueError("ATOM_ENGLISH_DEFAULT_GRADIENT_ACCUMULATION is invalid")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "self-test"),
        default=default_mode,
    )
    parser.add_argument(
        "--stage",
        choices=("foundation", "dialogue", "context"),
        default=default_stage,
    )
    parser.add_argument("--profile", default="scale-227m")
    parser.add_argument(
        "--output-directory",
        default="/kaggle/working/atom-english",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=default_sequence_length,
    )
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--maximum-wall-seconds", type=int, default=38_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=default_gradient_accumulation,
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--context-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--context-tail-tokens", type=int, default=256)
    parser.add_argument(
        "--context-differentiable-tokens",
        type=int,
        default=2_048,
    )
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    parser.add_argument("--sample-tokens", type=int, default=96)
    parser.add_argument("--evaluation-examples", type=int)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--teacher-model-id")
    parser.add_argument("--teacher-revision")
    parser.add_argument("--require-accelerator", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.mode == "self-test":
        result = run_self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.mode == "evaluate":
        result = run_evaluation(args)
    elif args.stage == "context":
        result = run_context_training(args)
    else:
        result = run_training(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
