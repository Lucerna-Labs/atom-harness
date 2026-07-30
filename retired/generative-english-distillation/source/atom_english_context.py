"""Long-context conditioning for the generative Atom English model."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from atom_english_core import AtomCausalLanguageModel
from atom_english_evaluation import (
    LONG_CONTEXT_TASK_FAMILIES,
    build_long_context_probe,
)
from atom_english_training import (
    AtomTrainingConfig,
    load_optimizer_checkpoint,
    save_atom_english_checkpoint,
)

ATOM_ENGLISH_CONTEXT_RUNTIME = "atom-english-long-context-conditioning-v1"
CONTEXT_TARGET_OPTIMIZER_STEPS = 2_600
CONTEXT_TRAINING_LENGTHS = (
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    264_000,
    524_288,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit_context_event(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "runtime": ATOM_ENGLISH_CONTEXT_RUNTIME,
                "event": event,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _autocast_context(device: torch.device, precision: str) -> Any:
    enabled = precision != "float32"
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def context_length_for_step(
    optimizer_step: int,
    *,
    target_steps: int = CONTEXT_TARGET_OPTIMIZER_STEPS,
    lengths: Sequence[int] = CONTEXT_TRAINING_LENGTHS,
) -> int:
    """Move from differentiable short spans to repeated 264K/512K exposure."""

    if optimizer_step < 0 or target_steps < 4:
        raise ValueError("context schedule bounds are invalid")
    ordered = tuple(int(value) for value in lengths)
    if not ordered or any(value < 256 for value in ordered):
        raise ValueError("context schedule contains an invalid length")
    if tuple(sorted(set(ordered))) != ordered:
        raise ValueError("context schedule must be strictly increasing")
    progress = min(optimizer_step / max(target_steps - 1, 1), 1.0)
    if len(ordered) == 1:
        return ordered[0]
    if progress < 0.62:
        pool = ordered[: min(3, len(ordered))]
    elif progress < 0.85:
        pool = ordered[: min(5, len(ordered))]
    elif progress < 0.95:
        pool = ordered[: min(7, len(ordered))]
    else:
        pool = ordered
    return pool[optimizer_step % len(pool)]


def build_context_training_tensors(
    tokenizer: Any,
    distractor_tokens: Sequence[int],
    *,
    context_tokens: int,
    family: str,
    seed: int,
    trainable_tail_tokens: int,
) -> dict[str, Any]:
    """Create an exact-answer example with a separately streamable prefix."""

    if trainable_tail_tokens < 64:
        raise ValueError("trainable context tail is too short")
    prompt_budget = context_tokens - 64
    if prompt_budget <= trainable_tail_tokens:
        raise ValueError("context training example has no frozen prefix")
    prompt_ids, expected = build_long_context_probe(
        tokenizer,
        distractor_tokens,
        prompt_tokens=prompt_budget,
        family=family,
        seed=seed,
    )
    answer_ids = [
        int(value)
        for value in tokenizer.encode(
            " " + expected,
            add_special_tokens=False,
        )
    ]
    answer_ids.append(int(tokenizer.eos_token_id))
    if len(answer_ids) >= 64:
        raise ValueError("context answer exceeds its reserved token budget")
    tail_start = len(prompt_ids) - trainable_tail_tokens
    prefix_ids = prompt_ids[:tail_start]
    tail_prompt_ids = prompt_ids[tail_start:]
    tail_sequence = [*tail_prompt_ids, *answer_ids]
    input_ids = torch.tensor(tail_sequence[:-1], dtype=torch.long)
    labels = torch.tensor(tail_sequence[1:], dtype=torch.long)
    answer_prediction_start = len(tail_prompt_ids) - 1
    labels[:answer_prediction_start] = -100
    return {
        "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
        "input_ids": input_ids,
        "labels": labels,
        "expected": expected,
        "family": family,
        "context_tokens": context_tokens,
        "prompt_tokens": len(prompt_ids),
        "supervised_tokens": int((labels != -100).sum().item()),
    }


def _learning_rate(
    config: AtomTrainingConfig,
    optimizer_step: int,
) -> float:
    if optimizer_step < config.warmup_steps:
        return (
            config.learning_rate
            * (optimizer_step + 1)
            / max(
                config.warmup_steps,
                1,
            )
        )
    progress = (optimizer_step - config.warmup_steps) / max(
        config.maximum_steps - config.warmup_steps,
        1,
    )
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
    ratio = config.minimum_learning_rate_ratio
    return float(config.learning_rate * (ratio + (1.0 - ratio) * cosine.item()))


def train_atom_english_context_stage(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    distractor_tokens: Sequence[int],
    *,
    config: AtomTrainingConfig,
    output_directory: Path,
    tokenizer_manifest: Mapping[str, Any],
    curriculum_manifest: Mapping[str, Any],
    device: torch.device,
    resume_directory: Path | None = None,
    training_lengths: Sequence[int] = CONTEXT_TRAINING_LENGTHS,
    trainable_tail_tokens: int = 256,
    differentiable_context_limit: int = 2_048,
) -> dict[str, Any]:
    """Condition bounded persistent memory without a full-context gradient tape."""

    if config.stage != "context":
        raise ValueError("long-context trainer requires the context stage")
    ordered_lengths = tuple(int(value) for value in training_lengths)
    if ordered_lengths[-1] > model.config.max_seq_len:
        raise ValueError("training schedule exceeds model context")
    if not distractor_tokens:
        raise ValueError("context training requires distractor tokens")
    if not 256 <= differentiable_context_limit <= model.config.max_seq_len:
        raise ValueError("differentiable context limit is invalid")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    optimizer_step = 0
    scanned_tokens = 0
    supervised_tokens = 0
    examples = 0
    per_length: dict[str, int] = {str(value): 0 for value in ordered_lengths}
    per_family: dict[str, int] = {family: 0 for family in LONG_CONTEXT_TASK_FAMILIES}
    if resume_directory is not None:
        manifest = load_optimizer_checkpoint(resume_directory, optimizer)
        if manifest["training_config"]["stage"] != "context":
            raise ValueError("context resume checkpoint has another stage")
        state = manifest["training_state"]
        optimizer_step = int(state["optimizer_step"])
        scanned_tokens = int(state["scanned_tokens"])
        supervised_tokens = int(state["consumed_tokens"])
        examples = int(state["examples"])
        for key, value in state["per_length_examples"].items():
            if key in per_length:
                per_length[key] = int(value)
        for key, value in state["per_family_examples"].items():
            if key in per_family:
                per_family[key] = int(value)
    if optimizer_step > config.maximum_steps:
        raise ValueError("context resume step exceeds configured target")
    for group in optimizer.param_groups:
        group["lr"] = _learning_rate(config, optimizer_step)

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(config.precision == "float16"),
    )
    model.train()
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    stop_reason = "step_limit"
    _emit_context_event(
        "stage_started",
        optimizer_step=optimizer_step,
        target_steps=config.maximum_steps,
        context_ceiling=model.config.max_seq_len,
        training_lengths=list(ordered_lengths),
        trainable_tail_tokens=trainable_tail_tokens,
        differentiable_context_limit=differentiable_context_limit,
    )
    while optimizer_step < config.maximum_steps:
        if time.monotonic() - started >= config.maximum_wall_seconds:
            stop_reason = "wall_time_limit"
            break
        context_tokens = context_length_for_step(
            optimizer_step,
            target_steps=config.maximum_steps,
            lengths=ordered_lengths,
        )
        family = LONG_CONTEXT_TASK_FAMILIES[
            optimizer_step % len(LONG_CONTEXT_TASK_FAMILIES)
        ]
        seed = config.seed + optimizer_step * 104_729
        example = build_context_training_tensors(
            tokenizer,
            distractor_tokens,
            context_tokens=context_tokens,
            family=family,
            seed=seed,
            trainable_tail_tokens=trainable_tail_tokens,
        )
        prefix_ids = (
            example["prefix_ids"]
            .to(
                device,
                non_blocking=True,
            )
            .unsqueeze(0)
        )
        input_ids = (
            example["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
            .unsqueeze(0)
        )
        labels = (
            example["labels"]
            .to(
                device,
                non_blocking=True,
            )
            .unsqueeze(0)
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if context_tokens <= differentiable_context_limit:
            full_input_ids = torch.cat((prefix_ids, input_ids), dim=1)
            prefix_labels = torch.full(
                prefix_ids.shape,
                -100,
                dtype=torch.long,
                device=device,
            )
            full_labels = torch.cat((prefix_labels, labels), dim=1)
            with _autocast_context(device, config.precision):
                output = model(full_input_ids, full_labels)
                assert output.cross_entropy is not None
                language_loss = output.cross_entropy
                criticality = output.criticality_loss
                loss = language_loss + model.config.criticality_weight * criticality
            gradient_mode = "end-to-end"
        else:
            model.eval()
            with (
                torch.no_grad(),
                _autocast_context(
                    device,
                    config.precision,
                ),
            ):
                _, states, _ = model.prefill(prefix_ids)
            model.train()
            with _autocast_context(device, config.precision):
                logits, _, diagnostics = model.forward_stream(
                    input_ids,
                    states,
                )
                language_loss = F.cross_entropy(
                    logits.reshape(-1, model.config.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                entropy = torch.stack(
                    [item.edge_entropy for item in diagnostics]
                ).mean()
                criticality = (entropy - model.config.criticality_target).square()
                loss = language_loss + model.config.criticality_weight * criticality
            gradient_mode = "bounded-tail"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer_step += 1
        examples += 1
        scanned_tokens += context_tokens
        supervised_tokens += int(example["supervised_tokens"])
        per_length[str(context_tokens)] += 1
        per_family[family] += 1
        learning_rate = _learning_rate(config, optimizer_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        record = {
            "optimizer_step": optimizer_step,
            "context_tokens": context_tokens,
            "family": family,
            "gradient_mode": gradient_mode,
            "loss": float(loss.detach().item()),
            "cross_entropy": float(language_loss.detach().item()),
            "criticality": float(criticality.detach().item()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": learning_rate,
            "scanned_tokens": scanned_tokens,
            "supervised_tokens": supervised_tokens,
            "elapsed_seconds": time.monotonic() - started,
        }
        if optimizer_step % config.log_interval == 0 or optimizer_step == 1:
            history.append(record)
            _emit_context_event("optimizer_progress", **record)
        if optimizer_step % config.save_interval == 0:
            state = {
                "stage": "context",
                "optimizer_step": optimizer_step,
                "micro_step": optimizer_step,
                "consumed_tokens": supervised_tokens,
                "scanned_tokens": scanned_tokens,
                "examples": examples,
                "per_length_examples": per_length,
                "per_family_examples": per_family,
                "elapsed_seconds": time.monotonic() - started,
                "stop_reason": "periodic_save",
            }
            checkpoint = save_atom_english_checkpoint(
                output_directory / "latest",
                model,
                optimizer=optimizer,
                training_state=state,
                tokenizer_manifest=tokenizer_manifest,
                curriculum_manifest=curriculum_manifest,
                training_config=config,
            )
            _emit_context_event(
                "checkpoint_saved",
                optimizer_step=optimizer_step,
                checkpoint_manifest_hash=checkpoint["manifest_hash"],
            )

    elapsed = time.monotonic() - started
    state = {
        "stage": "context",
        "optimizer_step": optimizer_step,
        "micro_step": optimizer_step,
        "consumed_tokens": supervised_tokens,
        "scanned_tokens": scanned_tokens,
        "examples": examples,
        "per_length_examples": per_length,
        "per_family_examples": per_family,
        "elapsed_seconds": elapsed,
        "stop_reason": stop_reason,
    }
    checkpoint = save_atom_english_checkpoint(
        output_directory,
        model,
        optimizer=optimizer,
        training_state=state,
        tokenizer_manifest=tokenizer_manifest,
        curriculum_manifest=curriculum_manifest,
        training_config=config,
    )
    report = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_CONTEXT_RUNTIME,
        "training_config": asdict(config),
        "training_state": state,
        "history": history,
        "checkpoint_manifest_hash": checkpoint["manifest_hash"],
    }
    _write_json(output_directory / "context_training_report.json", report)
    _emit_context_event(
        "stage_stopped",
        optimizer_step=optimizer_step,
        stop_reason=stop_reason,
        scanned_tokens=scanned_tokens,
        elapsed_seconds=elapsed,
        checkpoint_manifest_hash=checkpoint["manifest_hash"],
    )
    return report


class _ContextSelfTestTokenizer:
    eos_token_id = 0

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


def atom_english_context_self_test() -> dict[str, bool]:
    tokenizer = _ContextSelfTestTokenizer()
    reservoir = tokenizer.encode(
        "Persistent scientific context. ",
        add_special_tokens=False,
    )
    example = build_context_training_tensors(
        tokenizer,
        reservoir,
        context_tokens=512,
        family="state_update",
        seed=20260724,
        trainable_tail_tokens=64,
    )
    final_lengths = {
        context_length_for_step(
            step,
            target_steps=CONTEXT_TARGET_OPTIMIZER_STEPS,
        )
        for step in range(
            int(CONTEXT_TARGET_OPTIMIZER_STEPS * 0.95),
            CONTEXT_TARGET_OPTIMIZER_STEPS,
        )
    }
    return {
        "frozen_prefix_present": int(example["prefix_ids"].numel()) > 0,
        "answer_tail_supervised": int(example["supervised_tokens"]) > 1,
        "264k_repeated": 264_000 in final_lengths,
        "512k_repeated": 524_288 in final_lengths,
    }
