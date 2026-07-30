"""Integrated natural-English shell over the evidence-bound Atom neural field."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from atom_english_language_dataset import (
    EnglishLanguageProgram,
    build_english_language_program,
    english_language_self_tests,
)
from atom_english_language_model import (
    ENGLISH_LANGUAGE_RUNTIME,
    english_language_model_payload,
    english_model_hash,
    english_model_self_tests,
    load_english_language_model,
    run_english_inference_request,
)
from atom_english_language_side_view import (
    ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_english_language_artifact,
)
from atom_field_proof import PROCESS_NAMES
from atom_neural_language_experiment import (
    _ablation_report,
    _controller_summary,
    _coverage_balanced_training_rows,
    _evidence_boundary_report,
)
from atom_neural_language_model import (
    AtomNeuralLanguageField,
    FlatNeuralLanguageBaseline,
    NeuralArchitectureConfig,
    NeuralVocabulary,
    adapt_neural_stream,
    clone_neural_model,
    evaluate_neural_model,
    neural_language_model_payload,
    neural_parameter_count,
    set_neural_deterministic,
    tensor_state_payload,
    train_neural_model,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_english_language_graph,
    retrieve_atom_context,
)


ENGLISH_EXPERIMENT_SCHEMA = 1
ENGLISH_EXPERIMENT_SEED = 2026072201
ENGLISH_EXPERIMENT_RUNTIME = "atom-english-language-experiment-v1"


def write_english_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_english_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _train_base_models(
    program: EnglishLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> tuple[AtomNeuralLanguageField, FlatNeuralLanguageBaseline, dict[str, Any]]:
    set_neural_deterministic(ENGLISH_EXPERIMENT_SEED)
    atom = AtomNeuralLanguageField(vocabulary, config)
    atom_training = train_neural_model(
        atom,
        program.stages["base_train"],
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=ENGLISH_EXPERIMENT_SEED + 1,
        device=device,
    )
    set_neural_deterministic(ENGLISH_EXPERIMENT_SEED)
    flat = FlatNeuralLanguageBaseline(vocabulary, config)
    flat_training = train_neural_model(
        flat,
        program.stages["base_train"],
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=ENGLISH_EXPERIMENT_SEED + 1,
        device=device,
    )
    return atom, flat, {"atom": atom_training, "flat": flat_training}


def _adapt_models(
    base: AtomNeuralLanguageField,
    program: EnglishLanguageProgram,
    recovery_train: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> tuple[AtomNeuralLanguageField, AtomNeuralLanguageField, dict[str, Any]]:
    stream = [
        (stage, program.stages[stage])
        for stage in (
            "transfer_adaptation",
            "transfer_noise",
        )
    ]
    stream.append(("transfer_recovery", recovery_train))
    adaptive = clone_neural_model(base)
    fixed = clone_neural_model(base)
    adaptive_trace = adapt_neural_stream(
        adaptive,
        stream,
        vocabulary,
        config,
        adaptive=True,
        device=device,
    )
    fixed_trace = adapt_neural_stream(
        fixed,
        stream,
        vocabulary,
        config,
        adaptive=False,
        device=device,
    )
    return adaptive, fixed, {"adaptive": adaptive_trace, "fixed": fixed_trace}


def _transfer_recovery_split(
    program: EnglishLanguageProgram,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Hold out two of every six recovery cases for policy selection."""

    rows = program.stages["transfer_recovery"]
    training = tuple(row for index, row in enumerate(rows) if index % 6 < 4)
    validation = tuple(row for index, row in enumerate(rows) if index % 6 >= 4)
    if len(training) != 84 or len(validation) != 42:
        raise AssertionError("unexpected transfer recovery split")
    return training, validation


def _select_adaptation_policy(
    adaptive: AtomNeuralLanguageField,
    fixed: AtomNeuralLanguageField,
    validation_rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    training: Mapping[str, Any],
    device: torch.device,
) -> tuple[str, AtomNeuralLanguageField, dict[str, Any]]:
    """Select a learning policy without consulting the final composition set."""

    candidates = {"adaptive": adaptive, "fixed": fixed}
    validation = {
        name: evaluate_neural_model(
            model,
            validation_rows,
            vocabulary,
            config,
            batch_size=96,
            device=device,
        )
        for name, model in candidates.items()
    }
    update_counts = {
        name: int(training["adaptation"][name]["update_count"]) for name in candidates
    }
    selected_name = max(
        candidates,
        key=lambda name: (
            validation[name]["joint_accuracy"],
            validation[name]["state_accuracy"],
            -validation[name]["continuous_mae"],
            -update_counts[name],
            name,
        ),
    )
    return (
        selected_name,
        candidates[selected_name],
        {
            "final_evaluation_hidden_during_selection": True,
            "policy": selected_name,
            "selection_cases": len(validation_rows),
            "selection_rule": (
                "joint accuracy, state accuracy, lower continuous error, "
                "then fewer updates"
            ),
            "update_counts": update_counts,
            "validation": validation,
        },
    )


def _evaluate_models(
    models: Mapping[str, torch.nn.Module],
    program: EnglishLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> dict[str, Any]:
    stages = (
        "base_validation",
        "base_composition",
        "transfer_composition",
        "zero_shot_composition",
    )
    return {
        model_name: {
            stage: evaluate_neural_model(
                model,
                program.stages[stage],
                vocabulary,
                config,
                batch_size=96,
                device=device,
            )
            for stage in stages
        }
        for model_name, model in models.items()
    }


def _sample_efficiency_probe(
    program: EnglishLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> dict[str, Any]:
    rows = _coverage_balanced_training_rows(program)  # type: ignore[arg-type]
    set_neural_deterministic(ENGLISH_EXPERIMENT_SEED)
    model = AtomNeuralLanguageField(vocabulary, config)
    training = train_neural_model(
        model,
        rows,
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=ENGLISH_EXPERIMENT_SEED + 1,
        device=device,
    )
    evaluation = evaluate_neural_model(
        model,
        program.stages["base_composition"],
        vocabulary,
        config,
        batch_size=96,
        device=device,
    )
    return {
        "curriculum": "one-world-variant-per-language-composition-query-cell",
        "evaluation": evaluation,
        "example_fraction": len(rows) / len(program.stages["base_train"]),
        "training": training,
        "training_examples": len(rows),
    }


def validate_english_workflow_request(payload: Mapping[str, Any]) -> None:
    expected = {"request_id", "schema_version", "turns"}
    if set(payload) != expected:
        raise ValueError(f"English workflow fields must be {sorted(expected)}")
    if payload["schema_version"] != ENGLISH_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported English workflow schema")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("English workflow request_id must be non-empty")
    turns = payload["turns"]
    if not isinstance(turns, list) or not 1 <= len(turns) <= 12:
        raise ValueError("English workflow requires between 1 and 12 turns")
    for turn in turns:
        if not isinstance(turn, Mapping) or set(turn) != {
            "adjacency",
            "node_features",
            "turn_id",
            "utterance",
        }:
            raise ValueError("English workflow turn fields are invalid")


def run_english_workflow(
    model_payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    validate_english_workflow_request(request)
    loaded = load_english_language_model(model_payload)
    graph = build_english_language_graph()
    context = retrieve_atom_context(
        graph,
        "natural English evidence answer synonym adaptation language codec",
        limit=8,
    )
    turns: list[dict[str, Any]] = []
    for turn in request["turns"]:
        inference = run_english_inference_request(
            loaded,
            {
                "adjacency": turn["adjacency"],
                "node_features": turn["node_features"],
                "request_id": turn["turn_id"],
                "utterance": turn["utterance"],
            },
        )
        turns.append(
            {
                "artifact": inference["artifact"],
                "knowledge_context": context,
                "turn_id": turn["turn_id"],
            }
        )
    return {
        "core_model_hash": loaded.core.model_hash,
        "model_hash": loaded.model_hash,
        "request_id": request["request_id"],
        "runtime": {
            "english_runtime": ENGLISH_LANGUAGE_RUNTIME,
            "experiment_runtime": ENGLISH_EXPERIMENT_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        },
        "schema_version": ENGLISH_EXPERIMENT_SCHEMA,
        "turns": turns,
    }


def _build_workflow(
    program: EnglishLanguageProgram,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supported_rows = program.stages["transfer_composition"]
    unsupported_rows = program.stages["zero_shot_composition"]
    selected = [
        *(
            {"row": supported_rows[index], "should_assert": True}
            for index in (0, 17, 35, 52, 71, 90)
        ),
        *(
            {"row": unsupported_rows[index], "should_assert": False}
            for index in (0, 31)
        ),
    ]
    turns = []
    for index, entry in enumerate(selected):
        row = entry["row"]
        truth = program.evaluator_truth[str(row["event_id"])]
        turns.append(
            {
                "adjacency": row["adjacency"],
                "node_features": row["node_features"],
                "turn_id": f"english-turn-{index + 1}",
                "utterance": truth["user_utterance"],
            }
        )
    return {
        "request_id": "evidence-bound-natural-english-workflow",
        "schema_version": ENGLISH_EXPERIMENT_SCHEMA,
        "turns": turns,
    }, selected


def _score_workflow(
    response: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    program: EnglishLanguageProgram,
) -> dict[str, Any]:
    supported = 0
    correct = 0
    state_success = 0
    unsupported = 0
    abstentions = 0
    context_turns = 0
    for turn, expectation in zip(response["turns"], expected, strict=True):
        artifact = turn["artifact"]
        row = expectation["row"]
        truth = program.evaluator_truth[str(row["event_id"])]
        if expectation["should_assert"]:
            supported += 1
            continuous = torch.tensor(artifact["continuous"])
            target_continuous = torch.tensor(row["target_continuous"])
            binary = torch.tensor(artifact["binary_probability"]) >= 0.5
            target_binary = torch.tensor(row["target_binary"]) >= 0.5
            state_correct = bool(
                (continuous - target_continuous).abs().mean() <= 0.16
                and torch.equal(binary, target_binary)
            )
            answer_correct = artifact["answer"] == truth["english_answer"]
            correct += int(state_correct and answer_correct)
            state_success += int(state_correct)
        else:
            unsupported += 1
            abstained = (
                artifact["claim_status"] == "unknown"
                and artifact["response"] is None
                and artifact["reasoning"]["execution_skipped"]
                and "enough grounded evidence" in artifact["answer"]
            )
            abstentions += int(abstained)
        context_turns += int(bool(turn["knowledge_context"]))
    joint = correct / max(supported, 1)
    passed = (
        joint >= 0.80 and abstentions == unsupported and context_turns == len(expected)
    )
    return {
        "all_turns_have_graph_context": context_turns == len(expected),
        "correct": correct,
        "correct_abstentions": abstentions,
        "joint_accuracy": joint,
        "passed": passed,
        "state_success": state_success,
        "supported_turns": supported,
        "turns": len(expected),
        "unsupported_turns": unsupported,
    }


def _reseal(payload: Mapping[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(dict(payload))
    base = {key: value for key, value in changed.items() if key != "model_hash"}
    changed["model_hash"] = english_model_hash(base)
    return changed


def _corruption_checks(payload: Mapping[str, Any]) -> dict[str, Any]:
    corruptions: dict[str, dict[str, Any]] = {}
    extra = copy.deepcopy(dict(payload))
    extra["extra"] = True
    corruptions["unknown_field"] = extra
    bad_hash = copy.deepcopy(dict(payload))
    bad_hash["model_hash"] = "0" * 64
    corruptions["model_hash"] = bad_hash
    bad_shell = copy.deepcopy(dict(payload))
    bad_shell["shell"]["internal_marker"] = "wrong"
    corruptions["shell_marker"] = _reseal(bad_shell)
    bad_dataset = copy.deepcopy(dict(payload))
    bad_dataset["dataset_hash"] = "short"
    corruptions["dataset_hash"] = _reseal(bad_dataset)
    bad_core = copy.deepcopy(dict(payload))
    bad_core["core_model"]["model_hash"] = "f" * 64
    corruptions["nested_core_hash"] = _reseal(bad_core)
    checks: dict[str, bool] = {}
    for name, corruption in corruptions.items():
        try:
            load_english_language_model(corruption)
        except (KeyError, TypeError, ValueError):
            checks[name] = True
        else:
            checks[name] = False
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "rejected": sum(checks.values()),
    }


def _architecture_audit(model: AtomNeuralLanguageField) -> dict[str, Any]:
    graph = build_english_language_graph()
    runtime_path = Path(__file__)
    experiment_source = runtime_path.read_text(encoding="utf-8")
    model_path = runtime_path.with_name("atom_english_language_model.py")
    dataset_path = runtime_path.with_name("atom_english_language_dataset.py")
    model_source = (model_path if model_path.exists() else runtime_path).read_text(
        encoding="utf-8"
    )
    dataset_source = (
        dataset_path if dataset_path.exists() else runtime_path
    ).read_text(encoding="utf-8")
    checks = {
        "English_codec_runtime_wired": "normalize_english_request" in model_source
        and "render_english_answer" in model_source,
        "evidence_gate_runtime_wired": "run_neural_inference_request" in model_source,
        "factorized_surface_memory": tuple(model.surface_table.shape[1:]) == (2, 7),
        "graph_resolves_to_seven_primitives": set(
            graph.expand("evidence_bound_english_answer")
        )
        == set(PROCESS_NAMES),
        "natural_English_curriculum_present": "ENGLISH_OPERATOR_LEXICONS"
        in dataset_source,
        "metaplastic_policy_selection_wired": "_select_adaptation_policy"
        in experiment_source,
        "strict_outer_loader": "load_english_language_model" in model_source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}


def _behavior_contract(
    evaluations: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash cross-platform decisions without assuming identical weight bytes."""

    base = {
        "evaluation_scope": "selected-transfer-composition-and-English-workflow",
        "schema_version": 1,
        "transfer_decisions": [
            {
                "event_id": prediction["event_id"],
                "response": prediction["response"],
                "response_correct": prediction["response_correct"],
                "state_success": prediction["state_success"],
            }
            for prediction in evaluations["selected"]["transfer_composition"][
                "predictions"
            ]
        ],
        "workflow_decisions": [
            {
                "answer": turn["artifact"]["answer"],
                "claim_status": turn["artifact"]["claim_status"],
                "execution_skipped": turn["artifact"]["reasoning"]["execution_skipped"],
                "response": turn["artifact"]["response"],
            }
            for turn in workflow["turns"]
        ],
    }
    return {**base, "behavior_sha256": english_model_hash(base)}


def _experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = report["evaluations"]
    selected = evaluation["selected"]["transfer_composition"]
    flat = evaluation["flat"]["base_composition"]
    retention = evaluation["selected"]["base_composition"]
    evidence = report["evidence_boundary"]
    sample = report["sample_efficiency"]
    selection = report["selection"]
    behavior = report["behavior_contract"]
    behavior_payload = {
        key: value for key, value in behavior.items() if key != "behavior_sha256"
    }
    checks = {
        "selected_English_transfer": selected["joint_accuracy"] >= 0.80,
        "both_adaptation_policies_measured": all(
            evaluation[name]["transfer_composition"]["cases"] == 96
            for name in ("adaptive", "fixed")
        ),
        "heldout_metaplastic_selection": selection[
            "final_evaluation_hidden_during_selection"
        ]
        and selection["selection_cases"] == 42
        and selection["policy"] in {"adaptive", "fixed"},
        "chaos_budget_bounded": report["controller"]["maximum_chaos_load"] <= 1.150001,
        "behavior_contract_sealed": behavior["behavior_sha256"]
        == english_model_hash(behavior_payload),
        "all_field_operators_causal": report["ablations"]["all_field_operators_causal"],
        "all_text_operators_causal": report["ablations"]["all_text_operators_causal"],
        "architecture_audit": report["architecture_audit"]["passed"],
        "corruptions_rejected": report["corruption_checks"]["passed"],
        "dataset_audit": report["dataset"]["audit"]["passed"],
        "deterministic_training": report["deterministic_training"]["passed"],
        "English_workflow": report["serialized_workflow"]["passed"],
        "evidence_assertion_accuracy": evidence["supported"]["assertion_accuracy"]
        >= 0.85,
        "evidence_supported_coverage": evidence["supported"]["coverage"] >= 0.85,
        "flat_composition_control": flat["joint_accuracy"] <= 0.10,
        "graph_rag_all_turns": report["serialized_workflow"][
            "all_turns_have_graph_context"
        ],
        "quarter_data_English": sample["evaluation"]["joint_accuracy"] >= 0.80,
        "quarter_data_fraction": sample["example_fraction"] == 0.25,
        "retention": retention["joint_accuracy"] >= 0.80,
        "roundtrip_exact": report["serialization"]["roundtrip_exact"],
        "self_tests": report["self_tests"]["passed"],
        "side_view_bound": report["side_view_contract"]["model_hash"]
        == report["model_hash"],
        "unknown_English_abstention": evidence["unsupported"]["correct_abstention_rate"]
        == 1.0,
        "unnecessary_compute_reduced": evidence["compute"]["reduction"] >= 0.50,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}


def _run_training(
    program: EnglishLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> tuple[
    AtomNeuralLanguageField,
    AtomNeuralLanguageField,
    AtomNeuralLanguageField,
    FlatNeuralLanguageBaseline,
    dict[str, Any],
]:
    recovery_train, recovery_validation = _transfer_recovery_split(program)
    base, flat, base_training = _train_base_models(program, vocabulary, config, device)
    adaptive, fixed, adaptation = _adapt_models(
        base, program, recovery_train, vocabulary, config, device
    )
    return (
        base,
        adaptive,
        fixed,
        flat,
        {
            "adaptation": adaptation,
            "base": base_training,
            "selection_split": {
                "recovery_training_cases": len(recovery_train),
                "selection_cases": len(recovery_validation),
                "strategy": "four-train-two-select-within-each-six-case-family",
            },
        },
    )


def run_english_language_experiment(output_dir: Path) -> dict[str, Any]:
    device = torch.device("cpu")
    program = build_english_language_program()
    vocabulary = NeuralVocabulary.build(program.vocabulary, program.response_vocabulary)
    config = NeuralArchitectureConfig(maximum_tokens=28)
    config.validate()
    base, adaptive, fixed, flat, training = _run_training(
        program, vocabulary, config, device
    )
    evaluations = _evaluate_models(
        {"base": base, "adaptive": adaptive, "fixed": fixed, "flat": flat},
        program,
        vocabulary,
        config,
        device,
    )
    _, selection_rows = _transfer_recovery_split(program)
    selected_name, selected, selection = _select_adaptation_policy(
        adaptive,
        fixed,
        selection_rows,
        vocabulary,
        config,
        training,
        device,
    )
    evaluations["selected"] = evaluations[selected_name]
    sample_efficiency = _sample_efficiency_probe(program, vocabulary, config, device)
    training_summary = {
        "selected_policy": selected_name,
        "selected_transfer": evaluations["selected"]["transfer_composition"][
            "joint_accuracy"
        ],
        "base_composition": evaluations["base"]["base_composition"]["joint_accuracy"],
        "retention": evaluations["selected"]["base_composition"]["joint_accuracy"],
        "seed": ENGLISH_EXPERIMENT_SEED,
    }
    core_payload = neural_language_model_payload(
        selected, training_summary=training_summary
    )
    model_payload = english_language_model_payload(
        core_payload,
        dataset_hash=str(program.manifest["dataset_hash"]),
    )
    loaded = load_english_language_model(model_payload)
    roundtrip_exact = tensor_state_payload(loaded.core.model) == tensor_state_payload(
        selected
    )

    replay_base, replay_adaptive, replay_fixed, _, replay_training = _run_training(
        program, vocabulary, config, device
    )
    del replay_base
    replay_selected_name, replay_selected, _ = _select_adaptation_policy(
        replay_adaptive,
        replay_fixed,
        selection_rows,
        vocabulary,
        config,
        replay_training,
        device,
    )
    replay_core = neural_language_model_payload(
        replay_selected, training_summary=training_summary
    )
    replay_payload = english_language_model_payload(
        replay_core,
        dataset_hash=str(program.manifest["dataset_hash"]),
    )
    deterministic = {
        "first_model_hash": model_payload["model_hash"],
        "passed": replay_payload["model_hash"] == model_payload["model_hash"]
        and replay_selected_name == selected_name,
        "selected_policy": selected_name,
        "second_model_hash": replay_payload["model_hash"],
    }

    workflow_request, workflow_truth = _build_workflow(program)
    workflow_response = run_english_workflow(model_payload, workflow_request)
    workflow_score = _score_workflow(workflow_response, workflow_truth, program)
    evidence = _evidence_boundary_report(
        selected,
        flat,
        program.stages["transfer_composition"],
        program.stages["zero_shot_composition"],
        vocabulary,
        config,
    )
    ablations = _ablation_report(
        selected,
        program.stages["transfer_composition"],
        vocabulary,
        config,
    )
    self_test_sections = {
        "dataset": english_language_self_tests(),
        "model": english_model_self_tests(),
    }
    self_tests = {
        "checks": {
            name: section["passed"] for name, section in self_test_sections.items()
        },
        "failed": sorted(
            name
            for name, section in self_test_sections.items()
            if not section["passed"]
        ),
        "passed": all(section["passed"] for section in self_test_sections.values()),
        "sections": self_test_sections,
    }
    report: dict[str, Any] = {
        "ablations": ablations,
        "architecture": ENGLISH_LANGUAGE_RUNTIME,
        "architecture_audit": _architecture_audit(selected),
        "behavior_contract": _behavior_contract(evaluations, workflow_response),
        "controller": _controller_summary(training["adaptation"]["adaptive"]),
        "core_model_hash": core_payload["model_hash"],
        "corruption_checks": _corruption_checks(model_payload),
        "dataset": program.manifest,
        "deterministic_training": deterministic,
        "evaluations": evaluations,
        "evidence_boundary": evidence,
        "experiment": ENGLISH_EXPERIMENT_RUNTIME,
        "knowledge_runtime": {
            "rag_runtime": ATOM_RAG_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        },
        "model_hash": model_payload["model_hash"],
        "parameter_counts": {
            "atom": neural_parameter_count(selected),
            "flat": neural_parameter_count(flat),
            "shell_trainable": 0,
        },
        "sample_efficiency": sample_efficiency,
        "schema_version": ENGLISH_EXPERIMENT_SCHEMA,
        "selection": selection,
        "self_tests": self_tests,
        "serialization": {"roundtrip_exact": roundtrip_exact},
        "serialized_workflow": workflow_score,
        "side_view_contract": {
            "binding": "render_english_language_artifact",
            "model_hash": model_payload["model_hash"],
            "runtime": ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME,
        },
        "training": training,
    }
    report["experiment_gates"] = _experiment_gates(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_english_json(
        output_dir / "atom_english_language_dataset_manifest.json",
        program.manifest,
    )
    for stage, rows in program.stages.items():
        write_english_jsonl(output_dir / f"atom_english_language_{stage}.jsonl", rows)
    write_english_json(
        output_dir / "atom_english_language_evaluator_truth.json",
        program.evaluator_truth,
    )
    write_english_json(output_dir / "atom_english_language_model.json", model_payload)
    write_english_json(output_dir / "atom_english_language_report.json", report)
    write_english_json(
        output_dir / "atom_english_language_workflow_request.json",
        workflow_request,
    )
    write_english_json(
        output_dir / "atom_english_language_workflow_response.json",
        workflow_response,
    )
    document = render_english_language_artifact(
        model_payload, report, workflow_response
    )
    side_path = output_dir / "atom_english_language_side_view.html"
    side_path.write_text(document, encoding="utf-8", newline="\n")
    report["artifacts"] = {
        "model": "atom_english_language_model.json",
        "report": "atom_english_language_report.json",
        "side_view": side_path.name,
        "side_view_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "workflow_request": "atom_english_language_workflow_request.json",
        "workflow_response": "atom_english_language_workflow_response.json",
    }
    write_english_json(output_dir / "atom_english_language_report.json", report)
    return report


def parse_english_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("english_language_outputs")
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_english_args()
    if args.self_test:
        payload = {
            "dataset": english_language_self_tests(),
            "model": english_model_self_tests(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if all(section["passed"] for section in payload.values()) else 1
    if args.model or args.request or args.response:
        if not (args.model and args.request and args.response):
            raise SystemExit(
                "--model, --request, and --response must be supplied together"
            )
        model_payload = json.loads(args.model.read_text(encoding="utf-8"))
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = run_english_workflow(model_payload, request)
        write_english_json(args.response, response)
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0
    report = run_english_language_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["experiment_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
