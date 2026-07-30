"""Integrated lifelong neural language-field experiment."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from atom_field_proof import PROCESS_NAMES
from atom_neural_language_dataset import (
    RUNTIME_ROW_KEYS,
    NeuralLanguageProgram,
    build_neural_language_program,
    neural_language_self_tests,
)
from atom_neural_language_model import (
    AtomNeuralLanguageField,
    EvidencePolicyConfig,
    FlatNeuralLanguageBaseline,
    NeuralArchitectureConfig,
    NeuralVocabulary,
    adapt_neural_stream,
    clone_neural_model,
    collate_runtime_rows,
    evaluate_neural_model,
    evidence_preflight,
    load_neural_language_model,
    neural_language_model_payload,
    neural_model_hash,
    neural_model_self_tests,
    neural_parameter_count,
    run_neural_inference_request,
    set_neural_deterministic,
    tensor_state_payload,
    train_neural_model,
)
from atom_neural_language_side_view import (
    ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_neural_language_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_neural_language_graph,
    retrieve_atom_context,
)


NEURAL_EXPERIMENT_SCHEMA = 2
NEURAL_EXPERIMENT_SEED = 2026072118
NEURAL_EXPERIMENT_RUNTIME = "atom-neural-language-field-v2"


def write_neural_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_neural_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _metric_view(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(report[key])
        for key in (
            "binary_accuracy",
            "continuous_mae",
            "joint_accuracy",
            "response_accuracy",
            "state_accuracy",
        )
    }


def _train_base_models(
    program: NeuralLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> tuple[AtomNeuralLanguageField, FlatNeuralLanguageBaseline, dict[str, Any]]:
    set_neural_deterministic(NEURAL_EXPERIMENT_SEED)
    atom_model = AtomNeuralLanguageField(vocabulary, config)
    atom_training = train_neural_model(
        atom_model,
        program.stages["base_train"],
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=NEURAL_EXPERIMENT_SEED + 1,
        device=device,
    )
    set_neural_deterministic(NEURAL_EXPERIMENT_SEED)
    flat_model = FlatNeuralLanguageBaseline(vocabulary, config)
    flat_training = train_neural_model(
        flat_model,
        program.stages["base_train"],
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=NEURAL_EXPERIMENT_SEED + 1,
        device=device,
    )
    return (
        atom_model,
        flat_model,
        {
            "atom": atom_training,
            "flat": flat_training,
        },
    )


def _adapt_models(
    base_model: AtomNeuralLanguageField,
    program: NeuralLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> tuple[AtomNeuralLanguageField, AtomNeuralLanguageField, dict[str, Any]]:
    stream = [
        (stage, program.stages[stage])
        for stage in (
            "transfer_adaptation",
            "transfer_noise",
            "transfer_recovery",
        )
    ]
    adaptive = clone_neural_model(base_model)
    fixed = clone_neural_model(base_model)
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


def _evaluate_all(
    *,
    base: AtomNeuralLanguageField,
    adaptive: AtomNeuralLanguageField,
    fixed: AtomNeuralLanguageField,
    flat: FlatNeuralLanguageBaseline,
    program: NeuralLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> dict[str, Any]:
    stages = {
        "base_validation": program.stages["base_validation"],
        "base_composition": program.stages["base_composition"],
        "transfer_composition": program.stages["transfer_composition"],
        "zero_shot_composition": program.stages["zero_shot_composition"],
    }
    models = {
        "base": base,
        "adaptive": adaptive,
        "fixed": fixed,
        "flat": flat,
    }
    result: dict[str, Any] = {}
    for model_name, model in models.items():
        result[model_name] = {}
        for stage_name, rows in stages.items():
            result[model_name][stage_name] = evaluate_neural_model(
                model,
                rows,
                vocabulary,
                config,
                batch_size=96,
                device=device,
            )
    return result


def _coverage_balanced_training_rows(
    program: NeuralLanguageProgram,
) -> tuple[Mapping[str, Any], ...]:
    """Keep one world variant for every language/composition/question cell."""

    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in program.stages["base_train"]:
        truth = program.evaluator_truth[str(row["event_id"])]
        key = (
            str(truth["language"]),
            str(truth["family"]),
            str(truth["query_type"]),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    expected = len(program.stages["base_train"]) // 4
    if len(selected) != expected:
        raise AssertionError(
            f"coverage-balanced curriculum selected {len(selected)} rows, expected {expected}"
        )
    return tuple(selected)


def _sample_efficiency_probe(
    program: NeuralLanguageProgram,
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    device: torch.device,
) -> dict[str, Any]:
    rows = _coverage_balanced_training_rows(program)
    set_neural_deterministic(NEURAL_EXPERIMENT_SEED)
    model = AtomNeuralLanguageField(vocabulary, config)
    training = train_neural_model(
        model,
        rows,
        vocabulary,
        config,
        epochs=5,
        batch_size=48,
        learning_rate=0.003,
        seed=NEURAL_EXPERIMENT_SEED + 1,
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


def _evidence_boundary_report(
    model: AtomNeuralLanguageField,
    flat: FlatNeuralLanguageBaseline,
    supported_rows: Sequence[Mapping[str, Any]],
    unsupported_rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
) -> dict[str, Any]:
    policy = EvidencePolicyConfig()
    policy.validate(config)

    def atom_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch = collate_runtime_rows(rows, vocabulary, config)
        model.eval()
        with torch.no_grad():
            preflight = evidence_preflight(
                model, batch["token_ids"], batch["token_mask"], policy
            )
            outputs = model(
                batch["token_ids"],
                batch["token_mask"],
                batch["node_features"],
                batch["adjacency"],
                text_tick_budget=policy.fast_text_ticks,
            )
        asserted = preflight["eligible"] & outputs["memory_used"]
        response = outputs["response_logits"].argmax(dim=1)
        response_correct = response == batch["response_ids"]
        continuous_error = (
            (outputs["continuous"] - batch["target_continuous"]).abs().mean(dim=(1, 2))
        )
        binary = torch.sigmoid(outputs["binary_logits"]) >= 0.5
        state_correct = (continuous_error <= 0.16) & (
            binary == (batch["target_binary"] >= 0.5)
        ).all(dim=2).all(dim=1)
        return {
            "asserted": asserted,
            "eligible": preflight["eligible"],
            "response_correct": response_correct,
            "state_correct": state_correct,
        }

    supported = atom_cases(supported_rows)
    unsupported = atom_cases(unsupported_rows)
    unsupported_batch = collate_runtime_rows(unsupported_rows, vocabulary, config)
    flat.eval()
    with torch.no_grad():
        flat_outputs = flat(
            unsupported_batch["token_ids"],
            unsupported_batch["token_mask"],
            unsupported_batch["node_features"],
            unsupported_batch["adjacency"],
        )
    flat_emits = torch.isfinite(flat_outputs["response_logits"]).all(dim=1)
    supported_assertions = int(supported["asserted"].sum())
    unsupported_assertions = int(unsupported["asserted"].sum())
    supported_total = len(supported_rows)
    unsupported_total = len(unsupported_rows)
    supported_correct = int(
        (
            supported["asserted"]
            & supported["response_correct"]
            & supported["state_correct"]
        ).sum()
    )
    unconditional_updates = (supported_total + unsupported_total) * (
        config.text_ticks + config.field_ticks
    )
    actual_updates = int(supported["eligible"].sum()) * (
        policy.fast_text_ticks + config.field_ticks
    ) + int(unsupported["eligible"].sum()) * (
        policy.fast_text_ticks + config.field_ticks
    )
    return {
        "policy": asdict(policy),
        "supported": {
            "assertion_accuracy": supported_correct / max(supported_assertions, 1),
            "assertions": supported_assertions,
            "cases": supported_total,
            "correct_assertion_rate": supported_correct / max(supported_total, 1),
            "coverage": supported_assertions / max(supported_total, 1),
        },
        "unsupported": {
            "assertions": unsupported_assertions,
            "cases": unsupported_total,
            "correct_abstention_rate": 1.0
            - unsupported_assertions / max(unsupported_total, 1),
            "flat_assertion_rate": int(flat_emits.sum()) / max(unsupported_total, 1),
        },
        "compute": {
            "actual_recurrent_updates": actual_updates,
            "reduction": 1.0 - actual_updates / max(unconditional_updates, 1),
            "unconditional_recurrent_updates": unconditional_updates,
        },
        "surface_memory": {
            "factorized_cells": len(vocabulary.tokens) * 2 * 7,
            "legacy_cells": len(vocabulary.tokens) * 6 * 7,
            "reduction": 2.0 / 3.0,
        },
    }


def _ablation_report(
    model: AtomNeuralLanguageField,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
) -> dict[str, Any]:
    sample = tuple(rows[:48])
    batch = collate_runtime_rows(sample, vocabulary, config)
    model.eval()
    with torch.no_grad():
        baseline = model(
            batch["token_ids"],
            batch["token_mask"],
            batch["node_features"],
            batch["adjacency"],
        )
        text: dict[str, Any] = {}
        field: dict[str, Any] = {}
        for index, name in enumerate(PROCESS_NAMES):
            ablated = model(
                batch["token_ids"],
                batch["token_mask"],
                batch["node_features"],
                batch["adjacency"],
                text_ablate=index,
            )
            delta = max(
                float((ablated["controls"] - baseline["controls"]).abs().max()),
                float((ablated["continuous"] - baseline["continuous"]).abs().max()),
                float(
                    (ablated["binary_logits"] - baseline["binary_logits"]).abs().max()
                ),
            )
            text[name] = {"causal": delta > 1e-7, "maximum_delta": delta}
        for index, name in enumerate(PROCESS_NAMES):
            ablated = model(
                batch["token_ids"],
                batch["token_mask"],
                batch["node_features"],
                batch["adjacency"],
                field_ablate=index,
            )
            delta = max(
                float((ablated["continuous"] - baseline["continuous"]).abs().max()),
                float(
                    (ablated["binary_logits"] - baseline["binary_logits"]).abs().max()
                ),
            )
            field[name] = {"causal": delta > 1e-7, "maximum_delta": delta}
    return {
        "all_field_operators_causal": all(row["causal"] for row in field.values()),
        "all_text_operators_causal": all(row["causal"] for row in text.values()),
        "field": field,
        "text": text,
    }


def _architecture_audit(model: AtomNeuralLanguageField) -> dict[str, Any]:
    model_source = Path(__file__).with_name("atom_neural_language_model.py")
    dataset_source = Path(__file__).with_name("atom_neural_language_dataset.py")
    runtime_source = Path(__file__)
    source = (model_source if model_source.exists() else runtime_source).read_text(
        encoding="utf-8"
    )
    dataset_text = (
        dataset_source if dataset_source.exists() else runtime_source
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    graph = build_neural_language_graph()
    checks = {
        "consequence_inducer_runtime_wired": "induce_control_from_consequence"
        in source,
        "evaluator_fields_absent_from_runtime_schema": set(RUNTIME_ROW_KEYS).isdisjoint(
            {
                "family",
                "global_features",
                "is_noise",
                "language",
                "process_signature",
                "query_type",
                "semantic_answer",
            }
        ),
        "frozen_root_field_kernel": all(
            not parameter.requires_grad for parameter in model.field_cell.parameters()
        ),
        "evidence_preflight_runtime_wired": "evidence_preflight" in source
        and "claim_status" in source,
        "factorized_surface_memory": tuple(model.surface_table.shape[1:]) == (2, 7),
        "graph_resolves_to_seven_primitives": set(
            graph.expand("lifelong_language_adapt")
        )
        == set(PROCESS_NAMES),
        "neural_field_class_present": "AtomNeuralLanguageField" in class_names,
        "operator_lattice_present": "TRAIN_COMPOSITIONS" in dataset_text
        and "HELDOUT_COMPOSITIONS" in dataset_text,
        "query_surface_memory_present": "QuerySurfaceMemory" in class_names,
        "recurrent_fast_path_present": "text_tick_budget" in source,
        "seven_text_branches_present": all(
            token in source
            for token in (
                "radiation",
                "gravitation",
                "attraction",
                "dissipation",
                "nucleation",
                "conserved",
                "decay",
            )
        ),
        "strict_loader_present": "load_neural_language_model" in source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "checks": checks,
        "failed": failed,
        "model_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "passed": not failed,
    }


def _reseal(payload: Mapping[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(dict(payload))
    base = {key: value for key, value in changed.items() if key != "model_hash"}
    changed["model_hash"] = neural_model_hash(base)
    return changed


def _corruption_checks(payload: Mapping[str, Any]) -> dict[str, Any]:
    corruptions: dict[str, dict[str, Any]] = {}
    unknown = copy.deepcopy(dict(payload))
    unknown["extra"] = True
    corruptions["unknown_field"] = unknown

    bad_hash = copy.deepcopy(dict(payload))
    bad_hash["model_hash"] = "0" * 64
    corruptions["model_hash"] = bad_hash

    duplicate_vocab = copy.deepcopy(dict(payload))
    duplicate_vocab["token_vocabulary"][1] = duplicate_vocab["token_vocabulary"][2]
    corruptions["duplicate_vocabulary"] = _reseal(duplicate_vocab)

    bad_shape = copy.deepcopy(dict(payload))
    tensor_name = next(iter(bad_shape["weights"]))
    bad_shape["weights"][tensor_name]["shape"] = [999]
    corruptions["weight_shape"] = _reseal(bad_shape)

    bad_surface = copy.deepcopy(dict(payload))
    bad_surface["weights"]["surface_table"]["values"][0] = (
        len(bad_surface["response_vocabulary"]) + 4
    )
    corruptions["surface_table_range"] = _reseal(bad_surface)

    bad_config = copy.deepcopy(dict(payload))
    bad_config["config"]["hidden_dim"] = 17
    corruptions["config"] = _reseal(bad_config)

    bad_policy = copy.deepcopy(dict(payload))
    bad_policy["inference_policy"]["minimum_query_support"] = 1.5
    corruptions["inference_policy"] = _reseal(bad_policy)

    checks: dict[str, bool] = {}
    for name, corruption in corruptions.items():
        try:
            load_neural_language_model(corruption)
        except (KeyError, TypeError, ValueError):
            checks[name] = True
        else:
            checks[name] = False
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "rejected": sum(checks.values()),
    }


def validate_neural_workflow_request(payload: Mapping[str, Any]) -> None:
    expected = {"request_id", "schema_version", "turns"}
    if set(payload) != expected:
        raise ValueError(f"workflow fields must be {sorted(expected)}")
    if payload["schema_version"] != NEURAL_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported workflow schema")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be non-empty text")
    turns = payload["turns"]
    if not isinstance(turns, list) or not 1 <= len(turns) <= 12:
        raise ValueError("turns must contain between 1 and 12 requests")
    for index, turn in enumerate(turns):
        expected_turn = {"adjacency", "node_features", "turn_id", "utterance"}
        if not isinstance(turn, Mapping) or set(turn) != expected_turn:
            raise ValueError(f"turn {index} fields must be {sorted(expected_turn)}")


def run_neural_workflow(
    model_payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    validate_neural_workflow_request(request)
    loaded = load_neural_language_model(model_payload)
    graph = build_neural_language_graph()
    context = retrieve_atom_context(
        graph,
        "evidence bound claim adaptive compute lifelong language consequence query surface response",
        limit=8,
    )
    turns: list[dict[str, Any]] = []
    for turn in request["turns"]:
        inference = run_neural_inference_request(
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
        "model_hash": loaded.model_hash,
        "request_id": request["request_id"],
        "runtime": {
            "neural_language_runtime": NEURAL_EXPERIMENT_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        },
        "schema_version": NEURAL_EXPERIMENT_SCHEMA,
        "turns": turns,
    }


def _build_workflow(
    supported_rows: Sequence[Mapping[str, Any]],
    unsupported_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supported = [supported_rows[index] for index in (0, 17, 35, 52, 71, 90)]
    unsupported = [unsupported_rows[index] for index in (0, 31)]
    selected = [
        *({"row": row, "should_assert": True} for row in supported),
        *({"row": row, "should_assert": False} for row in unsupported),
    ]
    request = {
        "request_id": "neural-language-evidence-bound-workflow",
        "schema_version": NEURAL_EXPERIMENT_SCHEMA,
        "turns": [
            {
                "adjacency": entry["row"]["adjacency"],
                "node_features": entry["row"]["node_features"],
                "turn_id": f"turn-{index + 1}",
                "utterance": entry["row"]["utterance"],
            }
            for index, entry in enumerate(selected)
        ],
    }
    return request, selected


def _score_workflow(
    response: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    correct = 0
    correct_abstentions = 0
    state_success = 0
    context_turns = 0
    supported_turns = 0
    unsupported_turns = 0
    for turn, expectation in zip(response["turns"], expected_rows, strict=True):
        expected = expectation["row"]
        artifact = turn["artifact"]
        if expectation["should_assert"]:
            supported_turns += 1
            response_correct = artifact["response"] == expected["response"]
            continuous = torch.tensor(artifact["continuous"])
            target_continuous = torch.tensor(expected["target_continuous"])
            binary = torch.tensor(artifact["binary_probability"]) >= 0.5
            target_binary = torch.tensor(expected["target_binary"]) >= 0.5
            state_correct = bool(
                (continuous - target_continuous).abs().mean() <= 0.16
                and torch.equal(binary, target_binary)
            )
            correct += int(response_correct and state_correct)
            state_success += int(state_correct)
        else:
            unsupported_turns += 1
            abstained = (
                artifact["claim_status"] == "unknown"
                and artifact["response"] is None
                and artifact["reasoning"]["execution_skipped"]
            )
            correct_abstentions += int(abstained)
        context_turns += int(bool(turn["knowledge_context"]))
    turn_count = len(expected_rows)
    joint_accuracy = correct / max(supported_turns, 1)
    return {
        "all_turns_have_graph_context": context_turns == turn_count,
        "correct_abstentions": correct_abstentions,
        "correct": correct,
        "joint_accuracy": joint_accuracy,
        "minimum_joint_accuracy": 0.80,
        "passed": joint_accuracy >= 0.80
        and state_success == supported_turns
        and correct_abstentions == unsupported_turns
        and context_turns == turn_count,
        "state_success": state_success,
        "supported_turns": supported_turns,
        "turns": turn_count,
        "unsupported_turns": unsupported_turns,
    }


def _controller_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    history = trace["controller_history"]
    action_counts = dict(sorted(Counter(row["action"] for row in history).items()))
    stage_counts: dict[str, dict[str, int]] = {}
    for stage in sorted({str(row["stage"]) for row in history}):
        rows = [row for row in history if row["stage"] == stage]
        stage_counts[stage] = {
            "accepted": sum(bool(row["accepted"]) for row in rows),
            "rejected": sum(not bool(row["accepted"]) for row in rows),
            "windows": len(rows),
        }
    return {
        "action_counts": action_counts,
        "maximum_chaos_load": trace["maximum_chaos_load"],
        "stage_counts": stage_counts,
        "windows": trace["windows"],
    }


def _experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluations = report["evaluations"]
    adaptive_transfer = evaluations["adaptive"]["transfer_composition"]
    fixed_transfer = evaluations["fixed"]["transfer_composition"]
    base_composition = evaluations["base"]["base_composition"]
    retention = evaluations["adaptive"]["base_composition"]
    flat_composition = evaluations["flat"]["base_composition"]
    zero_shot = evaluations["adaptive"]["zero_shot_composition"]
    controller = report["controller"]
    evidence = report["evidence_boundary"]
    sample_efficiency = report["sample_efficiency"]
    noise = controller["stage_counts"]["transfer_noise"]
    coherent_windows = (
        controller["stage_counts"]["transfer_adaptation"]["accepted"]
        + controller["stage_counts"]["transfer_recovery"]["accepted"]
    )
    coherent_total = (
        controller["stage_counts"]["transfer_adaptation"]["windows"]
        + controller["stage_counts"]["transfer_recovery"]["windows"]
    )
    checks = {
        "adaptive_beats_fixed_joint": adaptive_transfer["joint_accuracy"]
        >= fixed_transfer["joint_accuracy"] + 0.30,
        "adaptive_transfer_joint": adaptive_transfer["joint_accuracy"] >= 0.84,
        "adaptive_transfer_response": adaptive_transfer["response_accuracy"] >= 0.88,
        "adaptive_transfer_state": adaptive_transfer["state_accuracy"] >= 0.90,
        "all_field_operators_causal": report["ablations"]["all_field_operators_causal"],
        "all_text_operators_causal": report["ablations"]["all_text_operators_causal"],
        "architecture_audit": report["architecture_audit"]["passed"],
        "base_composition_joint": base_composition["joint_accuracy"] >= 0.80,
        "base_query_memory": report["training"]["base"]["atom"]["query_surface_memory"][
            "query_laws"
        ]
        == 12,
        "chaos_budget": controller["maximum_chaos_load"] <= 1.1500001,
        "coherent_windows_accepted": coherent_windows / coherent_total >= 0.60,
        "corruptions_rejected": report["corruption_checks"]["passed"],
        "dataset_audit": report["dataset"]["audit"]["passed"],
        "deterministic_training": report["deterministic_training"]["passed"],
        "evidence_bound_supported_accuracy": evidence["supported"]["assertion_accuracy"]
        >= 0.90,
        "evidence_bound_supported_coverage": evidence["supported"]["coverage"] >= 0.85,
        "evidence_bound_unknown_abstention": evidence["unsupported"][
            "correct_abstention_rate"
        ]
        == 1.0,
        "factorized_surface_memory": evidence["surface_memory"]["reduction"] >= 0.66,
        "flat_baseline_fails_composition": flat_composition["joint_accuracy"] <= 0.10,
        "flat_baseline_asserts_without_evidence": evidence["unsupported"][
            "flat_assertion_rate"
        ]
        == 1.0,
        "fixed_schedule_lags": fixed_transfer["joint_accuracy"] <= 0.60,
        "graph_rag_workflow": report["serialized_workflow"]["passed"],
        "noise_windows_rejected": noise["rejected"] / noise["windows"] >= 0.65,
        "raw_events_forgotten": report["adaptation"]["adaptive"]["raw_event_count"] == 0
        and report["adaptation"]["adaptive"]["lexicon_memory"]["raw_event_count"] == 0
        and report["adaptation"]["adaptive"]["query_surface_memory"]["raw_event_count"]
        == 0,
        "retention_joint": retention["joint_accuracy"] >= 0.85,
        "roundtrip_exact": report["serialization"]["roundtrip_exact"],
        "sample_efficient_composition": sample_efficiency["evaluation"][
            "joint_accuracy"
        ]
        >= 0.80,
        "sample_efficient_quarter_curriculum": sample_efficiency["example_fraction"]
        == 0.25,
        "self_tests": report["self_tests"]["passed"],
        "side_view_model_bound": report["side_view_contract"]["model_hash"]
        == report["model_hash"],
        "transfer_query_memory": len(
            report["adaptation"]["adaptive"]["query_surface_memory"]["query_laws"]
        )
        == 6,
        "unnecessary_recurrent_work_reduced": evidence["compute"]["reduction"] >= 0.50,
        "zero_shot_control_remains_negative": zero_shot["joint_accuracy"] <= 0.10,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}


def _run_primary_training(
    program: NeuralLanguageProgram,
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
    base, flat, base_training = _train_base_models(program, vocabulary, config, device)
    adaptive, fixed, adaptation = _adapt_models(
        base, program, vocabulary, config, device
    )
    return (
        base,
        adaptive,
        fixed,
        flat,
        {
            "adaptation": adaptation,
            "base": base_training,
        },
    )


def run_neural_language_experiment(output_dir: Path) -> dict[str, Any]:
    device = torch.device("cpu")
    program = build_neural_language_program()
    vocabulary = NeuralVocabulary.build(program.vocabulary, program.response_vocabulary)
    config = NeuralArchitectureConfig()
    config.validate()
    base, adaptive, fixed, flat, training = _run_primary_training(
        program, vocabulary, config, device
    )
    evaluations = _evaluate_all(
        base=base,
        adaptive=adaptive,
        fixed=fixed,
        flat=flat,
        program=program,
        vocabulary=vocabulary,
        config=config,
        device=device,
    )
    sample_efficiency = _sample_efficiency_probe(program, vocabulary, config, device)
    training_summary = {
        "adaptive_transfer": _metric_view(
            evaluations["adaptive"]["transfer_composition"]
        ),
        "base_composition": _metric_view(evaluations["base"]["base_composition"]),
        "controller": _controller_summary(training["adaptation"]["adaptive"]),
        "retention": _metric_view(evaluations["adaptive"]["base_composition"]),
        "training_seed": NEURAL_EXPERIMENT_SEED,
    }
    model_payload = neural_language_model_payload(
        adaptive,
        training_summary=training_summary,
    )
    loaded = load_neural_language_model(model_payload)
    roundtrip_exact = tensor_state_payload(loaded.model) == tensor_state_payload(
        adaptive
    )

    replay_base, replay_adaptive, _, _, _ = _run_primary_training(
        program, vocabulary, config, device
    )
    del replay_base
    replay_payload = neural_language_model_payload(
        replay_adaptive,
        training_summary=training_summary,
    )
    deterministic_training = {
        "first_model_hash": model_payload["model_hash"],
        "passed": replay_payload["model_hash"] == model_payload["model_hash"],
        "second_model_hash": replay_payload["model_hash"],
    }

    workflow_request, workflow_rows = _build_workflow(
        program.stages["transfer_composition"],
        program.stages["zero_shot_composition"],
    )
    workflow_response = run_neural_workflow(model_payload, workflow_request)
    workflow_score = _score_workflow(workflow_response, workflow_rows)
    evidence_boundary = _evidence_boundary_report(
        adaptive,
        flat,
        program.stages["transfer_composition"],
        program.stages["zero_shot_composition"],
        vocabulary,
        config,
    )
    ablations = _ablation_report(
        adaptive,
        program.stages["transfer_composition"],
        vocabulary,
        config,
    )
    architecture = _architecture_audit(adaptive)
    corruptions = _corruption_checks(model_payload)
    self_test_sections = {
        "dataset": neural_language_self_tests(),
        "model": neural_model_self_tests(),
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
    controller = _controller_summary(training["adaptation"]["adaptive"])
    report: dict[str, Any] = {
        "ablations": ablations,
        "adaptation": training["adaptation"],
        "architecture": "atom-neural-language-field-v2",
        "architecture_audit": architecture,
        "controller": controller,
        "corruption_checks": corruptions,
        "dataset": program.manifest,
        "deterministic_training": deterministic_training,
        "evidence_boundary": evidence_boundary,
        "evaluations": evaluations,
        "experiment": "atom-lifelong-neural-language-field-v2",
        "knowledge_runtime": {
            "rag_runtime": ATOM_RAG_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        },
        "model_hash": model_payload["model_hash"],
        "parameter_counts": {
            "atom": neural_parameter_count(adaptive),
            "flat": neural_parameter_count(flat),
        },
        "schema_version": NEURAL_EXPERIMENT_SCHEMA,
        "sample_efficiency": sample_efficiency,
        "self_tests": self_tests,
        "serialization": {"roundtrip_exact": roundtrip_exact},
        "serialized_workflow": workflow_score,
        "side_view_contract": {
            "binding": "render_neural_language_artifact",
            "model_hash": model_payload["model_hash"],
            "runtime": ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME,
        },
        "training": training,
    }
    report["experiment_gates"] = _experiment_gates(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_neural_json(
        output_dir / "atom_neural_language_dataset_manifest.json", program.manifest
    )
    for stage, rows in program.stages.items():
        write_neural_jsonl(output_dir / f"atom_neural_language_{stage}.jsonl", rows)
    write_neural_json(
        output_dir / "atom_neural_language_evaluator_truth.json",
        program.evaluator_truth,
    )
    write_neural_json(output_dir / "atom_neural_language_model.json", model_payload)
    write_neural_json(output_dir / "atom_neural_language_report.json", report)
    write_neural_json(
        output_dir / "atom_neural_language_workflow_request.json",
        workflow_request,
    )
    write_neural_json(
        output_dir / "atom_neural_language_workflow_response.json",
        workflow_response,
    )
    html = render_neural_language_artifact(model_payload, report, workflow_response)
    (output_dir / "atom_neural_language_side_view.html").write_text(
        html,
        encoding="utf-8",
        newline="\n",
    )
    report["artifacts"] = {
        "model": "atom_neural_language_model.json",
        "report": "atom_neural_language_report.json",
        "side_view": "atom_neural_language_side_view.html",
        "side_view_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "workflow_request": "atom_neural_language_workflow_request.json",
        "workflow_response": "atom_neural_language_workflow_response.json",
    }
    write_neural_json(output_dir / "atom_neural_language_report.json", report)
    return report


def parse_neural_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        Path("/kaggle/working/neural_language_outputs")
        if Path("/kaggle/working").exists()
        else Path("neural_language_outputs")
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_neural_args()
    if args.self_test:
        payload = {
            "dataset": neural_language_self_tests(),
            "model": neural_model_self_tests(),
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
        response = run_neural_workflow(model_payload, request)
        write_neural_json(args.response, response)
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0
    report = run_neural_language_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["experiment_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
