"""Run the deterministic Atom homeostatic-governor experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from atom_homeostatic_dataset import (
    build_homeostatic_program,
    write_homeostatic_program,
)
from atom_homeostatic_governor import (
    ATOM_HOMEOSTATIC_RUNTIME,
    HomeostaticPrimitive,
    evaluate_final_laws,
    evaluate_prequential,
    homeostatic_architecture_audit,
    homeostatic_hash,
    homeostatic_model_payload,
    load_homeostatic_model,
    run_homeostatic_request,
    run_homeostatic_self_tests,
    train_homeostatic_field,
    write_homeostatic_json,
)
from atom_homeostatic_side_view import (
    ATOM_HOMEOSTATIC_ARTIFACT_BINDING,
    ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME,
    render_homeostatic_artifact,
)
from atom_runtime_knowledge import ATOM_RAG_RUNTIME, ATOM_WIKI_GRAPH_RUNTIME


EXPERIMENT_NAME = "atom-homeostatic-criticality-governor-v1"


def _rehash_model(model: dict[str, Any]) -> None:
    core = deepcopy(model)
    core.pop("model_hash", None)
    model["model_hash"] = homeostatic_hash(core)


def _corruption_checks(model: Mapping[str, Any]) -> dict[str, Any]:
    corruptions: dict[str, Callable[[dict[str, Any]], None]] = {
        "model_hash": lambda item: item.__setitem__("model_hash", "0" * 64),
        "temperature_bound": lambda item: item["controller"].__setitem__(
            "temperature", 9.0
        ),
        "chaos_load": lambda item: item["controller"].__setitem__(
            "chaos_load", 9.0
        ),
        "history_order": lambda item: item["controller"]["history"][0].__setitem__(
            "window", 99
        ),
        "duplicate_law": lambda item: item["laws"].append(
            deepcopy(item["laws"][0])
        ),
        "raw_evidence": lambda item: item["training"].__setitem__(
            "raw_evidence_count", 1
        ),
        "unknown_field": lambda item: item.__setitem__("surprise_truth", "leak"),
    }
    checks: dict[str, bool] = {}
    for name, mutation in corruptions.items():
        candidate = deepcopy(dict(model))
        mutation(candidate)
        if name != "model_hash":
            _rehash_model(candidate)
        try:
            load_homeostatic_model(candidate)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rejected": sum(checks.values()),
    }


def _ablation_report(
    observations: list[dict[str, Any]],
    final_truth: Mapping[str, str],
    reference_accuracy: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for primitive in HomeostaticPrimitive:
        try:
            result = train_homeostatic_field(
                observations,
                adaptive=True,
                disabled=(primitive,),
            )
            final = evaluate_final_laws(result.runtime, final_truth)
            try:
                homeostatic_model_payload(result.runtime)
            except ValueError:
                serialization_rejected = True
            else:
                serialization_rejected = False
            rows[primitive.value] = {
                "disabled": primitive.value,
                "final_accuracy": final["accuracy"],
                "laws": len(result.runtime.state.committed),
                "reheats": result.runtime.state.reheats,
                "cools": result.runtime.state.cools,
                "raw_evidence_count": result.runtime.state.raw_evidence_count,
                "serialization_rejected": serialization_rejected,
                "causal_effect_observed": bool(
                    final["accuracy"] < reference_accuracy
                    or serialization_rejected
                ),
            }
        except (AssertionError, KeyError, RuntimeError, ValueError) as error:
            rows[primitive.value] = {
                "disabled": primitive.value,
                "training_rejected": True,
                "error": str(error),
                "causal_effect_observed": True,
            }
    return rows


def _regime_commitments(
    commitments: tuple[dict[str, Any], ...],
    evaluator_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in commitments:
        regime = str(evaluator_truth[str(item["event_id"])]["regime"])
        if item["previous"] is not None and item["previous"] != item["current"]:
            counts[regime] += 1
    return dict(sorted(counts.items()))


def _controller_summary(model: Mapping[str, Any]) -> dict[str, Any]:
    history = model["controller"]["history"]
    temperatures = [float(row["temperature_after"]) for row in history]
    phases = [float(row["phase_after"]) for row in history]
    thresholds = [float(row["threshold_after"]) for row in history]
    loads = [float(row["chaos_load"]) for row in history]
    actions = Counter(str(row["action"]) for row in history)
    proposed = int(model["training"]["proposed_uphill_moves"])
    accepted = int(model["training"]["accepted_uphill_moves"])
    return {
        "windows": len(history),
        "action_counts": dict(sorted(actions.items())),
        "temperature_range": [min(temperatures), max(temperatures)],
        "phase_strength_range": [min(phases), max(phases)],
        "nucleation_threshold_range": [min(thresholds), max(thresholds)],
        "maximum_chaos_load": max(loads),
        "uphill_acceptance_ratio": accepted / max(proposed, 1),
        "uphill_proposals": proposed,
        "uphill_accepts": accepted,
        "uphill_rejections": int(model["training"]["rejected_uphill_moves"]),
        "order_parameter_range": [
            min(float(row["order_parameter"]) for row in history),
            max(float(row["order_parameter"]) for row in history),
        ],
        "near_criticality_claimed": False,
        "self_organized_criticality_claimed": False,
    }


def _model_isolation(model: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(model, sort_keys=True)
    forbidden = (
        "initial_crystallization",
        "noise_burst",
        "recovery",
        "law_shift",
        "consolidation",
        "expected_effect",
        "is_noise",
    )
    findings = [item for item in forbidden if item in serialized]
    return {"passed": not findings, "findings": findings}


def homeostatic_experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    adaptive = report["comparison"]["adaptive"]
    fixed = report["comparison"]["fixed"]
    controller = report["controller"]
    actions = controller["action_counts"]
    ablations = report["primitive_ablations"]
    checks = {
        "runtime_rows_hide_evaluator_truth": report["dataset"][
            "evaluator_truth_separate"
        ],
        "adaptive_final_laws": adaptive["final"]["accuracy"] == 1.0,
        "fixed_schedule_control_fails_shift": fixed["final"]["accuracy"] <= 0.5,
        "governor_beats_fixed_control": adaptive["final"]["accuracy"]
        > fixed["final"]["accuracy"],
        "noise_is_rejected_without_replacement": adaptive["prequential"][
            "noise_burst"
        ]["accuracy"]
        == 1.0
        and adaptive["replacement_counts"].get("noise_burst", 0) == 0,
        "coherent_change_is_learned": adaptive["prequential"]["consolidation"][
            "accuracy"
        ]
        == 1.0
        and adaptive["replacement_counts"].get("law_shift", 0) == 4,
        "feedback_reheats": actions.get("reheat_coherent_shift", 0) > 0,
        "feedback_cools_noise": actions.get("cool_incoherent_disturbance", 0) > 0,
        "temperature_is_controlled": controller["temperature_range"][0]
        < controller["temperature_range"][1],
        "phase_strength_is_controlled": controller["phase_strength_range"][0]
        < controller["phase_strength_range"][1],
        "nucleation_threshold_is_controlled": controller[
            "nucleation_threshold_range"
        ][0]
        < controller["nucleation_threshold_range"][1],
        "both_uphill_outcomes_occur": controller["uphill_accepts"] > 0
        and controller["uphill_rejections"] > 0,
        "order_parameter_is_observed": 0.0
        <= controller["order_parameter_range"][0]
        <= controller["order_parameter_range"][1]
        <= 1.0,
        "chaos_budget_holds": controller["maximum_chaos_load"]
        <= report["model_config"]["chaos_budget"] + 1e-9,
        "scope_does_not_claim_criticality": not controller[
            "near_criticality_claimed"
        ]
        and not controller["self_organized_criticality_claimed"],
        "raw_evidence_is_forgotten": report["model_training"][
            "raw_evidence_count"
        ]
        == 0
        and report["model_training"]["raw_event_count"] == 0,
        "serialized_roundtrip_is_exact": report["serialization"][
            "roundtrip_exact"
        ],
        "strict_corruption_rejection": report["corruption_checks"]["passed"],
        "deterministic_replay": report["deterministic_replay"]["passed"],
        "all_primitives_are_causal": all(
            row["causal_effect_observed"] for row in ablations.values()
        ),
        "all_seven_primitives_run": set(report["model_training"]["operator_counts"])
        == {primitive.value for primitive in HomeostaticPrimitive}
        and all(report["model_training"]["operator_counts"].values()),
        "wiki_graph_and_rag_exercised": report["serialized_workflow"][
            "all_turns_have_context"
        ],
        "serialized_workflow": report["serialized_workflow"]["passed"],
        "model_excludes_evaluator_labels": report["model_isolation"]["passed"],
        "side_view_is_model_bound": report["side_view_contract"]["model_hash"]
        == report["model_hash"],
        "architecture_audit": report["architecture_audit"]["passed"],
        "self_tests": report["self_tests"]["passed"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def run_homeostatic_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_homeostatic_program(output_dir)
    program = build_homeostatic_program()
    observations = program["observations"]

    adaptive_start = time.perf_counter()
    adaptive = train_homeostatic_field(observations, adaptive=True)
    adaptive_seconds = time.perf_counter() - adaptive_start
    fixed_start = time.perf_counter()
    fixed = train_homeostatic_field(observations, adaptive=False)
    fixed_seconds = time.perf_counter() - fixed_start

    adaptive_final = evaluate_final_laws(adaptive.runtime, program["final_truth"])
    fixed_final = evaluate_final_laws(fixed.runtime, program["final_truth"])
    adaptive_prequential = evaluate_prequential(adaptive, program["evaluator_truth"])
    fixed_prequential = evaluate_prequential(fixed, program["evaluator_truth"])
    adaptive_model = homeostatic_model_payload(adaptive.runtime)
    fixed_model = homeostatic_model_payload(fixed.runtime)
    model_path = output_dir / "atom_homeostatic_model.json"
    write_homeostatic_json(model_path, adaptive_model)
    restored = load_homeostatic_model(
        json.loads(model_path.read_text(encoding="utf-8"))
    )

    workflow_request = {
        "schema_version": 1,
        "request_id": "homeostatic-final-laws",
        "queries": [
            {"turn_id": f"turn-{index + 1}", "cue": cue}
            for index, cue in enumerate(sorted(program["final_truth"]))
        ],
    }
    workflow_response = run_homeostatic_request(restored, workflow_request)
    request_path = output_dir / "atom_homeostatic_workflow_request.json"
    response_path = output_dir / "atom_homeostatic_workflow_response.json"
    write_homeostatic_json(request_path, workflow_request)
    write_homeostatic_json(response_path, workflow_response)
    workflow_correct = sum(
        turn["effect"] == program["final_truth"][turn["cue"]]
        for turn in workflow_response["turns"]
    )

    replay = train_homeostatic_field(observations, adaptive=True)
    replay_model = homeostatic_model_payload(replay.runtime)
    ablations = _ablation_report(
        observations,
        program["final_truth"],
        adaptive_final["accuracy"],
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "runtime": ATOM_HOMEOSTATIC_RUNTIME,
        "dataset": program["manifest"],
        "comparison": {
            "adaptive": {
                "training_seconds": adaptive_seconds,
                "final": adaptive_final,
                "prequential": adaptive_prequential,
                "replacement_counts": _regime_commitments(
                    adaptive.commitment_events,
                    program["evaluator_truth"],
                ),
                "model_hash": adaptive_model["model_hash"],
            },
            "fixed": {
                "training_seconds": fixed_seconds,
                "final": fixed_final,
                "prequential": fixed_prequential,
                "replacement_counts": _regime_commitments(
                    fixed.commitment_events,
                    program["evaluator_truth"],
                ),
                "model_hash": fixed_model["model_hash"],
            },
        },
        "controller": _controller_summary(adaptive_model),
        "model_config": adaptive_model["config"],
        "model_training": adaptive_model["training"],
        "model_hash": adaptive_model["model_hash"],
        "model_isolation": _model_isolation(adaptive_model),
        "serialization": {
            "roundtrip_exact": restored.payload == adaptive_model,
        },
        "corruption_checks": _corruption_checks(adaptive_model),
        "deterministic_replay": {
            "first_model_hash": adaptive_model["model_hash"],
            "second_model_hash": replay_model["model_hash"],
            "passed": replay_model == adaptive_model,
        },
        "primitive_ablations": ablations,
        "serialized_workflow": {
            "turns": len(workflow_response["turns"]),
            "correct": workflow_correct,
            "passed": workflow_correct == len(workflow_response["turns"]),
            "all_turns_have_context": all(
                turn["knowledge_context"] for turn in workflow_response["turns"]
            ),
        },
        "knowledge_runtime": {
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
        },
        "atomic_governor": {
            "radiation": "inject bounded reheating and phase pulses",
            "gravitation": "aggregate order, surprise, coherence, acceptance, and churn",
            "attraction_repulsion": "derive signed errors around target bands",
            "dissipation": "cool the field and damp control oscillation",
            "nucleation": "commit laws and sustained control changes",
            "conservation": "project evidence and chaos into bounded budgets",
            "decay": "remove stale windows and raw evidence",
        },
        "side_view_contract": {
            "runtime": ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME,
            "binding": ATOM_HOMEOSTATIC_ARTIFACT_BINDING,
            "model_hash": adaptive_model["model_hash"],
        },
        "architecture_audit": homeostatic_architecture_audit(),
        "self_tests": run_homeostatic_self_tests(),
    }
    report["experiment_gates"] = homeostatic_experiment_gates(report)
    side_document = render_homeostatic_artifact(
        adaptive_model,
        report,
        workflow_response,
    )
    side_path = output_dir / "atom_homeostatic_side_view.html"
    side_path.write_text(side_document, encoding="utf-8", newline="\n")
    report["artifacts"] = {
        "model": model_path.name,
        "workflow_request": request_path.name,
        "workflow_response": response_path.name,
        "side_view": side_path.name,
        "side_view_sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
    }
    write_homeostatic_json(output_dir / "atom_homeostatic_report.json", report)
    return report


def parse_homeostatic_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = (
        Path("/kaggle/working/homeostatic_outputs")
        if Path("/kaggle/working").exists()
        else Path("homeostatic_outputs")
    )
    parser.add_argument("--output-dir", type=Path, default=default)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_homeostatic_args()
    if args.self_test:
        result = run_homeostatic_self_tests()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    inference = any(item is not None for item in (args.model, args.request, args.response))
    if inference:
        if not all(item is not None for item in (args.model, args.request, args.response)):
            raise SystemExit("--model, --request, and --response are required together")
        model = load_homeostatic_model(
            json.loads(args.model.read_text(encoding="utf-8"))
        )
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = run_homeostatic_request(model, request)
        write_homeostatic_json(args.response, response)
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0
    report = run_homeostatic_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["experiment_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
