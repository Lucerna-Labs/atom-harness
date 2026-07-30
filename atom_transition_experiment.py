"""Run evaluator-separated emergent transition-law discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    build_language_graph,
    retrieve_atom_context,
)
from atom_transition_dataset import (
    TRANSITION_AGENTS,
    TRANSITION_LEXICON,
    TRANSITION_LOCATIONS,
    TRANSITION_OBJECTS,
    apply_evaluator_transition,
    build_transition_discovery_program,
    render_transition,
    transition_world,
)
from atom_transition_discovery import (
    ATOM_TRANSITION_RUNTIME,
    TRANSITION_MODEL_SCHEMA,
    TransitionConfig,
    TransitionPrimitive,
    apply_transition_text,
    evaluate_transition_rows,
    evaluator_law_mapping,
    generate_transition_text,
    run_transition_self_tests,
    run_transition_workflow,
    runtime_from_transition_model,
    surface_law_maps,
    train_transition_field,
    transition_architecture_audit,
    transition_hash,
    transition_model_payload,
    write_transition_json,
)
from atom_transition_side_view import (
    ATOM_TRANSITION_ARTIFACT_BINDING,
    ATOM_TRANSITION_SIDE_VIEW_RUNTIME,
    render_transition_artifact,
)


TRANSITION_EXPERIMENT_NAME = "atom_emergent_transition_law_discovery_v1"


def transition_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def exact_surface_transition_baseline(
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known = {str(row["text"]) for row in train}
    covered = [row for row in heldout if str(row["text"]) in known]
    return {
        "cases": len(heldout),
        "covered": len(covered),
        "coverage": len(covered) / max(1, len(heldout)),
    }


def fixed_predicate_delta_baseline(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    novel_labels: set[str],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        before = row["before"]
        after = row["after"]
        location_changes = [
            agent
            for agent in before["locations"]
            if before["locations"][agent] != after["locations"][agent]
        ]
        holder_changes = [
            item
            for item in before["holders"]
            if before["holders"][item] != after["holders"][item]
        ]
        fixed_predicate = None
        if len(location_changes) == 1 and not holder_changes:
            fixed_predicate = "MOVE"
        elif len(holder_changes) == 1 and not location_changes:
            item = holder_changes[0]
            old_holder = before["holders"][item]
            new_holder = after["holders"][item]
            if old_holder is None and new_holder is not None:
                fixed_predicate = "TAKE"
            elif (
                old_holder is not None
                and new_holder is not None
                and old_holder != new_holder
            ):
                fixed_predicate = "GIVE"
        label = str(truth[str(row["case_id"])]["semantic_label"])
        predictions.append(
            {
                "case_id": row["case_id"],
                "semantic_label": label,
                "fixed_predicate": fixed_predicate,
                "recognized": fixed_predicate is not None,
                "novel": label in novel_labels,
            }
        )
    novel = [row for row in predictions if row["novel"]]
    return {
        "cases": len(predictions),
        "recognized": sum(row["recognized"] for row in predictions),
        "coverage": sum(row["recognized"] for row in predictions)
        / max(1, len(predictions)),
        "novel_cases": len(novel),
        "novel_recognized": sum(row["recognized"] for row in novel),
        "novel_coverage": sum(row["recognized"] for row in novel)
        / max(1, len(novel)),
        "predictions": predictions,
    }


def transition_lexicon_score(runtime: Any) -> dict[str, Any]:
    surface_to_concept, _ = surface_law_maps(runtime.state.surface_laws)
    expected = {
        surface: concept for concept, surface in TRANSITION_LEXICON.items()
    }
    rows = [
        {
            "surface": surface,
            "expected": concept,
            "predicted": surface_to_concept.get(surface),
            "correct": surface_to_concept.get(surface) == concept,
        }
        for surface, concept in sorted(expected.items())
    ]
    extra = sorted(set(surface_to_concept) - set(expected))
    correct = sum(row["correct"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "unexpected_surfaces": extra,
        "rows": rows,
    }


def evaluator_semantics_absent_from_model(
    model: Mapping[str, Any],
    labels: Sequence[str],
) -> dict[str, Any]:
    strings: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            strings.append(value.lower())
        elif isinstance(value, Mapping):
            for key, child in value.items():
                strings.append(str(key).lower())
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for child in value:
                visit(child)

    visit(model)
    forbidden = {str(label).lower() for label in labels} | {
        "move",
        "take",
        "give",
    }
    found = sorted(forbidden & set(strings))
    return {"forbidden_exact_strings": found, "passed": not found}


def transition_corruption_checks(model: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def check(name: str, payload: Mapping[str, Any]) -> None:
        rejected = False
        error = None
        try:
            runtime_from_transition_model(payload)
        except (KeyError, TypeError, ValueError) as exc:
            rejected = True
            error = str(exc)
        cases.append({"name": name, "rejected": rejected, "error": error})

    extra = deepcopy(dict(model))
    extra["untrusted"] = True
    check("unknown_top_level_field", extra)

    bad_hash = deepcopy(dict(model))
    bad_hash["model_hash"] = "0" * 64
    check("model_hash_tampering", bad_hash)

    bad_effect = deepcopy(dict(model))
    bad_effect["transition_laws"][0]["effects"][0]["after"] = "unbounded"
    core = {key: value for key, value in bad_effect.items() if key != "model_hash"}
    bad_effect["model_hash"] = transition_hash(core)
    check("unsupported_effect_expression", bad_effect)

    bad_schema = deepcopy(dict(model))
    bad_schema["schema_version"] = 999
    core = {key: value for key, value in bad_schema.items() if key != "model_hash"}
    bad_schema["model_hash"] = transition_hash(core)
    check("unknown_schema", bad_schema)

    return {
        "cases": len(cases),
        "rejected": sum(row["rejected"] for row in cases),
        "passed": all(row["rejected"] for row in cases),
        "rows": cases,
    }


def run_transition_ablation(
    program: Mapping[str, Any],
    primitive: TransitionPrimitive,
    config: TransitionConfig,
) -> dict[str, Any]:
    runtime, _, diagnostics = train_transition_field(
        program["train"],
        config=config,
        disabled=(primitive,),
    )
    laws_before = len(runtime.state.transition_laws)
    surfaces_before = len(runtime.state.surface_laws)
    traces_before = len(runtime.state.traces)
    evidence_before = runtime.state.raw_evidence_count
    temperature_before = runtime.state.temperature
    conservation_before = runtime.state.conservation_applications
    runtime.forget_raw(f"ablation-forget-{primitive.value}")
    traces_after = len(runtime.state.traces)
    evidence_after = runtime.state.raw_evidence_count
    causal = {
        TransitionPrimitive.RADIATION: laws_before == 0 and surfaces_before == 0,
        TransitionPrimitive.GRAVITATION: laws_before == 0 and surfaces_before == 0,
        TransitionPrimitive.ATTRACTION_REPULSION: laws_before == 0,
        TransitionPrimitive.NUCLEATION: laws_before == 0 and surfaces_before == 0,
        TransitionPrimitive.CONSERVATION: conservation_before == 0,
        TransitionPrimitive.DISSIPATION: math.isclose(
            temperature_before,
            config.initial_temperature,
            abs_tol=1e-12,
        ),
        TransitionPrimitive.DECAY: traces_after == traces_before
        and evidence_after == evidence_before
        and traces_before > 0,
    }[primitive]
    return {
        "primitive": primitive.value,
        "surface_laws": surfaces_before,
        "transition_laws": laws_before,
        "unresolved_cases": len(diagnostics["unresolved_case_ids"]),
        "temperature": temperature_before,
        "conservation_applications": conservation_before,
        "raw_traces_before_forget": traces_before,
        "raw_traces_after_forget": traces_after,
        "raw_evidence_before_forget": evidence_before,
        "raw_evidence_after_forget": evidence_after,
        "causal_effect_observed": causal,
    }


def build_transition_workflow() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a0, _, a2, a3, _ = TRANSITION_AGENTS
    item = TRANSITION_OBJECTS[0]
    destination = TRANSITION_LOCATIONS[4]
    specifications = (
        ("relocate", {"actor": a0, "destination": destination}),
        ("exchange_locations", {"actor": a0, "other": a2}),
        ("acquire", {"actor": a0, "object": item}),
        (
            "transfer",
            {"actor": a0, "recipient": a3, "object": item},
        ),
        ("release", {"actor": a3, "object": item}),
    )
    world = transition_world()
    turns: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    for index, (label, participants) in enumerate(specifications, start=1):
        text = render_transition(label, participants)
        world = apply_evaluator_transition(label, participants, world)
        turn_id = f"transition-{index}"
        turns.append({"turn_id": turn_id, "text": text})
        expected.append(
            {
                "turn_id": turn_id,
                "semantic_label": label,
                "text": text,
                "world_after": world,
            }
        )
    return {
        "schema_version": TRANSITION_MODEL_SCHEMA,
        "request_id": "atom-transition-workflow-001",
        "world": transition_world(),
        "turns": turns,
    }, expected


def score_transition_workflow(
    response: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    targets = {str(row["turn_id"]): row for row in expected}
    rows: list[dict[str, Any]] = []
    for turn in response["turns"]:
        target = targets[str(turn["turn_id"])]
        passed = (
            turn["law_id"] == mapping[str(target["semantic_label"])]
            and turn["generated"] == target["text"]
            and turn["world_after"] == target["world_after"]
            and bool(turn["knowledge_context"])
        )
        rows.append(
            {
                "turn_id": turn["turn_id"],
                "semantic_label": target["semantic_label"],
                "law_id": turn["law_id"],
                "passed": passed,
            }
        )
    correct = sum(row["passed"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "passed": correct == len(rows),
        "turns": rows,
    }


def transition_experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = report["evaluation"]
    training = report["training"]
    baselines = report["baselines"]
    gates = {
        "architecture_uses_only_universe_core": report["architecture_audit"][
            "passed"
        ],
        "self_tests_pass": report["self_tests"]["passed"],
        "training_observations_hide_semantic_labels": not report[
            "dataset"
        ]["semantic_labels_in_runtime_observations"],
        "all_fifteen_entity_lexemes_crystallize": report["lexicon"]["accuracy"]
        == 1.0
        and not report["lexicon"]["unexpected_surfaces"],
        "five_latent_effect_laws_crystallize": training["transition_laws"] == 5,
        "validation_executes_and_identifies_every_law": evaluation["validation"][
            "execution_accuracy"
        ]
        == 1.0
        and evaluation["validation"]["law_accuracy"] == 1.0,
        "heldout_executes_every_unseen_utterance": evaluation["heldout"][
            "execution_accuracy"
        ]
        == 1.0,
        "heldout_recovers_stable_law_identities": evaluation["heldout"][
            "law_accuracy"
        ]
        == 1.0,
        "heldout_generation_roundtrips": evaluation["heldout"][
            "generation_accuracy"
        ]
        == 1.0,
        "novel_exchange_and_release_laws_execute": evaluation[
            "novel_transitions"
        ]["execution_accuracy"]
        == 1.0
        and evaluation["novel_transitions"]["law_accuracy"] == 1.0,
        "exact_surface_memory_cannot_cover_heldout": baselines["exact_surface"][
            "coverage"
        ]
        == 0.0,
        "fixed_predicate_inventory_misses_novel_laws": baselines[
            "fixed_predicate_delta"
        ]["novel_coverage"]
        == 0.0
        and baselines["fixed_predicate_delta"]["coverage"] < 1.0,
        "runtime_model_contains_no_evaluator_semantics": report[
            "semantic_separation"
        ]["passed"],
        "raw_episodes_and_evidence_are_forgotten": training[
            "raw_episodes_after_forget"
        ]
        == 0
        and training["raw_evidence_after_forget"] == 0,
        "all_seven_primitives_execute": set(training["operator_counts"])
        == set(UNIVERSE_PRIMITIVE_NAMES)
        and all(training["operator_counts"].values()),
        "all_seven_single_primitive_ablations_are_causal": all(
            row["causal_effect_observed"]
            for row in report["primitive_ablations"].values()
        ),
        "phase_mixing_and_thermal_annealing_are_active": report[
            "controlled_chaos"
        ]["cumulative_phase_energy"]
        > 0.0
        and report["controlled_chaos"]["accepted_worse_moves"] > 0,
        "serialized_model_reloads": report["serialized_model_reloads"],
        "all_corrupt_models_fail_closed": report["corruption_checks"]["passed"],
        "deterministic_replay_matches": report["deterministic_replay"],
        "stateful_serialized_workflow_passes": report["serialized_workflow"][
            "passed"
        ],
        "graph_rag_runs_inside_the_workflow": report["knowledge_runtime"][
            "passed"
        ],
        "side_view_is_bound_to_the_real_model": report["side_view_contract"][
            "model_hash"
        ]
        == report["primary_model_hash"],
    }
    return {
        "gates": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
    }


def run_transition_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_tests = run_transition_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Transition self-tests failed: {self_tests['failed']}")
    program = build_transition_discovery_program()
    for split in ("train", "validation", "heldout"):
        transition_write_jsonl(
            output_dir / f"atom_transition_{split}.jsonl",
            program[split],
        )
    for name in ("training_truth", "validation_truth", "evaluation_truth"):
        write_transition_json(output_dir / f"atom_transition_{name}.json", program[name])
    write_transition_json(
        output_dir / "atom_transition_dataset_manifest.json",
        program["manifest"],
    )

    config = TransitionConfig()
    runtime, history, diagnostics = train_transition_field(
        program["train"],
        config=config,
    )
    mapping = evaluator_law_mapping(
        runtime,
        program["train"],
        program["training_truth"],
    )
    validation = evaluate_transition_rows(
        runtime,
        program["validation"],
        program["validation_truth"],
        mapping,
    )
    heldout = evaluate_transition_rows(
        runtime,
        program["heldout"],
        program["evaluation_truth"],
        mapping,
    )
    novel_labels = set(program["manifest"]["novel_transition_labels"])
    novel_rows = [
        row
        for row in program["heldout"]
        if program["evaluation_truth"][str(row["case_id"])]["semantic_label"]
        in novel_labels
    ]
    novel_truth = {
        str(row["case_id"]): program["evaluation_truth"][str(row["case_id"])]
        for row in novel_rows
    }
    novel_evaluation = evaluate_transition_rows(
        runtime,
        novel_rows,
        novel_truth,
        mapping,
    )
    lexicon = transition_lexicon_score(runtime)
    observations = runtime.state.observations
    raw_episodes_before = len(runtime.state.traces)
    raw_evidence_before = runtime.state.raw_evidence_count
    operator_counts = dict(runtime.state.operator_counts)
    cumulative_phase_energy = runtime.state.cumulative_phase_energy
    maximum_phase_energy = runtime.state.maximum_phase_energy
    accepted_improving = runtime.state.accepted_improving_moves
    accepted_worse = runtime.state.accepted_worse_moves
    final_temperature = runtime.state.temperature
    transition_law_count = len(runtime.state.transition_laws)
    surface_law_count = len(runtime.state.surface_laws)
    runtime.forget_raw("final-cognitive-forget")
    model = transition_model_payload(runtime)
    model_path = output_dir / "atom_transition_model.json"
    write_transition_json(model_path, model)
    write_transition_json(output_dir / "atom_transition_history.json", history)
    restored = runtime_from_transition_model(json.loads(json.dumps(model)))
    post_forget = evaluate_transition_rows(
        restored,
        program["heldout"],
        program["evaluation_truth"],
        mapping,
    )
    serialized_model_reloads = (
        transition_model_payload(restored)["model_hash"] == model["model_hash"]
    )

    replay, replay_history, _ = train_transition_field(
        program["train"],
        config=config,
    )
    replay.forget_raw("final-cognitive-forget")
    replay_model = transition_model_payload(replay)
    deterministic_replay = (
        replay_model["model_hash"] == model["model_hash"]
        and transition_hash(replay_history) == transition_hash(history)
    )

    ablations = {
        primitive.value: run_transition_ablation(program, primitive, config)
        for primitive in TransitionPrimitive
    }
    exact_surface = exact_surface_transition_baseline(
        program["train"],
        program["heldout"],
    )
    fixed_delta = fixed_predicate_delta_baseline(
        program["heldout"],
        program["evaluation_truth"],
        novel_labels,
    )
    semantic_separation = evaluator_semantics_absent_from_model(
        model,
        sorted(program["evaluator_oracle"]["patterns"]),
    )
    corruption_checks = transition_corruption_checks(model)

    workflow_request, workflow_expected = build_transition_workflow()
    request_path = output_dir / "atom_transition_workflow_request.json"
    response_path = output_dir / "atom_transition_workflow_response.json"
    write_transition_json(request_path, workflow_request)
    workflow_response = run_transition_workflow(
        model_path,
        request_path,
        response_path,
    )
    workflow_score = score_transition_workflow(
        workflow_response,
        workflow_expected,
        mapping,
    )

    graph = build_language_graph()
    retrieval = retrieve_atom_context(
        graph,
        "ground learn remember forget abstract transition consequence law",
        limit=20,
    )
    retrieved_names = {str(row["name"]) for row in retrieval}
    workflow_rag = all(turn["knowledge_context"] for turn in workflow_response["turns"])
    knowledge_runtime = {
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "graph_hash": transition_hash(graph.manifest()),
        "retrieved": retrieval,
        "workflow_turns_with_rag": sum(
            bool(turn["knowledge_context"]) for turn in workflow_response["turns"]
        ),
        "passed": workflow_rag
        and {"ground", "learn", "remember", "forget", "abstract"}
        <= retrieved_names,
    }

    example = next(
        row
        for row in program["heldout"]
        if program["evaluation_truth"][str(row["case_id"])]["semantic_label"]
        == "exchange_locations"
    )
    interaction = apply_transition_text(restored, str(example["text"]), example["before"])
    generated = generate_transition_text(
        restored,
        str(interaction["law_id"]),
        interaction["slots"],
    )

    report: dict[str, Any] = {
        "schema_version": TRANSITION_MODEL_SCHEMA,
        "experiment": TRANSITION_EXPERIMENT_NAME,
        "primary_model_hash": model["model_hash"],
        "manifest": {
            "standard_neural_network": False,
            "gradient_descent": False,
            "backpropagation": False,
            "pretrained_model": False,
            "fixed_predicate_inventory_in_runtime": False,
            "semantic_labels_visible_to_runtime": False,
            "runtime": ATOM_TRANSITION_RUNTIME,
            "universe_primitives": list(UNIVERSE_PRIMITIVE_NAMES),
        },
        "dataset": program["manifest"],
        "self_tests": self_tests,
        "architecture_audit": transition_architecture_audit(),
        "lexicon": lexicon,
        "evaluator_law_mapping": mapping,
        "evaluation": {
            "validation": validation,
            "heldout": heldout,
            "novel_transitions": novel_evaluation,
            "post_forget_heldout": post_forget,
        },
        "training": {
            "observations": observations,
            "surface_laws": surface_law_count,
            "transition_laws": transition_law_count,
            "raw_episodes_before_forget": raw_episodes_before,
            "raw_evidence_before_forget": raw_evidence_before,
            "raw_episodes_after_forget": len(runtime.state.traces),
            "raw_evidence_after_forget": runtime.state.raw_evidence_count,
            "operator_counts": operator_counts,
            "history_hash": transition_hash(history),
        },
        "baselines": {
            "exact_surface": exact_surface,
            "fixed_predicate_delta": fixed_delta,
        },
        "semantic_separation": semantic_separation,
        "primitive_ablations": ablations,
        "controlled_chaos": {
            "initial_temperature": config.initial_temperature,
            "final_temperature": final_temperature,
            "cumulative_phase_energy": cumulative_phase_energy,
            "maximum_phase_energy": maximum_phase_energy,
            "accepted_improving_moves": accepted_improving,
            "accepted_worse_moves": accepted_worse,
        },
        "serialized_model_reloads": serialized_model_reloads,
        "corruption_checks": corruption_checks,
        "deterministic_replay": deterministic_replay,
        "serialized_workflow": workflow_score,
        "knowledge_runtime": knowledge_runtime,
        "side_view_interaction": {
            "case_id": example["case_id"],
            "utterance": example["text"],
            "law_id": interaction["law_id"],
            "evaluator_label": "exchange_locations",
            "slots": interaction["slots"],
            "effects": interaction["effects"],
            "generated": generated["text"],
            "world_before": interaction["world_before"],
            "world_after": interaction["world_after"],
        },
        "side_view_contract": {
            "runtime": ATOM_TRANSITION_SIDE_VIEW_RUNTIME,
            "binding": ATOM_TRANSITION_ARTIFACT_BINDING,
            "model_hash": model["model_hash"],
        },
        "model": model,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["experiment_gates"] = transition_experiment_gates(report)
    if not report["experiment_gates"]["passed"]:
        raise RuntimeError(
            "Transition experiment gates failed: "
            + str(report["experiment_gates"]["failed"])
        )
    report_path = output_dir / "atom_transition_report.json"
    write_transition_json(report_path, report)
    side_path = render_transition_artifact(
        report,
        model,
        output_dir / "atom_transition_side_view.html",
    )
    report["artifact_side_view"] = {
        "path": side_path.name,
        "sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
        "runtime_marker": ATOM_TRANSITION_SIDE_VIEW_RUNTIME,
        "binding_marker": ATOM_TRANSITION_ARTIFACT_BINDING,
        "model_hash_bound": model["model_hash"],
    }
    write_transition_json(report_path, report)
    return report


def parse_transition_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transition_outputs"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def transition_main() -> None:
    args = parse_transition_args()
    result = (
        run_transition_self_tests()
        if args.self_test
        else run_transition_experiment(args.output_dir.resolve())
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    transition_main()
