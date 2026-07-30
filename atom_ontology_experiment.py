"""End-to-end experiment for alias-invariant ontology discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_ontology_dataset import (
    ONTOLOGY_RELATION_ALIASES,
    apply_ontology_evaluator_transition,
    build_ontology_discovery_program,
    opaque_world,
    render_ontology_transition,
)
from atom_ontology_discovery import (
    ATOM_ONTOLOGY_RUNTIME,
    ONTOLOGY_MODEL_SCHEMA,
    OntologyPrimitive,
    discover_ontology,
    evaluate_ontology_rows,
    evaluator_ontology_law_mapping,
    ontology_architecture_audit,
    ontology_hash,
    ontology_model_payload,
    run_ontology_self_tests,
    run_ontology_workflow,
    runtime_from_ontology_model,
    train_ontology_field,
    write_ontology_json,
)
from atom_ontology_side_view import (
    ATOM_ONTOLOGY_ARTIFACT_BINDING,
    ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME,
    render_ontology_artifact,
)
from atom_runtime_knowledge import ATOM_RAG_RUNTIME, ATOM_WIKI_GRAPH_RUNTIME


def write_ontology_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def exact_surface_baseline(
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seen = {str(row["text"]) for row in train}
    recognized = sum(str(row["text"]) in seen for row in heldout)
    return {
        "cases": len(heldout),
        "recognized": recognized,
        "accuracy": recognized / max(len(heldout), 1),
    }


def fixed_schema_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = sum(set(row["before"]) == {"locations", "holders"} for row in rows)
    return {
        "cases": len(rows),
        "accepted": accepted,
        "accuracy": accepted / max(len(rows), 1),
    }


def alias_memorization_baseline(
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known = {alias for row in train for alias in row["before"]}
    recognized = sum(set(row["before"]) <= known for row in heldout)
    return {
        "known_aliases": sorted(known),
        "cases": len(heldout),
        "recognized": recognized,
        "accuracy": recognized / max(len(heldout), 1),
    }


def ontology_lexicon_score(runtime: Any, oracle: Mapping[str, str]) -> dict[str, Any]:
    learned = {law.entity: law.surface for law in runtime.surface_laws}
    correct = sum(learned.get(entity) == surface for entity, surface in oracle.items())
    return {
        "expected": len(oracle),
        "learned": len(learned),
        "correct": correct,
        "accuracy": correct / max(len(oracle), 1),
    }


def model_isolation_check(
    model: Mapping[str, Any],
    relation_aliases: Sequence[str],
) -> dict[str, Any]:
    forbidden_exact = set(relation_aliases) | {
        "relocate",
        "acquire",
        "transfer",
        "exchange_locations",
        "release",
        "locations",
        "holders",
    }
    forbidden_prefixes = ("agent-", "object-", "location-")
    findings: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(str(key), path + ".key")
                visit(child, path + "." + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str):
            normalized = value.lower()
            if normalized in forbidden_exact or normalized.startswith(
                forbidden_prefixes
            ):
                findings.append(f"{path}={value}")

    visit(model, "model")
    return {
        "forbidden_relation_aliases": sorted(relation_aliases),
        "findings": findings,
        "passed": not findings,
    }


def _rehash_model(model: dict[str, Any]) -> None:
    core = {key: value for key, value in model.items() if key != "model_hash"}
    model["model_hash"] = ontology_hash(core)


def ontology_corruption_checks(model: Mapping[str, Any]) -> dict[str, Any]:
    outcomes: dict[str, bool] = {}

    def rejected(name: str, candidate: dict[str, Any]) -> None:
        try:
            runtime_from_ontology_model(candidate)
        except (KeyError, TypeError, ValueError):
            outcomes[name] = True
        else:
            outcomes[name] = False

    bad_hash = deepcopy(model)
    bad_hash["model_hash"] = "0" * 64
    rejected("model_hash", bad_hash)

    bad_type = deepcopy(model)
    bad_type["ontology"]["types"][0]["type_id"] = "type-corrupt0000"
    _rehash_model(bad_type)
    rejected("type_identity", bad_type)

    bad_relation = deepcopy(model)
    bad_relation["ontology"]["relations"][0]["relation_id"] = "relation-corrupt"
    _rehash_model(bad_relation)
    rejected("relation_identity", bad_relation)

    raw_evidence = deepcopy(model)
    raw_evidence["raw_evidence_count"] = 1
    _rehash_model(raw_evidence)
    rejected("raw_evidence", raw_evidence)

    bad_effect = deepcopy(model)
    bad_effect["transition_laws"][0]["effects"][0]["relation_id"] = "relation-unknown"
    _rehash_model(bad_effect)
    rejected("effect_relation", bad_effect)
    return {
        "checks": outcomes,
        "rejected": sum(outcomes.values()),
        "passed": bool(outcomes) and all(outcomes.values()),
    }


def run_ontology_ablation(
    primitive: OntologyPrimitive,
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    evaluator_mapping: Mapping[str, str],
) -> dict[str, Any]:
    runtime = train_ontology_field(train, disabled=(primitive,))
    evaluation = evaluate_ontology_rows(runtime, heldout, truth, evaluator_mapping)
    serialization_rejected = False
    try:
        ontology_model_payload(runtime)
    except ValueError:
        serialization_rejected = True
    causal = (
        evaluation["execution_accuracy"] < 1.0
        or runtime.state.raw_evidence_count > 0
        or runtime.state.conservation_applications == 0
        or runtime.state.ontology is None
        or serialization_rejected
    )
    return {
        "disabled": primitive.value,
        "surface_laws": len(runtime.surface_laws),
        "transition_laws": len(runtime.transition_laws),
        "execution_accuracy": evaluation["execution_accuracy"],
        "raw_evidence_count": runtime.state.raw_evidence_count,
        "conservation_applications": runtime.state.conservation_applications,
        "ontology_nucleated": runtime.state.ontology is not None,
        "serialization_rejected": serialization_rejected,
        "causal_effect_observed": causal,
    }


def build_ontology_workflow() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = "heldout"
    world = opaque_world(split)
    specifications = (
        ("relocate", {"actor": "n00", "destination": "n12"}),
        ("acquire", {"actor": "n00", "object": "n05"}),
        (
            "transfer",
            {"actor": "n00", "recipient": "n01", "object": "n05"},
        ),
        ("exchange_locations", {"actor": "n01", "other": "n02"}),
        ("release", {"actor": "n01", "object": "n05"}),
    )
    turns: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    for index, (label, participants) in enumerate(specifications, start=1):
        text = render_ontology_transition(label, participants)
        world = apply_ontology_evaluator_transition(label, participants, world, split)
        turn_id = f"ontology-turn-{index}"
        turns.append({"turn_id": turn_id, "text": text})
        expected.append(
            {
                "turn_id": turn_id,
                "text": text,
                "semantic_label": label,
                "world_after": deepcopy(world),
            }
        )
    request = {
        "schema_version": ONTOLOGY_MODEL_SCHEMA,
        "request_id": "opaque-heldout-workflow-v1",
        "world": opaque_world(split),
        "turns": turns,
    }
    return request, expected


def score_ontology_workflow(
    response: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    evaluator_mapping: Mapping[str, str],
) -> dict[str, Any]:
    correct = 0
    bindings_are_heldout = True
    heldout_aliases = set(ONTOLOGY_RELATION_ALIASES["heldout"].values())
    train_aliases = set(ONTOLOGY_RELATION_ALIASES["train"].values())
    for actual, target in zip(response["turns"], expected, strict=True):
        binding_aliases = set(actual["ontology_binding"])
        bindings_are_heldout &= (
            binding_aliases == heldout_aliases and not binding_aliases & train_aliases
        )
        if (
            actual["turn_id"] == target["turn_id"]
            and actual["text"] == target["text"]
            and actual["generated"] == target["text"]
            and actual["law_id"] == evaluator_mapping[target["semantic_label"]]
            and actual["world_after"] == target["world_after"]
            and actual["knowledge_context"]
        ):
            correct += 1
    return {
        "turns": len(expected),
        "correct": correct,
        "heldout_alias_binding": bindings_are_heldout,
        "passed": correct == len(expected) and bindings_are_heldout,
    }


def ontology_experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    validation = report["evaluation"]["validation"]
    heldout = report["evaluation"]["heldout"]
    novel = report["evaluation"]["novel_transitions"]
    checks = {
        "self_tests": report["self_tests"]["passed"],
        "architecture_audit": report["architecture_audit"]["passed"],
        "runtime_observations_hide_evaluator_semantics": not report["dataset"][
            "semantic_labels_in_runtime_observations"
        ],
        "entity_identifiers_are_opaque": not report["dataset"][
            "typed_entity_prefixes_present"
        ],
        "relation_aliases_are_disjoint": report["dataset"]["relation_alias_overlap"]
        == 0,
        "heldout_surfaces_are_unseen": report["dataset"]["heldout_surface_overlap"]
        == 0,
        "three_type_atoms_discovered": report["ontology"]["type_atoms"] == 3,
        "two_relation_atoms_discovered": report["ontology"]["relation_atoms"] == 2,
        "split_ontology_signatures_match": report["ontology"]["split_signatures_match"],
        "model_excludes_raw_aliases_and_semantic_names": report["model_isolation"][
            "passed"
        ],
        "lexicon_is_grounded": report["lexicon"]["accuracy"] == 1.0,
        "five_effect_laws_discovered": report["training"]["transition_laws"] == 5,
        "validation_execution": validation["execution_accuracy"] == 1.0,
        "validation_law_identity": validation["law_accuracy"] == 1.0,
        "validation_generation": validation["generation_accuracy"] == 1.0,
        "heldout_execution": heldout["execution_accuracy"] == 1.0,
        "heldout_law_identity": heldout["law_accuracy"] == 1.0,
        "heldout_generation": heldout["generation_accuracy"] == 1.0,
        "novel_effects_execute": novel["execution_correct"] == novel["cases"] == 10,
        "exact_surface_baseline_fails": report["baselines"]["exact_surface"][
            "recognized"
        ]
        == 0,
        "fixed_schema_baseline_fails": report["baselines"]["fixed_schema"]["accepted"]
        == 0,
        "alias_memorization_baseline_fails": report["baselines"]["alias_memorization"][
            "recognized"
        ]
        == 0,
        "strict_corruption_rejection": report["corruption_checks"]["passed"],
        "all_primitives_are_causal": all(
            row["causal_effect_observed"]
            for row in report["primitive_ablations"].values()
        ),
        "raw_evidence_is_forgotten": report["training"]["raw_evidence_count"] == 0
        and report["training"]["raw_episode_count"] == 0,
        "controlled_chaos_is_active": report["controlled_chaos"][
            "cumulative_phase_energy"
        ]
        > 0.0
        and report["controlled_chaos"]["accepted_worse_moves"] > 0,
        "serialized_roundtrip_is_exact": report["serialization"]["roundtrip_exact"],
        "deterministic_replay": report["deterministic_replay"]["passed"],
        "heldout_multiturn_workflow": report["serialized_workflow"]["passed"],
        "wiki_graph_and_rag_exercised": report["knowledge_runtime"][
            "all_turns_have_context"
        ],
        "side_view_is_model_bound": report["side_view_contract"]["model_hash"]
        == report["model_hash"],
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def run_ontology_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    program = build_ontology_discovery_program()
    for split in ("train", "validation", "heldout"):
        write_ontology_jsonl(
            output_dir / f"atom_ontology_{split}.jsonl", program[split]
        )
    write_ontology_json(
        output_dir / "atom_ontology_dataset_manifest.json", program["manifest"]
    )
    write_ontology_json(
        output_dir / "atom_ontology_training_truth.json", program["training_truth"]
    )
    write_ontology_json(
        output_dir / "atom_ontology_validation_truth.json", program["validation_truth"]
    )
    write_ontology_json(
        output_dir / "atom_ontology_evaluation_truth.json", program["evaluation_truth"]
    )

    started = time.perf_counter()
    runtime = train_ontology_field(program["train"])
    training_seconds = time.perf_counter() - started
    evaluator_mapping = evaluator_ontology_law_mapping(
        runtime, program["train"], program["training_truth"]
    )
    validation = evaluate_ontology_rows(
        runtime, program["validation"], program["validation_truth"], evaluator_mapping
    )
    heldout = evaluate_ontology_rows(
        runtime, program["heldout"], program["evaluation_truth"], evaluator_mapping
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
    novel = evaluate_ontology_rows(runtime, novel_rows, novel_truth, evaluator_mapping)

    model = ontology_model_payload(runtime)
    model_path = output_dir / "atom_ontology_model.json"
    write_ontology_json(model_path, model)
    restored = runtime_from_ontology_model(
        json.loads(model_path.read_text(encoding="utf-8"))
    )
    restored_payload = ontology_model_payload(restored)
    split_signatures = {
        split: discover_ontology(program[split][0]["before"]).signature
        for split in ("train", "validation", "heldout")
    }
    aliases = [
        alias
        for values in program["manifest"]["relation_aliases"].values()
        for alias in values
    ]
    corruption = ontology_corruption_checks(model)
    isolation = model_isolation_check(model, aliases)
    ablations = {
        primitive.value: run_ontology_ablation(
            primitive,
            program["train"],
            program["heldout"],
            program["evaluation_truth"],
            evaluator_mapping,
        )
        for primitive in OntologyPrimitive
    }

    request, expected_workflow = build_ontology_workflow()
    request_path = output_dir / "atom_ontology_workflow_request.json"
    response_path = output_dir / "atom_ontology_workflow_response.json"
    write_ontology_json(request_path, request)
    workflow = run_ontology_workflow(model_path, request_path, response_path)
    workflow_score = score_ontology_workflow(
        workflow, expected_workflow, evaluator_mapping
    )
    write_ontology_json(
        output_dir / "atom_ontology_knowledge_graph.json", runtime.knowledge.manifest()
    )

    replay = train_ontology_field(program["train"])
    replay_model = ontology_model_payload(replay)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "atom-emergent-ontology-discovery-v1",
        "runtime": ATOM_ONTOLOGY_RUNTIME,
        "model_hash": model["model_hash"],
        "dataset": program["manifest"],
        "ontology": {
            "signature": model["ontology"]["signature"],
            "type_atoms": len(model["ontology"]["types"]),
            "relation_atoms": len(model["ontology"]["relations"]),
            "split_signatures": split_signatures,
            "split_signatures_match": len(set(split_signatures.values())) == 1,
        },
        "training": {
            "seconds": training_seconds,
            "observations": runtime.state.observations,
            "surface_laws": len(runtime.surface_laws),
            "transition_laws": len(runtime.transition_laws),
            "raw_episode_count": len(runtime.state.traces),
            "raw_evidence_count": runtime.state.raw_evidence_count,
            "forgotten": runtime.state.forgotten,
            "operator_counts": dict(sorted(runtime.state.operator_counts.items())),
            "outcome_counts": dict(sorted(runtime.state.outcome_counts.items())),
        },
        "lexicon": ontology_lexicon_score(
            runtime, program["evaluator_oracle"]["concept_to_surface"]
        ),
        "evaluator_law_mapping": evaluator_mapping,
        "evaluation": {
            "validation": validation,
            "heldout": heldout,
            "novel_transitions": novel,
        },
        "baselines": {
            "exact_surface": exact_surface_baseline(
                program["train"], program["heldout"]
            ),
            "fixed_schema": fixed_schema_baseline(program["heldout"]),
            "alias_memorization": alias_memorization_baseline(
                program["train"], program["heldout"]
            ),
        },
        "model_isolation": isolation,
        "corruption_checks": corruption,
        "primitive_ablations": ablations,
        "controlled_chaos": {
            "temperature": runtime.state.temperature,
            "energy": runtime.state.energy,
            "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
            "maximum_phase_energy": runtime.state.maximum_phase_energy,
            "accepted_improving_moves": runtime.state.accepted_improving_moves,
            "accepted_worse_moves": runtime.state.accepted_worse_moves,
        },
        "serialization": {
            "roundtrip_exact": restored_payload == model,
            "isolation_findings": isolation["findings"],
        },
        "deterministic_replay": {
            "first_model_hash": model["model_hash"],
            "second_model_hash": replay_model["model_hash"],
            "passed": replay_model == model,
        },
        "serialized_workflow": workflow_score,
        "knowledge_runtime": {
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "all_turns_have_context": all(
                turn["knowledge_context"] for turn in workflow["turns"]
            ),
        },
        "side_view_contract": {
            "runtime": ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME,
            "binding": ATOM_ONTOLOGY_ARTIFACT_BINDING,
            "model_hash": model["model_hash"],
        },
        "self_tests": run_ontology_self_tests(),
        "architecture_audit": ontology_architecture_audit(),
    }
    report["experiment_gates"] = ontology_experiment_gates(report)
    side_document = render_ontology_artifact(model, report, workflow)
    side_path = output_dir / "atom_ontology_side_view.html"
    side_path.write_text(side_document, encoding="utf-8", newline="\n")
    report["artifacts"] = {
        "model": model_path.name,
        "workflow_request": request_path.name,
        "workflow_response": response_path.name,
        "side_view": side_path.name,
        "side_view_sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
    }
    write_ontology_json(output_dir / "atom_ontology_report.json", report)
    return report


def parse_ontology_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = (
        Path("/kaggle/working/ontology_outputs")
        if Path("/kaggle/working").exists()
        else Path("ontology_outputs")
    )
    parser.add_argument("--output-dir", type=Path, default=default)
    return parser.parse_args()


def ontology_main() -> None:
    args = parse_ontology_args()
    report = run_ontology_experiment(args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    ontology_main()
