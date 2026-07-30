"""Run the disjoint-lexicon compositional transfer experiment on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_language_dataset import build_grounded_language_program
from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    LANGUAGE_MODEL_SCHEMA,
    UNIVERSE_PRIMITIVE_NAMES,
    LanguageConfig,
    adapt_language_field,
    architecture_audit,
    character_span_f1,
    evaluate_language_rows,
    interact_text,
    language_model_payload,
    lexeme_maps,
    make_frame,
    run_language_field_self_tests,
    run_language_workflow,
    runtime_from_language_model,
    stable_hash,
    train_language_field,
)
from atom_language_side_view import (
    ATOM_LANGUAGE_ARTIFACT_BINDING,
    ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_language_artifact,
)
from atom_language_transfer_dataset import (
    TRANSFER_AGENTS,
    TRANSFER_LEXICON,
    TRANSFER_LOCATIONS,
    TRANSFER_OBJECTS,
    build_language_transfer_program,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_language_graph,
    retrieve_atom_context,
)


EXPERIMENT_NAME = "atom_language_disjoint_lexicon_transfer_v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_transfer_self_tests() -> dict[str, Any]:
    field = run_language_field_self_tests()
    base = build_grounded_language_program()
    transfer = build_language_transfer_program()
    base_concepts = set(base["evaluator_oracle"]["concept_to_surface"])
    transfer_concepts = set(transfer["evaluator_oracle"]["concept_to_surface"])
    grounding_ids = {row["case_id"] for row in transfer["grounding"]}
    heldout_ids = {row["case_id"] for row in transfer["heldout"]}
    checks = {
        "field_self_tests": field["passed"],
        "transfer_counts_are_12_2_48": transfer["manifest"]["counts"]
        == {"grounding": 12, "transient": 2, "heldout": 48},
        "transfer_uses_nine_new_concepts": len(transfer_concepts) == 9,
        "base_and_transfer_concepts_are_disjoint": not base_concepts
        & transfer_concepts,
        "grounding_and_heldout_are_disjoint": not grounding_ids & heldout_ids,
        "grounding_has_no_gold_meanings": all(
            "frame" not in row and "family" not in row for row in transfer["grounding"]
        ),
        "truth_is_evaluator_only": set(transfer["evaluation_truth"]) == heldout_ids,
        "heldout_actions_do_not_repeat_grounding_surfaces": transfer["manifest"][
            "heldout_action_surface_overlap"
        ]
        == 0,
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
        "field": field,
    }


def exact_demo_baseline(
    grounding: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    known = {str(row["text"]) for row in grounding}
    covered = sum(str(row["text"]) in known for row in heldout)
    return {
        "cases": len(heldout),
        "covered": covered,
        "coverage": covered / len(heldout),
        "grounded_accuracy": 0.0,
    }


def _lexicon_score(runtime: Any, oracle: Mapping[str, str]) -> dict[str, Any]:
    surface_to_concept, _ = lexeme_maps(runtime.state.lexeme_laws)
    rows = [
        {
            "surface": surface,
            "expected": concept,
            "predicted": surface_to_concept.get(surface),
            "correct": surface_to_concept.get(surface) == concept,
        }
        for surface, concept in sorted(oracle.items())
    ]
    correct = sum(row["correct"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "rows": rows,
        "noise_surface_absent": "florp" not in surface_to_concept,
        "corrected_lumi_binding": surface_to_concept.get("lumi") == "agent-4",
    }


def _stage_transfer(
    base_program: Mapping[str, Any],
    transfer_program: Mapping[str, Any],
    stage: str,
    config: LanguageConfig,
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    base_runtime, base_history, _ = train_language_field(
        base_program["train"], stage, config=config
    )
    base_runtime.abstract(f"{stage}-base-transfer-source-abstract")
    base_model = language_model_payload(base_runtime)

    frozen = runtime_from_language_model(json.loads(json.dumps(base_model)))
    frozen_transfer = evaluate_language_rows(
        frozen,
        transfer_program["heldout"],
        transfer_program["evaluation_truth"],
    )

    runtime = runtime_from_language_model(json.loads(json.dumps(base_model)))
    adaptation_history, diagnostics = adapt_language_field(
        runtime,
        transfer_program["grounding"],
        transient_rows=transfer_program["transient"],
        bootstrap_epochs=2,
        grounded_epochs=3,
        adaptation_id=f"transfer-{stage}",
    )
    transfer_before = evaluate_language_rows(
        runtime,
        transfer_program["heldout"],
        transfer_program["evaluation_truth"],
    )
    base_retention = evaluate_language_rows(
        runtime,
        base_program["heldout"],
        base_program["evaluation_truth"],
    )
    span_score = character_span_f1(runtime, transfer_program["heldout"])
    lexicon = _lexicon_score(
        runtime,
        transfer_program["evaluator_oracle"]["surface_to_concept"],
    )
    transfer_phase_energy = runtime.state.cumulative_phase_energy
    accepted_improving = runtime.state.accepted_improving_moves
    accepted_worse = runtime.state.accepted_worse_moves
    operator_counts = dict(runtime.state.operator_counts)
    raw_before = len(runtime.state.traces)
    transient_trace_retained = any(
        "florp" in row.tokens for row in runtime.state.traces
    )

    runtime.abstract(f"{stage}-transfer-final-abstract")
    adapted_model = language_model_payload(runtime)
    restored = runtime_from_language_model(json.loads(json.dumps(adapted_model)))
    post_transfer = evaluate_language_rows(
        restored,
        transfer_program["heldout"],
        transfer_program["evaluation_truth"],
    )
    post_base = evaluate_language_rows(
        restored,
        base_program["heldout"],
        base_program["evaluation_truth"],
    )
    result = {
        "stage": stage,
        "frozen_base_heldout": frozen_transfer,
        "heldout": transfer_before,
        "post_abstraction_heldout": post_transfer,
        "base_retention": base_retention,
        "post_abstraction_base_retention": post_base,
        "character_spans": span_score,
        "lexicon": lexicon,
        "adaptation": {
            **diagnostics,
            "raw_episodes_before_abstraction": raw_before,
            "raw_episodes_after_abstraction": len(runtime.state.traces),
            "raw_evidence_after_abstraction": (
                len(runtime.state.association_evidence)
                + len(runtime.state.template_evidence)
                + len(runtime.state.reference_evidence)
            ),
            "transient_noise_trace_retained": transient_trace_retained,
            "lexeme_laws": len(runtime.state.lexeme_laws),
            "frame_laws": len(runtime.state.frame_laws),
            "reference_laws": len(runtime.state.reference_laws),
            "character_span_laws": len(
                [row for row in runtime.state.character_span_laws if row.active]
            ),
            "operator_counts": operator_counts,
            "cumulative_phase_energy": transfer_phase_energy,
            "accepted_improving_moves": accepted_improving,
            "accepted_worse_moves": accepted_worse,
            "history_hash": stable_hash(adaptation_history),
            "base_history_hash": stable_hash(base_history),
        },
        "base_model_hash": base_model["model_hash"],
        "adapted_model_hash": adapted_model["model_hash"],
        "serialized_model_reloads": language_model_payload(restored)["model_hash"]
        == adapted_model["model_hash"],
    }
    return runtime, result, adapted_model, adaptation_history


def build_transfer_workflow(
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a0, a1, a2 = TRANSFER_AGENTS
    item = TRANSFER_OBJECTS[2]
    l0, _, l2 = TRANSFER_LOCATIONS
    world = {
        "locations": {
            a0: TRANSFER_LOCATIONS[0],
            a1: TRANSFER_LOCATIONS[1],
            a2: TRANSFER_LOCATIONS[2],
        },
        "holders": {value: None for value in TRANSFER_OBJECTS},
    }
    turns = [
        {
            "turn_id": "move-cross",
            "mode": "interact",
            "text": f"move {TRANSFER_LEXICON[a0]} to {TRANSFER_LEXICON[l2]}",
        },
        {"turn_id": "where-context", "mode": "interact", "text": "where are they"},
        {
            "turn_id": "take-cross",
            "mode": "interact",
            "text": f"{TRANSFER_LEXICON[a0]} take {TRANSFER_LEXICON[item]}",
        },
        {
            "turn_id": "has-query",
            "mode": "interact",
            "text": f"does {TRANSFER_LEXICON[a0]} hold {TRANSFER_LEXICON[item]}",
        },
        {
            "turn_id": "give-novel",
            "mode": "interact",
            "text": (
                f"{TRANSFER_LEXICON[a0]} give {TRANSFER_LEXICON[item]} "
                f"to {TRANSFER_LEXICON[a1]}"
            ),
        },
        {"turn_id": "who-context", "mode": "interact", "text": "who holds it"},
        {
            "turn_id": "generate-new-lexicon",
            "mode": "generate",
            "meaning": make_frame(
                "assertion", "AT", agent=a2, destination=l0
            ).payload(),
        },
    ]
    expected = [
        {"turn_id": "move-cross", "predicate": "MOVE", "answer": None},
        {
            "turn_id": "where-context",
            "predicate": "WHERE",
            "answer": f"{TRANSFER_LEXICON[a0]} is at {TRANSFER_LEXICON[l2]}",
        },
        {"turn_id": "take-cross", "predicate": "TAKE", "answer": None},
        {"turn_id": "has-query", "predicate": "HAS_QUERY", "answer": "yes"},
        {"turn_id": "give-novel", "predicate": "GIVE", "answer": None},
        {
            "turn_id": "who-context",
            "predicate": "WHO_HAS",
            "answer": f"{TRANSFER_LEXICON[a1]} holds {TRANSFER_LEXICON[item]}",
        },
        {
            "turn_id": "generate-new-lexicon",
            "predicate": "AT",
            "answer": f"{TRANSFER_LEXICON[a2]} is at {TRANSFER_LEXICON[l0]}",
        },
    ]
    return {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "request_id": "atom-language-transfer-workflow-001",
        "stage": model["stage"],
        "world": world,
        "turns": turns,
    }, expected


def score_transfer_workflow(
    response: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_by_id = {row["turn_id"]: row for row in expected}
    rows = []
    for turn in response["turns"]:
        target = expected_by_id[turn["turn_id"]]
        meaning = turn.get("meaning") or {}
        predicate = meaning.get("predicate")
        if turn["mode"] == "generate":
            answer = turn.get("text")
        else:
            generated = turn.get("answer")
            answer = (
                generated.get("text")
                if isinstance(generated, dict)
                and generated.get("status") == "generated"
                else None
            )
        rows.append(
            {
                "turn_id": turn["turn_id"],
                "predicate": predicate,
                "answer": answer,
                "passed": predicate == target["predicate"]
                and answer == target["answer"],
            }
        )
    correct = sum(row["passed"] for row in rows)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "passed": correct == len(rows),
        "turns": rows,
    }


def transfer_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    word = report["stages"]["word"]
    character = report["stages"]["character"]
    gates = {
        "architecture_still_uses_only_universe_core": report["architecture_audit"][
            "passed"
        ],
        "transfer_program_is_12_2_48": report["transfer_program"]["counts"]
        == {"grounding": 12, "transient": 2, "heldout": 48},
        "base_and_transfer_concepts_are_disjoint": report["transfer_program"][
            "disjoint_from_base_concepts"
        ],
        "heldout_actions_are_surface_novel": report["transfer_program"][
            "heldout_action_surface_overlap"
        ]
        == 0,
        "frozen_base_cannot_use_new_lexicon": word["frozen_base_heldout"][
            "grounded_accuracy"
        ]
        <= 0.05,
        "word_transfer_grounding_reaches_threshold": word["heldout"][
            "grounded_accuracy"
        ]
        >= 0.90,
        "character_transfer_grounding_reaches_threshold": character["heldout"][
            "grounded_accuracy"
        ]
        >= 0.90,
        "word_transfer_generation_reaches_threshold": word["heldout"][
            "generation_roundtrip_accuracy"
        ]
        >= 0.90,
        "character_transfer_generation_reaches_threshold": character["heldout"][
            "generation_roundtrip_accuracy"
        ]
        >= 0.90,
        "reference_transfer_reaches_threshold": word["heldout"]["reference_accuracy"]
        >= 0.90
        and character["heldout"]["reference_accuracy"] >= 0.90,
        "all_nine_new_lexemes_are_grounded": word["lexicon"]["accuracy"] == 1.0
        and character["lexicon"]["accuracy"] == 1.0,
        "low_salience_false_binding_is_corrected": word["lexicon"][
            "corrected_lumi_binding"
        ]
        and character["lexicon"]["corrected_lumi_binding"],
        "one_off_noise_does_not_nucleate": word["lexicon"]["noise_surface_absent"]
        and character["lexicon"]["noise_surface_absent"],
        "one_off_noise_trace_is_forgotten": not word["adaptation"][
            "transient_noise_trace_retained"
        ]
        and not character["adaptation"]["transient_noise_trace_retained"],
        "base_language_behavior_is_retained": word["base_retention"][
            "grounded_accuracy"
        ]
        >= 0.98
        and character["base_retention"]["grounded_accuracy"] >= 0.98,
        "base_lexical_and_frame_laws_are_retained": all(
            stage["adaptation"]["retained_base_lexemes"]
            == stage["adaptation"]["base_lexemes"]
            and stage["adaptation"]["retained_base_frames"]
            == stage["adaptation"]["base_frames"]
            for stage in report["stages"].values()
        ),
        "character_span_transfer_reaches_threshold": character["character_spans"]["f1"]
        >= 0.95,
        "abstraction_preserves_transfer": all(
            stage["post_abstraction_heldout"]["grounded_accuracy"] + 0.02
            >= stage["heldout"]["grounded_accuracy"]
            for stage in report["stages"].values()
        ),
        "adapted_models_retain_no_raw_evidence": all(
            stage["adaptation"]["raw_episodes_after_abstraction"] == 0
            and stage["adaptation"]["raw_evidence_after_abstraction"] == 0
            for stage in report["stages"].values()
        ),
        "all_seven_primitives_run_during_adaptation": all(
            set(stage["adaptation"]["operator_counts"]) == set(UNIVERSE_PRIMITIVE_NAMES)
            and all(stage["adaptation"]["operator_counts"].values())
            for stage in report["stages"].values()
        ),
        "phase_mixing_and_annealing_are_active": all(
            stage["adaptation"]["cumulative_phase_energy"] > 0.0
            and stage["adaptation"]["accepted_worse_moves"] > 0
            for stage in report["stages"].values()
        ),
        "serialized_adapted_models_reload": all(
            stage["serialized_model_reloads"] for stage in report["stages"].values()
        ),
        "adaptation_replay_is_deterministic": report["deterministic_replay"],
        "serialized_transfer_workflow_passes": report["serialized_workflow"]["passed"],
        "graph_rag_and_side_view_are_bound": report["knowledge_runtime"]["passed"]
        and report["side_view_contract"]["model_hash"] == report["primary_model_hash"],
    }
    return {
        "gates": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
    }


def run_transfer_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_tests = run_transfer_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Transfer self-tests failed: {self_tests['failed']}")

    base_program = build_grounded_language_program()
    transfer_program = build_language_transfer_program()
    write_jsonl(
        output_dir / "atom_transfer_grounding.jsonl", transfer_program["grounding"]
    )
    write_jsonl(
        output_dir / "atom_transfer_transient.jsonl", transfer_program["transient"]
    )
    write_jsonl(output_dir / "atom_transfer_heldout.jsonl", transfer_program["heldout"])
    write_json(
        output_dir / "atom_transfer_evaluation_truth.json",
        transfer_program["evaluation_truth"],
    )
    write_json(
        output_dir / "atom_transfer_dataset_manifest.json", transfer_program["manifest"]
    )

    config = LanguageConfig()
    word_runtime, word_result, word_model, word_history = _stage_transfer(
        base_program, transfer_program, "word", config
    )
    char_runtime, char_result, char_model, char_history = _stage_transfer(
        base_program, transfer_program, "character", config
    )
    write_json(output_dir / "atom_transfer_word_model.json", word_model)
    write_json(output_dir / "atom_transfer_character_model.json", char_model)
    write_json(output_dir / "atom_transfer_word_history.json", word_history)
    write_json(output_dir / "atom_transfer_character_history.json", char_history)

    replay_base, _, _ = train_language_field(
        base_program["train"], "word", config=config
    )
    replay_base.abstract("word-base-transfer-source-abstract")
    replay_runtime = runtime_from_language_model(
        json.loads(json.dumps(language_model_payload(replay_base)))
    )
    replay_history, _ = adapt_language_field(
        replay_runtime,
        transfer_program["grounding"],
        transient_rows=transfer_program["transient"],
        bootstrap_epochs=2,
        grounded_epochs=3,
        adaptation_id="transfer-word",
    )
    replay_runtime.abstract("word-transfer-final-abstract")
    replay_model = language_model_payload(replay_runtime)
    deterministic_replay = replay_model["model_hash"] == word_model[
        "model_hash"
    ] and stable_hash(replay_history) == stable_hash(word_history)

    request, expected = build_transfer_workflow(char_model)
    request_path = output_dir / "atom_transfer_workflow_request.json"
    response_path = output_dir / "atom_transfer_workflow_response.json"
    write_json(request_path, request)
    response = run_language_workflow(
        output_dir / "atom_transfer_character_model.json",
        request_path,
        response_path,
    )
    workflow_score = score_transfer_workflow(response, expected)

    graph = build_language_graph()
    retrieval = retrieve_atom_context(
        graph,
        "learn revise remember forget ground lexeme role transfer context speak",
        limit=20,
    )
    retrieved_names = {row["name"] for row in retrieval}
    knowledge_runtime = {
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "graph_hash": stable_hash(graph.manifest()),
        "retrieved": retrieval,
        "passed": {
            "learn",
            "revise",
            "remember",
            "forget",
            "ground",
            "lexical_nucleation",
            "role_bind",
            "speak",
        }
        <= retrieved_names,
    }

    example = next(
        row
        for row in transfer_program["heldout"]
        if row.get("answer_text") and not row.get("context_text")
    )
    interaction = interact_text(
        char_runtime,
        str(example["text"]),
        example["before"],
        {},
    )
    answer = None
    if interaction.get("answer"):
        answer = interaction["answer"].get("text")
    meaning = interaction["meaning"]

    report: dict[str, Any] = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "primary_model_hash": char_model["model_hash"],
        "manifest": {
            "standard_neural_network": False,
            "gradient_descent": False,
            "backpropagation": False,
            "pretrained_model": False,
            "trainable_weight_matrix": False,
            "base_model_is_laws_only_before_adaptation": True,
            "word_and_character_runs_are_independent": True,
            "language_runtime": ATOM_LANGUAGE_RUNTIME,
            "universe_primitives": list(UNIVERSE_PRIMITIVE_NAMES),
        },
        "transfer_program": transfer_program["manifest"],
        "self_tests": self_tests,
        "architecture_audit": architecture_audit(),
        "knowledge_runtime": knowledge_runtime,
        "stages": {"word": word_result, "character": char_result},
        "baselines": {
            "exact_demo": exact_demo_baseline(
                transfer_program["grounding"], transfer_program["heldout"]
            ),
            "frozen_base_word": word_result["frozen_base_heldout"],
        },
        "deterministic_replay": deterministic_replay,
        "serialized_workflow": workflow_score,
        "side_view_interaction": {
            "case_id": example["case_id"],
            "utterance": example["text"],
            "meaning": meaning,
            "answer": answer,
            "semantic_mass": (
                "unknown"
                if meaning is None
                else f"{len(meaning['roles'])} roles -> 0 unexpressed"
            ),
            "world_after": interaction["world_after"],
        },
        "controlled_chaos": {
            "initial_temperature": config.initial_temperature,
            "final_temperature": char_runtime.state.temperature,
            "word_transfer_phase_energy": word_result["adaptation"][
                "cumulative_phase_energy"
            ],
            "character_transfer_phase_energy": char_result["adaptation"][
                "cumulative_phase_energy"
            ],
            "word_accepted_worse_moves": word_result["adaptation"][
                "accepted_worse_moves"
            ],
            "character_accepted_worse_moves": char_result["adaptation"][
                "accepted_worse_moves"
            ],
        },
        "side_view_contract": {
            "runtime": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
            "binding": ATOM_LANGUAGE_ARTIFACT_BINDING,
            "model_hash": char_model["model_hash"],
        },
        "models": {"word": word_model, "character": char_model},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["experiment_gates"] = transfer_gates(report)
    report_path = output_dir / "atom_transfer_report.json"
    write_json(report_path, report)
    side_path = render_language_artifact(
        report,
        char_model,
        output_dir / "atom_transfer_side_view.html",
    )
    report["artifact_side_view"] = {
        "path": side_path.name,
        "sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
        "runtime_marker": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
        "binding_marker": ATOM_LANGUAGE_ARTIFACT_BINDING,
        "model_hash_bound": char_model["model_hash"],
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transfer_outputs"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_transfer_self_tests()
    else:
        result = run_transfer_experiment(args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
