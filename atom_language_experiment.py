"""Run the grounded Atom Language Field v1 experiment.

The experiment trains independent word-pulse and character-pulse fields from a
192-episode micro-world, evaluates systematic held-out combinations, runs
primitive ablations and baselines, exports strict serialized models, and binds
the resulting artifact into a user-visible side view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_language_dataset import (
    AGENTS,
    LOCATIONS,
    OBJECTS,
    LANGUAGE_SEED,
    build_grounded_language_program,
)
from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    LANGUAGE_MODEL_SCHEMA,
    LanguageConfig,
    Primitive,
    architecture_audit,
    character_span_f1,
    config_with,
    evaluate_language_rows,
    frame_from_world_delta,
    interact_text,
    language_model_payload,
    make_frame,
    run_language_workflow,
    run_language_field_self_tests,
    runtime_from_language_model,
    stable_hash,
    tokenize_word_pulses,
    train_language_field,
)
from atom_language_side_view import (
    ATOM_LANGUAGE_ARTIFACT_BINDING,
    ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_language_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    build_language_graph,
    retrieve_atom_context,
)


EXPERIMENT_NAME = "atom_grounded_language_field_v1"


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


def run_self_tests() -> dict[str, Any]:
    field = run_language_field_self_tests()
    program = build_grounded_language_program()
    train_ids = {row["case_id"] for row in program["train"]}
    validation_ids = {row["case_id"] for row in program["validation"]}
    heldout_ids = {row["case_id"] for row in program["heldout"]}
    graph = build_language_graph()
    retrieval = retrieve_atom_context(
        graph,
        "ground character lexeme grammar role understand context pronoun speak",
        limit=16,
    )
    retrieved = {row["name"] for row in retrieval}
    checks = {
        "field_self_tests": field["passed"],
        "dataset_has_exact_192_rows": sum(
            len(program[name]) for name in ("train", "validation", "heldout")
        )
        == 192,
        "dataset_split_is_120_24_48": (
            len(program["train"]),
            len(program["validation"]),
            len(program["heldout"]),
        )
        == (120, 24, 48),
        "dataset_splits_do_not_overlap": not (
            train_ids & validation_ids
            or train_ids & heldout_ids
            or validation_ids & heldout_ids
        ),
        "training_rows_have_no_gold_frames": all(
            "frame" not in row and "family" not in row for row in program["train"]
        ),
        "context_questions_have_grounded_paraphrases": all(
            row.get("paraphrase_text")
            for row in program["train"]
            if str(row["text"]).startswith(("who holds", "are they"))
        ),
        "evaluator_truth_is_separate": set(program["evaluation_truth"])
        == validation_ids
        | heldout_ids
        | {
            row["case_id"]
            for row in program["train"]
            if row["case_id"] in program["evaluation_truth"]
        },
        "language_graph_retrieval_is_live": {
            "ground",
            "lexical_nucleation",
            "role_bind",
            "understand",
            "resolve_reference",
            "speak",
        }
        <= retrieved,
        "every_action_delta_is_groundable": all(
            frame_from_world_delta(row) is not None
            for row in program["train"]
            if row["before"] != row["after"]
        ),
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
        "field": field,
    }


def exact_table_baseline(
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    table = {
        str(row["text"]): truth[row["case_id"]]["frame"]
        for row in train_rows
        if row["case_id"] in truth
    }
    correct = 0
    covered = 0
    for row in evaluation_rows:
        prediction = table.get(str(row["text"]))
        covered += int(prediction is not None)
        correct += int(prediction == truth[row["case_id"]]["frame"])
    total = len(evaluation_rows)
    return {
        "cases": total,
        "correct": correct,
        "covered": covered,
        "grounded_accuracy": correct / total if total else 0.0,
        "coverage": covered / total if total else 0.0,
    }


def trigram_surface_baseline(
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trigrams: Counter[tuple[str, str, str]] = Counter()
    for row in train_rows:
        tokens = ("<s>", *tokenize_word_pulses(str(row["text"])), "</s>")
        trigrams.update(zip(tokens, tokens[1:], tokens[2:]))
    covered = 0
    for row in evaluation_rows:
        tokens = ("<s>", *tokenize_word_pulses(str(row["text"])), "</s>")
        row_trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        covered += int(
            bool(row_trigrams) and all(value in trigrams for value in row_trigrams)
        )
    total = len(evaluation_rows)
    return {
        "cases": total,
        "surface_trigram_coverage": covered / total if total else 0.0,
        "grounded_accuracy": 0.0,
        "reason": "surface statistics carry no world-atom binding",
    }


def _stage_run(
    program: Mapping[str, Any],
    stage: str,
    config: LanguageConfig,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    runtime, history, diagnostics = train_language_field(
        program["train"], stage, config=config
    )
    raw_before = len(runtime.state.traces)
    before = {
        "train": evaluate_language_rows(
            runtime,
            [
                row
                for row in program["train"]
                if row["case_id"] in program["evaluation_truth"]
            ],
            program["evaluation_truth"],
        ),
        "validation": evaluate_language_rows(
            runtime, program["validation"], program["evaluation_truth"]
        ),
        "heldout": evaluate_language_rows(
            runtime, program["heldout"], program["evaluation_truth"]
        ),
    }
    spans = character_span_f1(runtime, program["heldout"])
    phase_before = runtime.state.cumulative_phase_energy
    maximum_phase = runtime.state.maximum_phase_energy
    improving = runtime.state.accepted_improving_moves
    worse = runtime.state.accepted_worse_moves
    observations = runtime.state.observations
    operator_counts = dict(runtime.state.operator_counts)
    temperature = runtime.state.temperature
    conservation_applications = runtime.state.conservation_applications
    runtime.abstract(f"{stage}-final-abstraction")
    model = language_model_payload(runtime)
    restored = runtime_from_language_model(json.loads(json.dumps(model)))
    after = evaluate_language_rows(
        restored, program["heldout"], program["evaluation_truth"]
    )
    result = {
        "stage": stage,
        "diagnostics": diagnostics,
        "train": before["train"],
        "validation": before["validation"],
        "heldout": before["heldout"],
        "post_abstraction_heldout": after,
        "character_spans": spans,
        "training": {
            "observations": observations,
            "raw_episodes_before_abstraction": raw_before,
            "raw_episodes_after_abstraction": len(runtime.state.traces),
            "lexeme_laws": len(runtime.state.lexeme_laws),
            "frame_laws": len(runtime.state.frame_laws),
            "reference_laws": len(runtime.state.reference_laws),
            "character_span_laws": len(runtime.state.character_span_laws),
            "compression_ratio": observations
            / max(
                1,
                len(runtime.state.lexeme_laws)
                + len(runtime.state.frame_laws)
                + len(runtime.state.reference_laws),
            ),
            "temperature": temperature,
            "accepted_improving_moves": improving,
            "accepted_worse_moves": worse,
            "cumulative_phase_energy": phase_before,
            "maximum_phase_energy": maximum_phase,
            "operator_counts": operator_counts,
            "conservation_applications": conservation_applications,
            "conservation_excess": runtime.state.conservation_excess,
            "history_hash": stable_hash(history),
        },
        "model_hash": model["model_hash"],
        "serialized_model_reloads": language_model_payload(restored)["model_hash"]
        == model["model_hash"],
    }
    return runtime, result, history, model


def run_primitive_ablation(
    program: Mapping[str, Any],
    primitive: Primitive,
    config: LanguageConfig,
) -> dict[str, Any]:
    runtime, _, diagnostics = train_language_field(
        program["train"], "word", config=config, disabled=(primitive,)
    )
    evaluation = evaluate_language_rows(
        runtime, program["heldout"], program["evaluation_truth"]
    )
    raw_before = len(runtime.state.traces)
    runtime.abstract(f"ablation-{primitive.value}-abstract")
    signals = {
        Primitive.RADIATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.GRAVITATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.ATTRACTION_REPULSION: len(runtime.state.frame_laws) == 0
        and evaluation["grounded_accuracy"] == 0.0,
        Primitive.NUCLEATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.CONSERVATION: runtime.state.conservation_applications == 0,
        Primitive.DISSIPATION: math.isclose(
            runtime.state.temperature, config.initial_temperature, abs_tol=1e-12
        ),
        Primitive.DECAY: len(runtime.state.traces) == raw_before and raw_before > 0,
    }
    return {
        "primitive": primitive.value,
        "grounded_accuracy": evaluation["grounded_accuracy"],
        "generation_roundtrip_accuracy": evaluation["generation_roundtrip_accuracy"],
        "lexeme_laws": len(runtime.state.lexeme_laws),
        "frame_laws": len(runtime.state.frame_laws),
        "raw_before_abstraction": raw_before,
        "raw_after_abstraction": len(runtime.state.traces),
        "temperature": runtime.state.temperature,
        "conservation_applications": runtime.state.conservation_applications,
        "unresolved_cases": len(diagnostics["unresolved_case_ids"]),
        "causal_signal": {
            "lexeme_laws_missing": len(runtime.state.lexeme_laws) == 0,
            "frame_laws_missing": len(runtime.state.frame_laws) == 0,
            "conservation_bypassed": runtime.state.conservation_applications == 0,
            "cooling_bypassed": math.isclose(
                runtime.state.temperature,
                config.initial_temperature,
                abs_tol=1e-12,
            ),
            "raw_episodes_retained": len(runtime.state.traces) == raw_before
            and raw_before > 0,
        },
        "causal_effect_observed": bool(signals[primitive]),
    }


def build_workflow(
    program: Mapping[str, Any], model: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lexicon = program["evaluator_oracle"]["concept_to_surface"]
    a0, a1 = AGENTS[:2]
    item = OBJECTS[0]
    l0, l1 = LOCATIONS[:2]
    world = {
        "locations": {
            agent: LOCATIONS[index % len(LOCATIONS)]
            for index, agent in enumerate(AGENTS)
        },
        "holders": {value: None for value in OBJECTS},
    }
    world["locations"][a0] = l0
    turns = [
        {
            "turn_id": "move-novel",
            "mode": "interact",
            "text": f"move {lexicon[a0]} to {lexicon[l1]}",
        },
        {
            "turn_id": "where-context",
            "mode": "interact",
            "text": "where are they",
        },
        {
            "turn_id": "take-novel",
            "mode": "interact",
            "text": f"{lexicon[a0]} take {lexicon[item]}",
        },
        {
            "turn_id": "has-query",
            "mode": "interact",
            "text": f"does {lexicon[a0]} hold {lexicon[item]}",
        },
        {
            "turn_id": "give-novel",
            "mode": "interact",
            "text": f"{lexicon[a0]} give {lexicon[item]} to {lexicon[a1]}",
        },
        {
            "turn_id": "who-context",
            "mode": "interact",
            "text": "who holds it",
        },
        {
            "turn_id": "generate-state",
            "mode": "generate",
            "meaning": make_frame(
                "assertion", "AT", agent=a1, destination=l0
            ).payload(),
        },
    ]
    expected = [
        {"turn_id": "move-novel", "predicate": "MOVE", "answer": None},
        {
            "turn_id": "where-context",
            "predicate": "WHERE",
            "answer": f"{lexicon[a0]} is at {lexicon[l1]}",
        },
        {"turn_id": "take-novel", "predicate": "TAKE", "answer": None},
        {"turn_id": "has-query", "predicate": "HAS_QUERY", "answer": "yes"},
        {"turn_id": "give-novel", "predicate": "GIVE", "answer": None},
        {
            "turn_id": "who-context",
            "predicate": "WHO_HAS",
            "answer": f"{lexicon[a1]} holds {lexicon[item]}",
        },
        {
            "turn_id": "generate-state",
            "predicate": "AT",
            "answer": f"{lexicon[a1]} is at {lexicon[l0]}",
        },
    ]
    request = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "request_id": "atom-language-real-workflow-001",
        "stage": model["stage"],
        "world": world,
        "turns": turns,
    }
    return request, expected


def score_workflow(
    response: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_map = {row["turn_id"]: row for row in expected}
    correct = 0
    rows = []
    for turn in response["turns"]:
        target = expected_map[turn["turn_id"]]
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
        passed = predicate == target["predicate"] and answer == target["answer"]
        correct += int(passed)
        rows.append(
            {
                "turn_id": turn["turn_id"],
                "predicate": predicate,
                "answer": answer,
                "passed": passed,
            }
        )
    return {
        "cases": len(expected),
        "correct": correct,
        "accuracy": correct / len(expected),
        "passed": correct == len(expected),
        "turns": rows,
    }


def experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    word = report["stages"]["word"]
    character = report["stages"]["character"]
    word_heldout = word["heldout"]
    char_heldout = character["heldout"]
    baselines = report["baselines"]
    chaos = report["controlled_chaos"]
    gates = {
        "architecture_is_universe_core_only": report["architecture_audit"]["passed"],
        "wiki_graph_and_graph_rag_are_runtime_wired": report["knowledge_runtime"][
            "passed"
        ],
        "dataset_is_exactly_120_24_48": report["dataset"]["counts"]
        == {"train": 120, "validation": 24, "heldout": 48},
        "training_has_no_gold_frames": not report["dataset"][
            "training_has_gold_frames"
        ],
        "word_grounded_language_reaches_threshold": word_heldout["grounded_accuracy"]
        >= 0.90,
        "word_generation_roundtrip_reaches_threshold": word_heldout[
            "generation_roundtrip_accuracy"
        ]
        >= 0.90,
        "word_grammar_is_valid": word_heldout["grammar_validity"] == 1.0,
        "word_reference_resolution_reaches_threshold": word_heldout[
            "reference_accuracy"
        ]
        >= 0.85,
        "character_grounded_language_reaches_threshold": char_heldout[
            "grounded_accuracy"
        ]
        >= 0.90,
        "character_generation_roundtrip_reaches_threshold": char_heldout[
            "generation_roundtrip_accuracy"
        ]
        >= 0.90,
        "character_span_f1_reaches_threshold": character["character_spans"]["f1"]
        >= 0.90,
        "character_stage_stays_within_five_points": char_heldout["grounded_accuracy"]
        + 0.05
        >= word_heldout["grounded_accuracy"],
        "abstraction_removes_raw_episodes": all(
            stage["training"]["raw_episodes_after_abstraction"] == 0
            for stage in report["stages"].values()
        ),
        "abstraction_preserves_language_behavior": all(
            stage["post_abstraction_heldout"]["grounded_accuracy"] + 0.02
            >= stage["heldout"]["grounded_accuracy"]
            for stage in report["stages"].values()
        ),
        "language_laws_compress_observations": all(
            stage["training"]["compression_ratio"] >= 5.0
            for stage in report["stages"].values()
        ),
        "beats_exact_table_by_twenty_points": word_heldout["grounded_accuracy"]
        >= baselines["exact_table"]["grounded_accuracy"] + 0.20,
        "beats_trigram_grounding_by_twenty_points": word_heldout["grounded_accuracy"]
        >= baselines["trigram"]["grounded_accuracy"] + 0.20,
        "every_universe_primitive_has_causal_effect": all(
            row["causal_effect_observed"]
            for row in report["primitive_ablations"].values()
        ),
        "all_seven_primitives_are_exercised": set(word["training"]["operator_counts"])
        == set(UNIVERSE_PRIMITIVE_NAMES)
        and all(word["training"]["operator_counts"].values()),
        "semantic_mass_conservation_holds": all(
            stage["training"]["conservation_excess"] <= 1e-9
            and stage["training"]["conservation_applications"] > 0
            for stage in report["stages"].values()
        ),
        "phase_mixing_is_active_and_causal": chaos["phase_active"]
        and chaos["phase_changes_trajectory"],
        "thermal_annealing_cools_and_is_causal": chaos["temperature_monotonic"]
        and chaos["temperature_drop"] > 0.0
        and chaos["thermal_changes_trajectory"]
        and chaos["accepted_worse_moves"] > 0,
        "training_replay_is_deterministic": chaos["deterministic_replay"],
        "serialized_models_reload": all(
            stage["serialized_model_reloads"] for stage in report["stages"].values()
        ),
        "serialized_real_workflow_passes": report["serialized_workflow"]["passed"],
        "artifact_side_view_binds_real_model": report["side_view_contract"][
            "model_hash"
        ]
        == report["primary_model_hash"]
        and report["side_view_contract"]["runtime"] == ATOM_LANGUAGE_SIDE_VIEW_RUNTIME
        and report["side_view_contract"]["binding"] == ATOM_LANGUAGE_ARTIFACT_BINDING,
    }
    return {
        "gates": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
    }


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_tests = run_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Language self-tests failed: {self_tests['failed']}")
    program = build_grounded_language_program()
    write_jsonl(output_dir / "atom_language_train.jsonl", program["train"])
    write_jsonl(output_dir / "atom_language_validation.jsonl", program["validation"])
    write_jsonl(output_dir / "atom_language_heldout.jsonl", program["heldout"])
    write_json(
        output_dir / "atom_language_evaluation_truth.json",
        program["evaluation_truth"],
    )
    write_json(output_dir / "atom_language_dataset_manifest.json", program["manifest"])

    config = LanguageConfig()
    word_runtime, word_result, word_history, word_model = _stage_run(
        program, "word", config
    )
    char_runtime, char_result, char_history, char_model = _stage_run(
        program, "character", config
    )
    write_json(output_dir / "atom_language_word_model.json", word_model)
    write_json(output_dir / "atom_language_character_model.json", char_model)
    write_json(output_dir / "atom_language_word_history.json", word_history)
    write_json(output_dir / "atom_language_character_history.json", char_history)

    exact = exact_table_baseline(
        program["train"], program["heldout"], program["evaluation_truth"]
    )
    trigram = trigram_surface_baseline(program["train"], program["heldout"])
    ablations = {
        primitive.value: run_primitive_ablation(program, primitive, config)
        for primitive in Primitive
    }

    zero_phase_runtime, zero_phase_history, _ = train_language_field(
        program["train"],
        "word",
        config=config_with(config, phase_mix_strength=0.0),
    )
    zero_phase_runtime.abstract("zero-phase-abstract")
    zero_phase_model = language_model_payload(zero_phase_runtime)
    no_cooling_runtime, no_cooling_history, _ = train_language_field(
        program["train"],
        "word",
        config=config_with(
            config,
            cooling_rate=1.0,
            temperature_floor=config.initial_temperature,
        ),
    )
    no_cooling_runtime.abstract("no-cooling-abstract")
    replay_runtime, replay_history, _ = train_language_field(
        program["train"], "word", config=config
    )
    replay_runtime.abstract("word-final-abstraction")
    replay_model = language_model_payload(replay_runtime)

    request, expected_workflow = build_workflow(program, char_model)
    request_path = output_dir / "atom_language_workflow_request.json"
    response_path = output_dir / "atom_language_workflow_response.json"
    write_json(request_path, request)
    workflow_response = run_language_workflow(
        output_dir / "atom_language_character_model.json",
        request_path,
        response_path,
    )
    workflow_score = score_workflow(workflow_response, expected_workflow)

    graph = build_language_graph()
    retrieval_query = (
        "ground character lexeme grammar role understand context pronoun speak"
    )
    retrieval = retrieve_atom_context(graph, retrieval_query, limit=16)
    retrieval_names = {row["name"] for row in retrieval}
    knowledge_runtime = {
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "graph_hash": stable_hash(graph.manifest()),
        "query": retrieval_query,
        "retrieved": retrieval,
        "passed": {
            "ground",
            "lexical_nucleation",
            "role_bind",
            "understand",
            "resolve_reference",
            "speak",
        }
        <= retrieval_names,
    }

    example_row = next(
        row
        for row in program["heldout"]
        if row.get("answer_text") and not row.get("context_text")
    )
    interaction = interact_text(
        char_runtime,
        str(example_row["text"]),
        example_row["before"],
        {},
    )
    interaction_answer = None
    if interaction.get("answer"):
        interaction_answer = interaction["answer"].get("text")
    meaning = interaction["meaning"]
    semantic_mass = (
        "unknown"
        if meaning is None
        else f"{len(meaning['roles'])} roles -> 0 unexpressed"
    )

    word_temperatures = [row["temperature"] for row in word_history]
    report: dict[str, Any] = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "primary_model_hash": char_model["model_hash"],
        "manifest": {
            "seed": LANGUAGE_SEED,
            "standard_neural_network": False,
            "gradient_descent": False,
            "backpropagation": False,
            "pretrained_model": False,
            "trainable_weight_matrix": False,
            "state_transition_authority": "UniverseLanguageKernel only",
            "universe_primitives": list(UNIVERSE_PRIMITIVE_NAMES),
            "runtime": ATOM_LANGUAGE_RUNTIME,
            "word_and_character_runs_are_independent": True,
            "training_receives_gold_frames": False,
        },
        "dataset": program["manifest"],
        "self_tests": self_tests,
        "architecture_audit": architecture_audit(),
        "knowledge_runtime": knowledge_runtime,
        "stages": {"word": word_result, "character": char_result},
        "baselines": {"exact_table": exact, "trigram": trigram},
        "primitive_ablations": ablations,
        "controlled_chaos": {
            "initial_temperature": config.initial_temperature,
            "final_temperature": word_result["training"]["temperature"],
            "temperature_drop": config.initial_temperature
            - word_result["training"]["temperature"],
            "temperature_monotonic": all(
                later <= earlier + 1e-12
                for earlier, later in zip(word_temperatures, word_temperatures[1:])
            ),
            "phase_active": word_result["training"]["cumulative_phase_energy"] > 0.0,
            "phase_changes_trajectory": stable_hash(word_history)
            != stable_hash(zero_phase_history)
            and word_model["model_hash"] != zero_phase_model["model_hash"],
            "thermal_changes_trajectory": stable_hash(word_history)
            != stable_hash(no_cooling_history),
            "accepted_improving_moves": word_result["training"][
                "accepted_improving_moves"
            ],
            "accepted_worse_moves": word_result["training"]["accepted_worse_moves"],
            "cumulative_phase_energy": word_result["training"][
                "cumulative_phase_energy"
            ],
            "maximum_phase_energy": word_result["training"]["maximum_phase_energy"],
            "deterministic_replay": word_model["model_hash"]
            == replay_model["model_hash"]
            and stable_hash(word_history) == stable_hash(replay_history),
        },
        "serialized_workflow": workflow_score,
        "side_view_interaction": {
            "case_id": example_row["case_id"],
            "utterance": example_row["text"],
            "meaning": meaning,
            "answer": interaction_answer,
            "semantic_mass": semantic_mass,
            "world_after": interaction["world_after"],
        },
        "side_view_contract": {
            "runtime": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
            "binding": ATOM_LANGUAGE_ARTIFACT_BINDING,
            "model_hash": char_model["model_hash"],
        },
        "models": {"word": word_model, "character": char_model},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["experiment_gates"] = experiment_gates(report)
    report_path = output_dir / "atom_language_report.json"
    write_json(report_path, report)
    side_view_path = render_language_artifact(
        report, char_model, output_dir / "atom_language_side_view.html"
    )
    report["artifact_side_view"] = {
        "path": side_view_path.name,
        "sha256": hashlib.sha256(side_view_path.read_bytes()).hexdigest(),
        "runtime_marker": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
        "binding_marker": ATOM_LANGUAGE_ARTIFACT_BINDING,
        "model_hash_bound": char_model["model_hash"],
    }
    write_json(report_path, report)
    write_json(output_dir / "atom_language_manifest.json", report["manifest"])
    write_json(output_dir / "atom_language_knowledge_graph.json", graph.manifest())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("language_outputs"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow_mode = any((args.model, args.request, args.response))
    if args.self_test:
        result = run_self_tests()
    elif workflow_mode:
        if not all((args.model, args.request, args.response)):
            raise ValueError("--model, --request, and --response are required together")
        result = run_language_workflow(
            args.model.resolve(), args.request.resolve(), args.response.resolve()
        )
    else:
        result = run_experiment(args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
