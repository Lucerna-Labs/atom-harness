"""Run opaque compositional grammar induction on the universe-core field."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_language_dataset import build_grounded_language_program
from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    LANGUAGE_MODEL_SCHEMA,
    UNIVERSE_PRIMITIVE_NAMES,
    LanguageConfig,
    Primitive,
    architecture_audit,
    character_span_f1,
    evaluate_language_rows,
    generate_text,
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
from atom_language_opaque_dataset import (
    OPAQUE_AGENTS,
    OPAQUE_LEXICON,
    OPAQUE_LOCATIONS,
    OPAQUE_OBJECTS,
    OPAQUE_PATTERNS,
    build_opaque_language_program,
    render_opaque_frame,
)
from atom_language_side_view import (
    ATOM_LANGUAGE_ARTIFACT_BINDING,
    ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_language_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_language_graph,
    retrieve_atom_context,
)


OPAQUE_EXPERIMENT_NAME = "atom_opaque_compositional_grammar_v1"


def opaque_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def opaque_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def opaque_surface_tokens(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    tokens: set[str] = set()
    for row in rows:
        for field in (
            "text",
            "answer_text",
            "context_text",
            "paraphrase_text",
        ):
            value = row.get(field)
            if isinstance(value, str):
                tokens.update(value.split())
    return tokens


def run_opaque_self_tests() -> dict[str, Any]:
    field = run_language_field_self_tests()
    opaque = build_opaque_language_program()
    base = build_grounded_language_program()
    opaque_rows = opaque["train"] + opaque["validation"] + opaque["heldout"]
    base_rows = base["train"] + base["validation"] + base["heldout"]
    opaque_tokens = opaque_surface_tokens(opaque_rows)
    base_tokens = opaque_surface_tokens(base_rows)
    opaque_concepts = set(opaque["evaluator_oracle"]["concept_to_surface"])
    base_concepts = set(base["evaluator_oracle"]["concept_to_surface"])
    truth_ids = set(opaque["evaluation_truth"])
    heldout_ids = {str(row["case_id"]) for row in opaque["heldout"]}
    literal_tokens = {
        piece
        for pattern in OPAQUE_PATTERNS.values()
        for piece in pattern
        if not piece.startswith("{")
    }
    checks = {
        "field_self_tests": field["passed"],
        "program_is_48_20_52": opaque["manifest"]["counts"]
        == {"train": 48, "validation": 20, "heldout": 52},
        "twelve_new_concepts": len(opaque_concepts) == 12,
        "twelve_opaque_grammar_patterns": len(OPAQUE_PATTERNS) == 12,
        "surface_vocabulary_is_disjoint_from_base": not opaque_tokens & base_tokens,
        "concept_space_is_disjoint_from_base": not opaque_concepts & base_concepts,
        "opaque_answers_are_not_english": {"aya", "nox"} <= literal_tokens
        and not {"yes", "no"} & opaque_tokens,
        "observations_contain_no_evaluator_meanings": all(
            "frame" not in row and "family" not in row for row in opaque_rows
        ),
        "heldout_truth_is_evaluator_only": truth_ids == heldout_ids,
        "heldout_action_surfaces_are_novel": opaque["manifest"][
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


def opaque_exact_surface_baseline(
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known = {str(row["text"]) for row in train}
    repeated = [row for row in heldout if str(row["text"]) in known]
    repeated_actions = [row for row in repeated if row["before"] != row["after"]]
    return {
        "cases": len(heldout),
        "surface_covered": len(repeated),
        "surface_coverage": len(repeated) / len(heldout),
        "action_surface_covered": len(repeated_actions),
        "action_surface_coverage": len(repeated_actions)
        / max(1, sum(row["before"] != row["after"] for row in heldout)),
    }


def opaque_lexicon_score(runtime: Any) -> dict[str, Any]:
    expected = {surface: concept for concept, surface in OPAQUE_LEXICON.items()}
    surface_to_concept, _ = lexeme_maps(runtime.state.lexeme_laws)
    rows = [
        {
            "surface": surface,
            "expected": concept,
            "predicted": surface_to_concept.get(surface),
            "correct": surface_to_concept.get(surface) == concept,
        }
        for surface, concept in sorted(expected.items())
    ]
    correct = sum(row["correct"] for row in rows)
    extra = sorted(set(surface_to_concept) - set(expected))
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "rows": rows,
        "unexpected_lexeme_surfaces": extra,
    }


def opaque_grammar_score(runtime: Any) -> dict[str, Any]:
    active = {
        (law.direction, law.frame_key, law.pattern)
        for law in runtime.state.frame_laws
        if law.active
    }
    rows: list[dict[str, Any]] = []
    for frame_key, pattern in sorted(OPAQUE_PATTERNS.items()):
        for direction in ("parse", "speak"):
            present = (direction, frame_key, tuple(pattern)) in active
            rows.append(
                {
                    "direction": direction,
                    "frame_key": frame_key,
                    "pattern": list(pattern),
                    "present": present,
                }
            )
    present = sum(row["present"] for row in rows)
    return {
        "cases": len(rows),
        "present": present,
        "coverage": present / len(rows),
        "rows": rows,
    }


def opaque_generated_vocabulary_score(
    evaluation: Mapping[str, Any],
    allowed_tokens: set[str],
) -> dict[str, Any]:
    generated = [
        str(row["generated"])
        for row in evaluation["predictions"]
        if isinstance(row.get("generated"), str)
    ]
    foreign = sorted(
        {
            token
            for text in generated
            for token in text.split()
            if token not in allowed_tokens
        }
    )
    return {
        "generated_cases": len(generated),
        "foreign_tokens": foreign,
        "opaque_only": not foreign and len(generated) == evaluation["cases"],
    }


def run_opaque_stage(
    program: Mapping[str, Any],
    stage: str,
    config: LanguageConfig,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    runtime, history, diagnostics = train_language_field(
        program["train"],
        stage,
        config=config,
    )
    validation = evaluate_language_rows(
        runtime,
        program["validation"],
        program["validation_truth"],
    )
    heldout = evaluate_language_rows(
        runtime,
        program["heldout"],
        program["evaluation_truth"],
    )
    spans = character_span_f1(runtime, program["heldout"])
    lexicon = opaque_lexicon_score(runtime)
    grammar = opaque_grammar_score(runtime)
    allowed_tokens = set(program["manifest"]["surface_vocabulary"])
    generation_vocabulary = opaque_generated_vocabulary_score(
        heldout,
        allowed_tokens,
    )
    raw_before = len(runtime.state.traces)
    observations = runtime.state.observations
    phase_energy = runtime.state.cumulative_phase_energy
    maximum_phase_energy = runtime.state.maximum_phase_energy
    accepted_improving = runtime.state.accepted_improving_moves
    accepted_worse = runtime.state.accepted_worse_moves
    temperature = runtime.state.temperature
    operator_counts = dict(runtime.state.operator_counts)
    runtime.abstract(f"opaque-{stage}-final-abstraction")
    model = language_model_payload(runtime)
    restored = runtime_from_language_model(json.loads(json.dumps(model)))
    post_heldout = evaluate_language_rows(
        restored,
        program["heldout"],
        program["evaluation_truth"],
    )
    answer_surfaces = {
        "YES": generate_text(restored, make_frame("answer", "YES")),
        "NO": generate_text(restored, make_frame("answer", "NO")),
    }
    result = {
        "stage": stage,
        "diagnostics": diagnostics,
        "validation": validation,
        "heldout": heldout,
        "post_abstraction_heldout": post_heldout,
        "character_spans": spans,
        "lexicon": lexicon,
        "grammar": grammar,
        "generation_vocabulary": generation_vocabulary,
        "answer_surfaces": answer_surfaces,
        "training": {
            "observations": observations,
            "raw_episodes_before_abstraction": raw_before,
            "raw_episodes_after_abstraction": len(runtime.state.traces),
            "raw_evidence_after_abstraction": (
                len(runtime.state.association_evidence)
                + len(runtime.state.template_evidence)
                + len(runtime.state.reference_evidence)
            ),
            "lexeme_laws": len(runtime.state.lexeme_laws),
            "frame_laws": len(runtime.state.frame_laws),
            "reference_laws": len(runtime.state.reference_laws),
            "temperature": temperature,
            "accepted_improving_moves": accepted_improving,
            "accepted_worse_moves": accepted_worse,
            "cumulative_phase_energy": phase_energy,
            "maximum_phase_energy": maximum_phase_energy,
            "operator_counts": operator_counts,
            "history_hash": stable_hash(history),
        },
        "model_hash": model["model_hash"],
        "serialized_model_reloads": language_model_payload(restored)["model_hash"]
        == model["model_hash"],
    }
    return restored, result, history, model


def run_opaque_ablation(
    program: Mapping[str, Any],
    primitive: Primitive,
    config: LanguageConfig,
) -> dict[str, Any]:
    runtime, _, diagnostics = train_language_field(
        program["train"],
        "word",
        config=config,
        disabled=(primitive,),
    )
    evaluation = evaluate_language_rows(
        runtime,
        program["heldout"],
        program["evaluation_truth"],
    )
    raw_before = len(runtime.state.traces)
    runtime.abstract(f"opaque-ablation-{primitive.value}")
    signals = {
        Primitive.RADIATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.GRAVITATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.ATTRACTION_REPULSION: len(runtime.state.frame_laws) == 0
        and evaluation["grounded_accuracy"] == 0.0,
        Primitive.NUCLEATION: len(runtime.state.lexeme_laws) == 0,
        Primitive.CONSERVATION: runtime.state.conservation_applications == 0,
        Primitive.DISSIPATION: math.isclose(
            runtime.state.temperature,
            config.initial_temperature,
            abs_tol=1e-12,
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
        "causal_effect_observed": bool(signals[primitive]),
    }


def build_opaque_workflow(
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a0, _, a2, a3 = OPAQUE_AGENTS
    item = OPAQUE_OBJECTS[3]
    destination = OPAQUE_LOCATIONS[3]
    false_destination = OPAQUE_LOCATIONS[2]
    generated_destination = OPAQUE_LOCATIONS[1]
    world = {
        "locations": {
            agent: OPAQUE_LOCATIONS[index] for index, agent in enumerate(OPAQUE_AGENTS)
        },
        "holders": {value: None for value in OPAQUE_OBJECTS},
    }
    move = make_frame("command", "MOVE", agent=a0, destination=destination)
    where = make_frame("question", "WHERE", agent=a0)
    take = make_frame("command", "TAKE", agent=a0, patient=item)
    has_query = make_frame("question", "HAS_QUERY", agent=a0, patient=item)
    give = make_frame("command", "GIVE", agent=a0, patient=item, recipient=a2)
    who = make_frame("question", "WHO_HAS", patient=item)
    at_query = make_frame(
        "question", "AT_QUERY", agent=a0, destination=false_destination
    )
    generated = make_frame(
        "assertion", "AT", agent=a3, destination=generated_destination
    )
    turns = [
        {
            "turn_id": "move-opaque",
            "mode": "interact",
            "text": render_opaque_frame(move.payload()),
        },
        {
            "turn_id": "where-opaque",
            "mode": "interact",
            "text": render_opaque_frame(where.payload()),
        },
        {
            "turn_id": "take-opaque",
            "mode": "interact",
            "text": render_opaque_frame(take.payload()),
        },
        {
            "turn_id": "has-opaque",
            "mode": "interact",
            "text": render_opaque_frame(has_query.payload()),
        },
        {
            "turn_id": "give-opaque",
            "mode": "interact",
            "text": render_opaque_frame(give.payload()),
        },
        {
            "turn_id": "who-opaque",
            "mode": "interact",
            "text": render_opaque_frame(who.payload()),
        },
        {
            "turn_id": "at-false-opaque",
            "mode": "interact",
            "text": render_opaque_frame(at_query.payload()),
        },
        {
            "turn_id": "generate-opaque",
            "mode": "generate",
            "meaning": generated.payload(),
        },
    ]
    expected = [
        {"turn_id": "move-opaque", "predicate": "MOVE", "text": None},
        {
            "turn_id": "where-opaque",
            "predicate": "WHERE",
            "text": render_opaque_frame(
                make_frame(
                    "assertion", "AT", agent=a0, destination=destination
                ).payload()
            ),
        },
        {"turn_id": "take-opaque", "predicate": "TAKE", "text": None},
        {"turn_id": "has-opaque", "predicate": "HAS_QUERY", "text": "aya"},
        {"turn_id": "give-opaque", "predicate": "GIVE", "text": None},
        {
            "turn_id": "who-opaque",
            "predicate": "WHO_HAS",
            "text": render_opaque_frame(
                make_frame("assertion", "HAS", agent=a2, patient=item).payload()
            ),
        },
        {
            "turn_id": "at-false-opaque",
            "predicate": "AT_QUERY",
            "text": "nox",
        },
        {
            "turn_id": "generate-opaque",
            "predicate": "AT",
            "text": render_opaque_frame(generated.payload()),
        },
    ]
    return {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "request_id": "atom-opaque-language-workflow-001",
        "stage": model["stage"],
        "world": world,
        "turns": turns,
    }, expected


def score_opaque_workflow(
    response: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_by_id = {str(row["turn_id"]): row for row in expected}
    rows: list[dict[str, Any]] = []
    for turn in response["turns"]:
        target = expected_by_id[str(turn["turn_id"])]
        meaning = turn.get("meaning") or {}
        predicate = meaning.get("predicate")
        if turn["mode"] == "generate":
            surface = turn.get("text")
        else:
            answer = turn.get("answer")
            surface = (
                answer.get("text")
                if isinstance(answer, Mapping) and answer.get("status") == "generated"
                else None
            )
        rows.append(
            {
                "turn_id": turn["turn_id"],
                "predicate": predicate,
                "surface": surface,
                "passed": predicate == target["predicate"]
                and surface == target["text"],
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


def opaque_experiment_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    word = report["stages"]["word"]
    character = report["stages"]["character"]
    gates = {
        "architecture_uses_only_universe_core": report["architecture_audit"]["passed"],
        "program_is_48_20_52": report["opaque_program"]["counts"]
        == {"train": 48, "validation": 20, "heldout": 52},
        "surface_vocabulary_is_disjoint_from_base": report["self_tests"]["checks"][
            "surface_vocabulary_is_disjoint_from_base"
        ],
        "heldout_action_surfaces_are_novel": report["baselines"]["exact_surface"][
            "action_surface_covered"
        ]
        == 0,
        "frozen_english_field_cannot_parse_opaque_language": report["baselines"][
            "frozen_english_word"
        ]["grounded_accuracy"]
        == 0.0,
        "word_validation_reaches_threshold": word["validation"]["grounded_accuracy"]
        >= 0.95,
        "character_validation_reaches_threshold": character["validation"][
            "grounded_accuracy"
        ]
        >= 0.95,
        "word_heldout_grounding_reaches_threshold": word["heldout"]["grounded_accuracy"]
        >= 0.95,
        "character_heldout_grounding_reaches_threshold": character["heldout"][
            "grounded_accuracy"
        ]
        >= 0.95,
        "word_heldout_generation_reaches_threshold": word["heldout"][
            "generation_roundtrip_accuracy"
        ]
        >= 0.95,
        "character_heldout_generation_reaches_threshold": character["heldout"][
            "generation_roundtrip_accuracy"
        ]
        >= 0.95,
        "all_twelve_concept_surfaces_are_grounded": word["lexicon"]["accuracy"] == 1.0
        and character["lexicon"]["accuracy"] == 1.0,
        "grammar_tokens_do_not_become_false_concepts": not word["lexicon"][
            "unexpected_lexeme_surfaces"
        ]
        and not character["lexicon"]["unexpected_lexeme_surfaces"],
        "all_opaque_parse_and_speak_patterns_crystallize": word["grammar"]["coverage"]
        == 1.0
        and character["grammar"]["coverage"] == 1.0,
        "opaque_truth_answers_are_learned_from_consequences": all(
            stage["answer_surfaces"][predicate]["status"] == "generated"
            and stage["answer_surfaces"][predicate]["text"] == expected
            for stage in (word, character)
            for predicate, expected in (("YES", "aya"), ("NO", "nox"))
        ),
        "generation_uses_only_opaque_vocabulary": word["generation_vocabulary"][
            "opaque_only"
        ]
        and character["generation_vocabulary"]["opaque_only"],
        "character_span_induction_reaches_threshold": character["character_spans"]["f1"]
        >= 0.95,
        "abstraction_preserves_opaque_language": all(
            stage["post_abstraction_heldout"]["grounded_accuracy"] + 0.02
            >= stage["heldout"]["grounded_accuracy"]
            for stage in (word, character)
        ),
        "models_retain_no_raw_evidence": all(
            stage["training"]["raw_episodes_after_abstraction"] == 0
            and stage["training"]["raw_evidence_after_abstraction"] == 0
            for stage in (word, character)
        ),
        "all_seven_primitives_run": all(
            set(stage["training"]["operator_counts"]) == set(UNIVERSE_PRIMITIVE_NAMES)
            and all(stage["training"]["operator_counts"].values())
            for stage in (word, character)
        ),
        "all_seven_single_primitive_ablations_are_causal": all(
            row["causal_effect_observed"]
            for row in report["primitive_ablations"].values()
        ),
        "phase_mixing_and_annealing_are_active": all(
            stage["training"]["cumulative_phase_energy"] > 0.0
            and stage["training"]["accepted_worse_moves"] > 0
            for stage in (word, character)
        ),
        "serialized_models_reload": all(
            stage["serialized_model_reloads"] for stage in (word, character)
        ),
        "deterministic_replay_matches": report["deterministic_replay"],
        "opaque_serialized_workflow_passes": report["serialized_workflow"]["passed"],
        "graph_rag_and_side_view_are_bound": report["knowledge_runtime"]["passed"]
        and report["side_view_contract"]["model_hash"] == report["primary_model_hash"],
    }
    return {
        "gates": gates,
        "failed": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
    }


def run_opaque_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_tests = run_opaque_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Opaque self-tests failed: {self_tests['failed']}")

    program = build_opaque_language_program()
    opaque_write_jsonl(output_dir / "atom_opaque_train.jsonl", program["train"])
    opaque_write_jsonl(
        output_dir / "atom_opaque_validation.jsonl", program["validation"]
    )
    opaque_write_jsonl(output_dir / "atom_opaque_heldout.jsonl", program["heldout"])
    opaque_write_json(
        output_dir / "atom_opaque_validation_truth.json",
        program["validation_truth"],
    )
    opaque_write_json(
        output_dir / "atom_opaque_evaluation_truth.json",
        program["evaluation_truth"],
    )
    opaque_write_json(
        output_dir / "atom_opaque_dataset_manifest.json", program["manifest"]
    )

    config = LanguageConfig()
    word_runtime, word_result, word_history, word_model = run_opaque_stage(
        program,
        "word",
        config,
    )
    character_runtime, character_result, character_history, character_model = (
        run_opaque_stage(program, "character", config)
    )
    opaque_write_json(output_dir / "atom_opaque_word_model.json", word_model)
    opaque_write_json(output_dir / "atom_opaque_character_model.json", character_model)
    opaque_write_json(output_dir / "atom_opaque_word_history.json", word_history)
    opaque_write_json(
        output_dir / "atom_opaque_character_history.json", character_history
    )

    base_program = build_grounded_language_program()
    frozen_base, _, _ = train_language_field(
        base_program["train"],
        "word",
        config=config,
    )
    frozen_base.abstract("opaque-frozen-english-base")
    frozen_english = evaluate_language_rows(
        frozen_base,
        program["heldout"],
        program["evaluation_truth"],
    )

    replay_runtime, replay_history, _ = train_language_field(
        program["train"],
        "word",
        config=config,
    )
    replay_runtime.abstract("opaque-word-final-abstraction")
    replay_model = language_model_payload(replay_runtime)
    deterministic_replay = replay_model["model_hash"] == word_model[
        "model_hash"
    ] and stable_hash(replay_history) == stable_hash(word_history)

    ablations = {
        primitive.value: run_opaque_ablation(program, primitive, config)
        for primitive in Primitive
    }

    request, expected_workflow = build_opaque_workflow(character_model)
    request_path = output_dir / "atom_opaque_workflow_request.json"
    response_path = output_dir / "atom_opaque_workflow_response.json"
    opaque_write_json(request_path, request)
    response = run_language_workflow(
        output_dir / "atom_opaque_character_model.json",
        request_path,
        response_path,
    )
    workflow = score_opaque_workflow(response, expected_workflow)

    graph = build_language_graph()
    retrieval = retrieve_atom_context(
        graph,
        "ground learn abstract grammar role answer consequence speak compose",
        limit=20,
    )
    retrieved_names = {row["name"] for row in retrieval}
    knowledge_runtime = {
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "graph_hash": stable_hash(graph.manifest()),
        "retrieved": retrieval,
        "passed": {
            "ground",
            "learn",
            "abstract",
            "role_bind",
            "speak",
        }
        <= retrieved_names,
    }

    example = next(
        row
        for row in program["heldout"]
        if program["evaluation_truth"][str(row["case_id"])]["family"]
        == "where_systematic"
    )
    interaction = interact_text(
        character_runtime,
        str(example["text"]),
        example["before"],
        {},
    )
    answer = None
    if isinstance(interaction.get("answer"), Mapping):
        answer = interaction["answer"].get("text")
    meaning = interaction["meaning"]

    report: dict[str, Any] = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "experiment": OPAQUE_EXPERIMENT_NAME,
        "primary_model_hash": character_model["model_hash"],
        "manifest": {
            "standard_neural_network": False,
            "gradient_descent": False,
            "backpropagation": False,
            "pretrained_model": False,
            "trainable_weight_matrix": False,
            "word_and_character_runs_are_independent": True,
            "answer_meanings_are_grounded_from_world_consequences": True,
            "language_runtime": ATOM_LANGUAGE_RUNTIME,
            "universe_primitives": list(UNIVERSE_PRIMITIVE_NAMES),
        },
        "opaque_program": program["manifest"],
        "self_tests": self_tests,
        "architecture_audit": architecture_audit(),
        "knowledge_runtime": knowledge_runtime,
        "stages": {"word": word_result, "character": character_result},
        "baselines": {
            "exact_surface": opaque_exact_surface_baseline(
                program["train"], program["heldout"]
            ),
            "frozen_english_word": frozen_english,
        },
        "primitive_ablations": ablations,
        "deterministic_replay": deterministic_replay,
        "serialized_workflow": workflow,
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
            "final_temperature": character_runtime.state.temperature,
            "word_phase_energy": word_result["training"]["cumulative_phase_energy"],
            "character_phase_energy": character_result["training"][
                "cumulative_phase_energy"
            ],
            "word_accepted_worse_moves": word_result["training"][
                "accepted_worse_moves"
            ],
            "character_accepted_worse_moves": character_result["training"][
                "accepted_worse_moves"
            ],
        },
        "side_view_contract": {
            "runtime": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
            "binding": ATOM_LANGUAGE_ARTIFACT_BINDING,
            "model_hash": character_model["model_hash"],
        },
        "models": {"word": word_model, "character": character_model},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["experiment_gates"] = opaque_experiment_gates(report)
    report_path = output_dir / "atom_opaque_report.json"
    opaque_write_json(report_path, report)
    side_path = render_language_artifact(
        report,
        character_model,
        output_dir / "atom_opaque_side_view.html",
    )
    report["artifact_side_view"] = {
        "path": side_path.name,
        "sha256": hashlib.sha256(side_path.read_bytes()).hexdigest(),
        "runtime_marker": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
        "binding_marker": ATOM_LANGUAGE_ARTIFACT_BINDING,
        "model_hash_bound": character_model["model_hash"],
    }
    opaque_write_json(report_path, report)
    return report


def parse_opaque_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opaque_outputs"),
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def opaque_main() -> None:
    args = parse_opaque_args()
    result = (
        run_opaque_self_tests()
        if args.self_test
        else run_opaque_experiment(args.output_dir.resolve())
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    opaque_main()
