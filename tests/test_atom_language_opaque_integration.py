from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    LanguageConfig,
    evaluate_language_rows,
    generate_text,
    language_model_payload,
    make_frame,
    run_language_workflow,
    runtime_from_language_model,
    train_language_field,
)
from atom_language_opaque_dataset import build_opaque_language_program
from atom_language_opaque_experiment import (
    build_opaque_workflow,
    opaque_grammar_score,
    opaque_lexicon_score,
    run_opaque_self_tests,
    score_opaque_workflow,
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


class AtomOpaqueLanguageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = build_opaque_language_program()
        cls.runtime, cls.history, cls.diagnostics = train_language_field(
            cls.program["train"],
            "word",
            config=LanguageConfig(),
        )
        cls.validation = evaluate_language_rows(
            cls.runtime,
            cls.program["validation"],
            cls.program["validation_truth"],
        )
        cls.heldout = evaluate_language_rows(
            cls.runtime,
            cls.program["heldout"],
            cls.program["evaluation_truth"],
        )
        cls.lexicon = opaque_lexicon_score(cls.runtime)
        cls.grammar = opaque_grammar_score(cls.runtime)
        cls.operator_counts = dict(cls.runtime.state.operator_counts)
        cls.phase_energy = cls.runtime.state.cumulative_phase_energy
        cls.accepted_worse_moves = cls.runtime.state.accepted_worse_moves
        cls.runtime.abstract("opaque-integration-final")
        cls.model = language_model_payload(cls.runtime)
        cls.restored = runtime_from_language_model(json.loads(json.dumps(cls.model)))

    def test_program_is_opaque_disjoint_and_evaluator_separated(self) -> None:
        self_tests = run_opaque_self_tests()
        self.assertTrue(self_tests["passed"], self_tests)
        self.assertEqual(
            self.program["manifest"]["counts"],
            {"train": 48, "validation": 20, "heldout": 52},
        )
        self.assertFalse(
            any(
                "frame" in row or "family" in row
                for split in ("train", "validation", "heldout")
                for row in self.program[split]
            )
        )

    def test_opaque_grammar_systematically_generalizes(self) -> None:
        self.assertEqual(self.diagnostics["unresolved_case_ids"], [])
        self.assertEqual(self.validation["grounded_accuracy"], 1.0)
        self.assertEqual(self.heldout["grounded_accuracy"], 1.0)
        self.assertEqual(self.heldout["generation_roundtrip_accuracy"], 1.0)
        self.assertEqual(self.heldout["answer_accuracy"], 1.0)
        self.assertEqual(self.lexicon["accuracy"], 1.0)
        self.assertEqual(self.lexicon["unexpected_lexeme_surfaces"], [])
        self.assertEqual(self.grammar["coverage"], 1.0)

    def test_truth_surfaces_are_learned_from_consequences(self) -> None:
        yes = generate_text(self.restored, make_frame("answer", "YES"))
        no = generate_text(self.restored, make_frame("answer", "NO"))
        self.assertEqual((yes["status"], yes["text"]), ("generated", "aya"))
        self.assertEqual((no["status"], no["text"]), ("generated", "nox"))
        observed_answers = {
            str(row["answer_text"])
            for row in self.program["train"]
            if row.get("answer_text")
        }
        self.assertNotIn("yes", observed_answers)
        self.assertNotIn("no", observed_answers)

    def test_core_dynamics_and_laws_only_serialization(self) -> None:
        self.assertEqual(set(self.operator_counts), set(UNIVERSE_PRIMITIVE_NAMES))
        self.assertTrue(all(self.operator_counts.values()))
        self.assertGreater(self.phase_energy, 0.0)
        self.assertGreater(self.accepted_worse_moves, 0)
        self.assertEqual(self.model["raw_episode_count"], 0)
        self.assertEqual(self.model["raw_evidence_count"], 0)
        self.assertEqual(
            language_model_payload(self.restored)["model_hash"],
            self.model["model_hash"],
        )

    def test_workflow_graph_rag_and_side_view_share_the_model(self) -> None:
        graph = build_language_graph()
        retrieved = retrieve_atom_context(
            graph,
            "ground learn abstract grammar role answer consequence speak compose",
            limit=20,
        )
        names = {row["name"] for row in retrieved}
        self.assertTrue({"ground", "learn", "abstract", "role_bind", "speak"} <= names)

        request, expected = build_opaque_workflow(self.model)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.json"
            request_path = root / "request.json"
            response_path = root / "response.json"
            model_path.write_text(json.dumps(self.model), encoding="utf-8")
            request_path.write_text(json.dumps(request), encoding="utf-8")
            response = run_language_workflow(
                model_path,
                request_path,
                response_path,
            )
            workflow = score_opaque_workflow(response, expected)
            self.assertTrue(workflow["passed"], workflow)

            first_meaning = next(
                turn["meaning"] for turn in response["turns"] if turn.get("meaning")
            )
            report = {
                "experiment": "atom_opaque_language_integration",
                "primary_model_hash": self.model["model_hash"],
                "stages": {
                    "word": {"heldout": self.heldout},
                    "character": {
                        "heldout": self.heldout,
                        "character_spans": {"f1": 1.0},
                    },
                },
                "experiment_gates": {
                    "gates": {"graph_rag": True, "serialized_workflow": True}
                },
                "side_view_interaction": {
                    "utterance": request["turns"][0]["text"],
                    "meaning": first_meaning,
                    "answer": None,
                    "semantic_mass": "2 roles -> 0 unexpressed",
                },
                "controlled_chaos": {
                    "initial_temperature": LanguageConfig().initial_temperature,
                    "final_temperature": self.restored.state.temperature,
                },
            }
            side_path = render_language_artifact(
                report,
                self.model,
                root / "side.html",
            )
            document = side_path.read_text(encoding="utf-8")
            self.assertIn(self.model["model_hash"], document)
            self.assertIn(ATOM_LANGUAGE_SIDE_VIEW_RUNTIME, document)
            self.assertIn(ATOM_LANGUAGE_ARTIFACT_BINDING, document)

        self.assertEqual(response["runtime"]["language_runtime"], ATOM_LANGUAGE_RUNTIME)
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)


if __name__ == "__main__":
    unittest.main()
