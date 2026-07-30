from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_language_dataset import build_grounded_language_program
from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    LanguageConfig,
    LanguageRuntime,
    adapt_language_field,
    character_span_f1,
    evaluate_language_rows,
    language_model_payload,
    lexeme_maps,
    run_language_workflow,
    runtime_from_language_model,
    train_language_field,
)
from atom_language_side_view import (
    ATOM_LANGUAGE_ARTIFACT_BINDING,
    ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_language_artifact,
)
from atom_language_transfer_dataset import (
    TRANSFER_LEXICON,
    build_language_transfer_program,
)
from atom_language_transfer_experiment import (
    build_transfer_workflow,
    run_transfer_self_tests,
    score_transfer_workflow,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_language_graph,
    retrieve_atom_context,
)


class AtomLanguageTransferIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_program = build_grounded_language_program()
        cls.transfer_program = build_language_transfer_program()
        base_runtime, _, cls.base_diagnostics = train_language_field(
            cls.base_program["train"],
            "word",
            config=LanguageConfig(),
        )
        base_runtime.abstract("transfer-integration-base")
        cls.base_model = language_model_payload(base_runtime)
        cls.runtime = runtime_from_language_model(
            json.loads(json.dumps(cls.base_model))
        )
        cls.history, cls.adaptation = adapt_language_field(
            cls.runtime,
            cls.transfer_program["grounding"],
            transient_rows=cls.transfer_program["transient"],
            bootstrap_epochs=2,
            grounded_epochs=3,
            adaptation_id="transfer-integration",
        )
        cls.transfer_evaluation = evaluate_language_rows(
            cls.runtime,
            cls.transfer_program["heldout"],
            cls.transfer_program["evaluation_truth"],
        )
        cls.base_retention = evaluate_language_rows(
            cls.runtime,
            cls.base_program["heldout"],
            cls.base_program["evaluation_truth"],
        )
        cls.character_spans = character_span_f1(
            cls.runtime,
            cls.transfer_program["heldout"],
        )
        cls.operator_counts = dict(cls.runtime.state.operator_counts)
        cls.cumulative_phase_energy = cls.runtime.state.cumulative_phase_energy
        cls.accepted_worse_moves = cls.runtime.state.accepted_worse_moves
        cls.transient_noise_retained = any(
            "florp" in trace.tokens for trace in cls.runtime.state.traces
        )
        cls.runtime.abstract("transfer-integration-final")
        cls.model = language_model_payload(cls.runtime)
        cls.restored = runtime_from_language_model(json.loads(json.dumps(cls.model)))

    def test_program_is_strictly_disjoint_and_evaluator_truth_is_separate(
        self,
    ) -> None:
        self_tests = run_transfer_self_tests()
        self.assertTrue(self_tests["passed"], self_tests)
        self.assertEqual(
            self.transfer_program["manifest"]["counts"],
            {"grounding": 12, "transient": 2, "heldout": 48},
        )
        self.assertEqual(
            self.transfer_program["manifest"]["heldout_action_surface_overlap"],
            0,
        )

    def test_adaptation_transfers_and_retains_without_memorizing_noise(self) -> None:
        self.assertEqual(self.base_diagnostics["unresolved_case_ids"], [])
        self.assertEqual(self.adaptation["unresolved_case_ids"], [])
        self.assertEqual(self.transfer_evaluation["grounded_accuracy"], 1.0)
        self.assertEqual(
            self.transfer_evaluation["generation_roundtrip_accuracy"],
            1.0,
        )
        self.assertEqual(self.transfer_evaluation["reference_accuracy"], 1.0)
        self.assertEqual(self.base_retention["grounded_accuracy"], 1.0)
        self.assertEqual(
            self.adaptation["retained_base_lexemes"],
            self.adaptation["base_lexemes"],
        )
        self.assertEqual(
            self.adaptation["retained_base_frames"],
            self.adaptation["base_frames"],
        )

        surface_to_concept, _ = lexeme_maps(self.restored.state.lexeme_laws)
        expected = {surface: concept for concept, surface in TRANSFER_LEXICON.items()}
        self.assertEqual(
            {surface: surface_to_concept.get(surface) for surface in expected},
            expected,
        )
        self.assertEqual(surface_to_concept["lumi"], "agent-4")
        self.assertNotIn("florp", surface_to_concept)
        self.assertFalse(self.transient_noise_retained)

    def test_adaptation_uses_the_full_core_and_serializes_laws_only(self) -> None:
        self.assertEqual(set(self.operator_counts), set(UNIVERSE_PRIMITIVE_NAMES))
        self.assertTrue(all(self.operator_counts.values()))
        self.assertGreater(self.cumulative_phase_energy, 0.0)
        self.assertGreater(self.accepted_worse_moves, 0)
        self.assertEqual(self.model["raw_episode_count"], 0)
        self.assertEqual(self.model["raw_evidence_count"], 0)
        self.assertEqual(
            language_model_payload(self.restored)["model_hash"],
            self.model["model_hash"],
        )

    def test_transfer_workflow_graph_rag_and_side_view_share_the_model(self) -> None:
        graph = build_language_graph()
        retrieved = retrieve_atom_context(
            graph,
            "learn revise remember forget ground lexeme role transfer context speak",
            limit=20,
        )
        names = {row["name"] for row in retrieved}
        self.assertTrue(
            {"learn", "remember", "forget", "ground", "role_bind", "speak"} <= names
        )

        request, expected = build_transfer_workflow(self.model)
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
            workflow = score_transfer_workflow(response, expected)
            self.assertTrue(workflow["passed"], workflow)

            first_meaning = next(
                turn["meaning"] for turn in response["turns"] if turn.get("meaning")
            )
            report = {
                "experiment": "atom_language_transfer_integration",
                "primary_model_hash": self.model["model_hash"],
                "stages": {
                    "word": {"heldout": self.transfer_evaluation},
                    "character": {
                        "heldout": self.transfer_evaluation,
                        "character_spans": self.character_spans,
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

    def test_adaptation_rejects_a_runtime_without_persistent_laws(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires persistent lexical and frame laws",
        ):
            adapt_language_field(
                LanguageRuntime("word"),
                self.transfer_program["grounding"],
            )


if __name__ == "__main__":
    unittest.main()
