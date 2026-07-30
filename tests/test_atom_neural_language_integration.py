from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_field_proof import PROCESS_NAMES
from atom_neural_language_dataset import (
    RUNTIME_ROW_KEYS,
    build_neural_language_program,
)
from atom_neural_language_experiment import (
    NEURAL_EXPERIMENT_RUNTIME,
    run_neural_language_experiment,
    run_neural_workflow,
)
from atom_neural_language_model import load_neural_language_model
from atom_neural_language_side_view import (
    ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_neural_language_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_neural_language_graph,
)


class AtomNeuralLanguageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_neural_language_experiment(cls.output_dir)
        cls.program = build_neural_language_program()
        cls.model = json.loads(
            (cls.output_dir / "atom_neural_language_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow_request = json.loads(
            (cls.output_dir / "atom_neural_language_workflow_request.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow_response = json.loads(
            (cls.output_dir / "atom_neural_language_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_runtime_language_is_opaque_and_evaluator_truth_is_separate(self) -> None:
        evaluator_fields = {
            "composition_id",
            "control",
            "language",
            "operator_names",
            "query_type",
            "stage",
        }
        self.assertEqual(sum(self.report["dataset"]["stages"].values()), 2230)
        self.assertTrue(self.report["dataset"]["audit"]["passed"])
        for rows in self.program.stages.values():
            self.assertTrue(all(set(row) == RUNTIME_ROW_KEYS for row in rows))
            self.assertTrue(all(not evaluator_fields & set(row) for row in rows))
        serialized_runtime = json.dumps(self.program.stages, sort_keys=True)
        for operator in PROCESS_NAMES:
            self.assertNotIn(operator, serialized_runtime)

    def test_compositional_transfer_beats_flat_and_fixed_controls(self) -> None:
        evaluations = self.report["evaluations"]
        adaptive = evaluations["adaptive"]["transfer_composition"]
        fixed = evaluations["fixed"]["transfer_composition"]
        flat = evaluations["flat"]["base_composition"]
        retention = evaluations["adaptive"]["base_composition"]
        zero_shot = evaluations["adaptive"]["zero_shot_composition"]
        self.assertGreaterEqual(adaptive["joint_accuracy"], 0.84)
        self.assertGreaterEqual(adaptive["state_accuracy"], 0.90)
        self.assertGreaterEqual(adaptive["response_accuracy"], 0.88)
        self.assertGreaterEqual(
            adaptive["joint_accuracy"] - fixed["joint_accuracy"], 0.30
        )
        self.assertLessEqual(fixed["joint_accuracy"], 0.60)
        self.assertLessEqual(flat["joint_accuracy"], 0.10)
        self.assertGreaterEqual(retention["joint_accuracy"], 0.85)
        self.assertLessEqual(zero_shot["joint_accuracy"], 0.10)

    def test_homeostasis_learns_coherence_rejects_noise_and_forgets_events(
        self,
    ) -> None:
        controller = self.report["controller"]
        adaptation = self.report["adaptation"]["adaptive"]
        noise = controller["stage_counts"]["transfer_noise"]
        coherent_accepted = sum(
            controller["stage_counts"][stage]["accepted"]
            for stage in ("transfer_adaptation", "transfer_recovery")
        )
        coherent_windows = sum(
            controller["stage_counts"][stage]["windows"]
            for stage in ("transfer_adaptation", "transfer_recovery")
        )
        self.assertGreaterEqual(noise["rejected"] / noise["windows"], 0.65)
        self.assertGreaterEqual(coherent_accepted / coherent_windows, 0.60)
        self.assertLessEqual(controller["maximum_chaos_load"], 1.1500001)
        self.assertEqual(adaptation["raw_event_count"], 0)
        self.assertEqual(adaptation["lexicon_memory"]["raw_event_count"], 0)
        self.assertEqual(adaptation["query_surface_memory"]["raw_event_count"], 0)
        self.assertEqual(len(adaptation["query_surface_memory"]["query_laws"]), 6)

    def test_every_text_and_field_operator_is_causal_and_loading_fails_closed(
        self,
    ) -> None:
        self.assertEqual(set(self.report["ablations"]["text"]), set(PROCESS_NAMES))
        self.assertEqual(set(self.report["ablations"]["field"]), set(PROCESS_NAMES))
        self.assertTrue(self.report["ablations"]["all_text_operators_causal"])
        self.assertTrue(self.report["ablations"]["all_field_operators_causal"])
        self.assertTrue(self.report["architecture_audit"]["passed"])
        self.assertTrue(self.report["corruption_checks"]["passed"])
        self.assertEqual(self.report["corruption_checks"]["rejected"], 7)
        loaded = load_neural_language_model(self.model)
        self.assertEqual(loaded.model_hash, self.model["model_hash"])
        self.assertTrue(self.report["serialization"]["roundtrip_exact"])
        self.assertTrue(self.report["deterministic_training"]["passed"])

    def test_evidence_gate_sample_efficiency_and_adaptive_compute(self) -> None:
        evidence = self.report["evidence_boundary"]
        self.assertGreaterEqual(evidence["supported"]["coverage"], 0.85)
        self.assertGreaterEqual(evidence["supported"]["assertion_accuracy"], 0.90)
        self.assertEqual(evidence["unsupported"]["correct_abstention_rate"], 1.0)
        self.assertEqual(evidence["unsupported"]["flat_assertion_rate"], 1.0)
        self.assertGreaterEqual(evidence["compute"]["reduction"], 0.50)
        self.assertGreaterEqual(evidence["surface_memory"]["reduction"], 0.66)
        sample = self.report["sample_efficiency"]
        self.assertEqual(sample["example_fraction"], 0.25)
        self.assertGreaterEqual(sample["evaluation"]["joint_accuracy"], 0.80)

    def test_graph_rag_workflow_and_side_view_bind_the_serialized_model(self) -> None:
        graph = build_neural_language_graph()
        self.assertEqual(
            set(graph.expand("lifelong_language_adapt")), set(PROCESS_NAMES)
        )
        self.assertEqual(set(graph.expand("evidence_bound_claim")), set(PROCESS_NAMES))
        response = run_neural_workflow(self.model, self.workflow_request)
        self.assertEqual(response, self.workflow_response)
        self.assertEqual(
            response["runtime"]["neural_language_runtime"],
            NEURAL_EXPERIMENT_RUNTIME,
        )
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        unknown_turns = [
            turn
            for turn in response["turns"]
            if turn["artifact"]["claim_status"] == "unknown"
        ]
        self.assertEqual(len(unknown_turns), 2)
        self.assertTrue(
            all(turn["artifact"]["response"] is None for turn in unknown_turns)
        )
        self.assertTrue(
            all(
                turn["artifact"]["reasoning"]["execution_skipped"]
                for turn in unknown_turns
            )
        )
        self.assertTrue(self.report["serialized_workflow"]["passed"])
        document = (self.output_dir / "atom_neural_language_side_view.html").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            render_neural_language_artifact(
                self.model,
                self.report,
                self.workflow_response,
            ),
            document,
        )
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME, document)
        self.assertIn("render_neural_language_artifact", document)
        self.assertNotIn("<button", document)
        self.assertNotIn("<input", document)
        self.assertTrue(self.report["experiment_gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
