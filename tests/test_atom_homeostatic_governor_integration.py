from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_homeostatic_dataset import build_homeostatic_program
from atom_homeostatic_experiment import run_homeostatic_experiment
from atom_homeostatic_governor import (
    ATOM_HOMEOSTATIC_RUNTIME,
    HomeostaticPrimitive,
    load_homeostatic_model,
    run_homeostatic_request,
)
from atom_homeostatic_side_view import (
    ATOM_HOMEOSTATIC_ARTIFACT_BINDING,
    ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME,
    render_homeostatic_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    build_homeostatic_graph,
)


class AtomHomeostaticGovernorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_homeostatic_experiment(cls.output_dir)
        cls.program = build_homeostatic_program()
        cls.model = json.loads(
            (cls.output_dir / "atom_homeostatic_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow = json.loads(
            (cls.output_dir / "atom_homeostatic_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_runtime_stream_does_not_expose_regimes_or_truth(self) -> None:
        forbidden = {"regime", "expected_effect", "is_noise"}
        self.assertTrue(self.program["manifest"]["evaluator_truth_separate"])
        self.assertTrue(
            all(not forbidden & set(row) for row in self.program["observations"])
        )
        serialized = json.dumps(self.model, sort_keys=True)
        for label in (
            "initial_crystallization",
            "noise_burst",
            "recovery",
            "law_shift",
            "consolidation",
        ):
            self.assertNotIn(label, serialized)

    def test_governor_rejects_noise_and_adapts_to_coherent_change(self) -> None:
        adaptive = self.report["comparison"]["adaptive"]
        fixed = self.report["comparison"]["fixed"]
        self.assertEqual(adaptive["final"]["accuracy"], 1.0)
        self.assertEqual(fixed["final"]["accuracy"], 0.0)
        self.assertEqual(adaptive["prequential"]["noise_burst"]["accuracy"], 1.0)
        self.assertEqual(adaptive["prequential"]["consolidation"]["accuracy"], 1.0)
        self.assertEqual(adaptive["replacement_counts"].get("noise_burst", 0), 0)
        self.assertEqual(adaptive["replacement_counts"]["law_shift"], 4)

    def test_three_controls_move_and_conservation_bounds_chaos(self) -> None:
        controller = self.report["controller"]
        self.assertLess(*controller["temperature_range"])
        self.assertLess(*controller["phase_strength_range"])
        self.assertLess(*controller["nucleation_threshold_range"])
        self.assertGreater(
            controller["action_counts"].get("reheat_coherent_shift", 0), 0
        )
        self.assertGreater(
            controller["action_counts"].get("cool_incoherent_disturbance", 0), 0
        )
        self.assertLessEqual(
            controller["maximum_chaos_load"],
            self.report["model_config"]["chaos_budget"],
        )
        self.assertGreater(controller["uphill_accepts"], 0)
        self.assertGreater(controller["uphill_rejections"], 0)
        self.assertFalse(controller["near_criticality_claimed"])
        self.assertFalse(controller["self_organized_criticality_claimed"])

    def test_seven_primitives_are_runtime_causal_and_model_fails_closed(self) -> None:
        self.assertEqual(
            set(self.report["model_training"]["operator_counts"]),
            {primitive.value for primitive in HomeostaticPrimitive},
        )
        self.assertEqual(
            set(self.report["model_training"]["operator_counts"]),
            set(UNIVERSE_PRIMITIVE_NAMES),
        )
        self.assertTrue(
            all(
                row["causal_effect_observed"]
                for row in self.report["primitive_ablations"].values()
            )
        )
        self.assertEqual(self.model["training"]["raw_event_count"], 0)
        self.assertEqual(self.model["training"]["raw_evidence_count"], 0)
        self.assertEqual(self.report["corruption_checks"]["rejected"], 7)
        self.assertTrue(self.report["corruption_checks"]["passed"])
        loaded = load_homeostatic_model(self.model)
        self.assertEqual(loaded.payload, self.model)

    def test_graph_rag_workflow_and_side_view_share_the_real_model(self) -> None:
        graph = build_homeostatic_graph()
        self.assertEqual(
            set(graph.expand("homeostatic_govern")),
            set(UNIVERSE_PRIMITIVE_NAMES),
        )
        self.assertEqual(
            self.workflow["runtime"]["homeostatic_runtime"],
            ATOM_HOMEOSTATIC_RUNTIME,
        )
        self.assertEqual(
            self.workflow["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME
        )
        self.assertEqual(self.workflow["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)
        self.assertTrue(all(turn["knowledge_context"] for turn in self.workflow["turns"]))
        request = json.loads(
            (self.output_dir / "atom_homeostatic_workflow_request.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            run_homeostatic_request(load_homeostatic_model(self.model), request),
            self.workflow,
        )
        document = (
            self.output_dir / "atom_homeostatic_side_view.html"
        ).read_text(encoding="utf-8")
        rendered_document = render_homeostatic_artifact(
            self.model,
            self.report,
            self.workflow,
        )
        self.assertEqual(rendered_document, document)
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME, document)
        self.assertIn(ATOM_HOMEOSTATIC_ARTIFACT_BINDING, document)
        self.assertNotIn("<button", document)
        self.assertNotIn("<input", document)
        self.assertTrue(self.report["experiment_gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
