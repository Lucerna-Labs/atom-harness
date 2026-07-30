from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_runtime_knowledge import ATOM_RAG_RUNTIME, ATOM_WIKI_GRAPH_RUNTIME
from atom_transition_dataset import build_transition_discovery_program
from atom_transition_discovery import (
    ATOM_TRANSITION_RUNTIME,
    TransitionPrimitive,
    apply_transition_text,
    evaluate_transition_rows,
    runtime_from_transition_model,
    transition_model_payload,
)
from atom_transition_experiment import run_transition_experiment
from atom_transition_side_view import (
    ATOM_TRANSITION_ARTIFACT_BINDING,
    ATOM_TRANSITION_SIDE_VIEW_RUNTIME,
)


class AtomTransitionDiscoveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_transition_experiment(cls.output_dir)
        cls.program = build_transition_discovery_program()
        cls.model = json.loads(
            (cls.output_dir / "atom_transition_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.runtime = runtime_from_transition_model(cls.model)
        cls.mapping = cls.report["evaluator_law_mapping"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_observations_hide_semantics_and_surfaces_are_unseen(self) -> None:
        forbidden = {"semantic_label", "participants", "predicate", "roles"}
        self.assertEqual(
            self.program["manifest"]["counts"],
            {"train": 50, "validation": 15, "heldout": 25},
        )
        self.assertEqual(self.program["manifest"]["heldout_surface_overlap"], 0)
        self.assertTrue(
            all(
                not forbidden & set(row)
                for split in ("train", "validation", "heldout")
                for row in self.program[split]
            )
        )

    def test_runtime_discovers_and_executes_five_latent_laws(self) -> None:
        self.assertEqual(len(self.model["surface_laws"]), 15)
        self.assertEqual(len(self.model["transition_laws"]), 5)
        self.assertEqual(len(set(self.mapping.values())), 5)
        heldout = evaluate_transition_rows(
            self.runtime,
            self.program["heldout"],
            self.program["evaluation_truth"],
            self.mapping,
        )
        self.assertEqual(heldout["execution_accuracy"], 1.0)
        self.assertEqual(heldout["law_accuracy"], 1.0)
        self.assertEqual(heldout["generation_accuracy"], 1.0)

    def test_novel_exchange_and_release_exceed_fixed_inventory(self) -> None:
        novel = self.report["evaluation"]["novel_transitions"]
        fixed = self.report["baselines"]["fixed_predicate_delta"]
        self.assertEqual((novel["cases"], novel["execution_correct"]), (10, 10))
        self.assertEqual((fixed["novel_cases"], fixed["novel_recognized"]), (10, 0))
        exchange = next(
            row
            for row in self.program["heldout"]
            if self.program["evaluation_truth"][str(row["case_id"])][
                "semantic_label"
            ]
            == "exchange_locations"
        )
        result = apply_transition_text(
            self.runtime,
            str(exchange["text"]),
            exchange["before"],
        )
        self.assertEqual(len(result["effects"]), 2)
        self.assertEqual(
            result["world_after"],
            self.program["evaluation_truth"][str(exchange["case_id"])][
                "expected_after"
            ],
        )

    def test_core_dynamics_serialization_and_corruption_guards(self) -> None:
        training = self.report["training"]
        self.assertEqual(
            set(training["operator_counts"]),
            {primitive.value for primitive in TransitionPrimitive},
        )
        self.assertTrue(all(training["operator_counts"].values()))
        self.assertGreater(
            self.report["controlled_chaos"]["cumulative_phase_energy"],
            0.0,
        )
        self.assertGreater(
            self.report["controlled_chaos"]["accepted_worse_moves"],
            0,
        )
        self.assertEqual(self.model["raw_episode_count"], 0)
        self.assertEqual(self.model["raw_evidence_count"], 0)
        self.assertEqual(
            transition_model_payload(self.runtime)["model_hash"],
            self.model["model_hash"],
        )
        self.assertTrue(self.report["corruption_checks"]["passed"])
        self.assertEqual(self.report["corruption_checks"]["rejected"], 4)
        self.assertTrue(
            all(
                row["causal_effect_observed"]
                for row in self.report["primitive_ablations"].values()
            )
        )

    def test_real_workflow_graph_rag_and_side_view_share_one_model(self) -> None:
        self.assertTrue(self.report["serialized_workflow"]["passed"])
        self.assertEqual(self.report["serialized_workflow"]["correct"], 5)
        response = json.loads(
            (self.output_dir / "atom_transition_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            response["runtime"]["transition_runtime"],
            ATOM_TRANSITION_RUNTIME,
        )
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        side_path = self.output_dir / "atom_transition_side_view.html"
        document = side_path.read_text(encoding="utf-8")
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_TRANSITION_SIDE_VIEW_RUNTIME, document)
        self.assertIn(ATOM_TRANSITION_ARTIFACT_BINDING, document)
        self.assertTrue(self.report["experiment_gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
