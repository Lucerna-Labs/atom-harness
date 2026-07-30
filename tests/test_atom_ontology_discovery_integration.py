from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_ontology_dataset import (
    ONTOLOGY_RELATION_ALIASES,
    build_ontology_discovery_program,
)
from atom_ontology_discovery import (
    ATOM_ONTOLOGY_RUNTIME,
    OntologyPrimitive,
    apply_ontology_text,
    discover_ontology,
    evaluate_ontology_rows,
    ontology_model_payload,
    runtime_from_ontology_model,
)
from atom_ontology_experiment import run_ontology_experiment
from atom_ontology_side_view import (
    ATOM_ONTOLOGY_ARTIFACT_BINDING,
    ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME,
    render_ontology_artifact,
)
from atom_runtime_knowledge import ATOM_RAG_RUNTIME, ATOM_WIKI_GRAPH_RUNTIME


class AtomOntologyDiscoveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_ontology_experiment(cls.output_dir)
        cls.program = build_ontology_discovery_program()
        cls.model = json.loads(
            (cls.output_dir / "atom_ontology_model.json").read_text(encoding="utf-8")
        )
        cls.runtime = runtime_from_ontology_model(cls.model)
        cls.mapping = cls.report["evaluator_law_mapping"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_dataset_hides_schema_types_and_evaluator_semantics(self) -> None:
        forbidden = {"semantic_label", "participants", "predicate", "roles"}
        aliases = {
            split: set(values.values())
            for split, values in ONTOLOGY_RELATION_ALIASES.items()
        }
        self.assertFalse(aliases["train"] & aliases["validation"])
        self.assertFalse(aliases["train"] & aliases["heldout"])
        self.assertFalse(aliases["validation"] & aliases["heldout"])
        self.assertEqual(self.program["manifest"]["relation_alias_overlap"], 0)
        self.assertFalse(self.program["manifest"]["typed_entity_prefixes_present"])
        self.assertTrue(
            all(
                not forbidden & set(row)
                for split in ("train", "validation", "heldout")
                for row in self.program[split]
            )
        )

    def test_structural_ontology_is_stable_across_renamed_worlds(self) -> None:
        signatures = {
            discover_ontology(self.program[split][0]["before"]).signature
            for split in ("train", "validation", "heldout")
        }
        self.assertEqual(len(signatures), 1)
        self.assertEqual(len(self.model["ontology"]["types"]), 3)
        self.assertEqual(len(self.model["ontology"]["relations"]), 2)
        serialized = json.dumps(self.model, sort_keys=True)
        for values in ONTOLOGY_RELATION_ALIASES.values():
            for alias in values.values():
                self.assertNotIn(alias, serialized)
        for prefix in ("agent-", "object-", "location-"):
            self.assertNotIn(prefix, serialized)

    def test_one_model_executes_all_unseen_alias_splits(self) -> None:
        self.assertEqual(len(self.model["surface_laws"]), 15)
        self.assertEqual(len(self.model["transition_laws"]), 5)
        for split, truth_name in (
            ("validation", "validation_truth"),
            ("heldout", "evaluation_truth"),
        ):
            result = evaluate_ontology_rows(
                self.runtime,
                self.program[split],
                self.program[truth_name],
                self.mapping,
            )
            self.assertEqual(result["execution_accuracy"], 1.0)
            self.assertEqual(result["law_accuracy"], 1.0)
            self.assertEqual(result["generation_accuracy"], 1.0)
        exchange = next(
            row
            for row in self.program["heldout"]
            if self.program["evaluation_truth"][str(row["case_id"])]["semantic_label"]
            == "exchange_locations"
        )
        result = apply_ontology_text(
            self.runtime, str(exchange["text"]), exchange["before"]
        )
        self.assertEqual(len(result["effects"]), 2)

    def test_primitive_dynamics_serialization_and_fail_closed_guards(self) -> None:
        self.assertEqual(
            set(self.report["training"]["operator_counts"]),
            {primitive.value for primitive in OntologyPrimitive},
        )
        self.assertTrue(all(self.report["training"]["operator_counts"].values()))
        self.assertGreater(
            self.report["controlled_chaos"]["cumulative_phase_energy"], 0.0
        )
        self.assertGreater(self.report["controlled_chaos"]["accepted_worse_moves"], 0)
        self.assertEqual(self.model["raw_episode_count"], 0)
        self.assertEqual(self.model["raw_evidence_count"], 0)
        self.assertEqual(ontology_model_payload(self.runtime), self.model)
        self.assertEqual(self.report["corruption_checks"]["rejected"], 5)
        self.assertTrue(self.report["corruption_checks"]["passed"])
        self.assertTrue(
            all(
                row["causal_effect_observed"]
                for row in self.report["primitive_ablations"].values()
            )
        )

    def test_real_workflow_graph_rag_and_side_view_share_the_model(self) -> None:
        self.assertTrue(self.report["serialized_workflow"]["passed"])
        self.assertEqual(self.report["serialized_workflow"]["correct"], 5)
        response = json.loads(
            (self.output_dir / "atom_ontology_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(response["runtime"]["ontology_runtime"], ATOM_ONTOLOGY_RUNTIME)
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        self.assertTrue(
            all(
                set(turn["ontology_binding"]) == {"b3x", "y8o"}
                for turn in response["turns"]
            )
        )
        document = (self.output_dir / "atom_ontology_side_view.html").read_text(
            encoding="utf-8"
        )
        rendered_document = render_ontology_artifact(
            self.model,
            self.report,
            response,
        )
        self.assertEqual(rendered_document, document)
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME, document)
        self.assertIn(ATOM_ONTOLOGY_ARTIFACT_BINDING, document)
        self.assertNotIn("<button", document)
        self.assertNotIn("<input", document)
        self.assertTrue(self.report["experiment_gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
