from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from atom_causal_memory import (
    CAUSAL_MEMORY_RUNTIME,
    CausalMemoryClient,
    build_query_for_primitive,
    build_release_binary,
    build_structural_query,
    load_forge,
    structural_features_for,
)
from atom_causal_memory_experiment import (
    CAUSAL_MEMORY_EXPERIMENT_RUNTIME,
    run_causal_memory_experiment,
)
from atom_causal_memory_knowledge import (
    CAUSAL_MEMORY_RAG_RUNTIME,
    CAUSAL_MEMORY_WIKI_RUNTIME,
    CausalMemoryWikiGraph,
    retrieve_causal_memory_context,
    validate_causal_memory_knowledge,
)
from atom_causal_memory_side_view import (
    CAUSAL_MEMORY_SIDE_VIEW_RUNTIME,
    render_causal_memory_artifact,
)
from atom_causal_world_schema import canonical_hash
from atom_primitive_forge import PrimitiveForge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORGE_PATH = (
    PROJECT_ROOT
    / "primitive_forge_outputs"
    / "atom_primitive_graph.json"
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} did not contain an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AtomCausalMemoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.binary = build_release_binary()
        cls.report = run_causal_memory_experiment(cls.output_dir)
        cls.store = cls.output_dir / "atom_causal_memory.atomdb"
        cls.inventory = _read_json(
            cls.output_dir / "atom_causal_memory_inventory.json"
        )
        cls.workflow = _read_json(
            cls.output_dir / "atom_causal_memory_workflow.json"
        )
        cls.knowledge = _read_json(
            cls.output_dir / "atom_causal_memory_knowledge.json"
        )
        cls.forge = load_forge(FORGE_PATH)
        cls.client = CausalMemoryClient(cls.store, cls.binary)
        cls.query_wire = (
            cls.output_dir / "atom_causal_memory_query.txt"
        ).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_real_forge_graph_is_persisted_as_causal_glyphs(self) -> None:
        self.assertEqual(
            self.report["runtime"],
            CAUSAL_MEMORY_EXPERIMENT_RUNTIME,
        )
        self.assertEqual(
            self.report["memory_runtime"],
            CAUSAL_MEMORY_RUNTIME,
        )
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report["storage"]["glyph_count"], 69)
        self.assertEqual(self.report["storage"]["root_count"], 7)
        self.assertEqual(len(self.inventory["glyphs"]), 69)
        self.assertGreater(self.report["storage"]["motif_count"], 3_000)
        self.assertGreater(
            self.report["storage"]["durable_bonds"],
            10_000,
        )

    def test_structural_query_has_no_lexical_retrieval_features(
        self,
    ) -> None:
        target = self.report["retrieval"]["target_glyph"]
        features = structural_features_for(self.forge, target)
        forbidden = {
            "alias",
            "provenance",
            "evidence",
            "counterexample",
        }
        self.assertTrue(features)
        self.assertFalse(
            any(
                role.split("/", 1)[0] in forbidden
                for role, _ in features
            )
        )
        result = self.client.query(
            build_query_for_primitive(self.forge, target)
        )
        self.assertTrue(result["answerable"])
        self.assertEqual(result["hits"][0]["primitive_id"], target)
        self.assertEqual(
            result["hits"][0]["coverage_per_million"],
            1_000_000,
        )

    def test_retrieval_exposes_exact_motif_evidence_paths(self) -> None:
        result = self.client.query(self.query_wire)
        motifs = result["hits"][0]["motifs"]
        self.assertGreater(len(motifs), 8)
        self.assertTrue(
            all(
                len(item["motif"]) == 64
                and item["role"]
                and item["value"]
                and 1 <= item["conductance_per_mille"] <= 4_000
                for item in motifs
            )
        )
        self.assertTrue(
            any(item["strengthen_count"] >= 2 for item in motifs)
        )

    def test_prediction_feedback_is_durable_and_directional(self) -> None:
        wrong = self.report["learning"]["wrong_prediction"]
        correct = self.report["learning"]["correct_prediction"]
        target = self.report["retrieval"]["target_glyph"]
        competitor = self.report["retrieval"]["competing_glyph"]
        self.assertFalse(wrong["prediction_correct"])
        self.assertTrue(
            any(
                item["primitive_id"] == target
                and item["polarity"] == "strengthen"
                for item in wrong["adjustments"]
            )
        )
        self.assertTrue(
            any(
                item["primitive_id"] == competitor
                and item["polarity"] == "weaken"
                for item in wrong["adjustments"]
            )
        )
        self.assertTrue(correct["prediction_correct"])
        self.assertTrue(
            all(
                item["primitive_id"] == target
                and item["polarity"] == "strengthen"
                for item in correct["adjustments"]
            )
        )
        self.assertEqual(
            self.report["retrieval"]["after_correct_prediction"],
            self.report["retrieval"]["persisted"],
        )
        self.assertGreater(
            self.report["retrieval"]["target_score_increase"],
            0,
        )

    def test_queries_are_read_only_at_the_store_boundary(self) -> None:
        before = _sha256(self.store)
        first = self.client.query(self.query_wire)
        second = self.client.query(self.query_wire)
        after = _sha256(self.store)
        self.assertEqual(first, second)
        self.assertEqual(before, after)

    def test_reimport_is_idempotent_and_preserves_learning(self) -> None:
        before_hash = _sha256(self.store)
        before_query = self.client.query(self.query_wire)
        imported = self.client.import_forge(self.forge)
        after_query = self.client.query(self.query_wire)
        self.assertFalse(imported["committed"])
        self.assertEqual(before_hash, _sha256(self.store))
        self.assertEqual(before_query, after_query)
        self.assertEqual(imported["root_history_versions"], 1)

    def test_unknown_required_topology_abstains(self) -> None:
        unknown = self.report["retrieval"]["unknown"]
        self.assertFalse(unknown["answerable"])
        self.assertTrue(unknown["insufficient_evidence"])
        self.assertEqual(unknown["hits"], [])
        query = build_structural_query(
            query_id="integration:unknown-topology",
            features=(
                ("domain", "mathematical_scalar_field"),
                ("kind", "derived"),
                ("component/7777", "never-observed"),
            ),
            required_roles=(
                "domain",
                "kind",
                "component/7777",
            ),
            minimum_support=2,
        )
        self.assertTrue(
            self.client.query(query)["insufficient_evidence"]
        )

    def test_corrupt_forge_artifact_fails_before_import(self) -> None:
        model = self.forge.model_payload()
        corrupt = copy.deepcopy(model)
        derived = next(
            item for item in corrupt["primitives"] if not item["root"]
        )
        derived["confidence"] = 0.123456789
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            PrimitiveForge.from_model_payload(corrupt)

        cyclic = copy.deepcopy(model)
        records = [
            item for item in cyclic["primitives"] if not item["root"]
        ][:2]
        records[0]["recipe"]["components"][0] = records[1][
            "primitive_id"
        ]
        records[1]["recipe"]["components"][0] = records[0][
            "primitive_id"
        ]
        core = {
            key: cyclic[key] for key in cyclic if key != "graph_hash"
        }
        cyclic["graph_hash"] = canonical_hash(core)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            PrimitiveForge.from_model_payload(cyclic)

    def test_wiki_rag_and_side_view_are_runtime_bound(self) -> None:
        self.assertEqual(
            self.knowledge["wiki_runtime"],
            CAUSAL_MEMORY_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.knowledge["rag_runtime"],
            CAUSAL_MEMORY_RAG_RUNTIME,
        )
        validate_causal_memory_knowledge(
            self.knowledge,
            graph_hash=self.forge.graph_hash,
            catalog_identity=self.inventory["catalog_identity"],
        )
        graph = CausalMemoryWikiGraph(
            self.client,
            self.forge,
            self.inventory,
        )
        context = retrieve_causal_memory_context(
            graph,
            self.query_wire,
        )
        self.assertEqual(
            context[0]["primitive_id"],
            self.report["retrieval"]["target_glyph"],
        )
        self.assertTrue(context[0]["evidence_paths"])
        side_view = render_causal_memory_artifact(
            self.report,
            self.inventory,
            self.workflow,
            self.knowledge,
        )
        self.assertIn(CAUSAL_MEMORY_SIDE_VIEW_RUNTIME, str(
            self.report["side_view_contract"]["runtime"]
        ))
        self.assertIn("Causal Atom Memory", side_view)
        self.assertIn("Structural resonance", side_view)
        self.assertIn("Prediction-driven metaplasticity", side_view)
        self.assertIn(
            "render_causal_memory_artifact",
            Path(
                PROJECT_ROOT / "atom_causal_memory_side_view.py"
            ).read_text(encoding="utf-8"),
        )

    def test_side_view_and_knowledge_reject_tampering(self) -> None:
        report = copy.deepcopy(self.report)
        report["retrieval"]["target_glyph"] = "detached"
        with self.assertRaisesRegex(ValueError, "report hash mismatch"):
            render_causal_memory_artifact(
                report,
                self.inventory,
                self.workflow,
                self.knowledge,
            )
        knowledge = copy.deepcopy(self.knowledge)
        knowledge["nodes"][0]["status"] = "detached"
        with self.assertRaisesRegex(ValueError, "knowledge hash mismatch"):
            validate_causal_memory_knowledge(
                knowledge,
                graph_hash=self.forge.graph_hash,
                catalog_identity=self.inventory["catalog_identity"],
            )


if __name__ == "__main__":
    unittest.main()
