from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from atom_causal_experience import (
    EXPERIENCE_MEMORY_RUNTIME,
    build_experience_query,
    load_experience_corpus,
)
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
    run_causal_experience_experiment,
)
from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
    validate_causal_experience_knowledge,
)
from atom_causal_experience_side_view import (
    CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME,
    render_causal_experience_artifact,
)
from atom_causal_world_schema import canonical_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AtomCausalExperienceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="atom-causal-experience-integration-"
        )
        cls.output_dir = Path(cls.temporary.name) / "runtime"
        cls.report = run_causal_experience_experiment(
            cls.output_dir,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        cls.inventory = json.loads(
            (cls.output_dir / "atom_causal_experience_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        cls.knowledge = json.loads(
            (cls.output_dir / "atom_causal_experience_knowledge.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow = json.loads(
            (cls.output_dir / "atom_causal_experience_workflow.json").read_text(
                encoding="utf-8"
            )
        )
        cls.side_view = (
            cls.output_dir / "atom_causal_experience_side_view.html"
        ).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_full_saved_world_is_durable(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(
            self.report["memory_runtime"],
            EXPERIENCE_MEMORY_RUNTIME,
        )
        self.assertEqual(
            self.report["source"]["observation_revisions"],
            2304,
        )
        self.assertEqual(self.report["source"]["laws"], 395)
        self.assertEqual(
            self.report["experience"]["experience_count"],
            2699,
        )
        self.assertEqual(len(self.inventory["experiences"]), 2699)

    def test_revisions_are_preserved_instead_of_overwritten(self) -> None:
        observations = [
            item
            for item in self.inventory["experiences"]
            if item["kind"] == "observation"
        ]
        self.assertEqual(len(observations), 2304)
        source = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        labels = {
            record.feature_values("source/id")[0] for record in source.observations
        }
        self.assertEqual(len(labels), 845)
        self.assertEqual(
            len({item["experience_id"] for item in observations}),
            2304,
        )

    def test_batches_append_and_replay_is_idempotent(self) -> None:
        observation = self.report["experience"]["observation_ingest"]
        laws = self.report["experience"]["law_ingest"]
        duplicate = self.report["experience"]["duplicate_ingest"]
        self.assertTrue(observation["committed"])
        self.assertTrue(laws["committed"])
        self.assertFalse(duplicate["committed"])
        self.assertGreater(
            laws["snapshot_sequence"],
            observation["snapshot_sequence"],
        )
        self.assertEqual(
            duplicate["snapshot_sequence"],
            laws["snapshot_sequence"],
        )
        self.assertEqual(len(self.inventory["batches"]), 2)

    def test_structural_recall_and_outcome_learning_survive_reopen(
        self,
    ) -> None:
        recall = self.report["recall"]
        target = recall["target_experience"]
        self.assertEqual(
            recall["before_feedback"]["hits"][0]["experience_id"],
            target,
        )
        self.assertEqual(
            recall["before_feedback"]["hits"][0]["coverage_per_million"],
            1_000_000,
        )
        self.assertGreater(recall["target_score_increase"], 0)
        self.assertEqual(
            recall["persisted"]["hits"][0]["experience_id"],
            target,
        )
        self.assertTrue(self.report["checks"]["learning_survives_process_reopen"])

    def test_recall_is_read_only_and_unknown_structure_abstains(
        self,
    ) -> None:
        self.assertTrue(self.report["checks"]["recall_is_read_only"])
        unknown = self.report["recall"]["unknown"]
        self.assertTrue(unknown["insufficient_evidence"])
        self.assertEqual(unknown["hits"], [])
        with self.assertRaisesRegex(ValueError, "audit-only"):
            build_experience_query(
                query_id="audit-metadata",
                features=[("provenance/hash", "0" * 64, True)],
                minimum_support=1,
            )

    def test_evidence_links_and_all_domains_are_graph_native(self) -> None:
        supported = [
            edge
            for edge in self.knowledge["edges"]
            if edge["relation"] == "supported_by"
        ]
        self.assertTrue(supported)
        domains = {item["domain"] for item in self.inventory["experiences"]}
        self.assertEqual(
            domains,
            {
                "agent",
                "biological",
                "chemical",
                "ecological",
                "language",
                "physical",
                "social",
                "symbolic",
            },
        )

    def test_wiki_rag_and_side_view_bind_the_real_store(self) -> None:
        source = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        validate_causal_experience_knowledge(
            self.knowledge,
            inventory=self.inventory,
            corpus=source,
        )
        self.assertEqual(
            self.knowledge["wiki_runtime"],
            CAUSAL_EXPERIENCE_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.knowledge["rag_runtime"],
            CAUSAL_EXPERIENCE_RAG_RUNTIME,
        )
        rendered = render_causal_experience_artifact(
            self.report,
            self.inventory,
            self.workflow,
            self.knowledge,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn(
            CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME,
            rendered,
        )
        self.assertIn("Persistent causal experience", rendered)
        self.assertIn(
            self.report["recall"]["target_experience"],
            rendered,
        )

    def test_artifact_tampering_fails_closed(self) -> None:
        report = copy.deepcopy(self.report)
        report["recall"]["target_experience"] = "detached"
        with self.assertRaisesRegex(ValueError, "report hash mismatch"):
            render_causal_experience_artifact(
                report,
                self.inventory,
                self.workflow,
                self.knowledge,
            )
        knowledge = copy.deepcopy(self.knowledge)
        knowledge["nodes"][0]["label"] = "detached"
        source = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        with self.assertRaisesRegex(ValueError, "knowledge hash mismatch"):
            validate_causal_experience_knowledge(
                knowledge,
                inventory=self.inventory,
                corpus=source,
            )

    def test_corrupt_source_model_fails_before_ingest(self) -> None:
        payload = json.loads((PROJECT_ROOT / DEFAULT_MODEL).read_text(encoding="utf-8"))
        payload["model_hash"] = "0" * 64
        corrupt = Path(self.temporary.name) / "corrupt-model.json"
        corrupt.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_experience_corpus(
                PROJECT_ROOT / DEFAULT_EVIDENCE,
                corrupt,
            )

    def test_active_runtime_declarations_advance_to_live_workflow(
        self,
    ) -> None:
        architecture = json.loads(
            (PROJECT_ROOT / "atom-causal-experience-architecture.json").read_text(
                encoding="utf-8"
            )
        )
        registry = json.loads(
            (PROJECT_ROOT / "ai-runtime-registry.json").read_text(encoding="utf-8")
        )
        causal_live = registry["runtimes"]["causal-live"]
        self.assertEqual(
            causal_live["runtime_entrypoint"],
            "atom_causal_live_experiment.py",
        )
        self.assertEqual(
            causal_live["wiki_runtime"],
            "CAUSAL_EXPERIENCE_WIKI_RUNTIME",
        )
        self.assertEqual(
            causal_live["rag_runtime"],
            "CAUSAL_EXPERIENCE_RAG_RUNTIME",
        )
        self.assertEqual(
            causal_live["side_view_runtime"],
            "CAUSAL_LIVE_SIDE_VIEW_RUNTIME",
        )
        self.assertEqual(
            causal_live["artifact_binding_marker"],
            "render_causal_live_artifact",
        )
        self.assertEqual(
            causal_live["integration_test"],
            "tests/test_atom_causal_live_integration.py",
        )
        self.assertEqual(
            architecture["runtime_entrypoint"],
            "atom_causal_experience_experiment.py",
        )
        self.assertEqual(
            architecture["integration_test"],
            "tests/test_atom_causal_experience_integration.py",
        )
        self.assertEqual(
            architecture["retrieval"]["runtime"],
            EXPERIENCE_MEMORY_RUNTIME,
        )

    def test_store_and_report_hashes_bind_exact_files(self) -> None:
        store = self.output_dir / "atom_causal_experience.atomdb"
        self.assertEqual(
            self.workflow["store_sha256"],
            hashlib.sha256(store.read_bytes()).hexdigest(),
        )
        core = {
            key: self.report[key] for key in sorted(self.report) if key != "report_hash"
        }
        self.assertEqual(
            self.report["report_hash"],
            canonical_hash(core),
        )


if __name__ == "__main__":
    unittest.main()
