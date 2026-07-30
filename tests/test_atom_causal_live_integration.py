from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from atom_causal_experience import (
    ExperienceMemoryClient,
    load_experience_corpus,
)
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
    validate_causal_experience_knowledge,
)
from atom_causal_live import (
    CAUSAL_LIVE_RUNTIME,
    LiveCausalEvent,
    LiveCausalRuntime,
    extend_corpus_from_inventory,
)
from atom_causal_live_experiment import run_causal_live_experiment
from atom_causal_live_side_view import (
    CAUSAL_LIVE_SIDE_VIEW_RUNTIME,
    render_causal_live_artifact,
)
from atom_causal_memory import RELEASE_BINARY
from atom_causal_world_schema import canonical_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AtomCausalLiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="atom-causal-live-integration-"
        )
        cls.output_dir = Path(cls.temporary.name) / "runtime"
        cls.report = run_causal_live_experiment(
            cls.output_dir,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        cls.store_path = (
            cls.output_dir / "base_experience" / "atom_causal_experience.atomdb"
        )
        cls.inventory = _read_json(cls.output_dir / "atom_causal_live_inventory.json")
        cls.knowledge = _read_json(cls.output_dir / "atom_causal_live_knowledge.json")
        cls.workflow = _read_json(cls.output_dir / "atom_causal_live_workflow.json")
        cls.side_view = (cls.output_dir / "atom_causal_live_side_view.html").read_text(
            encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_novel_outcome_changes_the_next_session_prediction(
        self,
    ) -> None:
        self.assertTrue(self.report["passed"])
        self.assertEqual(
            self.report["live_runtime"],
            CAUSAL_LIVE_RUNTIME,
        )
        first = self.report["cycles"]["first_novel_outcome"]
        second = self.report["cycles"]["second_outcome"]
        self.assertFalse(first["prediction_correct"])
        self.assertTrue(second["prediction_correct"])
        self.assertEqual(
            second["selected_experience"],
            first["experience_id"],
        )
        self.assertEqual(
            second["selected_effect"],
            first["observed_effect"],
        )

    def test_exact_replay_cannot_apply_feedback_twice(self) -> None:
        replay = self.report["cycles"]["replay"]
        self.assertTrue(replay["replayed"])
        self.assertFalse(replay["ingest"]["committed"])
        self.assertFalse(replay["feedback"]["committed"])
        self.assertIsNone(replay["feedback"]["cell_identity"])
        self.assertEqual(replay["feedback"]["adjustments"], [])
        self.assertEqual(
            replay["store_hash_before"],
            replay["store_hash_after"],
        )

    def test_outcome_key_conflict_fails_without_mutating_store(
        self,
    ) -> None:
        first = self.report["cycles"]["first_novel_outcome"]
        before = _sha256(self.store_path)
        client = ExperienceMemoryClient(
            self.store_path,
            RELEASE_BINARY,
        )
        with self.assertRaisesRegex(RuntimeError, "outcome key conflicts"):
            client.observe_outcome_once(
                first["outcome_query"],
                outcome_key=first["event_hash"],
                expected_experience=first["experience_id"],
                selected_experience=first["experience_id"],
            )
        self.assertEqual(_sha256(self.store_path), before)

    def test_untrusted_or_malformed_outcomes_fail_closed(self) -> None:
        payload = copy.deepcopy(self.report["events"]["first"])
        payload["outcome"]["authority_kind"] = "self_generated"
        with self.assertRaisesRegex(ValueError, "not trusted"):
            LiveCausalEvent.from_manifest(payload)
        payload = copy.deepcopy(self.report["events"]["first"])
        payload["atom_program"] = ["unknown_root"]
        with self.assertRaisesRegex(ValueError, "unknown universe"):
            LiveCausalEvent.from_manifest(payload)
        payload = copy.deepcopy(self.report["events"]["first"])
        payload["outcome"]["magnitude"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            LiveCausalEvent.from_manifest(payload)
        client = ExperienceMemoryClient(
            self.store_path,
            RELEASE_BINARY,
        )
        runtime = LiveCausalRuntime(
            client,
            authorities={"deterministic-live-world-v1": "simulator"},
        )
        event = LiveCausalEvent.from_manifest(self.report["events"]["first"])
        tampered_evidence = copy.deepcopy(self.report["events"]["first_evidence"])
        tampered_evidence["trace"]["label"] = "forged-trace"
        before = _sha256(self.store_path)
        with self.assertRaisesRegex(ValueError, "hash binding"):
            runtime.observe(event, tampered_evidence)
        self.assertEqual(_sha256(self.store_path), before)

    def test_full_inventory_exposes_bound_structural_features(
        self,
    ) -> None:
        self.assertEqual(len(self.inventory["experiences"]), 2701)
        self.assertEqual(len(self.inventory["batches"]), 4)
        for item in self.inventory["experiences"]:
            self.assertEqual(
                item["feature_count"],
                len(item["features"]),
            )
        first_id = self.report["cycles"]["first_novel_outcome"]["experience_id"]
        first = next(
            item
            for item in self.inventory["experiences"]
            if item["experience_id"] == first_id
        )
        self.assertIn(
            {"role": "authority/kind", "value": "simulator"},
            first["features"],
        )

    def test_live_sessions_are_in_the_runtime_wiki_and_rag(
        self,
    ) -> None:
        sessions = [
            node for node in self.knowledge["nodes"] if node["kind"] == "live_session"
        ]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(
            len(
                [
                    edge
                    for edge in self.knowledge["edges"]
                    if edge["relation"] == "observed_in"
                ]
            ),
            2,
        )
        first_id = self.report["cycles"]["first_novel_outcome"]["experience_id"]
        self.assertTrue(
            any(
                item["experience_id"] == first_id for item in self.report["rag_context"]
            )
        )
        base = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        extended = extend_corpus_from_inventory(
            base,
            self.inventory,
        )
        validate_causal_experience_knowledge(
            self.knowledge,
            inventory=self.inventory,
            corpus=extended,
        )

    def test_side_view_renders_the_bound_live_artifact(self) -> None:
        rendered = render_causal_live_artifact(
            self.report,
            self.inventory,
            self.workflow,
            self.knowledge,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn(CAUSAL_LIVE_SIDE_VIEW_RUNTIME, rendered)
        self.assertIn("Live causal learning", rendered)
        self.assertIn("live-session-alpha", json.dumps(self.knowledge))
        tampered = copy.deepcopy(self.report)
        tampered["experience_count"] = 0
        with self.assertRaisesRegex(ValueError, "report hash mismatch"):
            render_causal_live_artifact(
                tampered,
                self.inventory,
                self.workflow,
                self.knowledge,
            )

    def test_runtime_declarations_preserve_live_learning(self) -> None:
        registry = _read_json(PROJECT_ROOT / "ai-runtime-registry.json")
        knowledge_contract = _read_json(PROJECT_ROOT / "ai-runtime-knowledge.json")
        side_view_contract = _read_json(PROJECT_ROOT / "ai-artifact-side-view.json")
        self.assertEqual(registry["active_runtime"], "language-harness-v3")
        self.assertNotIn("generative-english", registry["runtimes"])
        self.assertIn("language-harness-v2", registry["runtimes"])
        self.assertIn("language-harness-v3", registry["runtimes"])
        causal_live = registry["runtimes"]["causal-live"]
        architecture = _read_json(PROJECT_ROOT / "atom-causal-live-architecture.json")
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
            "atom_causal_live_experiment.py",
        )
        self.assertEqual(
            architecture["integration_test"],
            "tests/test_atom_causal_live_integration.py",
        )
        self.assertEqual(
            knowledge_contract["runtime_entrypoint"],
            "atom_harness_experiment.py",
        )
        self.assertEqual(
            knowledge_contract["wiki_graph"]["module_path"],
            "atom_harness_knowledge.py",
        )
        self.assertEqual(
            knowledge_contract["rag"]["module_path"],
            "atom_harness_knowledge.py",
        )
        self.assertEqual(
            side_view_contract["runtime_entrypoint"],
            "atom_harness_experiment.py",
        )
        self.assertEqual(
            side_view_contract["side_view"]["module_path"],
            "atom_harness_side_view.py",
        )

    def test_store_report_and_workflow_hashes_bind_exact_files(
        self,
    ) -> None:
        self.assertEqual(
            self.report["store_sha256"],
            _sha256(self.store_path),
        )
        report_core = {
            key: self.report[key] for key in sorted(self.report) if key != "report_hash"
        }
        self.assertEqual(
            self.report["report_hash"],
            canonical_hash(report_core),
        )
        workflow_core = {
            key: self.workflow[key]
            for key in sorted(self.workflow)
            if key != "workflow_hash"
        }
        self.assertEqual(
            self.workflow["workflow_hash"],
            canonical_hash(workflow_core),
        )
        self.assertEqual(
            self.workflow["inventory_hash"],
            canonical_hash(self.inventory),
        )

    def test_reopened_recall_is_read_only_and_prefers_learning(
        self,
    ) -> None:
        self.assertTrue(self.report["checks"]["live_learning_survives_process_reopen"])
        self.assertTrue(self.report["checks"]["recall_remains_read_only"])
        first_id = self.report["cycles"]["first_novel_outcome"]["experience_id"]
        self.assertEqual(
            self.report["persisted_recall"]["hits"][0]["experience_id"],
            first_id,
        )
        self.assertEqual(
            self.knowledge["wiki_runtime"],
            CAUSAL_EXPERIENCE_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.knowledge["rag_runtime"],
            CAUSAL_EXPERIENCE_RAG_RUNTIME,
        )


if __name__ == "__main__":
    unittest.main()
