from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from atom_causal_world_schema import canonical_hash
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_harness_operator import (
    ATOM_HARNESS_OPERATOR_JOURNAL_RUNTIME,
    AtomHarnessOperator,
    OperatorCapacityError,
)
from atom_llm_protocol import CancellationToken
from scripts.certify_atom_harness_operator import (
    MAX_SETTLED_WORKING_SET_GROWTH_BYTES,
    MAX_WORKING_SET_BYTES,
    _working_set_evidence,
)


class _FakeSession:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.started = threading.Event()
        self.closed = False
        self.fail_next = False
        self.answers = 0

    def preload_runtime(self) -> Mapping[str, Any]:
        return {
            "schema": 1,
            "runtime": "fake-session",
            "operation": "session-runtime-preload",
            "knowledge": {
                "wiki_runtime": ATOM_HARNESS_WIKI_RUNTIME,
                "rag_runtime": ATOM_HARNESS_RAG_RUNTIME,
                "knowledge_hash": "a" * 64,
            },
            "providers": {
                "runtime": "fake-fabric",
                "preload_hash": "b" * 64,
            },
            "secrets_persisted": False,
        }

    def answer(
        self,
        question: str,
        *,
        output_dir: Path,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        del output_dir
        token = cancellation or CancellationToken()
        self.started.set()
        while not self.gate.wait(0.01):
            token.raise_if_cancelled()
        token.raise_if_cancelled()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic provider failure with private detail")
        self.answers += 1
        artifact_hash = canonical_hash({"question": question, "ordinal": self.answers})
        return {
            "artifact_hash": artifact_hash,
            "request_id": canonical_hash({"question": question}),
            "response": {
                "answerable": True,
                "answer": f"Answer {self.answers}",
                "citations": ["experience:test"],
                "limitations": "fixture",
            },
            "knowledge": {
                "knowledge_hash": "a" * 64,
                "graph_knowledge_hash": "c" * 64,
            },
            "timings": {"total_ms": 1},
            "provider_routes": [{"route_hash": "d" * 64}],
        }

    def manifest(self) -> Mapping[str, Any]:
        return {
            "schema": 1,
            "runtime": "fake-session",
            "providers": [
                {
                    "provider": "fake-provider",
                    "lane": {
                        "alive": not self.closed,
                        "model_load_count": 1,
                        "restart_count": 0,
                    },
                }
            ],
            "closed": self.closed,
        }

    def close(self) -> None:
        self.closed = True


def _verified_transaction(_path: Path) -> dict[str, Any]:
    return {"transaction_id": "e" * 64}


class AtomHarnessOperatorTests(unittest.TestCase):
    def test_working_set_evidence_separates_process_generations(self) -> None:
        gibibyte = 1024 * 1024 * 1024
        samples = []
        for ordinal in range(20):
            samples.extend(
                (
                    {
                        "process_generation": 1,
                        "process_id": 101,
                        "working_set_bytes": 12 * gibibyte + ordinal * 1024,
                    },
                    {
                        "process_generation": 2,
                        "process_id": 202,
                        "working_set_bytes": 7 * gibibyte + ordinal * 2048,
                    },
                )
            )

        evidence = _working_set_evidence(samples)

        self.assertTrue(evidence["observed"])
        self.assertEqual(evidence["process_count"], 2)
        self.assertTrue(evidence["growth_bounded"])
        self.assertTrue(evidence["ceiling_bounded"])
        global_values = [item["working_set_bytes"] for item in samples]
        self.assertGreater(
            max(global_values) - min(global_values),
            MAX_SETTLED_WORKING_SET_GROWTH_BYTES,
        )

        leaking = [
            {
                "process_generation": 3,
                "process_id": 303,
                "working_set_bytes": (
                    8 * gibibyte
                    if ordinal < 17
                    else 8 * gibibyte + MAX_SETTLED_WORKING_SET_GROWTH_BYTES + 1
                ),
            }
            for ordinal in range(20)
        ]
        self.assertFalse(_working_set_evidence(leaking)["growth_bounded"])

        over_ceiling = [
            {
                "process_generation": 4,
                "process_id": 404,
                "working_set_bytes": MAX_WORKING_SET_BYTES + 1,
            }
        ]
        self.assertFalse(_working_set_evidence(over_ceiling)["ceiling_bounded"])

    def test_bounded_queue_cancel_retry_and_safe_journal(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="atom-operator-unit-")
        root = Path(temporary.name)
        session = _FakeSession()
        operator = AtomHarnessOperator(session, state_root=root, max_queue_depth=1)
        with patch(
            "atom_harness_operator.verify_committed_run",
            side_effect=_verified_transaction,
        ):
            try:
                operator.start()
                first = operator.submit("first question")
                self.assertTrue(session.started.wait(5))
                queued = operator.submit("queued question")
                with self.assertRaises(OperatorCapacityError):
                    operator.submit("over capacity")
                cancelled = operator.cancel(queued["request_id"])
                self.assertEqual(cancelled["status"], "cancelled")
                session.gate.set()
                self.assertEqual(
                    operator.wait_for_terminal(first["request_id"])["status"],
                    "completed",
                )
                self.assertEqual(
                    operator.wait_for_terminal(queued["request_id"])["status"],
                    "cancelled",
                )
                retried = operator.retry(queued["request_id"])
                retried_terminal = operator.wait_for_terminal(retried["request_id"])
                self.assertEqual(retried_terminal["status"], "completed")
                self.assertEqual(
                    retried_terminal["parent_request_id"],
                    queued["request_id"],
                )
                self.assertEqual(retried_terminal["attempt"], 2)

                journal = json.loads(operator.journal_path.read_text(encoding="utf-8"))
                journal_hash = journal.pop("journal_hash")
                self.assertEqual(journal_hash, canonical_hash(journal))
                self.assertEqual(
                    journal["runtime"],
                    ATOM_HARNESS_OPERATOR_JOURNAL_RUNTIME,
                )
                self.assertFalse(journal["secrets_persisted"])
                self.assertNotIn(
                    "synthetic provider failure with private detail",
                    json.dumps(journal),
                )
                signals = {item["signal"] for item in journal["flow_events"]}
                self.assertIn("operator-queue-capacity", signals)
                self.assertIn("operator-cancellation", signals)
            finally:
                session.gate.set()
                operator.shutdown(wait=True, cancel_pending=True)
                temporary.cleanup()

    def test_active_cancel_is_preemptible_and_error_text_is_hashed(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="atom-operator-cancel-")
        root = Path(temporary.name)
        session = _FakeSession()
        operator = AtomHarnessOperator(session, state_root=root)
        with patch(
            "atom_harness_operator.verify_committed_run",
            side_effect=_verified_transaction,
        ):
            try:
                operator.start()
                submitted = operator.submit("cancel this active request")
                self.assertTrue(session.started.wait(5))
                operator.cancel(submitted["request_id"])
                terminal = operator.wait_for_terminal(submitted["request_id"])
                self.assertEqual(terminal["status"], "cancelled")
                self.assertEqual(
                    terminal["error"]["type"],
                    "ProviderCancelledError",
                )
                journal_text = operator.journal_path.read_text(encoding="utf-8")
                self.assertNotIn("cancelled by Atom operator", journal_text)
                self.assertIn("message_sha256", journal_text)
            finally:
                session.gate.set()
                operator.shutdown(wait=True, cancel_pending=True)
                temporary.cleanup()

    def test_restart_recovers_inflight_journal_record_as_interrupted(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="atom-operator-recover-")
        root = Path(temporary.name)
        first_session = _FakeSession()
        first_session.gate.set()
        first_operator = AtomHarnessOperator(first_session, state_root=root)
        with patch(
            "atom_harness_operator.verify_committed_run",
            side_effect=_verified_transaction,
        ):
            first_operator.start()
            submitted = first_operator.submit("recover this journal record")
            first_operator.wait_for_terminal(submitted["request_id"])
            first_operator.shutdown(wait=True)

        journal = json.loads(first_operator.journal_path.read_text(encoding="utf-8"))
        journal.pop("journal_hash")
        record = journal["requests"][submitted["request_id"]]
        record["status"] = "running"
        record["finished_at"] = None
        record["error"] = None
        journal["journal_hash"] = canonical_hash(journal)
        first_operator.journal_path.write_text(
            json.dumps(journal, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        second_session = _FakeSession()
        second_session.gate.set()
        recovered = AtomHarnessOperator(second_session, state_root=root)
        try:
            request = recovered.request_snapshot(submitted["request_id"])
            self.assertEqual(request["status"], "interrupted")
            self.assertEqual(request["error"]["kind"], "operator-restart")
            snapshot = recovered.start()
            self.assertEqual(snapshot["state"], "ready")
            self.assertTrue(
                any(
                    item["signal"] == "operator-restart-recovery"
                    for item in snapshot["flow"]["events"]
                )
            )
        finally:
            recovered.shutdown(wait=True, cancel_pending=True)
            temporary.cleanup()

    def test_failed_request_can_retry_without_persisting_raw_error(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="atom-operator-retry-")
        root = Path(temporary.name)
        session = _FakeSession()
        session.gate.set()
        session.fail_next = True
        operator = AtomHarnessOperator(session, state_root=root)
        with patch(
            "atom_harness_operator.verify_committed_run",
            side_effect=_verified_transaction,
        ):
            try:
                operator.start()
                failed = operator.submit("retry after provider failure")
                terminal = operator.wait_for_terminal(failed["request_id"])
                self.assertEqual(terminal["status"], "failed")
                self.assertEqual(terminal["error"]["type"], "RuntimeError")
                retry = operator.retry(failed["request_id"])
                self.assertEqual(
                    operator.wait_for_terminal(retry["request_id"])["status"],
                    "completed",
                )
                journal_text = operator.journal_path.read_text(encoding="utf-8")
                self.assertNotIn("private detail", journal_text)
            finally:
                operator.shutdown(wait=True, cancel_pending=True)
                temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
