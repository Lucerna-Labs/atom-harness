"""Durable interactive operator host for the resident Atom harness."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash
from atom_harness_session import AtomHarnessSession
from atom_llm_protocol import CancellationToken, ProviderCancelledError
from atom_run_transaction import verify_committed_run
from atom_tool_fabric import PermissionedToolFabric


ATOM_HARNESS_OPERATOR_RUNTIME = "atom-language-harness-operator-v5"
ATOM_HARNESS_OPERATOR_JOURNAL_RUNTIME = "atom-harness-operator-journal-v1"
ATOM_HARNESS_OPERATOR_FLOW_RUNTIME = "atom-harness-operator-spiderweb-flow-v2"
MAX_OPERATOR_QUESTION_CHARS = 4096
MAX_OPERATOR_HISTORY = 1000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            if stream.write(data) != len(data):
                raise OSError("operator journal write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("operator journal is not an object")
    return payload


def _terminal(status: str) -> bool:
    return status in {"completed", "cancelled", "failed", "interrupted"}


class OperatorCapacityError(RuntimeError):
    """The operator queue has reached its declared bound."""


class OperatorStateError(RuntimeError):
    """The requested control is invalid for the current operator state."""


class AtomHarnessOperator:
    """Keep the model, knowledge graph, queue, and session journal resident."""

    def __init__(
        self,
        session: AtomHarnessSession,
        *,
        state_root: Path,
        max_queue_depth: int = 8,
        tool_fabric: PermissionedToolFabric | None = None,
    ) -> None:
        if not 1 <= int(max_queue_depth) <= 256:
            raise ValueError("operator queue depth must be between one and 256")
        self.session = session
        self.state_root = Path(state_root).resolve()
        self.runs_root = self.state_root / "runs"
        self.journal_path = self.state_root / "atom_harness_operator_journal.json"
        self.max_queue_depth = int(max_queue_depth)
        self.tool_fabric = tool_fabric
        self._lock = threading.RLock()
        self._maintenance_lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self.max_queue_depth)
        self._tokens: dict[str, CancellationToken] = {}
        self._worker: threading.Thread | None = None
        self._state = "created"
        self._accepting = False
        self._active_request_id: str | None = None
        self._preload: Mapping[str, Any] | None = None
        self._restart_count = 0
        self._requests: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._created_at = _utc_now()
        self._updated_at = self._created_at
        self._load_journal()

    def _load_journal(self) -> None:
        if not self.journal_path.exists():
            return
        if not self.journal_path.is_file() or self.journal_path.is_symlink():
            raise ValueError("operator journal path is unsafe")
        payload = _read_object(self.journal_path)
        supplied_hash = payload.pop("journal_hash", None)
        if supplied_hash != canonical_hash(payload):
            raise ValueError("operator journal hash mismatch")
        if (
            payload.get("schema") != 1
            or payload.get("runtime") != ATOM_HARNESS_OPERATOR_JOURNAL_RUNTIME
        ):
            raise ValueError("operator journal contract is invalid")
        raw_requests = payload.get("requests")
        raw_order = payload.get("request_order")
        if not isinstance(raw_requests, dict) or not isinstance(raw_order, list):
            raise ValueError("operator journal request index is invalid")
        if set(raw_order) != set(raw_requests):
            raise ValueError("operator journal request order is invalid")
        if len(raw_order) > MAX_OPERATOR_HISTORY:
            raise ValueError("operator journal exceeds the history bound")
        self._created_at = str(payload.get("created_at", self._created_at))
        self._updated_at = str(payload.get("updated_at", self._updated_at))
        self._restart_count = int(payload.get("operator_restart_count", 0)) + 1
        self._requests = {
            str(identity): dict(record)
            for identity, record in raw_requests.items()
            if isinstance(record, Mapping)
        }
        if set(self._requests) != set(raw_requests):
            raise ValueError("operator journal contains an invalid request")
        self._order = [str(identity) for identity in raw_order]
        events = payload.get("flow_events", [])
        if not isinstance(events, list) or any(
            not isinstance(item, Mapping) for item in events
        ):
            raise ValueError("operator journal flow events are invalid")
        self._events = [dict(item) for item in events[-MAX_OPERATOR_HISTORY:]]
        recovered = 0
        for identity in self._order:
            request = self._requests[identity]
            if request.get("status") in {"queued", "running"}:
                request["status"] = "interrupted"
                request["finished_at"] = _utc_now()
                request["error"] = {
                    "kind": "operator-restart",
                    "type": "InterruptedError",
                    "message_sha256": canonical_hash(
                        {"request_id": identity, "reason": "operator-restart"}
                    ),
                }
                recovered += 1
        if recovered:
            self._events.append(
                self._flow_event(
                    signal="operator-restart-recovery",
                    request_id=None,
                    detail={"interrupted_request_count": recovered},
                )
            )
            self._persist_locked()

    def _flow_event(
        self,
        *,
        signal: str,
        request_id: str | None,
        detail: Mapping[str, Any],
    ) -> dict[str, Any]:
        core = {
            "schema": 1,
            "runtime": ATOM_HARNESS_OPERATOR_FLOW_RUNTIME,
            "kind": "vertical",
            "signal": signal,
            "request_id": request_id,
            "origin": "L0:operator-transport",
            "propagates_to": [
                "L1:operator-message",
                "L2:operator-flow",
                "L3:operator-orchestration",
            ],
            "observed_at": _utc_now(),
            "detail": dict(detail),
        }
        return {**core, "event_hash": canonical_hash(core)}

    def _journal_core_locked(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "runtime": ATOM_HARNESS_OPERATOR_JOURNAL_RUNTIME,
            "operator_runtime": ATOM_HARNESS_OPERATOR_RUNTIME,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "operator_restart_count": self._restart_count,
            "max_queue_depth": self.max_queue_depth,
            "request_order": list(self._order),
            "requests": {
                identity: dict(self._requests[identity]) for identity in self._order
            },
            "flow_events": list(self._events[-MAX_OPERATOR_HISTORY:]),
            "atom_store_mutation_allowed": False,
            "secrets_persisted": False,
        }

    def _persist_locked(self) -> None:
        self._updated_at = _utc_now()
        core = self._journal_core_locked()
        _atomic_json(
            self.journal_path,
            {**core, "journal_hash": canonical_hash(core)},
        )

    def start(self) -> Mapping[str, Any]:
        """Preload the runtime fully before opening the operator on-ramp."""

        with self._lock:
            if self._state == "ready":
                return self.snapshot()
            if self._state != "created":
                raise OperatorStateError(
                    f"operator cannot start from state {self._state}"
                )
            self._state = "preloading"
            self._persist_locked()
        try:
            preload = self.session.preload_runtime()
            hands_preload = (
                self.tool_fabric.start() if self.tool_fabric is not None else None
            )
        except BaseException:
            if self.tool_fabric is not None:
                try:
                    self.tool_fabric.shutdown(wait=True, cancel_pending=True)
                except BaseException:
                    pass
            try:
                self.session.close()
            except BaseException:
                pass
            with self._lock:
                self._state = "failed"
                self._accepting = False
                self._persist_locked()
            raise
        with self._lock:
            self._preload = dict(preload)
            self._state = "ready"
            self._accepting = True
            self._events.append(
                self._flow_event(
                    signal="resident-operator-lane-ready",
                    request_id=None,
                    detail={
                        "knowledge_hash": preload["knowledge"]["knowledge_hash"],
                        "provider_preload_hash": preload["providers"]["preload_hash"],
                        "permissioned_hands_ready": hands_preload is not None,
                    },
                )
            )
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="atom-harness-operator-worker",
                daemon=False,
            )
            self._worker.start()
            self._persist_locked()
            return self.snapshot()

    def submit(
        self,
        question: str,
        *,
        parent_request_id: str | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        normalized = str(question).strip()
        if (
            not normalized
            or "\x00" in normalized
            or len(normalized) > MAX_OPERATOR_QUESTION_CHARS
        ):
            raise ValueError("operator question is invalid")
        with self._lock:
            if not self._accepting or self._state != "ready":
                raise OperatorStateError("operator is not accepting requests")
            if len(self._order) >= MAX_OPERATOR_HISTORY:
                raise OperatorCapacityError("operator history bound is reached")
            if self._queue.full():
                self._events.append(
                    self._flow_event(
                        signal="operator-queue-capacity",
                        request_id=None,
                        detail={"max_queue_depth": self.max_queue_depth},
                    )
                )
                self._persist_locked()
                raise OperatorCapacityError("operator queue is at capacity")
            identity = uuid.uuid4().hex
            now = _utc_now()
            output_dir = self.runs_root / f"request-{identity}"
            record = {
                "schema": 1,
                "request_id": identity,
                "parent_request_id": parent_request_id,
                "attempt": int(attempt),
                "question": normalized,
                "status": "queued",
                "submitted_at": now,
                "started_at": None,
                "finished_at": None,
                "cancel_requested_at": None,
                "output_dir": str(output_dir),
                "artifact": None,
                "error": None,
            }
            token = CancellationToken()
            self._requests[identity] = record
            self._order.append(identity)
            self._tokens[identity] = token
            try:
                self._queue.put_nowait(identity)
            except queue.Full as error:
                self._requests.pop(identity, None)
                self._order.remove(identity)
                self._tokens.pop(identity, None)
                raise OperatorCapacityError("operator queue is at capacity") from error
            queued = self._queue.qsize()
            self._events.append(
                self._flow_event(
                    signal="operator-thread-formed",
                    request_id=identity,
                    detail={
                        "on_ramp": "OperatorQuestion",
                        "queued_requests": queued,
                        "flow_created": True,
                    },
                )
            )
            if queued > 1:
                self._events.append(
                    self._flow_event(
                        signal="operator-queue-backpressure",
                        request_id=identity,
                        detail={
                            "queued_requests": queued,
                            "max_queue_depth": self.max_queue_depth,
                        },
                    )
                )
            self._persist_locked()
            return dict(record)

    def _worker_loop(self) -> None:
        while True:
            identity = self._queue.get()
            try:
                if identity is None:
                    return
                self._run_request(identity)
            finally:
                self._queue.task_done()

    def _run_request(self, identity: str) -> None:
        with self._lock:
            record = self._requests[identity]
            if record["status"] == "cancelled":
                self._tokens.pop(identity, None)
                return
            record["status"] = "running"
            record["started_at"] = _utc_now()
            self._active_request_id = identity
            token = self._tokens[identity]
            self._events.append(
                self._flow_event(
                    signal="operator-thread-promoted",
                    request_id=identity,
                    detail={
                        "intersection": "resident-knowledge-language-intersection",
                        "emergent": True,
                    },
                )
            )
            self._persist_locked()
            question = str(record["question"])
            output_dir = Path(str(record["output_dir"]))
        try:
            artifact = self.session.answer(
                question,
                output_dir=output_dir,
                cancellation=token,
            )
            transaction = verify_committed_run(output_dir)
            response = artifact["response"]
            artifact_reference = {
                "artifact_hash": artifact["artifact_hash"],
                "transaction_id": transaction["transaction_id"],
                "request_id": artifact["request_id"],
                "output_dir": str(output_dir),
                "side_view": str(output_dir / "atom_harness_side_view.html"),
                "answerable": bool(response["answerable"]),
                "answer": str(response["answer"]),
                "citations": list(response["citations"]),
                "limitations": str(response["limitations"]),
                "knowledge_hash": artifact["knowledge"]["knowledge_hash"],
                "graph_knowledge_hash": artifact["knowledge"]["graph_knowledge_hash"],
                "total_ms": int(artifact["timings"]["total_ms"]),
                "provider_route_hashes": [
                    item["route_hash"] for item in artifact["provider_routes"]
                ],
            }
            with self._lock:
                record = self._requests[identity]
                record["status"] = "completed"
                record["finished_at"] = _utc_now()
                record["artifact"] = artifact_reference
                record["error"] = None
                self._events.append(
                    self._flow_event(
                        signal="operator-artifact-demoted",
                        request_id=identity,
                        detail={
                            "off_ramp": "CommittedAtomArtifact",
                            "artifact_hash": artifact["artifact_hash"],
                            "transaction_id": transaction["transaction_id"],
                        },
                    )
                )
                self._persist_locked()
        except ProviderCancelledError as error:
            self._finish_error(identity, "cancelled", error, "operator-cancel")
        except BaseException as error:
            self._finish_error(identity, "failed", error, "operator-failure")
        finally:
            with self._lock:
                if self._active_request_id == identity:
                    self._active_request_id = None
                self._tokens.pop(identity, None)
                self._persist_locked()

    def _finish_error(
        self,
        identity: str,
        status: str,
        error: BaseException,
        signal: str,
    ) -> None:
        safe_error = {
            "kind": signal,
            "type": type(error).__name__,
            "message_sha256": canonical_hash(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ),
        }
        with self._lock:
            record = self._requests[identity]
            record["status"] = status
            record["finished_at"] = _utc_now()
            record["error"] = safe_error
            self._events.append(
                self._flow_event(
                    signal=signal,
                    request_id=identity,
                    detail={"error_type": type(error).__name__},
                )
            )
            self._persist_locked()

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._requests.get(str(request_id))
            if record is None:
                raise KeyError("operator request does not exist")
            status = str(record["status"])
            if _terminal(status):
                return dict(record)
            record["cancel_requested_at"] = _utc_now()
            token = self._tokens.get(str(request_id))
            if token is not None:
                token.cancel("cancelled by Atom operator")
            if status == "queued":
                record["status"] = "cancelled"
                record["finished_at"] = _utc_now()
                record["error"] = {
                    "kind": "operator-cancel",
                    "type": "ProviderCancelledError",
                    "message_sha256": canonical_hash(
                        {"request_id": request_id, "reason": "operator-cancel"}
                    ),
                }
            self._events.append(
                self._flow_event(
                    signal="operator-cancellation",
                    request_id=str(request_id),
                    detail={"status_when_requested": status},
                )
            )
            self._persist_locked()
            return dict(record)

    def retry(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._requests.get(str(request_id))
            if record is None:
                raise KeyError("operator request does not exist")
            if str(record["status"]) not in {
                "cancelled",
                "failed",
                "interrupted",
            }:
                raise OperatorStateError(
                    "only a failed, cancelled, or interrupted request can be retried"
                )
            question = str(record["question"])
            attempt = int(record["attempt"]) + 1
        return self.submit(
            question,
            parent_request_id=str(request_id),
            attempt=attempt,
        )

    def restart_resident_lane(self) -> Mapping[str, Any]:
        """Restart and immediately rewarm resident providers while idle."""

        with self._maintenance_lock:
            with self._lock:
                if self._state != "ready" or not self._accepting:
                    raise OperatorStateError("operator is not ready")
                if self._active_request_id is not None or self._queue.qsize():
                    raise OperatorStateError(
                        "resident lane restart requires an idle operator"
                    )
                self._accepting = False
            try:
                restarted = []
                for provider in self.session.provider_fabric.providers:
                    terminate = getattr(provider, "terminate_lane_for_recovery", None)
                    if callable(terminate):
                        terminate("operator requested resident lane restart")
                        restarted.append(provider.capabilities().provider_id)
                preload = self.session.preload_runtime()
            finally:
                with self._lock:
                    if self._state == "ready":
                        self._accepting = True
            with self._lock:
                self._preload = dict(preload)
                self._events.append(
                    self._flow_event(
                        signal="operator-resident-lane-restarted",
                        request_id=None,
                        detail={"providers": restarted},
                    )
                )
                self._persist_locked()
                return {
                    "restarted_providers": restarted,
                    "preload": dict(preload),
                }

    def request_snapshot(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._requests.get(str(request_id))
            if record is None:
                raise KeyError("operator request does not exist")
            return dict(record)

    def submit_tool_task(
        self,
        task: str,
        *,
        parent_proposal_id: str | None = None,
    ) -> dict[str, Any]:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        with self._lock:
            if self._state != "ready" or not self._accepting:
                raise OperatorStateError("operator is not accepting tool tasks")
        return self.tool_fabric.submit_task(
            task,
            parent_proposal_id=parent_proposal_id,
        )

    def approve_tool(
        self,
        proposal_id: str,
        *,
        manifest_hash: str,
        decision_nonce: str,
    ) -> dict[str, Any]:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        return self.tool_fabric.approve(
            proposal_id,
            manifest_hash=manifest_hash,
            decision_nonce=decision_nonce,
        )

    def deny_tool(
        self,
        proposal_id: str,
        *,
        manifest_hash: str,
        decision_nonce: str,
    ) -> dict[str, Any]:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        return self.tool_fabric.deny(
            proposal_id,
            manifest_hash=manifest_hash,
            decision_nonce=decision_nonce,
        )

    def cancel_tool(self, proposal_id: str) -> dict[str, Any]:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        return self.tool_fabric.cancel(proposal_id)

    def tool_proposal_snapshot(self, proposal_id: str) -> dict[str, Any]:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        return self.tool_fabric.proposal_snapshot(proposal_id)

    def side_view_path(self, request_id: str) -> Path:
        record = self.request_snapshot(request_id)
        if record["status"] != "completed" or not isinstance(
            record.get("artifact"), Mapping
        ):
            raise OperatorStateError("operator request has no committed side view")
        output_dir = Path(str(record["output_dir"])).resolve()
        if self.runs_root.resolve() not in output_dir.parents:
            raise ValueError("operator artifact escaped its run root")
        verify_committed_run(output_dir)
        side_view = output_dir / "atom_harness_side_view.html"
        if not side_view.is_file() or side_view.is_symlink():
            raise ValueError("operator side view is unavailable")
        return side_view

    def tool_side_view_path(self, proposal_id: str) -> Path:
        if self.tool_fabric is None:
            raise OperatorStateError("permissioned hands are unavailable")
        return self.tool_fabric.side_view_path(proposal_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            recent = [dict(self._requests[identity]) for identity in self._order[-100:]]
            status_counts: dict[str, int] = {}
            for record in self._requests.values():
                status = str(record["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
            session = dict(self.session.manifest())
            hands = (
                self.tool_fabric.snapshot()
                if self.tool_fabric is not None
                else {
                    "schema": 1,
                    "enabled": False,
                    "state": "unavailable",
                    "permission_required_for_every_execution": True,
                }
            )
            core = {
                "schema": 1,
                "runtime": ATOM_HARNESS_OPERATOR_RUNTIME,
                "state": self._state,
                "accepting": self._accepting,
                "active_request_id": self._active_request_id,
                "queue_depth": self._queue.qsize(),
                "max_queue_depth": self.max_queue_depth,
                "request_count": len(self._order),
                "status_counts": status_counts,
                "requests": recent,
                "preload": dict(self._preload) if self._preload else None,
                "session": session,
                "hands": hands,
                "flow": {
                    "runtime": ATOM_HARNESS_OPERATOR_FLOW_RUNTIME,
                    "events": list(self._events[-100:]),
                    "typed_on_ramp": "OperatorQuestion",
                    "typed_off_ramp": "CommittedAtomArtifact",
                    "vertical_backpressure": True,
                    "flow_created_threads": True,
                    "flow_created_intersections": True,
                },
                "journal_path": str(self.journal_path),
                "atom_store_mutation_allowed": False,
                "secrets_persisted": False,
            }
            return {**core, "snapshot_hash": canonical_hash(core)}

    def wait_for_terminal(
        self,
        request_id: str,
        *,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            record = self.request_snapshot(request_id)
            if _terminal(str(record["status"])):
                return record
            time.sleep(0.025)
        raise TimeoutError("operator request did not reach a terminal state")

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = False,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self._state == "closed":
                return self.snapshot()
            self._accepting = False
            self._state = "closing"
            pending = [
                identity
                for identity in self._order
                if self._requests[identity]["status"] in {"queued", "running"}
            ]
        if cancel_pending:
            for identity in pending:
                self.cancel(identity)
        self._queue.put(None)
        worker = self._worker
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join()
        if worker is None or not worker.is_alive():
            if self.tool_fabric is not None:
                self.tool_fabric.shutdown(
                    wait=True,
                    cancel_pending=cancel_pending,
                )
            self.session.close()
            with self._lock:
                self._state = "closed"
                self._active_request_id = None
                self._persist_locked()
        return self.snapshot()

    def close(self) -> None:
        self.shutdown(wait=True, cancel_pending=True)

    def __enter__(self) -> AtomHarnessOperator:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
