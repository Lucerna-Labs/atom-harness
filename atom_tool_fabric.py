"""Permissioned Spiderweb action fabric for Atom Harness Phase 6."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from atom_causal_world_schema import canonical_hash
from atom_harness_knowledge import HarnessKnowledge
from atom_llm_protocol import (
    CancellationToken,
    JsonGenerationRequest,
    ProviderCancelledError,
)
from atom_run_transaction import (
    ATOM_RUN_TRANSACTION_RUNTIME,
    RunTransaction,
    recover_transactions,
    verify_committed_run,
)
from atom_tool_capabilities import (
    ATOM_TOOL_CAPABILITY_RUNTIME,
    RISK_ORDER,
    AtomCapabilityRegistry,
    PreparedCapabilityAction,
)
from atom_tool_protocol import (
    ATOM_PERMISSION_GRANT_RUNTIME,
    ATOM_TOOL_PLANNER_RUNTIME,
    ATOM_TOOL_RESULT_RUNTIME,
    canonicalize_tool_plan_candidate,
    MAX_TOOL_CONTEXT_BYTES,
    TOOL_PLANNER_SYSTEM_PROMPT,
    detect_injection_signals,
    normalize_tool_task,
    normalize_untrusted_context,
    planner_payload,
    tool_plan_schema,
    tool_task_sha256,
    validate_tool_plan,
)
from atom_tool_side_view import (
    ATOM_TOOL_ARTIFACT_BINDING,
    ATOM_TOOL_SIDE_VIEW_RUNTIME,
    render_atom_tool_artifact,
)


ATOM_PERMISSIONED_HANDS_RUNTIME = "atom-permissioned-hands-fabric-v1"
ATOM_PERMISSIONED_HANDS_JOURNAL_RUNTIME = "atom-permissioned-hands-journal-v1"
ATOM_PERMISSIONED_HANDS_FLOW_RUNTIME = "atom-permissioned-hands-spiderweb-flow-v1"
ATOM_TOOL_ARTIFACT_RUNTIME = "atom-permissioned-hands-artifact-v1"
ATOM_TOOL_WORKFLOW_RUNTIME = "atom-permissioned-hands-workflow-v1"
MAX_TOOL_HISTORY = 200
DEFAULT_PERMISSION_TTL_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    raw = (
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
            if stream.write(raw) != len(raw):
                raise OSError("permissioned hands journal write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _terminal(status: str) -> bool:
    return status in {
        "completed",
        "failed",
        "cancelled",
        "denied",
        "interrupted",
        "expired",
        "no-actions",
    }


class ToolCapacityError(RuntimeError):
    """The bounded planning or execution queue is full."""


class ToolStateError(RuntimeError):
    """The proposed control is invalid for the current action state."""


class ToolPermissionError(RuntimeError):
    """An exact one-time permission grant is missing or invalid."""


class PermissionedToolFabric:
    """Plan broadly, execute only after one exact trusted approval."""

    def __init__(
        self,
        *,
        provider_fabric: Any,
        knowledge_loader: Callable[[], HarnessKnowledge],
        workspace_root: Path,
        state_root: Path,
        max_queue_depth: int = 8,
        permission_ttl_seconds: int = DEFAULT_PERMISSION_TTL_SECONDS,
    ) -> None:
        if not 1 <= int(max_queue_depth) <= 64:
            raise ValueError("tool queue depth must be between one and 64")
        if not 30 <= int(permission_ttl_seconds) <= 3600:
            raise ValueError(
                "tool permission lifetime must be between 30 and 3600 seconds"
            )
        self.provider_fabric = provider_fabric
        self.knowledge_loader = knowledge_loader
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = self.state_root / "runs"
        self.journal_path = self.state_root / "permissioned_hands_journal.json"
        self.registry = AtomCapabilityRegistry(
            workspace_root=Path(workspace_root),
            state_root=self.state_root / "runtime",
        )
        self.max_queue_depth = int(max_queue_depth)
        self.permission_ttl_seconds = int(permission_ttl_seconds)
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue(
            maxsize=self.max_queue_depth
        )
        self._worker: threading.Thread | None = None
        self._state = "created"
        self._accepting = False
        self._active_proposal_id: str | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._tokens: dict[str, CancellationToken] = {}
        self._grants: dict[str, dict[str, str]] = {}
        self._grant_key = secrets.token_bytes(32)
        self._created_at = _utc_now()
        self._updated_at = self._created_at
        self._restart_count = 0
        self._load_journal()

    def _load_journal(self) -> None:
        if not self.journal_path.exists():
            return
        if not self.journal_path.is_file() or self.journal_path.is_symlink():
            raise ValueError("permissioned hands journal path is unsafe")
        payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("permissioned hands journal is not an object")
        supplied_hash = payload.pop("journal_hash", None)
        if supplied_hash != canonical_hash(payload):
            raise ValueError("permissioned hands journal hash mismatch")
        if (
            payload.get("schema") != 1
            or payload.get("runtime") != ATOM_PERMISSIONED_HANDS_JOURNAL_RUNTIME
        ):
            raise ValueError("permissioned hands journal contract is invalid")
        order = payload.get("proposal_order")
        records = payload.get("proposals")
        if not isinstance(order, list) or not isinstance(records, dict):
            raise ValueError("permissioned hands journal index is invalid")
        if set(order) != set(records) or len(order) > MAX_TOOL_HISTORY:
            raise ValueError("permissioned hands journal order is invalid")
        self._created_at = str(payload.get("created_at", self._created_at))
        self._updated_at = str(payload.get("updated_at", self._updated_at))
        self._restart_count = int(payload.get("restart_count", 0)) + 1
        self._order = [str(item) for item in order]
        self._records = {
            str(identity): dict(record)
            for identity, record in records.items()
            if isinstance(record, Mapping)
        }
        if set(self._records) != set(records):
            raise ValueError("permissioned hands journal contains an invalid proposal")
        raw_events = payload.get("flow_events", [])
        if not isinstance(raw_events, list) or any(
            not isinstance(item, Mapping) for item in raw_events
        ):
            raise ValueError("permissioned hands journal events are invalid")
        self._events = [dict(item) for item in raw_events[-MAX_TOOL_HISTORY:]]
        recovered = 0
        for identity in self._order:
            record = self._records[identity]
            if not _terminal(str(record.get("status"))):
                record["status"] = "interrupted"
                record["finished_at"] = _utc_now()
                record["error"] = {
                    "kind": "hands-restart",
                    "type": "InterruptedError",
                    "message_sha256": canonical_hash(
                        {"proposal_id": identity, "reason": "hands-restart"}
                    ),
                }
                recovered += 1
        if recovered:
            self._events.append(
                self._flow_event(
                    signal="hands-restart-recovery",
                    proposal_id=None,
                    detail={"interrupted_proposal_count": recovered},
                )
            )
            self._persist_locked()

    def _flow_event(
        self,
        *,
        signal: str,
        proposal_id: str | None,
        detail: Mapping[str, Any],
    ) -> dict[str, Any]:
        core = {
            "schema": 1,
            "runtime": ATOM_PERMISSIONED_HANDS_FLOW_RUNTIME,
            "kind": "vertical",
            "signal": signal,
            "proposal_id": proposal_id,
            "origin": "L0:permissioned-tool-transport",
            "propagates_to": [
                "L1:typed-capability-message",
                "L2:permissioned-action-flow",
                "L3:operator-authority",
            ],
            "observed_at": _utc_now(),
            "detail": dict(detail),
        }
        return {**core, "event_hash": canonical_hash(core)}

    def _journal_core_locked(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "runtime": ATOM_PERMISSIONED_HANDS_JOURNAL_RUNTIME,
            "hands_runtime": ATOM_PERMISSIONED_HANDS_RUNTIME,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "restart_count": self._restart_count,
            "workspace_root": str(self.registry.workspace_root),
            "max_queue_depth": self.max_queue_depth,
            "proposal_order": list(self._order),
            "proposals": {
                identity: dict(self._records[identity]) for identity in self._order
            },
            "flow_events": list(self._events[-MAX_TOOL_HISTORY:]),
            "permission_required_for_every_execution": True,
            "permission_grants_persisted": False,
            "provider_secrets_persisted": False,
            "tool_results_trusted_as_instructions": False,
        }

    def _persist_locked(self) -> None:
        self._updated_at = _utc_now()
        core = self._journal_core_locked()
        _atomic_json(
            self.journal_path,
            {**core, "journal_hash": canonical_hash(core)},
        )

    def start(self) -> Mapping[str, Any]:
        with self._lock:
            if self._state == "ready":
                return self.snapshot()
            if self._state != "created":
                raise ToolStateError(f"tool fabric cannot start from {self._state}")
            self._state = "ready"
            self._accepting = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="atom-permissioned-hands-worker",
                daemon=False,
            )
            self._worker.start()
            self._events.append(
                self._flow_event(
                    signal="hands-capability-highway-ready",
                    proposal_id=None,
                    detail={
                        "registry_hash": self.registry.manifest()["registry_hash"],
                        "capability_count": len(self.registry.names),
                        "preloaded": True,
                    },
                )
            )
            self._persist_locked()
            return self.snapshot()

    def _parent_context(self, parent_proposal_id: str | None) -> list[dict[str, Any]]:
        if parent_proposal_id is None:
            return []
        parent = self._records.get(str(parent_proposal_id))
        if parent is None:
            raise KeyError("parent tool proposal does not exist")
        if parent.get("status") != "completed" or not parent.get("results"):
            raise ToolStateError("parent tool proposal has no completed results")
        rows = []
        used = 0
        for result in parent["results"]:
            content = json.dumps(
                result,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            encoded = content.encode("utf-8")
            remaining = MAX_TOOL_CONTEXT_BYTES - used
            if remaining <= 0:
                break
            if len(encoded) > remaining:
                content = encoded[:remaining].decode("utf-8", errors="ignore")
                encoded = content.encode("utf-8")
            used += len(encoded)
            rows.append(
                {
                    "source": f"proposal:{parent_proposal_id}:{result['action_id']}",
                    "content": content,
                    "content_sha256": canonical_hash({"content": content}),
                }
            )
        return rows

    def submit_task(
        self,
        task: str,
        *,
        parent_proposal_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_tool_task(task)
        with self._lock:
            if self._state != "ready" or not self._accepting:
                raise ToolStateError("permissioned hands are not accepting tasks")
            if len(self._order) >= MAX_TOOL_HISTORY:
                raise ToolCapacityError("permissioned hands history bound is reached")
            if self._queue.full():
                self._events.append(
                    self._flow_event(
                        signal="hands-queue-backpressure",
                        proposal_id=None,
                        detail={"max_queue_depth": self.max_queue_depth},
                    )
                )
                self._persist_locked()
                raise ToolCapacityError("permissioned hands queue is at capacity")
            context_rows = self._parent_context(parent_proposal_id)
            context, context_hash = normalize_untrusted_context(context_rows)
            identity = uuid.uuid4().hex
            now = _utc_now()
            record = {
                "schema": 1,
                "proposal_id": identity,
                "parent_proposal_id": parent_proposal_id,
                "task": normalized,
                "task_sha256": tool_task_sha256(normalized),
                "status": "planning",
                "submitted_at": now,
                "planned_at": None,
                "decision_at": None,
                "started_at": None,
                "finished_at": None,
                "context": context,
                "context_sha256": context_hash,
                "injection_signals": detect_injection_signals(
                    normalized,
                    *[str(item["content"]) for item in context],
                ),
                "provider_route": None,
                "proposal": None,
                "execution_manifest": None,
                "decision_nonce": None,
                "permission_expires_at": None,
                "permission": None,
                "grant_consumed": False,
                "grant_consumed_at": None,
                "results": [],
                "output_dir": str(self.runs_root / f"proposal-{identity}"),
                "artifact": None,
                "error": None,
            }
            self._records[identity] = record
            self._order.append(identity)
            self._tokens[identity] = CancellationToken()
            try:
                self._queue.put_nowait(("plan", identity))
            except queue.Full as error:
                self._records.pop(identity, None)
                self._order.remove(identity)
                self._tokens.pop(identity, None)
                raise ToolCapacityError(
                    "permissioned hands queue is at capacity"
                ) from error
            self._events.append(
                self._flow_event(
                    signal="hands-planning-thread-formed",
                    proposal_id=identity,
                    detail={
                        "typed_on_ramp": "ToolTask",
                        "parent_proposal_id": parent_proposal_id,
                        "outside_influence_signal_count": len(
                            record["injection_signals"]
                        ),
                    },
                )
            )
            self._persist_locked()
            return self._summary_record(record)

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                operation, identity = job
                try:
                    if operation == "plan":
                        self._plan(identity)
                    elif operation == "execute":
                        self._execute(identity)
                    else:  # pragma: no cover
                        raise RuntimeError(
                            "permissioned hands worker received an invalid job"
                        )
                except BaseException as error:  # pragma: no cover - last-resort guard
                    with self._lock:
                        record = self._records.get(identity)
                        should_record = record is not None and not _terminal(
                            str(record.get("status"))
                        )
                    if should_record:
                        self._finish_error(
                            identity,
                            "failed",
                            error,
                            "hands-worker-failed",
                        )
            finally:
                self._queue.task_done()

    def _plan(self, identity: str) -> None:
        with self._lock:
            record = self._records[identity]
            if record["status"] == "cancelled":
                return
            token = self._tokens[identity]
            self._active_proposal_id = identity
            task = str(record["task"])
            context = list(record["context"])
            context_hash = str(record["context_sha256"])
        try:
            capability_manifest = self.registry.planner_manifest()
            request = JsonGenerationRequest(
                stage="tool.plan",
                system_prompt=TOOL_PLANNER_SYSTEM_PROMPT,
                payload=planner_payload(
                    task=task,
                    context=context,
                    context_sha256=context_hash,
                    capabilities=capability_manifest,
                ),
                schema=tool_plan_schema(
                    capability_manifest,
                    task_sha256=str(record["task_sha256"]),
                    context_sha256=context_hash,
                ),
                max_tokens=4096,
                validator=lambda payload: canonicalize_tool_plan_candidate(
                    payload,
                    task=task,
                    context_sha256=context_hash,
                    capabilities=capability_manifest,
                ),
                data_sensitivity="private-operator-intent",
            )
            completion = self.provider_fabric.generate_json(
                request,
                cancellation=token,
            )
            if {
                "planner_normalizations",
                "proposal_payload_hash",
            }.intersection(completion.payload):
                proposal = validate_tool_plan(
                    completion.payload,
                    task=task,
                    context_sha256=context_hash,
                    capability_names=self.registry.names,
                )
            else:
                proposal = canonicalize_tool_plan_candidate(
                    completion.payload,
                    task=task,
                    context_sha256=context_hash,
                    capabilities=capability_manifest,
                )
            prepared = [
                self.registry.prepare_action(action) for action in proposal["actions"]
            ]
            registry_manifest = self.registry.manifest()
            planned_at = datetime.now(timezone.utc)
            expires_at = planned_at + timedelta(seconds=self.permission_ttl_seconds)
            manifest_core = {
                "schema": 1,
                "runtime": "atom-exact-tool-execution-manifest-v1",
                "proposal_id": identity,
                "task_sha256": record["task_sha256"],
                "context_sha256": context_hash,
                "workspace_root": str(self.registry.workspace_root),
                "registry_hash": registry_manifest["registry_hash"],
                "actions": [item.manifest() for item in prepared],
                "planner_normalizations": list(proposal["planner_normalizations"]),
                "action_count": len(prepared),
                "maximum_risk": max(
                    (item.risk for item in prepared),
                    default="low",
                    key=lambda item: RISK_ORDER[item],
                ),
                "permission_required": True,
                "permission_scope": "exact-manifest-one-time",
                "planned_at": planned_at.isoformat(),
                "expires_at": expires_at.isoformat(),
            }
            execution_manifest = {
                **manifest_core,
                "manifest_hash": canonical_hash(manifest_core),
            }
            with self._lock:
                record = self._records[identity]
                record["planned_at"] = planned_at.isoformat()
                record["provider_route"] = dict(completion.route)
                record["proposal"] = proposal
                record["execution_manifest"] = execution_manifest
                record["decision_nonce"] = secrets.token_urlsafe(24)
                record["permission_expires_at"] = expires_at.isoformat()
                record["error"] = None
                if prepared:
                    record["status"] = "awaiting-permission"
                    self._events.append(
                        self._flow_event(
                            signal="hands-permission-requested",
                            proposal_id=identity,
                            detail={
                                "manifest_hash": execution_manifest["manifest_hash"],
                                "action_count": len(prepared),
                                "maximum_risk": execution_manifest["maximum_risk"],
                                "operator_permission_required": True,
                            },
                        )
                    )
                else:
                    record["status"] = "no-actions"
                    record["finished_at"] = _utc_now()
                    self._tokens.pop(identity, None)
                self._persist_locked()
        except ProviderCancelledError as error:
            self._finish_error(identity, "cancelled", error, "hands-planning-cancelled")
        except BaseException as error:
            self._finish_error(identity, "failed", error, "hands-planning-failed")
        finally:
            with self._lock:
                if self._active_proposal_id == identity:
                    self._active_proposal_id = None
                self._persist_locked()

    def _permission_material(
        self,
        *,
        proposal_id: str,
        manifest_hash: str,
        decision_nonce: str,
        expires_at: str,
        grant_id: str,
    ) -> bytes:
        payload = {
            "proposal_id": proposal_id,
            "manifest_hash": manifest_hash,
            "decision_nonce": decision_nonce,
            "expires_at": expires_at,
            "grant_id": grant_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    def approve(
        self,
        proposal_id: str,
        *,
        manifest_hash: str,
        decision_nonce: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(proposal_id))
            if record is None:
                raise KeyError("tool proposal does not exist")
            self._expire_record_locked(record)
            if record["status"] != "awaiting-permission":
                raise ToolPermissionError("tool proposal is not awaiting permission")
            manifest = record["execution_manifest"]
            if (
                not isinstance(manifest, Mapping)
                or not secrets.compare_digest(
                    str(manifest["manifest_hash"]), str(manifest_hash)
                )
                or not secrets.compare_digest(
                    str(record["decision_nonce"]), str(decision_nonce)
                )
            ):
                raise ToolPermissionError("tool permission binding is invalid")
            if self._queue.full():
                raise ToolCapacityError("permissioned hands queue is at capacity")
            grant_id = uuid.uuid4().hex
            material = self._permission_material(
                proposal_id=str(proposal_id),
                manifest_hash=str(manifest_hash),
                decision_nonce=str(decision_nonce),
                expires_at=str(record["permission_expires_at"]),
                grant_id=grant_id,
            )
            signature = hmac.new(self._grant_key, material, hashlib.sha256).hexdigest()
            grant_hash = canonical_hash(
                {
                    "grant_id": grant_id,
                    "manifest_hash": manifest_hash,
                    "signature": signature,
                }
            )
            self._grants[str(proposal_id)] = {
                "grant_id": grant_id,
                "signature": signature,
                "grant_hash": grant_hash,
            }
            permission_core = {
                "schema": 1,
                "runtime": ATOM_PERMISSION_GRANT_RUNTIME,
                "decision": "approved",
                "authority": "trusted-local-operator",
                "proposal_id": str(proposal_id),
                "manifest_hash": str(manifest_hash),
                "decision_nonce": str(decision_nonce),
                "approved_at": _utc_now(),
                "expires_at": str(record["permission_expires_at"]),
                "single_use": True,
                "grant_hash": grant_hash,
                "grant_secret_persisted": False,
            }
            record["permission"] = {
                **permission_core,
                "permission_hash": canonical_hash(permission_core),
            }
            record["decision_at"] = permission_core["approved_at"]
            record["status"] = "approved"
            self._queue.put_nowait(("execute", str(proposal_id)))
            self._events.append(
                self._flow_event(
                    signal="hands-permission-granted",
                    proposal_id=str(proposal_id),
                    detail={
                        "manifest_hash": str(manifest_hash),
                        "grant_hash": grant_hash,
                        "single_use": True,
                    },
                )
            )
            self._persist_locked()
            return self._summary_record(record)

    def deny(
        self,
        proposal_id: str,
        *,
        manifest_hash: str,
        decision_nonce: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(proposal_id))
            if record is None:
                raise KeyError("tool proposal does not exist")
            self._expire_record_locked(record)
            if record["status"] != "awaiting-permission":
                raise ToolPermissionError("tool proposal is not awaiting permission")
            manifest = record["execution_manifest"]
            if (
                not isinstance(manifest, Mapping)
                or not secrets.compare_digest(
                    str(manifest["manifest_hash"]), str(manifest_hash)
                )
                or not secrets.compare_digest(
                    str(record["decision_nonce"]), str(decision_nonce)
                )
            ):
                raise ToolPermissionError("tool denial binding is invalid")
            decision_at = _utc_now()
            permission_core = {
                "schema": 1,
                "runtime": ATOM_PERMISSION_GRANT_RUNTIME,
                "decision": "denied",
                "authority": "trusted-local-operator",
                "proposal_id": str(proposal_id),
                "manifest_hash": str(manifest_hash),
                "decision_nonce": str(decision_nonce),
                "decided_at": decision_at,
                "single_use": True,
                "grant_hash": None,
                "grant_secret_persisted": False,
            }
            record["permission"] = {
                **permission_core,
                "permission_hash": canonical_hash(permission_core),
            }
            record["decision_at"] = decision_at
            record["finished_at"] = decision_at
            record["status"] = "denied"
            self._tokens.pop(str(proposal_id), None)
            self._events.append(
                self._flow_event(
                    signal="hands-permission-denied",
                    proposal_id=str(proposal_id),
                    detail={"manifest_hash": str(manifest_hash)},
                )
            )
            self._persist_locked()
            return self._summary_record(record)

    def _consume_grant_locked(self, identity: str) -> dict[str, str]:
        record = self._records[identity]
        manifest = record.get("execution_manifest")
        grant = self._grants.pop(identity, None)
        permission = record.get("permission")
        if (
            not isinstance(manifest, Mapping)
            or not isinstance(permission, Mapping)
            or grant is None
        ):
            raise ToolPermissionError("one-time execution grant is absent")
        material = self._permission_material(
            proposal_id=identity,
            manifest_hash=str(manifest["manifest_hash"]),
            decision_nonce=str(record["decision_nonce"]),
            expires_at=str(record["permission_expires_at"]),
            grant_id=str(grant["grant_id"]),
        )
        expected = hmac.new(self._grant_key, material, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(grant["signature"])):
            raise ToolPermissionError("one-time execution grant signature is invalid")
        if permission.get("grant_hash") != grant.get("grant_hash"):
            raise ToolPermissionError("one-time execution grant hash is invalid")
        return grant

    def _reconstruct_actions(
        self,
        manifest: Mapping[str, Any],
    ) -> list[PreparedCapabilityAction]:
        registry_hash = self.registry.manifest()["registry_hash"]
        if manifest.get("registry_hash") != registry_hash:
            raise ToolPermissionError("capability registry changed after permission")
        prepared: list[PreparedCapabilityAction] = []
        for action in manifest["actions"]:
            if not isinstance(action, Mapping) or set(action) != {
                "action_id",
                "capability",
                "arguments",
                "rationale",
                "category",
                "risk",
                "effects",
                "action_hash",
            }:
                raise ToolPermissionError("approved action fields are invalid")
            if action["capability"] not in self.registry.names:
                raise ToolPermissionError("approved capability disappeared")
            action_core = {
                key: action[key]
                for key in (
                    "action_id",
                    "capability",
                    "arguments",
                    "rationale",
                    "category",
                    "risk",
                    "effects",
                )
            }
            if action.get("action_hash") != canonical_hash(action_core):
                raise ToolPermissionError("approved action changed before execution")
            prepared.append(
                PreparedCapabilityAction(
                    action_id=str(action["action_id"]),
                    capability=str(action["capability"]),
                    arguments=dict(action["arguments"]),
                    rationale=str(action["rationale"]),
                    category=str(action["category"]),
                    risk=str(action["risk"]),
                    effects=dict(action["effects"]),
                    action_hash=str(action["action_hash"]),
                )
            )
        core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if manifest.get("manifest_hash") != canonical_hash(core):
            raise ToolPermissionError("execution manifest hash is invalid")
        return prepared

    def _execute(self, identity: str) -> None:
        with self._lock:
            record = self._records[identity]
            if record["status"] == "cancelled":
                return
            self._expire_record_locked(record)
            if record["status"] != "approved":
                return
            grant = self._consume_grant_locked(identity)
            manifest = dict(record["execution_manifest"])
            actions = self._reconstruct_actions(manifest)
            record["status"] = "executing"
            record["started_at"] = _utc_now()
            record["grant_consumed_at"] = record["started_at"]
            record["grant_consumed"] = True
            token = self._tokens[identity]
            self._active_proposal_id = identity
            self._events.append(
                self._flow_event(
                    signal="hands-action-thread-promoted",
                    proposal_id=identity,
                    detail={
                        "typed_on_ramp": "ApprovedToolManifest",
                        "grant_hash": grant["grant_hash"],
                        "action_count": len(actions),
                    },
                )
            )
            self._persist_locked()

        results: list[dict[str, Any]] = []
        try:
            knowledge = self.knowledge_loader()
            memory_before = _sha256(knowledge.store_path)
            graph_before = str(knowledge.graph_manifest["knowledge_hash"])
            for action in actions:
                token.raise_if_cancelled()
                with self._lock:
                    self._events.append(
                        self._flow_event(
                            signal="hands-capability-on-ramp",
                            proposal_id=identity,
                            detail={
                                "action_id": action.action_id,
                                "capability": action.capability,
                                "action_hash": action.action_hash,
                            },
                        )
                    )
                    self._persist_locked()
                try:
                    result = self.registry.execute_action(
                        action,
                        cancellation=token,
                    )
                except BaseException as error:
                    failed_core = {
                        "schema": 1,
                        "runtime": ATOM_TOOL_RESULT_RUNTIME,
                        "action_id": action.action_id,
                        "action_hash": action.action_hash,
                        "capability": action.capability,
                        "status": "failed",
                        "trust": "untrusted-tool-output",
                        "error_type": type(error).__name__,
                        "error_sha256": canonical_hash(
                            {"type": type(error).__name__, "message": str(error)}
                        ),
                    }
                    results.append(
                        {**failed_core, "result_hash": canonical_hash(failed_core)}
                    )
                    break
                results.append(result)
                with self._lock:
                    self._events.append(
                        self._flow_event(
                            signal="hands-capability-off-ramp",
                            proposal_id=identity,
                            detail={
                                "action_id": action.action_id,
                                "result_hash": result["result_hash"],
                                "trust": "untrusted-tool-output",
                            },
                        )
                    )
                    self._persist_locked()
                if result["status"] != "completed":
                    break
            memory_after = _sha256(knowledge.store_path)
            graph_after = str(knowledge.graph.manifest()["knowledge_hash"])
            finished_at = _utc_now()
            artifact, transaction = self._publish_artifact(
                record=dict(record),
                manifest=manifest,
                results=results,
                knowledge=knowledge,
                memory_before=memory_before,
                memory_after=memory_after,
                graph_before=graph_before,
                graph_after=graph_after,
                finished_at=finished_at,
            )
            with self._lock:
                record = self._records[identity]
                record["status"] = "completed" if artifact["passed"] else "failed"
                record["finished_at"] = finished_at
                record["results"] = results
                record["artifact"] = {
                    "artifact_hash": artifact["artifact_hash"],
                    "transaction_id": transaction["transaction_id"],
                    "side_view": str(
                        Path(record["output_dir"]) / "atom_tool_side_view.html"
                    ),
                    "status": artifact["status"],
                    "passed": artifact["passed"],
                    "action_count": len(results),
                    "manifest_hash": manifest["manifest_hash"],
                }
                record["error"] = None
                self._events.append(
                    self._flow_event(
                        signal="hands-artifact-demoted",
                        proposal_id=identity,
                        detail={
                            "typed_off_ramp": "CommittedToolArtifact",
                            "artifact_hash": artifact["artifact_hash"],
                            "transaction_id": transaction["transaction_id"],
                            "passed": artifact["passed"],
                        },
                    )
                )
                self._persist_locked()
        except ProviderCancelledError as error:
            self._finish_error(
                identity, "cancelled", error, "hands-execution-cancelled"
            )
        except BaseException as error:
            self._finish_error(identity, "failed", error, "hands-execution-failed")
        finally:
            with self._lock:
                if self._active_proposal_id == identity:
                    self._active_proposal_id = None
                self._tokens.pop(identity, None)
                self._persist_locked()

    def _publish_artifact(
        self,
        *,
        record: Mapping[str, Any],
        manifest: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        knowledge: HarnessKnowledge,
        memory_before: str,
        memory_after: str,
        graph_before: str,
        graph_after: str,
        finished_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_dir = Path(str(record["output_dir"])).resolve()
        recovery_events = recover_transactions(final_dir.parent)
        with RunTransaction(final_dir) as transaction:
            transaction.snapshot_file(
                "runtime/atom_harness_knowledge.atomdb",
                knowledge.store_path,
            )
            checks = {
                "permission_approved": record["permission"]["decision"] == "approved",
                "permission_exact_manifest": record["permission"]["manifest_hash"]
                == manifest["manifest_hash"],
                "grant_consumed_once": record.get("grant_consumed") is True,
                "every_result_bound_to_approved_action": all(
                    result.get("action_hash") == action.get("action_hash")
                    for result, action in zip(
                        results, manifest["actions"], strict=False
                    )
                ),
                "execution_stopped_after_failure": not any(
                    result.get("status") != "completed" for result in results[:-1]
                ),
                "all_actions_completed": len(results) == manifest["action_count"]
                and all(result.get("status") == "completed" for result in results),
                "results_quarantined": all(
                    result.get("trust") == "untrusted-tool-output" for result in results
                ),
                "atom_memory_unchanged": memory_before == memory_after,
                "wiki_graph_unchanged": graph_before == graph_after,
            }
            passed = all(checks.values())
            status = "completed" if passed else "failed-closed"
            knowledge_manifest = knowledge.manifest()
            artifact_core = {
                "schema": 1,
                "runtime": ATOM_TOOL_ARTIFACT_RUNTIME,
                "proposal_id": record["proposal_id"],
                "parent_proposal_id": record["parent_proposal_id"],
                "task": record["task"],
                "task_sha256": record["task_sha256"],
                "status": status,
                "passed": passed,
                "submitted_at": record["submitted_at"],
                "planned_at": record["planned_at"],
                "started_at": record["started_at"],
                "finished_at": finished_at,
                "proposal": record["proposal"],
                "execution_manifest": dict(manifest),
                "permission": dict(record["permission"]),
                "grant_consumed_at": record["grant_consumed_at"],
                "results": [dict(item) for item in results],
                "injection_signals": list(record["injection_signals"]),
                "context_sha256": record["context_sha256"],
                "knowledge": knowledge_manifest,
                "memory": {
                    "store_sha256_before": memory_before,
                    "store_sha256_after": memory_after,
                    "unchanged": memory_before == memory_after,
                    "mutation_allowed": False,
                },
                "checks": checks,
                "transaction": {
                    "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
                    "transaction_id": transaction.transaction_id,
                    "atomic_publication": True,
                    "overwrite_allowed": False,
                    "recovery_event_count": len(recovery_events),
                },
                "side_view_contract": {
                    "runtime": ATOM_TOOL_SIDE_VIEW_RUNTIME,
                    "artifact_binding_marker": ATOM_TOOL_ARTIFACT_BINDING,
                    "placement": "side",
                    "user_visible": True,
                    "bound_to_real_output": True,
                },
            }
            artifact = {
                **artifact_core,
                "artifact_hash": canonical_hash(artifact_core),
            }
            workflow_core = {
                "schema": 1,
                "runtime": ATOM_TOOL_WORKFLOW_RUNTIME,
                "hands_runtime": ATOM_PERMISSIONED_HANDS_RUNTIME,
                "planner_runtime": ATOM_TOOL_PLANNER_RUNTIME,
                "capability_runtime": ATOM_TOOL_CAPABILITY_RUNTIME,
                "transaction_runtime": ATOM_RUN_TRANSACTION_RUNTIME,
                "transaction_id": transaction.transaction_id,
                "artifact_hash": artifact["artifact_hash"],
                "manifest_hash": manifest["manifest_hash"],
                "permission_hash": record["permission"]["permission_hash"],
                "result_hashes": [item["result_hash"] for item in results],
                "knowledge_hash": knowledge_manifest["knowledge_hash"],
                "graph_knowledge_hash": knowledge.graph_manifest["knowledge_hash"],
                "wiki_runtime": knowledge_manifest["wiki_runtime"],
                "rag_runtime": knowledge_manifest["rag_runtime"],
                "side_view_runtime": ATOM_TOOL_SIDE_VIEW_RUNTIME,
            }
            workflow = {
                **workflow_core,
                "workflow_hash": canonical_hash(workflow_core),
            }
            side_view = render_atom_tool_artifact(
                artifact,
                workflow,
                knowledge.graph_manifest,
            )
            transaction.write_json("atom_tool_artifact.json", artifact)
            transaction.write_json("atom_tool_workflow.json", workflow)
            transaction.write_json("atom_tool_permission.json", record["permission"])
            transaction.write_json("atom_tool_results.json", {"results": list(results)})
            transaction.write_json("atom_harness_knowledge.json", knowledge_manifest)
            transaction.write_json(
                "atom_harness_wiki_graph.json", knowledge.graph_manifest
            )
            transaction.write_text("atom_tool_side_view.html", side_view)
            transaction.seal(
                required_files=(
                    "atom_tool_artifact.json",
                    "atom_tool_workflow.json",
                    "atom_tool_permission.json",
                    "atom_tool_results.json",
                    "atom_harness_knowledge.json",
                    "atom_harness_wiki_graph.json",
                    "atom_tool_side_view.html",
                    "runtime/atom_harness_knowledge.atomdb",
                )
            )
            transaction.commit()
        return artifact, verify_committed_run(final_dir)

    def _finish_error(
        self,
        identity: str,
        status: str,
        error: BaseException,
        signal: str,
    ) -> None:
        route = getattr(error, "route", None)
        provider_failures: list[dict[str, Any]] = []
        route_hash = None
        if isinstance(route, Mapping):
            route_hash = route.get("route_hash")
            attempts = route.get("attempts", [])
            if isinstance(attempts, list):
                for attempt in attempts[:32]:
                    if not isinstance(attempt, Mapping):
                        continue
                    provider_failures.append(
                        {
                            "provider_id": str(attempt.get("provider_id", "unknown")),
                            "model": str(attempt.get("model", "unknown")),
                            "location": str(attempt.get("location", "unknown")),
                            "outcome": str(attempt.get("outcome", "unknown")),
                            "failure_kind": str(attempt.get("failure_kind", "unknown")),
                            "retryable": attempt.get("retryable") is True,
                            "error_sha256": attempt.get("error_sha256"),
                        }
                    )
        operator_message = (
            "The local planner could not produce a valid exact manifest. "
            "No tool action ran."
            if signal.startswith("hands-planning")
            else "The approved tool run stopped safely. No unapproved action ran."
        )
        safe_error = {
            "kind": signal,
            "type": type(error).__name__,
            "message_sha256": canonical_hash(
                {"type": type(error).__name__, "message": str(error)}
            ),
            "operator_message": operator_message,
            "provider_route_hash": route_hash,
            "provider_failures": provider_failures,
        }
        with self._lock:
            record = self._records[identity]
            record["status"] = status
            record["finished_at"] = _utc_now()
            record["error"] = safe_error
            self._grants.pop(identity, None)
            self._tokens.pop(identity, None)
            self._events.append(
                self._flow_event(
                    signal=signal,
                    proposal_id=identity,
                    detail={"error_type": type(error).__name__},
                )
            )
            self._persist_locked()

    def _expire_record_locked(self, record: dict[str, Any]) -> None:
        if record.get("status") not in {"awaiting-permission", "approved"}:
            return
        expires = datetime.fromisoformat(str(record["permission_expires_at"]))
        if datetime.now(timezone.utc) < expires:
            return
        record["status"] = "expired"
        record["finished_at"] = _utc_now()
        self._grants.pop(str(record["proposal_id"]), None)
        self._tokens.pop(str(record["proposal_id"]), None)
        self._events.append(
            self._flow_event(
                signal="hands-permission-expired",
                proposal_id=str(record["proposal_id"]),
                detail={"manifest_hash": record["execution_manifest"]["manifest_hash"]},
            )
        )
        self._persist_locked()

    def cancel(self, proposal_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(proposal_id))
            if record is None:
                raise KeyError("tool proposal does not exist")
            if _terminal(str(record["status"])):
                return self._summary_record(record)
            token = self._tokens.get(str(proposal_id))
            if token is not None:
                token.cancel("cancelled by Atom operator")
            if record["status"] in {"planning", "awaiting-permission", "approved"}:
                record["status"] = "cancelled"
                record["finished_at"] = _utc_now()
                self._grants.pop(str(proposal_id), None)
                self._tokens.pop(str(proposal_id), None)
            self._events.append(
                self._flow_event(
                    signal="hands-cancellation",
                    proposal_id=str(proposal_id),
                    detail={"status_when_requested": record["status"]},
                )
            )
            self._persist_locked()
            return self._summary_record(record)

    def _summary_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        proposal = record.get("proposal")
        manifest = record.get("execution_manifest")
        return {
            "schema": 1,
            "proposal_id": record["proposal_id"],
            "parent_proposal_id": record["parent_proposal_id"],
            "task": record["task"],
            "status": record["status"],
            "submitted_at": record["submitted_at"],
            "planned_at": record["planned_at"],
            "decision_at": record["decision_at"],
            "finished_at": record["finished_at"],
            "summary": proposal.get("summary")
            if isinstance(proposal, Mapping)
            else None,
            "completion_condition": (
                proposal.get("completion_condition")
                if isinstance(proposal, Mapping)
                else None
            ),
            "manifest_hash": (
                manifest.get("manifest_hash") if isinstance(manifest, Mapping) else None
            ),
            "action_count": (
                manifest.get("action_count") if isinstance(manifest, Mapping) else 0
            ),
            "maximum_risk": (
                manifest.get("maximum_risk") if isinstance(manifest, Mapping) else None
            ),
            "planner_normalizations": (
                list(manifest.get("planner_normalizations", []))
                if isinstance(manifest, Mapping)
                else []
            ),
            "actions": (
                list(manifest.get("actions", []))
                if isinstance(manifest, Mapping)
                and record["status"] in {"awaiting-permission", "approved", "executing"}
                else []
            ),
            "decision_nonce": record["decision_nonce"],
            "permission_expires_at": record["permission_expires_at"],
            "permission": record["permission"],
            "injection_signals": list(record["injection_signals"]),
            "artifact": record["artifact"],
            "error": record["error"],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            for identity in self._order:
                self._expire_record_locked(self._records[identity])
            recent = [
                self._summary_record(self._records[identity])
                for identity in self._order[-50:]
            ]
            counts: dict[str, int] = {}
            for record in self._records.values():
                status = str(record["status"])
                counts[status] = counts.get(status, 0) + 1
            core = {
                "schema": 1,
                "runtime": ATOM_PERMISSIONED_HANDS_RUNTIME,
                "state": self._state,
                "accepting": self._accepting,
                "active_proposal_id": self._active_proposal_id,
                "queue_depth": self._queue.qsize(),
                "max_queue_depth": self.max_queue_depth,
                "proposal_count": len(self._order),
                "status_counts": counts,
                "proposals": recent,
                "workspace_root": str(self.registry.workspace_root),
                "registry": self.registry.manifest(),
                "flow": {
                    "runtime": ATOM_PERMISSIONED_HANDS_FLOW_RUNTIME,
                    "events": list(self._events[-100:]),
                    "typed_on_ramps": ["ToolTask", "ApprovedToolManifest"],
                    "typed_off_ramp": "CommittedToolArtifact",
                    "threads_form_from_observed_tasks": True,
                    "vertical_backpressure": True,
                    "capability_lane_preloaded": True,
                },
                "permission_required_for_every_execution": True,
                "permission_grants_persisted": False,
                "tool_results_trusted_as_instructions": False,
            }
            return {**core, "snapshot_hash": canonical_hash(core)}

    def proposal_snapshot(self, proposal_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(proposal_id))
            if record is None:
                raise KeyError("tool proposal does not exist")
            self._expire_record_locked(record)
            return dict(record)

    def side_view_path(self, proposal_id: str) -> Path:
        record = self.proposal_snapshot(proposal_id)
        if not isinstance(record.get("artifact"), Mapping):
            raise ToolStateError("tool proposal has no committed side view")
        output_dir = Path(str(record["output_dir"])).resolve()
        if self.runs_root.resolve() not in output_dir.parents:
            raise ValueError("tool artifact escaped its run root")
        verify_committed_run(output_dir)
        side_view = output_dir / "atom_tool_side_view.html"
        if not side_view.is_file() or side_view.is_symlink():
            raise ValueError("tool side view is unavailable")
        return side_view

    def wait_for_status(
        self,
        proposal_id: str,
        statuses: Sequence[str],
        *,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        expected = frozenset(str(item) for item in statuses)
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            record = self.proposal_snapshot(proposal_id)
            if str(record["status"]) in expected:
                return record
            time.sleep(0.025)
        raise TimeoutError("tool proposal did not reach the requested state")

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = True,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self._state == "closed":
                return self.snapshot()
            self._accepting = False
            self._state = "closing"
            pending = [
                identity
                for identity in self._order
                if not _terminal(str(self._records[identity]["status"]))
            ]
        if cancel_pending:
            for identity in pending:
                self.cancel(identity)
        self._queue.put(None)
        worker = self._worker
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._lock:
            if worker is None or not worker.is_alive():
                self._state = "closed"
                self._active_proposal_id = None
                self._persist_locked()
        return self.snapshot()

    def close(self) -> None:
        self.shutdown(wait=True, cancel_pending=True)
