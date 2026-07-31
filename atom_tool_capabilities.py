"""Real, permission-gated capability adapters for Atom Harness Phase 6."""

from __future__ import annotations

import difflib
import hashlib
import http.client
import ipaddress
import json
import os
import signal
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from atom_causal_world_schema import canonical_hash
from atom_llm_protocol import CancellationToken


ATOM_TOOL_CAPABILITY_RUNTIME = "atom-permissioned-capability-registry-v1"
MAX_TEXT_BYTES = 1024 * 1024
MAX_PROCESS_OUTPUT_BYTES = 512 * 1024
MAX_WEB_BYTES = 512 * 1024
MAX_WEB_ADDRESSES = 16
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    root = Path(path)
    if root.is_file():
        return _sha256(root)
    rows: list[dict[str, Any]] = []
    for child in sorted(root.rglob("*")):
        if child.is_symlink():
            raise ValueError("workspace tree contains a symbolic link")
        relative = child.relative_to(root).as_posix()
        if child.is_file():
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": child.stat().st_size,
                    "sha256": _sha256(child),
                }
            )
        elif child.is_dir():
            rows.append({"path": relative, "kind": "directory"})
    return canonical_hash({"tree": rows})


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = content.encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            if stream.write(encoded) != len(encoded):
                raise OSError("tool write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _strict_text(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or "\x00" in value or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    normalized = value if allow_empty else value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{label} is invalid")
    return normalized


def _strict_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _exact_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("tool arguments must be an object")
    if not required <= set(payload) or set(payload) - required - optional:
        raise ValueError("tool argument fields are invalid")


@dataclass(frozen=True)
class PreparedCapabilityAction:
    action_id: str
    capability: str
    arguments: Mapping[str, Any]
    rationale: str
    category: str
    risk: str
    effects: Mapping[str, Any]
    action_hash: str

    def manifest(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "capability": self.capability,
            "arguments": dict(self.arguments),
            "rationale": self.rationale,
            "category": self.category,
            "risk": self.risk,
            "effects": dict(self.effects),
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True)
class ToolCapability:
    name: str
    title: str
    category: str
    description: str
    risk: str
    arguments_schema: Mapping[str, Any]
    prepare: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any], str]]
    execute: Callable[[Mapping[str, Any], CancellationToken], Mapping[str, Any]]

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "risk": self.risk,
            "arguments_schema": dict(self.arguments_schema),
            "operator_permission_required": True,
        }


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connect to an approved address while retaining the reviewed Host value."""

    def __init__(
        self,
        host: str,
        *,
        pinned_address: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_address = pinned_address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Use an approved address and the reviewed hostname for TLS verification."""

    def __init__(
        self,
        host: str,
        *,
        pinned_address: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_address = pinned_address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except BaseException:
            raw_socket.close()
            raise


class _BoundedPipeCapture:
    """Drain a process pipe without retaining unbounded output."""

    def __init__(self, stream: Any, maximum: int) -> None:
        self.stream = stream
        self.maximum = maximum
        self.preview = bytearray()
        self.digest = hashlib.sha256()
        self.total = 0
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for block in iter(lambda: self.stream.read(64 * 1024), b""):
                self.digest.update(block)
                self.total += len(block)
                remaining = self.maximum - len(self.preview)
                if remaining > 0:
                    self.preview.extend(block[:remaining])
        except BaseException as error:
            self.error = error
        finally:
            self.stream.close()

    def result(self) -> tuple[bytes, str, int]:
        if self.error is not None:
            raise RuntimeError("process output capture failed") from self.error
        return bytes(self.preview), self.digest.hexdigest(), self.total


class AtomCapabilityRegistry:
    """Validate and execute exact capability manifests inside one workspace."""

    def __init__(self, *, workspace_root: Path, state_root: Path) -> None:
        workspace = Path(workspace_root).resolve()
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("tool workspace must be an existing regular directory")
        self.workspace_root = workspace
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root = self.state_root / "quarantine"
        self._capabilities = self._build_capabilities()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def manifest(self) -> dict[str, Any]:
        rows = [self._capabilities[name].manifest() for name in self.names]
        core = {
            "schema": 1,
            "runtime": ATOM_TOOL_CAPABILITY_RUNTIME,
            "workspace_root": str(self.workspace_root),
            "capabilities": rows,
            "capability_count": len(rows),
            "permission_required_for_every_execution": True,
            "shell_expansion": False,
            "tool_results_are_untrusted": True,
        }
        return {**core, "registry_hash": canonical_hash(core)}

    def planner_manifest(self) -> list[dict[str, Any]]:
        return [self._capabilities[name].manifest() for name in self.names]

    def _relative(self, value: Any, label: str, *, allow_root: bool = False) -> str:
        text = _strict_text(value, label, maximum=1024)
        relative = Path(text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} must stay inside the approved workspace")
        normalized = relative.as_posix()
        if normalized in {"", "."} and not allow_root:
            raise ValueError(f"{label} may not name the workspace root")
        target = (self.workspace_root / relative).resolve(strict=False)
        if target != self.workspace_root and self.workspace_root not in target.parents:
            raise ValueError(f"{label} escaped the approved workspace")
        return (
            "."
            if target == self.workspace_root
            else target.relative_to(self.workspace_root).as_posix()
        )

    def _path(
        self,
        relative: str,
        *,
        must_exist: bool | None = None,
        file_only: bool = False,
        directory_only: bool = False,
    ) -> Path:
        target = (self.workspace_root / Path(relative)).resolve(strict=False)
        if target != self.workspace_root and self.workspace_root not in target.parents:
            raise ValueError("tool path escaped the approved workspace")
        current = self.workspace_root
        for part in Path(relative).parts:
            if part == ".":
                continue
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError("tool path crosses a symbolic link")
        if must_exist is True and not target.exists():
            raise FileNotFoundError("approved tool target does not exist", target)
        if must_exist is False and target.exists():
            raise FileExistsError("approved tool target already exists", target)
        if file_only and (not target.is_file() or target.is_symlink()):
            raise ValueError("approved tool target is not a regular file")
        if directory_only and (not target.is_dir() or target.is_symlink()):
            raise ValueError("approved tool target is not a regular directory")
        return target

    def _json_schema(
        self,
        properties: Mapping[str, Any],
        required: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": dict(properties),
        }

    def _build_capabilities(self) -> dict[str, ToolCapability]:
        path_field = {"type": "string", "minLength": 1, "maxLength": 1024}
        text_field = {"type": "string", "maxLength": MAX_TEXT_BYTES}
        return {
            "workspace.list": ToolCapability(
                name="workspace.list",
                title="List workspace files",
                category="workspace-read",
                description="List a bounded directory tree without following links.",
                risk="low",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "recursive": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                    },
                    ("path", "recursive", "limit"),
                ),
                prepare=self._prepare_list,
                execute=self._execute_list,
            ),
            "workspace.read_text": ToolCapability(
                name="workspace.read_text",
                title="Read a text file",
                category="workspace-read",
                description="Read one bounded UTF-8 text file as untrusted data.",
                risk="medium",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_TEXT_BYTES,
                        },
                    },
                    ("path", "max_bytes"),
                ),
                prepare=self._prepare_read,
                execute=self._execute_read,
            ),
            "workspace.search_text": ToolCapability(
                name="workspace.search_text",
                title="Search workspace text",
                category="workspace-read",
                description="Search regular text files for one literal bounded query.",
                risk="medium",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "query": {"type": "string", "minLength": 1, "maxLength": 512},
                        "file_glob": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    ("path", "query", "file_glob", "limit"),
                ),
                prepare=self._prepare_search,
                execute=self._execute_search,
            ),
            "workspace.write_text": ToolCapability(
                name="workspace.write_text",
                title="Write a text file",
                category="workspace-write",
                description="Create or hash-guardedly replace one UTF-8 file atomically.",
                risk="high",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "content": text_field,
                        "mode": {"type": "string", "enum": ["create", "replace"]},
                        "expected_sha256": {"type": ["string", "null"]},
                    },
                    ("path", "content", "mode", "expected_sha256"),
                ),
                prepare=self._prepare_write,
                execute=self._execute_write,
            ),
            "workspace.patch_text": ToolCapability(
                name="workspace.patch_text",
                title="Patch a text file",
                category="workspace-write",
                description="Replace an exact text fragment under a required file hash.",
                risk="high",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "expected_sha256": {
                            "type": "string",
                            "minLength": 64,
                            "maxLength": 64,
                        },
                        "old_text": text_field,
                        "new_text": text_field,
                        "expected_occurrences": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                    (
                        "path",
                        "expected_sha256",
                        "old_text",
                        "new_text",
                        "expected_occurrences",
                    ),
                ),
                prepare=self._prepare_patch,
                execute=self._execute_patch,
            ),
            "workspace.make_directory": ToolCapability(
                name="workspace.make_directory",
                title="Create a directory",
                category="workspace-write",
                description="Create an exact directory path inside the workspace.",
                risk="medium",
                arguments_schema=self._json_schema(
                    {"path": path_field, "parents": {"type": "boolean"}},
                    ("path", "parents"),
                ),
                prepare=self._prepare_directory,
                execute=self._execute_directory,
            ),
            "workspace.move": ToolCapability(
                name="workspace.move",
                title="Move a workspace item",
                category="workspace-management",
                description="Move one hash-bound file or tree without overwriting.",
                risk="high",
                arguments_schema=self._json_schema(
                    {
                        "source": path_field,
                        "destination": path_field,
                        "expected_sha256": {
                            "type": "string",
                            "minLength": 64,
                            "maxLength": 64,
                        },
                    },
                    ("source", "destination", "expected_sha256"),
                ),
                prepare=self._prepare_move,
                execute=self._execute_move,
            ),
            "workspace.quarantine": ToolCapability(
                name="workspace.quarantine",
                title="Quarantine a workspace item",
                category="workspace-management",
                description="Reversibly move a hash-bound file or tree out of the workspace.",
                risk="critical",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "expected_sha256": {
                            "type": "string",
                            "minLength": 64,
                            "maxLength": 64,
                        },
                    },
                    ("path", "expected_sha256"),
                ),
                prepare=self._prepare_quarantine,
                execute=self._execute_quarantine,
            ),
            "process.run": ToolCapability(
                name="process.run",
                title="Run a development process",
                category="code-and-build",
                description="Run an exact executable and argument array without shell expansion.",
                risk="critical",
                arguments_schema=self._process_schema(),
                prepare=self._prepare_process,
                execute=self._execute_process,
            ),
            "simulation.run": ToolCapability(
                name="simulation.run",
                title="Run simulation cases",
                category="simulation",
                description="Run an exact program over bounded named cases and collect measurements.",
                risk="critical",
                arguments_schema=self._simulation_schema(),
                prepare=self._prepare_simulation,
                execute=self._execute_simulation,
            ),
            "document.create": ToolCapability(
                name="document.create",
                title="Create a document",
                category="documents",
                description="Create or hash-guardedly replace Markdown, text, HTML, or JSON.",
                risk="high",
                arguments_schema=self._json_schema(
                    {
                        "path": path_field,
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "text", "html", "json"],
                        },
                        "content": text_field,
                        "mode": {"type": "string", "enum": ["create", "replace"]},
                        "expected_sha256": {"type": ["string", "null"]},
                    },
                    ("path", "format", "content", "mode", "expected_sha256"),
                ),
                prepare=self._prepare_document,
                execute=self._execute_document,
            ),
            "web.fetch": ToolCapability(
                name="web.fetch",
                title="Fetch a public web resource",
                category="network-read",
                description="Fetch one exact public HTTP or HTTPS URL without redirects or credentials.",
                risk="high",
                arguments_schema=self._json_schema(
                    {
                        "url": {"type": "string", "minLength": 1, "maxLength": 2048},
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_WEB_BYTES,
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 60,
                        },
                    },
                    ("url", "max_bytes", "timeout_seconds"),
                ),
                prepare=self._prepare_web,
                execute=self._execute_web,
            ),
        }

    def _process_schema(self) -> dict[str, Any]:
        return self._json_schema(
            {
                "program": {"type": "string", "minLength": 1, "maxLength": 1024},
                "arguments": {
                    "type": "array",
                    "maxItems": 128,
                    "items": {"type": "string", "maxLength": 8192},
                },
                "cwd": {"type": "string", "minLength": 1, "maxLength": 1024},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                "stdin": {"type": "string", "maxLength": 65536},
            },
            ("program", "arguments", "cwd", "timeout_seconds", "stdin"),
        )

    def _simulation_schema(self) -> dict[str, Any]:
        return self._json_schema(
            {
                "program": {"type": "string", "minLength": 1, "maxLength": 1024},
                "base_arguments": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {"type": "string", "maxLength": 8192},
                },
                "cases": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "arguments"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "arguments": {
                                "type": "array",
                                "maxItems": 64,
                                "items": {"type": "string", "maxLength": 8192},
                            },
                        },
                    },
                },
                "cwd": {"type": "string", "minLength": 1, "maxLength": 1024},
                "timeout_seconds_per_case": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            (
                "program",
                "base_arguments",
                "cases",
                "cwd",
                "timeout_seconds_per_case",
            ),
        )

    def prepare_action(self, action: Mapping[str, Any]) -> PreparedCapabilityAction:
        if not isinstance(action, Mapping) or set(action) != {
            "action_id",
            "capability",
            "arguments",
            "rationale",
        }:
            raise ValueError("tool action is invalid")
        capability_name = str(action["capability"])
        capability = self._capabilities.get(capability_name)
        if capability is None:
            raise ValueError("tool capability is not registered")
        arguments, effects, dynamic_risk = capability.prepare(action["arguments"])
        risk = max(
            (capability.risk, dynamic_risk),
            key=lambda item: RISK_ORDER[item],
        )
        core = {
            "action_id": str(action["action_id"]),
            "capability": capability_name,
            "arguments": arguments,
            "rationale": str(action["rationale"]),
            "category": capability.category,
            "risk": risk,
            "effects": effects,
        }
        return PreparedCapabilityAction(**core, action_hash=canonical_hash(core))

    def execute_action(
        self,
        action: PreparedCapabilityAction,
        *,
        cancellation: CancellationToken,
    ) -> dict[str, Any]:
        cancellation.raise_if_cancelled()
        capability = self._capabilities.get(action.capability)
        if capability is None:
            raise ValueError("approved tool capability disappeared")
        started = time.perf_counter()
        result = capability.execute(action.arguments, cancellation)
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError("tool result exceeds the safe byte limit")
        succeeded = True
        if action.capability == "process.run":
            succeeded = result.get("exit_code") == 0
        elif action.capability == "simulation.run":
            succeeded = result.get("all_passed") is True
        core = {
            "schema": 1,
            "runtime": "atom-quarantined-tool-result-v1",
            "action_id": action.action_id,
            "action_hash": action.action_hash,
            "capability": action.capability,
            "status": "completed" if succeeded else "failed",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "trust": "untrusted-tool-output",
            "output": dict(result),
        }
        return {**core, "result_hash": canonical_hash(core)}

    def _prepare_list(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "recursive", "limit"})
        path = self._relative(raw["path"], "list path", allow_root=True)
        arguments = {
            "path": path,
            "recursive": _strict_bool(raw["recursive"], "list recursive"),
            "limit": _strict_int(raw["limit"], "list limit", 1, 2000),
        }
        return arguments, {"reads": [path]}, "low"

    def _execute_list(self, args: Mapping[str, Any], token: CancellationToken):
        root = self._path(str(args["path"]), must_exist=True, directory_only=True)
        iterator = root.rglob("*") if args["recursive"] else root.iterdir()
        rows = []
        for item in sorted(iterator):
            token.raise_if_cancelled()
            if len(rows) >= int(args["limit"]):
                break
            relative = item.relative_to(self.workspace_root).as_posix()
            kind = (
                "link"
                if item.is_symlink()
                else "directory"
                if item.is_dir()
                else "file"
            )
            row: dict[str, Any] = {"path": relative, "kind": kind}
            if kind == "file":
                row["bytes"] = item.stat().st_size
            rows.append(row)
        return {
            "path": str(args["path"]),
            "entries": rows,
            "entry_count": len(rows),
            "truncated": len(rows) >= int(args["limit"]),
        }

    def _prepare_read(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "max_bytes"})
        path = self._relative(raw["path"], "read path")
        arguments = {
            "path": path,
            "max_bytes": _strict_int(
                raw["max_bytes"], "read byte limit", 1, MAX_TEXT_BYTES
            ),
        }
        return (
            arguments,
            {"reads": [path], "content_enters_model_context": True},
            "medium",
        )

    def _execute_read(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        path = self._path(str(args["path"]), must_exist=True, file_only=True)
        maximum = int(args["max_bytes"])
        if path.stat().st_size > maximum:
            raise ValueError("approved text file exceeds the requested byte limit")
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("approved file is not UTF-8 text") from error
        return {
            "path": str(args["path"]),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content": content,
        }

    def _prepare_search(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "query", "file_glob", "limit"})
        path = self._relative(raw["path"], "search path", allow_root=True)
        query = _strict_text(raw["query"], "search query", maximum=512)
        file_glob = _strict_text(raw["file_glob"], "search glob", maximum=128)
        if ".." in Path(file_glob).parts or Path(file_glob).is_absolute():
            raise ValueError("search glob is invalid")
        arguments = {
            "path": path,
            "query": query,
            "file_glob": file_glob,
            "limit": _strict_int(raw["limit"], "search limit", 1, 500),
        }
        return (
            arguments,
            {"reads": [path], "content_enters_model_context": True},
            "medium",
        )

    def _execute_search(self, args: Mapping[str, Any], token: CancellationToken):
        root = self._path(str(args["path"]), must_exist=True, directory_only=True)
        query = str(args["query"])
        rows: list[dict[str, Any]] = []
        for path in sorted(root.rglob(str(args["file_glob"]))):
            token.raise_if_cancelled()
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_TEXT_BYTES
            ):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if query not in line:
                    continue
                rows.append(
                    {
                        "path": path.relative_to(self.workspace_root).as_posix(),
                        "line": line_number,
                        "text": line[:2048],
                    }
                )
                if len(rows) >= int(args["limit"]):
                    return {
                        "matches": rows,
                        "match_count": len(rows),
                        "truncated": True,
                    }
        return {"matches": rows, "match_count": len(rows), "truncated": False}

    def _normalize_hash(
        self, value: Any, label: str, *, nullable: bool = False
    ) -> str | None:
        if nullable and value is None:
            return None
        text = _strict_text(value, label, maximum=64)
        if len(text) != 64 or any(
            character not in "0123456789abcdef" for character in text
        ):
            raise ValueError(f"{label} is invalid")
        return text

    def _prepare_write(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "content", "mode", "expected_sha256"})
        path = self._relative(raw["path"], "write path")
        content = _strict_text(
            raw["content"], "write content", maximum=MAX_TEXT_BYTES, allow_empty=True
        )
        mode = _strict_text(raw["mode"], "write mode", maximum=16)
        if mode not in {"create", "replace"}:
            raise ValueError("write mode is invalid")
        expected = self._normalize_hash(
            raw["expected_sha256"], "expected file hash", nullable=True
        )
        if mode == "create" and expected is not None:
            raise ValueError("create mode may not carry an existing file hash")
        if mode == "replace" and expected is None:
            raise ValueError("replace mode requires the current file hash")
        arguments = {
            "path": path,
            "content": content,
            "mode": mode,
            "expected_sha256": expected,
        }
        return (
            arguments,
            {"writes": [path], "mode": mode, "bytes": len(content.encode("utf-8"))},
            "high",
        )

    def _execute_write(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        path = self._path(str(args["path"]))
        before = None
        if args["mode"] == "create":
            if path.exists():
                raise FileExistsError("create action refuses to overwrite", path)
        else:
            self._path(str(args["path"]), must_exist=True, file_only=True)
            before = _sha256(path)
            if before != args["expected_sha256"]:
                raise ValueError("file changed after permission was granted")
        _atomic_text(path, str(args["content"]))
        return {
            "path": str(args["path"]),
            "mode": str(args["mode"]),
            "sha256_before": before,
            "sha256_after": _sha256(path),
            "bytes": path.stat().st_size,
        }

    def _prepare_patch(self, raw: Mapping[str, Any]):
        _exact_fields(
            raw,
            required={
                "path",
                "expected_sha256",
                "old_text",
                "new_text",
                "expected_occurrences",
            },
        )
        path = self._relative(raw["path"], "patch path")
        old = _strict_text(raw["old_text"], "old patch text", maximum=MAX_TEXT_BYTES)
        new = _strict_text(
            raw["new_text"], "new patch text", maximum=MAX_TEXT_BYTES, allow_empty=True
        )
        expected = self._normalize_hash(raw["expected_sha256"], "expected file hash")
        occurrences = _strict_int(
            raw["expected_occurrences"], "patch occurrence count", 1, 1000
        )
        arguments = {
            "path": path,
            "expected_sha256": expected,
            "old_text": old,
            "new_text": new,
            "expected_occurrences": occurrences,
        }
        return (
            arguments,
            {"writes": [path], "mode": "hash-bound-patch", "occurrences": occurrences},
            "high",
        )

    def _execute_patch(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        path = self._path(str(args["path"]), must_exist=True, file_only=True)
        before_hash = _sha256(path)
        if before_hash != args["expected_sha256"]:
            raise ValueError("file changed after permission was granted")
        content = path.read_text(encoding="utf-8")
        count = content.count(str(args["old_text"]))
        if count != int(args["expected_occurrences"]):
            raise ValueError("patch occurrence count changed after permission")
        updated = content.replace(str(args["old_text"]), str(args["new_text"]))
        diff = "\n".join(
            difflib.unified_diff(
                content.splitlines(),
                updated.splitlines(),
                fromfile=str(args["path"]),
                tofile=str(args["path"]),
                lineterm="",
            )
        )
        _atomic_text(path, updated)
        return {
            "path": str(args["path"]),
            "sha256_before": before_hash,
            "sha256_after": _sha256(path),
            "occurrences": count,
            "diff": diff[: 128 * 1024],
            "diff_truncated": len(diff) > 128 * 1024,
        }

    def _prepare_directory(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "parents"})
        path = self._relative(raw["path"], "directory path")
        arguments = {
            "path": path,
            "parents": _strict_bool(raw["parents"], "directory parents"),
        }
        return arguments, {"creates_directory": [path]}, "medium"

    def _execute_directory(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        path = self._path(str(args["path"]), must_exist=False)
        path.mkdir(parents=bool(args["parents"]), exist_ok=False)
        return {"path": str(args["path"]), "created": True}

    def _prepare_move(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"source", "destination", "expected_sha256"})
        source = self._relative(raw["source"], "move source")
        destination = self._relative(raw["destination"], "move destination")
        if source == destination:
            raise ValueError("move source and destination are identical")
        expected = self._normalize_hash(raw["expected_sha256"], "expected item hash")
        arguments = {
            "source": source,
            "destination": destination,
            "expected_sha256": expected,
        }
        return (
            arguments,
            {"moves": [{"source": source, "destination": destination}]},
            "high",
        )

    def _execute_move(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        source = self._path(str(args["source"]), must_exist=True)
        destination = self._path(str(args["destination"]), must_exist=False)
        if _tree_sha256(source) != args["expected_sha256"]:
            raise ValueError("move source changed after permission was granted")
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                "move destination parent does not exist", destination.parent
            )
        os.replace(source, destination)
        return {
            "source": str(args["source"]),
            "destination": str(args["destination"]),
            "sha256": str(args["expected_sha256"]),
        }

    def _prepare_quarantine(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"path", "expected_sha256"})
        path = self._relative(raw["path"], "quarantine path")
        expected = self._normalize_hash(raw["expected_sha256"], "expected item hash")
        arguments = {"path": path, "expected_sha256": expected}
        return arguments, {"quarantines": [path], "recoverable": True}, "critical"

    def _execute_quarantine(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        source = self._path(str(args["path"]), must_exist=True)
        if _tree_sha256(source) != args["expected_sha256"]:
            raise ValueError("quarantine target changed after permission was granted")
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = self.quarantine_root / f"{uuid.uuid4().hex}-{source.name}"
        os.replace(source, destination)
        return {
            "path": str(args["path"]),
            "quarantined": True,
            "recovery_path": str(destination),
            "sha256": str(args["expected_sha256"]),
        }

    def _program(self, value: Any) -> tuple[str, str]:
        program = _strict_text(value, "process program", maximum=1024)
        candidate = Path(program)
        if candidate.is_absolute() or len(candidate.parts) > 1:
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=False)
            else:
                relative = self._relative(program, "process program")
                resolved = self._path(relative, must_exist=True, file_only=True)
            if not resolved.is_file():
                raise FileNotFoundError("process executable does not exist", resolved)
            return str(resolved), str(resolved)
        resolved_text = shutil.which(program)
        if not resolved_text:
            raise FileNotFoundError("process executable is unavailable", program)
        return program, str(Path(resolved_text).resolve())

    def _string_array(self, value: Any, label: str, maximum: int) -> list[str]:
        if not isinstance(value, list) or len(value) > maximum:
            raise ValueError(f"{label} is invalid")
        return [
            _strict_text(item, f"{label} item", maximum=8192, allow_empty=True)
            for item in value
        ]

    def _prepare_process(self, raw: Mapping[str, Any]):
        _exact_fields(
            raw, required={"program", "arguments", "cwd", "timeout_seconds", "stdin"}
        )
        program, resolved_program = self._program(raw["program"])
        resolved_program_sha256 = _sha256(Path(resolved_program))
        cwd = self._relative(raw["cwd"], "process cwd", allow_root=True)
        arguments = {
            "program": program,
            "resolved_program": resolved_program,
            "resolved_program_sha256": resolved_program_sha256,
            "arguments": self._string_array(raw["arguments"], "process arguments", 128),
            "cwd": cwd,
            "timeout_seconds": _strict_int(
                raw["timeout_seconds"], "process timeout", 1, 600
            ),
            "stdin": _strict_text(
                raw["stdin"], "process stdin", maximum=65536, allow_empty=True
            ),
        }
        shell_names = {
            "cmd",
            "cmd.exe",
            "powershell",
            "powershell.exe",
            "pwsh",
            "pwsh.exe",
            "sh",
            "bash",
        }
        dynamic_risk = (
            "critical" if Path(program).name.casefold() in shell_names else "high"
        )
        effects = {
            "executes": {
                "program": resolved_program,
                "program_sha256": resolved_program_sha256,
                "arguments": list(arguments["arguments"]),
                "cwd": cwd,
                "shell_expansion": False,
            },
            "may_modify_workspace": True,
            "environment_secrets_forwarded": False,
        }
        return arguments, effects, dynamic_risk

    def _sanitized_environment(self) -> dict[str, str]:
        names = {
            "PATH",
            "PATHEXT",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "LOCALAPPDATA",
            "APPDATA",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "ProgramData",
            "RUSTUP_HOME",
            "CARGO_HOME",
            "DOTNET_ROOT",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
        }
        environment = {
            name: value for name, value in os.environ.items() if name in names
        }
        environment["ATOM_TOOL_WORKSPACE"] = str(self.workspace_root)
        environment["ATOM_TOOL_PERMISSIONED"] = "1"
        return environment

    def _run_process(
        self,
        args: Mapping[str, Any],
        token: CancellationToken,
    ) -> dict[str, Any]:
        cwd = self._path(str(args["cwd"]), must_exist=True, directory_only=True)
        resolved_program = Path(str(args["resolved_program"]))
        if (
            not resolved_program.is_file()
            or resolved_program.is_symlink()
            or _sha256(resolved_program) != args["resolved_program_sha256"]
        ):
            raise ValueError("process executable changed after permission was granted")
        started = time.perf_counter()
        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            [
                str(resolved_program),
                *[str(item) for item in args["arguments"]],
            ],
            cwd=cwd,
            env=self._sanitized_environment(),
            stdin=subprocess.PIPE if args["stdin"] else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )
        if process.stdout is None or process.stderr is None:
            self._terminate_process_tree(process)
            raise RuntimeError("process output pipes were not created")
        stdout_capture = _BoundedPipeCapture(
            process.stdout,
            MAX_PROCESS_OUTPUT_BYTES,
        )
        stderr_capture = _BoundedPipeCapture(
            process.stderr,
            MAX_PROCESS_OUTPUT_BYTES,
        )
        capture_threads = [
            threading.Thread(
                target=stdout_capture.run,
                name=f"atom-tool-stdout-{process.pid}",
                daemon=True,
            ),
            threading.Thread(
                target=stderr_capture.run,
                name=f"atom-tool-stderr-{process.pid}",
                daemon=True,
            ),
        ]
        for thread in capture_threads:
            thread.start()

        stdin_thread: threading.Thread | None = None
        if args["stdin"] and process.stdin is not None:
            stdin_bytes = str(args["stdin"]).encode("utf-8")

            def write_stdin() -> None:
                try:
                    if process.stdin is not None:
                        process.stdin.write(stdin_bytes)
                        process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    if process.stdin is not None:
                        process.stdin.close()

            stdin_thread = threading.Thread(
                target=write_stdin,
                name=f"atom-tool-stdin-{process.pid}",
                daemon=True,
            )
            stdin_thread.start()

        deadline = time.monotonic() + int(args["timeout_seconds"])
        try:
            while process.poll() is None:
                token.raise_if_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("approved process exceeded its time limit")
                time.sleep(0.025)
        except BaseException:
            self._terminate_process_tree(process)
            raise
        finally:
            if stdin_thread is not None:
                stdin_thread.join(timeout=2)
            for thread in capture_threads:
                thread.join(timeout=5)
        if any(thread.is_alive() for thread in capture_threads):
            self._terminate_process_tree(process)
            raise RuntimeError("process output pipe remained open after process exit")
        stdout_preview, stdout_sha256, stdout_bytes = stdout_capture.result()
        stderr_preview, stderr_sha256, stderr_bytes = stderr_capture.result()
        return {
            "program": str(resolved_program),
            "program_sha256": str(args["resolved_program_sha256"]),
            "arguments": list(args["arguments"]),
            "cwd": str(args["cwd"]),
            "exit_code": int(process.returncode),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "stdout": stdout_preview.decode("utf-8", errors="replace"),
            "stderr": stderr_preview.decode("utf-8", errors="replace"),
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
            "stdout_sha256": stdout_sha256,
            "stderr_sha256": stderr_sha256,
            "stdout_truncated": stdout_bytes > len(stdout_preview),
            "stderr_truncated": stderr_bytes > len(stderr_preview),
            "shell_expansion": False,
            "environment_secrets_forwarded": False,
        }

    def _terminate_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            taskkill = system_root / "System32" / "taskkill.exe"
            if taskkill.is_file():
                try:
                    subprocess.run(
                        [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=self._sanitized_environment(),
                        timeout=5,
                        check=False,
                        shell=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def _execute_process(self, args: Mapping[str, Any], token: CancellationToken):
        return self._run_process(args, token)

    def _prepare_simulation(self, raw: Mapping[str, Any]):
        _exact_fields(
            raw,
            required={
                "program",
                "base_arguments",
                "cases",
                "cwd",
                "timeout_seconds_per_case",
            },
        )
        program, resolved_program = self._program(raw["program"])
        resolved_program_sha256 = _sha256(Path(resolved_program))
        cwd = self._relative(raw["cwd"], "simulation cwd", allow_root=True)
        base = self._string_array(
            raw["base_arguments"], "simulation base arguments", 64
        )
        raw_cases = raw["cases"]
        if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= 16:
            raise ValueError("simulation cases are invalid")
        cases = []
        seen = set()
        for raw_case in raw_cases:
            _exact_fields(raw_case, required={"name", "arguments"})
            name = _strict_text(raw_case["name"], "simulation case name", maximum=128)
            if name in seen:
                raise ValueError("simulation case names must be unique")
            seen.add(name)
            cases.append(
                {
                    "name": name,
                    "arguments": self._string_array(
                        raw_case["arguments"], "simulation case arguments", 64
                    ),
                }
            )
        timeout = _strict_int(
            raw["timeout_seconds_per_case"], "simulation timeout", 1, 300
        )
        arguments = {
            "program": program,
            "resolved_program": resolved_program,
            "resolved_program_sha256": resolved_program_sha256,
            "base_arguments": base,
            "cases": cases,
            "cwd": cwd,
            "timeout_seconds_per_case": timeout,
        }
        return (
            arguments,
            {
                "executes_simulation_cases": [item["name"] for item in cases],
                "program": resolved_program,
                "program_sha256": resolved_program_sha256,
                "cwd": cwd,
                "may_modify_workspace": True,
            },
            "critical",
        )

    def _execute_simulation(self, args: Mapping[str, Any], token: CancellationToken):
        results = []
        for case in args["cases"]:
            token.raise_if_cancelled()
            process_args = {
                "resolved_program": args["resolved_program"],
                "resolved_program_sha256": args["resolved_program_sha256"],
                "arguments": [*args["base_arguments"], *case["arguments"]],
                "cwd": args["cwd"],
                "timeout_seconds": args["timeout_seconds_per_case"],
                "stdin": "",
            }
            result = self._run_process(process_args, token)
            results.append({"name": case["name"], **result})
            if result["exit_code"] != 0:
                break
        return {
            "cases": results,
            "case_count": len(results),
            "all_passed": len(results) == len(args["cases"])
            and all(item["exit_code"] == 0 for item in results),
        }

    def _prepare_document(self, raw: Mapping[str, Any]):
        _exact_fields(
            raw, required={"path", "format", "content", "mode", "expected_sha256"}
        )
        path = self._relative(raw["path"], "document path")
        format_name = _strict_text(raw["format"], "document format", maximum=16)
        extensions = {
            "markdown": {".md"},
            "text": {".txt"},
            "html": {".html", ".htm"},
            "json": {".json"},
        }
        if (
            format_name not in extensions
            or Path(path).suffix.casefold() not in extensions[format_name]
        ):
            raise ValueError("document extension does not match its format")
        content = _strict_text(
            raw["content"], "document content", maximum=MAX_TEXT_BYTES, allow_empty=True
        )
        if format_name == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as error:
                raise ValueError("document JSON content is invalid") from error
        write_args, effects, risk = self._prepare_write(
            {
                "path": path,
                "content": content,
                "mode": raw["mode"],
                "expected_sha256": raw["expected_sha256"],
            }
        )
        return (
            {**write_args, "format": format_name},
            {**effects, "document_format": format_name},
            risk,
        )

    def _execute_document(self, args: Mapping[str, Any], token: CancellationToken):
        result = self._execute_write(args, token)
        return {**result, "format": str(args["format"]), "document_created": True}

    def _validated_public_url(self, value: Any) -> tuple[str, str, list[str]]:
        url = _strict_text(value, "web URL", maximum=2048)
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.port not in {None, 80, 443}
        ):
            raise ValueError("web URL is outside the public fetch policy")
        try:
            addresses = sorted(
                {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        parsed.hostname,
                        parsed.port or (443 if parsed.scheme == "https" else 80),
                        type=socket.SOCK_STREAM,
                    )
                }
            )
        except socket.gaierror as error:
            raise ValueError("web host could not be resolved") from error
        if not addresses:
            raise ValueError("web host resolved to no addresses")
        if len(addresses) > MAX_WEB_ADDRESSES:
            raise ValueError("web host resolved to too many addresses")
        for address in addresses:
            parsed_ip = ipaddress.ip_address(address)
            if not parsed_ip.is_global:
                raise ValueError("web URL resolves to a non-public address")
        canonical = urllib.parse.urlunsplit(parsed)
        return canonical, str(parsed.hostname), addresses

    def _prepare_web(self, raw: Mapping[str, Any]):
        _exact_fields(raw, required={"url", "max_bytes", "timeout_seconds"})
        url, host, addresses = self._validated_public_url(raw["url"])
        arguments = {
            "url": url,
            "host": host,
            "resolved_addresses": addresses,
            "max_bytes": _strict_int(
                raw["max_bytes"], "web byte limit", 1, MAX_WEB_BYTES
            ),
            "timeout_seconds": _strict_int(
                raw["timeout_seconds"], "web timeout", 1, 60
            ),
        }
        return (
            arguments,
            {
                "network_read": url,
                "permission_bound_addresses": addresses,
                "redirects_followed": False,
                "credentials_sent": False,
                "content_enters_model_context": True,
            },
            "high",
        )

    def _connect_pinned_web(
        self,
        *,
        scheme: str,
        host: str,
        port: int,
        addresses: Sequence[str],
        deadline: float,
    ) -> tuple[http.client.HTTPConnection, str]:
        last_error: BaseException | None = None
        connection_type = (
            _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
        )
        for address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            connection = connection_type(
                host,
                pinned_address=address,
                port=port,
                timeout=remaining,
            )
            try:
                connection.connect()
                return connection, address
            except (OSError, http.client.HTTPException) as error:
                last_error = error
                connection.close()
        raise ValueError(
            "web connection to the approved address set failed"
        ) from last_error

    def _execute_web(self, args: Mapping[str, Any], token: CancellationToken):
        token.raise_if_cancelled()
        url, host, addresses = self._validated_public_url(args["url"])
        if host != args["host"] or addresses != args["resolved_addresses"]:
            raise ValueError("web destination changed after permission was granted")
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        deadline = time.monotonic() + int(args["timeout_seconds"])
        connection, connected_address = self._connect_pinned_web(
            scheme=parsed.scheme,
            host=host,
            port=port,
            addresses=addresses,
            deadline=deadline,
        )
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": (
                        "text/plain, text/html, application/json;q=0.9, */*;q=0.1"
                    ),
                    "User-Agent": "AtomHarnessPermissionedHands/1",
                },
            )
            response = connection.getresponse()
            status = int(response.status)
            if 300 <= status < 400:
                raise ValueError("web redirect requires a new exact permission")
            if status >= 400:
                raise ValueError(f"web fetch returned HTTP {status}")
            maximum = int(args["max_bytes"])
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise ValueError(
                        "web response has an invalid content length"
                    ) from error
                if declared_length < 0 or declared_length > maximum:
                    raise ValueError("web response exceeds the approved byte limit")
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                token.raise_if_cancelled()
                time_remaining = deadline - time.monotonic()
                if time_remaining <= 0:
                    raise TimeoutError("web fetch exceeded the approved time limit")
                if connection.sock is None:
                    raise ValueError(
                        "web connection closed before the response completed"
                    )
                connection.sock.settimeout(time_remaining)
                chunk = response.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > maximum:
                raise ValueError("web response exceeds the approved byte limit")
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            content = raw.decode(charset, errors="replace")
            return {
                "url": url,
                "status": status,
                "content_type": content_type,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": content,
                "connected_address": connected_address,
                "permission_bound_addresses": list(addresses),
                "redirects_followed": False,
                "credentials_sent": False,
            }
        except (OSError, http.client.HTTPException) as error:
            raise ValueError(
                "web fetch failed after connecting to an approved address"
            ) from error
        finally:
            connection.close()
