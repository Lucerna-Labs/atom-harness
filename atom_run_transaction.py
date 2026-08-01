"""Crash-safe, concurrent run publication for Atom Harness V2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import canonical_hash


ATOM_RUN_TRANSACTION_RUNTIME = "atom-run-transaction-v2"
ATOM_RUN_TRANSACTION_FILENAME = "atom_harness_transaction.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RunTransactionError(RuntimeError):
    """Base error for a run that cannot be published atomically."""


class RunLockedError(RunTransactionError):
    """Another process owns the exact output target."""


class RunIntegrityError(RunTransactionError):
    """A staged or committed run fails its file manifest."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            written = stream.write(data)
            if written != len(data):
                raise OSError("transaction file write was incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _safe_child(root: Path, candidate: Path) -> bool:
    resolved_root = Path(root).resolve()
    resolved_candidate = Path(candidate).resolve()
    return resolved_candidate.parent == resolved_root


def _is_link(path: Path) -> bool:
    path = Path(path)
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (callable(is_junction) and bool(is_junction()))


def bind_recorded_run_directory(
    runs_root: Path,
    recorded_output_dir: object,
    *,
    kind: str,
    identity: str,
) -> Path:
    """Bind a journal record to its deterministic run directory.

    Windows packaged launchers can virtualize LocalAppData writes. Resolving a
    recorded path through the filesystem can then expose the physical package
    path even though the runtime must continue opening the logical path. Keep
    the containment comparison lexical, reject links and junctions explicitly,
    and return the runtime-owned deterministic path instead of the journal
    string.
    """

    if kind not in {"request", "proposal"}:
        raise ValueError("run directory kind is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", str(identity)):
        raise ValueError("run directory identity is invalid")
    root = Path(runs_root)
    recorded = Path(str(recorded_output_dir))
    expected = root / f"{kind}-{identity}"
    if not recorded.is_absolute():
        raise ValueError("recorded run directory must be absolute")

    def logical(path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    if logical(recorded) != logical(expected):
        raise ValueError("recorded run directory escaped its run root")
    if _is_link(root) or _is_link(expected):
        raise ValueError("recorded run directory is an unsafe link")
    return expected


def _ensure_direct_directory(path: Path, parent: Path) -> None:
    path = Path(path)
    parent = Path(parent).resolve()
    if path.exists():
        if _is_link(path) or not path.is_dir() or path.resolve().parent != parent:
            raise RunIntegrityError("transaction control directory is unsafe")
        return
    path.mkdir(parents=False, exist_ok=False)
    if path.resolve().parent != parent:
        raise RunIntegrityError("transaction control directory escaped its parent")


def _valid_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    if "\\" in value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and value not in {".", ".."}
        and ".." not in path.parts
        and all(part not in {"", "."} for part in path.parts)
    )


def _transaction_core(
    *,
    transaction_id: str,
    target_name: str,
    state: str,
    created_at: str,
    sealed_at: str | None,
    pid: int,
    required_files: Sequence[str],
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": 1,
        "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
        "transaction_id": transaction_id,
        "target_name": target_name,
        "state": state,
        "created_at": created_at,
        "sealed_at": sealed_at,
        "pid": pid,
        "required_files": list(required_files),
        "files": [dict(item) for item in files],
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


def _validate_manifest(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    expected_state: str,
    expected_target_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise RunIntegrityError("transaction manifest is not an object")
    expected_fields = {
        "schema",
        "runtime",
        "transaction_id",
        "target_name",
        "state",
        "created_at",
        "sealed_at",
        "pid",
        "required_files",
        "files",
        "total_bytes",
        "manifest_hash",
    }
    if set(manifest) != expected_fields:
        raise RunIntegrityError("transaction manifest fields are invalid")
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise RunIntegrityError("transaction schema is invalid")
    if manifest["runtime"] != ATOM_RUN_TRANSACTION_RUNTIME:
        raise RunIntegrityError("transaction runtime is invalid")
    if not isinstance(manifest["transaction_id"], str) or not _SHA256_PATTERN.fullmatch(
        manifest["transaction_id"]
    ):
        raise RunIntegrityError("transaction identity is invalid")
    if manifest["state"] != expected_state:
        raise RunIntegrityError(
            f"transaction state is {manifest['state']}, expected {expected_state}"
        )
    if (
        not isinstance(manifest["created_at"], str)
        or not manifest["created_at"]
        or not isinstance(manifest["sealed_at"], str)
        or not manifest["sealed_at"]
    ):
        raise RunIntegrityError("transaction timestamps are invalid")
    if type(manifest["pid"]) is not int or manifest["pid"] <= 0:
        raise RunIntegrityError("transaction process identity is invalid")
    if type(manifest["total_bytes"]) is not int or manifest["total_bytes"] < 0:
        raise RunIntegrityError("transaction total byte count is invalid")
    if not isinstance(manifest["manifest_hash"], str) or not _SHA256_PATTERN.fullmatch(
        manifest["manifest_hash"]
    ):
        raise RunIntegrityError("transaction manifest hash is invalid")
    core = {key: manifest[key] for key in sorted(manifest) if key != "manifest_hash"}
    try:
        expected_hash = canonical_hash(core)
    except (TypeError, ValueError) as error:
        raise RunIntegrityError("transaction manifest is not canonical JSON") from error
    if manifest["manifest_hash"] != expected_hash:
        raise RunIntegrityError("transaction manifest hash mismatch")
    target_name = expected_target_name or Path(run_dir).name
    if (
        not isinstance(manifest["target_name"], str)
        or "/" in manifest["target_name"]
        or "\\" in manifest["target_name"]
        or Path(manifest["target_name"]).name != manifest["target_name"]
        or manifest["target_name"] != target_name
    ):
        raise RunIntegrityError("transaction target name mismatch")

    if not isinstance(manifest["files"], list):
        raise RunIntegrityError("transaction file list is invalid")
    listed: dict[str, Mapping[str, Any]] = {}
    listed_casefold: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise RunIntegrityError("transaction file entry is invalid")
        relative = item["path"]
        folded = relative.casefold()
        if (
            not _valid_relative_path(relative)
            or relative in listed
            or folded in listed_casefold
        ):
            raise RunIntegrityError("transaction file path is invalid or duplicated")
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise RunIntegrityError(f"transaction byte count is invalid: {relative}")
        if not isinstance(item["sha256"], str) or not _SHA256_PATTERN.fullmatch(
            item["sha256"]
        ):
            raise RunIntegrityError(f"transaction file hash is invalid: {relative}")
        listed[relative] = item
        listed_casefold.add(folded)
    run_root = Path(run_dir).resolve()
    actual_paths: set[str] = set()
    for path in Path(run_dir).rglob("*"):
        if _is_link(path):
            raise RunIntegrityError("transaction bundle contains a symbolic link")
        if not path.is_file() or path.name == ATOM_RUN_TRANSACTION_FILENAME:
            continue
        resolved = path.resolve()
        if run_root not in resolved.parents:
            raise RunIntegrityError("transaction file escaped the run directory")
        actual_paths.add(path.relative_to(run_dir).as_posix())
    if actual_paths != set(listed):
        raise RunIntegrityError("transaction file set differs from the manifest")
    for relative, item in listed.items():
        path = Path(run_dir) / Path(relative)
        if path.stat().st_size != item["bytes"]:
            raise RunIntegrityError(f"transaction byte count mismatch: {relative}")
        if _sha256(path) != item["sha256"]:
            raise RunIntegrityError(f"transaction file hash mismatch: {relative}")
    if (
        not isinstance(manifest["required_files"], list)
        or any(not _valid_relative_path(item) for item in manifest["required_files"])
        or len(set(manifest["required_files"])) != len(manifest["required_files"])
        or len({item.casefold() for item in manifest["required_files"]})
        != len(manifest["required_files"])
    ):
        raise RunIntegrityError("transaction required-file list is invalid")
    required = set(manifest["required_files"])
    if not required <= actual_paths:
        missing = sorted(required - actual_paths)
        raise RunIntegrityError(
            "transaction is missing required files: " + ", ".join(missing)
        )
    if sum(int(item["bytes"]) for item in listed.values()) != manifest["total_bytes"]:
        raise RunIntegrityError("transaction total byte count mismatch")
    return dict(manifest)


def verify_committed_run(run_dir: Path) -> dict[str, Any]:
    """Verify the exact atomic bundle visible to users."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / ATOM_RUN_TRANSACTION_FILENAME
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise RunIntegrityError("committed run or transaction manifest is absent")
    if _is_link(manifest_path):
        raise RunIntegrityError("transaction manifest must not be a symbolic link")
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise RunIntegrityError("transaction manifest exceeds the safe byte limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunIntegrityError("transaction manifest cannot be decoded") from error
    return _validate_manifest(run_dir, manifest, expected_state="committed")


class RunTransaction:
    """Stage an entire run and expose it with one atomic directory rename."""

    def __init__(self, final_dir: Path) -> None:
        self.final_dir = Path(final_dir).resolve()
        if self.final_dir.name.casefold() in {"", ".", "..", ".atom-harness-v2"}:
            raise ValueError("run output directory is invalid")
        self.parent = self.final_dir.parent
        self.control_dir = self.parent / ".atom-harness-v2"
        target_hash = hashlib.sha256(
            str(self.final_dir).casefold().encode("utf-8")
        ).hexdigest()[:24]
        self.lock_path = self.control_dir / "locks" / f"{target_hash}.lock"
        self.transaction_id = canonical_hash(
            {
                "target": str(self.final_dir),
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "nonce": uuid.uuid4().hex,
            }
        )
        self.staging_dir = (
            self.control_dir
            / "staging"
            / f"{self.final_dir.name}.{self.transaction_id}"
        )
        self.recovery_dir = self.control_dir / "recovery"
        self.created_at = _utc_now()
        self._lock_descriptor: int | None = None
        self._owns_lock = False
        self._sealed_manifest: dict[str, Any] | None = None
        self._published = False

    def begin(self) -> "RunTransaction":
        if self._owns_lock or self.staging_dir.exists():
            raise RunTransactionError("transaction has already begun")
        self.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            raise FileExistsError(
                f"Atom harness refuses to overwrite an existing run: {self.final_dir}"
            )
        _ensure_direct_directory(self.control_dir, self.parent)
        _ensure_direct_directory(self.lock_path.parent, self.control_dir)
        _ensure_direct_directory(self.staging_dir.parent, self.control_dir)
        _ensure_direct_directory(self.recovery_dir, self.control_dir)
        try:
            self._lock_descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            raise RunLockedError(
                f"another Atom harness run owns {self.final_dir}"
            ) from error
        self._owns_lock = True
        try:
            lock_payload = {
                "schema": 1,
                "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
                "transaction_id": self.transaction_id,
                "target": str(self.final_dir),
                "pid": os.getpid(),
                "created_at": self.created_at,
            }
            lock_bytes = _canonical_json(lock_payload)
            offset = 0
            while offset < len(lock_bytes):
                written = os.write(self._lock_descriptor, lock_bytes[offset:])
                if written <= 0:
                    raise OSError("transaction lock write was incomplete")
                offset += written
            os.fsync(self._lock_descriptor)
            self.staging_dir.mkdir(parents=False, exist_ok=False)
            self._write_state("preparing", required_files=(), files=(), sealed_at=None)
        except BaseException:
            try:
                if self.staging_dir.exists():
                    destination = self.recovery_dir / (
                        f"{self.transaction_id}.begin-failed"
                    )
                    os.replace(self.staging_dir, destination)
            finally:
                self._release_lock()
            raise
        return self

    def __enter__(self) -> "RunTransaction":
        return self.begin()

    def _target(self, relative_path: str | Path) -> Path:
        if not self._owns_lock or not self.staging_dir.is_dir():
            raise RunTransactionError("transaction has not begun")
        if self._sealed_manifest is not None:
            raise RunTransactionError("sealed transaction is not writable")
        if self._published:
            raise RunTransactionError("published transaction is not writable")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("transaction path must stay inside the staged run")
        if relative.as_posix().casefold() == ATOM_RUN_TRANSACTION_FILENAME.casefold():
            raise ValueError("transaction manifest path is reserved")
        target = (self.staging_dir / relative).resolve()
        if self.staging_dir.resolve() not in target.parents:
            raise ValueError("transaction path escaped the staged run")
        return target

    def write_json(
        self,
        relative_path: str | Path,
        payload: Mapping[str, Any],
    ) -> Path:
        target = self._target(relative_path)
        _atomic_write(target, _canonical_json(payload))
        return target

    def write_text(
        self,
        relative_path: str | Path,
        text: str,
    ) -> Path:
        target = self._target(relative_path)
        _atomic_write(target, str(text).encode("utf-8"))
        return target

    def copy_file(
        self,
        relative_path: str | Path,
        source_path: Path,
    ) -> Path:
        """Copy a regular immutable input into the staged transaction safely."""

        source = Path(source_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise ValueError("transaction source must be a regular file")
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    written = writer.write(block)
                    if written != len(block):
                        raise OSError("transaction file copy was incomplete")
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def snapshot_file(
        self,
        relative_path: str | Path,
        source_path: Path,
    ) -> Path:
        """Bind an immutable snapshot by hard link, with safe copy fallback."""

        source = Path(source_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise ValueError("transaction snapshot source must be a regular file")
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            try:
                os.link(source, temporary)
            except OSError:
                with source.open("rb") as reader, temporary.open("xb") as writer:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        written = writer.write(block)
                        if written != len(block):
                            raise OSError("transaction snapshot copy was incomplete")
                    writer.flush()
                    os.fsync(writer.fileno())
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def _files(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.staging_dir.rglob("*")):
            if not path.is_file() or path.name == ATOM_RUN_TRANSACTION_FILENAME:
                continue
            rows.append(
                {
                    "path": path.relative_to(self.staging_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        return rows

    def _write_state(
        self,
        state: str,
        *,
        required_files: Sequence[str],
        files: Sequence[Mapping[str, Any]],
        sealed_at: str | None,
    ) -> dict[str, Any]:
        core = _transaction_core(
            transaction_id=self.transaction_id,
            target_name=self.final_dir.name,
            state=state,
            created_at=self.created_at,
            sealed_at=sealed_at,
            pid=os.getpid(),
            required_files=required_files,
            files=files,
        )
        manifest = {**core, "manifest_hash": canonical_hash(core)}
        _atomic_write(
            self.staging_dir / ATOM_RUN_TRANSACTION_FILENAME,
            _canonical_json(manifest),
        )
        return manifest

    def seal(self, *, required_files: Sequence[str]) -> dict[str, Any]:
        if self._sealed_manifest is not None:
            raise RunTransactionError("transaction is already sealed")
        normalized_required = tuple(dict.fromkeys(str(item) for item in required_files))
        if any(not _valid_relative_path(item) for item in normalized_required):
            raise RunIntegrityError("required transaction path is invalid")
        files = self._files()
        existing = {str(item["path"]) for item in files}
        missing = sorted(set(normalized_required) - existing)
        if missing:
            raise RunIntegrityError(
                "cannot seal run with missing files: " + ", ".join(missing)
            )
        self._sealed_manifest = self._write_state(
            "sealed",
            required_files=normalized_required,
            files=files,
            sealed_at=_utc_now(),
        )
        return dict(self._sealed_manifest)

    def commit(self) -> dict[str, Any]:
        if self._published:
            raise RunTransactionError("transaction is already published")
        if self._sealed_manifest is None:
            raise RunTransactionError("run must be sealed before commit")
        if self.final_dir.exists():
            raise FileExistsError(
                f"Atom harness refuses to overwrite an existing run: {self.final_dir}"
            )
        committed = self._write_state(
            "committed",
            required_files=self._sealed_manifest["required_files"],
            files=self._sealed_manifest["files"],
            sealed_at=self._sealed_manifest["sealed_at"],
        )
        _validate_manifest(
            self.staging_dir,
            committed,
            expected_state="committed",
            expected_target_name=self.final_dir.name,
        )
        os.replace(self.staging_dir, self.final_dir)
        self._published = True
        try:
            _fsync_directory(self.parent)
            return verify_committed_run(self.final_dir)
        except BaseException:
            destination = self.recovery_dir / (f"{self.transaction_id}.commit-failed")
            if self.final_dir.exists() and not destination.exists():
                os.replace(self.final_dir, destination)
                self._published = False
            raise
        finally:
            self._release_lock()

    def abort(self, reason: str) -> Path | None:
        if self._published or not self.staging_dir.exists():
            self._release_lock()
            return None
        destination: Path | None = None
        try:
            reason_hash = hashlib.sha256(
                str(reason).encode("utf-8", errors="replace")
            ).hexdigest()
            try:
                _atomic_write(
                    self.staging_dir / "atom_harness_failure.json",
                    _canonical_json(
                        {
                            "schema": 1,
                            "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
                            "transaction_id": self.transaction_id,
                            "state": "aborted",
                            "reason_sha256": reason_hash,
                            "failed_at": _utc_now(),
                        }
                    ),
                )
            except OSError:
                pass
            destination = self.recovery_dir / f"{self.transaction_id}.aborted"
            if destination.exists():
                destination = self.recovery_dir / (
                    f"{self.transaction_id}.{uuid.uuid4().hex}.aborted"
                )
            os.replace(self.staging_dir, destination)
            return destination
        finally:
            self._release_lock()

    def _release_lock(self) -> None:
        owned = self._owns_lock
        self._owns_lock = False
        if self._lock_descriptor is not None:
            try:
                os.close(self._lock_descriptor)
            finally:
                self._lock_descriptor = None
        if owned and self.lock_path.exists():
            try:
                if self.lock_path.stat().st_size > 64 * 1024:
                    payload = {}
                else:
                    payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = {}
            if payload.get("transaction_id") in {None, self.transaction_id}:
                self.lock_path.unlink()

    def __exit__(self, error_type, error, traceback) -> bool:
        if error is not None:
            self.abort(type(error).__name__)
        elif not self._published:
            self.abort("transaction context exited without commit")
        else:
            self._release_lock()
        return False


def recover_transactions(output_parent: Path) -> list[dict[str, Any]]:
    """Recover sealed crashes and quarantine incomplete, dead-process stages."""

    parent = Path(output_parent).resolve()
    control = parent / ".atom-harness-v2"
    staging_root = control / "staging"
    recovery_root = control / "recovery"
    lock_root = control / "locks"
    parent.mkdir(parents=True, exist_ok=True)
    _ensure_direct_directory(control, parent)
    _ensure_direct_directory(recovery_root, control)
    for optional_root in (staging_root, lock_root):
        if optional_root.exists() and (
            _is_link(optional_root)
            or not optional_root.is_dir()
            or optional_root.resolve().parent != control.resolve()
        ):
            raise RunIntegrityError("transaction control root is unsafe")
    events: list[dict[str, Any]] = []
    if staging_root.is_dir():
        for staging in sorted(path for path in staging_root.iterdir() if path.is_dir()):
            if _is_link(staging) or not _safe_child(staging_root, staging):
                events.append(
                    {
                        "transaction_id": staging.name,
                        "action": "ignored-unsafe-staging-path",
                    }
                )
                continue
            marker = staging / ATOM_RUN_TRANSACTION_FILENAME
            try:
                if marker.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("transaction marker is too large")
                manifest = json.loads(marker.read_text(encoding="utf-8"))
                pid = int(manifest["pid"])
                target = parent / str(manifest["target_name"])
                transaction_id = str(manifest["transaction_id"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pid = -1
                target = parent / "__invalid_target__"
                transaction_id = staging.name
                manifest = {}
            if _pid_running(pid):
                events.append(
                    {
                        "transaction_id": transaction_id,
                        "action": "active-skip",
                    }
                )
                continue
            if not _safe_child(parent, target):
                action = "quarantined-invalid-target"
            elif manifest.get("state") == "committed" and not target.exists():
                try:
                    _validate_manifest(
                        staging,
                        manifest,
                        expected_state="committed",
                        expected_target_name=target.name,
                    )
                except RunIntegrityError:
                    action = "quarantined-integrity-failure"
                else:
                    os.replace(staging, target)
                    _fsync_directory(parent)
                    action = "recovered-commit"
            else:
                action = "quarantined-incomplete"
            if staging.exists():
                destination = recovery_root / f"{transaction_id}.recovered"
                if destination.exists():
                    destination = recovery_root / (
                        f"{transaction_id}.{uuid.uuid4().hex}.recovered"
                    )
                os.replace(staging, destination)
            events.append(
                {
                    "transaction_id": transaction_id,
                    "action": action,
                    "target": str(target) if _safe_child(parent, target) else None,
                }
            )

    if lock_root.is_dir():
        for lock in sorted(path for path in lock_root.iterdir() if path.is_file()):
            if _is_link(lock) or not _safe_child(lock_root, lock):
                events.append(
                    {
                        "transaction_id": lock.name,
                        "action": "ignored-unsafe-lock-path",
                    }
                )
                continue
            try:
                if lock.stat().st_size > 64 * 1024:
                    raise ValueError("transaction lock is too large")
                payload = json.loads(lock.read_text(encoding="utf-8"))
                pid = int(payload["pid"])
                transaction_id = str(payload["transaction_id"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pid = -1
                transaction_id = lock.stem
            if _pid_running(pid):
                continue
            destination = recovery_root / f"{transaction_id}.stale-lock"
            if destination.exists():
                destination = recovery_root / (
                    f"{transaction_id}.{uuid.uuid4().hex}.stale-lock"
                )
            os.replace(lock, destination)
            events.append(
                {
                    "transaction_id": transaction_id,
                    "action": "recovered-stale-lock",
                }
            )
    return events
