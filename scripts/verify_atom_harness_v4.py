"""Fail-closed repository policy checks for Atom Harness Operator V4."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from verify_atom_harness_v3 import (  # noqa: E402
    PolicyFailure,
    _check_ci_contract,
    _check_git_surface,
    _check_required_files,
    _check_runtime_declarations,
    _check_rust_crate_sizes,
    _load_json,
)


V4_TEST = "tests/test_atom_language_harness_v4_integration.py"
V4_ENTRYPOINT = "atom_harness_operator_server.py"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    normalized = (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    return hashlib.sha256(normalized).hexdigest()


def _require_markers(path: str, markers: tuple[str, ...]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise PolicyFailure(f"{path} is missing V4 marker: {marker}")


def _check_v4_surface() -> dict[str, Any]:
    registry = _load_json("ai-runtime-registry.json")
    knowledge = _load_json("ai-runtime-knowledge.json")
    side_view = _load_json("ai-artifact-side-view.json")
    fabric = _load_json("ai-provider-fabric.json")
    transaction = _load_json("ai-run-transaction.json")
    architecture = _load_json("atom-language-harness-architecture.json")

    if registry.get("active_runtime") != "language-harness-v4":
        raise PolicyFailure("operator V4 is not active")
    active = registry.get("runtimes", {}).get("language-harness-v4")
    historical = registry.get("runtimes", {}).get("language-harness-v3")
    if not isinstance(active, dict) or not isinstance(historical, dict):
        raise PolicyFailure("V4 active or V3 historical registry entry is absent")
    if active.get("runtime_entrypoint") != V4_ENTRYPOINT:
        raise PolicyFailure("V4 registry entrypoint is invalid")
    if active.get("integration_test") != V4_TEST:
        raise PolicyFailure("V4 registry integration test is invalid")
    if active.get("artifact_binding_marker") != "render_operator_surface":
        raise PolicyFailure("V4 artifact binding is invalid")
    if historical.get("integration_test") != (
        "tests/test_atom_language_harness_v3_integration.py"
    ):
        raise PolicyFailure("V3 historical integration is not preserved")

    for declaration in (knowledge, side_view, fabric, transaction):
        if declaration.get("runtime_entrypoint") != V4_ENTRYPOINT:
            raise PolicyFailure("a V4 declaration has the wrong entrypoint")
        if declaration.get("integration_test") != V4_TEST:
            raise PolicyFailure("a V4 declaration has the wrong integration test")

    if architecture.get("schema") != 4:
        raise PolicyFailure("operator architecture schema is not V4")
    if architecture.get("runtime") != "atom-language-harness-operator-v4":
        raise PolicyFailure("operator architecture runtime is invalid")
    operator = architecture.get("operator")
    if not isinstance(operator, dict):
        raise PolicyFailure("operator architecture section is absent")
    for field in (
        "browser_token_in_memory_only",
        "host_header_validation",
        "same_origin_controls",
        "bounded_request_body",
        "durable_session_history",
    ):
        if operator.get(field) is not True:
            raise PolicyFailure(f"operator architecture must require {field}")
    if operator.get("cloud_allowed") is not False:
        raise PolicyFailure("operator architecture must keep cloud disabled")
    certification = operator.get("certification_evidence")
    if not isinstance(certification, dict):
        raise PolicyFailure("operator live certification evidence is absent")
    if (
        certification.get("status") != "certified-live-local"
        or certification.get("runtime") != "atom-harness-operator-certification-v1"
        or certification.get("source_hash_normalization") != "utf-8-lf-v1"
        or certification.get("mode") != "live"
        or certification.get("request_count") != 100
        or certification.get("elapsed_seconds", 0) < 3600
        or certification.get("all_checks_passed") is not True
    ):
        raise PolicyFailure("operator live certification evidence is invalid")
    for field in ("report_sha256", "report_hash", "source_manifest_hash"):
        if not _is_sha256(certification.get(field)):
            raise PolicyFailure(f"operator certification {field} is invalid")
    source_files = certification.get("source_files_sha256")
    if (
        not isinstance(source_files, dict)
        or len(source_files) != 15
        or any(not _is_sha256(value) for value in source_files.values())
    ):
        raise PolicyFailure("operator certification source binding is invalid")
    for relative_path, expected_hash in source_files.items():
        source_path = ROOT / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise PolicyFailure(
                f"operator certified source is unavailable: {relative_path}"
            )
        if _sha256_file(source_path) != expected_hash:
            raise PolicyFailure(f"operator certified source changed: {relative_path}")
    calculated_manifest_hash = hashlib.sha256(
        json.dumps(
            source_files,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if calculated_manifest_hash != certification["source_manifest_hash"]:
        raise PolicyFailure("operator certification source manifest hash is invalid")
    for field in (
        "cancellation_and_retry_passed",
        "resident_restart_and_rewarm_passed",
        "all_requests_passed",
        "knowledge_store_unchanged",
        "journal_hash_valid",
        "operator_closed_cleanly",
    ):
        if certification.get(field) is not True:
            raise PolicyFailure(f"operator certification must require {field}")

    side = side_view.get("side_view")
    if not isinstance(side, dict):
        raise PolicyFailure("operator side-view declaration is absent")
    for field in (
        "enabled",
        "user_visible",
        "bound_to_real_artifact_output",
        "real_artifact_loaded_from_committed_transaction",
        "authenticated_artifact_fetch",
        "artifact_token_absent_from_url",
        "http_only_same_site_artifact_cookie",
        "artifact_cookie_path_scoped",
        "sandboxed_artifact_frame",
    ):
        if side.get(field) is not True:
            raise PolicyFailure(f"operator side view must require {field}")
    if side.get("placement") != "side":
        raise PolicyFailure("operator artifact view is not placed at the side")
    if side.get("artifact_binding_marker") != "render_operator_surface":
        raise PolicyFailure("operator side-view binding marker is invalid")

    if knowledge.get("session_resident") is not True:
        raise PolicyFailure("operator knowledge is not session-resident")
    if knowledge.get("immutable_during_requests") is not True:
        raise PolicyFailure("operator knowledge is not immutable")
    if knowledge.get("same_volume_hard_link_with_copy_fallback") is not True:
        raise PolicyFailure("operator immutable snapshot fallback is absent")

    run_transaction = transaction.get("run_transaction")
    if not isinstance(run_transaction, dict):
        raise PolicyFailure("operator transaction declaration is absent")
    for field in (
        "durable_operator_journal",
        "journal_hash_bound",
        "journal_recovery_marks_interrupted_requests",
        "operator_history_references_committed_artifacts",
    ):
        if run_transaction.get(field) is not True:
            raise PolicyFailure(f"operator transaction must require {field}")

    _require_markers(
        V4_ENTRYPOINT,
        (
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
            "ATOM_HARNESS_OPERATOR_UI_RUNTIME",
            "render_operator_surface",
            "X-Atom-Operator-Token",
            "invalid-origin",
            "invalid-host",
            "MAX_REQUEST_BODY_BYTES",
            "AtomArtifactToken",
            "SameSite=Strict",
            "SAMEORIGIN",
            "127.0.0.1",
        ),
    )
    _require_markers(
        "atom_harness_operator.py",
        (
            "queue.Queue",
            "CancellationToken",
            "operator-restart-recovery",
            "operator-queue-backpressure",
            "operator-artifact-demoted",
            "verify_committed_run",
            "journal_hash",
            "restart_resident_lane",
        ),
    )
    _require_markers(
        "atom_harness_operator_ui.py",
        (
            "render_operator_surface",
            "REAL ARTIFACT SIDE VIEW",
            "sandbox=",
            'sandbox=""',
            'artifactFrame.removeAttribute("srcdoc")',
            "X-Atom-Operator-Token",
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
        ),
    )
    _require_markers(
        V4_TEST,
        (
            "render_operator_surface",
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
            "atom_harness_operator_server.py",
            "operator-thread-formed",
            "operator-artifact-demoted",
            "/api/shutdown",
        ),
    )
    _require_markers(
        "scripts/certify_atom_harness_operator.py",
        (
            "MIN_LIVE_REQUESTS = 100",
            "DEFAULT_LIVE_DURATION_SECONDS = 3600",
            "cancellation_and_retry",
            "resident_restart_and_rewarm",
            "knowledge_store_unchanged",
            "working_set_observed",
            "gpu_growth_bounded",
            "working_set_growth_bounded",
            "working_set_ceiling_bounded",
            "_working_set_evidence",
        ),
    )
    _require_markers(
        "run-atom-harness-operator.ps1",
        (
            "language-harness-v4",
            "atom_harness_operator_server.py",
            "render_operator_surface",
            "llama-server",
            "Preloading the Atom graph",
        ),
    )
    return {
        "operator_runtime": active["operator_runtime"],
        "operator_entrypoint": active["runtime_entrypoint"],
        "operator_integration_test": active["integration_test"],
        "operator_certification": active["operator_certification"],
    }


def main() -> int:
    try:
        _check_required_files()
        git_surface = _check_git_surface()
        declarations = _check_runtime_declarations()
        ci_contract = _check_ci_contract()
        rust_crates = _check_rust_crate_sizes()
        v4 = _check_v4_surface()
    except (
        PolicyFailure,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        print(f"POLICY FAILURE: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "policy": "atom-harness-v4",
                **git_surface,
                **declarations,
                **ci_contract,
                **v4,
                "rust_crate_lines": rust_crates,
                "rust_crate_line_limit": 4000,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
