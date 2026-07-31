"""Fail-closed repository policy checks for Atom Harness Phase 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from verify_atom_harness_v3 import (  # noqa: E402
    PolicyFailure,
    _check_git_surface,
    _check_rust_crate_sizes,
    _load_json,
)


PHASE6_TEST = "tests/test_atom_permissioned_hands_integration.py"
DESKTOP_TEST = "tests/test_atom_harness_desktop_v6_integration.py"
WORKFLOW = ".github/workflows/atom-harness-v6-ci.yml"

REQUIRED_FILES = (
    WORKFLOW,
    "README.md",
    "ATOM_LANGUAGE_HARNESS.md",
    "ATOM_HARNESS_OPERATOR.md",
    "ATOM_HARNESS_DESKTOP.md",
    "DEVELOPER_NOTES.md",
    "ai-artifact-side-view.json",
    "ai-provider-fabric.json",
    "ai-run-transaction.json",
    "ai-runtime-knowledge.json",
    "ai-runtime-registry.json",
    "ai-tool-fabric.json",
    "atom-harness-desktop-release-evidence.json",
    "atom-language-harness-architecture.json",
    "atom-harness-desktop-architecture.json",
    "atom-permissioned-hands-certification.json",
    "atom_harness_operator.py",
    "atom_harness_operator_server.py",
    "atom_harness_operator_ui.py",
    "atom_tool_capabilities.py",
    "atom_tool_fabric.py",
    "atom_tool_protocol.py",
    "atom_tool_side_view.py",
    "lucerna-update.json",
    "run-atom-harness-operator.ps1",
    "scripts/build_atom_harness_desktop.ps1",
    "scripts/certify_atom_permissioned_hands.py",
    "scripts/verify_atom_harness_v6.py",
    PHASE6_TEST,
    DESKTOP_TEST,
)


def _require_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise PolicyFailure("required Phase 6 files are absent: " + ", ".join(missing))
    symlinks = [name for name in REQUIRED_FILES if (ROOT / name).is_symlink()]
    if symlinks:
        raise PolicyFailure(
            "required Phase 6 files may not be symlinks: " + ", ".join(symlinks)
        )


def _require_markers(path: str, markers: tuple[str, ...]) -> str:
    source = (ROOT / path).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in source:
            raise PolicyFailure(f"{path} is missing Phase 6 marker: {marker}")
    return source


def _require_true(mapping: dict[str, Any], *fields: str) -> None:
    for field in fields:
        if mapping.get(field) is not True:
            raise PolicyFailure(f"required boolean is not true: {field}")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_file_hash(path: Path) -> str:
    data = (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def _require_sha256(mapping: dict[str, Any], *fields: str) -> None:
    for field in fields:
        if not re.fullmatch(r"[0-9a-f]{64}", str(mapping.get(field, ""))):
            raise PolicyFailure(f"required SHA-256 is invalid: {field}")


def _check_runtime_contracts() -> dict[str, str]:
    registry = _load_json("ai-runtime-registry.json")
    knowledge = _load_json("ai-runtime-knowledge.json")
    side_view = _load_json("ai-artifact-side-view.json")
    provider = _load_json("ai-provider-fabric.json")
    transaction = _load_json("ai-run-transaction.json")
    tools = _load_json("ai-tool-fabric.json")
    architecture = _load_json("atom-language-harness-architecture.json")

    if (
        registry.get("schema_version") != 1
        or registry.get("active_runtime") != "language-harness-v5"
    ):
        raise PolicyFailure("language-harness-v5 is not the active schema-1 runtime")
    runtimes = registry.get("runtimes")
    active = runtimes.get("language-harness-v5") if isinstance(runtimes, dict) else None
    historical = (
        runtimes.get("language-harness-v4") if isinstance(runtimes, dict) else None
    )
    if not isinstance(active, dict) or not isinstance(historical, dict):
        raise PolicyFailure("current V5 or historical V4 runtime entry is absent")
    expected_active = {
        "runtime_entrypoint": "atom_harness_operator_server.py",
        "session_entrypoint": "atom_harness_operator.py",
        "browser_surface": "atom_harness_operator_ui.py",
        "tool_fabric_runtime": "ATOM_PERMISSIONED_HANDS_RUNTIME",
        "tool_planner_runtime": "ATOM_TOOL_PLANNER_RUNTIME",
        "tool_capability_runtime": "ATOM_TOOL_CAPABILITY_RUNTIME",
        "tool_side_view_runtime": "ATOM_TOOL_SIDE_VIEW_RUNTIME",
        "tool_artifact_binding_marker": "render_atom_tool_artifact",
        "permission_runtime": "ATOM_PERMISSION_GRANT_RUNTIME",
        "integration_test": PHASE6_TEST,
    }
    for field, expected in expected_active.items():
        if active.get(field) != expected:
            raise PolicyFailure(f"active runtime field is invalid: {field}")
    if (
        historical.get("integration_test")
        != "tests/test_atom_language_harness_v4_integration.py"
    ):
        raise PolicyFailure("historical V4 runtime is not preserved")

    for declaration in (knowledge, side_view, provider, transaction, tools):
        if declaration.get("schema") != 1:
            raise PolicyFailure("a Phase 6 declaration is not schema 1")
        if declaration.get("project_kind") != "ai-harness":
            raise PolicyFailure("a Phase 6 declaration is not an AI harness")
        if declaration.get("runtime_entrypoint") != "atom_harness_operator_server.py":
            raise PolicyFailure("Phase 6 runtime entrypoints disagree")
        if declaration.get("integration_test") != PHASE6_TEST:
            raise PolicyFailure("Phase 6 integration-test declarations disagree")

    wiki = knowledge.get("wiki_graph")
    rag = knowledge.get("rag")
    if not isinstance(wiki, dict) or not isinstance(rag, dict):
        raise PolicyFailure("wiki graph or RAG declaration is absent")
    _require_true(wiki, "enabled")
    _require_true(rag, "enabled")
    _require_true(
        knowledge,
        "session_resident",
        "immutable_during_requests",
        "exact_snapshot_bound_into_each_committed_transaction",
        "permissioned_tool_artifacts_bind_the_same_immutable_snapshot",
    )

    side = side_view.get("side_view")
    if not isinstance(side, dict):
        raise PolicyFailure("artifact side-view declaration is absent")
    _require_true(
        side,
        "enabled",
        "user_visible",
        "bound_to_real_artifact_output",
        "real_artifact_loaded_from_committed_transaction",
        "authenticated_artifact_fetch",
        "sandboxed_artifact_frame",
        "permission_manifest_visible",
        "tool_results_visible",
        "tool_results_rendered_as_untrusted",
    )
    if side.get("placement") != "side":
        raise PolicyFailure(
            "artifact output is not declared beside the primary surface"
        )
    if side.get("artifact_binding_marker") != "render_operator_surface":
        raise PolicyFailure("evidence artifact binding marker is invalid")
    if side.get("tool_artifact_binding_marker") != "render_atom_tool_artifact":
        raise PolicyFailure("tool artifact binding marker is invalid")

    fabric = provider.get("provider_fabric")
    if not isinstance(fabric, dict):
        raise PolicyFailure("provider fabric declaration is absent")
    _require_true(
        fabric,
        "enabled",
        "strict_json_required",
        "bounded_concurrency",
        "cancellation",
        "backpressure_vibrations",
        "resident_model_preload",
        "authenticated_loopback_transport",
        "typed_resident_on_off_ramps",
    )

    run = transaction.get("run_transaction")
    if not isinstance(run, dict):
        raise PolicyFailure("run transaction declaration is absent")
    _require_true(
        run,
        "enabled",
        "staged_writes",
        "atomic_directory_publication",
        "target_locking",
        "committed_file_manifest",
        "durable_operator_journal",
        "journal_hash_bound",
        "permissioned_tool_artifacts_atomic",
        "permission_receipt_manifested",
        "quarantined_tool_results_manifested",
    )
    if run.get("overwrite_allowed") is not False:
        raise PolicyFailure("committed transactions may not overwrite an artifact")

    hands = tools.get("permissioned_hands")
    if not isinstance(hands, dict):
        raise PolicyFailure("permissioned-hands declaration is absent")
    _require_true(
        hands,
        "enabled",
        "experimental",
        "operator_permission_required_for_every_execution",
        "exact_manifest_hash_required",
        "model_candidate_reduced_to_capability_contract",
        "candidate_normalizations_user_visible",
        "missing_required_arguments_fail_closed",
        "process_executable_sha256_bound_into_manifest",
        "process_output_retention_bounded",
        "process_timeout_terminates_process_tree",
        "public_web_address_set_bound_into_manifest",
        "web_connect_uses_only_permission_bound_address",
        "one_time_non_replayable_grant",
        "changed_manifest_requires_new_permission",
        "tool_results_are_untrusted",
        "atomic_tool_artifacts",
        "user_visible_permission_surface",
        "user_visible_real_result_side_view",
    )
    for field in (
        "permission_grant_secret_in_memory_only",
        "tool_results_cannot_authorize_actions",
    ):
        if hands.get(field) is not True:
            raise PolicyFailure(f"permission boundary is missing {field}")
    for field in (
        "atom_memory_mutation_allowed",
        "shell_expansion",
        "provider_environment_secrets_forwarded",
        "web_redirects_allowed",
    ):
        if hands.get(field) is not False:
            raise PolicyFailure(f"permission boundary must prohibit {field}")

    if (
        architecture.get("schema") != 5
        or architecture.get("runtime") != "atom-language-harness-operator-v5"
        or architecture.get("integration_test") != PHASE6_TEST
    ):
        raise PolicyFailure("Phase 6 language architecture identity is invalid")
    permissioned = architecture.get("permissioned_hands")
    if not isinstance(permissioned, dict) or permissioned.get("experiment_phase") != 6:
        raise PolicyFailure("Phase 6 architecture lacks permissioned hands")
    _require_true(
        permissioned,
        "every_execution_requires_operator_permission",
        "changed_manifest_requires_new_permission",
        "tool_results_are_untrusted",
        "workspace_scope_is_explicit",
        "model_candidate_reduced_to_capability_contract",
        "candidate_normalizations_user_visible",
        "missing_required_arguments_fail_closed",
        "process_executable_sha256_bound_into_manifest",
        "process_output_retention_bounded",
        "process_timeout_terminates_process_tree",
        "public_web_address_set_bound_into_manifest",
        "web_connect_uses_only_permission_bound_address",
    )
    if permissioned.get("planner_may_grant_permission") is not False:
        raise PolicyFailure("the planner may not grant permission")
    if permissioned.get("planner_receives_executable_handles") is not False:
        raise PolicyFailure("the planner may not receive executable handles")
    if permissioned.get("atom_memory_mutation_allowed") is not False:
        raise PolicyFailure("tools may not mutate Atom memory")
    if permissioned.get("web_redirects_allowed") is not False:
        raise PolicyFailure("public web reads may not follow redirects")

    return {
        "active_runtime": str(registry["active_runtime"]),
        "runtime_integration_test": PHASE6_TEST,
        "historical_runtime": "language-harness-v4",
    }


def _check_runtime_wiring() -> dict[str, Any]:
    _require_markers(
        "atom_harness_operator_server.py",
        (
            "ATOM_PERMISSIONED_HANDS_RUNTIME",
            "ATOM_TOOL_SIDE_VIEW_RUNTIME",
            "/api/tools/propose",
            "/api/tools/approve",
            "/api/tools/deny",
            "/api/tools/cancel",
            "/api/tool-artifacts/",
            "SameSite=Strict",
            "X-Atom-Operator-Token",
        ),
    )
    _require_markers(
        "atom_harness_operator.py",
        (
            "tool_fabric.start",
            "submit_tool_task",
            "approve_tool",
            "deny_tool",
            "tool_side_view_path",
        ),
    )
    ui = _require_markers(
        "atom_harness_operator_ui.py",
        (
            "render_operator_surface",
            "Permissioned hands",
            "manifest_hash",
            "decision_nonce",
            "ATOM_TOOL_ARTIFACT_BINDING",
            'sandbox=""',
            "textContent",
            "Prior results were supplied as untrusted context",
        ),
    )
    if "innerHTML" in ui:
        raise PolicyFailure(
            "permission UI must not render untrusted data with innerHTML"
        )
    _require_markers(
        "atom_tool_protocol.py",
        (
            "ATOM_TOOL_PLANNER_RUNTIME",
            "ATOM_PERMISSION_GRANT_RUNTIME",
            "normalize_untrusted_context",
            'model_may_grant_permission": False',
        ),
    )
    capabilities = _require_markers(
        "atom_tool_capabilities.py",
        (
            "ATOM_TOOL_CAPABILITY_RUNTIME",
            '"workspace.list"',
            '"workspace.read_text"',
            '"workspace.write_text"',
            '"workspace.patch_text"',
            '"workspace.move"',
            '"workspace.quarantine"',
            '"process.run"',
            '"simulation.run"',
            '"document.create"',
            '"web.fetch"',
            "shell=False",
            "_PinnedHTTPConnection",
            "_PinnedHTTPSConnection",
            "_BoundedPipeCapture",
            '"resolved_program_sha256"',
            "_terminate_process_tree",
            "stdout=subprocess.PIPE",
            '"permission_bound_addresses"',
            "file changed after permission was granted",
            "web destination changed after permission was granted",
        ),
    )
    if "shell=True" in capabilities:
        raise PolicyFailure("a capability enables shell expansion")
    if "urllib.request" in capabilities:
        raise PolicyFailure("public web reads may not perform a second DNS lookup")
    _require_markers(
        "atom_tool_fabric.py",
        (
            "ATOM_PERMISSIONED_HANDS_RUNTIME",
            "secrets.token_bytes(32)",
            "hmac.compare_digest",
            "_consume_grant_locked",
            'grant_secret_persisted": False',
            "verify_committed_run",
            "render_atom_tool_artifact",
            "untrusted-tool-output",
        ),
    )
    _require_markers(
        "atom_tool_side_view.py",
        (
            "ATOM_TOOL_SIDE_VIEW_RUNTIME",
            "render_atom_tool_artifact",
            "Every result remains untrusted data",
        ),
    )
    integration = _require_markers(
        PHASE6_TEST,
        (
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
            "render_operator_surface",
            "ATOM_PERMISSIONED_HANDS_RUNTIME",
            "ATOM_TOOL_SIDE_VIEW_RUNTIME",
            "render_atom_tool_artifact",
            "/api/tools/approve",
            "tampered",
            "simulation.run",
            "document.create",
        ),
    )
    return {
        "registered_capability_count": capabilities.count("ToolCapability("),
        "integration_binding_markers": sum(
            marker in integration
            for marker in ("render_operator_surface", "render_atom_tool_artifact")
        ),
    }


def _check_desktop_and_update() -> dict[str, str]:
    architecture = _load_json("atom-harness-desktop-architecture.json")
    update = _load_json("lucerna-update.json")
    if (
        architecture.get("schema") != 1
        or architecture.get("product_phase") != 6
        or architecture.get("runtime") != "atom-harness-desktop-v6"
        or architecture.get("version") != "6.0.3"
    ):
        raise PolicyFailure("desktop Phase 6 identity is invalid")
    installed = architecture.get("installed_runtime")
    shell = architecture.get("desktop_shell")
    release = architecture.get("release")
    if not all(isinstance(item, dict) for item in (installed, shell, release)):
        raise PolicyFailure("desktop Phase 6 sections are absent")
    if installed.get("authority_runtime") != "language-harness-v5":
        raise PolicyFailure("desktop authority runtime is invalid")
    if (
        installed.get("local_only") is not True
        or installed.get("cloud_allowed") is not False
    ):
        raise PolicyFailure("desktop must remain local-only by default")
    _require_true(
        shell,
        "thin_shell_only",
        "atom_authority_unchanged",
        "per_user_install",
        "single_instance",
        "backend_job_object_kill_on_close",
        "real_artifact_side_view_embedded",
        "trusted_permission_controls_embedded",
        "exact_action_manifest_visible_before_approval",
    )
    desktop_hands = architecture.get("permissioned_hands")
    if not isinstance(desktop_hands, dict):
        raise PolicyFailure("desktop permissioned-hands contract is absent")
    _require_true(
        desktop_hands,
        "operator_permission_required_for_every_execution",
        "candidate_normalizations_user_visible",
        "process_executable_sha256_bound_into_manifest",
        "process_output_retention_bounded",
        "process_timeout_terminates_process_tree",
        "public_web_address_set_bound_into_manifest",
        "web_connect_uses_only_permission_bound_address",
        "exact_manifest_hash_bound",
        "one_time_grants",
        "tool_results_untrusted",
    )
    if desktop_hands.get("web_redirects_allowed") is not False:
        raise PolicyFailure("desktop web capabilities may not follow redirects")
    if release.get("integration_test") != DESKTOP_TEST:
        raise PolicyFailure("desktop Phase 6 integration declaration is invalid")

    policy = update.get("policy")
    if (
        update.get("schema") != 1
        or update.get("app_id") != "com.lucernalabs.atom-harness"
        or update.get("current_version") != "6.0.3"
        or update.get("platform") != "windows-x64"
        or not isinstance(policy, dict)
    ):
        raise PolicyFailure("desktop update contract identity is invalid")
    _require_true(
        policy,
        "explicit_user_consent_required",
        "artifact_sha256_required",
        "stage_outside_install_directory",
        "replace_only_after_app_exit",
        "rollback_backup_required",
    )
    if (
        policy.get("automatic_download") is not False
        or policy.get("automatic_install") is not False
    ):
        raise PolicyFailure("desktop updates may not download or install silently")

    _require_markers(
        "desktop/AtomHarness.Desktop/Program.cs",
        ("atom-harness-desktop-v6", "AtomHarness.Desktop.v6"),
    )
    _require_markers(
        "desktop/AtomHarness.Desktop/BackendSupervisor.cs",
        (
            "atom-harness-operator-loopback-server-v2",
            "atom-permissioned-hands-fabric-v1",
            "atom-permissioned-hands-side-view-v1",
            "ToolWorkspace",
        ),
    )
    _require_markers(
        "desktop/AtomHarness.Desktop/MainForm.cs",
        (
            "confirmUpdate",
            "confirmInstall",
            "DownloadAndVerifyAsync",
            "RequestGracefulShutdownAsync",
        ),
    )
    _require_markers(
        "scripts/build_atom_harness_desktop.ps1",
        (
            "6.0.3",
            "verify_atom_harness_v6.py",
            "--source-only",
            "PyInstaller",
            "cargo build",
            "dotnet publish",
            "Compress-Archive",
            "heat.exe",
            "candle.exe",
            "light.exe",
        ),
    )
    return {
        "desktop_runtime": str(architecture["runtime"]),
        "desktop_version": str(architecture["version"]),
        "update_runtime": str(update["runtime"]),
    }


def _check_release_evidence_and_certificate() -> dict[str, str]:
    evidence = _load_json("atom-harness-desktop-release-evidence.json")
    package = evidence.get("package")
    installation = evidence.get("installation")
    runtime = evidence.get("interactive_runtime")
    hands = evidence.get("permissioned_hands")
    model = evidence.get("model")
    update_boundary = evidence.get("update_boundary")
    if (
        evidence.get("schema") != 1
        or evidence.get("runtime") != "atom-harness-desktop-release-evidence-v2"
        or evidence.get("version") != "6.0.3"
        or evidence.get("platform") != "windows-x64"
        or evidence.get("all_checks_passed") is not True
        or not all(
            isinstance(section, dict)
            for section in (
                package,
                installation,
                runtime,
                hands,
                model,
                update_boundary,
            )
        )
    ):
        raise PolicyFailure("Phase 6 desktop release evidence identity is invalid")

    assert isinstance(package, dict)
    assert isinstance(installation, dict)
    assert isinstance(runtime, dict)
    assert isinstance(hands, dict)
    assert isinstance(model, dict)
    assert isinstance(update_boundary, dict)
    if (
        package.get("portable_zip") != "Atom-Harness-6.0.3-windows-x64.zip"
        or package.get("portable_zip_bytes") != 138764940
        or package.get("portable_zip_sha256")
        != "ede0d697dbb3351f513632fe572b68ea84010fb3f2bdcd97dd01594abed5fb63"
        or package.get("msi") != "Atom-Harness-6.0.3-windows-x64.msi"
        or package.get("msi_bytes") != 120206546
        or package.get("msi_sha256")
        != "2cb27e6ea84810b21935ee08418cc9aeadc117d3ca90e7cc38a7bcbf39656dc3"
        or package.get("file_count") != 157
    ):
        raise PolicyFailure("Phase 6 package evidence does not match release 6.0.3")
    _require_sha256(
        package,
        "portable_zip_sha256",
        "msi_sha256",
        "llama_server_sha256",
    )

    _require_true(
        installation,
        "passed",
        "desktop_shortcut",
        "start_menu_shortcut",
        "full_release_manifest_verified",
    )
    _require_sha256(installation, "verification_report_sha256")
    if installation.get("verification_runtime") != (
        "atom-harness-desktop-install-verification-v1"
    ):
        raise PolicyFailure("installed-layout release evidence is invalid")

    _require_true(
        runtime,
        "real_artifact_side_view_visible",
        "failed_attempt_recovered",
        "completed_artifact_recovered_after_restart",
        "graceful_process_tree_shutdown_verified",
    )
    _require_sha256(
        runtime,
        "request_sha256",
        "transaction_sha256",
        "transaction_manifest_sha256",
        "artifact_sha256",
        "side_view_sha256",
    )
    if (
        runtime.get("citations") != 1
        or runtime.get("wiki_nodes") != 2737
        or runtime.get("retrieved_passages") != 7
        or runtime.get("cloud_evidence_used") is not False
        or runtime.get("llm_memory_writes") != 0
    ):
        raise PolicyFailure("fresh local-model evidence transaction is invalid")

    if (
        model.get("model") != "Qwen/Qwen3-4B-Instruct-2507"
        or model.get("bytes") != 4280403520
        or model.get("sha256")
        != "ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1"
        or model.get("weights_bundled") is not False
        or model.get("resident_loads") != 1
        or model.get("resident_restarts") != 0
    ):
        raise PolicyFailure("release model evidence is invalid")

    _require_sha256(
        hands,
        "proposal_payload_sha256",
        "manifest_sha256",
        "action_sha256",
        "permission_sha256",
        "journal_sha256",
    )
    normalizations = hands.get("candidate_normalizations")
    if (
        hands.get("capability") != "workspace.write_text"
        or hands.get("action_count") != 1
        or hands.get("maximum_risk") != "high"
        or hands.get("decision") != "denied"
        or hands.get("result_count") != 0
        or hands.get("proposed_file_exists_after_denial") is not False
        or hands.get("grant_secret_persisted") is not False
        or hands.get("permission_grants_persisted") is not False
        or hands.get("tool_results_trusted_as_instructions") is not False
        or not isinstance(normalizations, list)
        or normalizations
        != [
            {
                "action_id": "action-1",
                "kind": "unsupported-argument-fields-omitted",
                "fields": ["format"],
            }
        ]
    ):
        raise PolicyFailure("live permission denial evidence is invalid")
    _require_true(
        hands,
        "exact_manifest_visible",
        "normalizations_visible",
        "approve_and_deny_controls_visible",
        "denied_proposal_recovered_after_restart",
        "deterministic_completed_tool_side_view_verified",
        "public_web_connection_pinned_to_permission_bound_address",
        "process_executable_hash_drift_blocked",
        "process_output_retention_bounded",
        "process_timeout_tree_cleanup_passed",
    )

    _require_true(
        update_boundary,
        "explicit_user_consent_required",
        "sha256_required",
        "external_staging_required",
        "replace_only_after_exit",
        "rollback_backup_required",
    )
    if (
        update_boundary.get("silent_download") is not False
        or update_boundary.get("silent_install") is not False
    ):
        raise PolicyFailure("release evidence permits a silent update")

    certificate = _load_json("atom-permissioned-hands-certification.json")
    report_hash = certificate.get("report_hash")
    certificate_core = dict(certificate)
    certificate_core.pop("report_hash", None)
    checks = certificate.get("checks")
    source_hashes = certificate.get("source_files_sha256")
    if (
        certificate.get("schema") != 1
        or certificate.get("runtime") != "atom-permissioned-hands-certification-v1"
        or certificate.get("return_code") != 0
        or certificate.get("all_checks_passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(source_hashes, dict)
        or not source_hashes
        or report_hash != _canonical_hash(certificate_core)
        or certificate.get("source_manifest_hash") != _canonical_hash(source_hashes)
    ):
        raise PolicyFailure("Phase 6 adversarial certificate is invalid")
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected)
        ):
            raise PolicyFailure("Phase 6 certificate source entry is invalid")
        source = (ROOT / relative).resolve()
        try:
            source.relative_to(ROOT)
        except ValueError as error:
            raise PolicyFailure("certificate source escapes repository") from error
        if not source.is_file() or source.is_symlink():
            raise PolicyFailure(f"certificate source is absent or unsafe: {relative}")
        if _normalized_file_hash(source) != expected:
            raise PolicyFailure(f"certificate source hash drifted: {relative}")

    return {
        "desktop_release_evidence": "verified-local-package-and-live-runtime",
        "permissioned_hands_live_evidence": "denied-no-side-effects-and-recovered",
        "permissioned_hands_certificate": str(report_hash),
    }


def _check_ci() -> dict[str, str]:
    workflow = _require_markers(
        WORKFLOW,
        (
            "Verify Desktop Phase 6 on Windows",
            "python scripts/verify_atom_harness_v6.py",
            "test_atom_harness_desktop_v6_integration.py",
            "test_atom_permissioned_hands_integration.py",
            "dotnet restore",
            "--locked-mode",
            "dotnet format",
            "dotnet build",
            "dotnet test",
            "cargo clippy",
            "-D warnings",
            "cargo test",
            "PyInstaller",
        ),
    )
    pinned = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "actions/setup-dotnet@a98b56852c35b8e3190ac28c8c2271da59106c68",
    )
    for action in pinned:
        if action not in workflow:
            raise PolicyFailure(f"Phase 6 CI action is not pinned: {action}")
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not re.search(
            r"@[0-9a-f]{40}(?:\s|$)", stripped
        ):
            raise PolicyFailure(f"Phase 6 CI contains an unpinned action: {stripped}")
    if 'dotnet-version: "9.0.305"' not in workflow:
        raise PolicyFailure("Phase 6 CI must pin .NET SDK 9.0.305")
    return {"phase6_ci": "full-sha-pinned-windows"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Atom Harness Phase 6 source and release policy."
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Verify the build inputs before new release evidence exists.",
    )
    arguments = parser.parse_args(argv)
    try:
        _require_files()
        git_surface = _check_git_surface()
        runtimes = _check_runtime_contracts()
        wiring = _check_runtime_wiring()
        desktop = _check_desktop_and_update()
        release = (
            {
                "desktop_release_evidence": "deferred-until-package-completes",
                "permissioned_hands_live_evidence": "deferred-until-package-completes",
                "permissioned_hands_certificate": "deferred-until-package-completes",
            }
            if arguments.source_only
            else _check_release_evidence_and_certificate()
        )
        ci = _check_ci()
        rust_crates = _check_rust_crate_sizes()
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
                "policy": "atom-harness-phase6-permissioned-hands",
                "verification_scope": (
                    "source-only" if arguments.source_only else "source-and-release"
                ),
                **git_surface,
                **runtimes,
                **wiring,
                **desktop,
                **release,
                **ci,
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
