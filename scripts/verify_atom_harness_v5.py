"""Fail-closed repository policy checks for Atom Harness Desktop Phase 5."""

from __future__ import annotations

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
    _check_ci_contract,
    _check_git_surface,
    _check_required_files,
    _check_runtime_declarations,
    _check_rust_crate_sizes,
    _load_json,
)
from verify_atom_harness_v4 import _check_v4_surface  # noqa: E402


V5_TEST = "tests/test_atom_harness_desktop_v5_integration.py"
V5_WORKFLOW = ".github/workflows/atom-harness-v5-ci.yml"
DESKTOP_ENTRYPOINT = "desktop/AtomHarness.Desktop/Program.cs"

REQUIRED_DESKTOP_FILES = (
    V5_WORKFLOW,
    "ATOM_HARNESS_DESKTOP.md",
    "global.json",
    "atom-harness-backend.spec",
    "atom-harness-desktop-architecture.json",
    "atom-harness-desktop-release-evidence.json",
    "atom_harness_desktop_backend.py",
    "lucerna-update.json",
    "scripts/build_atom_harness_desktop.ps1",
    "desktop/Directory.Build.props",
    "desktop/Directory.Packages.props",
    "desktop/AtomHarness.Desktop.Core/AtomHarness.Desktop.Core.csproj",
    "desktop/AtomHarness.Desktop.Core/CertifiedModelContract.cs",
    "desktop/AtomHarness.Desktop.Core/Integrity.cs",
    "desktop/AtomHarness.Desktop.Core/LucernaReleaseClient.cs",
    "desktop/AtomHarness.Desktop.Core/ReleaseManifest.cs",
    "desktop/AtomHarness.Desktop.Core/SafeUpdateInstaller.cs",
    "desktop/AtomHarness.Desktop.Core/UpdateContract.cs",
    "desktop/AtomHarness.Desktop/AtomHarness.Desktop.csproj",
    "desktop/AtomHarness.Desktop/BackendSupervisor.cs",
    "desktop/AtomHarness.Desktop/DesktopPaths.cs",
    "desktop/AtomHarness.Desktop/DesktopSettings.cs",
    "desktop/AtomHarness.Desktop/InstalledLayoutVerifier.cs",
    "desktop/AtomHarness.Desktop/MainForm.cs",
    "desktop/AtomHarness.Desktop/ModelProvisioner.cs",
    "desktop/AtomHarness.Desktop/ProcessJob.cs",
    "desktop/AtomHarness.Desktop/Program.cs",
    "desktop/AtomHarness.Desktop/SafeDiagnostics.cs",
    "desktop/AtomHarness.Updater/AtomHarness.Updater.csproj",
    "desktop/AtomHarness.Updater/Program.cs",
    "desktop/AtomHarness.Desktop.Tests/AtomHarness.Desktop.Tests.csproj",
    "desktop/packaging/AtomHarness.wxs",
    "desktop/packaging/PerUserHarvest.xslt",
    V5_TEST,
)


def _require_markers(path: str, markers: tuple[str, ...]) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    for marker in markers:
        if marker not in text:
            raise PolicyFailure(f"{path} is missing Phase 5 marker: {marker}")
    return text


def _check_desktop_surface() -> dict[str, Any]:
    for relative_path in REQUIRED_DESKTOP_FILES:
        path = ROOT / relative_path
        if not path.is_file() or path.is_symlink():
            raise PolicyFailure(f"required desktop file is absent: {relative_path}")

    architecture = _load_json("atom-harness-desktop-architecture.json")
    update = _load_json("lucerna-update.json")
    if (
        architecture.get("schema") != 1
        or architecture.get("product_phase") != 5
        or architecture.get("runtime") != "atom-harness-desktop-v5"
        or architecture.get("version") != "5.0.0"
        or architecture.get("desktop_entrypoint") != DESKTOP_ENTRYPOINT
    ):
        raise PolicyFailure("desktop architecture identity is invalid")
    installed = architecture.get("installed_runtime")
    shell = architecture.get("desktop_shell")
    model = architecture.get("model_provisioning")
    updates = architecture.get("updates")
    release = architecture.get("release")
    for name, value in (
        ("installed_runtime", installed),
        ("desktop_shell", shell),
        ("model_provisioning", model),
        ("updates", updates),
        ("release", release),
    ):
        if not isinstance(value, dict):
            raise PolicyFailure(f"desktop architecture section is absent: {name}")
    if (
        installed.get("authority_runtime") != "language-harness-v4"
        or installed.get("artifact_binding_marker") != "render_operator_surface"
        or installed.get("local_only") is not True
        or installed.get("cloud_allowed") is not False
    ):
        raise PolicyFailure("desktop authority boundary is invalid")
    for field in (
        "thin_shell_only",
        "atom_authority_unchanged",
        "per_user_install",
        "single_instance",
        "backend_job_object_kill_on_close",
        "graceful_ui_shutdown_first",
        "forced_process_tree_cleanup_fallback",
        "persistent_session_history",
        "real_artifact_side_view_embedded",
    ):
        if shell.get(field) is not True:
            raise PolicyFailure(f"desktop shell must require {field}")
    if shell.get("administrator_required") is not False:
        raise PolicyFailure("desktop packaging must remain per-user")
    for field in (
        "existing_model_discovery",
        "explicit_download_consent",
        "download_staged_outside_install_directory",
        "bytes_verified",
        "sha256_verified",
    ):
        if model.get(field) is not True:
            raise PolicyFailure(f"desktop model provisioning must require {field}")
    if model.get("weights_bundled") is not False:
        raise PolicyFailure("desktop package must not bundle model weights")
    for field in (
        "explicit_install_consent",
        "sha256_before_install",
        "staging_outside_install_directory",
        "wait_for_app_exit",
        "rollback_backup",
    ):
        if updates.get(field) is not True:
            raise PolicyFailure(f"desktop updates must require {field}")
    if updates.get("silent_install") is not False:
        raise PolicyFailure("desktop updates must never install silently")
    if release.get("integration_test") != V5_TEST:
        raise PolicyFailure("desktop integration-test declaration is invalid")

    policy = update.get("policy")
    if (
        update.get("schema") != 1
        or update.get("app_id") != "com.lucernalabs.atom-harness"
        or update.get("app_name") != "Atom Harness"
        or update.get("platform") != "windows-x64"
        or update.get("current_version") != "5.0.0"
        or update.get("runtime") != "lucerna-release-client-v1"
        or not str(update.get("feed_url", "")).startswith("https://")
        or not isinstance(policy, dict)
    ):
        raise PolicyFailure("lucerna-update.json identity is invalid")
    for field in (
        "explicit_user_consent_required",
        "artifact_sha256_required",
        "stage_outside_install_directory",
        "replace_only_after_app_exit",
        "rollback_backup_required",
    ):
        if policy.get(field) is not True:
            raise PolicyFailure(f"update policy must require {field}")
    if policy.get("automatic_download") is not False:
        raise PolicyFailure("updates must not download automatically")
    if policy.get("automatic_install") is not False:
        raise PolicyFailure("updates must not install automatically")

    _require_markers(
        DESKTOP_ENTRYPOINT,
        (
            "atom-harness-desktop-v5",
            "ATOM_HARNESS_OPERATOR_RUNTIME",
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
            "ATOM_HARNESS_OPERATOR_UI_RUNTIME",
            "render_operator_surface",
        ),
    )
    _require_markers(
        "atom_harness_desktop_backend.py",
        (
            "ATOM_HARNESS_DESKTOP_BACKEND_RUNTIME",
            "ATOM_HARNESS_BUNDLED_CAUSAL_RUNTIME",
            "_bind_bundled_causal_memory",
            "atom_causal_memory.RELEASE_BINARY = binary",
            "atom-causal-memory.exe",
            "atom_harness_operator_server",
            "operator_main",
        ),
    )
    _require_markers(
        "desktop/AtomHarness.Desktop/MainForm.cs",
        (
            "confirmUpdate",
            "confirmInstall",
            "LucernaReleaseClient",
            "DownloadAndVerifyAsync",
            "RequestGracefulShutdownAsync",
            "AtomHarness.Updater.exe",
        ),
    )
    _require_markers(
        "desktop/AtomHarness.Desktop/ProcessJob.cs",
        (
            "JobObjectLimitKillOnJobClose",
            "AssignProcessToJobObject",
        ),
    )
    _require_markers(
        "desktop/AtomHarness.Desktop.Core/LucernaReleaseClient.cs",
        (
            "Uri.UriSchemeHttps",
            "SHA256",
            "ArtifactSha256",
            "File.Move(partialPath, finalPath, true)",
        ),
    )
    installer_text = _require_markers(
        "desktop/AtomHarness.Desktop.Core/SafeUpdateInstaller.cs",
        (
            "WaitForExitAsync",
            "ReleaseManifest.Load",
            "VerifyDirectoryAsync",
            "previous-",
            "Directory.Move(installDirectory, backupDirectory)",
        ),
    )
    if installer_text.index("await WaitForExitAsync") > installer_text.index(
        "Directory.Move(installDirectory, backupDirectory)"
    ):
        raise PolicyFailure("update replacement occurs before the app exits")
    _require_markers(
        "scripts/build_atom_harness_desktop.ps1",
        (
            "PyInstaller",
            "cargo build",
            "atom_causal_memory_rust",
            "dotnet publish",
            "--locked-mode",
            "atom-harness-release-manifest-v1",
            "Compress-Archive",
            "heat.exe",
            "candle.exe",
            "light.exe",
        ),
    )
    _require_markers(
        "desktop/packaging/AtomHarness.wxs",
        (
            'InstallScope="perUser"',
            "LocalAppDataFolder",
            "StartMenuShortcut",
            "DesktopShortcut",
        ),
    )
    _require_markers(
        V5_TEST,
        (
            "ATOM_HARNESS_WIKI_RUNTIME",
            "ATOM_HARNESS_RAG_RUNTIME",
            "ATOM_HARNESS_OPERATOR_UI_RUNTIME",
            "render_operator_surface",
            "test_operator_runtime_wires_wiki_rag_api_and_real_side_view",
        ),
    )

    package_versions = (ROOT / "desktop" / "Directory.Packages.props").read_text(
        encoding="utf-8"
    )
    for marker in (
        'Microsoft.Web.WebView2" Version="1.0.4078.44"',
        'Microsoft.NET.Test.Sdk" Version="18.8.1"',
        'MSTest.TestAdapter" Version="4.3.3"',
        'MSTest.TestFramework" Version="4.3.3"',
    ):
        if marker not in package_versions:
            raise PolicyFailure(f".NET dependency is not pinned: {marker}")
    requirements = (
        (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    )
    if "pyinstaller==6.21.0" not in requirements:
        raise PolicyFailure("PyInstaller is not pinned")

    dotnet_sdk = _load_json("global.json").get("sdk")
    if (
        not isinstance(dotnet_sdk, dict)
        or dotnet_sdk.get("version") != "9.0.305"
        or dotnet_sdk.get("rollForward") != "disable"
        or dotnet_sdk.get("allowPrerelease") is not False
    ):
        raise PolicyFailure("The desktop .NET SDK boundary is not exact")

    workflow = _require_markers(
        V5_WORKFLOW,
        (
            "Verify Desktop Phase 5 on Windows",
            "python scripts/verify_atom_harness_v5.py",
            "test_atom_harness_desktop_v5_integration.py",
            "dotnet restore",
            "--locked-mode",
            "dotnet format",
            "dotnet build",
            "dotnet test",
            "cargo clippy",
            "-D warnings",
            "cargo test",
            "cargo build",
            "PyInstaller",
            "atom-causal-memory.exe",
        ),
    )
    expected_actions = (
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "actions/setup-dotnet@a98b56852c35b8e3190ac28c8c2271da59106c68",
    )
    for action in expected_actions:
        if action not in workflow:
            raise PolicyFailure(f"Desktop Phase 5 CI action is not pinned: {action}")
    if 'dotnet-version: "9.0.305"' not in workflow:
        raise PolicyFailure("Desktop Phase 5 CI must pin .NET SDK 9.0.305")
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not re.search(
            r"@[0-9a-f]{40}(?:\s|$)",
            stripped,
        ):
            raise PolicyFailure(
                f"Desktop Phase 5 CI has an unpinned action: {stripped}"
            )

    evidence_path = ROOT / "atom-harness-desktop-release-evidence.json"
    evidence_status = "pending-local-package"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        package = evidence.get("package")
        installation = evidence.get("installation")
        runtime = evidence.get("interactive_runtime")
        update_boundary = evidence.get("update_boundary")
        if (
            evidence.get("schema") != 1
            or evidence.get("runtime") != "atom-harness-desktop-release-evidence-v1"
            or evidence.get("version") != "5.0.0"
            or evidence.get("platform") != "windows-x64"
            or evidence.get("all_checks_passed") is not True
            or not isinstance(package, dict)
            or not isinstance(installation, dict)
            or not isinstance(runtime, dict)
            or not isinstance(update_boundary, dict)
        ):
            raise PolicyFailure("desktop release evidence is invalid")
        for field in ("portable_zip_sha256", "msi_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(package.get(field, ""))):
                raise PolicyFailure(f"desktop release evidence {field} is invalid")
        for field in ("portable_zip_bytes", "msi_bytes", "file_count"):
            if not isinstance(package.get(field), int) or package[field] <= 0:
                raise PolicyFailure(f"desktop release evidence {field} is invalid")
        if (
            installation.get("passed") is not True
            or installation.get("full_release_manifest_verified") is not True
            or installation.get("desktop_shortcut") is not True
            or installation.get("start_menu_shortcut") is not True
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(installation.get("verification_report_sha256", "")),
            )
        ):
            raise PolicyFailure("desktop installed-layout evidence is invalid")
        if (
            runtime.get("real_artifact_side_view_visible") is not True
            or runtime.get("failed_attempt_recovered") is not True
            or runtime.get("completed_artifact_recovered_after_restart") is not True
            or runtime.get("graceful_process_tree_shutdown_verified") is not True
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(runtime.get("transaction_sha256", "")),
            )
        ):
            raise PolicyFailure("desktop interactive runtime evidence is invalid")
        for field in (
            "explicit_user_consent_required",
            "sha256_required",
            "external_staging_required",
            "replace_only_after_exit",
            "rollback_backup_required",
        ):
            if update_boundary.get(field) is not True:
                raise PolicyFailure(
                    f"desktop release evidence update boundary is missing {field}"
                )
        if (
            update_boundary.get("silent_download") is not False
            or update_boundary.get("silent_install") is not False
        ):
            raise PolicyFailure("desktop release evidence permits a silent update")
        evidence_status = "verified-local-package"

    return {
        "desktop_runtime": architecture["runtime"],
        "desktop_version": architecture["version"],
        "desktop_integration_test": V5_TEST,
        "desktop_release_evidence": evidence_status,
        "update_runtime": update["runtime"],
    }


def main() -> int:
    if (
        _load_json("ai-runtime-registry.json").get("active_runtime")
        == "language-harness-v5"
    ):
        from verify_atom_harness_v6 import main as verify_current_phase

        return verify_current_phase()
    try:
        _check_required_files()
        git_surface = _check_git_surface()
        declarations = _check_runtime_declarations()
        ci_contract = _check_ci_contract()
        v4 = _check_v4_surface()
        rust_crates = _check_rust_crate_sizes()
        desktop = _check_desktop_surface()
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
                "policy": "atom-harness-v5-desktop",
                **git_surface,
                **declarations,
                **ci_contract,
                **v4,
                **desktop,
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
