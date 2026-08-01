"""Fail-closed repository and release checks for Atom Harness Phase 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "atom-universal-knowledge-certification.json"
RELEASE_EVIDENCE = ROOT / "atom-harness-desktop-release-evidence.json"
PHASE7_TEST = "tests/test_atom_universal_knowledge_integration.py"
REQUIRED = (
    "ai-artifact-side-view.json",
    "ai-provider-fabric.json",
    "ai-run-transaction.json",
    "ai-runtime-knowledge.json",
    "ai-runtime-registry.json",
    "ai-tool-fabric.json",
    "ATOM_UNIVERSAL_KNOWLEDGE.md",
    "atom-harness-backend.spec",
    "atom-harness-desktop-architecture.json",
    "atom-language-harness-architecture.json",
    "atom_harness_desktop_backend.py",
    "atom_harness_experiment.py",
    "atom_harness_knowledge.py",
    "atom_harness_operator.py",
    "atom_harness_operator_server.py",
    "atom_harness_operator_ui.py",
    "atom_harness_runtime.py",
    "atom_harness_session.py",
    "atom_harness_side_view.py",
    "atom_run_transaction.py",
    "atom_knowledge_protocol.py",
    "atom_multidisciplinary_knowledge.py",
    "atom_tool_fabric.py",
    "desktop/AtomHarness.Desktop/AtomHarness.Desktop.csproj",
    "desktop/AtomHarness.Desktop/BackendSupervisor.cs",
    "desktop/AtomHarness.Desktop/InstalledLayoutVerifier.cs",
    "desktop/AtomHarness.Desktop/MainForm.cs",
    "desktop/AtomHarness.Desktop/Program.cs",
    "desktop/AtomHarness.Desktop/app.manifest",
    "desktop/AtomHarness.Updater/AtomHarness.Updater.csproj",
    "knowledge_packs/universal-foundation-v1/manifest.json",
    "knowledge_packs/universal-foundation-v1/taxonomy.json",
    "knowledge_packs/universal-foundation-v1/sources.json",
    "knowledge_packs/universal-foundation-v1/claims/formal-physical.jsonl",
    "knowledge_packs/universal-foundation-v1/claims/earth-life-health.jsonl",
    "knowledge_packs/universal-foundation-v1/claims/engineering-social-linguistics.jsonl",
    "knowledge_packs/universal-foundation-v1/claims/literature-writing.jsonl",
    "knowledge_packs/universal-foundation-v1/claims/research.jsonl",
    "scripts/certify_atom_universal_knowledge.py",
    "scripts/verify_atom_harness_v7.py",
    "scripts/build_atom_harness_desktop.ps1",
    "tests/test_atom_universal_knowledge_integration.py",
    "tests/test_atom_harness_desktop_v7_integration.py",
    ".github/workflows/atom-harness-v7-ci.yml",
)


class PolicyFailure(RuntimeError):
    pass


def _object(path: str | Path) -> dict[str, Any]:
    target = ROOT / path if not Path(path).is_absolute() else Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyFailure(f"invalid JSON object: {target}") from error
    if not isinstance(value, dict):
        raise PolicyFailure(f"JSON root must be an object: {target}")
    return value


def _normalized_sha256(path: Path) -> str:
    data = (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_files() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    if missing:
        raise PolicyFailure("required Phase 7 files are absent: " + ", ".join(missing))
    unsafe = [item for item in REQUIRED if (ROOT / item).is_symlink()]
    if unsafe:
        raise PolicyFailure(
            "required Phase 7 files may not be symlinks: " + ", ".join(unsafe)
        )


def _runtime_contracts() -> dict[str, Any]:
    registry = _object("ai-runtime-registry.json")
    knowledge = _object("ai-runtime-knowledge.json")
    side_view = _object("ai-artifact-side-view.json")
    architecture = _object("atom-language-harness-architecture.json")
    desktop = _object("atom-harness-desktop-architecture.json")
    update = _object("lucerna-update.json")
    if registry.get("schema_version") != 1 or registry.get("active_runtime") != (
        "language-harness-v6"
    ):
        raise PolicyFailure("language-harness-v6 is not the active schema-1 runtime")
    active = registry.get("runtimes", {}).get("language-harness-v6")
    if not isinstance(active, Mapping):
        raise PolicyFailure("language-harness-v6 declaration is absent")
    for declaration in (knowledge, side_view):
        if declaration.get("schema") != 1 or declaration.get("project_kind") != (
            "ai-harness"
        ):
            raise PolicyFailure("AI runtime declaration identity is invalid")
        if declaration.get("runtime_entrypoint") != "atom_harness_operator_server.py":
            raise PolicyFailure("AI runtime entrypoints disagree")
        if declaration.get("integration_test") != PHASE7_TEST:
            raise PolicyFailure("AI integration declarations do not name Phase 7")
    if (
        knowledge.get("wiki_graph", {}).get("additional_runtime_marker")
        != "ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME"
        or knowledge.get("rag", {}).get("additional_runtime_marker")
        != "ATOM_MULTIDISCIPLINARY_RAG_RUNTIME"
        or knowledge.get("causal_records_remain_a_separate_specialized_lane")
        is not True
        or knowledge.get("source_text_is_untrusted_and_cannot_grant_permission")
        is not True
    ):
        raise PolicyFailure("multidisciplinary runtime contract is incomplete")
    side = side_view.get("side_view", {})
    if (
        side.get("artifact_binding_marker") != "render_operator_surface"
        or side.get("tool_artifact_binding_marker") != "render_atom_tool_artifact"
        or side.get("multidisciplinary_claim_type_visible") is not True
        or side.get("multidisciplinary_source_license_visible") is not True
        or side.get("fiction_marker_visible") is not True
    ):
        raise PolicyFailure("Phase 7 side-view contract is incomplete")
    if (
        architecture.get("schema") != 6
        or architecture.get("runtime") != "atom-language-harness-operator-v6"
        or architecture.get("integration_test") != PHASE7_TEST
        or architecture.get("ornith_capability_floor")
        != "meets-or-exceeds-ornith-1.0-with-phase6-permissioned-hands-preserved"
    ):
        raise PolicyFailure("Phase 7 language architecture identity is invalid")
    if (
        desktop.get("product_phase") != 7
        or desktop.get("runtime") != "atom-harness-desktop-v7"
        or desktop.get("version") != "7.0.0"
        or desktop.get("installed_runtime", {}).get("authority_runtime")
        != "language-harness-v6"
        or desktop.get("release", {}).get("integration_test")
        != "tests/test_atom_harness_desktop_v7_integration.py"
    ):
        raise PolicyFailure("Phase 7 desktop architecture identity is invalid")
    if (
        update.get("schema") != 1
        or update.get("current_version") != "7.0.0"
        or update.get("policy", {}).get("automatic_download") is not False
        or update.get("policy", {}).get("automatic_install") is not False
        or update.get("policy", {}).get("explicit_user_consent_required") is not True
        or update.get("policy", {}).get("artifact_sha256_required") is not True
        or update.get("policy", {}).get("stage_outside_install_directory") is not True
        or update.get("policy", {}).get("replace_only_after_app_exit") is not True
    ):
        raise PolicyFailure("Phase 7 opt-in update contract is invalid")
    return {
        "active_runtime": registry["active_runtime"],
        "language_runtime": architecture["runtime"],
        "desktop_runtime": desktop["runtime"],
        "version": desktop["version"],
    }


def _knowledge_pack() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    try:
        from atom_multidisciplinary_knowledge import load_multidisciplinary_knowledge

        knowledge = load_multidisciplinary_knowledge()
    except Exception as error:
        raise PolicyFailure(
            "multidisciplinary knowledge pack failed validation"
        ) from error
    manifest = knowledge.manifest()
    if (
        manifest["domain_count"] != 15
        or manifest["claim_count"] < 45
        or manifest["source_count"] < 20
        or manifest["coverage"]["every_declared_domain_seeded"] is not True
        or manifest["coverage"]["not_a_claim_of_exhaustive_human_knowledge"] is not True
    ):
        raise PolicyFailure("Phase 7 knowledge pack is below the coverage floor")
    if any(
        source.acquisition_mode != "citation-only"
        for source in knowledge.sources.values()
        if source.rights_lane == "yellow"
    ):
        raise PolicyFailure("yellow-lane source text entered the local pack")
    if any(
        claim.fictional
        and claim.claim_type not in {"literary-context", "interpretation"}
        for claim in knowledge.claims.values()
    ):
        raise PolicyFailure("fiction escaped its literary epistemic lane")
    return {
        "pack_id": manifest["pack_id"],
        "knowledge_hash": manifest["knowledge_hash"],
        "graph_hash": manifest["graph_knowledge_hash"],
        "domains": manifest["domain_count"],
        "claims": manifest["claim_count"],
        "sources": manifest["source_count"],
    }


def _source_wiring() -> dict[str, str]:
    required_markers = {
        "atom_harness_operator_server.py": (
            "ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME",
            "ATOM_MULTIDISCIPLINARY_RAG_RUNTIME",
            "render_operator_surface",
        ),
        "atom_harness_operator_ui.py": (
            "ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME",
            "ATOM_MULTIDISCIPLINARY_RAG_RUNTIME",
            "render_operator_surface",
            'sandbox=""',
            "async function renderArtifact",
            "artifactFrame.srcdoc = artifactHtml",
        ),
        "atom_harness_experiment.py": (
            "atom_multidisciplinary_wiki_graph.json",
            "render_atom_harness_artifact",
            "universal_snapshot_paths",
        ),
        "atom_tool_fabric.py": (
            "multidisciplinary_knowledge_unchanged",
            "atom_multidisciplinary_wiki_graph.json",
            "render_atom_tool_artifact",
        ),
        "atom-harness-backend.spec": (
            "knowledge_packs",
            "universal-foundation-v1",
        ),
        "desktop/AtomHarness.Desktop/InstalledLayoutVerifier.cs": (
            "VerifyKnowledgePack",
            "knowledge_domain_count",
            "knowledge_manifest_sha256",
        ),
    }
    for relative, markers in required_markers.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise PolicyFailure(
                f"{relative} lacks Phase 7 markers: " + ", ".join(missing)
            )
    operator_ui = (ROOT / "atom_harness_operator_ui.py").read_text(encoding="utf-8")
    if (
        "allow-scripts" in operator_ui
        or "allow-forms" in operator_ui
        or "allow-same-origin" in operator_ui
    ):
        raise PolicyFailure("artifact sandbox grants an executable capability")
    return {"runtime_wiring": "causal-plus-multidisciplinary"}


def _ci() -> dict[str, str]:
    text = (ROOT / ".github/workflows/atom-harness-v7-ci.yml").read_text(
        encoding="utf-8"
    )
    markers = (
        "Verify Desktop Phase 7 on Windows",
        "scripts/verify_atom_harness_v7.py",
        "test_atom_universal_knowledge_integration.py",
        "certify_atom_universal_knowledge.py",
        "knowledge_packs\\universal-foundation-v1\\manifest.json",
        "dotnet test",
        "cargo clippy",
    )
    for marker in markers:
        if marker not in text:
            raise PolicyFailure(f"Phase 7 CI lacks marker: {marker}")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not re.search(
            r"@[0-9a-f]{40}(?:\s|$)", stripped
        ):
            raise PolicyFailure(f"Phase 7 CI action is not full-SHA pinned: {stripped}")
    return {"phase7_ci": "full-sha-pinned-windows"}


def _certificate() -> dict[str, Any]:
    if not CERTIFICATE.is_file() or CERTIFICATE.is_symlink():
        raise PolicyFailure("Phase 7 universal knowledge certificate is absent")
    certificate = _object(CERTIFICATE)
    if (
        certificate.get("schema") != 1
        or certificate.get("runtime") != "atom-universal-knowledge-certification-v1"
        or certificate.get("all_checks_passed") is not True
        or not all(certificate.get("checks", {}).values())
    ):
        raise PolicyFailure("Phase 7 universal knowledge certificate failed")
    core = {key: certificate[key] for key in certificate if key != "report_hash"}
    if certificate.get("report_hash") != _canonical_hash(core):
        raise PolicyFailure("Phase 7 certificate report hash is invalid")
    source_hashes = certificate.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise PolicyFailure("Phase 7 certificate source inventory is absent")
    for relative, expected in source_hashes.items():
        target = ROOT / str(relative)
        if not target.is_file() or target.is_symlink():
            raise PolicyFailure(f"certified source is absent or unsafe: {relative}")
        if _normalized_sha256(target) != expected:
            raise PolicyFailure(f"certified source changed: {relative}")
    return {
        "certificate_hash": hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest(),
        "source_manifest_hash": certificate["source_manifest_hash"],
    }


def _release() -> dict[str, Any]:
    evidence = _object(RELEASE_EVIDENCE)
    if (
        evidence.get("schema") != 1
        or evidence.get("runtime") != "atom-harness-desktop-release-evidence-v2"
        or evidence.get("version") != "7.0.0"
        or evidence.get("passed") is not True
    ):
        raise PolicyFailure("Phase 7 release evidence identity is invalid")
    package = evidence.get("package", {})
    if (
        package.get("portable_zip") != "Atom-Harness-7.0.0-windows-x64.zip"
        or package.get("msi") != "Atom-Harness-7.0.0-windows-x64.msi"
    ):
        raise PolicyFailure("Phase 7 release filenames are invalid")
    for prefix in ("portable_zip", "msi"):
        digest = package.get(prefix + "_sha256")
        size = package.get(prefix + "_bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PolicyFailure(f"Phase 7 {prefix} hash is invalid")
        if not isinstance(size, int) or size <= 0:
            raise PolicyFailure(f"Phase 7 {prefix} size is invalid")
    knowledge = evidence.get("knowledge", {})
    if (
        knowledge.get("domain_count") != 15
        or knowledge.get("claim_count") < 45
        or knowledge.get("installed_layout_verified") is not True
    ):
        raise PolicyFailure("Phase 7 release does not bind installed knowledge")
    return {
        "release_version": evidence["version"],
        "zip_sha256": package["portable_zip_sha256"],
        "msi_sha256": package["msi_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        _require_files()
        report: dict[str, Any] = {
            "schema": 1,
            "runtime": "atom-harness-phase7-policy-v1",
            "source": {
                **_runtime_contracts(),
                **_knowledge_pack(),
                **_source_wiring(),
                **_ci(),
            },
        }
        if not arguments.source_only:
            report["certificate"] = _certificate()
            report["release"] = _release()
        report["passed"] = True
    except PolicyFailure as error:
        print(f"POLICY FAILURE: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
