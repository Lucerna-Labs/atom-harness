"""Create a source-bound Phase 7 universal-knowledge certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "atom-universal-knowledge-certification-v1"
TEST_MODULES = ("tests.test_atom_universal_knowledge_integration",)
SOURCE_FILES = (
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
    "atom_harness_experiment.py",
    "atom_harness_desktop_backend.py",
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
    "desktop/AtomHarness.Desktop/BackendSupervisor.cs",
    "desktop/AtomHarness.Desktop/AtomHarness.Desktop.csproj",
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
    "lucerna-update.json",
    ".github/workflows/atom-harness-v7-ci.yml",
    "scripts/build_atom_harness_desktop.ps1",
    "scripts/certify_atom_universal_knowledge.py",
    "scripts/verify_atom_harness_v7.py",
    "tests/test_atom_harness_desktop_v7_integration.py",
    "tests/test_atom_universal_knowledge_integration.py",
)


def _normalized_sha256(path: Path) -> str:
    data = (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and bind the Phase 7 universal-knowledge suite."
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Also publish atom-universal-knowledge-certification.json at project root.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    local_results = (ROOT / "local-results").resolve()
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else local_results / f"phase7-universal-certification-{stamp}"
    )
    try:
        output_root.relative_to(local_results)
    except ValueError:
        print("output root must stay below local-results", file=sys.stderr)
        return 2
    if output_root.exists():
        print("output root already exists", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True)

    command = [sys.executable, "-m", "unittest", *TEST_MODULES, "-v"]
    started = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = datetime.now(UTC)
    output = completed.stdout + completed.stderr
    (output_root / "test-output.txt").write_text(output, encoding="utf-8")

    source_hashes = {
        relative: _normalized_sha256(ROOT / relative) for relative in SOURCE_FILES
    }
    expected_tests = (
        "test_phase7_wires_multidisciplinary_graph_rag_and_real_side_view",
        "test_every_declared_domain_is_seeded_routable_and_citable",
        "test_fiction_interpretation_craft_and_science_never_collapse",
        "test_threads_intersections_preload_and_injection_boundary_are_bound",
        "test_pack_tampering_and_unknown_knowledge_fail_closed",
        "test_phase7_preserves_phase6_permissioned_hands_and_operator_view",
    )
    observed = {name: name in output for name in expected_tests}
    all_passed = completed.returncode == 0 and all(observed.values())
    report_core = {
        "schema": 1,
        "runtime": RUNTIME,
        "mode": "deterministic-multidisciplinary-and-permission-regression",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "python": sys.version.split()[0],
        "command": ["python", "-m", "unittest", *TEST_MODULES, "-v"],
        "return_code": completed.returncode,
        "source_hash_normalization": "utf-8-lf-v1",
        "source_manifest_hash": _canonical_hash(source_hashes),
        "source_files_sha256": source_hashes,
        "knowledge_pack": {
            "pack_id": "atom-universal-foundation-v1",
            "domain_count": 15,
            "claim_count": 45,
            "source_count": 22,
            "not_exhaustive_human_knowledge": True,
        },
        "checks": {
            "all_declared_domains_seeded": observed[expected_tests[1]],
            "all_declared_domains_routable": observed[expected_tests[1]],
            "every_retrieved_claim_citable": observed[expected_tests[1]],
            "wiki_graph_and_rag_runtime_wired": observed[expected_tests[0]],
            "real_artifact_side_view_bound": observed[expected_tests[0]],
            "immutable_transaction_snapshot_bound": observed[expected_tests[0]],
            "formal_empirical_fictional_and_craft_types_separate": observed[
                expected_tests[2]
            ],
            "fiction_cannot_enter_science_lane": observed[expected_tests[2]],
            "query_threads_and_intersections_emerge": observed[expected_tests[3]],
            "neighbor_preload_declared": observed[expected_tests[3]],
            "source_injection_cannot_grant_permission": observed[expected_tests[3]],
            "pack_tamper_fails_closed": observed[expected_tests[4]],
            "unknown_knowledge_abstains": observed[expected_tests[4]],
            "phase6_permissioned_hands_preserved": observed[expected_tests[5]],
            "operator_and_tool_side_views_preserved": observed[expected_tests[5]],
            "ornith_1_0_capability_floor_preserved": observed[expected_tests[5]],
        },
        "all_checks_passed": all_passed,
        "claim_boundary": (
            "This certificate proves the checked Phase 7 foundation pack, "
            "runtime wiring, epistemic separation, immutable retrieval, side "
            "view, transaction binding, tamper rejection, and Phase 6 permission "
            "regression. It establishes domain-level seed coverage, not exhaustive "
            "human knowledge, perfect retrieval, medical authority, universal "
            "prompt-injection resistance, or unattended autonomy safety."
        ),
    }
    report = {**report_core, "report_hash": _canonical_hash(report_core)}
    report_path = output_root / "atom-universal-knowledge-certification.json"
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    if arguments.promote and all_passed:
        (ROOT / "atom-universal-knowledge-certification.json").write_text(
            encoded,
            encoding="utf-8",
        )
    result = {
        "passed": all_passed,
        "runtime": RUNTIME,
        "report": str(report_path),
        "promoted": arguments.promote and all_passed,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_manifest_hash": report_core["source_manifest_hash"],
        "elapsed_seconds": report_core["elapsed_seconds"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
