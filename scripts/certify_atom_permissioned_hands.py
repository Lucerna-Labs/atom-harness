"""Create a source-bound Phase 6 adversarial permission certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "atom-permissioned-hands-certification-v1"
TEST_MODULES = (
    "tests.test_atom_permissioned_hands",
    "tests.test_atom_permissioned_hands_integration",
    "tests.test_atom_harness_desktop_v6_integration",
)
SOURCE_FILES = (
    "ai-artifact-side-view.json",
    "ai-provider-fabric.json",
    "ai-run-transaction.json",
    "ai-runtime-knowledge.json",
    "ai-runtime-registry.json",
    "ai-tool-fabric.json",
    "atom-harness-desktop-architecture.json",
    "atom-harness-desktop-release-evidence.json",
    "atom-language-harness-architecture.json",
    "atom_harness_knowledge.py",
    "atom_harness_operator.py",
    "atom_harness_operator_server.py",
    "atom_harness_operator_ui.py",
    "atom_llm_protocol.py",
    "atom_provider_fabric.py",
    "atom_run_transaction.py",
    "atom_tool_capabilities.py",
    "atom_tool_fabric.py",
    "atom_tool_protocol.py",
    "atom_tool_side_view.py",
    "lucerna-update.json",
    "scripts/certify_atom_permissioned_hands.py",
    "scripts/verify_atom_harness_v6.py",
    "tests/test_atom_permissioned_hands.py",
    "tests/test_atom_permissioned_hands_integration.py",
    "tests/test_atom_harness_desktop_v6_integration.py",
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


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and bind the Phase 6 permissioned-hands adversarial suite."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New report directory below local-results. A timestamped directory is used by default.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    local_results = (ROOT / "local-results").resolve()
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else local_results / f"phase6-permission-certification-{stamp}"
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
    reported_command = ["python", "-m", "unittest", *TEST_MODULES, "-v"]
    started = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    finished = datetime.now(UTC)
    combined_output = completed.stdout + completed.stderr
    (output_root / "test-output.txt").write_text(combined_output, encoding="utf-8")

    source_hashes = {
        relative: _normalized_sha256(ROOT / relative) for relative in SOURCE_FILES
    }
    source_manifest_hash = _canonical_hash(source_hashes)
    expected_tests = (
        "test_exact_denial_and_tamper_block_all_side_effects",
        "test_untrusted_model_candidate_is_normalized_before_permission",
        "test_real_code_simulation_and_document_workflow_is_transaction_bound",
        "test_indirect_injection_is_tainted_and_cannot_escape_workspace",
        "test_hash_guard_fails_closed_after_approval_time_drift",
        "test_phase6_wires_wiki_rag_permission_hands_and_both_side_views",
        "test_desktop_chain_exercises_wiki_rag_permission_and_tool_side_view",
        "test_release_evidence_binds_live_model_denial_and_recovery",
        "test_web_fetch_connects_only_to_the_permission_bound_address",
        "test_process_output_is_streamed_into_a_bounded_preview",
        "test_process_executable_hash_drift_fails_before_spawn",
        "test_process_timeout_terminates_parent_and_child",
    )
    observed = {name: name in combined_output for name in expected_tests}
    all_passed = completed.returncode == 0 and all(observed.values())
    report_core = {
        "schema": 1,
        "runtime": RUNTIME,
        "mode": "scripted-adversarial-and-loopback-integration",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "python": sys.version.split()[0],
        "command": reported_command,
        "return_code": completed.returncode,
        "source_hash_normalization": "utf-8-lf-v1",
        "source_manifest_hash": source_manifest_hash,
        "source_files_sha256": source_hashes,
        "checks": {
            "exact_denial_blocks_side_effects": observed[expected_tests[0]],
            "tampered_manifest_blocks_side_effects": observed[expected_tests[0]],
            "model_candidate_reduced_before_permission": observed[expected_tests[1]],
            "candidate_normalizations_visible": observed[expected_tests[1]],
            "permission_replay_blocked": observed[expected_tests[2]],
            "real_code_simulation_and_document_passed": observed[expected_tests[2]],
            "tool_results_quarantined": observed[expected_tests[2]],
            "indirect_injection_tainted": observed[expected_tests[3]],
            "workspace_escape_blocked": observed[expected_tests[3]],
            "toctou_drift_blocked": observed[expected_tests[4]],
            "wiki_graph_and_rag_runtime_wired": observed[expected_tests[5]],
            "evidence_side_view_runtime_wired": observed[expected_tests[5]],
            "tool_side_view_runtime_wired": observed[expected_tests[5]],
            "authenticated_loopback_permission_flow_passed": observed[
                expected_tests[5]
            ],
            "desktop_chain_passed": observed[expected_tests[6]],
            "live_model_and_denial_release_evidence_bound": observed[expected_tests[7]],
            "public_web_connection_pinned_to_permission_bound_address": observed[
                expected_tests[8]
            ],
            "process_output_retention_bounded": observed[expected_tests[9]],
            "process_executable_hash_drift_blocked": observed[expected_tests[10]],
            "process_timeout_tree_cleanup_passed": observed[expected_tests[11]],
        },
        "all_checks_passed": all_passed,
        "claim_boundary": (
            "This certificate covers the deterministic adversarial suite and the "
            "real local loopback integration using schema-valid scripted language "
            "outputs. It proves that the tested permission and transaction paths "
            "fail closed. It does not prove universal prompt-injection resistance "
            "or unattended autonomy safety."
        ),
    }
    report = {**report_core, "report_hash": _canonical_hash(report_core)}
    report_path = output_root / "atom-permissioned-hands-certification.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "passed": all_passed,
        "runtime": RUNTIME,
        "report": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_manifest_hash": source_manifest_hash,
        "elapsed_seconds": report_core["elapsed_seconds"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
