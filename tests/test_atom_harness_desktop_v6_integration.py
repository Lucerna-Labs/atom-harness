from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.test_atom_permissioned_hands_integration import (
    PermissionedHandsIntegrationTests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AtomHarnessDesktopV6IntegrationTests(unittest.TestCase):
    def test_desktop_entrypoint_exposes_permissioned_hands_and_opt_in_updates(
        self,
    ) -> None:
        architecture = json.loads(
            (PROJECT_ROOT / "atom-harness-desktop-architecture.json").read_text(
                encoding="utf-8"
            )
        )
        update = json.loads(
            (PROJECT_ROOT / "lucerna-update.json").read_text(encoding="utf-8")
        )
        program = (PROJECT_ROOT / "desktop/AtomHarness.Desktop/Program.cs").read_text(
            encoding="utf-8"
        )
        supervisor = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/BackendSupervisor.cs"
        ).read_text(encoding="utf-8")
        main_form = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/MainForm.cs"
        ).read_text(encoding="utf-8")
        backend = (PROJECT_ROOT / "atom_harness_desktop_backend.py").read_text(
            encoding="utf-8"
        )
        package_builder = (
            PROJECT_ROOT / "scripts/build_atom_harness_desktop.ps1"
        ).read_text(encoding="utf-8")

        self.assertEqual(architecture["product_phase"], 7)
        self.assertEqual(architecture["runtime"], "atom-harness-desktop-v7")
        self.assertEqual(architecture["version"], "7.0.0")
        self.assertEqual(
            architecture["installed_runtime"]["authority_runtime"],
            "language-harness-v6",
        )
        self.assertTrue(
            architecture["permissioned_hands"][
                "operator_permission_required_for_every_execution"
            ]
        )
        self.assertTrue(
            architecture["permissioned_hands"]["candidate_normalizations_user_visible"]
        )
        self.assertTrue(
            architecture["desktop_shell"][
                "exact_action_manifest_visible_before_approval"
            ]
        )
        self.assertFalse(architecture["installed_runtime"]["cloud_allowed"])
        self.assertEqual(update["schema"], 1)
        self.assertEqual(update["current_version"], "7.0.0")
        self.assertTrue(update["policy"]["explicit_user_consent_required"])
        self.assertTrue(update["policy"]["artifact_sha256_required"])
        self.assertTrue(update["policy"]["stage_outside_install_directory"])
        self.assertTrue(update["policy"]["replace_only_after_app_exit"])
        self.assertIn("atom-harness-desktop-v7", program)
        self.assertIn("atom-harness-operator-loopback-server-v3", supervisor)
        self.assertIn("atom-permissioned-hands-fabric-v1", supervisor)
        self.assertIn("atom-permissioned-hands-side-view-v1", supervisor)
        self.assertIn("atom_harness_operator_server", backend)
        self.assertIn("atom-harness-desktop-backend-v7", backend)
        self.assertIn("cargo build", package_builder)
        self.assertIn("confirmUpdate", main_form)
        self.assertIn("Atom Harness Desktop 7", main_form)
        self.assertIn("permission registry", main_form)
        self.assertIn("confirmInstall", main_form)
        self.assertIn("DownloadAndVerifyAsync", main_form)
        self.assertIn("RequestGracefulShutdownAsync", main_form)

    def test_release_evidence_binds_live_model_denial_and_recovery(self) -> None:
        evidence = json.loads(
            (PROJECT_ROOT / "atom-harness-desktop-release-evidence.json").read_text(
                encoding="utf-8"
            )
        )
        package = evidence["package"]
        installation = evidence["installation"]
        runtime = evidence["interactive_runtime"]
        hands = evidence["permissioned_hands"]

        self.assertEqual(evidence["schema"], 1)
        self.assertEqual(
            evidence["runtime"], "atom-harness-desktop-release-evidence-v2"
        )
        self.assertEqual(evidence["version"], "7.0.0")
        self.assertTrue(evidence["all_checks_passed"])
        self.assertGreater(package["file_count"], 157)
        self.assertGreater(package["portable_zip_bytes"], 0)
        self.assertRegex(package["portable_zip_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(package["msi_bytes"], 0)
        self.assertRegex(package["msi_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(installation["passed"])
        self.assertTrue(installation["full_release_manifest_verified"])
        self.assertTrue(runtime["real_artifact_side_view_visible"])
        self.assertFalse(runtime["cloud_evidence_used"])
        self.assertEqual(runtime["llm_memory_writes"], 0)
        self.assertGreaterEqual(runtime["wiki_nodes"], 2737)
        self.assertGreater(runtime["retrieved_passages"], 0)
        self.assertEqual(hands["capability"], "workspace.write_text")
        self.assertEqual(hands["action_count"], 1)
        self.assertEqual(hands["maximum_risk"], "high")
        self.assertEqual(
            hands["candidate_normalizations"],
            [
                {
                    "action_id": "action-1",
                    "kind": "unsupported-argument-fields-omitted",
                    "fields": ["format"],
                }
            ],
        )
        self.assertTrue(hands["exact_manifest_visible"])
        self.assertTrue(hands["normalizations_visible"])
        self.assertTrue(hands["approve_and_deny_controls_visible"])
        self.assertEqual(hands["decision"], "denied")
        self.assertEqual(hands["result_count"], 0)
        self.assertFalse(hands["proposed_file_exists_after_denial"])
        self.assertFalse(hands["grant_secret_persisted"])
        self.assertFalse(hands["permission_grants_persisted"])
        self.assertFalse(hands["tool_results_trusted_as_instructions"])
        self.assertTrue(hands["denied_proposal_recovered_after_restart"])
        self.assertTrue(hands["deterministic_completed_tool_side_view_verified"])
        self.assertTrue(
            hands["public_web_connection_pinned_to_permission_bound_address"]
        )
        self.assertTrue(hands["process_executable_hash_drift_blocked"])
        self.assertTrue(hands["process_output_retention_bounded"])
        self.assertTrue(hands["process_timeout_tree_cleanup_passed"])
        for section, fields in (
            (
                runtime,
                (
                    "transaction_sha256",
                    "transaction_manifest_sha256",
                    "artifact_sha256",
                    "side_view_sha256",
                ),
            ),
            (
                hands,
                (
                    "proposal_payload_sha256",
                    "manifest_sha256",
                    "action_sha256",
                    "permission_sha256",
                    "journal_sha256",
                ),
            ),
        ):
            for field in fields:
                self.assertRegex(section[field], r"^[0-9a-f]{64}$")

    def test_desktop_chain_exercises_wiki_rag_permission_and_tool_side_view(
        self,
    ) -> None:
        runtime_case = PermissionedHandsIntegrationTests(
            "test_phase6_wires_wiki_rag_permission_hands_and_both_side_views"
        )
        runtime_case.test_phase6_wires_wiki_rag_permission_hands_and_both_side_views()


if __name__ == "__main__":
    unittest.main()
