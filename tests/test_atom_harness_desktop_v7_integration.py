from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.test_atom_universal_knowledge_integration import (
    AtomUniversalKnowledgeIntegrationTests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


class AtomHarnessDesktopV7IntegrationTests(unittest.TestCase):
    def test_desktop_v7_declares_universal_knowledge_and_preserves_hands(
        self,
    ) -> None:
        architecture = _json("atom-harness-desktop-architecture.json")
        update = _json("lucerna-update.json")
        registry = _json("ai-runtime-registry.json")
        runtime_knowledge = _json("ai-runtime-knowledge.json")
        spec = (PROJECT_ROOT / "atom-harness-backend.spec").read_text(encoding="utf-8")
        supervisor = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/BackendSupervisor.cs"
        ).read_text(encoding="utf-8")
        install_verifier = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/InstalledLayoutVerifier.cs"
        ).read_text(encoding="utf-8")

        self.assertEqual(architecture["product_phase"], 7)
        self.assertEqual(architecture["runtime"], "atom-harness-desktop-v7")
        self.assertEqual(architecture["version"], "7.0.0")
        self.assertEqual(
            architecture["installed_runtime"]["authority_runtime"],
            "language-harness-v6",
        )
        self.assertEqual(registry["active_runtime"], "language-harness-v6")
        self.assertEqual(
            registry["runtimes"]["language-harness-v6"]["integration_test"],
            "tests/test_atom_universal_knowledge_integration.py",
        )
        self.assertTrue(
            runtime_knowledge["causal_records_remain_a_separate_specialized_lane"]
        )
        self.assertTrue(runtime_knowledge["ornith_1_0_capability_floor_preserved"])
        universal = architecture["universal_knowledge"]
        self.assertTrue(universal["causal_memory_remains_separate"])
        self.assertEqual(universal["domain_count"], 15)
        self.assertEqual(universal["seed_claim_count"], 45)
        self.assertEqual(universal["source_count"], 22)
        self.assertTrue(
            universal[
                "formal_empirical_interpretive_fictional_and_craft_types_separate"
            ]
        )
        self.assertTrue(
            architecture["permissioned_hands"][
                "operator_permission_required_for_every_execution"
            ]
        )
        self.assertEqual(update["current_version"], "7.0.0")
        self.assertFalse(update["policy"]["automatic_download"])
        self.assertFalse(update["policy"]["automatic_install"])
        self.assertTrue(update["policy"]["explicit_user_consent_required"])
        self.assertIn("knowledge_packs", spec)
        self.assertIn("universal-foundation-v1", spec)
        self.assertIn("atom-multidisciplinary-wiki-v1", supervisor)
        self.assertIn("atom-multidisciplinary-graph-rag-v1", supervisor)
        self.assertIn("VerifyKnowledgePack", install_verifier)
        self.assertIn("knowledge_manifest_sha256", install_verifier)

    def test_release_evidence_binds_package_install_and_live_knowledge(
        self,
    ) -> None:
        evidence = _json("atom-harness-desktop-release-evidence.json")
        self.assertEqual(evidence["schema"], 1)
        self.assertEqual(
            evidence["runtime"], "atom-harness-desktop-release-evidence-v2"
        )
        self.assertEqual(evidence["version"], "7.0.0")
        self.assertTrue(evidence["passed"])
        self.assertTrue(evidence["all_checks_passed"])
        package = evidence["package"]
        self.assertEqual(package["portable_zip"], "Atom-Harness-7.0.0-windows-x64.zip")
        self.assertEqual(package["msi"], "Atom-Harness-7.0.0-windows-x64.msi")
        self.assertGreater(package["portable_zip_bytes"], 0)
        self.assertGreater(package["msi_bytes"], 0)
        self.assertRegex(package["portable_zip_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(package["msi_sha256"], r"^[0-9a-f]{64}$")
        knowledge = evidence["knowledge"]
        self.assertEqual(knowledge["pack_id"], "atom-universal-foundation-v1")
        self.assertEqual(knowledge["domain_count"], 15)
        self.assertEqual(knowledge["claim_count"], 45)
        self.assertEqual(knowledge["source_count"], 22)
        self.assertTrue(knowledge["installed_layout_verified"])
        self.assertTrue(knowledge["live_multidisciplinary_answer_verified"])
        self.assertTrue(knowledge["causal_lane_regression_verified"])
        self.assertTrue(knowledge["permissioned_hands_regression_verified"])
        self.assertTrue(knowledge["real_artifact_side_view_verified"])
        self.assertRegex(knowledge["knowledge_hash"], r"^[0-9a-f]{64}$")
        self.assertRegex(knowledge["graph_knowledge_hash"], r"^[0-9a-f]{64}$")
        installation = evidence["installation"]
        self.assertTrue(installation["passed"])
        self.assertTrue(installation["full_release_manifest_verified"])
        self.assertTrue(installation["knowledge_pack_hashes_verified"])

    def test_desktop_chain_exercises_phase7_and_phase6_capability_floor(
        self,
    ) -> None:
        runtime = AtomUniversalKnowledgeIntegrationTests(
            "test_phase7_wires_multidisciplinary_graph_rag_and_real_side_view"
        )
        runtime.test_phase7_wires_multidisciplinary_graph_rag_and_real_side_view()
        hands = AtomUniversalKnowledgeIntegrationTests(
            "test_phase7_preserves_phase6_permissioned_hands_and_operator_view"
        )
        hands.test_phase7_preserves_phase6_permissioned_hands_and_operator_view()


if __name__ == "__main__":
    unittest.main()
