from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.test_atom_language_harness_v4_integration import (
    AtomLanguageHarnessV4IntegrationTests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATOM_HARNESS_WIKI_RUNTIME = "ATOM_HARNESS_WIKI_RUNTIME"
ATOM_HARNESS_RAG_RUNTIME = "ATOM_HARNESS_RAG_RUNTIME"
ATOM_HARNESS_OPERATOR_UI_RUNTIME = "ATOM_HARNESS_OPERATOR_UI_RUNTIME"
ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING = "render_operator_surface"


class AtomHarnessDesktopV5IntegrationTests(unittest.TestCase):
    def test_phase5_compatibility_gate_accepts_phase6_successor_boundaries(
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
        main_form = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/MainForm.cs"
        ).read_text(encoding="utf-8")
        backend = (PROJECT_ROOT / "atom_harness_desktop_backend.py").read_text(
            encoding="utf-8"
        )
        backend_spec = (PROJECT_ROOT / "atom-harness-backend.spec").read_text(
            encoding="utf-8"
        )
        package_builder = (
            PROJECT_ROOT / "scripts/build_atom_harness_desktop.ps1"
        ).read_text(encoding="utf-8")

        self.assertEqual(architecture["product_phase"], 6)
        self.assertEqual(architecture["runtime"], "atom-harness-desktop-v6")
        self.assertEqual(
            architecture["installed_runtime"]["authority_runtime"],
            "language-harness-v5",
        )
        self.assertTrue(architecture["desktop_shell"]["thin_shell_only"])
        self.assertTrue(
            architecture["desktop_shell"]["real_artifact_side_view_embedded"]
        )
        self.assertFalse(architecture["installed_runtime"]["cloud_allowed"])
        self.assertEqual(update["schema"], 1)
        self.assertTrue(update["policy"]["explicit_user_consent_required"])
        self.assertTrue(update["policy"]["artifact_sha256_required"])
        self.assertTrue(update["policy"]["stage_outside_install_directory"])
        self.assertTrue(update["policy"]["replace_only_after_app_exit"])
        installed_verifier = (
            PROJECT_ROOT / "desktop/AtomHarness.Desktop/InstalledLayoutVerifier.cs"
        ).read_text(encoding="utf-8")
        runtime_sources = program + installed_verifier
        for marker in (
            ATOM_HARNESS_WIKI_RUNTIME,
            ATOM_HARNESS_RAG_RUNTIME,
            ATOM_HARNESS_OPERATOR_UI_RUNTIME,
            ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING,
        ):
            self.assertIn(marker, runtime_sources)
        self.assertIn("atom_harness_operator_server", backend)
        self.assertIn("atom-harness-bundled-causal-memory-v1", backend)
        self.assertIn("_bind_bundled_causal_memory", backend)
        self.assertIn("atom_causal_memory.RELEASE_BINARY = binary", backend)
        self.assertIn("atom-causal-memory.exe", backend_spec)
        self.assertIn("cargo build", package_builder)
        self.assertIn("confirmUpdate", main_form)
        self.assertIn("confirmInstall", main_form)
        self.assertIn("DownloadAndVerifyAsync", main_form)
        self.assertIn("RequestGracefulShutdownAsync", main_form)

    def test_desktop_chain_preserves_real_v4_wiki_rag_and_side_view(self) -> None:
        runtime_case = AtomLanguageHarnessV4IntegrationTests(
            "test_operator_runtime_wires_wiki_rag_api_and_real_side_view"
        )
        runtime_case.test_operator_runtime_wires_wiki_rag_api_and_real_side_view()


if __name__ == "__main__":
    unittest.main()
