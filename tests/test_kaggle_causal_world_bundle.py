from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.build_kaggle_causal_world_bundle import (
    PROJECT_ROOT,
    build_causal_world_bundle,
    causal_bundle_main,
)
from scripts.audit_wilson_portability import audit_wilson_portability


class KaggleCausalWorldBundleTests(unittest.TestCase):
    def test_gpu_bundle_contains_formal_decimal_runtime_and_side_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            manifest = build_causal_world_bundle(
                output_dir,
                accelerator="gpu",
            )
            metadata = json.loads(
                (output_dir / "kernel-metadata.json").read_text(encoding="utf-8")
            )
            source = (output_dir / "atom_causal_world_kaggle.py").read_text(
                encoding="utf-8"
            )
        self.assertEqual(manifest["accelerator"], "GPU")
        self.assertEqual(
            metadata["id"],
            "jessealicea/atom-massive-causal-world-gpu-v1",
        )
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["enable_tpu"], "false")
        for marker in (
            'KAGGLE_BUNDLE_ACCELERATOR = "gpu"',
            'CONTEXT_FACTOR_GRAPH_RUNTIME = "atom-causal-context-factor-graph-v2"',
            'TRANSFER_POLICY_RUNTIME = "atom-causal-metaplastic-transfer-policy-v7"',
            'TRANSFER_RISK_METHOD = "wilson_score_upper_bound_decimal12"',
            'FORMAL_DOMAIN_RUNTIME = "atom-formal-domain-runtime-v1"',
            'FORMAL_TRUTH_ORACLE_RUNTIME = "atom-formal-truth-oracle-v1"',
            'ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME = "atom-causal-world-side-view-v10"',
            "render_causal_world_artifact",
            "from decimal import Decimal, ROUND_HALF_EVEN, localcontext",
            "from fractions import Fraction",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

    def test_wilson_boundary_is_exhaustively_decimal_stable(self) -> None:
        result = audit_wilson_portability(576)
        self.assertTrue(result["passed"])
        self.assertEqual(result["count_pairs"], 166753)
        self.assertEqual(result["legacy_boundary_difference_count"], 5)
        self.assertEqual(
            result["deterministic_value_sha256"],
            "b3b8fb4d202bf1bf0cb945484a1b87b4dca1d5616f7f78618cb3f4215036f702",
        )

    def test_default_output_lane_follows_selected_accelerator(self) -> None:
        for accelerator, lane in (
            ("tpu", "causal-world-v1"),
            ("gpu", "causal-world-gpu-v1"),
        ):
            with self.subTest(accelerator=accelerator):
                argv = [
                    "build_kaggle_causal_world_bundle.py",
                    "--accelerator",
                    accelerator,
                ]
                with (
                    patch("sys.argv", argv),
                    patch(
                        "scripts.build_kaggle_causal_world_bundle."
                        "build_causal_world_bundle",
                        return_value={"accelerator": accelerator.upper()},
                    ) as builder,
                    redirect_stdout(io.StringIO()),
                ):
                    causal_bundle_main()
                self.assertEqual(
                    builder.call_args.args[0],
                    (PROJECT_ROOT / "kaggle" / lane).resolve(),
                )
                self.assertEqual(
                    builder.call_args.kwargs,
                    {"accelerator": accelerator},
                )

    def test_explicit_output_lane_is_preserved(self) -> None:
        selected = Path("custom-kaggle-lane")
        argv = [
            "build_kaggle_causal_world_bundle.py",
            "--accelerator",
            "gpu",
            "--output-dir",
            str(selected),
        ]
        with (
            patch("sys.argv", argv),
            patch(
                "scripts.build_kaggle_causal_world_bundle."
                "build_causal_world_bundle",
                return_value={"accelerator": "GPU"},
            ) as builder,
            redirect_stdout(io.StringIO()),
        ):
            causal_bundle_main()
        self.assertEqual(builder.call_args.args[0], selected.resolve())
        self.assertEqual(builder.call_args.kwargs, {"accelerator": "gpu"})


if __name__ == "__main__":
    unittest.main()
