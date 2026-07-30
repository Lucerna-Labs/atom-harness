from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import atom_field_proof as proof


class AtomFieldProofTests(unittest.TestCase):
    def test_full_self_test_gate(self) -> None:
        report = proof.run_self_tests()
        self.assertTrue(report["passed"], report["failed"])

    def test_tiny_split_contract(self) -> None:
        splits = proof.generate_splits()
        audit = proof.audit_splits(splits)
        self.assertTrue(audit["passed"], audit["failures"])
        self.assertEqual(
            audit["counts"], {"train": 140, "validation": 40, "heldout": 36}
        )
        self.assertFalse(audit["heldout_family_overlap"])

    def test_closed_simulator_conserves_mass(self) -> None:
        case = proof.build_case(
            "aggregate_preserve_expire", proof.SEED + 5, "heldout", 5
        )
        start = sum(node[1] * node[5] for node in case["node_features"])
        finish = sum(node[1] for node in case["target_continuous"])
        self.assertLess(abs(start - finish), 1e-5)

    def test_model_forward_is_finite(self) -> None:
        rows = proof.generate_splits()["validation"][:2]
        batch = next(iter(proof.make_loader(rows, 2, False, proof.SEED)))
        model = proof.AtomFieldNet(ticks=2)
        outputs = model(
            batch["node_features"], batch["adjacency"], batch["global_features"]
        )
        self.assertTrue(bool(torch.isfinite(outputs["continuous"]).all()))
        self.assertTrue(bool(torch.isfinite(outputs["binary_logits"]).all()))
        self.assertTrue(
            bool(
                torch.allclose(
                    outputs["route_mean"].sum(dim=-1),
                    torch.ones(2),
                    atol=1e-5,
                )
            )
        )

    def test_serialized_workflow_round_trip(self) -> None:
        case = proof.build_case(
            "full_cycle", proof.SEED + 800_001, "workflow-test", 0
        )
        request = {
            "request_id": "unit-workflow-001",
            "node_features": case["node_features"],
            "adjacency": case["adjacency"],
            "global_features": case["global_features"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            weights_path = root / "weights.pt"
            response_path = root / "response.json"
            proof.write_json(request_path, request)
            torch.save(proof.AtomFieldNet().state_dict(), weights_path)
            response = proof.run_serialized_field_workflow(
                request_path,
                weights_path,
                response_path,
                torch.device("cpu"),
            )
            persisted = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(response["status"], "ok")
            self.assertEqual(persisted["request_id"], "unit-workflow-001")
            self.assertEqual(len(persisted["predicted_continuous"]), proof.NODE_COUNT)
            self.assertLess(persisted["invariant"]["absolute_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
