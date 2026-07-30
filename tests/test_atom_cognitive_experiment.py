from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import atom_cognitive_experiment as cognitive


class AtomCognitiveExperimentTests(unittest.TestCase):
    def test_self_test_contract(self) -> None:
        report = cognitive.run_self_tests()
        self.assertTrue(report["passed"], report["failed"])

    def test_context_codes_are_separable(self) -> None:
        cue_a = cognitive.encode_cue(2, 0)
        cue_b = cognitive.encode_cue(2, 1)
        cue_other = cognitive.encode_cue(3, 0)
        self.assertGreater(cognitive.cosine_similarity(cue_a, cue_a), 0.99)
        self.assertLess(cognitive.cosine_similarity(cue_a, cue_b), 0.70)
        self.assertLess(cognitive.cosine_similarity(cue_a, cue_other), 0.70)

    def test_local_learning_revision_and_forgetting(self) -> None:
        memory = cognitive.AtomCognitiveMemory()
        durable = cognitive.encode_cue(0, 0)
        transient = cognitive.encode_cue(7, 1)
        for _ in range(3):
            memory.observe(durable, 1)
        for _ in range(6):
            memory.observe(durable, 3, salience=1.15)
        memory.observe(transient, 2, salience=0.75)
        memory.idle(18)
        durable_result = memory.retrieve(durable)
        self.assertIsNotNone(durable_result)
        self.assertEqual(durable_result["value"], 3)
        self.assertIsNone(memory.retrieve(transient))

    def test_interleaved_exposures_survive_until_reinforcement(self) -> None:
        program = cognitive.build_tiny_program()
        memory = cognitive.AtomCognitiveMemory()
        metrics = cognitive.evaluate_trained_system(memory, program)
        self.assertGreaterEqual(metrics["category_accuracy"]["full"], 0.90)
        self.assertGreaterEqual(metrics["category_accuracy"]["partial"], 0.80)
        self.assertGreaterEqual(metrics["category_accuracy"]["context"], 0.90)
        self.assertGreaterEqual(
            metrics["category_accuracy"]["noise_rejection"], 0.80
        )

    def test_state_round_trip_preserves_retrieval(self) -> None:
        memory = cognitive.AtomCognitiveMemory()
        cue = cognitive.encode_cue(4, 0)
        for _ in range(3):
            memory.observe(cue, 2)
        restored = cognitive.AtomCognitiveMemory.from_state(memory.to_state())
        self.assertEqual(
            cognitive.stable_hash(memory.to_state()),
            cognitive.stable_hash(restored.to_state()),
        )
        self.assertEqual(restored.retrieve(cue)["value"], 2)

    def test_state_loader_rejects_corruption(self) -> None:
        memory = cognitive.AtomCognitiveMemory()
        memory.observe(cognitive.encode_cue(0, 0), 1)
        valid = memory.to_state()
        corruptions = []

        state = copy.deepcopy(valid)
        state["traces"][0]["mass"] = float("nan")
        corruptions.append(("nonfinite_mass", state))

        state = copy.deepcopy(valid)
        state["traces"][0]["evidence"][1] = -1.0
        corruptions.append(("negative_evidence", state))

        state = copy.deepcopy(valid)
        state["traces"][0]["support"] = 1.1
        corruptions.append(("support_range", state))

        state = copy.deepcopy(valid)
        state["traces"][0]["active"] = "yes"
        corruptions.append(("activity_type", state))

        state = copy.deepcopy(valid)
        state["forgotten_count"] = 1
        corruptions.append(("forgotten_count", state))

        state = copy.deepcopy(valid)
        state["action_counts"]["nucleate"] = -1
        corruptions.append(("negative_action_count", state))

        for name, state in corruptions:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cognitive.AtomCognitiveMemory.from_state(state)

    def test_serialized_request_to_response(self) -> None:
        cue = cognitive.encode_cue(1, 0)
        request = {
            "request_id": "unit-cognitive-001",
            "experiences": [
                {
                    "event_id": f"event-{index}",
                    "cue": cue,
                    "value": 2,
                    "salience": 1.0,
                }
                for index in range(3)
            ],
            "idle_steps": 2,
            "queries": [
                {
                    "query_id": "partial-query",
                    "cue": cognitive.partial_cue(cue, (0, 2, 4)),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            response_path = root / "response.json"
            cognitive.write_json(request_path, request)
            response = cognitive.run_serialized_workflow(
                request_path, response_path
            )
            persisted = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(response["status"], "ok")
            self.assertEqual(persisted["predictions"][0]["prediction"], 2)
            self.assertGreater(persisted["memory"]["compression_ratio"], 2.0)

    def test_non_finite_request_is_rejected(self) -> None:
        cue = cognitive.encode_cue(0, 0)
        cue[0] = float("nan")
        request = {
            "request_id": "invalid",
            "experiences": [
                {"event_id": "bad", "cue": cue, "value": 0, "salience": 1.0}
            ],
            "idle_steps": 0,
            "queries": [{"query_id": "q", "cue": cognitive.encode_cue(0, 0)}],
        }
        with self.assertRaises(ValueError):
            cognitive.validate_workflow_request(request)

    def test_request_rejects_ambiguous_ids_and_boolean_numbers(self) -> None:
        cue = cognitive.encode_cue(0, 0)
        valid = {
            "request_id": "strict-request",
            "experiences": [
                {"event_id": "event", "cue": cue, "value": 0, "salience": 1.0}
            ],
            "idle_steps": 0,
            "queries": [{"query_id": "query", "cue": cue}],
        }
        invalid_requests = []

        request = copy.deepcopy(valid)
        request["experiences"].append(copy.deepcopy(request["experiences"][0]))
        invalid_requests.append(("duplicate_event_id", request))

        request = copy.deepcopy(valid)
        request["queries"].append(copy.deepcopy(request["queries"][0]))
        invalid_requests.append(("duplicate_query_id", request))

        request = copy.deepcopy(valid)
        request["experiences"][0]["value"] = True
        invalid_requests.append(("boolean_value", request))

        request = copy.deepcopy(valid)
        request["idle_steps"] = True
        invalid_requests.append(("boolean_idle", request))

        request = copy.deepcopy(valid)
        request["queries"][0]["cue"][0] = False
        invalid_requests.append(("boolean_cue", request))

        for name, request in invalid_requests:
            with self.subTest(name=name), self.assertRaises(ValueError):
                cognitive.validate_workflow_request(request)


if __name__ == "__main__":
    unittest.main()
