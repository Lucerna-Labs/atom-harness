from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import atom_universe_composition as universe


class AtomUniverseCompositionTests(unittest.TestCase):
    def test_seven_primitives_are_the_only_recipe_leaves(self) -> None:
        self.assertEqual(len(universe.UNIVERSE_PRIMITIVES), 7)
        primitive_names = set(universe.UNIVERSE_PRIMITIVES)
        book = universe.RecipeBook()
        for name in book.names:
            expanded = book.expand(name)
            self.assertTrue(expanded)
            self.assertTrue({step.value for step in expanded}.issubset(primitive_names))

    def test_substrate_is_immutable_and_mutation_boundary_is_enforced(self) -> None:
        kernel = universe.UniverseKernel()
        state = kernel.initial_state()
        with self.assertRaises(FrozenInstanceError):
            state.tick = 1  # type: ignore[misc]
        audit = universe.architecture_audit()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["replace_calls_outside_universe_kernel"], [])

    def test_behavior_and_every_primitive_ablation(self) -> None:
        program = universe.build_tiny_program()
        runtime = universe.CompositionRuntime()
        metrics = universe.evaluate_atom_runtime(runtime, program)
        self.assertEqual(metrics["behavior_score"], 1.0)
        self.assertEqual(metrics["active_memory_units"], 10)
        self.assertLessEqual(metrics["mass_excess"], 1e-9)
        self.assertEqual(
            set(metrics["operator_counts"]), set(universe.UNIVERSE_PRIMITIVES)
        )
        ablations = universe.run_ablations(program)
        self.assertEqual(set(ablations), set(universe.UNIVERSE_PRIMITIVES))
        self.assertTrue(
            all(row["causal_effect_observed"] for row in ablations.values())
        )

    def test_phase_mixing_and_thermal_annealing_are_controlled(self) -> None:
        controls = universe.run_chaos_controls(universe.build_tiny_program())
        phase = controls["phase_mixing"]
        thermal = controls["thermal_annealing"]
        self.assertGreater(phase["cumulative_phase_energy"], 0.0)
        self.assertEqual(phase["zero_phase_energy"], 0.0)
        self.assertTrue(phase["bounded"])
        self.assertTrue(phase["changes_trajectory"])
        self.assertGreater(thermal["temperature_drop"], 0.0)
        self.assertTrue(thermal["monotonic"])
        self.assertTrue(thermal["changes_trajectory"])
        self.assertTrue(controls["behavior_preserved"])
        self.assertTrue(all(controls["determinism"].values()))

    def test_state_round_trip_and_corruption_rejection(self) -> None:
        runtime = universe.CompositionRuntime()
        universe.evaluate_atom_runtime(runtime, universe.build_tiny_program())
        payload = universe.runtime_payload(runtime)
        restored = universe.runtime_from_payload(json.loads(json.dumps(payload)))
        self.assertEqual(
            universe.stable_hash(payload),
            universe.stable_hash(universe.runtime_payload(restored)),
        )

        corruptions = []
        corrupt = copy.deepcopy(payload)
        corrupt["state"]["traces"][0]["mass"] = float("nan")
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(payload)
        corrupt["state"]["operator_counts"]["radiation"] = -1
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(payload)
        corrupt["state"]["transition_hash"] = "invalid"
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(payload)
        corrupt["recipes"]["learn"]["declared_steps"] = ["direct_write"]
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(payload)
        corrupt["state"]["phase_energy"] = 1.0
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(payload)
        corrupt["state"]["temperature"] = 0.0
        corruptions.append(corrupt)

        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index), self.assertRaises(ValueError):
                universe.runtime_from_payload(corrupt)

    def test_serialized_request_response_uses_composed_atoms(self) -> None:
        program = universe.build_tiny_program()
        request = {
            "request_id": "composition-test",
            "experiences": [
                {
                    "event_id": row["event_id"],
                    "atom": row["atom"],
                    "cue": row["cue"],
                    "value": row["value"],
                    "salience": row["salience"],
                }
                for row in program["experiences"]
            ],
            "idle_steps": program["idle_steps"],
            "queries": [
                {
                    "query_id": row["query_id"],
                    "atom": row["atom"],
                    "cue": row["cue"],
                }
                for row in program["queries"]
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            response_path = root / "response.json"
            universe.write_json(request_path, request)
            response = universe.run_serialized_workflow(request_path, response_path)
            metrics = universe.score_workflow_response(response, program["queries"])
            self.assertTrue(metrics["passed"])
            self.assertEqual(len(response["predictions"]), 31)
            self.assertEqual(
                set(response["runtime"]["operator_counts"]),
                set(universe.UNIVERSE_PRIMITIVES),
            )

    def test_request_rejects_direct_write_and_duplicate_ids(self) -> None:
        cue = universe.encode_cue(0, 0)
        valid = {
            "request_id": "strict-composition",
            "experiences": [
                {
                    "event_id": "event",
                    "atom": "remember",
                    "cue": cue,
                    "value": 0,
                    "salience": 1.0,
                }
            ],
            "idle_steps": 0,
            "queries": [{"query_id": "query", "atom": "retrieve", "cue": cue}],
        }
        invalid_requests = []
        invalid = copy.deepcopy(valid)
        invalid["experiences"][0]["atom"] = "direct_write"
        invalid_requests.append(invalid)
        invalid = copy.deepcopy(valid)
        invalid["experiences"].append(copy.deepcopy(invalid["experiences"][0]))
        invalid_requests.append(invalid)
        invalid = copy.deepcopy(valid)
        invalid["queries"][0]["atom"] = "remember"
        invalid_requests.append(invalid)
        invalid = copy.deepcopy(valid)
        invalid["idle_steps"] = True
        invalid_requests.append(invalid)

        for index, invalid in enumerate(invalid_requests):
            with self.subTest(index=index), self.assertRaises(ValueError):
                universe.validate_workflow_request(invalid)

    def test_self_test_contract(self) -> None:
        report = universe.run_self_tests()
        self.assertTrue(report["passed"], report["failed"])


if __name__ == "__main__":
    unittest.main()
