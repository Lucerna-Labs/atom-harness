from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import atom_phase_law_experiment as phase


class AtomPhaseLawExperimentTests(unittest.TestCase):
    def test_dataset_holds_out_pairs_and_all_compositions(self) -> None:
        program = phase.build_tiny_world()
        train = {(row["source"], row["operators"][0]) for row in program["train"]}
        heldout = {
            (row["source"], row["operators"][0])
            for row in program["heldout_single_step"]
        }
        self.assertFalse(train & heldout)
        self.assertEqual(len(program["symbols"]), 8)
        self.assertEqual(len(program["operators"]), 4)
        self.assertEqual(len(program["unseen_two_step"]), 128)
        self.assertEqual(len(program["unseen_three_step"]), 512)

    def test_substrate_is_frozen_and_only_kernel_replaces_state(self) -> None:
        program = phase.build_tiny_world()
        kernel = phase.UniversePhaseKernel()
        state = kernel.initial_state(program["symbols"])
        with self.assertRaises(FrozenInstanceError):
            state.tick = 1  # type: ignore[misc]
        audit = phase.architecture_audit()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["replace_calls_outside_kernel"], [])

    def test_repeated_evidence_nucleates_a_phase_field(self) -> None:
        program = phase.build_tiny_world()
        row = program["train"][0]
        runtime = phase.PhaseRuntime(program["symbols"])
        runtime.observe(row["source"], row["operators"][0], row["target"], "first")
        runtime.observe(row["source"], row["operators"][0], row["target"], "second")
        second = program["train"][1]
        runtime.observe(
            second["source"], second["operators"][0], second["target"], "third"
        )
        runtime.observe(
            second["source"], second["operators"][0], second["target"], "fourth"
        )
        self.assertTrue(any(trace.active for trace in runtime.state.traces))
        self.assertTrue(runtime.state.laws)
        result = runtime.predict(
            row["source"], row["operators"], event_id="phase-query"
        )
        self.assertIsNotNone(result)
        self.assertGreater(runtime.state.cumulative_phase_energy, 0.0)

    def test_conservation_caps_information_mass(self) -> None:
        program = phase.build_tiny_world()
        config = phase.config_with(
            phase.PhaseConfig(), information_mass_budget=0.55, epochs=2
        )
        runtime = phase.PhaseRuntime(
            program["symbols"], kernel=phase.UniversePhaseKernel(config)
        )
        row = program["train"][0]
        for index in range(12):
            runtime.observe(
                row["source"],
                row["operators"][0],
                row["target"],
                event_id=f"mass-{index}",
            )
        self.assertLessEqual(
            runtime.state.information_mass, config.information_mass_budget + 1e-12
        )

    def test_model_round_trip_and_corruption_rejection(self) -> None:
        program = phase.build_tiny_world()
        config = phase.config_with(
            phase.PhaseConfig(), epochs=3, crystallization_coherence=0.0
        )
        runtime, _ = phase.train_phase_model(program, config=config)
        runtime.consolidate("unit-abstract")
        model = phase.model_payload(runtime, program)
        restored = phase.runtime_from_model(json.loads(json.dumps(model)))
        self.assertEqual(
            model["model_hash"], phase.model_payload(restored, program)["model_hash"]
        )

        corruptions = []
        corrupt = copy.deepcopy(model)
        corrupt["symbols"][0]["slot"] = corrupt["symbols"][1]["slot"]
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["laws"][0]["coherence"] = float("nan")
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["raw_trace_count"] = 1
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["unknown"] = True
        corruptions.append(corrupt)
        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index), self.assertRaises(ValueError):
                phase.runtime_from_model(corrupt)

    def test_serialized_request_rejects_unknown_fields(self) -> None:
        program = phase.build_tiny_world()
        request = {
            "request_id": "strict",
            "queries": [
                {
                    "query_id": "q",
                    "source": program["symbols"][0],
                    "operators": [program["operators"][0]],
                    "unexpected": 1,
                }
            ],
        }
        with self.assertRaises(ValueError):
            phase.validate_prediction_request(
                request, set(program["symbols"]), set(program["operators"])
            )

    def test_public_prediction_scalars_are_cross_platform_canonical(self) -> None:
        program = phase.build_tiny_world()
        config = phase.config_with(
            phase.PhaseConfig(), epochs=3, crystallization_coherence=0.0
        )
        runtime, _ = phase.train_phase_model(program, config=config)
        row = program["heldout_single_step"][0]
        result = runtime.predict(
            row["source"], row["operators"], "canonical-public-scalars"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["confidence"], round(result["confidence"], 12))
        self.assertEqual(result["wave_strength"], round(result["wave_strength"], 12))

    def test_missing_law_is_only_tolerated_by_ablation_scoring(self) -> None:
        program = phase.build_tiny_world()
        config = phase.config_with(phase.PhaseConfig(), epochs=2)
        runtime, _ = phase.train_phase_model(
            program, config=config, disabled=(phase.Primitive.NUCLEATION,)
        )
        row = program["heldout_single_step"][0]
        with self.assertRaises(ValueError):
            runtime.predict(row["source"], row["operators"], "strict-missing-law")
        metrics = phase.evaluate_rows(
            runtime,
            program["heldout_single_step"],
            "ablation-missing-law",
            allow_unknown_laws=True,
        )
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["coverage"], 0.0)

    def test_side_view_rejects_unbound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "view.html"
            with self.assertRaises(ValueError):
                phase.render_phase_artifact({}, {}, path)

    def test_self_test_contract(self) -> None:
        report = phase.run_self_tests()
        self.assertTrue(report["passed"], report["failed"])


if __name__ == "__main__":
    unittest.main()
