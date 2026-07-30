from __future__ import annotations

import copy
import json
import unittest

from atom_causal_world_schema import canonical_hash
from atom_formal_domains import (
    FORMAL_DOMAIN_NAMES,
    FORMAL_DOMAIN_RUNTIME,
    FORMAL_DOMAIN_SCHEMA,
    FORMAL_EPISTEMIC_STATES,
    FORMAL_PRIMITIVES,
    execute_formal_program,
    formal_domain_manifest,
    run_formal_domain_benchmark,
    solve_formal_request,
)


def formal_request(
    primitive: str,
    arguments: dict[str, object],
    *,
    query_id: str = "test",
) -> dict[str, object]:
    return {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "runtime": FORMAL_DOMAIN_RUNTIME,
        "query_id": query_id,
        "primitive": primitive,
        "arguments": arguments,
    }


class AtomFormalDomainTests(unittest.TestCase):
    def test_registry_covers_every_domain_and_is_hash_bound(self) -> None:
        manifest = formal_domain_manifest()
        core = {key: value for key, value in manifest.items() if key != "registry_hash"}
        self.assertEqual(set(manifest["domains"]), set(FORMAL_DOMAIN_NAMES))
        self.assertEqual(len(manifest["primitives"]), len(FORMAL_PRIMITIVES))
        self.assertTrue(all(manifest["domain_counts"].values()))
        self.assertEqual(manifest["registry_hash"], canonical_hash(core))
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertIn("proven", FORMAL_EPISTEMIC_STATES)
        self.assertIn("contradicted", FORMAL_EPISTEMIC_STATES)
        self.assertIn("unknown", FORMAL_EPISTEMIC_STATES)

    def test_logic_proves_truth_and_contradicts_false_candidates(self) -> None:
        request = formal_request(
            "logic_implies",
            {"premise": True, "conclusion": False},
        )
        response = solve_formal_request(request)
        self.assertEqual(response["value"], False)
        self.assertEqual(response["claim_status"], "proven")
        challenged = {**request, "candidate": True}
        contradiction = solve_formal_request(challenged)
        self.assertEqual(contradiction["claim_status"], "contradicted")
        self.assertFalse(contradiction["candidate_matches"])
        self.assertEqual(
            contradiction["proof_trace"][-1]["epistemic_state"],
            "contradicted",
        )

    def test_algebra_geometry_and_calculus_remain_exact(self) -> None:
        linear = solve_formal_request(
            formal_request(
                "algebra_solve_linear",
                {"coefficient": 6, "offset": 3, "result": 8},
            )
        )
        distance = solve_formal_request(
            formal_request(
                "geometry_distance_squared",
                {"left": [-2, 5], "right": [4, -3]},
            )
        )
        derivative = solve_formal_request(
            formal_request(
                "calculus_polynomial_derivative",
                {"coefficients": [7, -2, 3, 4]},
            )
        )
        integral = solve_formal_request(
            formal_request(
                "calculus_definite_integral",
                {"coefficients": [0, 0, 3], "lower": 0, "upper": 2},
            )
        )
        self.assertEqual(linear["value"], "5/6")
        self.assertEqual(distance["value"], 100)
        self.assertEqual(derivative["value"], [-2, 6, 12])
        self.assertEqual(integral["value"], "8/1")
        self.assertTrue(
            all(
                response["precision"] == "exact"
                for response in (linear, distance, derivative, integral)
            )
        )

    def test_chemistry_and_biology_enforce_conservation(self) -> None:
        balanced = solve_formal_request(
            formal_request(
                "chemistry_mass_conservation",
                {"reactant_masses": [2, 32], "product_masses": [34]},
            )
        )
        extent = solve_formal_request(
            formal_request(
                "chemistry_stoichiometric_extent",
                {"available_moles": [5, 12], "coefficients": [2, 3]},
            )
        )
        offspring = solve_formal_request(
            formal_request(
                "biology_mendelian_distribution",
                {"parent_a": "Aa", "parent_b": "Aa"},
            )
        )
        feedback = solve_formal_request(
            formal_request(
                "biology_homeostatic_error",
                {"target": 98, "observed": 105},
            )
        )
        self.assertTrue(balanced["value"])
        self.assertEqual(extent["value"], "5/2")
        self.assertEqual(
            offspring["value"],
            {"AA": "1/4", "Aa": "1/2", "aa": "1/4"},
        )
        self.assertEqual(feedback["value"], -7)

    def test_information_theory_uses_deterministic_decimal_projection(self) -> None:
        maximum_entropy = solve_formal_request(
            formal_request(
                "information_binary_entropy",
                {"successes": 1, "trials": 2},
            )
        )
        deterministic = solve_formal_request(
            formal_request(
                "information_binary_entropy",
                {"successes": 0, "trials": 23},
            )
        )
        independent = solve_formal_request(
            formal_request(
                "information_mutual_information",
                {"joint_counts": [10, 10, 10, 10]},
            )
        )
        correlated = solve_formal_request(
            formal_request(
                "information_mutual_information",
                {"joint_counts": [20, 0, 0, 20]},
            )
        )
        self.assertEqual(maximum_entropy["value"], "1.000000000000")
        self.assertEqual(deterministic["value"], "0.000000000000")
        self.assertEqual(independent["value"], "0.000000000000")
        self.assertEqual(correlated["value"], "1.000000000000")

    def test_cross_domain_program_preserves_each_proof_trace(self) -> None:
        result = execute_formal_program(
            "derive-then-evaluate",
            [
                {
                    "primitive": "calculus_polynomial_derivative",
                    "arguments": {"coefficients": [1, 2, 3]},
                },
                {
                    "primitive": "algebra_polynomial_value",
                    "arguments": {"coefficients": {"$ref": 0}, "x": 3},
                },
            ],
        )
        self.assertEqual(result["domains"], ["calculus", "algebra"])
        self.assertEqual(result["value"], 20)
        self.assertEqual(result["claim_status"], "proven")
        self.assertEqual(result["stage_count"], 2)
        self.assertTrue(all(stage["proof_trace"] for stage in result["stages"]))
        self.assertEqual(len(result["program_hash"]), 64)

    def test_serialized_requests_fail_closed(self) -> None:
        valid = formal_request(
            "algebra_solve_linear",
            {"coefficient": 2, "offset": 1, "result": 7},
        )
        unknown_field = {**valid, "unexpected": True}
        unknown_primitive = {**valid, "primitive": "invent_answer"}
        missing_argument = copy.deepcopy(valid)
        del missing_argument["arguments"]["result"]
        zero_coefficient = copy.deepcopy(valid)
        zero_coefficient["arguments"]["coefficient"] = 0
        with self.assertRaises(ValueError):
            solve_formal_request(unknown_field)
        with self.assertRaises(ValueError):
            solve_formal_request(unknown_primitive)
        with self.assertRaises(ValueError):
            solve_formal_request(missing_argument)
        with self.assertRaises(ValueError):
            solve_formal_request(zero_coefficient)
        with self.assertRaises(TypeError):
            solve_formal_request(
                formal_request(
                    "logic_implies",
                    {"premise": 1, "conclusion": True},
                )
            )

    def test_curriculum_has_disjoint_partitions_and_oracle_agreement(self) -> None:
        artifact = run_formal_domain_benchmark(cases_per_primitive=24)
        report = artifact["report"]
        self.assertTrue(report["passed"])
        self.assertEqual(report["domain_count"], 7)
        self.assertEqual(report["case_count"], len(FORMAL_PRIMITIVES) * 24)
        self.assertGreater(report["partition_counts"]["demonstration"], 0)
        self.assertGreater(report["partition_counts"]["validation"], 0)
        self.assertGreater(report["partition_counts"]["heldout"], 0)
        self.assertTrue(
            report["gates"]["runtime_and_oracle_implementations_are_disjoint"]
        )
        self.assertTrue(report["gates"]["runtime_matches_independent_oracle"])
        self.assertTrue(report["gates"]["false_candidates_are_contradicted"])
        self.assertTrue(report["gates"]["cross_domain_programs_are_proven"])
        self.assertTrue(
            all(domain["accuracy"] == 1.0 for domain in report["per_domain"].values())
        )
        self.assertEqual(len(report["samples"]), 7)
        self.assertEqual(len(report["report_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
