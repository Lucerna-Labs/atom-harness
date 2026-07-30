from __future__ import annotations

import copy
import json
import unittest
from dataclasses import FrozenInstanceError

from atom_language_dataset import build_grounded_language_program
from atom_language_field import (
    LANGUAGE_MODEL_SCHEMA,
    LanguageConfig,
    Primitive,
    UniverseLanguageKernel,
    architecture_audit,
    evaluate_language_rows,
    language_model_payload,
    runtime_from_language_model,
    train_language_field,
    validate_language_request,
)


class AtomLanguageFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = build_grounded_language_program()
        cls.runtime, cls.history, cls.diagnostics = train_language_field(
            cls.program["train"],
            "word",
            config=LanguageConfig(),
        )

    def test_dataset_is_exact_and_keeps_evaluator_truth_separate(self) -> None:
        self.assertEqual(
            tuple(
                len(self.program[name]) for name in ("train", "validation", "heldout")
            ),
            (120, 24, 48),
        )
        self.assertTrue(
            all(
                "frame" not in row and "family" not in row
                for row in self.program["train"]
            )
        )
        context_only = [
            row
            for row in self.program["train"]
            if str(row["text"]).startswith(("who holds", "are they"))
        ]
        self.assertTrue(context_only)
        self.assertTrue(all(row["paraphrase_text"] for row in context_only))

    def test_only_the_universe_kernel_can_replace_immutable_state(self) -> None:
        state = UniverseLanguageKernel().initial_state("word")
        with self.assertRaises(FrozenInstanceError):
            state.temperature = 0.0  # type: ignore[misc]
        audit = architecture_audit()
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["replace_calls_outside_kernel"], [])

    def test_grounding_generalizes_and_language_is_bidirectional(self) -> None:
        evaluation = evaluate_language_rows(
            self.runtime,
            self.program["heldout"],
            self.program["evaluation_truth"],
        )
        self.assertEqual(evaluation["grounded_accuracy"], 1.0)
        self.assertEqual(evaluation["generation_roundtrip_accuracy"], 1.0)
        self.assertEqual(evaluation["grammar_validity"], 1.0)
        self.assertEqual(evaluation["reference_accuracy"], 1.0)
        self.assertEqual(self.diagnostics["unresolved_case_ids"], [])

    def test_abstraction_model_roundtrip_and_corruption_rejection(self) -> None:
        runtime, _, _ = train_language_field(
            self.program["train"],
            "word",
            config=LanguageConfig(),
        )
        runtime.abstract("unit-language-abstraction")
        model = language_model_payload(runtime)
        restored = runtime_from_language_model(json.loads(json.dumps(model)))
        self.assertEqual(
            language_model_payload(restored)["model_hash"], model["model_hash"]
        )
        self.assertEqual(model["raw_episode_count"], 0)
        self.assertEqual(model["raw_evidence_count"], 0)

        corruptions = []
        corrupt = copy.deepcopy(model)
        corrupt["unknown"] = True
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["lexeme_laws"][0]["mass"] = float("nan")
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["raw_episode_count"] = 1
        corruptions.append(corrupt)
        corrupt = copy.deepcopy(model)
        corrupt["model_hash"] = "0" * 64
        corruptions.append(corrupt)
        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index), self.assertRaises(ValueError):
                runtime_from_language_model(corrupt)

    def test_all_seven_primitives_are_exercised(self) -> None:
        operator_counts = dict(self.runtime.state.operator_counts)
        self.assertEqual(
            set(operator_counts),
            {primitive.value for primitive in Primitive},
        )
        self.assertTrue(all(operator_counts.values()))

    def test_serialized_request_fails_closed_on_unknown_or_invalid_input(self) -> None:
        valid = {
            "schema_version": LANGUAGE_MODEL_SCHEMA,
            "request_id": "strict-language-request",
            "stage": "word",
            "world": self.program["heldout"][0]["before"],
            "turns": [
                {
                    "turn_id": "one",
                    "mode": "interact",
                    "text": self.program["heldout"][0]["text"],
                }
            ],
        }
        validate_language_request(valid, self.runtime)

        unknown = copy.deepcopy(valid)
        unknown["unexpected"] = True
        with self.assertRaises(ValueError):
            validate_language_request(unknown, self.runtime)

        invalid = copy.deepcopy(valid)
        invalid["turns"][0]["text"] = ""
        with self.assertRaises(ValueError):
            validate_language_request(invalid, self.runtime)


if __name__ == "__main__":
    unittest.main()
