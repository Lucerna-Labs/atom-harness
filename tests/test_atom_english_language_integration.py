from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_english_language_dataset import (
    ENGLISH_FUNCTION_WORDS,
    build_english_language_program,
    normalize_english_request,
)
from atom_english_language_experiment import (
    ENGLISH_EXPERIMENT_RUNTIME,
    run_english_language_experiment,
    run_english_workflow,
)
from atom_english_language_model import (
    ENGLISH_LANGUAGE_RUNTIME,
    english_model_hash,
    load_english_language_model,
    run_english_inference_request,
)
from atom_english_language_side_view import (
    ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_english_language_artifact,
)
from atom_field_proof import PROCESS_NAMES
from atom_neural_language_dataset import RUNTIME_ROW_KEYS
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_english_language_graph,
)


class AtomEnglishLanguageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_english_language_experiment(cls.output_dir)
        cls.program = build_english_language_program()
        cls.model = json.loads(
            (cls.output_dir / "atom_english_language_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow_request = json.loads(
            (cls.output_dir / "atom_english_language_workflow_request.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow_response = json.loads(
            (cls.output_dir / "atom_english_language_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_natural_english_surface_is_separate_from_compact_runtime(self) -> None:
        self.assertEqual(sum(self.report["dataset"]["stages"].values()), 2230)
        self.assertTrue(self.report["dataset"]["audit"]["passed"])
        self.assertEqual(
            normalize_english_request(
                "Please spread, then report the node with greatest signal."
            ),
            "eng spread signal",
        )
        for rows in self.program.stages.values():
            self.assertTrue(all(set(row) == RUNTIME_ROW_KEYS for row in rows))
            for row in rows:
                tokens = str(row["utterance"]).split()
                self.assertEqual(tokens[0], "eng")
                self.assertFalse(set(tokens) & ENGLISH_FUNCTION_WORDS)
                truth = self.program.evaluator_truth[str(row["event_id"])]
                self.assertNotEqual(truth["user_utterance"], row["utterance"])
                self.assertFalse(str(truth["user_utterance"]).startswith("eng "))

    def test_metaplastic_policy_is_selected_before_final_evaluation(self) -> None:
        selection = self.report["selection"]
        evaluations = self.report["evaluations"]
        selected = evaluations["selected"]
        self.assertTrue(selection["final_evaluation_hidden_during_selection"])
        self.assertEqual(selection["selection_cases"], 42)
        self.assertIn(selection["policy"], {"adaptive", "fixed"})
        self.assertEqual(selected, evaluations[selection["policy"]])
        self.assertGreaterEqual(
            selected["transfer_composition"]["joint_accuracy"], 0.90
        )
        self.assertGreaterEqual(selected["base_composition"]["joint_accuracy"], 0.80)
        self.assertLessEqual(
            evaluations["flat"]["base_composition"]["joint_accuracy"], 0.10
        )
        self.assertEqual(self.report["sample_efficiency"]["example_fraction"], 0.25)
        self.assertGreaterEqual(
            self.report["sample_efficiency"]["evaluation"]["joint_accuracy"],
            0.80,
        )

    def test_evidence_gate_abstains_on_unsupported_and_oov_english(self) -> None:
        evidence = self.report["evidence_boundary"]
        self.assertGreaterEqual(evidence["supported"]["coverage"], 0.90)
        self.assertGreaterEqual(evidence["supported"]["assertion_accuracy"], 0.90)
        self.assertEqual(evidence["unsupported"]["correct_abstention_rate"], 1.0)
        self.assertGreaterEqual(evidence["compute"]["reduction"], 0.50)

        loaded = load_english_language_model(self.model)
        source = self.workflow_request["turns"][0]
        response = run_english_inference_request(
            loaded,
            {
                "adjacency": source["adjacency"],
                "node_features": source["node_features"],
                "request_id": "oov-English-request",
                "utterance": "Please teleport, then report the strongest signal.",
            },
        )
        artifact = response["artifact"]
        self.assertEqual(artifact["claim_status"], "unknown")
        self.assertIsNone(artifact["assertion"])
        self.assertEqual(artifact["unknown_tokens"], ["teleport"])
        self.assertTrue(artifact["reasoning"]["execution_skipped"])
        self.assertIn("grounded evidence", artifact["answer"])

    def test_all_root_operators_remain_causal_and_state_is_bounded(self) -> None:
        self.assertEqual(set(self.report["ablations"]["text"]), set(PROCESS_NAMES))
        self.assertEqual(set(self.report["ablations"]["field"]), set(PROCESS_NAMES))
        self.assertTrue(self.report["ablations"]["all_text_operators_causal"])
        self.assertTrue(self.report["ablations"]["all_field_operators_causal"])
        self.assertLessEqual(self.report["controller"]["maximum_chaos_load"], 1.150001)
        self.assertEqual(
            self.report["training"]["adaptation"]["adaptive"]["raw_event_count"],
            0,
        )
        self.assertTrue(self.report["architecture_audit"]["passed"])
        self.assertTrue(self.report["corruption_checks"]["passed"])
        self.assertEqual(self.report["corruption_checks"]["rejected"], 5)
        self.assertTrue(self.report["serialization"]["roundtrip_exact"])
        self.assertTrue(self.report["deterministic_training"]["passed"])
        behavior = self.report["behavior_contract"]
        payload = {
            key: value for key, value in behavior.items() if key != "behavior_sha256"
        }
        self.assertEqual(behavior["behavior_sha256"], english_model_hash(payload))

    def test_graph_rag_workflow_and_side_view_bind_the_model(self) -> None:
        graph = build_english_language_graph()
        self.assertEqual(
            set(graph.expand("evidence_bound_english_answer")), set(PROCESS_NAMES)
        )
        response = run_english_workflow(self.model, self.workflow_request)
        self.assertEqual(response, self.workflow_response)
        self.assertEqual(
            response["runtime"]["english_runtime"], ENGLISH_LANGUAGE_RUNTIME
        )
        self.assertEqual(
            response["runtime"]["experiment_runtime"], ENGLISH_EXPERIMENT_RUNTIME
        )
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)
        self.assertEqual(len(response["turns"]), 8)
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        self.assertTrue(self.report["serialized_workflow"]["passed"])

        document = (self.output_dir / "atom_english_language_side_view.html").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            render_english_language_artifact(
                self.model,
                self.report,
                self.workflow_response,
            ),
            document,
        )
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME, document)
        self.assertIn("render_english_language_artifact", document)
        self.assertNotIn("<button", document)
        self.assertNotIn("<input", document)
        self.assertTrue(self.report["experiment_gates"]["passed"])


if __name__ == "__main__":
    unittest.main()
