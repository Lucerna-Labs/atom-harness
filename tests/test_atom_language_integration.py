from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_language_dataset import build_grounded_language_program
from atom_language_experiment import build_workflow, score_workflow
from atom_language_field import (
    ATOM_LANGUAGE_RUNTIME,
    LanguageConfig,
    character_span_f1,
    evaluate_language_rows,
    language_model_payload,
    run_language_workflow,
    train_language_field,
)
from atom_language_side_view import (
    ATOM_LANGUAGE_ARTIFACT_BINDING,
    ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
    render_language_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    build_language_graph,
    retrieve_atom_context,
)


class AtomLanguageRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = build_grounded_language_program()
        cls.runtime, _, cls.diagnostics = train_language_field(
            cls.program["train"],
            "character",
            config=LanguageConfig(),
        )
        cls.evaluation = evaluate_language_rows(
            cls.runtime,
            cls.program["heldout"],
            cls.program["evaluation_truth"],
        )
        cls.spans = character_span_f1(cls.runtime, cls.program["heldout"])
        cls.final_temperature = cls.runtime.state.temperature
        cls.runtime.abstract("integration-language-abstraction")
        cls.model = language_model_payload(cls.runtime)

    def test_graph_rag_workflow_and_side_view_share_real_artifact(self) -> None:
        graph = build_language_graph()
        retrieved = retrieve_atom_context(
            graph,
            "ground character lexeme grammar role context pronoun speak",
            limit=16,
        )
        names = {row["name"] for row in retrieved}
        self.assertTrue(
            {
                "ground",
                "lexical_nucleation",
                "role_bind",
                "understand",
                "resolve_reference",
                "speak",
            }
            <= names
        )

        request, expected = build_workflow(self.program, self.model)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model.json"
            request_path = root / "request.json"
            response_path = root / "response.json"
            model_path.write_text(json.dumps(self.model), encoding="utf-8")
            request_path.write_text(json.dumps(request), encoding="utf-8")
            response = run_language_workflow(
                model_path,
                request_path,
                response_path,
            )
            workflow = score_workflow(response, expected)
            self.assertTrue(workflow["passed"], workflow)

            first_meaning = next(
                turn["meaning"] for turn in response["turns"] if turn.get("meaning")
            )
            report = {
                "experiment": "atom_grounded_language_field_integration",
                "primary_model_hash": self.model["model_hash"],
                "stages": {
                    "word": {"heldout": self.evaluation},
                    "character": {
                        "heldout": self.evaluation,
                        "character_spans": self.spans,
                    },
                },
                "experiment_gates": {
                    "gates": {"graph_rag": True, "serialized_workflow": True}
                },
                "side_view_interaction": {
                    "utterance": request["turns"][0]["text"],
                    "meaning": first_meaning,
                    "answer": None,
                    "semantic_mass": "2 roles -> 0 unexpressed",
                },
                "controlled_chaos": {
                    "initial_temperature": LanguageConfig().initial_temperature,
                    "final_temperature": self.final_temperature,
                },
            }
            side_path = render_language_artifact(
                report,
                self.model,
                root / "side.html",
            )
            document = side_path.read_text(encoding="utf-8")
            self.assertIn(self.model["model_hash"], document)
            self.assertIn(ATOM_LANGUAGE_SIDE_VIEW_RUNTIME, document)
            self.assertIn(ATOM_LANGUAGE_ARTIFACT_BINDING, document)
            self.assertIn("Real held-out interaction", document)

        self.assertEqual(response["runtime"]["language_runtime"], ATOM_LANGUAGE_RUNTIME)
        self.assertEqual(response["runtime"]["wiki_runtime"], ATOM_WIKI_GRAPH_RUNTIME)
        self.assertEqual(response["runtime"]["rag_runtime"], ATOM_RAG_RUNTIME)


if __name__ == "__main__":
    unittest.main()
