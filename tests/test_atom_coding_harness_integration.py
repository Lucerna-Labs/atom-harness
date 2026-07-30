from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_causal_world_schema import canonical_hash
from atom_coding_experiment import (
    run_coding_experiment,
    run_coding_workflow,
)
from atom_coding_harness import (
    CODE_GRAPH_RUNTIME,
    CodeCausalGraph,
    CodeStructureGraph,
    HiddenPlatformEvaluator,
    IsolatedPythonInterventionRunner,
    coding_harness_self_test,
)
from atom_coding_knowledge import (
    CODING_RAG_RUNTIME,
    CODING_WIKI_RUNTIME,
    CodingWikiGraph,
    coding_knowledge_self_test,
    retrieve_coding_context,
)
from atom_coding_side_view import (
    ATOM_CODING_SIDE_VIEW_RUNTIME,
    render_coding_artifact,
)
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    PLATFORM_PRIMITIVES,
    SPIDERWEB_LAYERS,
    platform_curriculum,
    platform_synthesis_self_test,
)


class AtomCodingHarnessIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_coding_experiment(cls.output_dir)
        cls.model = json.loads(
            (cls.output_dir / "atom_coding_model.json").read_text(encoding="utf-8")
        )
        cls.request = json.loads(
            (
                cls.output_dir / "atom_coding_workflow_request.json"
            ).read_text(encoding="utf-8")
        )
        cls.workflow = json.loads(
            (
                cls.output_dir / "atom_coding_workflow_response.json"
            ).read_text(encoding="utf-8")
        )
        cls.source = (
            cls.output_dir / "atom_generated_platform.py"
        ).read_text(encoding="utf-8")
        cls.side_view = (
            cls.output_dir / "atom_coding_side_view.html"
        ).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_mathematical_platform_registry_and_generated_runtime(self) -> None:
        self.assertEqual(len(PLATFORM_PRIMITIVES), 9)
        self.assertEqual(len(PLATFORM_CAPABILITY_PRIMITIVES), 9)
        self.assertEqual(len(SPIDERWEB_LAYERS), 4)
        self.assertTrue(all(platform_synthesis_self_test().values()))
        self.assertIn("def execute(request):", self.source)
        self.assertIn("def platform_manifest():", self.source)

    def test_learned_graph_beats_baseline_on_unseen_platform_specs(self) -> None:
        benchmark = self.report["benchmark"]
        self.assertEqual(benchmark["atom"]["capability_passes"], 31)
        self.assertEqual(benchmark["atom"]["capability_count"], 31)
        self.assertEqual(benchmark["atom"]["full_passes"], 7)
        self.assertEqual(benchmark["baseline"]["capability_passes"], 15)
        self.assertEqual(benchmark["baseline"]["full_passes"], 2)
        self.assertGreater(
            benchmark["atom_capability_score"],
            benchmark["baseline_capability_score"],
        )
        self.assertEqual(benchmark["partitions"], ["validation", "heldout"])

    def test_model_is_hash_bound_persistent_and_corruption_fails_closed(self) -> None:
        restored = CodeCausalGraph.from_model_payload(self.model)
        self.assertEqual(restored.model_payload(), self.model)
        changed = json.loads(json.dumps(self.model))
        changed["laws"][0]["confidence"] = 0.0
        with self.assertRaises(ValueError):
            CodeCausalGraph.from_model_payload(changed)

    def test_causal_laws_preserve_conjunctive_causes(self) -> None:
        graph = CodeCausalGraph.from_model_payload(self.model)
        recognized = graph.recognize(
            ("parallel_promotion", "emergent_topology")
        )
        self.assertIsNotNone(recognized)
        assert recognized is not None
        self.assertEqual(
            set(recognized["parallel_promotion"]),
            {"directed_relation", "composition"},
        )
        self.assertEqual(
            set(recognized["emergent_topology"]),
            {"directed_relation", "topology"},
        )

    def test_live_workflow_uses_knowledge_and_executes_hidden_behaviors(self) -> None:
        response, source = run_coding_workflow(self.model, self.request)
        self.assertEqual(response["claim_status"], "derived")
        self.assertTrue(response["evaluation"]["passed"])
        self.assertEqual(response["runtime"]["wiki"], CODING_WIKI_RUNTIME)
        self.assertEqual(response["runtime"]["rag"], CODING_RAG_RUNTIME)
        self.assertTrue(response["knowledge_context"])
        self.assertEqual(
            response["artifact_hash"],
            canonical_hash({"source": source}),
        )
        full = next(
            spec
            for spec in platform_curriculum()
            if spec.spec_id == "heldout-spiderweb-platform"
        )
        evaluator = HiddenPlatformEvaluator(IsolatedPythonInterventionRunner())
        self.assertTrue(evaluator.evaluate(source, full).passed)

    def test_code_structure_graph_is_bound_to_the_real_artifact(self) -> None:
        structure = CodeStructureGraph.from_source(self.source)
        manifest = structure.manifest()
        self.assertEqual(manifest["runtime"], CODE_GRAPH_RUNTIME)
        self.assertEqual(manifest["source_hash"], self.workflow["artifact_hash"])
        self.assertTrue(
            any(edge["relation"] == "defines" for edge in manifest["edges"])
        )
        self.assertTrue(
            any(edge["relation"] == "calls" for edge in manifest["edges"])
        )

    def test_ablations_show_memory_composition_and_persistence_are_required(
        self,
    ) -> None:
        ablations = self.report["ablations"]
        self.assertLess(ablations["no_causal_memory_score"], 1.0)
        self.assertLess(ablations["no_phase_mixing_score"], 1.0)
        self.assertTrue(ablations["no_topological_persistence_abstained"])
        self.assertTrue(ablations["unexperienced_capability_abstained"])

    def test_wiki_graph_and_rag_are_runtime_wired(self) -> None:
        graph = CodingWikiGraph()
        manifest = graph.manifest()
        hits = retrieve_coding_context(
            graph,
            "parallel routing with backpressure and emergent intersections",
        )
        self.assertEqual(manifest["wiki_runtime"], CODING_WIKI_RUNTIME)
        self.assertEqual(manifest["rag_runtime"], CODING_RAG_RUNTIME)
        self.assertTrue(hits)
        self.assertTrue(all(coding_knowledge_self_test().values()))

    def test_user_visible_side_view_binds_the_real_artifact(self) -> None:
        rendered = render_coding_artifact(
            self.model,
            self.report,
            self.workflow,
            self.source,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn('<aside aria-label="Causal artifact evidence">', rendered)
        self.assertIn("Generated mathematical-primitive platform", rendered)
        self.assertIn(ATOM_CODING_SIDE_VIEW_RUNTIME, str(self.report))
        changed_source = self.source + "\n"
        with self.assertRaises(ValueError):
            render_coding_artifact(
                self.model,
                self.report,
                self.workflow,
                changed_source,
            )

    def test_candidate_runtime_rejects_invalid_and_unsupported_requests(self) -> None:
        runner = IsolatedPythonInterventionRunner()
        invalid = runner.run(self.source, {"action": "route"})
        unsupported = runner.run(
            self.source,
            {"action": "delete_everything", "payload": {}},
        )
        self.assertFalse(invalid.succeeded)
        self.assertFalse(unsupported.succeeded)

    def test_all_declared_experiment_gates_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(all(self.report["checks"].values()))
        self.assertTrue(all(coding_harness_self_test().values()))


if __name__ == "__main__":
    unittest.main()
