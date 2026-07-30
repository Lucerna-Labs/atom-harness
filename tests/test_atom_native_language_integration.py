from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atom_causal_world_schema import ROOT_MECHANICS
from atom_coding_harness import CodeCausalGraph
from atom_coding_knowledge import (
    CODING_RAG_RUNTIME,
    CODING_WIKI_RUNTIME,
    CodingWikiGraph,
    coding_knowledge_self_test,
    retrieve_coding_context,
)
from atom_frontend_target import (
    ATOM_FRONTEND_TARGET_RUNTIME,
    frontend_target_self_test,
)
from atom_language import (
    ATOM_LANGUAGE_RUNTIME,
    ATOM_TRAINING_RUNTIME,
    atom_language_self_test,
    parse_atom_source,
)
from atom_native_experiment import (
    ATOM_NATIVE_EXPERIMENT_RUNTIME,
    run_atom_native_experiment,
)
from atom_native_side_view import (
    ATOM_NATIVE_SIDE_VIEW_RUNTIME,
    render_atom_native_artifact,
)
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    platform_curriculum,
)
from atom_rust_target import (
    ATOM_RUST_TARGET_RUNTIME,
    rust_target_self_test,
)


class AtomNativeLanguageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_atom_native_experiment(cls.output_dir)
        cls.model = json.loads(
            (cls.output_dir / "atom_native_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.workflow = json.loads(
            (
                cls.output_dir / "atom_native_workflow_response.json"
            ).read_text(encoding="utf-8")
        )
        cls.atom_source = (
            cls.output_dir / "atom_generated_platform.atom"
        ).read_text(encoding="utf-8")
        rust_file = "lib." + "r" + "s"
        cls.rust_source = (
            cls.output_dir
            / "atom_generated_rust_platform"
            / "src"
            / rust_file
        ).read_text(encoding="utf-8")
        cls.frontend_component = (
            cls.output_dir
            / "atom_generated_frontend"
            / "src"
            / "AtomPlatform.svelte"
        ).read_text(encoding="utf-8")
        cls.side_view = (
            cls.output_dir / "atom_native_side_view.html"
        ).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_atom_is_primary_language_and_only_root_substrate(self) -> None:
        program = parse_atom_source(self.atom_source)
        self.assertEqual(program.roots, ROOT_MECHANICS)
        self.assertEqual(
            self.report["runtime"],
            ATOM_NATIVE_EXPERIMENT_RUNTIME,
        )
        self.assertEqual(
            self.workflow["runtime"]["atom_language"],
            ATOM_LANGUAGE_RUNTIME,
        )
        self.assertEqual(
            self.report["learning"]["runtime"],
            ATOM_TRAINING_RUNTIME,
        )
        self.assertIn("Primary construction language", self.side_view)
        self.assertIn("<h1>Atom</h1>", self.side_view)

    def test_atom_parser_round_trips_and_rejects_invalid_root_expansion(
        self,
    ) -> None:
        parsed = parse_atom_source(self.atom_source)
        self.assertEqual(
            parsed.manifest()["program_hash"],
            self.report["program"]["program_hash"],
        )
        invalid = self.atom_source.replace(
            "primitive identity <- conservation",
            "primitive identity <- radiation",
            1,
        )
        with self.assertRaises(ValueError):
            parse_atom_source(invalid)
        with self.assertRaises(ValueError):
            parse_atom_source(self.atom_source + "unknown directive\n")

    def test_atom_native_learning_discovers_every_conjunctive_binding(
        self,
    ) -> None:
        graph = CodeCausalGraph.from_model_payload(self.model)
        self.assertTrue(
            all(
                "substrate:atom" in context
                for law in self.model["laws"]
                for context in law["contexts"]
            )
        )
        for capability, expected in PLATFORM_CAPABILITY_PRIMITIVES.items():
            recognized = graph.recognize((capability,))
            self.assertIsNotNone(recognized)
            assert recognized is not None
            self.assertIn(expected, recognized[capability])
        self.assertEqual(
            set(graph.recognize(("parallel_promotion",))["parallel_promotion"]),
            {"directed_relation", "composition"},
        )
        self.assertEqual(
            set(graph.recognize(("emergent_topology",))["emergent_topology"]),
            {"directed_relation", "topology"},
        )
        self.assertIsNone(graph.recognize(("never_experienced",)))

    def test_atom_beats_fixed_baseline_on_every_unseen_program(self) -> None:
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

    def test_rust_is_a_hash_bound_executable_atom_projection(self) -> None:
        benchmark = self.report["benchmark"]
        self.assertEqual(
            self.workflow["runtime"]["rust_target"],
            ATOM_RUST_TARGET_RUNTIME,
        )
        self.assertEqual(benchmark["rust_full_passes"], 7)
        self.assertTrue(
            all(
                record["compiled"]
                and record["executed"]
                and record["passed"]
                for record in benchmark["rust_records"]
            )
        )
        self.assertIn(self.report["program"]["program_hash"], self.rust_source)
        self.assertTrue(self.report["cargo_validation"]["passed"])
        self.assertIn(
            "generated_atom_program_behaves_as_declared",
            self.rust_source,
        )
        self.assertIn("invalid_requests_fail_closed", self.rust_source)
        self.assertIn(
            "2 passed; 0 failed",
            self.report["cargo_validation"]["stdout"],
        )

    def test_frontend_is_a_thin_wired_projection(self) -> None:
        validation = self.report["frontend_validation"]
        self.assertEqual(
            self.workflow["runtime"]["frontend_target"],
            ATOM_FRONTEND_TARGET_RUNTIME,
        )
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["result"]["warningCount"], 0)
        self.assertIn("onclick={runProjection}", self.frontend_component)
        self.assertIn("bridge.execute", self.frontend_component)
        self.assertIn("Run through Rust", self.frontend_component)
        self.assertNotIn("function route", self.frontend_component)
        self.assertNotIn("function project", self.frontend_component)

    def test_wiki_graph_and_rag_are_on_the_runtime_path(self) -> None:
        graph = CodingWikiGraph()
        context = retrieve_coding_context(
            graph,
            "Atom Rust Svelte typed causal platform",
        )
        self.assertEqual(self.workflow["runtime"]["wiki"], CODING_WIKI_RUNTIME)
        self.assertEqual(self.workflow["runtime"]["rag"], CODING_RAG_RUNTIME)
        self.assertTrue(self.workflow["knowledge_context"])
        self.assertTrue(context)
        self.assertTrue(all(coding_knowledge_self_test().values()))

    def test_side_view_renders_and_binds_all_three_real_artifacts(self) -> None:
        base_rust = self.rust_source.split("\n\n#[cfg(test)]", 1)[0]
        rendered = render_atom_native_artifact(
            self.model,
            self.report,
            self.workflow,
            self.atom_source,
            base_rust,
            self.frontend_component,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn(
            '<aside aria-label="Replaceable target projections">',
            rendered,
        )
        self.assertIn("Rust execution projection", rendered)
        self.assertIn("Thin Svelte projection", rendered)
        self.assertEqual(
            self.report["side_view_contract"]["runtime"],
            ATOM_NATIVE_SIDE_VIEW_RUNTIME,
        )
        with self.assertRaises(ValueError):
            render_atom_native_artifact(
                self.model,
                self.report,
                self.workflow,
                self.atom_source + "\n",
                base_rust,
                self.frontend_component,
            )

    def test_language_target_and_frontend_self_tests_pass(self) -> None:
        program = parse_atom_source(self.atom_source)
        full_spec = next(
            spec
            for spec in platform_curriculum()
            if spec.spec_id == "heldout-spiderweb-platform"
        )
        self.assertTrue(all(atom_language_self_test().values()))
        self.assertTrue(all(rust_target_self_test(program, full_spec).values()))
        validator_dir = (
            Path(__file__).resolve().parents[1]
            / "tooling"
            / "svelte-validator"
        )
        self.assertTrue(
            all(frontend_target_self_test(program, validator_dir).values())
        )

    def test_all_experiment_gates_pass(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(all(self.report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
