from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash
from atom_primitive_experiment import (
    PRIMITIVE_CONTINUAL_RUNTIME,
    PRIMITIVE_EXPERIMENT_RUNTIME,
    build_use_observation_request,
    ingest_use_observation,
    run_primitive_forge_experiment,
)
from atom_primitive_forge import (
    CANDIDATE_STATUS,
    CRYSTALLIZED_STATUS,
    QUARANTINED_STATUS,
    RETIRED_STATUS,
    REVISED_STATUS,
    Dimension,
    PortSignature,
    PrimitiveForge,
    TypeSignature,
)
from atom_primitive_knowledge import (
    PRIMITIVE_RAG_RUNTIME,
    PRIMITIVE_WIKI_RUNTIME,
    PrimitiveWikiGraph,
    retrieve_primitive_context,
    validate_knowledge_manifest,
)
from atom_primitive_side_view import (
    PRIMITIVE_SIDE_VIEW_RUNTIME,
    render_primitive_forge_artifact,
)
from atom_primitive_simulation import (
    counterfactual_world,
    evaluate_primitive,
    evaluate_root_expansion,
)


def _reseal(payload: dict[str, object]) -> None:
    core = {key: payload[key] for key in payload if key != "graph_hash"}
    payload["graph_hash"] = canonical_hash(core)


class AtomPrimitiveForgeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_primitive_forge_experiment(cls.output_dir)
        cls.model = json.loads(
            (cls.output_dir / "atom_primitive_graph.json").read_text(encoding="utf-8")
        )
        cls.workflow = json.loads(
            (cls.output_dir / "atom_primitive_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )
        cls.knowledge_manifest = json.loads(
            (cls.output_dir / "atom_primitive_knowledge_graph.json").read_text(
                encoding="utf-8"
            )
        )
        cls.side_view = (cls.output_dir / "atom_primitive_side_view.html").read_text(
            encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_inventory_grows_beyond_a_fixed_primitive_vocabulary(self) -> None:
        forge = PrimitiveForge.from_model_payload(self.model)
        self.assertEqual(forge.root_ids, tuple(ROOT_MECHANICS))
        self.assertEqual(self.report["inventory"]["root_count"], 7)
        self.assertGreater(self.report["inventory"]["derived_count"], 20)
        self.assertEqual(self.report["inventory"]["derived_count"], 62)
        before = len(forge.derived_records)
        derived = forge.derived_records
        novel = forge.compose(
            "serial",
            (derived[3].primitive_id, derived[11].primitive_id),
            provenance=("integration-test:open-growth",),
        )
        self.assertEqual(novel.status, QUARANTINED_STATUS)
        self.assertEqual(len(forge.derived_records), before + 1)

    def test_recursive_primitives_reuse_discoveries_and_expand_to_roots(
        self,
    ) -> None:
        forge = PrimitiveForge.from_model_payload(self.model)
        recursive = [
            record
            for record in forge.derived_records
            if record.recipe is not None
            and any(
                not forge.get(component).root for component in record.recipe.components
            )
        ]
        self.assertGreaterEqual(len(recursive), 18)
        for record in recursive:
            roots = forge.expand_to_roots(record.primitive_id)
            self.assertGreaterEqual(len(roots), 3)
            self.assertLessEqual(set(roots), set(ROOT_MECHANICS))

    def test_canonical_equivalence_merges_recipes_and_provenance(self) -> None:
        forge = PrimitiveForge()
        forward = forge.compose(
            "parallel",
            ("radiation", "dissipation"),
            provenance=("path:forward",),
        )
        reverse = forge.compose(
            "parallel",
            ("dissipation", "radiation"),
            provenance=("path:reverse",),
        )
        self.assertEqual(forward.primitive_id, reverse.primitive_id)
        merged = forge.get(forward.primitive_id)
        self.assertEqual(len(merged.equivalent_recipes), 1)
        self.assertTrue(
            {"path:forward", "path:reverse"}.issubset(set(merged.provenance))
        )

        left = forge.compose("serial", ("radiation", "dissipation"))
        nested_left = forge.compose(
            "serial",
            (left.primitive_id, "gravitation"),
        )
        right = forge.compose("serial", ("dissipation", "gravitation"))
        nested_right = forge.compose(
            "serial",
            ("radiation", right.primitive_id),
        )
        self.assertEqual(
            nested_left.primitive_id,
            nested_right.primitive_id,
        )

    def test_type_dimension_and_unknown_reference_rejection(self) -> None:
        forge = PrimitiveForge()
        length_port = PortSignature(
            kind="bounded_scalar_field",
            dimension=Dimension((("length", 1),)),
        )
        wrong_signature = TypeSignature(
            domain="mathematical_scalar_field",
            inputs=(length_port,),
            output=length_port,
        )
        with self.assertRaisesRegex(ValueError, "type/dimension"):
            forge.compose(
                "serial",
                ("radiation", "dissipation"),
                expected_signature=wrong_signature,
            )
        with self.assertRaisesRegex(ValueError, "unknown primitive"):
            forge.compose("serial", ("radiation", "not-observed"))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            forge.compose(
                "feedback",
                ("radiation", "dissipation", "gravitation"),
            )

    def test_serialization_rejects_corruption_cycles_and_root_mutation(
        self,
    ) -> None:
        corrupt = copy.deepcopy(self.model)
        derived = next(item for item in corrupt["primitives"] if not item["root"])
        derived["confidence"] = 0.123
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            PrimitiveForge.from_model_payload(corrupt)

        unknown = copy.deepcopy(self.model)
        unknown_derived = next(
            item for item in unknown["primitives"] if not item["root"]
        )
        unknown_derived["recipe"]["components"][0] = "missing-reference"
        _reseal(unknown)
        with self.assertRaisesRegex(ValueError, "unknown primitive"):
            PrimitiveForge.from_model_payload(unknown)

        cyclic = copy.deepcopy(self.model)
        cyclic_derived = [item for item in cyclic["primitives"] if not item["root"]][:2]
        cyclic_derived[0]["recipe"]["components"][0] = cyclic_derived[1]["primitive_id"]
        cyclic_derived[1]["recipe"]["components"][0] = cyclic_derived[0]["primitive_id"]
        _reseal(cyclic)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            PrimitiveForge.from_model_payload(cyclic)

        root_mutation = copy.deepcopy(self.model)
        root = next(item for item in root_mutation["primitives"] if item["root"])
        root["confidence"] = 0.9
        _reseal(root_mutation)
        with self.assertRaisesRegex(ValueError, "immutable root"):
            PrimitiveForge.from_model_payload(root_mutation)

    def test_continual_learning_promotes_revises_and_decays(self) -> None:
        forge = PrimitiveForge()
        proposed = forge.compose(
            "serial",
            ("radiation", "dissipation"),
        )
        self.assertEqual(proposed.status, QUARANTINED_STATUS)
        for trial in range(3):
            updated = forge.observe(
                proposed.primitive_id,
                context_id=f"use:{trial}",
                predicted=0.2 + trial,
                observed=0.2 + trial,
            )
        self.assertEqual(updated.status, CRYSTALLIZED_STATUS)
        revised = forge.observe(
            proposed.primitive_id,
            context_id="use:contradiction",
            predicted=0.3,
            observed=0.8,
        )
        self.assertEqual(revised.status, REVISED_STATUS)
        retired = forge.apply_decay(
            proposed.primitive_id,
            amount=0.8,
        )
        self.assertEqual(retired.status, RETIRED_STATUS)
        with self.assertRaisesRegex(ValueError, "immutable root"):
            forge.apply_decay("radiation", amount=0.5)
        with self.assertRaises(FrozenInstanceError):
            forge.get("radiation").status = CANDIDATE_STATUS  # type: ignore[misc]

    def test_hash_bound_live_use_path_persists_an_observation(self) -> None:
        forge = PrimitiveForge.from_model_payload(self.model)
        target = next(
            record
            for record in forge.derived_records
            if record.status == QUARANTINED_STATUS
        )
        model_path = self.output_dir / "live-input-model.json"
        request_path = self.output_dir / "live-observation.json"
        output_path = self.output_dir / "live-output-model.json"
        forge.save(model_path)
        request = build_use_observation_request(
            primitive_id=target.primitive_id,
            context_id="integration-live-use",
            predicted=0.42,
            observed=0.42,
            source="integration-test",
        )
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        response = ingest_use_observation(
            model_path,
            request_path,
            output_path,
        )
        updated = PrimitiveForge.load(output_path)
        self.assertEqual(response["runtime"], PRIMITIVE_CONTINUAL_RUNTIME)
        self.assertNotEqual(
            response["before_graph_hash"],
            response["after_graph_hash"],
        )
        self.assertEqual(
            updated.get(target.primitive_id).status,
            CANDIDATE_STATUS,
        )

        tampered = copy.deepcopy(request)
        tampered["observed"] = 0.9
        request_path.write_text(
            json.dumps(tampered),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ingest_use_observation(model_path, request_path, output_path)

    def test_heldout_and_counterfactual_transfer_are_measured(self) -> None:
        evaluation = self.report["evaluation"]
        self.assertEqual(evaluation["heldout_count"], 36)
        self.assertEqual(evaluation["counterfactual_count"], 36)
        self.assertEqual(evaluation["heldout_accuracy"], 1.0)
        self.assertEqual(evaluation["counterfactual_accuracy"], 1.0)
        self.assertTrue(all(item["passed"] for item in evaluation["heldout_records"]))
        self.assertTrue(
            all(item["passed"] for item in evaluation["counterfactual_records"])
        )
        forge = PrimitiveForge.from_model_payload(self.model)
        recursive = next(
            record
            for record in forge.derived_records
            if record.recipe is not None
            and any(
                not forge.get(component).root for component in record.recipe.components
            )
        )
        world = counterfactual_world(999)
        prediction = evaluate_primitive(
            forge,
            recursive.primitive_id,
            0.271,
            world,
        )
        observed = evaluate_root_expansion(
            forge.expand_to_roots(recursive.primitive_id),
            0.271,
            world,
        )
        self.assertAlmostEqual(prediction, observed, places=12)

    def test_graph_native_wiki_and_rag_are_artifact_bound(self) -> None:
        forge = PrimitiveForge.from_model_payload(self.model)
        graph = PrimitiveWikiGraph(forge)
        graph.assert_bound_to(forge)
        target = self.workflow["knowledge_context"][0]["primitive_id"]
        context = retrieve_primitive_context(
            graph,
            f"{target} recursive root expansion",
        )
        self.assertEqual(context[0]["primitive_id"], target)
        self.assertTrue(context[0]["root_expansion"])
        self.assertTrue(context[0]["neighbors"])
        self.assertEqual(
            self.workflow["runtime"]["wiki"],
            PRIMITIVE_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.workflow["runtime"]["rag"],
            PRIMITIVE_RAG_RUNTIME,
        )
        validate_knowledge_manifest(self.knowledge_manifest, forge)
        tampered = copy.deepcopy(self.knowledge_manifest)
        tampered["nodes"][0]["description"] += " changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_knowledge_manifest(tampered, forge)

    def test_side_view_binds_the_real_graph_and_rejects_tampering(self) -> None:
        rendered = render_primitive_forge_artifact(
            self.model,
            self.report,
            self.workflow,
            self.knowledge_manifest,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn(
            '<aside aria-label="Primitive artifact side view">',
            rendered,
        )
        self.assertIn(self.model["graph_hash"], rendered)
        self.assertIn("Discovered recursive inventory", rendered)
        self.assertEqual(
            self.report["side_view_contract"]["runtime"],
            PRIMITIVE_SIDE_VIEW_RUNTIME,
        )
        tampered_model = copy.deepcopy(self.model)
        tampered_model["sequence"] += 1
        with self.assertRaises(ValueError):
            render_primitive_forge_artifact(
                tampered_model,
                self.report,
                self.workflow,
                self.knowledge_manifest,
            )
        tampered_workflow = copy.deepcopy(self.workflow)
        tampered_workflow["artifact_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            render_primitive_forge_artifact(
                self.model,
                self.report,
                tampered_workflow,
                self.knowledge_manifest,
            )

    def test_architecture_and_runtime_declarations_are_universe_first(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        architecture = json.loads(
            (root / "atom-primitive-forge-architecture.json").read_text(
                encoding="utf-8"
            )
        )
        registry = json.loads(
            (root / "ai-runtime-registry.json").read_text(encoding="utf-8")
        )
        causal_live = registry["runtimes"]["causal-live"]
        causal_memory = json.loads(
            (root / "atom-causal-memory-architecture.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            architecture["root_substrate"],
            list(ROOT_MECHANICS),
        )
        self.assertEqual(
            architecture["hierarchy"][1],
            "unbounded recursively discovered primitives",
        )
        self.assertEqual(
            architecture["later_projections"],
            ["coding", "Rust", "frontend"],
        )
        self.assertEqual(
            causal_live["runtime_entrypoint"],
            "atom_causal_live_experiment.py",
        )
        self.assertEqual(
            causal_live["integration_test"],
            "tests/test_atom_causal_live_integration.py",
        )
        self.assertEqual(
            causal_memory["causal_glyph"]["catalog"],
            ("hash-bound Primitive Forge graph selected through a causal Atom DB root"),
        )
        self.assertEqual(
            causal_memory["retrieval"]["method"],
            (
                "exact graph-motif resonance with role-specific "
                "activation and durable conductance"
            ),
        )
        self.assertEqual(
            self.report["runtime"],
            PRIMITIVE_EXPERIMENT_RUNTIME,
        )
        self.assertIn("does not establish complete physics", self.report["claim_scope"])


if __name__ == "__main__":
    unittest.main()
