from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atom_causal_graph import (
    CONTEXT_FACTOR_GRAPH_RUNTIME,
    CausalCognition,
    CausalGraph,
    CausalQuery,
    law_condition_signature,
    project_context_factor_trace,
    stable_condition_signature,
)
from atom_formal_domains import FORMAL_DOMAIN_NAMES
from atom_causal_world_accelerator import (
    build_accelerator_plan,
    run_jax_massive_shard,
)
from atom_causal_world_curriculum import (
    curriculum_manifest,
    curriculum_programs,
    decode_world_program,
    world_program_space_size,
)
from atom_causal_world_experiment import (
    _run_accelerator_learning,
    build_causal_workflow_request,
    build_causal_resume_cursor,
    load_causal_resume_state,
    run_causal_workflow,
    run_causal_world_experiment,
)
from atom_causal_world_knowledge import (
    CAUSAL_WORLD_RAG_RUNTIME,
    CAUSAL_WORLD_WIKI_RUNTIME,
    CausalWorldWikiGraph,
    retrieve_causal_context,
)
from atom_causal_world_schema import (
    ARCHITECTURE_COMPONENTS,
    DOMAIN_NAMES,
    CausalEvidence,
    FEATURE_INDEX,
    canonical_hash,
    get_profile,
)
from atom_causal_world_side_view import (
    ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME,
    render_causal_world_artifact,
)
from atom_causal_world_language import (
    parse_causal_question,
    render_causal_question,
)
from atom_causal_world_simulator import (
    ProceduralWorldCompiler,
    _apply_programmed_relations,
    _apply_programmed_state,
    advance_world,
)
from atom_causal_world_transfer import (
    TRANSFER_RISK_METHOD,
    build_transfer_request,
    evaluate_transfer_response,
    factor_projection_lattice_hash,
    factor_probe_trace_hash,
    fit_transfer_policy,
    run_transfer_workflow,
    selective_error_upper_bound,
    select_heldout_program_ids,
    stable_transfer_evidence_provenance,
    validate_transfer_policy_artifact,
)


class AtomCausalWorldIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temporary.name)
        cls.report = run_causal_world_experiment(
            cls.output_dir,
            profile="test",
            backend="numpy",
        )
        cls.model = json.loads(
            (cls.output_dir / "atom_causal_world_model.json").read_text(
                encoding="utf-8"
            )
        )
        cls.request = json.loads(
            (cls.output_dir / "atom_causal_world_workflow_request.json").read_text(
                encoding="utf-8"
            )
        )
        cls.response = json.loads(
            (cls.output_dir / "atom_causal_world_workflow_response.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_pure_graph_runtime_learns_across_all_world_domains(self) -> None:
        self.assertEqual(
            self.model["architecture"],
            "pure-executable-causal-phase-hypergraph",
        )
        self.assertEqual(set(self.report["world"]["domain_counts"]), set(DOMAIN_NAMES))
        self.assertTrue(
            all(count > 0 for count in self.report["world"]["domain_counts"].values())
        )
        self.assertGreaterEqual(self.report["learning"]["crystallized_laws"], 2)
        self.assertTrue(self.report["experiment_gates"]["passed"])
        self.assertTrue(self.report["experiment_gates"]["checks"]["mamba_not_imported"])

    def test_live_workflow_runs_all_components_and_abstains_without_a_path(
        self,
    ) -> None:
        response = run_causal_workflow(self.model, self.request)
        self.assertEqual(response, self.response)
        self.assertEqual(response["runtime"]["wiki"], CAUSAL_WORLD_WIKI_RUNTIME)
        self.assertEqual(response["runtime"]["rag"], CAUSAL_WORLD_RAG_RUNTIME)
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        for turn in response["turns"]:
            components = {
                entry["component"] for entry in turn["artifact"]["execution_trace"]
            }
            self.assertEqual(components, set(ARCHITECTURE_COMPONENTS))
        statuses = [turn["artifact"]["claim_status"] for turn in response["turns"]]
        self.assertGreaterEqual(statuses.count("derived"), 2)
        self.assertEqual(statuses.count("unknown"), 2)
        self.assertTrue(
            any(turn["artifact"]["path_length"] > 1 for turn in response["turns"])
        )
        self.assertTrue(
            self.report["experiment_gates"]["checks"]["multi_hop_causal_workflow"]
        )

    def test_saved_graph_is_hash_bound_and_corruption_fails_closed(self) -> None:
        graph = CausalGraph.from_model_payload(self.model)
        self.assertEqual(graph.model_payload(), self.model)
        self.assertTrue(self.report["corruption_checks"]["passed"])
        self.assertTrue(self.report["deterministic_replay"]["passed"])
        altered = json.loads(json.dumps(self.model))
        altered["graph"]["laws"][0]["confidence"] = 0.0
        with self.assertRaises(ValueError):
            CausalGraph.from_model_payload(altered)

    def test_massive_plan_is_sharded_and_exceeds_a_billion_updates(self) -> None:
        config = get_profile("tpu-massive")
        plan = build_accelerator_plan(config)
        scale = config.scale_manifest()
        self.assertGreater(scale["entity_updates"], 1_000_000_000)
        self.assertGreater(scale["relation_updates"], 10_000_000_000)
        self.assertGreater(plan["microbatches_per_shard"], 1)
        self.assertEqual(plan["matched_rollouts_per_world"], 2)
        self.assertEqual(plan["persistent_graph_location"], "host")
        self.assertEqual(plan["effects_retained_per_intervention"], 8)
        self.assertEqual(plan["expected_evidence_per_shard"], 65536)
        self.assertEqual(plan["device_parallelism"], "pmap-all-local-devices")
        self.assertEqual(plan["world_programs_per_shard"], 4)
        self.assertEqual(plan["curriculum"]["exercised_programs"], 64)

    def test_compositional_curriculum_is_massive_and_changes_runtime_tensors(
        self,
    ) -> None:
        manifest = curriculum_manifest(shard_count=16)
        self.assertEqual(world_program_space_size(), 52_500_000)
        self.assertEqual(manifest["contextual_domain_world_space"], 420_000_000)
        self.assertEqual(manifest["unique_exercised_programs"], 64)
        self.assertEqual(len(manifest["primary_roots_exercised"]), 7)
        self.assertTrue(
            all(
                len(values) == len(manifest["axes"][axis])
                for axis, values in manifest["axis_values_exercised"].items()
            )
        )
        first, second = curriculum_programs(0)[:2]
        feature_count = 32
        relation_count = 12
        first_state = np.full((1, 3, feature_count), 0.5, dtype=np.float32)
        second_state = first_state.copy()
        _apply_programmed_state(
            first_state, np.asarray([first.program_id], dtype=np.int64)
        )
        _apply_programmed_state(
            second_state, np.asarray([second.program_id], dtype=np.int64)
        )
        first_relations = np.full((1, 3, 2, relation_count), 0.5, dtype=np.float32)
        second_relations = first_relations.copy()
        _apply_programmed_relations(
            first_relations, np.asarray([first.program_id], dtype=np.int64)
        )
        _apply_programmed_relations(
            second_relations, np.asarray([second.program_id], dtype=np.int64)
        )
        self.assertFalse(np.array_equal(first_state, second_state))
        self.assertFalse(np.array_equal(first_relations, second_relations))

    def test_conserved_budgets_survive_total_entity_expiry_pressure(self) -> None:
        config = get_profile("test")
        batch = ProceduralWorldCompiler(config).compile_shard(0)
        batch.state[..., FEATURE_INDEX["lifetime"]] = 0.021
        batch.state[..., FEATURE_INDEX["integrity"]] = 0.01
        batch.state[..., FEATURE_INDEX["health"]] = 0.01
        batch.state[..., FEATURE_INDEX["structure"]] = 0.0

        for _ in range(4):
            diagnostics = advance_world(batch)
            self.assertLess(diagnostics["maximum_invariant_error"], 1e-5)
            self.assertTrue(np.all(batch.active_mask.sum(axis=1) >= 1.0))

        actual_budgets = np.stack(
            tuple(
                (
                    batch.state[..., FEATURE_INDEX[feature]]
                    * batch.active_mask
                ).sum(axis=1)
                for feature in ("mass", "energy", "resource")
            ),
            axis=1,
        )
        np.testing.assert_allclose(
            actual_budgets,
            batch.initial_budgets,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_english_question_roundtrips_the_complete_world_regime(self) -> None:
        conditions = curriculum_programs(0)[0].condition_signature()
        request = render_causal_question(
            query_id="condition-roundtrip",
            domain="ecological",
            cause_feature="resource",
            effect_feature="health",
            variant=3,
            condition_signature=conditions,
        )
        query = parse_causal_question(request)
        self.assertEqual(query.cause_feature, "resource")
        self.assertEqual(query.effect_feature, "health")
        self.assertEqual(
            stable_condition_signature(query.context_signature),
            stable_condition_signature(conditions),
        )
        self.assertIn("world regime", request["text"].lower())

    def test_compiler_can_execute_a_program_excluded_from_training(self) -> None:
        config = get_profile("test")
        training_ids = [program.program_id for program in curriculum_programs(0)]
        heldout_id = select_heldout_program_ids(training_ids, count=1)[0]
        batch = ProceduralWorldCompiler(config).compile_shard(
            0, program_id=heldout_id
        )
        self.assertEqual(set(batch.program_ids.tolist()), {heldout_id})
        self.assertNotIn(heldout_id, training_ids)

    def test_semantic_phase_state_does_not_depend_on_query_identifier(self) -> None:
        graph = CausalGraph.from_model_payload(self.model)
        law = next(
            value for value in graph.laws.values() if value.status == "crystallized"
        )
        common = {
            "domain": law.domain,
            "cause_feature": law.cause_feature,
            "effect_feature": law.effect_feature,
            "context_signature": law_condition_signature(law),
        }
        cognition = CausalCognition(graph)
        first = cognition.answer(CausalQuery(query_id="semantic-a", **common))
        second = cognition.answer(CausalQuery(query_id="semantic-b", **common))
        self.assertEqual(first.pop("query_id"), "semantic-a")
        self.assertEqual(second.pop("query_id"), "semantic-b")
        self.assertEqual(first, second)

    def test_contextual_transfer_composes_three_persistent_regimes(self) -> None:
        graph = CausalGraph(maximum_laws=64)
        for program_id in (0, 1, 2):
            conditions = decode_world_program(program_id).condition_signature()
            for repetition in range(5):
                graph.observe(
                    CausalEvidence(
                        evidence_id=f"transfer-{program_id}-{repetition}",
                        domain="physical",
                        cause_feature="energy",
                        effect_feature="temperature",
                        direction=1,
                        magnitude=0.5,
                        delay=1,
                        context_signature=(
                            "domain:physical",
                            "cause:energy",
                            *conditions,
                        ),
                        treated_worlds=32,
                        baseline_worlds=32,
                        variance=0.005,
                        invariant_error=0.001,
                        provenance_hash=canonical_hash(
                            {"program_id": program_id, "run": repetition}
                        ),
                    )
                )
        cognition = CausalCognition(graph)
        cognition.persistence.consolidate(graph)
        query = CausalQuery(
            query_id="unseen-regime",
            domain="physical",
            cause_feature="energy",
            effect_feature="temperature",
            context_signature=decode_world_program(3).condition_signature(),
        )
        exact = cognition.answer(query, allow_contextual_transfer=False)
        transferred = cognition.answer(query, allow_contextual_transfer=True)
        self.assertEqual(exact["claim_status"], "unknown")
        self.assertEqual(transferred["claim_status"], "derived")
        self.assertEqual(transferred["derivation_kind"], "contextual_transfer")
        self.assertEqual(transferred["direction"], 1)
        self.assertGreaterEqual(transferred["source_count"], 3)
        self.assertGreaterEqual(len(transferred["source_condition_signatures"]), 3)
        self.assertEqual(
            transferred["context_factor_trace"]["pair_motif_count"],
            55,
        )
        self.assertEqual(
            transferred["context_factor_trace"]["runtime"],
            CONTEXT_FACTOR_GRAPH_RUNTIME,
        )
        self.assertEqual(
            project_context_factor_trace(
                transferred["context_factor_trace"],
                transferred["transfer_policy"],
            ),
            transferred["context_factor_projection"],
        )
        self.assertIn(
            "pair_motif_power",
            transferred["transfer_policy"],
        )
        for field in (
            "expected_magnitude",
            "expected_delay",
            "confidence",
            "persistence",
            "probability",
            "margin",
            "context_similarity",
            "maximum_context_similarity",
            "transfer_consensus",
        ):
            self.assertEqual(transferred[field], round(transferred[field], 12))

    def test_transfer_provenance_is_semantic_and_platform_portable(self) -> None:
        evidence_fields = {
            "evidence_id": "portable-transfer-evidence",
            "domain": "physical",
            "cause_feature": "energy",
            "effect_feature": "temperature",
            "direction": 1,
            "magnitude": 0.123456789012345,
            "delay": 2,
            "context_signature": (
                "domain:physical",
                "cause:energy",
                *decode_world_program(11).condition_signature(),
            ),
            "treated_worlds": 32,
            "baseline_worlds": 32,
            "variance": 0.000123456789012345,
            "invariant_error": 0.000001,
        }
        first = CausalEvidence(
            **evidence_fields,
            provenance_hash=canonical_hash({"raw_array_digest": "linux"}),
        )
        second = CausalEvidence(
            **evidence_fields,
            provenance_hash=canonical_hash({"raw_array_digest": "windows"}),
        )
        first_hash = stable_transfer_evidence_provenance(
            first,
            program_id=11,
            replica=0,
        )
        self.assertEqual(
            first_hash,
            stable_transfer_evidence_provenance(
                second,
                program_id=11,
                replica=0,
            ),
        )
        self.assertNotEqual(
            first_hash,
            stable_transfer_evidence_provenance(
                second,
                program_id=11,
                replica=1,
            ),
        )

    def test_transfer_runtime_is_hash_bound_balanced_and_graph_rag_visible(
        self,
    ) -> None:
        names = {
            "validation": "atom_causal_world_transfer_validation_truth.json",
            "policy": "atom_causal_world_transfer_policy.json",
            "truth": "atom_causal_world_transfer_truth.json",
            "request": "atom_causal_world_transfer_request.json",
            "exact": "atom_causal_world_transfer_exact_response.json",
            "response": "atom_causal_world_transfer_response.json",
            "report": "atom_causal_world_transfer_report.json",
        }
        payloads = {
            key: json.loads((self.output_dir / name).read_text(encoding="utf-8"))
            for key, name in names.items()
        }
        truth = payloads["truth"]
        validation_truth = payloads["validation"]
        transfer_policy = payloads["policy"]
        request = payloads["request"]
        response = payloads["response"]
        transfer_report = payloads["report"]
        self.assertEqual(build_transfer_request(truth), request)
        self.assertEqual(
            run_transfer_workflow(
                self.model,
                request,
                allow_contextual_transfer=True,
                transfer_policy=transfer_policy,
            ),
            response,
        )
        self.assertEqual(
            validate_transfer_policy_artifact(
                transfer_policy,
                model_hash=self.model["model_hash"],
            ),
            transfer_policy,
        )
        self.assertEqual(
            fit_transfer_policy(self.model, validation_truth),
            transfer_policy,
        )
        self.assertEqual(
            evaluate_transfer_response(response, truth),
            transfer_report["contextual_transfer"],
        )
        self.assertEqual(transfer_report, self.report["transfer_benchmark"])
        self.assertEqual(transfer_report["model_hash"], self.model["model_hash"])
        self.assertEqual(
            transfer_report["transfer_policy_hash"],
            transfer_policy["policy_hash"],
        )
        self.assertEqual(validation_truth["truth_role"], "validation")
        self.assertEqual(truth["truth_role"], "evaluation")
        self.assertEqual(len(validation_truth["heldout_program_ids"]), 4)
        self.assertEqual(len(truth["heldout_program_ids"]), 3)
        self.assertEqual(transfer_policy["evaluated_policy_count"], 5000)
        self.assertEqual(
            transfer_policy["risk_contract"],
            {
                "method": TRANSFER_RISK_METHOD,
                "confidence_level": 0.95,
                "overall_selective_error_upper_limit": 0.10,
                "direction_selective_error_upper_limit": 0.15,
            },
        )
        self.assertTrue(
            transfer_policy["gates"]["single_factor_probe_reused"]
        )
        self.assertTrue(
            transfer_policy["gates"]["pair_motif_controls_exercised"]
        )
        validation_request = build_transfer_request(validation_truth)
        validation_response = run_transfer_workflow(
            self.model,
            validation_request,
            allow_contextual_transfer=True,
            transfer_policy=transfer_policy,
        )
        portable_probe_hash = factor_probe_trace_hash(validation_response)
        self.assertEqual(
            set(transfer_policy["probe_response_hashes"]),
            {"policy_neutral_projection_lattice"},
        )
        diagnostic_variant = json.loads(json.dumps(validation_response))
        diagnostic_variant["turns"][0]["knowledge_context"][0]["score"] += 0.125
        self.assertEqual(
            factor_probe_trace_hash(diagnostic_variant),
            portable_probe_hash,
        )
        trace_variant = json.loads(json.dumps(validation_response))
        traced_artifact = trace_variant["turns"][0]["artifact"]
        trace = traced_artifact["context_factor_trace"]
        if isinstance(trace, dict):
            trace["portable_digest_test"] = True
        else:
            traced_artifact["context_factor_trace"] = {
                "portable_digest_test": True
            }
        self.assertNotEqual(
            factor_probe_trace_hash(trace_variant),
            portable_probe_hash,
        )
        lattice_hash = factor_projection_lattice_hash(
            model_hash=self.model["model_hash"],
            request_hash=validation_request["request_hash"],
            truth_hash=validation_truth["truth_hash"],
            evaluations=(
                {
                    "policy": transfer_policy["selected_policy"],
                    "evaluation_hash": transfer_policy[
                        "selected_validation_evaluation"
                    ]["evaluation_hash"],
                },
            ),
        )
        altered_evaluation_hash = canonical_hash(
            {"original": lattice_hash}
        )
        self.assertNotEqual(
            lattice_hash,
            factor_projection_lattice_hash(
                model_hash=self.model["model_hash"],
                request_hash=validation_request["request_hash"],
                truth_hash=validation_truth["truth_hash"],
                evaluations=(
                    {
                        "policy": transfer_policy["selected_policy"],
                        "evaluation_hash": altered_evaluation_hash,
                    },
                ),
            ),
        )
        self.assertTrue(
            transfer_policy["gates"]["selected_policy_overall_risk_bound"]
        )
        self.assertTrue(
            transfer_policy["gates"][
                "selected_policy_directional_risk_bounds"
            ]
        )
        self.assertGreaterEqual(
            selective_error_upper_bound(0, 100),
            0.0,
        )
        self.assertEqual(selective_error_upper_bound(0, 0), 1.0)
        self.assertEqual(
            selective_error_upper_bound(28, 396),
            0.100293089682,
        )
        self.assertEqual(
            selective_error_upper_bound(21, 367),
            0.08588756027,
        )
        for false_assertions, asserted, expected in (
            (47, 180, 0.329792132438),
            (158, 246, 0.699565225213),
            (257, 457, 0.607140064743),
            (76, 462, 0.201047948618),
            (204, 568, 0.39943375185),
        ):
            with self.subTest(
                false_assertions=false_assertions,
                asserted=asserted,
            ):
                self.assertEqual(
                    selective_error_upper_bound(
                        false_assertions,
                        asserted,
                    ),
                    expected,
                )
        for false_assertions, asserted in (
            (False, 0),
            (0, False),
            (-1, 1),
            (2, 1),
        ):
            with self.subTest(
                invalid_false_assertions=false_assertions,
                invalid_asserted=asserted,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "selective error counts are invalid",
                ):
                    selective_error_upper_bound(
                        false_assertions,
                        asserted,
                    )
        self.assertFalse(
            set(validation_truth["heldout_program_ids"]).intersection(
                truth["heldout_program_ids"]
            )
        )
        self.assertTrue(
            all(
                case["cause_feature"] != case["effect_feature"]
                for case in truth["cases"]
            )
        )
        direction_counts = transfer_report["truth_direction_counts"]
        self.assertGreaterEqual(
            min(direction_counts.values()) / sum(direction_counts.values()),
            0.30,
        )
        self.assertTrue(all(turn["knowledge_context"] for turn in response["turns"]))
        wiki = CausalWorldWikiGraph(CausalGraph.from_model_payload(self.model))
        hits = retrieve_causal_context(
            wiki, "held out causal transfer in an unseen regime", limit=8
        )
        self.assertTrue(
            any(hit["name"] == "contextual_causal_transfer" for hit in hits)
        )
        calibration_hits = retrieve_causal_context(
            wiki, "metaplastic calibration on validation worlds", limit=8
        )
        self.assertTrue(
            any(
                hit["name"] == "metaplastic_transfer_governor"
                for hit in calibration_hits
            )
        )
        risk_hits = retrieve_causal_context(
            wiki, "pairwise context factor selective error risk", limit=8
        )
        self.assertTrue(
            any(
                hit["name"] == "context_factor_risk_governor"
                for hit in risk_hits
            )
        )
        altered_policy = json.loads(json.dumps(transfer_policy))
        altered_policy["selected_policy"]["pair_motif_power"] = 0.5
        with self.assertRaises(ValueError):
            run_transfer_workflow(
                self.model,
                request,
                allow_contextual_transfer=True,
                transfer_policy=altered_policy,
            )

    def test_accelerator_resume_state_is_cumulative_and_fails_closed(self) -> None:
        config = get_profile("tpu-massive")
        graph = CausalGraph(maximum_laws=config.maximum_laws)
        evidence = CausalEvidence(
            evidence_id="resume-contract-evidence",
            domain="physical",
            cause_feature="energy",
            effect_feature="temperature",
            direction=1,
            magnitude=0.5,
            delay=1,
            context_signature=("domain:physical", "backend:xla"),
            treated_worlds=512,
            baseline_worlds=512,
            variance=0.01,
            invariant_error=0.001,
            provenance_hash=canonical_hash({"source": "resume-contract"}),
        )
        graph.observe(evidence)
        model = graph.model_payload()
        cursor = build_causal_resume_cursor(
            config,
            graph,
            completed_shards=[0],
            shard_evidence_hashes=[canonical_hash([evidence.evidence_id])],
            model_lineage=[model["model_hash"]],
        )
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            (state_dir / "atom_causal_world_model.json").write_text(
                json.dumps(model), encoding="utf-8"
            )
            (state_dir / "atom_causal_world_resume_cursor.json").write_text(
                json.dumps(cursor), encoding="utf-8"
            )
            restored, restored_cursor = load_causal_resume_state(
                state_dir,
                config,
                expected_next_shard=1,
            )
            self.assertEqual(restored.model_payload(), model)
            self.assertEqual(restored_cursor, cursor)

            altered = json.loads(json.dumps(cursor))
            altered["cumulative_evidence_count"] += 1
            altered_core = {
                key: altered[key] for key in sorted(set(altered) - {"cursor_hash"})
            }
            altered["cursor_hash"] = canonical_hash(altered_core)
            (state_dir / "atom_causal_world_resume_cursor.json").write_text(
                json.dumps(altered), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_causal_resume_state(
                    state_dir,
                    config,
                    expected_next_shard=1,
                )

    def test_multiple_accelerator_shards_extend_one_graph_and_resume(self) -> None:
        config = get_profile("tpu-massive")

        def fake_shard(
            _config: object,
            shard_index: int,
            *,
            require_tpu: bool,
            require_gpu: bool,
        ) -> tuple[tuple[CausalEvidence, ...], dict[str, object]]:
            self.assertTrue(require_tpu)
            self.assertFalse(require_gpu)
            evidence = CausalEvidence(
                evidence_id=f"fake-accelerator-{shard_index}",
                domain="physical",
                cause_feature="energy",
                effect_feature="temperature",
                direction=1,
                magnitude=0.4 + 0.01 * shard_index,
                delay=1,
                context_signature=("domain:physical", "backend:xla"),
                treated_worlds=512,
                baseline_worlds=512,
                variance=0.01,
                invariant_error=0.001,
                provenance_hash=canonical_hash({"shard": shard_index}),
            )
            evidence_hash = canonical_hash([evidence.evidence_id])
            return (evidence,), {
                "runtime": "fake-accelerator",
                "probe": {
                    "jax_available": True,
                    "tpu_available": True,
                    "gpu_available": False,
                },
                "plan": {"microbatches_per_shard": 16},
                "shard_index": shard_index,
                "world_programs": [
                    program.manifest() for program in curriculum_programs(shard_index)
                ],
                "evidence_count": 1,
                "elapsed_seconds": 1.0,
                "entity_updates": 10,
                "relation_updates": 20,
                "maximum_invariant_error": 0.001,
                "devices_used": 8,
                "executor_mode": "pmap",
                "jit_executor_constructions": 1,
                "deterministic_replay": {"passed": True},
                "evidence_hash": evidence_hash,
            }

        state_writes: list[tuple[dict[str, object], dict[str, object]]] = []
        with patch(
            "atom_causal_world_experiment.run_jax_massive_shard",
            side_effect=fake_shard,
        ):
            graph, evidence, execution, cursor = _run_accelerator_learning(
                config,
                0,
                require_tpu=True,
                require_gpu=False,
                shards_per_run=2,
                state_writer=lambda model, state: state_writes.append(
                    (dict(model), dict(state))
                ),
            )
            self.assertEqual(graph.observation_count, 2)
            self.assertEqual(len(evidence), 2)
            self.assertEqual(execution["shards_executed"], 2)
            self.assertEqual(cursor["completed_shards"], [0, 1])
            self.assertEqual(cursor["next_shard"], 2)
            self.assertEqual(len(state_writes), 2)

            resumed_graph, resumed_evidence, resumed_execution, resumed_cursor = (
                _run_accelerator_learning(
                    config,
                    2,
                    require_tpu=True,
                    require_gpu=False,
                    shards_per_run=1,
                    initial_graph=graph,
                    prior_cursor=cursor,
                )
            )
        self.assertEqual(resumed_graph.observation_count, 3)
        self.assertEqual(len(resumed_evidence), 1)
        self.assertEqual(resumed_execution["shard_start"], 2)
        self.assertEqual(resumed_cursor["completed_shards"], [0, 1, 2])
        self.assertEqual(resumed_cursor["next_shard"], 3)

    def test_accelerator_requirements_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "cannot require both TPU and GPU",
        ):
            run_causal_world_experiment(
                self.output_dir / "invalid-dual-accelerator",
                profile="tpu-massive",
                backend="jax-xla",
                require_tpu=True,
                require_gpu=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            "needs the jax-xla backend",
        ):
            run_causal_world_experiment(
                self.output_dir / "invalid-gpu-fallback",
                profile="test",
                backend="numpy",
                require_gpu=True,
            )
        with patch(
            "atom_causal_world_accelerator.probe_jax_accelerator",
            return_value={
                "runtime": "atom-causal-world-xla-v1",
                "jax_available": True,
                "jax_version": "test",
                "tpu_available": False,
                "gpu_available": False,
                "devices": [
                    {
                        "id": 0,
                        "platform": "cpu",
                        "device_kind": "test-cpu",
                    }
                ],
                "error": None,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU_REQUIRED"):
                run_jax_massive_shard(
                    get_profile("tpu-massive"),
                    0,
                    require_tpu=False,
                    require_gpu=True,
                )

    def test_unobserved_answer_can_be_derived_through_a_supported_path(self) -> None:
        graph = CausalGraph(maximum_laws=64)
        for cause, effect, magnitude in (
            ("temperature", "energy", 0.5),
            ("energy", "structure", 0.7),
        ):
            for repetition in range(5):
                graph.observe(
                    CausalEvidence(
                        evidence_id=f"path-{cause}-{effect}-{repetition}",
                        domain="physical",
                        cause_feature=cause,
                        effect_feature=effect,
                        direction=1,
                        magnitude=magnitude,
                        delay=2,
                        context_signature=(
                            "domain:physical",
                            f"cause:{cause}",
                        ),
                        treated_worlds=32,
                        baseline_worlds=32,
                        variance=0.005,
                        invariant_error=0.001,
                        provenance_hash=canonical_hash(
                            {"cause": cause, "effect": effect, "run": repetition}
                        ),
                    )
                )
        cognition = CausalCognition(graph)
        cognition.persistence.consolidate(graph)
        artifact = cognition.answer(
            CausalQuery(
                query_id="integration-multi-hop",
                domain="physical",
                cause_feature="temperature",
                effect_feature="structure",
                context_signature=("domain:physical", "cause:temperature"),
            )
        )
        self.assertEqual(artifact["claim_status"], "derived")
        self.assertEqual(artifact["path_length"], 2)
        self.assertEqual(len(artifact["source_law_ids"]), 2)
        self.assertEqual(
            [item["law_id"] for item in artifact["evidence_path"][:-2]],
            artifact["source_law_ids"],
        )
        self.assertTrue(
            any(
                entry.get("composed_paths", 0) > 0
                for entry in artifact["execution_trace"]
            )
        )

    def test_missing_multi_hop_path_is_reportable_instead_of_fatal(self) -> None:
        graph = CausalGraph(maximum_laws=64)
        for domain, cause, effect in (
            ("physical", "energy", "temperature"),
            ("social", "trust", "memory_strength"),
        ):
            for repetition in range(5):
                graph.observe(
                    CausalEvidence(
                        evidence_id=f"disconnected-{domain}-{repetition}",
                        domain=domain,
                        cause_feature=cause,
                        effect_feature=effect,
                        direction=1,
                        magnitude=0.5,
                        delay=1,
                        context_signature=(f"domain:{domain}", f"cause:{cause}"),
                        treated_worlds=32,
                        baseline_worlds=32,
                        variance=0.005,
                        invariant_error=0.001,
                        provenance_hash=canonical_hash(
                            {"domain": domain, "run": repetition}
                        ),
                    )
                )
        CausalCognition(graph).persistence.consolidate(graph)
        request, truth = build_causal_workflow_request(graph)
        self.assertNotIn("composed-00", truth)
        self.assertFalse(
            any(turn["query_id"] == "composed-00" for turn in request["turns"])
        )
        response = run_causal_workflow(graph.model_payload(), request)
        self.assertEqual(len(response["turns"]), 4)
        self.assertEqual(
            [turn["artifact"]["claim_status"] for turn in response["turns"]],
            ["derived", "derived", "unknown", "unknown"],
        )

    def test_graph_rag_and_side_view_render_the_real_artifact(self) -> None:
        document = (self.output_dir / "atom_causal_world_side_view.html").read_text(
            encoding="utf-8"
        )
        saved_report = json.loads(
            (self.output_dir / "atom_causal_world_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            render_causal_world_artifact(self.model, self.report, self.response),
            document,
        )
        self.assertEqual(
            render_causal_world_artifact(self.model, saved_report, self.response),
            document,
        )
        self.assertIn(self.model["model_hash"], document)
        self.assertIn(ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME, document)
        self.assertIn("render_causal_world_artifact", document)
        self.assertIn("Measured causal artifact", document)
        self.assertIn("Context-factor risk governor", document)
        self.assertIn("Pair motif power", document)
        self.assertIn("Projection lattice digest", document)
        self.assertIn(
            self.report["transfer_benchmark"]["metaplastic_calibration"][
                "probe_response_hashes"
            ]["policy_neutral_projection_lattice"],
            document,
        )
        self.assertIn("Declared overall risk limit", document)
        self.assertIn("Held-out causal transfer", document)
        self.assertIn("Negative / positive truth", document)
        self.assertIn(
            self.report["transfer_benchmark"]["report_hash"], document
        )
        self.assertIn(
            self.report["transfer_benchmark"]["transfer_policy_hash"],
            document,
        )
        self.assertIn("Compositional world curriculum", document)
        self.assertIn("Exercised world regimes", document)
        self.assertIn("Typed formal domains", document)
        self.assertIn("Cross-domain formal programs", document)
        self.assertIn(self.report["formal_domains"]["registry_hash"], document)
        self.assertIn(self.report["formal_domains"]["report_hash"], document)
        self.assertNotIn("<button", document)
        self.assertNotIn("<input", document)

    def test_formal_domains_are_bound_into_runtime_rag_and_artifacts(self) -> None:
        artifact = json.loads(
            (self.output_dir / "atom_causal_world_formal_domains.json").read_text(
                encoding="utf-8"
            )
        )
        knowledge = json.loads(
            (self.output_dir / "atom_causal_world_knowledge_graph.json").read_text(
                encoding="utf-8"
            )
        )
        formal_report = artifact["report"]
        self.assertEqual(formal_report, self.report["formal_domains"])
        self.assertTrue(formal_report["passed"])
        self.assertEqual(set(formal_report["per_domain"]), set(FORMAL_DOMAIN_NAMES))
        self.assertTrue(
            formal_report["gates"]["runtime_matches_independent_oracle"]
        )
        self.assertTrue(
            formal_report["gates"]["false_candidates_are_contradicted"]
        )
        self.assertTrue(
            formal_report["gates"]["cross_domain_programs_are_proven"]
        )
        self.assertEqual(
            knowledge["formal_domains"]["registry_hash"],
            formal_report["registry_hash"],
        )
        node_names = {node["name"] for node in knowledge["nodes"]}
        self.assertTrue(
            {
                "logic_implies",
                "algebra_solve_linear",
                "geometry_distance_squared",
                "calculus_polynomial_derivative",
                "chemistry_mass_conservation",
                "biology_mendelian_distribution",
                "information_binary_entropy",
            }
            <= node_names
        )
        wiki = CausalWorldWikiGraph(CausalGraph.from_model_payload(self.model))
        hits = retrieve_causal_context(
            wiki,
            "differentiate polynomial and calculate entropy",
            limit=16,
        )
        self.assertTrue(
            {"calculus_polynomial_derivative", "information_binary_entropy"}
            <= {hit["name"] for hit in hits}
        )
        for gate in (
            "formal_domain_curriculum_passed",
            "all_formal_domains_match_independent_oracles",
            "formal_epistemic_states_distinguish_proof_and_contradiction",
            "formal_cross_domain_composition_passed",
            "formal_registry_is_bound_into_graph_rag",
        ):
            with self.subTest(gate=gate):
                self.assertTrue(self.report["experiment_gates"]["checks"][gate])


if __name__ == "__main__":
    unittest.main()
