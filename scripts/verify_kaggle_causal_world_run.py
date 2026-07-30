"""Fail-closed verifier for downloaded Atom causal-world Kaggle runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom_causal_graph import (  # noqa: E402
    CausalGraph,
    project_context_factor_trace,
    stable_condition_signature,
)
from atom_formal_domains import (  # noqa: E402
    formal_domain_manifest,
    run_formal_domain_benchmark,
)
from atom_causal_world_experiment import (  # noqa: E402
    load_causal_resume_state,
    run_causal_workflow,
)
from atom_causal_world_knowledge import CausalWorldWikiGraph  # noqa: E402
from atom_causal_world_schema import (  # noqa: E402
    DOMAIN_NAMES,
    canonical_hash,
    get_profile,
)
from atom_causal_world_side_view import (  # noqa: E402
    render_causal_world_artifact,
)
from atom_causal_world_transfer import (  # noqa: E402
    build_transfer_evaluator_truth,
    build_transfer_validation_truth,
    evaluate_transfer_response,
    fit_transfer_policy,
    run_transfer_workflow,
    validate_transfer_policy_artifact,
)

EXPECTED_OUTPUT_FILES = (
    "atom_causal_world_evaluator_truth.json",
    "atom_causal_world_evidence.jsonl",
    "atom_causal_world_formal_domains.json",
    "atom_causal_world_knowledge_graph.json",
    "atom_causal_world_manifest.json",
    "atom_causal_world_model.json",
    "atom_causal_world_report.json",
    "atom_causal_world_resume_cursor.json",
    "atom_causal_world_side_view.html",
    "atom_causal_world_workflow_request.json",
    "atom_causal_world_workflow_response.json",
)

TRANSFER_OUTPUT_FILES = (
    "atom_causal_world_transfer_exact_response.json",
    "atom_causal_world_transfer_report.json",
    "atom_causal_world_transfer_request.json",
    "atom_causal_world_transfer_response.json",
    "atom_causal_world_transfer_truth.json",
)

CALIBRATED_TRANSFER_OUTPUT_FILES = (
    "atom_causal_world_transfer_policy.json",
    "atom_causal_world_transfer_validation_truth.json",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_source_sha256(path: Path) -> str:
    source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _artifact_dir(run_dir: Path) -> Path:
    direct = run_dir.resolve()
    nested = direct / "causal_world_outputs"
    selected = nested if nested.is_dir() else direct
    missing = [name for name in EXPECTED_OUTPUT_FILES if not (selected / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing causal-world artifacts: {missing}")
    return selected


def _runtime_api(source_file: Path | None) -> Any:
    if source_file is None:
        return SimpleNamespace(
            CausalGraph=CausalGraph,
            CausalWorldWikiGraph=CausalWorldWikiGraph,
            build_transfer_evaluator_truth=build_transfer_evaluator_truth,
            build_transfer_validation_truth=build_transfer_validation_truth,
            canonical_hash=canonical_hash,
            evaluate_transfer_response=evaluate_transfer_response,
            fit_transfer_policy=fit_transfer_policy,
            formal_domain_manifest=formal_domain_manifest,
            get_profile=get_profile,
            load_causal_resume_state=load_causal_resume_state,
            project_context_factor_trace=project_context_factor_trace,
            render_causal_world_artifact=render_causal_world_artifact,
            run_causal_workflow=run_causal_workflow,
            run_formal_domain_benchmark=run_formal_domain_benchmark,
            run_transfer_workflow=run_transfer_workflow,
            stable_condition_signature=stable_condition_signature,
            validate_transfer_policy_artifact=validate_transfer_policy_artifact,
        )
    resolved = source_file.resolve()
    module_name = f"atom_causal_world_downloaded_{_sha256(resolved)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the downloaded causal-world source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stream_evidence(path: Path, runtime: Any) -> dict[str, Any]:
    canonical_digest = hashlib.sha256()
    canonical_digest.update(b"[")
    row_count = 0
    domain_counts: Counter[str] = Counter()
    conditions_bound = True
    provenance_bound = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if row_count:
                canonical_digest.update(b",")
            canonical_digest.update(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            row_count += 1
            domain_counts[str(payload["domain"])] += 1
            conditions_bound &= (
                len(runtime.stable_condition_signature(payload["context_signature"]))
                == 11
            )
            provenance_bound &= len(str(payload["provenance_hash"])) == 64
    canonical_digest.update(b"]")
    return {
        "canonical_hash": canonical_digest.hexdigest(),
        "row_count": row_count,
        "domain_counts": dict(sorted(domain_counts.items())),
        "conditions_bound": conditions_bound,
        "provenance_bound": provenance_bound,
    }


def _axis_coverage(curriculum: dict[str, Any]) -> dict[str, list[str]]:
    schedule = curriculum["schedule"]
    return {
        axis: sorted({str(program[axis]) for program in schedule})
        for axis in curriculum["axes"]
    }


def verify_run(
    run_dir: Path,
    *,
    source_file: Path | None = None,
    expected_bundle_sha256: str | None = None,
    require_transfer_benchmark: bool = False,
    expected_accelerator: str = "tpu",
) -> dict[str, Any]:
    expected_accelerator = expected_accelerator.lower()
    if expected_accelerator not in {"tpu", "gpu"}:
        raise ValueError("expected accelerator must be TPU or GPU")
    artifacts = _artifact_dir(run_dir)
    runtime = _runtime_api(source_file)
    report = _load_json(artifacts / "atom_causal_world_report.json")
    model = _load_json(artifacts / "atom_causal_world_model.json")
    request = _load_json(artifacts / "atom_causal_world_workflow_request.json")
    response = _load_json(artifacts / "atom_causal_world_workflow_response.json")
    saved_knowledge = _load_json(
        artifacts / "atom_causal_world_knowledge_graph.json"
    )
    saved_formal = _load_json(
        artifacts / "atom_causal_world_formal_domains.json"
    )
    saved_side_view = (
        artifacts / "atom_causal_world_side_view.html"
    ).read_text(encoding="utf-8")
    evidence = _stream_evidence(
        artifacts / "atom_causal_world_evidence.jsonl", runtime
    )
    transfer_summary = report.get("transfer_benchmark")
    transfer_declared = isinstance(transfer_summary, dict)
    calibrated_transfer = transfer_declared and isinstance(
        transfer_summary.get("transfer_policy_hash"), str
    )
    required_transfer_files = list(TRANSFER_OUTPUT_FILES)
    if calibrated_transfer:
        required_transfer_files.extend(CALIBRATED_TRANSFER_OUTPUT_FILES)
    transfer_present = any(
        (artifacts / name).is_file() for name in required_transfer_files
    )
    verify_transfer = transfer_declared or transfer_present or require_transfer_benchmark
    missing_transfer = [
        name for name in required_transfer_files if not (artifacts / name).is_file()
    ]
    if verify_transfer and missing_transfer:
        raise FileNotFoundError(
            f"missing causal transfer benchmark artifacts: {missing_transfer}"
        )

    report_core = dict(report)
    saved_report_hash = str(report_core.pop("report_hash"))
    graph = runtime.CausalGraph.from_model_payload(model)
    replay_response = runtime.run_causal_workflow(model, request)
    expected_knowledge = runtime.CausalWorldWikiGraph(graph).manifest()
    expected_formal = runtime.run_formal_domain_benchmark(
        cases_per_primitive=int(report["formal_domains"]["cases_per_primitive"])
    )
    render_report = copy.deepcopy(report)
    if hasattr(runtime, "WORLD_PROGRAM_AXES"):
        saved_axes = render_report["world"]["curriculum"]["axes"]
        render_report["world"]["curriculum"]["axes"] = {
            axis: saved_axes[axis] for axis, _ in runtime.WORLD_PROGRAM_AXES
        }
    expected_side_view = runtime.render_causal_world_artifact(
        model, render_report, response
    )

    config = runtime.get_profile(str(report["world"]["config"]["profile"]))
    resumed_graph, cursor = runtime.load_causal_resume_state(
        artifacts,
        config,
        expected_next_shard=int(report["execution"]["shard_stop"]),
    )

    curriculum = dict(report["world"]["curriculum"])
    coverage = _axis_coverage(curriculum)
    expected_coverage = {
        axis: sorted(str(value) for value in values)
        for axis, values in curriculum["axes"].items()
    }
    schedule = list(curriculum["schedule"])
    execution = report["execution"]
    accelerator = execution["accelerator"]
    probe = report["accelerator_probe"]
    scale = config.scale_manifest()
    gate_checks = dict(report["experiment_gates"]["checks"])
    workflow_evaluation = report["evaluation"]
    source_file_hash = _sha256(source_file) if source_file is not None else None
    source_hash = (
        _canonical_source_sha256(source_file) if source_file is not None else None
    )
    normalized_expected_hash = (
        expected_bundle_sha256.lower() if expected_bundle_sha256 else None
    )
    log_files = sorted(run_dir.resolve().glob("*.log"))
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in log_files)

    checks = {
        "report_hash_exact": runtime.canonical_hash(report_core) == saved_report_hash,
        "evidence_hash_exact": (
            evidence["canonical_hash"] == report["learning"]["evidence_hash"]
        ),
        "model_roundtrip_exact": graph.model_payload() == model,
        "resume_model_roundtrip_exact": resumed_graph.model_payload() == model,
        "cursor_model_binding_exact": cursor["model_hash"] == model["model_hash"],
        "workflow_replay_exact": replay_response == response,
        "knowledge_graph_exact": runtime.canonical_hash(expected_knowledge)
        == runtime.canonical_hash(saved_knowledge),
        "formal_domain_artifact_exact": expected_formal == saved_formal,
        "formal_domain_report_bound": (
            saved_formal["report"] == report["formal_domains"]
            and saved_formal["manifest"]["registry_hash"]
            == report["formal_domains"]["registry_hash"]
            and saved_knowledge["formal_domains"]["registry_hash"]
            == report["formal_domains"]["registry_hash"]
        ),
        "formal_domain_oracle_replay_passed": (
            expected_formal["report"]["passed"] is True
            and expected_formal["report"]["gates"][
                "runtime_matches_independent_oracle"
            ]
            and expected_formal["report"]["gates"][
                "cross_domain_programs_are_proven"
            ]
        ),
        "side_view_exact": expected_side_view == saved_side_view,
        "all_expected_files_present": all(
            (artifacts / name).is_file() for name in EXPECTED_OUTPUT_FILES
        ),
        "all_evidence_conditions_bound": evidence["conditions_bound"],
        "all_evidence_provenance_bound": evidence["provenance_bound"],
        "all_domains_exercised": set(evidence["domain_counts"]) == set(DOMAIN_NAMES),
        "evidence_count_exact": evidence["row_count"]
        == report["accelerator_plan"]["expected_evidence_per_shard"]
        * execution["shards_executed"],
        "axis_coverage_exact": coverage == expected_coverage,
        "sixty_four_unique_programs": (
            len(schedule) == 64
            and len({int(program["program_id"]) for program in schedule}) == 64
        ),
        "all_primary_roots_exercised": len(
            {str(program["primary_root"]) for program in schedule}
        )
        == 7,
        "all_secondary_roots_exercised": len(
            {str(program["secondary_root"]) for program in schedule}
        )
        == 7,
        "expected_accelerator_devices_observed": (
            probe[f"{expected_accelerator}_available"] is True
            and len(probe["devices"]) >= 1
            and all(
                (
                    str(device["platform"]).lower() == "tpu"
                    if expected_accelerator == "tpu"
                    else str(device["platform"]).lower()
                    in {"gpu", "cuda", "rocm"}
                )
                for device in probe["devices"]
            )
            and accelerator["devices_used"] == len(probe["devices"])
            and (expected_accelerator != "tpu" or len(probe["devices"]) == 8)
        ),
        "expected_accelerator_executor_observed": (
            accelerator["executor_mode"] == "pmap"
            if expected_accelerator == "tpu"
            else accelerator["executor_mode"] in {"jit", "pmap"}
        ),
        "sixteen_shards_executed": (
            accelerator["shards_executed"] == 16
            and execution["atomic_state_writes"] == 16
        ),
        "paired_entity_updates_exact": (
            accelerator["entity_updates"] == scale["entity_updates"] * 2
        ),
        "paired_relation_updates_exact": (
            accelerator["relation_updates"] == scale["relation_updates"] * 2
        ),
        "conservation_bounded": execution["maximum_invariant_error"] < 0.25,
        "deterministic_replay_passed": report["deterministic_replay"]["passed"]
        is True,
        "all_experiment_gates_passed": (
            report["experiment_gates"]["passed"] is True
            and all(bool(value) for value in gate_checks.values())
        ),
        "workflow_exact": (
            workflow_evaluation["turns"] == 9
            and workflow_evaluation["correct"] == 9
            and workflow_evaluation["accuracy"] == 1.0
        ),
        "no_python_traceback": "Traceback (most recent call last)" not in log_text,
        "source_hash_exact": (
            source_hash == normalized_expected_hash
            if normalized_expected_hash is not None
            else True
        ),
    }
    transfer_metrics: dict[str, Any] | None = None
    if verify_transfer:
        transfer_truth = _load_json(
            artifacts / "atom_causal_world_transfer_truth.json"
        )
        transfer_request = _load_json(
            artifacts / "atom_causal_world_transfer_request.json"
        )
        exact_transfer_response = _load_json(
            artifacts / "atom_causal_world_transfer_exact_response.json"
        )
        transfer_response = _load_json(
            artifacts / "atom_causal_world_transfer_response.json"
        )
        transfer_report = _load_json(
            artifacts / "atom_causal_world_transfer_report.json"
        )
        truth_core = dict(transfer_truth)
        truth_hash = str(truth_core.pop("truth_hash"))
        request_core = dict(transfer_request)
        request_hash = str(request_core.pop("request_hash"))
        exact_core = dict(exact_transfer_response)
        exact_hash = str(exact_core.pop("response_hash"))
        transfer_core = dict(transfer_response)
        transfer_hash = str(transfer_core.pop("response_hash"))
        transfer_report_core = dict(transfer_report)
        transfer_report_hash = str(transfer_report_core.pop("report_hash"))
        replayed_exact_transfer = runtime.run_transfer_workflow(
            model, transfer_request, allow_contextual_transfer=False
        )
        validation_truth: dict[str, Any] | None = None
        transfer_policy: dict[str, Any] | None = None
        validation_truth_hash: str | None = None
        transfer_policy_hash: str | None = None
        regenerated_validation_truth: dict[str, Any] | None = None
        regenerated_policy: dict[str, Any] | None = None
        if calibrated_transfer:
            validation_truth = _load_json(
                artifacts / "atom_causal_world_transfer_validation_truth.json"
            )
            transfer_policy = _load_json(
                artifacts / "atom_causal_world_transfer_policy.json"
            )
            validation_truth_core = dict(validation_truth)
            validation_truth_hash = str(validation_truth_core.pop("truth_hash"))
            transfer_policy_core = dict(transfer_policy)
            transfer_policy_hash = str(transfer_policy_core.pop("policy_hash"))
            runtime.validate_transfer_policy_artifact(
                transfer_policy, model_hash=str(model["model_hash"])
            )
            replayed_contextual_transfer = runtime.run_transfer_workflow(
                model,
                transfer_request,
                allow_contextual_transfer=True,
                transfer_policy=transfer_policy,
            )
            regenerated_validation_truth = (
                runtime.build_transfer_validation_truth(
                    validation_truth["training_program_ids"],
                    profile_name=str(validation_truth["profile"]),
                )
            )
            regenerated_policy = runtime.fit_transfer_policy(
                model, validation_truth
            )
            regenerated_truth = runtime.build_transfer_evaluator_truth(
                transfer_truth["training_program_ids"],
                profile_name=str(transfer_truth["profile"]),
                excluded_program_ids=validation_truth["heldout_program_ids"],
            )
        else:
            replayed_contextual_transfer = runtime.run_transfer_workflow(
                model, transfer_request, allow_contextual_transfer=True
            )
            regenerated_truth = runtime.build_transfer_evaluator_truth(
                transfer_truth["training_program_ids"],
                profile_name=str(transfer_truth["profile"]),
            )
        recomputed_exact_evaluation = runtime.evaluate_transfer_response(
            exact_transfer_response, transfer_truth
        )
        recomputed_transfer_evaluation = runtime.evaluate_transfer_response(
            transfer_response, transfer_truth
        )
        direction_counts = {
            str(direction): sum(
                int(case["expected_direction"]) == direction
                for case in transfer_truth["cases"]
            )
            for direction in (-1, 1)
        }
        transfer_checks = {
            "transfer_benchmark_declared": transfer_declared,
            "all_transfer_files_present": not missing_transfer,
            "transfer_truth_hash_exact": runtime.canonical_hash(truth_core)
            == truth_hash,
            "transfer_request_hash_exact": runtime.canonical_hash(request_core)
            == request_hash,
            "exact_transfer_response_hash_exact": runtime.canonical_hash(exact_core)
            == exact_hash,
            "contextual_transfer_response_hash_exact": runtime.canonical_hash(
                transfer_core
            )
            == transfer_hash,
            "transfer_report_hash_exact": runtime.canonical_hash(
                transfer_report_core
            )
            == transfer_report_hash,
            "transfer_truth_regeneration_exact": regenerated_truth == transfer_truth,
            "exact_transfer_replay_exact": replayed_exact_transfer
            == exact_transfer_response,
            "contextual_transfer_replay_exact": replayed_contextual_transfer
            == transfer_response,
            "exact_transfer_evaluation_exact": recomputed_exact_evaluation
            == transfer_report["exact_baseline"],
            "contextual_transfer_evaluation_exact": recomputed_transfer_evaluation
            == transfer_report["contextual_transfer"],
            "transfer_model_binding_exact": (
                transfer_report["model_hash"] == model["model_hash"]
                and exact_transfer_response["model_hash"] == model["model_hash"]
                and transfer_response["model_hash"] == model["model_hash"]
                and report["transfer_benchmark"] == transfer_report
            ),
            "transfer_request_binding_exact": (
                transfer_request["sealed_truth_hash"] == truth_hash
                and exact_transfer_response["request_hash"] == request_hash
                and transfer_response["request_hash"] == request_hash
                and exact_transfer_response["sealed_truth_hash"] == truth_hash
                and transfer_response["sealed_truth_hash"] == truth_hash
            ),
            "heldout_programs_are_disjoint": not set(
                transfer_truth["training_program_ids"]
            ).intersection(transfer_truth["heldout_program_ids"]),
            "transfer_truth_is_cross_feature": all(
                case["cause_feature"] != case["effect_feature"]
                for case in transfer_truth["cases"]
            ),
            "transfer_truth_is_direction_balanced": min(direction_counts.values())
            / max(sum(direction_counts.values()), 1)
            >= 0.35,
            "all_transfer_gates_passed": (
                transfer_report["passed"] is True
                and all(bool(value) for value in transfer_report["gates"].values())
            ),
        }
        if calibrated_transfer:
            if (
                validation_truth is None
                or transfer_policy is None
                or validation_truth_hash is None
                or transfer_policy_hash is None
                or regenerated_validation_truth is None
                or regenerated_policy is None
            ):
                raise AssertionError("calibrated transfer artifacts were not loaded")
            validation_programs = set(validation_truth["heldout_program_ids"])
            evaluation_programs = set(transfer_truth["heldout_program_ids"])
            training_programs = set(transfer_truth["training_program_ids"])
            validation_direction_counts = {
                str(direction): sum(
                    int(case["expected_direction"]) == direction
                    for case in validation_truth["cases"]
                )
                for direction in (-1, 1)
            }
            transfer_checks.update(
                {
                    "validation_truth_hash_exact": runtime.canonical_hash(
                        validation_truth_core
                    )
                    == validation_truth_hash,
                    "transfer_policy_hash_exact": runtime.canonical_hash(
                        transfer_policy_core
                    )
                    == transfer_policy_hash,
                    "validation_truth_regeneration_exact": (
                        regenerated_validation_truth == validation_truth
                    ),
                    "transfer_policy_regeneration_exact": regenerated_policy
                    == transfer_policy,
                    "validation_programs_are_disjoint": not training_programs.intersection(
                        validation_programs
                    ),
                    "evaluation_excludes_validation_programs": not validation_programs.intersection(
                        evaluation_programs
                    ),
                    "evaluation_declares_validation_exclusions": set(
                        transfer_truth["excluded_program_ids"]
                    )
                    == validation_programs,
                    "validation_truth_is_cross_feature": all(
                        case["cause_feature"] != case["effect_feature"]
                        for case in validation_truth["cases"]
                    ),
                    "validation_truth_is_direction_balanced": min(
                        validation_direction_counts.values()
                    )
                    / max(sum(validation_direction_counts.values()), 1)
                    >= 0.35,
                    "validation_truth_uses_two_replicas": all(
                        len(program_report["replicas"]) == 2
                        for program_report in validation_truth["program_reports"]
                    ),
                    "transfer_policy_model_binding_exact": (
                        transfer_policy["model_hash"] == model["model_hash"]
                        and transfer_report["transfer_policy_hash"]
                        == transfer_policy_hash
                        and transfer_report["validation_truth_hash"]
                        == validation_truth_hash
                        and transfer_response["transfer_policy_hash"]
                        == transfer_policy_hash
                        and exact_transfer_response["transfer_policy_hash"] is None
                    ),
                    "transfer_truth_roles_are_separated": (
                        validation_truth["truth_role"] == "validation"
                        and transfer_truth["truth_role"] == "evaluation"
                        and transfer_request["truth_role"] == "evaluation"
                        and exact_transfer_response["truth_role"] == "evaluation"
                        and transfer_response["truth_role"] == "evaluation"
                    ),
                    "all_transfer_policy_gates_passed": (
                        transfer_policy["passed"] is True
                        and all(
                            bool(value)
                            for value in transfer_policy["gates"].values()
                        )
                    ),
                }
            )
            risk_contract = transfer_policy.get("risk_contract")
            if isinstance(risk_contract, dict):
                selected_evaluation = transfer_policy[
                    "selected_validation_evaluation"
                ]
                selected_policy = transfer_policy["selected_policy"]
                derived_turns = [
                    turn
                    for turn in transfer_response["turns"]
                    if turn["artifact"]["claim_status"] == "derived"
                ]
                search_space = transfer_policy["search_space"]
                expected_policy_count = (
                    len(search_space["direction_prior_power"])
                    * len(search_space["pair_motif_power"])
                    * len(search_space["consensus_thresholds"]) ** 2
                )
                policy_runtime = transfer_policy["runtime"]
                expected_probe_key = (
                    "policy_neutral_projection_lattice"
                    if policy_runtime
                    in {
                        "atom-causal-metaplastic-transfer-policy-v4",
                        "atom-causal-metaplastic-transfer-policy-v5",
                        "atom-causal-metaplastic-transfer-policy-v6",
                        "atom-causal-metaplastic-transfer-policy-v7",
                    }
                    else "policy_neutral_context_factor_trace"
                )
                persisted_risk_values = [
                    float(evaluation["selective_error_upper_bound"])
                    for evaluation in (
                        transfer_policy["default_validation_evaluation"],
                        selected_evaluation,
                    )
                ]
                persisted_risk_values.extend(
                    float(
                        evaluation["by_expected_direction"][str(direction)][
                            "selective_error_upper_bound"
                        ]
                    )
                    for evaluation in (
                        transfer_policy["default_validation_evaluation"],
                        selected_evaluation,
                    )
                    for direction in (-1, 1)
                )
                transfer_checks.update(
                    {
                        "risk_contract_report_binding_exact": (
                            transfer_report["metaplastic_calibration"][
                                "risk_contract"
                            ]
                            == risk_contract
                        ),
                        "single_factor_probe_reused_exact": (
                            len(transfer_policy["probe_response_hashes"]) == 1
                            and transfer_policy["evaluated_policy_count"]
                            == expected_policy_count
                            and transfer_policy["gates"][
                                "single_factor_probe_reused"
                            ]
                        ),
                        "portable_policy_probe_digest_bound": (
                            set(transfer_policy["probe_response_hashes"])
                            == {expected_probe_key}
                            and isinstance(
                                transfer_policy["probe_response_hashes"][
                                    expected_probe_key
                                ],
                                str,
                            )
                            and len(
                                transfer_policy["probe_response_hashes"][
                                    expected_probe_key
                                ]
                            )
                            == 64
                            and all(
                                character in "0123456789abcdef"
                                for character in transfer_policy[
                                    "probe_response_hashes"
                                ][expected_probe_key]
                            )
                        ),
                        "portable_wilson_statistics_bound": (
                            policy_runtime
                            not in {
                                "atom-causal-metaplastic-transfer-policy-v5",
                                "atom-causal-metaplastic-transfer-policy-v6",
                                "atom-causal-metaplastic-transfer-policy-v7",
                            }
                            or all(
                                value == round(value, 12)
                                for value in persisted_risk_values
                            )
                        ),
                        "deterministic_decimal_wilson_declared": (
                            policy_runtime
                            not in {
                                "atom-causal-metaplastic-transfer-policy-v6",
                                "atom-causal-metaplastic-transfer-policy-v7",
                            }
                            or risk_contract["method"]
                            == "wilson_score_upper_bound_decimal12"
                        ),
                        "pair_motif_policy_exercised": (
                            "pair_motif_power" in selected_policy
                            and len(search_space["pair_motif_power"]) >= 3
                            and transfer_policy["gates"][
                                "pair_motif_controls_exercised"
                            ]
                        ),
                        "validation_overall_risk_bound_passed": (
                            selected_evaluation[
                                "selective_error_upper_bound"
                            ]
                            <= risk_contract[
                                "overall_selective_error_upper_limit"
                            ]
                            or selected_evaluation["asserted"] == 0
                            and transfer_policy["gates"][
                                "quick_profile_abstention_is_explicit"
                            ]
                        ),
                        "validation_directional_risk_bounds_passed": (
                            all(
                                selected_evaluation[
                                    "by_expected_direction"
                                ][str(direction)][
                                    "selective_error_upper_bound"
                                ]
                                <= risk_contract[
                                    "direction_selective_error_upper_limit"
                                ]
                                for direction in (-1, 1)
                            )
                            or selected_evaluation["asserted"] == 0
                            and transfer_policy["gates"][
                                "quick_profile_abstention_is_explicit"
                            ]
                        ),
                        "context_factor_trace_runtime_wired": (
                            bool(derived_turns)
                            and all(
                                isinstance(
                                    turn["artifact"].get(
                                        "context_factor_trace"
                                    ),
                                    dict,
                                )
                                and isinstance(
                                    turn["artifact"].get(
                                        "context_factor_projection"
                                    ),
                                    dict,
                                )
                                and runtime.project_context_factor_trace(
                                    turn["artifact"]["context_factor_trace"],
                                    selected_policy,
                                )
                                == turn["artifact"][
                                    "context_factor_projection"
                                ]
                                for turn in derived_turns
                            )
                        ),
                        "deterministic_decimal_factor_runtime_declared": (
                            policy_runtime
                            != "atom-causal-metaplastic-transfer-policy-v7"
                            or all(
                                turn["artifact"]["context_factor_trace"][
                                    "runtime"
                                ]
                                == "atom-causal-context-factor-graph-v2"
                                for turn in derived_turns
                            )
                        ),
                    }
                )
        checks.update(transfer_checks)
        transfer_metrics = {
            "truth_hash": truth_hash,
            "request_hash": request_hash,
            "report_hash": transfer_report_hash,
            "heldout_programs": transfer_report["heldout_program_count"],
            "cases": transfer_report["case_count"],
            "turns": transfer_report["turn_count"],
            "direction_counts": direction_counts,
            "exact_coverage": transfer_report["exact_baseline"]["coverage"],
            "contextual_coverage": transfer_report["contextual_transfer"][
                "coverage"
            ],
            "selective_accuracy": transfer_report["contextual_transfer"][
                "selective_accuracy"
            ],
            "selective_error_upper_bound": transfer_report[
                "contextual_transfer"
            ].get("selective_error_upper_bound"),
            "false_assertion_rate": transfer_report["contextual_transfer"][
                "false_assertion_rate"
            ],
            "safe_direction_utility": transfer_report["contextual_transfer"][
                "safe_direction_utility"
            ],
        }
        if calibrated_transfer and transfer_policy is not None:
            transfer_metrics["calibration"] = {
                "validation_truth_hash": validation_truth_hash,
                "policy_hash": transfer_policy_hash,
                "validation_programs": transfer_report[
                    "validation_program_count"
                ],
                "evaluated_policies": transfer_policy[
                    "evaluated_policy_count"
                ],
                "eligible_policies": transfer_policy["eligible_policy_count"],
                "selected_policy": transfer_policy["selected_policy"],
                "risk_contract": transfer_policy.get("risk_contract"),
                "default_validation_safe_utility": transfer_policy[
                    "default_validation_evaluation"
                ]["safe_direction_utility"],
                "selected_validation_safe_utility": transfer_policy[
                    "selected_validation_evaluation"
                ]["safe_direction_utility"],
            }
    elif require_transfer_benchmark:
        checks["transfer_benchmark_required"] = False

    hashed_files = list(EXPECTED_OUTPUT_FILES)
    if verify_transfer:
        hashed_files.extend(required_transfer_files)
    file_hashes = {name: _sha256(artifacts / name) for name in hashed_files}
    if log_files:
        file_hashes.update({path.name: _sha256(path) for path in log_files})
    return {
        "schema": 1,
        "run_dir": str(run_dir.resolve()),
        "artifact_dir": str(artifacts),
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "metrics": {
            "evidence_rows": evidence["row_count"],
            "domain_counts": evidence["domain_counts"],
            "axis_coverage": coverage,
            "maximum_invariant_error": execution["maximum_invariant_error"],
            "entity_updates": accelerator["entity_updates"],
            "relation_updates": accelerator["relation_updates"],
            "accelerator_elapsed_seconds": accelerator["elapsed_seconds"],
            "learned_laws": report["learning"]["graph_laws"],
            "crystallized_laws": report["learning"]["crystallized_laws"],
            "model_hash": report["model_hash"],
            "report_hash": report["report_hash"],
            "evidence_hash": report["learning"]["evidence_hash"],
            "cursor_hash": cursor["cursor_hash"],
            "source_sha256": source_hash,
            "source_file_sha256": source_file_hash,
            "transfer": transfer_metrics,
            "formal_domains": {
                "registry_hash": report["formal_domains"]["registry_hash"],
                "report_hash": report["formal_domains"]["report_hash"],
                "primitive_count": report["formal_domains"]["primitive_count"],
                "case_count": report["formal_domains"]["case_count"],
                "heldout_cases": report["formal_domains"]["partition_counts"][
                    "heldout"
                ],
                "passed": report["formal_domains"]["passed"],
            },
        },
        "file_sha256": file_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--require-transfer-benchmark", action="store_true")
    parser.add_argument(
        "--expected-accelerator",
        choices=("tpu", "gpu"),
        default="tpu",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_run(
        args.run_dir,
        source_file=args.source_file,
        expected_bundle_sha256=args.expected_bundle_sha256,
        require_transfer_benchmark=args.require_transfer_benchmark,
        expected_accelerator=args.expected_accelerator,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
