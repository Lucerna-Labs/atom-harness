"""Sealed held-out transfer benchmark for the Atom causal-world graph.

The evaluator compiles procedural regimes that never appeared in training,
derives truth from two independent treated/control simulations, and only then
asks the persisted graph ordinary-English questions.  Runtime inference never
receives the evaluator direction labels.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
from typing import Any, Mapping, Sequence

from atom_causal_graph import (
    CausalCognition,
    CausalGraph,
    project_context_factor_trace,
    validated_contextual_transfer_policy,
)
from atom_causal_world_curriculum import (
    WORLD_PROGRAM_AXES,
    curriculum_program_ids,
    decode_world_program,
    world_program_space_size,
)
from atom_causal_world_knowledge import CausalWorldWikiGraph, retrieve_causal_context
from atom_causal_world_language import (
    CAUSAL_WORLD_LANGUAGE_RUNTIME,
    parse_causal_question,
    render_causal_answer,
    render_causal_question,
)
from atom_causal_world_schema import CausalWorldConfig, canonical_hash
from atom_causal_world_simulator import (
    ProceduralWorldCompiler,
    generate_interventions,
    rollout_counterfactual_pair,
)


CAUSAL_TRANSFER_SCHEMA = 1
CAUSAL_TRANSFER_RUNTIME = "atom-causal-heldout-transfer-v9"
TRANSFER_POLICY_SCHEMA = 1
TRANSFER_POLICY_RUNTIME = "atom-causal-metaplastic-transfer-policy-v7"
TRANSFER_PROVENANCE_RUNTIME = "atom-causal-transfer-semantic-provenance-v1"
TRANSFER_FACTOR_TRACE_DIGEST_RUNTIME = (
    "atom-causal-context-factor-trace-digest-v1"
)
TRANSFER_PROJECTION_LATTICE_RUNTIME = (
    "atom-causal-context-factor-projection-lattice-v4"
)
CAUSAL_TRANSFER_SEED = 2026072602

TRANSFER_PROFILE_SEEDS = {
    "quick": {"validation": 2026072501, "evaluation": 2026072502},
    "massive": {"validation": 2026072601, "evaluation": CAUSAL_TRANSFER_SEED},
}

TRANSFER_PRIOR_POWER_SEARCH = (0.0, 0.20, 0.40, 0.60, 0.80)
TRANSFER_PAIR_MOTIF_POWER_SEARCH = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
)
TRANSFER_CONSENSUS_THRESHOLD_SEARCH = (
    0.75,
    0.80,
    0.85,
    0.90,
    0.93,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)
TRANSFER_RISK_CONFIDENCE_LEVEL = 0.95
TRANSFER_OVERALL_SELECTIVE_ERROR_LIMIT = 0.10
TRANSFER_DIRECTION_SELECTIVE_ERROR_LIMIT = 0.15
TRANSFER_RISK_METHOD = "wilson_score_upper_bound_decimal12"
_TRANSFER_RISK_Z_SCORE_DECIMAL = Decimal("1.959963984540054")
_TRANSFER_RISK_QUANTUM = Decimal("0.000000000001")

TRANSFER_PROFILES: Mapping[str, Mapping[str, int | float]] = {
    "quick": {
        "validation_programs": 4,
        "evaluation_programs": 3,
        "worlds": 16,
        "entities": 24,
        "neighbors": 4,
        "steps": 5,
        "interventions": 6,
        "cases_per_program": 8,
        "minimum_magnitude": 2e-5,
    },
    "massive": {
        "validation_programs": 24,
        "evaluation_programs": 12,
        "worlds": 32,
        "entities": 32,
        "neighbors": 6,
        "steps": 8,
        "interventions": 12,
        "cases_per_program": 24,
        "minimum_magnitude": 5e-5,
    },
}


def _profile(name: str) -> dict[str, int | float]:
    try:
        return dict(TRANSFER_PROFILES[name])
    except KeyError as error:
        raise ValueError("unknown causal transfer benchmark profile") from error


def stable_transfer_evidence_provenance(
    evidence: Any,
    *,
    program_id: int,
    replica: int,
) -> str:
    """Bind transfer evidence to portable causal measurements, not raw arrays."""

    if isinstance(program_id, bool) or not isinstance(program_id, int):
        raise TypeError("transfer provenance program ID must be an integer")
    if isinstance(replica, bool) or not isinstance(replica, int) or replica < 0:
        raise ValueError("transfer provenance replica must be nonnegative")
    evidence.validate()
    payload = {
        "runtime": TRANSFER_PROVENANCE_RUNTIME,
        "program_id": program_id,
        "replica": replica,
        "evidence_id": evidence.evidence_id,
        "domain": evidence.domain,
        "cause_feature": evidence.cause_feature,
        "effect_feature": evidence.effect_feature,
        "direction": evidence.direction,
        "magnitude": round(float(evidence.magnitude), 12),
        "delay": evidence.delay,
        "variance": round(float(evidence.variance), 12),
        "context_signature": list(evidence.context_signature),
        "treated_worlds": evidence.treated_worlds,
        "baseline_worlds": evidence.baseline_worlds,
    }
    return canonical_hash(payload)


def validate_transfer_policy_artifact(
    policy: Mapping[str, Any], *, model_hash: str
) -> dict[str, Any]:
    """Fail closed unless a learned projection policy is hash and model bound."""

    expected = {
        "default_policy",
        "default_validation_evaluation",
        "eligible_policy_count",
        "evaluated_policy_count",
        "gates",
        "model_hash",
        "passed",
        "policy_hash",
        "probe_response_hashes",
        "profile",
        "risk_contract",
        "runtime",
        "schema",
        "search_space",
        "selected_policy",
        "selected_validation_evaluation",
        "validation_program_ids",
        "validation_request_hash",
        "validation_truth_hash",
    }
    if set(policy) != expected:
        raise ValueError("transfer policy artifact fields are invalid")
    core = {key: policy[key] for key in sorted(expected - {"policy_hash"})}
    if policy["policy_hash"] != canonical_hash(core):
        raise ValueError("transfer policy artifact hash mismatch")
    if policy["schema"] != TRANSFER_POLICY_SCHEMA:
        raise ValueError("unsupported transfer policy schema")
    if policy["runtime"] != TRANSFER_POLICY_RUNTIME:
        raise ValueError("transfer policy runtime mismatch")
    if policy["model_hash"] != model_hash:
        raise ValueError("transfer policy is not bound to this model")
    risk_contract = policy["risk_contract"]
    if not isinstance(risk_contract, Mapping) or set(risk_contract) != {
        "confidence_level",
        "direction_selective_error_upper_limit",
        "method",
        "overall_selective_error_upper_limit",
    }:
        raise ValueError("transfer policy risk contract is invalid")
    if (
        risk_contract["method"] != TRANSFER_RISK_METHOD
        or float(risk_contract["confidence_level"])
        != TRANSFER_RISK_CONFIDENCE_LEVEL
        or float(risk_contract["overall_selective_error_upper_limit"])
        != TRANSFER_OVERALL_SELECTIVE_ERROR_LIMIT
        or float(risk_contract["direction_selective_error_upper_limit"])
        != TRANSFER_DIRECTION_SELECTIVE_ERROR_LIMIT
    ):
        raise ValueError("transfer policy risk contract values are invalid")
    if policy["passed"] is not True or not all(
        bool(value) for value in policy["gates"].values()
    ):
        raise ValueError("transfer policy did not pass its validation gates")
    probe_hashes = policy["probe_response_hashes"]
    if (
        not isinstance(probe_hashes, Mapping)
        or set(probe_hashes) != {"policy_neutral_projection_lattice"}
    ):
        raise ValueError("transfer policy projection lattice digest is invalid")
    trace_digest = probe_hashes["policy_neutral_projection_lattice"]
    if (
        not isinstance(trace_digest, str)
        or len(trace_digest) != 64
        or any(character not in "0123456789abcdef" for character in trace_digest)
    ):
        raise ValueError("transfer policy projection lattice digest is malformed")
    selected = validated_contextual_transfer_policy(policy["selected_policy"])
    default = validated_contextual_transfer_policy(policy["default_policy"])
    if selected != policy["selected_policy"] or default != policy["default_policy"]:
        raise ValueError("transfer policy controls are not canonical")
    return dict(policy)


def _program_tokens(program_id: int) -> frozenset[str]:
    return frozenset(decode_world_program(program_id).condition_signature())


def select_heldout_program_ids(
    training_program_ids: Sequence[int],
    *,
    count: int,
    seed: int = CAUSAL_TRANSFER_SEED,
) -> tuple[int, ...]:
    """Select diverse program recombinations disjoint from the training schedule."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("held-out program count must be a positive integer")
    training = {int(value) for value in training_program_ids}
    if not training:
        raise ValueError("held-out selection requires training program IDs")
    space = world_program_space_size()
    candidates: list[int] = []
    serial = 0
    while len(candidates) < max(512, count * 64):
        candidate = int(
            canonical_hash({"seed": seed, "serial": serial})[:16], 16
        ) % space
        serial += 1
        if candidate in training or candidate in candidates:
            continue
        candidates.append(candidate)

    selected: list[int] = []
    covered: set[str] = set()
    while len(selected) < count:
        best = max(
            (value for value in candidates if value not in selected),
            key=lambda value: (
                len(_program_tokens(value) - covered),
                min(
                    (
                        len(_program_tokens(value) ^ _program_tokens(other))
                        for other in selected
                    ),
                    default=len(_program_tokens(value)),
                ),
                -value,
            ),
        )
        selected.append(best)
        covered.update(_program_tokens(best))
    if training.intersection(selected):
        raise AssertionError("held-out selector leaked a training program")
    return tuple(selected)


def _benchmark_config(
    profile: Mapping[str, int | float], *, seed: int
) -> CausalWorldConfig:
    return CausalWorldConfig(
        profile="test",
        worlds_per_shard=int(profile["worlds"]),
        shard_count=2,
        entity_count=int(profile["entities"]),
        neighbor_count=int(profile["neighbors"]),
        time_steps=int(profile["steps"]),
        intervention_candidates=int(profile["interventions"]),
        active_experiments=int(profile["interventions"]),
        phase_dimensions=8,
        maximum_laws=512,
        seed=seed,
    )


def _replica_evidence(
    compiler: ProceduralWorldCompiler,
    *,
    program_id: int,
    replica: int,
    intervention_count: int,
    steps: int,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    world = compiler.compile_shard(replica, program_id=program_id)
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    maximum_invariant_error = 0.0
    trace_hashes: list[str] = []
    for intervention in generate_interventions(intervention_count):
        evidence, diagnostics = rollout_counterfactual_pair(world, intervention, steps)
        maximum_invariant_error = max(
            maximum_invariant_error, float(diagnostics["maximum_invariant_error"])
        )
        trace_hashes.append(str(diagnostics["effect_trace_digest"]["sha256"]))
        for item in evidence:
            key = (item.domain, item.cause_feature, item.effect_feature)
            rows[key] = {
                "direction": item.direction,
                "magnitude": item.magnitude,
                "delay": item.delay,
                "variance": item.variance,
                "provenance_hash": stable_transfer_evidence_provenance(
                    item,
                    program_id=program_id,
                    replica=replica,
                ),
            }
    return rows, {
        "replica": replica,
        "program_id": program_id,
        "evidence_rows": len(rows),
        "maximum_invariant_error": maximum_invariant_error,
        "trace_hash": canonical_hash(trace_hashes),
    }


def _select_balanced_cross_feature_cases(
    stable: Sequence[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """Select non-trivial causal cases without letting one direction dominate."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
        raise ValueError("transfer case limit must be an integer of at least two")
    candidates = [
        dict(value)
        for value in stable
        if value["cause_feature"] != value["effect_feature"]
        and int(value["expected_direction"]) in {-1, 1}
    ]
    by_direction = {
        direction: sorted(
            (
                value
                for value in candidates
                if int(value["expected_direction"]) == direction
            ),
            key=lambda value: (
                -float(value["mean_magnitude"]),
                str(value["domain"]),
                str(value["cause_feature"]),
                str(value["effect_feature"]),
            ),
        )
        for direction in (-1, 1)
    }
    selected: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str, str]] = set()
    target = {-1: limit // 2, 1: limit - limit // 2}

    def take(direction: int, desired: int) -> None:
        for candidate in by_direction[direction]:
            pair = (
                str(candidate["domain"]),
                str(candidate["cause_feature"]),
                str(candidate["effect_feature"]),
            )
            if pair in used_pairs:
                continue
            selected.append(candidate)
            used_pairs.add(pair)
            if sum(
                int(value["expected_direction"]) == direction for value in selected
            ) >= desired:
                return

    for direction in (-1, 1):
        take(direction, target[direction])
    if len(selected) < limit:
        remainder = sorted(
            (value for value in candidates if value not in selected),
            key=lambda value: (
                -float(value["mean_magnitude"]),
                str(value["domain"]),
                str(value["cause_feature"]),
                str(value["effect_feature"]),
            ),
        )
        for candidate in remainder:
            pair = (
                str(candidate["domain"]),
                str(candidate["cause_feature"]),
                str(candidate["effect_feature"]),
            )
            if pair in used_pairs:
                continue
            selected.append(candidate)
            used_pairs.add(pair)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _build_transfer_truth(
    training_program_ids: Sequence[int],
    *,
    profile_name: str,
    truth_role: str,
    seed: int,
    excluded_program_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Generate role-bound truth from procedural regimes excluded from training."""

    if truth_role not in {"validation", "evaluation"}:
        raise ValueError("transfer truth role must be validation or evaluation")
    profile = _profile(profile_name)
    config = _benchmark_config(profile, seed=seed)
    compiler = ProceduralWorldCompiler(config)
    training = {int(value) for value in training_program_ids}
    excluded = {int(value) for value in excluded_program_ids}
    if training.intersection(excluded):
        raise ValueError("transfer exclusions overlap training programs")
    program_count_key = (
        "validation_programs"
        if truth_role == "validation"
        else "evaluation_programs"
    )
    heldout_ids = select_heldout_program_ids(
        sorted(training | excluded),
        count=int(profile[program_count_key]),
        seed=seed,
    )
    cases: list[dict[str, Any]] = []
    program_reports: list[dict[str, Any]] = []
    minimum_magnitude = float(profile["minimum_magnitude"])
    for program_id in heldout_ids:
        replicas = [
            _replica_evidence(
                compiler,
                program_id=program_id,
                replica=replica,
                intervention_count=int(profile["interventions"]),
                steps=int(profile["steps"]),
            )
            for replica in range(2)
        ]
        first, second = replicas[0][0], replicas[1][0]
        stable: list[dict[str, Any]] = []
        for domain, cause, effect in sorted(set(first).intersection(second)):
            left = first[(domain, cause, effect)]
            right = second[(domain, cause, effect)]
            magnitude = 0.5 * (float(left["magnitude"]) + float(right["magnitude"]))
            if (
                int(left["direction"]) != int(right["direction"])
                or magnitude < minimum_magnitude
            ):
                continue
            stable.append(
                {
                    "domain": domain,
                    "cause_feature": cause,
                    "effect_feature": effect,
                    "expected_direction": int(left["direction"]),
                    "mean_magnitude": magnitude,
                    "mean_delay": 0.5
                    * (float(left["delay"]) + float(right["delay"])),
                    "replica_provenance": [
                        str(left["provenance_hash"]),
                        str(right["provenance_hash"]),
                    ],
                }
            )
        selected = _select_balanced_cross_feature_cases(
            stable, limit=int(profile["cases_per_program"])
        )
        if len(selected) < min(4, int(profile["cases_per_program"])):
            raise RuntimeError("held-out program produced too little stable evaluator truth")
        program = decode_world_program(program_id)
        for candidate in selected:
            case_core = {
                "program_id": program_id,
                "condition_signature": list(program.condition_signature()),
                **candidate,
            }
            cases.append(
                {
                    "case_id": "transfer-" + canonical_hash(case_core)[:24],
                    **case_core,
                }
            )
        program_reports.append(
            {
                "program": program.manifest(),
                "stable_candidates": len(stable),
                "stable_cross_feature_candidates": sum(
                    value["cause_feature"] != value["effect_feature"]
                    for value in stable
                ),
                "selected_direction_counts": {
                    str(direction): sum(
                        int(value["expected_direction"]) == direction
                        for value in selected
                    )
                    for direction in (-1, 1)
                },
                "selected_cases": len(selected),
                "replicas": [item[1] for item in replicas],
            }
        )
    core = {
        "schema": CAUSAL_TRANSFER_SCHEMA,
        "runtime": CAUSAL_TRANSFER_RUNTIME,
        "profile": profile_name,
        "truth_role": truth_role,
        "truth_seed": seed,
        "training_program_ids": sorted(training),
        "excluded_program_ids": sorted(excluded),
        "heldout_program_ids": list(heldout_ids),
        "config": asdict(config),
        "profile_config": profile,
        "cases": cases,
        "program_reports": program_reports,
    }
    return {**core, "truth_hash": canonical_hash(core)}


def build_transfer_validation_truth(
    training_program_ids: Sequence[int], *, profile_name: str
) -> dict[str, Any]:
    """Create the disjoint worlds allowed to shape the projection policy."""

    return _build_transfer_truth(
        training_program_ids,
        profile_name=profile_name,
        truth_role="validation",
        seed=int(TRANSFER_PROFILE_SEEDS[profile_name]["validation"]),
    )


def build_transfer_evaluator_truth(
    training_program_ids: Sequence[int],
    *,
    profile_name: str,
    excluded_program_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Create final evaluator worlds hidden from projection-policy selection."""

    return _build_transfer_truth(
        training_program_ids,
        profile_name=profile_name,
        truth_role="evaluation",
        seed=int(TRANSFER_PROFILE_SEEDS[profile_name]["evaluation"]),
        excluded_program_ids=excluded_program_ids,
    )


def build_transfer_request(truth: Mapping[str, Any]) -> dict[str, Any]:
    if truth.get("runtime") != CAUSAL_TRANSFER_RUNTIME:
        raise ValueError("transfer truth runtime mismatch")
    turns: list[dict[str, Any]] = []
    for index, case in enumerate(truth["cases"]):
        for paraphrase_index, variant in enumerate((index * 8, index * 8 + 4)):
            request = render_causal_question(
                query_id=f"{case['case_id']}-p{paraphrase_index}",
                domain=str(case["domain"]),
                cause_feature=str(case["cause_feature"]),
                effect_feature=str(case["effect_feature"]),
                variant=variant,
                condition_signature=tuple(case["condition_signature"]),
            )
            turns.append(
                {
                    "case_id": str(case["case_id"]),
                    "paraphrase_index": paraphrase_index,
                    "request": request,
                }
            )
    core = {
        "schema": CAUSAL_TRANSFER_SCHEMA,
        "runtime": CAUSAL_TRANSFER_RUNTIME,
        "profile": truth["profile"],
        "truth_role": truth["truth_role"],
        "sealed_truth_hash": truth["truth_hash"],
        "turns": turns,
    }
    return {**core, "request_hash": canonical_hash(core)}


def _run_transfer_workflow_core(
    model_payload: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    *,
    allow_contextual_transfer: bool,
    selected_policy: Mapping[str, Any] | None,
    policy_hash: str | None,
) -> dict[str, Any]:
    expected = {
        "profile",
        "request_hash",
        "runtime",
        "schema",
        "sealed_truth_hash",
        "truth_role",
        "turns",
    }
    if set(request_payload) != expected:
        raise ValueError("transfer request fields are invalid")
    core = {key: request_payload[key] for key in sorted(expected - {"request_hash"})}
    if request_payload["request_hash"] != canonical_hash(core):
        raise ValueError("transfer request hash mismatch")
    if request_payload["schema"] != CAUSAL_TRANSFER_SCHEMA:
        raise ValueError("unsupported transfer request schema")
    if request_payload["runtime"] != CAUSAL_TRANSFER_RUNTIME:
        raise ValueError("transfer request runtime mismatch")
    if request_payload["truth_role"] not in {"validation", "evaluation"}:
        raise ValueError("transfer request truth role is invalid")
    controls = validated_contextual_transfer_policy(selected_policy)
    graph = CausalGraph.from_model_payload(model_payload)
    cognition = CausalCognition(graph, transfer_policy=controls)
    wiki = CausalWorldWikiGraph(graph)
    turns: list[dict[str, Any]] = []
    for wrapper in request_payload["turns"]:
        if set(wrapper) != {"case_id", "paraphrase_index", "request"}:
            raise ValueError("transfer turn fields are invalid")
        request = wrapper["request"]
        query = parse_causal_question(request)
        artifact = cognition.answer(
            query, allow_contextual_transfer=allow_contextual_transfer
        )
        turns.append(
            {
                "case_id": wrapper["case_id"],
                "paraphrase_index": wrapper["paraphrase_index"],
                "request": dict(request),
                "parsed_query": {
                    **asdict(query),
                    "context_signature": list(query.context_signature),
                },
                "artifact": artifact,
                "answer": render_causal_answer(artifact),
                "knowledge_context": retrieve_causal_context(
                    wiki,
                    str(request["text"]),
                    limit=4,
                    domain=query.domain,
                    cause_feature=query.cause_feature,
                    effect_feature=query.effect_feature,
                ),
            }
        )
    response_core = {
        "schema": CAUSAL_TRANSFER_SCHEMA,
        "runtime": CAUSAL_TRANSFER_RUNTIME,
        "mode": "contextual_transfer" if allow_contextual_transfer else "exact_only",
        "model_hash": model_payload["model_hash"],
        "request_hash": request_payload["request_hash"],
        "sealed_truth_hash": request_payload["sealed_truth_hash"],
        "truth_role": request_payload["truth_role"],
        "transfer_policy_hash": policy_hash,
        "turns": turns,
    }
    return {**response_core, "response_hash": canonical_hash(response_core)}


def run_transfer_workflow(
    model_payload: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    *,
    allow_contextual_transfer: bool,
    transfer_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_policy: Mapping[str, Any] | None = None
    policy_hash: str | None = None
    if transfer_policy is not None:
        if not allow_contextual_transfer:
            raise ValueError("exact-only transfer cannot receive a learned policy")
        validated = validate_transfer_policy_artifact(
            transfer_policy, model_hash=str(model_payload["model_hash"])
        )
        selected_policy = validated["selected_policy"]
        policy_hash = str(validated["policy_hash"])
    return _run_transfer_workflow_core(
        model_payload,
        request_payload,
        allow_contextual_transfer=allow_contextual_transfer,
        selected_policy=selected_policy,
        policy_hash=policy_hash,
    )


@lru_cache(maxsize=None)
def _deterministic_wilson_upper_bound(
    false_assertions: int,
    asserted: int,
) -> float:
    """Compute one validated Wilson bound with deterministic decimal arithmetic."""

    if asserted == 0:
        return 1.0
    with localcontext() as context:
        context.prec = 80
        false_decimal = Decimal(false_assertions)
        asserted_decimal = Decimal(asserted)
        rate = false_decimal / asserted_decimal
        squared = _TRANSFER_RISK_Z_SCORE_DECIMAL * _TRANSFER_RISK_Z_SCORE_DECIMAL
        denominator = Decimal(1) + squared / asserted_decimal
        center = rate + squared / (Decimal(2) * asserted_decimal)
        radius = _TRANSFER_RISK_Z_SCORE_DECIMAL * (
            rate * (Decimal(1) - rate) / asserted_decimal
            + squared / (Decimal(4) * asserted_decimal * asserted_decimal)
        ).sqrt()
        bound = ((center + radius) / denominator).quantize(
            _TRANSFER_RISK_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
    return float(bound)


def selective_error_upper_bound(
    false_assertions: int,
    asserted: int,
) -> float:
    """Return a deterministic twelve-decimal 95% Wilson error upper bound."""

    if (
        isinstance(false_assertions, bool)
        or not isinstance(false_assertions, int)
        or isinstance(asserted, bool)
        or not isinstance(asserted, int)
        or false_assertions < 0
        or asserted < 0
        or false_assertions > asserted
    ):
        raise ValueError("selective error counts are invalid")
    return _deterministic_wilson_upper_bound(false_assertions, asserted)


def evaluate_transfer_response(
    response: Mapping[str, Any], truth: Mapping[str, Any]
) -> dict[str, Any]:
    truth_by_id = {str(case["case_id"]): case for case in truth["cases"]}
    if response.get("sealed_truth_hash") != truth.get("truth_hash"):
        raise ValueError("transfer response is not bound to evaluator truth")
    correct = 0
    false_assertions = 0
    abstentions = 0
    transfer_assertions = 0
    by_case: dict[str, list[tuple[str, int | None, str]]] = {}
    by_direction = {
        direction: {
            "turns": 0,
            "correct_assertions": 0,
            "false_assertions": 0,
            "abstentions": 0,
        }
        for direction in (-1, 1)
    }
    for turn in response["turns"]:
        case = truth_by_id[str(turn["case_id"])]
        expected_direction = int(case["expected_direction"])
        artifact = turn["artifact"]
        status = str(artifact["claim_status"])
        direction = artifact.get("direction")
        derivation = str(artifact.get("derivation_kind"))
        by_case.setdefault(str(turn["case_id"]), []).append(
            (status, int(direction) if direction is not None else None, derivation)
        )
        direction_metrics = by_direction[expected_direction]
        direction_metrics["turns"] += 1
        if status == "unknown":
            abstentions += 1
            direction_metrics["abstentions"] += 1
        elif int(direction) == expected_direction:
            correct += 1
            direction_metrics["correct_assertions"] += 1
        else:
            false_assertions += 1
            direction_metrics["false_assertions"] += 1
        transfer_assertions += derivation == "contextual_transfer"
    total = len(response["turns"])
    asserted = correct + false_assertions
    consistent_cases = sum(
        len(values) == 2 and values[0] == values[1] for values in by_case.values()
    )
    rendered_by_direction = {}
    for direction, metrics in by_direction.items():
        direction_asserted = (
            metrics["correct_assertions"] + metrics["false_assertions"]
        )
        rendered_by_direction[str(direction)] = {
            **metrics,
            "asserted": direction_asserted,
            "coverage": direction_asserted / max(metrics["turns"], 1),
            "selective_accuracy": metrics["correct_assertions"]
            / max(direction_asserted, 1),
            "selective_error_rate": metrics["false_assertions"]
            / max(direction_asserted, 1),
            "selective_error_upper_bound": selective_error_upper_bound(
                metrics["false_assertions"], direction_asserted
            ),
        }
    core = {
        "turns": total,
        "cases": len(truth_by_id),
        "correct_assertions": correct,
        "false_assertions": false_assertions,
        "abstentions": abstentions,
        "asserted": asserted,
        "coverage": asserted / max(total, 1),
        "direction_accuracy": correct / max(total, 1),
        "selective_accuracy": correct / max(asserted, 1),
        "selective_error_rate": false_assertions / max(asserted, 1),
        "selective_error_upper_bound": selective_error_upper_bound(
            false_assertions, asserted
        ),
        "risk_confidence_level": TRANSFER_RISK_CONFIDENCE_LEVEL,
        "safe_direction_utility": (correct - false_assertions) / max(total, 1),
        "false_assertion_rate": false_assertions / max(total, 1),
        "abstention_rate": abstentions / max(total, 1),
        "contextual_transfer_assertions": transfer_assertions,
        "paraphrase_consistency": consistent_cases / max(len(by_case), 1),
        "by_expected_direction": rendered_by_direction,
    }
    return {**core, "evaluation_hash": canonical_hash(core)}


def factor_probe_trace_hash(probe_response: Mapping[str, Any]) -> str:
    """Hash complete factor traces for diagnostics, not policy identity."""

    if (
        probe_response.get("runtime") != CAUSAL_TRANSFER_RUNTIME
        or probe_response.get("mode") != "contextual_transfer"
        or probe_response.get("truth_role") != "validation"
    ):
        raise ValueError("factor probe response contract is invalid")
    turns = probe_response.get("turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError("factor probe turns must be a sequence")
    portable_turns: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError("factor probe turn must be an object")
        artifact = turn.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("factor probe artifact must be an object")
        trace = artifact.get("context_factor_trace")
        if trace is not None and not isinstance(trace, Mapping):
            raise ValueError("factor probe trace must be an object or null")
        portable_turns.append(
            {
                "case_id": str(turn["case_id"]),
                "paraphrase_index": int(turn["paraphrase_index"]),
                "context_factor_trace": (
                    dict(trace) if isinstance(trace, Mapping) else None
                ),
            }
        )
    core = {
        "runtime": TRANSFER_FACTOR_TRACE_DIGEST_RUNTIME,
        "model_hash": str(probe_response["model_hash"]),
        "request_hash": str(probe_response["request_hash"]),
        "sealed_truth_hash": str(probe_response["sealed_truth_hash"]),
        "turns": portable_turns,
    }
    return canonical_hash(core)


def factor_projection_lattice_hash(
    *,
    model_hash: str,
    request_hash: str,
    truth_hash: str,
    evaluations: Sequence[Mapping[str, Any]],
) -> str:
    """Hash every searchable policy and its observable factor-projection result."""

    for name, value in (
        ("model", model_hash),
        ("request", request_hash),
        ("truth", truth_hash),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"projection lattice {name} hash is malformed")
    rendered = []
    for item in evaluations:
        if not isinstance(item, Mapping) or set(item) != {
            "evaluation_hash",
            "policy",
        }:
            raise ValueError("projection lattice evaluation fields are invalid")
        evaluation_hash = item["evaluation_hash"]
        if (
            not isinstance(evaluation_hash, str)
            or len(evaluation_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in evaluation_hash
            )
        ):
            raise ValueError("projection lattice evaluation hash is malformed")
        rendered.append(
            {
                "policy": validated_contextual_transfer_policy(item["policy"]),
                "evaluation_hash": evaluation_hash,
            }
        )
    if not rendered:
        raise ValueError("projection lattice cannot be empty")
    return canonical_hash(
        {
            "runtime": TRANSFER_PROJECTION_LATTICE_RUNTIME,
            "model_hash": model_hash,
            "request_hash": request_hash,
            "truth_hash": truth_hash,
            "evaluations": rendered,
        }
    )


def _evaluate_factor_probe(
    probe_response: Mapping[str, Any],
    truth: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproject one policy-neutral factor trace without rerunning cognition."""

    controls = validated_contextual_transfer_policy(policy)
    filtered_turns: list[dict[str, Any]] = []
    for turn in probe_response["turns"]:
        artifact = turn["artifact"]
        trace = artifact.get("context_factor_trace")
        projection = (
            project_context_factor_trace(trace, controls)
            if isinstance(trace, Mapping)
            else None
        )
        direction = (
            int(projection["candidate_direction"])
            if projection is not None
            else None
        )
        accepted = (
            projection is not None
            and direction in {-1, 1}
            and bool(projection["structural_support"])
            and float(projection["consensus"])
            >= float(controls["consensus_thresholds"][str(direction)])
        )
        filtered_turns.append(
            {
                "case_id": turn["case_id"],
                "artifact": (
                    {
                        "claim_status": "derived",
                        "direction": int(direction),
                        "derivation_kind": "contextual_transfer",
                    }
                    if accepted
                    else {
                        "claim_status": "unknown",
                        "direction": None,
                        "derivation_kind": "contextual_transfer_rejected",
                    }
                ),
            }
        )
    return evaluate_transfer_response(
        {
            "sealed_truth_hash": probe_response["sealed_truth_hash"],
            "turns": filtered_turns,
        },
        truth,
    )


def _policy_selection_key(
    evaluation: Mapping[str, Any],
    selected_policy: Mapping[str, Any],
) -> tuple[float, ...]:
    by_direction = evaluation["by_expected_direction"]
    negative_coverage = float(by_direction["-1"]["coverage"])
    positive_coverage = float(by_direction["1"]["coverage"])
    thresholds = selected_policy["consensus_thresholds"]
    return (
        float(evaluation["safe_direction_utility"]),
        min(negative_coverage, positive_coverage),
        -abs(negative_coverage - positive_coverage),
        float(evaluation["coverage"]),
        float(evaluation["selective_accuracy"]),
        -float(evaluation["selective_error_upper_bound"]),
        -float(evaluation["false_assertion_rate"]),
        -float(thresholds["-1"]) - float(thresholds["1"]),
        -float(selected_policy["direction_prior_power"]),
        -float(selected_policy["pair_motif_power"]),
    )


def fit_transfer_policy(
    model_payload: Mapping[str, Any],
    validation_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Select metaplastic projection controls using validation worlds only."""

    if validation_truth.get("truth_role") != "validation":
        raise ValueError("transfer policy requires validation-role truth")
    validation_request = build_transfer_request(validation_truth)
    default_policy = validated_contextual_transfer_policy(None)
    probe_controls = validated_contextual_transfer_policy(
        {
            "direction_prior_power": 0.0,
            "token_likelihood_power": default_policy["token_likelihood_power"],
            "pair_motif_power": 0.0,
            "consensus_thresholds": {"-1": 0.50, "1": 0.50},
        }
    )
    factor_probe = _run_transfer_workflow_core(
        model_payload,
        validation_request,
        allow_contextual_transfer=True,
        selected_policy=probe_controls,
        policy_hash=None,
    )
    default_evaluation = _evaluate_factor_probe(
        factor_probe,
        validation_truth,
        default_policy,
    )
    default_by_direction = default_evaluation["by_expected_direction"]
    default_minimum_direction_coverage = min(
        float(default_by_direction[str(direction)]["coverage"])
        for direction in (-1, 1)
    )

    evaluated = 0
    projection_lattice: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for prior_power in TRANSFER_PRIOR_POWER_SEARCH:
        for pair_motif_power in TRANSFER_PAIR_MOTIF_POWER_SEARCH:
            for negative_threshold in TRANSFER_CONSENSUS_THRESHOLD_SEARCH:
                for positive_threshold in TRANSFER_CONSENSUS_THRESHOLD_SEARCH:
                    selected_policy = validated_contextual_transfer_policy(
                        {
                            "direction_prior_power": prior_power,
                            "token_likelihood_power": default_policy[
                                "token_likelihood_power"
                            ],
                            "pair_motif_power": pair_motif_power,
                            "consensus_thresholds": {
                                "-1": negative_threshold,
                                "1": positive_threshold,
                            },
                        }
                    )
                    evaluation = _evaluate_factor_probe(
                        factor_probe,
                        validation_truth,
                        selected_policy,
                    )
                    projection_lattice.append(
                        {
                            "policy": selected_policy,
                            "evaluation_hash": evaluation["evaluation_hash"],
                        }
                    )
                    evaluated += 1
                    by_direction = evaluation["by_expected_direction"]
                    minimum_direction_coverage = min(
                        float(by_direction[str(direction)]["coverage"])
                        for direction in (-1, 1)
                    )
                    directionally_accurate = all(
                        float(
                            by_direction[str(direction)]["selective_accuracy"]
                        )
                        >= 0.75
                        for direction in (-1, 1)
                    )
                    directionally_risk_bounded = all(
                        float(
                            by_direction[str(direction)][
                                "selective_error_upper_bound"
                            ]
                        )
                        <= TRANSFER_DIRECTION_SELECTIVE_ERROR_LIMIT
                        for direction in (-1, 1)
                    )
                    if (
                        float(evaluation["selective_error_upper_bound"])
                        <= TRANSFER_OVERALL_SELECTIVE_ERROR_LIMIT
                        and directionally_risk_bounded
                        and float(evaluation["paraphrase_consistency"]) == 1.0
                        and directionally_accurate
                        and float(evaluation["safe_direction_utility"])
                        >= float(default_evaluation["safe_direction_utility"])
                        and minimum_direction_coverage
                        >= default_minimum_direction_coverage
                    ):
                        eligible.append(
                            (
                                _policy_selection_key(
                                    evaluation, selected_policy
                                ),
                                selected_policy,
                                evaluation,
                            )
                        )
    quick_abstention_fallback = not eligible and validation_truth["profile"] == "quick"
    if quick_abstention_fallback:
        selected_policy = default_policy
        selected_evaluation = default_evaluation
    elif eligible:
        _, selected_policy, selected_evaluation = max(
            eligible,
            key=lambda item: (
                item[0],
                canonical_hash(item[1]),
            ),
        )
    else:
        raise RuntimeError("no transfer policy survived validation constraints")
    probe_hashes = {
        "policy_neutral_projection_lattice": factor_projection_lattice_hash(
            model_hash=str(model_payload["model_hash"]),
            request_hash=str(validation_request["request_hash"]),
            truth_hash=str(validation_truth["truth_hash"]),
            evaluations=projection_lattice,
        )
    }
    selected_by_direction = selected_evaluation["by_expected_direction"]
    direction_counts = {
        str(direction): sum(
            int(case["expected_direction"]) == direction
            for case in validation_truth["cases"]
        )
        for direction in (-1, 1)
    }
    validation_programs = set(
        int(value) for value in validation_truth["heldout_program_ids"]
    )
    training_programs = set(
        int(value) for value in validation_truth["training_program_ids"]
    )
    gates = {
        "validation_truth_is_role_bound": validation_truth["truth_role"]
        == "validation",
        "validation_programs_disjoint_from_training": not validation_programs.intersection(
            training_programs
        ),
        "validation_truth_uses_two_replicas": all(
            len(report["replicas"]) == 2
            for report in validation_truth["program_reports"]
        ),
        "validation_truth_is_direction_balanced": min(direction_counts.values())
        / max(sum(direction_counts.values()), 1)
        >= 0.35,
        "single_factor_probe_reused": (
            len(probe_hashes) == 1
            and evaluated
            == len(TRANSFER_PRIOR_POWER_SEARCH)
            * len(TRANSFER_PAIR_MOTIF_POWER_SEARCH)
            * len(TRANSFER_CONSENSUS_THRESHOLD_SEARCH) ** 2
        ),
        "multiple_prior_controls_exercised": len(TRANSFER_PRIOR_POWER_SEARCH)
        >= 3,
        "pair_motif_controls_exercised": len(
            TRANSFER_PAIR_MOTIF_POWER_SEARCH
        )
        >= 3,
        "selected_policy_is_selectively_accurate": (
            selected_evaluation["selective_accuracy"] >= 0.82
            or quick_abstention_fallback
            and selected_evaluation["asserted"] == 0
        ),
        "selected_policy_false_assertions_bounded": selected_evaluation[
            "false_assertion_rate"
        ]
        <= 0.10,
        "selected_policy_overall_risk_bound": (
            selected_evaluation["selective_error_upper_bound"]
            <= TRANSFER_OVERALL_SELECTIVE_ERROR_LIMIT
            or quick_abstention_fallback
            and selected_evaluation["asserted"] == 0
        ),
        "selected_policy_directional_risk_bounds": (
            all(
                selected_by_direction[str(direction)][
                    "selective_error_upper_bound"
                ]
                <= TRANSFER_DIRECTION_SELECTIVE_ERROR_LIMIT
                for direction in (-1, 1)
            )
            or quick_abstention_fallback
            and selected_evaluation["asserted"] == 0
        ),
        "selected_policy_preserves_both_directions": (
            all(
                selected_by_direction[str(direction)]["selective_accuracy"] >= 0.75
                and selected_by_direction[str(direction)]["coverage"]
                >= default_minimum_direction_coverage
                for direction in (-1, 1)
            )
            or quick_abstention_fallback
            and all(
                selected_by_direction[str(direction)]["false_assertions"] == 0
                for direction in (-1, 1)
            )
        ),
        "quick_profile_abstention_is_explicit": (
            not quick_abstention_fallback
            or selected_evaluation["abstention_rate"] == 1.0
        ),
        "selected_policy_utility_not_below_default": selected_evaluation[
            "safe_direction_utility"
        ]
        >= default_evaluation["safe_direction_utility"],
        "validation_paraphrases_are_invariant": selected_evaluation[
            "paraphrase_consistency"
        ]
        == 1.0,
    }
    core = {
        "schema": TRANSFER_POLICY_SCHEMA,
        "runtime": TRANSFER_POLICY_RUNTIME,
        "model_hash": model_payload["model_hash"],
        "profile": validation_truth["profile"],
        "validation_truth_hash": validation_truth["truth_hash"],
        "validation_request_hash": validation_request["request_hash"],
        "validation_program_ids": sorted(validation_programs),
        "risk_contract": {
            "method": TRANSFER_RISK_METHOD,
            "confidence_level": TRANSFER_RISK_CONFIDENCE_LEVEL,
            "overall_selective_error_upper_limit": (
                TRANSFER_OVERALL_SELECTIVE_ERROR_LIMIT
            ),
            "direction_selective_error_upper_limit": (
                TRANSFER_DIRECTION_SELECTIVE_ERROR_LIMIT
            ),
        },
        "default_policy": default_policy,
        "selected_policy": selected_policy,
        "search_space": {
            "direction_prior_power": list(TRANSFER_PRIOR_POWER_SEARCH),
            "token_likelihood_power": [
                default_policy["token_likelihood_power"]
            ],
            "pair_motif_power": list(TRANSFER_PAIR_MOTIF_POWER_SEARCH),
            "consensus_thresholds": list(
                TRANSFER_CONSENSUS_THRESHOLD_SEARCH
            ),
            "objective": (
                "risk_bounded_safe_utility_then_minimum_direction_coverage"
                "_then_balance"
            ),
        },
        "evaluated_policy_count": evaluated,
        "eligible_policy_count": len(eligible),
        "probe_response_hashes": probe_hashes,
        "default_validation_evaluation": default_evaluation,
        "selected_validation_evaluation": selected_evaluation,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return {**core, "policy_hash": canonical_hash(core)}


def _adversarial_language_checks(request: Mapping[str, Any]) -> dict[str, Any]:
    first = dict(request["turns"][0]["request"])
    variants = [
        {
            **first,
            "text": str(first["text"])
            + " The world regime has open boundary and closed boundary.",
        },
        {
            "query_id": "adversarial-ungrounded",
            "text": "What does changing quantum foam do in a physical world?",
            "language_runtime": CAUSAL_WORLD_LANGUAGE_RUNTIME,
        },
        {**first, "unexpected_truth": 1},
    ]
    rejected = 0
    errors: list[str] = []
    for variant in variants:
        try:
            parse_causal_question(variant)
        except (TypeError, ValueError) as error:
            rejected += 1
            errors.append(type(error).__name__)
    return {
        "variants": len(variants),
        "rejected": rejected,
        "errors": errors,
        "passed": rejected == len(variants),
    }


def run_causal_transfer_benchmark(
    model_payload: Mapping[str, Any],
    training_program_ids: Sequence[int],
    *,
    profile_name: str,
) -> dict[str, Any]:
    validation_truth = build_transfer_validation_truth(
        training_program_ids, profile_name=profile_name
    )
    transfer_policy = fit_transfer_policy(model_payload, validation_truth)
    validation_program_ids = tuple(validation_truth["heldout_program_ids"])
    truth = build_transfer_evaluator_truth(
        training_program_ids,
        profile_name=profile_name,
        excluded_program_ids=validation_program_ids,
    )
    request = build_transfer_request(truth)
    exact_response = run_transfer_workflow(
        model_payload, request, allow_contextual_transfer=False
    )
    transfer_response = run_transfer_workflow(
        model_payload,
        request,
        allow_contextual_transfer=True,
        transfer_policy=transfer_policy,
    )
    exact_evaluation = evaluate_transfer_response(exact_response, truth)
    transfer_evaluation = evaluate_transfer_response(transfer_response, truth)
    adversarial = _adversarial_language_checks(request)
    training = set(int(value) for value in training_program_ids)
    validation = set(int(value) for value in validation_program_ids)
    heldout = set(int(value) for value in truth["heldout_program_ids"])
    direction_counts = {
        str(direction): sum(
            int(case["expected_direction"]) == direction for case in truth["cases"]
        )
        for direction in (-1, 1)
    }
    majority_direction = max(
        (-1, 1), key=lambda direction: (direction_counts[str(direction)], direction)
    )
    majority_accuracy = direction_counts[str(majority_direction)] / max(
        len(truth["cases"]), 1
    )
    majority_safe_utility = (
        direction_counts[str(majority_direction)]
        - direction_counts[str(-majority_direction)]
    ) / max(len(truth["cases"]), 1)
    transfer_by_direction = transfer_evaluation["by_expected_direction"]
    gates = {
        "heldout_programs_disjoint_from_training": not training.intersection(heldout),
        "validation_programs_disjoint_from_training": not training.intersection(
            validation
        ),
        "evaluation_programs_disjoint_from_validation": not validation.intersection(
            heldout
        ),
        "metaplastic_policy_passed_validation": transfer_policy["passed"]
        and all(bool(value) for value in transfer_policy["gates"].values()),
        "metaplastic_policy_is_model_bound": (
            transfer_policy["model_hash"] == model_payload["model_hash"]
            and transfer_response["transfer_policy_hash"]
            == transfer_policy["policy_hash"]
        ),
        "evaluator_truth_sealed_from_runtime": (
            exact_response["sealed_truth_hash"] == truth["truth_hash"]
            and transfer_response["sealed_truth_hash"] == truth["truth_hash"]
            and exact_response["request_hash"] == request["request_hash"]
            and transfer_response["request_hash"] == request["request_hash"]
            and exact_response["truth_role"] == "evaluation"
            and transfer_response["truth_role"] == "evaluation"
        ),
        "two_independent_truth_replicas": all(
            len(report["replicas"]) == 2 for report in truth["program_reports"]
        ),
        "cross_feature_truth_only": all(
            case["cause_feature"] != case["effect_feature"]
            for case in truth["cases"]
        ),
        "truth_directions_are_balanced": min(direction_counts.values())
        / max(sum(direction_counts.values()), 1)
        >= (0.30 if profile_name == "quick" else 0.35),
        "contextual_transfer_improves_coverage": transfer_evaluation["coverage"]
        > exact_evaluation["coverage"],
        "contextual_transfer_covers_at_least_half": transfer_evaluation["coverage"]
        >= 0.50,
        "contextual_transfer_beats_majority_utility": transfer_evaluation[
            "safe_direction_utility"
        ]
        > majority_safe_utility,
        "contextual_transfer_is_selectively_accurate": transfer_evaluation[
            "selective_accuracy"
        ]
        >= 0.80,
        "both_directions_transfer_accurately": all(
            transfer_by_direction[str(direction)]["coverage"] >= 0.30
            and transfer_by_direction[str(direction)]["selective_accuracy"] >= 0.80
            for direction in (-1, 1)
        ),
        "false_assertions_below_ten_percent": transfer_evaluation[
            "false_assertion_rate"
        ]
        <= 0.10,
        "paraphrases_are_inference_invariant": transfer_evaluation[
            "paraphrase_consistency"
        ]
        == 1.0,
        "adversarial_language_fails_closed": adversarial["passed"],
        "derived_transfers_retain_sources": all(
            turn["artifact"]["claim_status"] == "unknown"
            or (
                turn["artifact"]["derivation_kind"] == "contextual_transfer"
                and turn["artifact"]["source_count"] >= 3
                and len(turn["artifact"]["source_law_ids"]) >= 3
                and len(turn["artifact"]["source_condition_signatures"]) >= 3
                and isinstance(
                    turn["artifact"]["context_factor_trace"], Mapping
                )
                and isinstance(
                    turn["artifact"]["context_factor_projection"], Mapping
                )
            )
            for turn in transfer_response["turns"]
        ),
    }
    samples = []
    truth_by_id = {str(case["case_id"]): case for case in truth["cases"]}
    for turn in transfer_response["turns"][:8]:
        case = truth_by_id[str(turn["case_id"])]
        samples.append(
            {
                "case_id": turn["case_id"],
                "question": turn["request"]["text"],
                "expected_direction": case["expected_direction"],
                "claim_status": turn["artifact"]["claim_status"],
                "predicted_direction": turn["artifact"]["direction"],
                "derivation_kind": turn["artifact"]["derivation_kind"],
                "source_count": turn["artifact"]["source_count"],
                "pair_motif_count": (
                    turn["artifact"]["context_factor_trace"]["pair_motif_count"]
                    if isinstance(
                        turn["artifact"]["context_factor_trace"], Mapping
                    )
                    else 0
                ),
                "answer": turn["answer"],
            }
        )
    report_core = {
        "schema": CAUSAL_TRANSFER_SCHEMA,
        "runtime": CAUSAL_TRANSFER_RUNTIME,
        "profile": profile_name,
        "model_hash": model_payload["model_hash"],
        "training_program_count": len(training),
        "validation_program_count": len(validation),
        "heldout_program_count": len(heldout),
        "validation_case_count": len(validation_truth["cases"]),
        "case_count": len(truth["cases"]),
        "turn_count": len(request["turns"]),
        "training_program_ids": sorted(training),
        "validation_program_ids": sorted(validation),
        "heldout_program_ids": sorted(heldout),
        "validation_truth_hash": validation_truth["truth_hash"],
        "transfer_policy_hash": transfer_policy["policy_hash"],
        "truth_hash": truth["truth_hash"],
        "request_hash": request["request_hash"],
        "exact_response_hash": exact_response["response_hash"],
        "transfer_response_hash": transfer_response["response_hash"],
        "exact_baseline": exact_evaluation,
        "truth_direction_counts": direction_counts,
        "majority_direction_baseline": {
            "direction": majority_direction,
            "accuracy": majority_accuracy,
            "safe_direction_utility": majority_safe_utility,
        },
        "contextual_transfer": transfer_evaluation,
        "metaplastic_calibration": {
            "runtime": transfer_policy["runtime"],
            "selected_policy": transfer_policy["selected_policy"],
            "default_validation_evaluation": transfer_policy[
                "default_validation_evaluation"
            ],
            "selected_validation_evaluation": transfer_policy[
                "selected_validation_evaluation"
            ],
            "risk_contract": transfer_policy["risk_contract"],
            "probe_response_hashes": transfer_policy["probe_response_hashes"],
            "evaluated_policy_count": transfer_policy[
                "evaluated_policy_count"
            ],
            "eligible_policy_count": transfer_policy["eligible_policy_count"],
            "gates": transfer_policy["gates"],
            "passed": transfer_policy["passed"],
        },
        "adversarial_language": adversarial,
        "gates": gates,
        "passed": all(gates.values()),
        "samples": samples,
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    return {
        "validation_truth": validation_truth,
        "transfer_policy": transfer_policy,
        "truth": truth,
        "request": request,
        "exact_response": exact_response,
        "transfer_response": transfer_response,
        "report": report,
    }


def causal_transfer_self_test() -> dict[str, bool]:
    training = tuple(
        program_id
        for shard in range(16)
        for program_id in curriculum_program_ids(shard)
    )
    validation = select_heldout_program_ids(
        training,
        count=int(TRANSFER_PROFILES["massive"]["validation_programs"]),
        seed=int(TRANSFER_PROFILE_SEEDS["massive"]["validation"]),
    )
    heldout = select_heldout_program_ids(
        (*training, *validation),
        count=int(TRANSFER_PROFILES["massive"]["evaluation_programs"]),
        seed=int(TRANSFER_PROFILE_SEEDS["massive"]["evaluation"]),
    )
    decoded = [decode_world_program(value) for value in heldout]
    return {
        "validation_programs_are_disjoint": not set(training).intersection(
            validation
        ),
        "evaluation_programs_exclude_validation": not set(validation).intersection(
            heldout
        ),
        "heldout_programs_are_unique": len(heldout) == len(set(heldout)),
        "heldout_programs_are_disjoint": not set(training).intersection(heldout),
        "heldout_programs_are_valid": all(
            program.program_id == value
            for program, value in zip(decoded, heldout, strict=True)
        ),
        "heldout_axes_are_broad": all(
            len({getattr(program, axis) for program in decoded}) == len(values)
            for axis, values in WORLD_PROGRAM_AXES
            if len(values) <= len(decoded)
        ),
    }
