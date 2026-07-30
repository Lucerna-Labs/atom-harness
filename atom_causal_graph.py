"""Pure graph-native causal cognition for the Atom causal world.

The persistent graph is the authority.  Phase dynamics, recognition,
persistence, annealing, and measurement operate on a retrieved working
subgraph; none of them owns an independent answer-generating memory.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from atom_causal_world_curriculum import WORLD_PROGRAM_AXES
from atom_causal_world_schema import (
    ARCHITECTURE_COMPONENTS,
    DOMAIN_NAMES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    ROOT_MECHANICS,
    CausalEvidence,
    Intervention,
    canonical_hash,
)


CAUSAL_GRAPH_MODEL_SCHEMA = 2
CAUSAL_GRAPH_RUNTIME = "atom-executable-causal-graph-v3"
PHASE_LOCKED_LOOP_RUNTIME = "atom-causal-phase-locked-loop-v1"
PHASE_MIXER_RUNTIME = "atom-causal-phase-mixer-v1"
MOLECULAR_RECOGNITION_RUNTIME = "atom-causal-molecular-recognition-v3"
TOPOLOGICAL_PERSISTENCE_RUNTIME = "atom-causal-topological-persistence-v1"
THERMAL_ANNEALING_RUNTIME = "atom-causal-thermal-annealing-v1"
PROJECTIVE_MEASUREMENT_RUNTIME = "atom-causal-projective-measurement-v5"
CONTEXT_FACTOR_GRAPH_RUNTIME = "atom-causal-context-factor-graph-v2"
ACTIVE_EXPERIMENT_RUNTIME = "atom-causal-active-experiment-v1"

_WORLD_CONDITION_WEIGHTS = {
    **{axis: 1.0 for axis, _ in WORLD_PROGRAM_AXES},
    "primary_root": 1.5,
    "secondary_root": 1.5,
}
_WORLD_CONDITION_CARDINALITY = {
    **{axis: len(values) for axis, values in WORLD_PROGRAM_AXES},
    "primary_root": 7,
    "secondary_root": 7,
}
DEFAULT_CONTEXTUAL_TRANSFER_POLICY = {
    "direction_prior_power": 0.60,
    "token_likelihood_power": 0.75,
    "pair_motif_power": 0.0,
    "consensus_thresholds": {"-1": 0.99, "1": 0.99},
}
_PROJECTED_DECIMAL_QUANTUM = Decimal("0.000000000001")


def _stable_projected_float(value: float) -> float:
    """Remove platform-only tail bits from persisted inference diagnostics."""

    return round(float(value), 12)


def _stable_projected_decimal(value: Decimal) -> float:
    """Project deterministic decimal cognition into the persisted float schema."""

    return float(
        value.quantize(
            _PROJECTED_DECIMAL_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
    )


def validated_contextual_transfer_policy(
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the small metaplastic control surface used during projection."""

    selected = (
        DEFAULT_CONTEXTUAL_TRANSFER_POLICY
        if policy is None
        else dict(policy)
    )
    if set(selected) != {
        "consensus_thresholds",
        "direction_prior_power",
        "pair_motif_power",
        "token_likelihood_power",
    }:
        raise ValueError("contextual transfer policy fields are invalid")
    thresholds = selected["consensus_thresholds"]
    if not isinstance(thresholds, Mapping) or set(thresholds) != {"-1", "1"}:
        raise ValueError("contextual transfer thresholds are invalid")
    prior_power = float(selected["direction_prior_power"])
    token_power = float(selected["token_likelihood_power"])
    pair_power = float(selected["pair_motif_power"])
    rendered_thresholds = {
        direction: float(thresholds[direction]) for direction in ("-1", "1")
    }
    if not 0.0 <= prior_power <= 1.5:
        raise ValueError("direction prior power must be within [0, 1.5]")
    if not 0.25 <= token_power <= 1.5:
        raise ValueError("token likelihood power must be within [0.25, 1.5]")
    if not 0.0 <= pair_power <= 0.5:
        raise ValueError("pair motif power must be within [0, 0.5]")
    if not all(
        0.50 <= threshold <= 0.999999
        for threshold in rendered_thresholds.values()
    ):
        raise ValueError("transfer consensus thresholds must be within [0.5, 0.999999]")
    return {
        "direction_prior_power": _stable_projected_float(prior_power),
        "token_likelihood_power": _stable_projected_float(token_power),
        "pair_motif_power": _stable_projected_float(pair_power),
        "consensus_thresholds": {
            direction: _stable_projected_float(rendered_thresholds[direction])
            for direction in ("-1", "1")
        },
    }


def project_context_factor_trace(
    trace: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project a direction from policy-neutral singleton and pairwise factors."""

    if set(trace) != {
        "directions",
        "forced_direction",
        "pair_motif_count",
        "runtime",
    }:
        raise ValueError("context factor trace fields are invalid")
    if trace["runtime"] != CONTEXT_FACTOR_GRAPH_RUNTIME:
        raise ValueError("context factor trace runtime mismatch")
    pair_motif_count = trace["pair_motif_count"]
    if (
        isinstance(pair_motif_count, bool)
        or not isinstance(pair_motif_count, int)
        or pair_motif_count < 0
    ):
        raise ValueError("context factor motif count is invalid")
    directions = trace["directions"]
    if not isinstance(directions, Mapping) or set(directions) != {"-1", "1"}:
        raise ValueError("context factor directions are invalid")
    expected_fields = {
        "direction_prior_log_probability",
        "mean_confidence",
        "mean_persistence",
        "pair_motif_log_likelihood",
        "singleton_log_likelihood",
        "source_count",
        "source_diversity",
    }
    rendered: dict[int, dict[str, float | int]] = {}
    for direction in (-1, 1):
        values = directions[str(direction)]
        if not isinstance(values, Mapping) or set(values) != expected_fields:
            raise ValueError("context factor direction fields are invalid")
        source_count = values["source_count"]
        source_diversity = values["source_diversity"]
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count < 0
            or isinstance(source_diversity, bool)
            or not isinstance(source_diversity, int)
            or not 0 <= source_diversity <= source_count
        ):
            raise ValueError("context factor source counts are invalid")
        numeric = {
            name: float(values[name])
            for name in expected_fields
            if name not in {"source_count", "source_diversity"}
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("context factor values must be finite")
        if not (
            0.0 <= numeric["mean_confidence"] <= 1.0
            and 0.0 <= numeric["mean_persistence"] <= 1.0
        ):
            raise ValueError("context factor confidence values are invalid")
        rendered[direction] = {
            **numeric,
            "source_count": source_count,
            "source_diversity": source_diversity,
        }

    controls = validated_contextual_transfer_policy(policy)
    singleton_scores = {
        direction: (
            controls["direction_prior_power"]
            * float(rendered[direction]["direction_prior_log_probability"])
            + controls["token_likelihood_power"]
            * float(rendered[direction]["singleton_log_likelihood"])
        )
        for direction in (-1, 1)
    }
    factor_scores = {
        direction: (
            singleton_scores[direction]
            + controls["pair_motif_power"]
            * float(rendered[direction]["pair_motif_log_likelihood"])
        )
        for direction in (-1, 1)
    }
    forced_direction = trace["forced_direction"]
    if forced_direction is not None and forced_direction not in {-1, 1}:
        raise ValueError("context factor forced direction is invalid")
    singleton_direction = max(singleton_scores, key=singleton_scores.get)
    if forced_direction is not None:
        winning_direction = int(forced_direction)
        consensus = 1.0
    else:
        winning_direction = max(factor_scores, key=factor_scores.get)
        losing_direction = -winning_direction
        with localcontext() as decimal_context:
            decimal_context.prec = 80
            score_gap = Decimal(str(factor_scores[winning_direction])) - Decimal(
                str(factor_scores[losing_direction])
            )
            consensus = _stable_projected_decimal(
                Decimal(1) / (Decimal(1) + (-score_gap).exp())
            )
    winner = rendered[winning_direction]
    structural_support = (
        int(winner["source_count"]) >= 3
        and int(winner["source_diversity"]) >= 3
        and float(winner["mean_confidence"]) >= 0.48
        and float(winner["mean_persistence"]) >= 0.56
    )
    return {
        "candidate_direction": winning_direction,
        "consensus": _stable_projected_float(consensus),
        "factor_agreement": singleton_direction == winning_direction,
        "factor_scores": {
            str(direction): _stable_projected_float(factor_scores[direction])
            for direction in (-1, 1)
        },
        "singleton_direction": singleton_direction,
        "structural_support": structural_support,
    }


@dataclass(frozen=True)
class CausalNode:
    node_id: str
    kind: str
    label: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CausalLaw:
    law_id: str
    domain: str
    cause_feature: str
    effect_feature: str
    direction: int
    magnitude_mean: float
    magnitude_m2: float
    effect_variance_mean: float
    delay_mean: float
    invariant_error_mean: float
    support: int
    contradictions: int
    treated_worlds: int
    baseline_worlds: int
    confidence: float
    persistence: float
    phase: float
    status: str
    atom_program: tuple[str, ...]
    source_law_ids: tuple[str, ...]
    contexts: dict[str, int] = field(default_factory=dict)
    provenance_hashes: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def sample_variance(self) -> float:
        return self.magnitude_m2 / max(self.support - 1, 1)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["atom_program"] = list(self.atom_program)
        payload["source_law_ids"] = list(self.source_law_ids)
        payload["sample_variance"] = self.sample_variance
        return payload


@dataclass(frozen=True)
class CausalQuery:
    query_id: str
    domain: str | None
    cause_feature: str
    effect_feature: str | None
    context_signature: tuple[str, ...]
    mode: str = "effect"

    def validate(self) -> None:
        if not self.query_id:
            raise ValueError("query ID cannot be empty")
        if self.domain is not None and self.domain not in DOMAIN_NAMES:
            raise ValueError("query domain is unknown")
        if self.cause_feature not in FEATURE_INDEX:
            raise ValueError("query cause feature is unknown")
        if self.effect_feature is not None and self.effect_feature not in FEATURE_INDEX:
            raise ValueError("query effect feature is unknown")
        if self.mode not in {"effect", "why", "counterfactual"}:
            raise ValueError("unsupported causal query mode")


def _node_id(kind: str, label: str) -> str:
    return f"{kind}:{label}"


def _stable_condition_signature(context_signature: Sequence[str]) -> tuple[str, ...]:
    transient_prefixes = (
        "backend:",
        "cause:",
        "domain:",
        "mode:",
        "polarity:",
        "target:",
        "world_tick:",
    )
    stable = {
        value for value in context_signature if not value.startswith(transient_prefixes)
    }
    return tuple(sorted(stable)) or ("condition:general",)


def stable_condition_signature(
    context_signature: Sequence[str],
) -> tuple[str, ...]:
    """Return only persistent world conditions from evidence or query context."""

    return _stable_condition_signature(context_signature)


def law_condition_signature(law: CausalLaw) -> tuple[str, ...]:
    return _stable_condition_signature(tuple(law.contexts))


def contextual_condition_similarity(
    requested: Sequence[str], candidate: Sequence[str]
) -> float:
    """Score a novel regime against one observed regime without claiming identity."""

    requested_map = {
        value.split(":", 1)[0]: value
        for value in _stable_condition_signature(requested)
        if value != "condition:general"
    }
    candidate_map = {
        value.split(":", 1)[0]: value
        for value in _stable_condition_signature(candidate)
        if value != "condition:general"
    }
    total = sum(_WORLD_CONDITION_WEIGHTS.get(key, 0.0) for key in requested_map)
    if total <= 0.0:
        return 0.0
    matched = sum(
        _WORLD_CONDITION_WEIGHTS.get(key, 0.0)
        for key, value in requested_map.items()
        if candidate_map.get(key) == value
    )
    return float(matched / total)


def _merge_stable_conditions(
    left: Sequence[str], right: Sequence[str]
) -> tuple[str, ...] | None:
    merged: dict[str, str] = {}
    for value in (*left, *right):
        if value == "condition:general":
            continue
        key = value.split(":", 1)[0]
        previous = merged.get(key)
        if previous is not None and previous != value:
            return None
        merged[key] = value
    return tuple(sorted(merged.values())) or ("condition:general",)


def _law_key(
    domain: str,
    cause_feature: str,
    effect_feature: str,
    condition_signature: Sequence[str],
) -> str:
    return canonical_hash(
        {
            "domain": domain,
            "cause_feature": cause_feature,
            "effect_feature": effect_feature,
            "condition_signature": list(condition_signature),
        }
    )[:24]


def _compose_causal_path(path: Sequence[CausalLaw]) -> CausalLaw:
    if len(path) < 2:
        raise ValueError("a composed causal path requires at least two laws")
    if any(
        left.effect_feature != right.cause_feature
        for left, right in zip(path, path[1:])
    ):
        raise ValueError("causal path laws are not connected")
    if len({law.domain for law in path}) != 1:
        raise ValueError("causal path cannot cross world domains")
    source_law_ids = tuple(
        source_id for law in path for source_id in (law.source_law_ids or (law.law_id,))
    )
    path_id = f"path:{canonical_hash({'source_law_ids': source_law_ids})[:24]}"
    contexts: dict[str, int] = {}
    for law in path:
        for context, count in law.contexts.items():
            contexts[context] = contexts.get(context, 0) + count
    magnitude = float(
        np.clip(math.prod(max(law.magnitude_mean, 1e-12) for law in path), 0.0, 1e6)
    )
    phase_vector = sum(
        complex(math.cos(law.phase), math.sin(law.phase)) for law in path
    )
    phase = math.atan2(phase_vector.imag, phase_vector.real)
    confidence = math.exp(
        sum(math.log(max(law.confidence, 1e-12)) for law in path) / len(path)
    ) * (0.94 ** (len(path) - 1))
    provenance_hashes = list(
        dict.fromkeys(value for law in path for value in law.provenance_hashes)
    )[-64:]
    evidence_ids = list(
        dict.fromkeys(value for law in path for value in law.evidence_ids)
    )[-64:]
    atom_program = tuple(
        dict.fromkeys(value for law in path for value in law.atom_program)
    )
    return CausalLaw(
        law_id=path_id,
        domain=path[0].domain,
        cause_feature=path[0].cause_feature,
        effect_feature=path[-1].effect_feature,
        direction=math.prod(law.direction for law in path),
        magnitude_mean=magnitude,
        magnitude_m2=sum(law.magnitude_m2 for law in path),
        effect_variance_mean=sum(law.effect_variance_mean for law in path),
        delay_mean=sum(law.delay_mean for law in path),
        invariant_error_mean=max(law.invariant_error_mean for law in path),
        support=min(law.support for law in path),
        contradictions=sum(law.contradictions for law in path),
        treated_worlds=min(law.treated_worlds for law in path),
        baseline_worlds=min(law.baseline_worlds for law in path),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        persistence=min(law.persistence for law in path),
        phase=phase,
        status=(
            "crystallized"
            if all(law.status == "crystallized" for law in path)
            else "hypothesis"
        ),
        atom_program=atom_program,
        source_law_ids=source_law_ids,
        contexts=contexts,
        provenance_hashes=provenance_hashes,
        evidence_ids=evidence_ids,
    )


def _infer_atom_program(evidence: CausalEvidence) -> tuple[str, ...]:
    program: list[str] = []
    effect = evidence.effect_feature
    if effect in {"signal", "belief", "language_alignment", "activation"}:
        program.append("radiation")
    if effect in {"temperature", "pressure", "energy", "uncertainty"}:
        program.append("dissipation" if evidence.direction < 0 else "radiation")
    if effect in {"mass", "position_x", "position_y", "position_z"}:
        program.append("gravitation")
    if effect in {"cohesion", "trust", "polarity", "value"}:
        program.append("attraction_repulsion")
    if effect in {"structure", "integrity", "memory_strength"}:
        program.append("nucleation")
    if evidence.invariant_error <= 0.10:
        program.append("conservation")
    if (
        effect in {"lifetime", "health", "existence", "resource"}
        and evidence.direction < 0
    ):
        program.append("decay")
    if not program:
        program.extend(("radiation", "attraction_repulsion"))
    return tuple(dict.fromkeys(program))


class CausalGraph:
    """Persistent executable hypergraph learned from interventional evidence."""

    def __init__(self, maximum_laws: int = 1_000_000) -> None:
        if isinstance(maximum_laws, bool) or not isinstance(maximum_laws, int):
            raise TypeError("maximum laws must be an integer")
        if maximum_laws <= 0:
            raise ValueError("maximum laws must be positive")
        self.maximum_laws = maximum_laws
        self.nodes: dict[str, CausalNode] = {}
        self.laws: dict[str, CausalLaw] = {}
        self._by_cause: dict[str, set[str]] = {}
        self._by_effect: dict[str, set[str]] = {}
        self.observation_count = 0
        self._initialize_ontology()

    def _initialize_ontology(self) -> None:
        for domain in DOMAIN_NAMES:
            self._add_node("domain", domain, {"persistent": True})
        for feature in FEATURE_NAMES:
            self._add_node("feature", feature, {"persistent": True})
        for mechanic in ROOT_MECHANICS:
            self._add_node("root_mechanic", mechanic, {"persistent": True})
        for component in ARCHITECTURE_COMPONENTS:
            self._add_node("architecture", component, {"persistent": True})

    def _add_node(
        self, kind: str, label: str, attributes: Mapping[str, Any] | None = None
    ) -> str:
        node_id = _node_id(kind, label)
        incoming = CausalNode(
            node_id=node_id,
            kind=kind,
            label=label,
            attributes=dict(attributes or {}),
        )
        existing = self.nodes.get(node_id)
        if existing is not None and existing != incoming:
            raise ValueError(f"causal node collision: {node_id}")
        self.nodes[node_id] = incoming
        return node_id

    def observe(self, evidence: CausalEvidence) -> CausalLaw:
        evidence.validate()
        key = _law_key(
            evidence.domain,
            evidence.cause_feature,
            evidence.effect_feature,
            _stable_condition_signature(evidence.context_signature),
        )
        law_id = f"law:{key}"
        law = self.laws.get(law_id)
        positive_evidence = evidence.magnitude >= 1e-6
        if law is None:
            if len(self.laws) >= self.maximum_laws:
                self._evict_weakest_law()
            law = CausalLaw(
                law_id=law_id,
                domain=evidence.domain,
                cause_feature=evidence.cause_feature,
                effect_feature=evidence.effect_feature,
                direction=evidence.direction,
                magnitude_mean=evidence.magnitude,
                magnitude_m2=0.0,
                effect_variance_mean=evidence.variance,
                delay_mean=float(evidence.delay),
                invariant_error_mean=evidence.invariant_error,
                support=1 if positive_evidence else 0,
                contradictions=0 if positive_evidence else 1,
                treated_worlds=evidence.treated_worlds,
                baseline_worlds=evidence.baseline_worlds,
                confidence=0.0,
                persistence=0.0,
                phase=0.0,
                status="hypothesis",
                atom_program=_infer_atom_program(evidence),
                source_law_ids=(law_id,),
            )
            self.laws[law_id] = law
            self._by_cause.setdefault(evidence.cause_feature, set()).add(law_id)
            self._by_effect.setdefault(evidence.effect_feature, set()).add(law_id)
        else:
            agrees = positive_evidence and evidence.direction == law.direction
            if agrees:
                previous_support = law.support
                law.support += 1
                delta = evidence.magnitude - law.magnitude_mean
                law.magnitude_mean += delta / law.support
                law.magnitude_m2 += delta * (evidence.magnitude - law.magnitude_mean)
                law.effect_variance_mean = (
                    law.effect_variance_mean * previous_support + evidence.variance
                ) / law.support
                law.delay_mean = (
                    law.delay_mean * previous_support + evidence.delay
                ) / law.support
                law.invariant_error_mean = (
                    law.invariant_error_mean * previous_support
                    + evidence.invariant_error
                ) / law.support
            else:
                law.contradictions += 1
        law.treated_worlds += evidence.treated_worlds
        law.baseline_worlds += evidence.baseline_worlds
        for context in evidence.context_signature:
            law.contexts[context] = law.contexts.get(context, 0) + 1
        if evidence.provenance_hash not in law.provenance_hashes:
            law.provenance_hashes.append(evidence.provenance_hash)
            del law.provenance_hashes[:-64]
        if evidence.evidence_id not in law.evidence_ids:
            law.evidence_ids.append(evidence.evidence_id)
            del law.evidence_ids[:-64]
        self.observation_count += 1
        self._refresh_law(law)
        self._add_node("domain", evidence.domain, {"persistent": True})
        self._add_node("feature", evidence.cause_feature, {"persistent": True})
        self._add_node("feature", evidence.effect_feature, {"persistent": True})
        return law

    @staticmethod
    def _refresh_law(law: CausalLaw) -> None:
        total = law.support + law.contradictions
        evidence_ratio = (law.support + 1.0) / (total + 2.0)
        variance_penalty = 1.0 / (1.0 + law.effect_variance_mean + law.sample_variance)
        invariant_score = math.exp(-3.0 * law.invariant_error_mean)
        breadth = min(1.0, math.log2(1.0 + len(law.contexts)) / 4.0)
        law.confidence = float(
            np.clip(
                evidence_ratio
                * (0.55 + 0.45 * variance_penalty)
                * (0.65 + 0.35 * invariant_score)
                * (0.75 + 0.25 * breadth),
                0.0,
                1.0,
            )
        )
        law.phase = math.atan2(
            law.direction * law.magnitude_mean,
            1.0 + law.effect_variance_mean,
        )

    def _evict_weakest_law(self) -> None:
        if not self.laws:
            raise RuntimeError("cannot evict from an empty causal graph")
        victim_id = min(
            self.laws,
            key=lambda law_id: (
                self.laws[law_id].status == "crystallized",
                self.laws[law_id].confidence * self.laws[law_id].persistence,
                self.laws[law_id].support,
                law_id,
            ),
        )
        victim = self.laws.pop(victim_id)
        self._by_cause.get(victim.cause_feature, set()).discard(victim_id)
        self._by_effect.get(victim.effect_feature, set()).discard(victim_id)

    def candidate_laws(
        self,
        *,
        cause_feature: str,
        effect_feature: str | None = None,
        domain: str | None = None,
        condition_signature: Sequence[str] | None = None,
    ) -> tuple[CausalLaw, ...]:
        if cause_feature not in FEATURE_INDEX:
            raise ValueError("unknown causal feature")
        laws = [self.laws[law_id] for law_id in self._by_cause.get(cause_feature, ())]
        if effect_feature is not None:
            laws = [law for law in laws if law.effect_feature == effect_feature]
        if domain is not None:
            laws = [law for law in laws if law.domain == domain]
        if condition_signature is not None:
            requested = _stable_condition_signature(condition_signature)
            if requested == ("condition:general",):
                laws = [
                    law
                    for law in laws
                    if law_condition_signature(law) == ("condition:general",)
                ]
            else:
                requested_set = set(requested)
                laws = [
                    law
                    for law in laws
                    if requested_set.issubset(set(law_condition_signature(law)))
                ]
        return tuple(sorted(laws, key=lambda law: law.law_id))

    def composed_paths(
        self,
        *,
        cause_feature: str,
        effect_feature: str,
        domain: str | None = None,
        max_depth: int = 4,
        limit: int = 64,
        condition_signature: Sequence[str] | None = None,
    ) -> tuple[CausalLaw, ...]:
        """Build transient, provenance-preserving multi-hop causal laws."""

        if cause_feature not in FEATURE_INDEX or effect_feature not in FEATURE_INDEX:
            raise ValueError("unknown causal path feature")
        if not 2 <= max_depth <= 6:
            raise ValueError("causal path depth must be within [2, 6]")
        if not 1 <= limit <= 512:
            raise ValueError("causal path limit must be within [1, 512]")
        paths: list[CausalLaw] = []
        initial_conditions = (
            _stable_condition_signature(condition_signature)
            if condition_signature is not None
            else ("condition:general",)
        )
        frontier: list[
            tuple[
                str, tuple[CausalLaw, ...], frozenset[str], tuple[str, ...], str | None
            ]
        ] = [
            (
                cause_feature,
                (),
                frozenset({cause_feature}),
                initial_conditions,
                domain,
            )
        ]
        exploration_budget = 4096
        explored = 0
        while frontier and explored < exploration_budget:
            current, path, visited, conditions, path_domain = frontier.pop()
            outgoing = list(
                self.candidate_laws(
                    cause_feature=current,
                    domain=path_domain,
                    condition_signature=conditions,
                )
            )
            outgoing.sort(
                key=lambda law: (
                    law.status != "crystallized",
                    -law.persistence,
                    -law.confidence,
                    -law.support,
                    law.law_id,
                )
            )
            for law in outgoing[:12]:
                explored += 1
                if explored > exploration_budget or law.status == "retired":
                    break
                if law.effect_feature in visited:
                    continue
                law_conditions = _stable_condition_signature(tuple(law.contexts))
                merged_conditions = _merge_stable_conditions(conditions, law_conditions)
                if merged_conditions is None:
                    continue
                next_domain = path_domain or law.domain
                if law.domain != next_domain:
                    continue
                next_path = (*path, law)
                if law.effect_feature == effect_feature and len(next_path) >= 2:
                    paths.append(_compose_causal_path(next_path))
                    if len(paths) >= limit * 4:
                        frontier.clear()
                        break
                elif len(next_path) < max_depth:
                    frontier.append(
                        (
                            law.effect_feature,
                            next_path,
                            visited | {law.effect_feature},
                            merged_conditions,
                            next_domain,
                        )
                    )
        paths.sort(
            key=lambda law: (
                law.status != "crystallized",
                -law.persistence,
                -law.confidence,
                len(law.source_law_ids),
                law.law_id,
            )
        )
        unique: dict[str, CausalLaw] = {}
        for law in paths:
            unique.setdefault(law.law_id, law)
            if len(unique) >= limit:
                break
        return tuple(unique.values())

    def manifest(self) -> dict[str, Any]:
        return {
            "runtime": CAUSAL_GRAPH_RUNTIME,
            "maximum_laws": self.maximum_laws,
            "observation_count": self.observation_count,
            "node_count": len(self.nodes),
            "law_count": len(self.laws),
            "nodes": [asdict(self.nodes[key]) for key in sorted(self.nodes)],
            "laws": [self.laws[key].manifest() for key in sorted(self.laws)],
        }

    def model_payload(self) -> dict[str, Any]:
        core = {
            "schema": CAUSAL_GRAPH_MODEL_SCHEMA,
            "architecture": "pure-executable-causal-phase-hypergraph",
            "graph": self.manifest(),
            "runtimes": {
                "causal_graph": CAUSAL_GRAPH_RUNTIME,
                "phase_locked_loop": PHASE_LOCKED_LOOP_RUNTIME,
                "phase_mixer": PHASE_MIXER_RUNTIME,
                "molecular_recognition": MOLECULAR_RECOGNITION_RUNTIME,
                "topological_persistence": TOPOLOGICAL_PERSISTENCE_RUNTIME,
                "thermal_annealing": THERMAL_ANNEALING_RUNTIME,
                "projective_measurement": PROJECTIVE_MEASUREMENT_RUNTIME,
                "active_experiment": ACTIVE_EXPERIMENT_RUNTIME,
            },
        }
        return {**core, "model_hash": canonical_hash(core)}

    @classmethod
    def from_model_payload(cls, payload: Mapping[str, Any]) -> "CausalGraph":
        expected = {"architecture", "graph", "model_hash", "runtimes", "schema"}
        if set(payload) != expected:
            raise ValueError("causal graph model fields are invalid")
        core = {key: payload[key] for key in sorted(expected - {"model_hash"})}
        if payload["model_hash"] != canonical_hash(core):
            raise ValueError("causal graph model hash mismatch")
        if payload["schema"] != CAUSAL_GRAPH_MODEL_SCHEMA:
            raise ValueError("unsupported causal graph model schema")
        if payload["architecture"] != "pure-executable-causal-phase-hypergraph":
            raise ValueError("unsupported causal graph architecture")
        graph_payload = payload["graph"]
        if not isinstance(graph_payload, Mapping):
            raise ValueError("causal graph payload must be an object")
        graph = cls(maximum_laws=int(graph_payload["maximum_laws"]))
        graph.observation_count = int(graph_payload["observation_count"])
        graph.laws.clear()
        graph._by_cause.clear()
        graph._by_effect.clear()
        for law_payload in graph_payload["laws"]:
            values = dict(law_payload)
            values.pop("sample_variance", None)
            values["atom_program"] = tuple(values["atom_program"])
            values["source_law_ids"] = tuple(values["source_law_ids"])
            values["contexts"] = {
                str(key): int(value) for key, value in values["contexts"].items()
            }
            values["provenance_hashes"] = list(values["provenance_hashes"])
            values["evidence_ids"] = list(values["evidence_ids"])
            law = CausalLaw(**values)
            graph.laws[law.law_id] = law
            graph._by_cause.setdefault(law.cause_feature, set()).add(law.law_id)
            graph._by_effect.setdefault(law.effect_feature, set()).add(law.law_id)
        if len(graph.laws) != int(graph_payload["law_count"]):
            raise ValueError("causal graph law count mismatch")
        if graph.model_payload() != dict(payload):
            raise ValueError("causal graph payload is not an exact runtime round trip")
        return graph


class MolecularRecognition:
    runtime = MOLECULAR_RECOGNITION_RUNTIME

    def retrieve(
        self,
        graph: CausalGraph,
        query: CausalQuery,
        limit: int = 96,
        *,
        allow_contextual_transfer: bool = True,
    ) -> tuple[tuple[CausalLaw, float], ...]:
        query.validate()
        if not 1 <= limit <= 512:
            raise ValueError("recognition limit must be within [1, 512]")
        direct_candidates = graph.candidate_laws(
            cause_feature=query.cause_feature,
            effect_feature=query.effect_feature,
            domain=query.domain,
            condition_signature=query.context_signature,
        )
        candidates = list(direct_candidates)
        if query.effect_feature is not None:
            candidates.extend(
                graph.composed_paths(
                    cause_feature=query.cause_feature,
                    effect_feature=query.effect_feature,
                    domain=query.domain,
                    limit=limit,
                    condition_signature=query.context_signature,
                )
            )
        if (
            not candidates
            and allow_contextual_transfer
            and query.effect_feature is not None
            and stable_condition_signature(query.context_signature)
            != ("condition:general",)
        ):
            candidates.extend(
                law
                for law in graph.candidate_laws(
                    cause_feature=query.cause_feature,
                    effect_feature=query.effect_feature,
                    domain=query.domain,
                )
                if law.status == "crystallized"
            )
        candidates = list({law.law_id: law for law in candidates}.values())
        query_context = set(query.context_signature)
        scored: list[tuple[CausalLaw, float]] = []
        for law in candidates:
            law_context = set(law.contexts)
            overlap = len(query_context & law_context) / max(len(query_context), 1)
            domain_fit = 1.0 if query.domain in {None, law.domain} else 0.0
            evidence_mass = 1.0 - math.exp(-law.support / 4.0)
            condition_fit = contextual_condition_similarity(
                query.context_signature, law_condition_signature(law)
            )
            path_depth = max(1, len(law.source_law_ids))
            path_preference = (
                (1.12 if path_depth > 1 else 0.88)
                if query.mode == "why"
                else 0.96 ** (path_depth - 1)
            )
            score = path_preference * (
                0.22 * law.confidence
                + 0.18 * law.persistence
                + 0.10 * overlap
                + 0.10 * domain_fit
                + 0.10 * evidence_mass
                + 0.30 * condition_fit
            )
            scored.append((law, float(score)))
        scored.sort(key=lambda item: (-item[1], item[0].law_id))
        return tuple(scored[:limit])


class PhaseLockedLoop:
    runtime = PHASE_LOCKED_LOOP_RUNTIME

    def synchronize(
        self,
        candidates: Sequence[tuple[CausalLaw, float]],
        query: CausalQuery,
        ticks: int = 12,
    ) -> dict[str, float]:
        if not candidates:
            return {}
        if not 1 <= ticks <= 128:
            raise ValueError("phase ticks must be within [1, 128]")
        phases = np.asarray([law.phase for law, _ in candidates], dtype=np.float64)
        base_scores = np.asarray([score for _, score in candidates], dtype=np.float64)
        cue_hash = int(
            canonical_hash(
                {
                    "domain": query.domain,
                    "cause_feature": query.cause_feature,
                    "effect_feature": query.effect_feature,
                    "context_signature": list(
                        stable_condition_signature(query.context_signature)
                    ),
                    "mode": query.mode,
                }
            )[:12],
            16,
        )
        cue_phase = (cue_hash % 1_000_003) / 1_000_003.0 * 2.0 * np.pi - np.pi
        natural = np.asarray(
            [
                ((int(law.law_id.split(":", 1)[1][:8], 16) % 1024) / 1024.0 - 0.5)
                * 0.08
                for law, _ in candidates
            ]
        )
        for _ in range(ticks):
            pairwise = phases[None, :] - phases[:, None]
            coupling = (np.sin(pairwise) * base_scores[None, :]).sum(axis=1)
            coupling /= max(base_scores.sum(), 1e-8)
            cue_pull = np.sin(cue_phase - phases)
            phases += natural + 0.16 * coupling + 0.22 * cue_pull * base_scores
            phases = np.arctan2(np.sin(phases), np.cos(phases))
        coherence = 0.5 + 0.5 * np.cos(phases - cue_phase)
        return {
            law.law_id: float(value)
            for (law, _), value in zip(candidates, coherence, strict=True)
        }


class PhaseMixer:
    runtime = PHASE_MIXER_RUNTIME

    def mix(
        self,
        candidates: Sequence[tuple[CausalLaw, float]],
        coherence: Mapping[str, float],
    ) -> dict[str, float]:
        mixed: dict[str, float] = {}
        for law, recognition_score in candidates:
            phase_score = float(coherence.get(law.law_id, 0.0))
            causal_strength = math.tanh(4.0 * law.magnitude_mean)
            contradiction_penalty = law.contradictions / max(
                law.support + law.contradictions, 1
            )
            mixed[law.law_id] = float(
                np.clip(
                    0.42 * recognition_score
                    + 0.28 * phase_score
                    + 0.18 * causal_strength
                    + 0.12 * law.persistence
                    - 0.30 * contradiction_penalty,
                    -1.0,
                    1.0,
                )
            )
        return mixed


class TopologicalPersistence:
    runtime = TOPOLOGICAL_PERSISTENCE_RUNTIME

    def consolidate(self, graph: CausalGraph) -> dict[str, Any]:
        # Causal effects span orders of magnitude. Logarithmic filtration keeps
        # weak-but-repeatable structure visible without granting persistence to
        # evidence that lacks support, confidence, or contradiction survival.
        thresholds = np.geomspace(1e-6, 0.20, 48)
        crystallized = 0
        retired = 0
        for law in graph.laws.values():
            signal = law.magnitude_mean / (
                1.0 + math.sqrt(max(law.effect_variance_mean, 0.0))
            )
            threshold_survival = float((signal >= thresholds).mean())
            context_survival = min(1.0, len(law.contexts) / 6.0)
            evidence_survival = min(1.0, law.support / 4.0)
            contradiction_survival = 1.0 - law.contradictions / max(
                law.support + law.contradictions, 1
            )
            law.persistence = float(
                np.clip(
                    0.40 * threshold_survival
                    + 0.20 * context_survival
                    + 0.25 * evidence_survival
                    + 0.15 * contradiction_survival,
                    0.0,
                    1.0,
                )
            )
            if law.support >= 3 and law.persistence >= 0.56 and law.confidence >= 0.48:
                law.status = "crystallized"
                crystallized += 1
            elif law.contradictions > max(4, 2 * law.support):
                law.status = "retired"
                retired += 1
            else:
                law.status = "hypothesis"
        return {
            "runtime": self.runtime,
            "threshold_count": len(thresholds),
            "crystallized_laws": crystallized,
            "retired_laws": retired,
            "hypothesis_laws": len(graph.laws) - crystallized - retired,
        }


class ThermalAnnealing:
    runtime = THERMAL_ANNEALING_RUNTIME

    def settle(
        self,
        laws: Mapping[str, CausalLaw],
        mixed_scores: Mapping[str, float],
        temperatures: Sequence[float] = (1.0, 0.72, 0.48, 0.28, 0.12),
    ) -> dict[str, float]:
        if not mixed_scores:
            return {}
        law_ids = sorted(mixed_scores)
        values = np.asarray(
            [mixed_scores[law_id] for law_id in law_ids], dtype=np.float64
        )
        probabilities = np.full(len(law_ids), 1.0 / len(law_ids))
        for temperature in temperatures:
            if temperature <= 0.0:
                raise ValueError("annealing temperature must be positive")
            conflict = np.zeros_like(values)
            for index, law_id in enumerate(law_ids):
                law = laws[law_id]
                conflict[index] = sum(
                    probabilities[other_index]
                    for other_index, other_id in enumerate(law_ids)
                    if laws[other_id].effect_feature == law.effect_feature
                    and laws[other_id].direction != law.direction
                )
            logits = (values - 0.22 * conflict) / temperature
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
        return {
            law_id: float(probability)
            for law_id, probability in zip(law_ids, probabilities, strict=True)
        }


class ProjectiveMeasurement:
    runtime = PROJECTIVE_MEASUREMENT_RUNTIME

    def __init__(
        self, transfer_policy: Mapping[str, Any] | None = None
    ) -> None:
        self.transfer_policy = validated_contextual_transfer_policy(transfer_policy)

    @staticmethod
    def _build_context_factor_trace(
        eligible: Sequence[tuple[CausalLaw, float, float, float]],
        requested: Sequence[str],
    ) -> dict[str, Any]:
        """Build reusable direction factors without binding a projection policy."""

        direction_evidence = {
            direction: sum(
                weight
                for law, weight, _, _ in eligible
                if law.direction == direction
            )
            for direction in (-1, 1)
        }
        total_evidence = sum(direction_evidence.values())
        pair_motifs = tuple(combinations(requested, 2))
        directions: dict[str, dict[str, Any]] = {}
        with localcontext() as decimal_context:
            decimal_context.prec = 80
            alpha = Decimal("0.10")
            total_decimal = Decimal(str(total_evidence))
            for direction in (-1, 1):
                direction_total = direction_evidence[direction]
                direction_decimal = Decimal(str(direction_total))
                singleton_log_likelihood = Decimal(0)
                for token in requested:
                    axis = token.split(":", 1)[0]
                    cardinality = _WORLD_CONDITION_CARDINALITY[axis]
                    token_evidence = sum(
                        weight
                        for law, weight, _, _ in eligible
                        if law.direction == direction
                        and token in law_condition_signature(law)
                    )
                    singleton_log_likelihood += (
                        (Decimal(str(token_evidence)) + alpha)
                        / (
                            direction_decimal
                            + alpha * Decimal(cardinality)
                        )
                    ).ln()
                pair_motif_log_likelihood = Decimal(0)
                for left, right in pair_motifs:
                    left_axis = left.split(":", 1)[0]
                    right_axis = right.split(":", 1)[0]
                    pair_evidence = sum(
                        weight
                        for law, weight, _, _ in eligible
                        if law.direction == direction
                        and left in law_condition_signature(law)
                        and right in law_condition_signature(law)
                    )
                    pair_motif_log_likelihood += (
                        (Decimal(str(pair_evidence)) + alpha)
                        / (
                            direction_decimal
                            + alpha
                            * Decimal(
                                _WORLD_CONDITION_CARDINALITY[left_axis]
                                * _WORLD_CONDITION_CARDINALITY[right_axis]
                            )
                        )
                    ).ln()
                winners = [
                    item for item in eligible if item[0].direction == direction
                ]
                winner_mass = sum(weight for _, weight, _, _ in winners)
                source_diversity = len(
                    {
                        law_condition_signature(law)
                        for law, _, _, _ in winners
                    }
                )
                mean_confidence = sum(
                    weight * law.confidence for law, weight, _, _ in winners
                ) / max(winner_mass, 1e-12)
                mean_persistence = sum(
                    weight * law.persistence for law, weight, _, _ in winners
                ) / max(winner_mass, 1e-12)
                direction_prior = (
                    (direction_decimal + alpha)
                    / (total_decimal + Decimal(2) * alpha)
                ).ln()
                directions[str(direction)] = {
                    "direction_prior_log_probability": (
                        _stable_projected_decimal(direction_prior)
                    ),
                    "singleton_log_likelihood": _stable_projected_decimal(
                        singleton_log_likelihood
                    ),
                    "pair_motif_log_likelihood": _stable_projected_decimal(
                        pair_motif_log_likelihood
                    ),
                    "source_count": len(winners),
                    "source_diversity": source_diversity,
                    "mean_confidence": _stable_projected_float(mean_confidence),
                    "mean_persistence": _stable_projected_float(mean_persistence),
                }
        forced_direction = (
            max(direction_evidence, key=direction_evidence.get)
            if min(direction_evidence.values()) <= 0.0
            else None
        )
        return {
            "runtime": CONTEXT_FACTOR_GRAPH_RUNTIME,
            "pair_motif_count": len(pair_motifs),
            "forced_direction": forced_direction,
            "directions": directions,
        }

    def _measure_contextual_transfer(
        self,
        laws: Mapping[str, CausalLaw],
        query: CausalQuery,
        probabilities: Mapping[str, float],
        mixed_scores: Mapping[str, float],
    ) -> dict[str, Any] | None:
        requested = stable_condition_signature(query.context_signature)
        if requested == ("condition:general",):
            return None
        requested_set = set(requested)
        if any(
            requested_set.issubset(set(law_condition_signature(law)))
            for law in laws.values()
        ):
            return None

        eligible: list[tuple[CausalLaw, float, float, float]] = []
        for law_id, probability in probabilities.items():
            law = laws[law_id]
            similarity = contextual_condition_similarity(
                requested, law_condition_signature(law)
            )
            mixed = float(mixed_scores.get(law_id, 0.0))
            if law.status != "crystallized":
                continue
            evidence_weight = (
                law.confidence
                * law.persistence
                * math.log1p(law.support)
                * (0.98 + 0.02 * max(mixed, 0.0))
            )
            eligible.append((law, evidence_weight, similarity, mixed))

        transfer_method = "local_regime_consensus"
        # Three independent crystallized regimes are the minimum evidence
        # required by the factor projection's structural-support contract.
        # Start the factor path at that same boundary so every supported
        # contextual claim carries a reusable, auditable factor trace.
        axis_conditioned = len(eligible) >= 3
        context_factor_trace: dict[str, Any] | None = None
        factor_projection: dict[str, Any] | None = None
        if axis_conditioned:
            transfer_method = "context_factor_evidence_composition"
            context_factor_trace = self._build_context_factor_trace(
                eligible, requested
            )
            factor_projection = project_context_factor_trace(
                context_factor_trace, self.transfer_policy
            )
            winning_direction = int(factor_projection["candidate_direction"])
            consensus = float(factor_projection["consensus"])
            weighted = eligible
            direction_mass = {
                winning_direction: consensus,
                -winning_direction: 1.0 - consensus,
            }
        else:
            weighted = []
            for law, _, similarity, mixed in eligible:
                if similarity < 0.30 or mixed < 0.24:
                    continue
                probability = max(float(probabilities[law.law_id]), 1e-12)
                weight = (
                    probability
                    * (0.45 + similarity)
                    * law.confidence
                    * law.persistence
                    * (0.50 + max(mixed, 0.0))
                )
                weighted.append((law, weight, similarity, mixed))
            direction_mass = {
                direction: sum(
                    weight
                    for law, weight, _, _ in weighted
                    if law.direction == direction
                )
                for direction in (-1, 1)
            }
            total_mass = sum(direction_mass.values())
            winning_direction = max(direction_mass, key=direction_mass.get)
            consensus = direction_mass[winning_direction] / max(total_mass, 1e-12)

        total_mass = sum(direction_mass.values())
        winners = [
            item for item in weighted if item[0].direction == winning_direction
        ]
        winner_mass = sum(weight for _, weight, _, _ in winners)
        source_diversity = len(
            {law_condition_signature(law) for law, _, _, _ in winners}
        )
        maximum_similarity = max(
            (similarity for _, _, similarity, _ in winners), default=0.0
        )
        mean_similarity = sum(
            weight * similarity for _, weight, similarity, _ in winners
        ) / max(winner_mass, 1e-12)
        mean_confidence = sum(
            weight * law.confidence for law, weight, _, _ in winners
        ) / max(winner_mass, 1e-12)
        mean_persistence = sum(
            weight * law.persistence for law, weight, _, _ in winners
        ) / max(winner_mass, 1e-12)
        mean_mixed = sum(
            weight * mixed for _, weight, _, mixed in winners
        ) / max(winner_mass, 1e-12)
        consensus_threshold = float(
            self.transfer_policy["consensus_thresholds"][str(winning_direction)]
        )
        structural_support = (
            bool(factor_projection["structural_support"])
            if axis_conditioned and factor_projection is not None
            else (
                len(winners) >= 3
                and source_diversity >= 3
                and maximum_similarity >= 0.50
                and mean_similarity >= 0.34
                and mean_confidence >= 0.60
                and mean_persistence >= 0.56
                and mean_mixed >= 0.30
            )
        )
        supported = structural_support and (
            consensus >= consensus_threshold
            if axis_conditioned
            else consensus >= 0.76
        )
        if not supported:
            unknown = self._unknown(query, "insufficient_transfer_consensus")
            unknown.update(
                {
                    "derivation_kind": "contextual_transfer_rejected",
                    "transfer_method": transfer_method,
                    "candidate_direction": winning_direction,
                    "source_count": len(winners),
                    "transfer_structural_support": structural_support,
                    "transfer_threshold": _stable_projected_float(
                        consensus_threshold if axis_conditioned else 0.76
                    ),
                    "transfer_policy": dict(self.transfer_policy),
                    "context_factor_trace": context_factor_trace,
                    "context_factor_projection": factor_projection,
                    "context_similarity": _stable_projected_float(mean_similarity),
                    "maximum_context_similarity": _stable_projected_float(
                        maximum_similarity
                    ),
                    "transfer_consensus": _stable_projected_float(consensus),
                    "source_condition_signatures": [
                        list(law_condition_signature(law))
                        for law, _, _, _ in sorted(
                            winners, key=lambda item: (-item[1], item[0].law_id)
                        )[:8]
                    ],
                }
            )
            return unknown

        winners.sort(key=lambda item: (-item[1], item[0].law_id))
        selected = winners[:8]
        selected_law_ids = [law.law_id for law, _, _, _ in selected]
        expected_magnitude = sum(
            weight * law.magnitude_mean for law, weight, _, _ in winners
        ) / winner_mass
        expected_delay = sum(
            weight * law.delay_mean for law, weight, _, _ in winners
        ) / winner_mass
        atom_program = list(
            dict.fromkeys(
                atom for law, _, _, _ in selected for atom in law.atom_program
            )
        )
        provenance_hashes = list(
            dict.fromkeys(
                value
                for law, _, _, _ in selected
                for value in law.provenance_hashes
            )
        )[:64]
        transfer_id = "transfer:" + canonical_hash(
            {
                "domain": query.domain,
                "cause_feature": query.cause_feature,
                "effect_feature": query.effect_feature,
                "conditions": list(requested),
                "direction": winning_direction,
                "sources": selected_law_ids,
            }
        )[:24]
        direction_word = "increases" if winning_direction > 0 else "decreases"
        return {
            "claim_status": "derived",
            "assertion": (
                f"Changing {query.cause_feature} {direction_word} "
                f"{query.effect_feature} in the {query.domain} context."
            ),
            "law_id": transfer_id,
            "source_law_ids": selected_law_ids,
            "source_count": len(winners),
            "path_length": 1,
            "derivation_kind": "contextual_transfer",
            "transfer_method": transfer_method,
            "candidate_direction": winning_direction,
            "transfer_structural_support": structural_support,
            "transfer_threshold": _stable_projected_float(
                consensus_threshold if axis_conditioned else 0.76
            ),
            "transfer_policy": dict(self.transfer_policy),
            "context_factor_trace": context_factor_trace,
            "context_factor_projection": factor_projection,
            "domain": query.domain,
            "cause_feature": query.cause_feature,
            "effect_feature": query.effect_feature,
            "direction": winning_direction,
            "expected_magnitude": _stable_projected_float(expected_magnitude),
            "expected_delay": _stable_projected_float(expected_delay),
            "confidence": _stable_projected_float(mean_confidence * consensus),
            "persistence": _stable_projected_float(mean_persistence),
            "support": sum(law.support for law, _, _, _ in winners),
            "contradictions": sum(
                law.contradictions for law, _, _, _ in winners
            ),
            "probability": _stable_projected_float(consensus),
            "margin": _stable_projected_float(
                (
                    direction_mass[winning_direction]
                    - direction_mass[-winning_direction]
                )
                / max(total_mass, 1e-12)
            ),
            "atom_program": atom_program,
            "condition_signature": list(requested),
            "source_condition_signatures": [
                list(law_condition_signature(law)) for law, _, _, _ in selected
            ],
            "context_similarity": _stable_projected_float(mean_similarity),
            "maximum_context_similarity": _stable_projected_float(
                maximum_similarity
            ),
            "transfer_consensus": _stable_projected_float(consensus),
            "provenance_hashes": provenance_hashes,
            "reason": None,
        }

    def measure(
        self,
        laws: Mapping[str, CausalLaw],
        query: CausalQuery,
        probabilities: Mapping[str, float],
        mixed_scores: Mapping[str, float],
    ) -> dict[str, Any]:
        if not probabilities:
            return self._unknown(query, "no_matching_causal_law")
        transfer = self._measure_contextual_transfer(
            laws, query, probabilities, mixed_scores
        )
        if transfer is not None:
            return transfer
        ranked = sorted(
            probabilities, key=lambda law_id: (-probabilities[law_id], law_id)
        )
        best_id = ranked[0]
        best = laws[best_id]
        probability = probabilities[best_id]
        runner_up = probabilities[ranked[1]] if len(ranked) > 1 else 0.0
        margin = probability - runner_up
        supported = (
            best.status == "crystallized"
            and best.support >= 3
            and best.confidence >= 0.48
            and best.persistence >= 0.56
            and mixed_scores.get(best_id, 0.0) >= 0.38
            and (probability >= 0.34 or margin >= 0.12)
        )
        if not supported:
            return self._unknown(query, "insufficient_causal_support")
        direction_word = "increases" if best.direction > 0 else "decreases"
        assertion = (
            f"Changing {best.cause_feature} {direction_word} "
            f"{best.effect_feature} in the {best.domain} context."
        )
        return {
            "claim_status": "derived",
            "assertion": assertion,
            "law_id": best.law_id,
            "source_law_ids": list(best.source_law_ids),
            "source_count": len(best.source_law_ids),
            "path_length": len(best.source_law_ids),
            "derivation_kind": (
                "composed_path" if len(best.source_law_ids) > 1 else "observed_law"
            ),
            "transfer_method": None,
            "candidate_direction": None,
            "transfer_structural_support": None,
            "transfer_threshold": None,
            "transfer_policy": None,
            "context_factor_trace": None,
            "context_factor_projection": None,
            "domain": best.domain,
            "cause_feature": best.cause_feature,
            "effect_feature": best.effect_feature,
            "direction": best.direction,
            "expected_magnitude": best.magnitude_mean,
            "expected_delay": best.delay_mean,
            "confidence": best.confidence,
            "persistence": best.persistence,
            "support": best.support,
            "contradictions": best.contradictions,
            "probability": probability,
            "margin": margin,
            "atom_program": list(best.atom_program),
            "condition_signature": list(law_condition_signature(best)),
            "source_condition_signatures": [
                list(law_condition_signature(best))
            ],
            "context_similarity": contextual_condition_similarity(
                query.context_signature, law_condition_signature(best)
            ),
            "maximum_context_similarity": contextual_condition_similarity(
                query.context_signature, law_condition_signature(best)
            ),
            "transfer_consensus": None,
            "provenance_hashes": list(best.provenance_hashes),
            "reason": None,
        }

    @staticmethod
    def _unknown(query: CausalQuery, reason: str) -> dict[str, Any]:
        return {
            "claim_status": "unknown",
            "assertion": None,
            "law_id": None,
            "source_law_ids": [],
            "source_count": 0,
            "path_length": 0,
            "derivation_kind": "unknown",
            "transfer_method": None,
            "candidate_direction": None,
            "transfer_structural_support": None,
            "transfer_threshold": None,
            "transfer_policy": None,
            "context_factor_trace": None,
            "context_factor_projection": None,
            "domain": query.domain,
            "cause_feature": query.cause_feature,
            "effect_feature": query.effect_feature,
            "direction": None,
            "expected_magnitude": None,
            "expected_delay": None,
            "confidence": 0.0,
            "persistence": 0.0,
            "support": 0,
            "contradictions": 0,
            "probability": 0.0,
            "margin": 0.0,
            "atom_program": [],
            "condition_signature": list(
                _stable_condition_signature(query.context_signature)
            ),
            "source_condition_signatures": [],
            "context_similarity": 0.0,
            "maximum_context_similarity": 0.0,
            "transfer_consensus": 0.0,
            "provenance_hashes": [],
            "reason": reason,
        }


class ActiveExperimentScheduler:
    runtime = ACTIVE_EXPERIMENT_RUNTIME

    def rank(
        self,
        graph: CausalGraph,
        interventions: Sequence[Intervention],
    ) -> tuple[tuple[Intervention, float], ...]:
        ranked: list[tuple[Intervention, float]] = []
        for intervention in interventions:
            laws = graph.candidate_laws(cause_feature=intervention.feature)
            if not laws:
                uncertainty = 1.0
                coverage = 0.0
                contradiction = 0.0
            else:
                uncertainty = 1.0 - float(np.mean([law.confidence for law in laws]))
                domains = {law.domain for law in laws}
                coverage = len(domains) / len(DOMAIN_NAMES)
                contradiction = float(
                    np.mean(
                        [
                            law.contradictions
                            / max(law.support + law.contradictions, 1)
                            for law in laws
                        ]
                    )
                )
            polarity_novelty = 0.08 if intervention.polarity < 0 else 0.0
            score = (
                0.52 * uncertainty
                + 0.28 * (1.0 - coverage)
                + 0.20 * contradiction
                + polarity_novelty
            )
            ranked.append((intervention, float(score)))
        ranked.sort(key=lambda item: (-item[1], item[0].intervention_id))
        return tuple(ranked)


class CausalCognition:
    """Runtime orchestration around one authoritative causal graph."""

    def __init__(
        self,
        graph: CausalGraph,
        *,
        transfer_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self.recognition = MolecularRecognition()
        self.phase_loop = PhaseLockedLoop()
        self.phase_mixer = PhaseMixer()
        self.persistence = TopologicalPersistence()
        self.annealing = ThermalAnnealing()
        self.measurement = ProjectiveMeasurement(transfer_policy)
        self.scheduler = ActiveExperimentScheduler()

    def learn(self, evidence: Iterable[CausalEvidence]) -> dict[str, Any]:
        observed = 0
        for item in evidence:
            self.graph.observe(item)
            observed += 1
        consolidation = self.persistence.consolidate(self.graph)
        return {
            "observed_evidence": observed,
            "graph_laws": len(self.graph.laws),
            "graph_nodes": len(self.graph.nodes),
            "consolidation": consolidation,
        }

    def answer(
        self, query: CausalQuery, *, allow_contextual_transfer: bool = True
    ) -> dict[str, Any]:
        query.validate()
        candidates = self.recognition.retrieve(
            self.graph,
            query,
            allow_contextual_transfer=allow_contextual_transfer,
        )
        working_laws = {law.law_id: law for law, _ in candidates}
        coherence = self.phase_loop.synchronize(candidates, query)
        mixed = self.phase_mixer.mix(candidates, coherence)
        probabilities = self.annealing.settle(working_laws, mixed)
        artifact = self.measurement.measure(working_laws, query, probabilities, mixed)
        artifact["query_id"] = query.query_id
        artifact["execution_trace"] = [
            {
                "component": "causal_graph",
                "runtime": CAUSAL_GRAPH_RUNTIME,
                "candidate_laws": len(candidates),
                "contextual_transfer_allowed": allow_contextual_transfer,
            },
            {
                "component": "molecular_recognition",
                "runtime": MOLECULAR_RECOGNITION_RUNTIME,
                "retrieved": len(candidates),
                "composed_paths": sum(
                    len(law.source_law_ids) > 1 for law, _ in candidates
                ),
            },
            {
                "component": "phase_locked_loop",
                "runtime": PHASE_LOCKED_LOOP_RUNTIME,
                "synchronized": len(coherence),
            },
            {
                "component": "phase_mixer",
                "runtime": PHASE_MIXER_RUNTIME,
                "mixed": len(mixed),
            },
            {
                "component": "topological_persistence",
                "runtime": TOPOLOGICAL_PERSISTENCE_RUNTIME,
                "eligible": sum(law.status == "crystallized" for law, _ in candidates),
            },
            {
                "component": "thermal_annealing",
                "runtime": THERMAL_ANNEALING_RUNTIME,
                "settled": len(probabilities),
            },
            {
                "component": "projective_measurement",
                "runtime": PROJECTIVE_MEASUREMENT_RUNTIME,
                "context_factor_runtime": CONTEXT_FACTOR_GRAPH_RUNTIME,
                "claim_status": artifact["claim_status"],
                "derivation_kind": artifact["derivation_kind"],
            },
        ]
        artifact["evidence_path"] = (
            [
                *[
                    {
                        "kind": "retrieved",
                        "source": "causal_graph",
                        "law_id": source_law_id,
                        "path_position": position,
                    }
                    for position, source_law_id in enumerate(
                        artifact["source_law_ids"], start=1
                    )
                ],
                {
                    "kind": "derived",
                    "source": "phase_locked_counterfactual_settlement",
                    "atom_program": artifact["atom_program"],
                },
                {
                    "kind": "measured",
                    "source": "projective_measurement",
                    "confidence": artifact["confidence"],
                },
            ]
            if artifact["claim_status"] == "derived"
            else []
        )
        return artifact


def causal_graph_self_test() -> dict[str, bool]:
    graph = CausalGraph(maximum_laws=64)
    base = {
        "domain": "physical",
        "cause_feature": "temperature",
        "effect_feature": "energy",
        "direction": 1,
        "magnitude": 0.24,
        "delay": 2,
        "context_signature": ("domain:physical", "cause:temperature"),
        "treated_worlds": 8,
        "baseline_worlds": 8,
        "variance": 0.01,
        "invariant_error": 0.02,
    }
    for index in range(6):
        evidence = CausalEvidence(
            evidence_id=f"self-{index}",
            provenance_hash=canonical_hash({"self": index}),
            **base,
        )
        graph.observe(evidence)
    graph.observe(
        CausalEvidence(
            evidence_id="self-conditional-variant",
            provenance_hash=canonical_hash({"self": "conditional-variant"}),
            **{
                **base,
                "direction": -1,
                "context_signature": (
                    "domain:physical",
                    "cause:temperature",
                    "condition:scarce",
                ),
            },
        )
    )
    chain_base = {
        **base,
        "cause_feature": "energy",
        "effect_feature": "structure",
        "magnitude": 0.62,
        "delay": 3,
        "context_signature": ("domain:physical", "cause:energy"),
    }
    for index in range(6):
        graph.observe(
            CausalEvidence(
                evidence_id=f"self-chain-{index}",
                provenance_hash=canonical_hash({"self-chain": index}),
                **chain_base,
            )
        )
    cognition = CausalCognition(graph)
    cognition.persistence.consolidate(graph)
    query = CausalQuery(
        query_id="self-query",
        domain="physical",
        cause_feature="temperature",
        effect_feature="energy",
        context_signature=("domain:physical", "cause:temperature"),
    )
    answer = cognition.answer(query)
    composed_answer = cognition.answer(
        CausalQuery(
            query_id="composed-query",
            domain="physical",
            cause_feature="temperature",
            effect_feature="structure",
            context_signature=("domain:physical", "cause:temperature"),
        )
    )
    model = graph.model_payload()
    restored = CausalGraph.from_model_payload(model)
    unknown = cognition.answer(
        CausalQuery(
            query_id="unsupported",
            domain="language",
            cause_feature="ownership",
            effect_feature="mass",
            context_signature=("domain:language",),
        )
    )
    components = {entry["component"] for entry in answer["execution_trace"]}
    checks = {
        "law_crystallized": next(iter(graph.laws.values())).status == "crystallized",
        "supported_query_derived": answer["claim_status"] == "derived",
        "unsupported_query_abstains": unknown["claim_status"] == "unknown",
        "all_seven_components_executed": components == set(ARCHITECTURE_COMPONENTS),
        "model_roundtrip_exact": restored.model_payload() == model,
        "provenance_retained": bool(answer["provenance_hashes"]),
        "conditional_law_variants_are_distinct": len(
            graph.candidate_laws(
                cause_feature="temperature",
                effect_feature="energy",
            )
        )
        == 2,
        "multi_hop_path_is_derived": composed_answer["claim_status"] == "derived"
        and composed_answer["path_length"] == 2
        and len(composed_answer["source_law_ids"]) == 2,
    }
    if not all(checks.values()):
        raise AssertionError(f"causal graph self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(causal_graph_self_test())
