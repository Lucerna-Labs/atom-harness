"""Emergent transition-law field built on the seven universe primitives.

This runtime is deliberately separate from the fixed language predicate
inventory.  It induces entity lexemes and executable effect programs from
opaque utterances paired only with before/after worlds.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    AtomWikiGraph,
    build_language_graph,
    retrieve_atom_context,
)
from atom_transition_dataset import build_transition_discovery_program


TRANSITION_MODEL_SCHEMA = 1
TRANSITION_DISCOVERY_SEED = 7_071_343
ATOM_TRANSITION_RUNTIME = "atom-emergent-transition-law-v1"


class TransitionPrimitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


def transition_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transition_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        json.dumps([seed, *parts], sort_keys=True, default=str).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def transition_clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def transition_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        raise ValueError("text must contain 1 to 256 characters")
    tokens = tuple(re.findall(r"[a-z0-9]+", text.lower()))
    if not tokens:
        raise ValueError("text does not contain any surface pulses")
    return tokens


def transition_concept_kind(concept: str) -> str:
    if concept.startswith("agent-"):
        return "agent"
    if concept.startswith("object-"):
        return "object"
    if concept.startswith("location-"):
        return "location"
    raise ValueError(f"Unknown world concept: {concept}")


def validate_transition_world(world: Mapping[str, Any]) -> None:
    if set(world) != {"locations", "holders"}:
        raise ValueError("world must contain exactly locations and holders")
    if not isinstance(world["locations"], dict) or not isinstance(
        world["holders"], dict
    ):
        raise ValueError("world collections must be objects")
    agents = set(world["locations"])
    if not agents:
        raise ValueError("world requires at least one agent")
    for agent, location in world["locations"].items():
        if transition_concept_kind(str(agent)) != "agent":
            raise ValueError("location keys must be agent concepts")
        if transition_concept_kind(str(location)) != "location":
            raise ValueError("location values must be location concepts")
    if not world["holders"]:
        raise ValueError("world requires at least one object")
    for item, holder in world["holders"].items():
        if transition_concept_kind(str(item)) != "object":
            raise ValueError("holder keys must be object concepts")
        if holder is not None and holder not in agents:
            raise ValueError("holder values must be null or known agents")


def copy_transition_world(world: Mapping[str, Any]) -> dict[str, Any]:
    validate_transition_world(world)
    return {
        "locations": dict(sorted(world["locations"].items())),
        "holders": dict(sorted(world["holders"].items())),
    }


def validate_transition_row(row: Mapping[str, Any]) -> None:
    expected = {"case_id", "text", "before", "after", "salience"}
    if set(row) != expected:
        raise ValueError(f"transition row fields must be {sorted(expected)}")
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise ValueError("case_id must be non-empty text")
    transition_tokens(str(row["text"]))
    validate_transition_world(row["before"])
    validate_transition_world(row["after"])
    if set(row["before"]["locations"]) != set(row["after"]["locations"]):
        raise ValueError("before/after location keys must match")
    if set(row["before"]["holders"]) != set(row["after"]["holders"]):
        raise ValueError("before/after holder keys must match")
    salience = row["salience"]
    if (
        isinstance(salience, bool)
        or not isinstance(salience, (int, float))
        or not math.isfinite(float(salience))
        or not 0.1 <= float(salience) <= 2.0
    ):
        raise ValueError("salience must be finite and within [0.1, 2.0]")
    if row["before"] == row["after"]:
        raise ValueError("transition row must change the world")


def delta_participants(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return causal concepts exposed by a transition, without semantic names."""

    validate_transition_row(row)
    before = row["before"]
    after = row["after"]
    participants: set[str] = set()
    for collection in ("locations", "holders"):
        for key in before[collection]:
            old = before[collection][key]
            new = after[collection][key]
            if old == new:
                continue
            participants.add(str(key))
            for value in (old, new):
                if value is not None:
                    participants.add(str(value))
    return tuple(sorted(participants))


@dataclass(frozen=True)
class TransitionConfig:
    initial_temperature: float = 1.45
    temperature_floor: float = 0.22
    cooling_rate: float = 0.94
    phase_mix_strength: float = 0.052
    surface_min_hits: int = 2
    surface_min_support: float = 0.70
    surface_margin: float = 1.50
    law_min_hits: int = 2
    trace_dissipation: float = 0.09
    trace_expiration: float = 0.12
    information_mass_budget: float = 128.0
    anneal_trials: int = 20
    chaos_seed: int = TRANSITION_DISCOVERY_SEED

    def validate(self) -> None:
        for name, minimum, maximum in (
            ("surface_min_hits", 2, 100),
            ("law_min_hits", 2, 100),
            ("anneal_trials", 1, 2_000),
            ("chaos_seed", 0, 2**63 - 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
        for name in (
            "initial_temperature",
            "temperature_floor",
            "information_mass_budget",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_floor > self.initial_temperature:
            raise ValueError("temperature floor cannot exceed initial temperature")
        for name in (
            "cooling_rate",
            "phase_mix_strength",
            "surface_min_support",
            "trace_dissipation",
            "trace_expiration",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not 1.0 <= self.surface_margin <= 10.0:
            raise ValueError("surface_margin must be within [1, 10]")


@dataclass(frozen=True)
class AssociationObservation:
    surface: str
    concept: str
    position: int
    token_count: int
    weight: float = 1.0


@dataclass(frozen=True)
class AssociationEvidence:
    surface: str
    concept: str
    hits: int
    weight: float
    phase_real: float
    phase_imaginary: float

    @property
    def coherence(self) -> float:
        return transition_clamp(
            math.hypot(self.phase_real, self.phase_imaginary)
            / max(self.weight, 1e-12),
            0.0,
            1.0,
        )


@dataclass(frozen=True)
class SurfaceExposure:
    surface: str
    hits: int


@dataclass(frozen=True)
class EffectAtom:
    collection: str
    key_slot: int
    before: str
    after: str

    def payload(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "key_slot": self.key_slot,
            "before": self.before,
            "after": self.after,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        slot_types: Sequence[str],
    ) -> EffectAtom:
        if set(payload) != {"collection", "key_slot", "before", "after"}:
            raise ValueError("effect atom shape is invalid")
        atom = cls(
            collection=str(payload["collection"]),
            key_slot=int(payload["key_slot"]),
            before=str(payload["before"]),
            after=str(payload["after"]),
        )
        atom.validate(slot_types)
        return atom

    def validate(self, slot_types: Sequence[str]) -> None:
        if self.collection not in {"locations", "holders"}:
            raise ValueError("effect collection is invalid")
        if (
            isinstance(self.key_slot, bool)
            or not isinstance(self.key_slot, int)
            or not 0 <= self.key_slot < len(slot_types)
        ):
            raise ValueError("effect key slot is out of range")
        expected_key_type = "agent" if self.collection == "locations" else "object"
        if slot_types[self.key_slot] != expected_key_type:
            raise ValueError("effect key slot has an incompatible concept type")
        for name, expression in (("before", self.before), ("after", self.after)):
            if expression == "none":
                continue
            if expression.startswith("slot:"):
                index = _expression_index(expression, "slot")
                if index >= len(slot_types):
                    raise ValueError(f"{name} slot expression is out of range")
                continue
            if expression.startswith("any:") and name == "before":
                kind = expression.split(":", 1)[1]
                if kind not in {"agent", "object", "location"}:
                    raise ValueError("effect wildcard type is invalid")
                continue
            if expression.startswith("old:") and name == "after":
                pieces = expression.split(":")
                if len(pieces) != 3 or pieces[1] not in {"locations", "holders"}:
                    raise ValueError("old-cell expression is invalid")
                index = _strict_index(pieces[2], "old-cell slot")
                if index >= len(slot_types):
                    raise ValueError("old-cell slot is out of range")
                key_type = "agent" if pieces[1] == "locations" else "object"
                if slot_types[index] != key_type:
                    raise ValueError("old-cell slot has an incompatible type")
                continue
            raise ValueError(f"Unsupported {name} effect expression: {expression}")


@dataclass(frozen=True)
class TransitionObservation:
    signature: str
    pattern: tuple[str, ...]
    slot_types: tuple[str, ...]
    effects: tuple[EffectAtom, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class TransitionEvidence:
    signature: str
    pattern: tuple[str, ...]
    slot_types: tuple[str, ...]
    effects: tuple[EffectAtom, ...]
    hits: int
    weight: float


@dataclass(frozen=True)
class SurfaceLaw:
    surface: str
    concept: str
    evidence_count: int
    mass: float
    support: float
    coherence: float
    active: bool = True

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionLaw:
    law_id: str
    signature: str
    pattern: tuple[str, ...]
    slot_types: tuple[str, ...]
    effects: tuple[EffectAtom, ...]
    evidence_count: int
    mass: float
    support: float
    active: bool = True

    def payload(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "signature": self.signature,
            "pattern": list(self.pattern),
            "slot_types": list(self.slot_types),
            "effects": [effect.payload() for effect in self.effects],
            "evidence_count": self.evidence_count,
            "mass": self.mass,
            "support": self.support,
            "active": self.active,
        }


@dataclass(frozen=True)
class TransitionTrace:
    event_id: str
    tokens: tuple[str, ...]
    signatures: tuple[str, ...]
    salience: float
    support: float
    age: int


@dataclass(frozen=True)
class TransitionStimulus:
    mode: str
    event_id: str
    tokens: tuple[str, ...] = ()
    associations: tuple[AssociationObservation, ...] = ()
    transitions: tuple[TransitionObservation, ...] = ()
    salience: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"observe", "anneal", "consolidate", "abstract"}:
            raise ValueError("Unknown transition stimulus mode")
        if not isinstance(self.event_id, str) or not 1 <= len(self.event_id) <= 200:
            raise ValueError("event_id must contain 1 to 200 characters")
        if (
            isinstance(self.salience, bool)
            or not isinstance(self.salience, (int, float))
            or not math.isfinite(float(self.salience))
            or not 0.1 <= float(self.salience) <= 2.0
        ):
            raise ValueError("salience must be finite and within [0.1, 2.0]")


@dataclass(frozen=True)
class TransitionState:
    exposures: tuple[SurfaceExposure, ...]
    association_evidence: tuple[AssociationEvidence, ...]
    transition_evidence: tuple[TransitionEvidence, ...]
    surface_laws: tuple[SurfaceLaw, ...]
    transition_laws: tuple[TransitionLaw, ...]
    traces: tuple[TransitionTrace, ...]
    temperature: float
    energy: float
    phase_energy: float
    cumulative_phase_energy: float
    maximum_phase_energy: float
    accepted_improving_moves: int
    accepted_worse_moves: int
    observations: int
    forgotten: int
    conservation_applications: int
    radiated_event: str
    gravitated_event: str
    bound_event: str
    operator_counts: tuple[tuple[str, int], ...]
    outcome_counts: tuple[tuple[str, int], ...]
    last_outcome: str
    transition_hash: str
    transitions: int

    @property
    def information_mass(self) -> float:
        return sum(row.mass for row in self.surface_laws if row.active) + sum(
            row.mass for row in self.transition_laws if row.active
        )

    @property
    def raw_evidence_count(self) -> int:
        return (
            len(self.exposures)
            + len(self.association_evidence)
            + len(self.transition_evidence)
        )


@dataclass(frozen=True)
class TransitionExecutionRecord:
    recipe: str
    event_id: str
    mode: str
    primitives: tuple[str, ...]
    before_hash: str
    after_hash: str
    temperature: float
    energy: float
    phase_energy: float
    outcome: str


def _strict_index(value: str, name: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _expression_index(expression: str, prefix: str) -> int:
    pieces = expression.split(":")
    if len(pieces) != 2 or pieces[0] != prefix:
        raise ValueError(f"Invalid {prefix} expression")
    return _strict_index(pieces[1], f"{prefix} index")


def _counter_payload(
    values: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    source = dict(values)
    return tuple(sorted((str(name), int(count)) for name, count in source.items()))


def _bump_transition_counter(
    values: Sequence[tuple[str, int]],
    name: str,
) -> tuple[tuple[str, int], ...]:
    counts = Counter(dict(values))
    counts[name] += 1
    return _counter_payload(counts)


class UniverseTransitionKernel:
    """The only authority allowed to replace the learned transition state."""

    def __init__(
        self,
        config: TransitionConfig | None = None,
        disabled: Iterable[TransitionPrimitive] = (),
    ) -> None:
        self.config = config or TransitionConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self) -> TransitionState:
        return TransitionState(
            exposures=(),
            association_evidence=(),
            transition_evidence=(),
            surface_laws=(),
            transition_laws=(),
            traces=(),
            temperature=self.config.initial_temperature,
            energy=0.0,
            phase_energy=0.0,
            cumulative_phase_energy=0.0,
            maximum_phase_energy=0.0,
            accepted_improving_moves=0,
            accepted_worse_moves=0,
            observations=0,
            forgotten=0,
            conservation_applications=0,
            radiated_event="",
            gravitated_event="",
            bound_event="",
            operator_counts=(),
            outcome_counts=(),
            last_outcome="initial",
            transition_hash="0" * 64,
            transitions=0,
        )

    def apply(
        self,
        state: TransitionState,
        primitive: TransitionPrimitive,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        stimulus.validate()
        if primitive in self.disabled:
            return state
        handlers = {
            TransitionPrimitive.RADIATION: self._radiation,
            TransitionPrimitive.DISSIPATION: self._dissipation,
            TransitionPrimitive.GRAVITATION: self._gravitation,
            TransitionPrimitive.ATTRACTION_REPULSION: self._attraction_repulsion,
            TransitionPrimitive.NUCLEATION: self._nucleation,
            TransitionPrimitive.CONSERVATION: self._conservation,
            TransitionPrimitive.DECAY: self._decay,
        }
        transitioned = handlers[primitive](state, stimulus)
        digest = transition_hash(
            {
                "previous": state.transition_hash,
                "primitive": primitive.value,
                "event": stimulus.event_id,
                "mode": stimulus.mode,
                "surface_laws": [
                    (row.surface, row.concept, round(row.support, 9))
                    for row in transitioned.surface_laws
                ],
                "transition_laws": [
                    (row.law_id, row.signature, row.pattern)
                    for row in transitioned.transition_laws
                ],
                "temperature": round(transitioned.temperature, 12),
                "energy": round(transitioned.energy, 12),
                "outcome": transitioned.last_outcome,
            }
        )
        return replace(
            transitioned,
            operator_counts=_bump_transition_counter(
                transitioned.operator_counts,
                primitive.value,
            ),
            outcome_counts=_bump_transition_counter(
                transitioned.outcome_counts,
                transitioned.last_outcome,
            ),
            transition_hash=digest,
            transitions=state.transitions + 1,
        )

    def _radiation(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        if stimulus.mode == "observe":
            exposure_counts = Counter({row.surface: row.hits for row in state.exposures})
            exposure_counts.update(stimulus.tokens)
            exposures = tuple(
                SurfaceExposure(surface=surface, hits=hits)
                for surface, hits in sorted(exposure_counts.items())
            )
            trace = TransitionTrace(
                event_id=stimulus.event_id,
                tokens=stimulus.tokens,
                signatures=tuple(row.signature for row in stimulus.transitions),
                salience=float(stimulus.salience),
                support=float(stimulus.salience),
                age=0,
            )
            return replace(
                state,
                exposures=exposures,
                traces=(*state.traces, trace),
                observations=state.observations + 1,
                radiated_event=stimulus.event_id,
                last_outcome="transition_pulse_radiated",
            )
        if stimulus.mode == "anneal":
            phase = self.config.phase_mix_strength * (
                2.0
                * transition_fraction(
                    self.config.chaos_seed,
                    stimulus.event_id,
                    state.transitions,
                    "radiation",
                )
                - 1.0
            )
            energy = abs(phase)
            return replace(
                state,
                phase_energy=energy,
                cumulative_phase_energy=state.cumulative_phase_energy + energy,
                maximum_phase_energy=max(state.maximum_phase_energy, energy),
                radiated_event=stimulus.event_id,
                last_outcome="phase_wave_radiated",
            )
        return replace(
            state,
            radiated_event=stimulus.event_id,
            last_outcome="consolidation_wave_radiated",
        )

    def _gravitation(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        if state.radiated_event != stimulus.event_id:
            return state
        if stimulus.mode == "observe":
            associations = {
                (row.surface, row.concept): row for row in state.association_evidence
            }
            phase_energy = 0.0
            for observation in stimulus.associations:
                key = (observation.surface, observation.concept)
                existing = associations.get(key)
                weight = float(stimulus.salience) * float(observation.weight)
                interference = self.config.phase_mix_strength * (
                    transition_fraction(
                        self.config.chaos_seed,
                        stimulus.event_id,
                        observation.surface,
                        observation.concept,
                    )
                    - 0.5
                )
                phase_energy += abs(interference)
                angle = (
                    2.0
                    * math.pi
                    * observation.position
                    / max(1, observation.token_count)
                )
                associations[key] = AssociationEvidence(
                    surface=observation.surface,
                    concept=observation.concept,
                    hits=(0 if existing is None else existing.hits) + 1,
                    weight=(0.0 if existing is None else existing.weight) + weight,
                    phase_real=(0.0 if existing is None else existing.phase_real)
                    + weight * math.cos(angle + interference),
                    phase_imaginary=(
                        0.0 if existing is None else existing.phase_imaginary
                    )
                    + weight * math.sin(angle + interference),
                )
            transition_evidence = {
                (row.signature, row.pattern): row for row in state.transition_evidence
            }
            for observation in stimulus.transitions:
                key = (observation.signature, observation.pattern)
                existing = transition_evidence.get(key)
                transition_evidence[key] = TransitionEvidence(
                    signature=observation.signature,
                    pattern=observation.pattern,
                    slot_types=observation.slot_types,
                    effects=observation.effects,
                    hits=(0 if existing is None else existing.hits) + 1,
                    weight=(0.0 if existing is None else existing.weight)
                    + float(stimulus.salience) * observation.weight,
                )
            return replace(
                state,
                association_evidence=tuple(
                    associations[key] for key in sorted(associations)
                ),
                transition_evidence=tuple(
                    transition_evidence[key] for key in sorted(transition_evidence)
                ),
                phase_energy=phase_energy,
                cumulative_phase_energy=state.cumulative_phase_energy + phase_energy,
                maximum_phase_energy=max(state.maximum_phase_energy, phase_energy),
                gravitated_event=stimulus.event_id,
                last_outcome="delta_attractor_formed",
            )
        return replace(
            state,
            gravitated_event=stimulus.event_id,
            last_outcome="consolidation_attractor_formed",
        )

    def _attraction_repulsion(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        if state.gravitated_event != stimulus.event_id:
            return state
        if stimulus.mode == "anneal":
            energy = state.energy
            improving = state.accepted_improving_moves
            worse = state.accepted_worse_moves
            for trial in range(self.config.anneal_trials):
                fraction = transition_fraction(
                    self.config.chaos_seed,
                    stimulus.event_id,
                    state.transitions,
                    trial,
                )
                delta = (fraction - 0.47) * 0.09
                if delta <= 0.0:
                    energy = max(0.0, energy + delta)
                    improving += 1
                    continue
                acceptance = math.exp(-delta / max(state.temperature, 1e-9))
                draw = transition_fraction(
                    self.config.chaos_seed,
                    "accept",
                    stimulus.event_id,
                    trial,
                )
                if draw < acceptance:
                    energy += delta
                    worse += 1
            return replace(
                state,
                energy=energy,
                accepted_improving_moves=improving,
                accepted_worse_moves=worse,
                bound_event=stimulus.event_id,
                last_outcome="annealed_effect_candidates",
            )
        by_surface: defaultdict[str, list[AssociationEvidence]] = defaultdict(list)
        for row in state.association_evidence:
            by_surface[row.surface].append(row)
        ambiguity = 0.0
        for rows in by_surface.values():
            weights = sorted((row.weight for row in rows), reverse=True)
            if len(weights) > 1 and weights[0] > 0.0:
                ambiguity += weights[1] / weights[0]
        return replace(
            state,
            energy=ambiguity / max(1, len(by_surface)),
            bound_event=stimulus.event_id,
            last_outcome="compatible_delta_roles_bound",
        )

    def _candidate_surface_laws(
        self,
        state: TransitionState,
    ) -> tuple[SurfaceLaw, ...]:
        exposures = {row.surface: row.hits for row in state.exposures}
        by_surface: defaultdict[str, list[AssociationEvidence]] = defaultdict(list)
        for row in state.association_evidence:
            by_surface[row.surface].append(row)
        candidates: list[SurfaceLaw] = []
        for surface, rows in by_surface.items():
            ranked = sorted(
                rows,
                key=lambda row: (
                    -(row.weight * (0.8 + 0.2 * row.coherence)),
                    -row.hits,
                    row.concept,
                ),
            )
            top = ranked[0]
            second_hits = ranked[1].hits if len(ranked) > 1 else 0
            support = top.hits / max(1, exposures.get(surface, 0))
            if (
                top.hits < self.config.surface_min_hits
                or support < self.config.surface_min_support
                or top.hits + 1e-12 < second_hits * self.config.surface_margin
            ):
                continue
            candidates.append(
                SurfaceLaw(
                    surface=surface,
                    concept=top.concept,
                    evidence_count=top.hits,
                    mass=top.weight,
                    support=transition_clamp(support, 0.0, 1.0),
                    coherence=round(top.coherence, 12),
                )
            )
        return tuple(sorted(candidates, key=lambda row: row.surface))

    def _candidate_transition_laws(
        self,
        evidence: Sequence[TransitionEvidence],
    ) -> tuple[TransitionLaw, ...]:
        candidates: list[TransitionLaw] = []
        for row in evidence:
            if row.hits < self.config.law_min_hits:
                continue
            law_id = "law-" + row.signature[:16]
            candidates.append(
                TransitionLaw(
                    law_id=law_id,
                    signature=row.signature,
                    pattern=row.pattern,
                    slot_types=row.slot_types,
                    effects=row.effects,
                    evidence_count=row.hits,
                    mass=row.weight,
                    support=transition_clamp(
                        row.hits / (self.config.law_min_hits + 1.0),
                        0.0,
                        1.0,
                    ),
                )
            )
        return tuple(sorted(candidates, key=lambda row: (row.law_id, row.pattern)))

    def _nucleation(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        if stimulus.mode not in {"observe", "consolidate", "abstract"}:
            return state
        if state.bound_event != stimulus.event_id:
            return state
        surface_by_name = {
            row.surface: row for row in state.surface_laws if row.active
        }
        for candidate in self._candidate_surface_laws(state):
            current = surface_by_name.get(candidate.surface)
            if current is None or current.concept == candidate.concept:
                surface_by_name[candidate.surface] = candidate
        law_by_key = {
            (row.law_id, row.pattern): row
            for row in state.transition_laws
            if row.active
        }
        for candidate in self._candidate_transition_laws(state.transition_evidence):
            law_by_key[(candidate.law_id, candidate.pattern)] = candidate
        return replace(
            state,
            surface_laws=tuple(
                surface_by_name[name] for name in sorted(surface_by_name)
            ),
            transition_laws=tuple(law_by_key[key] for key in sorted(law_by_key)),
            last_outcome="transition_laws_nucleated",
        )

    def _conservation(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        concept_winners: dict[str, SurfaceLaw] = {}
        for law in sorted(
            state.surface_laws,
            key=lambda row: (-row.support, -row.evidence_count, row.surface),
        ):
            concept_winners.setdefault(law.concept, law)
        surfaces = tuple(sorted(concept_winners.values(), key=lambda row: row.surface))
        transition_winners: dict[tuple[str, tuple[str, ...]], TransitionLaw] = {}
        for law in sorted(
            state.transition_laws,
            key=lambda row: (-row.support, -row.evidence_count, row.pattern),
        ):
            transition_winners.setdefault((law.law_id, law.pattern), law)
        laws = tuple(
            sorted(
                transition_winners.values(),
                key=lambda row: (row.law_id, row.pattern),
            )
        )
        mass = sum(row.mass for row in surfaces) + sum(row.mass for row in laws)
        if mass > self.config.information_mass_budget:
            scale = self.config.information_mass_budget / mass
            surfaces = tuple(replace(row, mass=row.mass * scale) for row in surfaces)
            laws = tuple(replace(row, mass=row.mass * scale) for row in laws)
        return replace(
            state,
            surface_laws=surfaces,
            transition_laws=laws,
            conservation_applications=state.conservation_applications + 1,
            last_outcome="transition_mass_conserved",
        )

    def _dissipation(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        traces = tuple(
            replace(
                row,
                support=max(
                    0.0,
                    row.support
                    - self.config.trace_dissipation
                    * (1.4 if row.salience < 0.5 else 0.35),
                ),
                age=row.age + 1,
            )
            for row in state.traces
        )
        return replace(
            state,
            traces=traces,
            temperature=max(
                self.config.temperature_floor,
                state.temperature * self.config.cooling_rate,
            ),
            last_outcome="transition_transients_dissipated",
        )

    def _decay(
        self,
        state: TransitionState,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        if stimulus.mode == "abstract":
            forgotten = state.forgotten + len(state.traces)
            return replace(
                state,
                exposures=(),
                association_evidence=(),
                transition_evidence=(),
                traces=(),
                forgotten=forgotten,
                last_outcome="raw_transition_episodes_coarse_grained",
            )
        kept = tuple(
            row for row in state.traces if row.support >= self.config.trace_expiration
        )
        return replace(
            state,
            traces=kept,
            forgotten=state.forgotten + len(state.traces) - len(kept),
            last_outcome="unsupported_transition_traces_decayed",
        )


class TransitionRuntime:
    def __init__(
        self,
        kernel: UniverseTransitionKernel | None = None,
        state: TransitionState | None = None,
        knowledge: AtomWikiGraph | None = None,
    ) -> None:
        self.kernel = kernel or UniverseTransitionKernel()
        self.knowledge = knowledge or build_language_graph()
        self.knowledge.assert_all_leaves_are_universe_primitives()
        self.state = state or self.kernel.initial_state()
        self.records: list[TransitionExecutionRecord] = []
        self.last_retrieval: list[dict[str, object]] = []

    def execute(
        self,
        recipe: str,
        stimulus: TransitionStimulus,
    ) -> TransitionState:
        primitives = self.knowledge.expand(recipe)
        before = self.state.transition_hash
        for name in primitives:
            self.state = self.kernel.apply(
                self.state,
                TransitionPrimitive(name),
                stimulus,
            )
        self.records.append(
            TransitionExecutionRecord(
                recipe=recipe,
                event_id=stimulus.event_id,
                mode=stimulus.mode,
                primitives=primitives,
                before_hash=before,
                after_hash=self.state.transition_hash,
                temperature=self.state.temperature,
                energy=self.state.energy,
                phase_energy=self.state.phase_energy,
                outcome=self.state.last_outcome,
            )
        )
        return self.state

    def observe(self, stimulus: TransitionStimulus) -> TransitionState:
        if stimulus.mode != "observe":
            raise ValueError("observe requires an observe stimulus")
        return self.execute("language_learn", stimulus)

    def anneal(self, event_id: str) -> TransitionState:
        return self.execute(
            "thermal_anneal",
            TransitionStimulus(mode="anneal", event_id=event_id),
        )

    def remember(self, event_id: str) -> TransitionState:
        return self.execute(
            "remember",
            TransitionStimulus(mode="consolidate", event_id=event_id),
        )

    def forget_raw(self, event_id: str) -> TransitionState:
        return self.execute(
            "forget",
            TransitionStimulus(mode="abstract", event_id=event_id),
        )

    def retrieve(self, query: str) -> list[dict[str, object]]:
        self.last_retrieval = retrieve_atom_context(self.knowledge, query, limit=12)
        return self.last_retrieval


def transition_config_with(
    base: TransitionConfig | None = None,
    **overrides: Any,
) -> TransitionConfig:
    values = asdict(base or TransitionConfig())
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"Unknown transition config values: {sorted(unknown)}")
    values.update(overrides)
    config = TransitionConfig(**values)
    config.validate()
    return config


def surface_law_maps(
    laws: Sequence[SurfaceLaw],
) -> tuple[dict[str, str], dict[str, str]]:
    surface_to_concept = {law.surface: law.concept for law in laws if law.active}
    concept_to_surface: dict[str, str] = {}
    for law in sorted(
        (row for row in laws if row.active),
        key=lambda row: (-row.support, -row.evidence_count, row.surface),
    ):
        concept_to_surface.setdefault(law.concept, law.surface)
    return surface_to_concept, concept_to_surface


def lexical_stimulus_from_row(row: Mapping[str, Any]) -> TransitionStimulus:
    validate_transition_row(row)
    tokens = transition_tokens(str(row["text"]))
    participants = delta_participants(row)
    associations = tuple(
        AssociationObservation(
            surface=surface,
            concept=concept,
            position=position,
            token_count=len(tokens),
        )
        for position, surface in enumerate(tokens)
        for concept in participants
    )
    return TransitionStimulus(
        mode="observe",
        event_id=f"lexical-{row['case_id']}",
        tokens=tokens,
        associations=associations,
        salience=float(row["salience"]),
    )


def _encode_before_expression(value: Any, slot_map: Mapping[str, int]) -> str:
    if value is None:
        return "none"
    concept = str(value)
    if concept in slot_map:
        return f"slot:{slot_map[concept]}"
    return f"any:{transition_concept_kind(concept)}"


def _encode_after_expression(
    value: Any,
    collection: str,
    before: Mapping[str, Any],
    slot_concepts: Sequence[str],
    slot_map: Mapping[str, int],
) -> str:
    if value is None:
        return "none"
    concept = str(value)
    if concept in slot_map:
        return f"slot:{slot_map[concept]}"
    for index, slot_concept in enumerate(slot_concepts):
        if slot_concept in before[collection]:
            if before[collection][slot_concept] == value:
                return f"old:{collection}:{index}"
    raise ValueError("after-state value is not expressible from utterance slots")


def transition_observation_from_row(
    row: Mapping[str, Any],
    surface_to_concept: Mapping[str, str],
) -> TransitionObservation:
    validate_transition_row(row)
    tokens = transition_tokens(str(row["text"]))
    slot_concepts: list[str] = []
    pattern: list[str] = []
    for token in tokens:
        concept = surface_to_concept.get(token)
        if concept is None:
            pattern.append(token)
            continue
        if concept in slot_concepts:
            index = slot_concepts.index(concept)
        else:
            index = len(slot_concepts)
            slot_concepts.append(concept)
        pattern.append("{" + str(index) + "}")
    if not slot_concepts:
        raise ValueError("utterance has no grounded entity slots")
    slot_map = {concept: index for index, concept in enumerate(slot_concepts)}
    before = row["before"]
    after = row["after"]
    effects: list[EffectAtom] = []
    for collection in ("locations", "holders"):
        for key in sorted(before[collection]):
            old = before[collection][key]
            new = after[collection][key]
            if old == new:
                continue
            if key not in slot_map:
                raise ValueError("changed world key is not grounded in the utterance")
            effects.append(
                EffectAtom(
                    collection=collection,
                    key_slot=slot_map[key],
                    before=_encode_before_expression(old, slot_map),
                    after=_encode_after_expression(
                        new,
                        collection,
                        before,
                        slot_concepts,
                        slot_map,
                    ),
                )
            )
    slot_types = tuple(transition_concept_kind(value) for value in slot_concepts)
    ordered_effects = tuple(
        sorted(
            effects,
            key=lambda row: (row.collection, row.key_slot, row.before, row.after),
        )
    )
    for effect in ordered_effects:
        effect.validate(slot_types)
    signature = transition_hash(
        {
            "slot_types": list(slot_types),
            "effects": [effect.payload() for effect in ordered_effects],
        }
    )
    return TransitionObservation(
        signature=signature,
        pattern=tuple(pattern),
        slot_types=slot_types,
        effects=ordered_effects,
    )


def transition_stimulus_from_row(
    row: Mapping[str, Any],
    surface_to_concept: Mapping[str, str],
) -> TransitionStimulus:
    observation = transition_observation_from_row(row, surface_to_concept)
    return TransitionStimulus(
        mode="observe",
        event_id=f"effect-{row['case_id']}",
        tokens=transition_tokens(str(row["text"])),
        transitions=(observation,),
        salience=float(row["salience"]),
    )


def train_transition_field(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TransitionConfig | None = None,
    disabled: Iterable[TransitionPrimitive] = (),
) -> tuple[TransitionRuntime, list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("transition training rows cannot be empty")
    runtime = TransitionRuntime(
        kernel=UniverseTransitionKernel(config=config, disabled=disabled)
    )
    history: list[dict[str, Any]] = []
    for row in rows:
        runtime.observe(lexical_stimulus_from_row(row))
    runtime.anneal("lexical-phase-anneal")
    runtime.remember("lexical-memory-consolidation")
    history.append(transition_training_snapshot(runtime, "lexical"))
    surface_to_concept, _ = surface_law_maps(runtime.state.surface_laws)
    unresolved: list[str] = []
    for row in rows:
        try:
            stimulus = transition_stimulus_from_row(row, surface_to_concept)
        except ValueError:
            unresolved.append(str(row["case_id"]))
            continue
        runtime.observe(stimulus)
    runtime.anneal("effect-phase-anneal")
    runtime.remember("effect-memory-consolidation")
    history.append(transition_training_snapshot(runtime, "effect"))
    diagnostics = {
        "training_rows": len(rows),
        "unresolved_case_ids": unresolved,
        "surface_laws": len(runtime.state.surface_laws),
        "transition_laws": len(runtime.state.transition_laws),
    }
    return runtime, history, diagnostics


def transition_training_snapshot(
    runtime: TransitionRuntime,
    stage: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "state_hash": runtime.state.transition_hash,
        "surface_laws": len(runtime.state.surface_laws),
        "transition_laws": len(runtime.state.transition_laws),
        "raw_traces": len(runtime.state.traces),
        "raw_evidence": runtime.state.raw_evidence_count,
        "temperature": runtime.state.temperature,
        "energy": runtime.state.energy,
        "phase_energy": runtime.state.cumulative_phase_energy,
    }


def _match_transition_law(
    runtime: TransitionRuntime,
    text: str,
) -> tuple[TransitionLaw, tuple[str, ...]]:
    tokens = transition_tokens(text)
    surface_to_concept, _ = surface_law_maps(runtime.state.surface_laws)
    candidates: list[tuple[TransitionLaw, tuple[str, ...]]] = []
    for law in runtime.state.transition_laws:
        if not law.active or len(law.pattern) != len(tokens):
            continue
        slots: list[str | None] = [None] * len(law.slot_types)
        valid = True
        for pattern_piece, token in zip(law.pattern, tokens, strict=True):
            if pattern_piece.startswith("{") and pattern_piece.endswith("}"):
                index = _strict_index(pattern_piece[1:-1], "pattern slot")
                if index >= len(slots):
                    valid = False
                    break
                concept = surface_to_concept.get(token)
                if concept is None or transition_concept_kind(concept) != law.slot_types[index]:
                    valid = False
                    break
                if slots[index] is not None and slots[index] != concept:
                    valid = False
                    break
                slots[index] = concept
            elif pattern_piece != token:
                valid = False
                break
        if valid and all(slot is not None for slot in slots):
            resolved = tuple(str(slot) for slot in slots)
            if len(resolved) == len(set(resolved)):
                candidates.append((law, resolved))
    if not candidates:
        raise ValueError("no learned transition law matches the utterance")
    if len(candidates) != 1:
        raise ValueError("utterance matches multiple transition laws")
    return candidates[0]


def _expected_before_value(
    expression: str,
    slots: Sequence[str],
) -> tuple[str, Any]:
    if expression == "none":
        return "exact", None
    if expression.startswith("slot:"):
        return "exact", slots[_expression_index(expression, "slot")]
    if expression.startswith("any:"):
        return "kind", expression.split(":", 1)[1]
    raise ValueError("invalid before-state expression")


def _resolved_after_value(
    expression: str,
    before: Mapping[str, Any],
    slots: Sequence[str],
) -> Any:
    if expression == "none":
        return None
    if expression.startswith("slot:"):
        return slots[_expression_index(expression, "slot")]
    if expression.startswith("old:"):
        pieces = expression.split(":")
        collection = pieces[1]
        key = slots[_strict_index(pieces[2], "old-cell slot")]
        return before[collection][key]
    raise ValueError("invalid after-state expression")


def apply_transition_text(
    runtime: TransitionRuntime,
    text: str,
    world: Mapping[str, Any],
) -> dict[str, Any]:
    before = copy_transition_world(world)
    runtime.retrieve("ground learn remember abstract transition effect law")
    law, slots = _match_transition_law(runtime, text)
    after = deepcopy(before)
    for effect in law.effects:
        key = slots[effect.key_slot]
        if key not in before[effect.collection]:
            raise ValueError("learned effect targets an absent world key")
        actual = before[effect.collection][key]
        mode, expected = _expected_before_value(effect.before, slots)
        if mode == "exact" and actual != expected:
            raise ValueError("learned transition precondition is not satisfied")
        if mode == "kind":
            if actual is None or transition_concept_kind(str(actual)) != expected:
                raise ValueError("learned transition type precondition is not satisfied")
        after[effect.collection][key] = _resolved_after_value(
            effect.after,
            before,
            slots,
        )
    validate_transition_world(after)
    return {
        "status": "applied",
        "law_id": law.law_id,
        "slots": list(slots),
        "pattern": list(law.pattern),
        "effects": [effect.payload() for effect in law.effects],
        "world_before": before,
        "world_after": copy_transition_world(after),
        "knowledge_context": runtime.last_retrieval,
    }


def generate_transition_text(
    runtime: TransitionRuntime,
    law_id: str,
    slots: Sequence[str],
) -> dict[str, Any]:
    matching = [
        law for law in runtime.state.transition_laws if law.active and law.law_id == law_id
    ]
    if len(matching) != 1:
        raise ValueError("law_id does not resolve to one active transition law")
    law = matching[0]
    if len(slots) != len(law.slot_types):
        raise ValueError("generation slot count does not match the law")
    _, concept_to_surface = surface_law_maps(runtime.state.surface_laws)
    output: list[str] = []
    for piece in law.pattern:
        if piece.startswith("{") and piece.endswith("}"):
            index = _strict_index(piece[1:-1], "generation slot")
            concept = str(slots[index])
            if transition_concept_kind(concept) != law.slot_types[index]:
                raise ValueError("generation concept type does not match the law slot")
            surface = concept_to_surface.get(concept)
            if surface is None:
                raise ValueError("generation concept has no learned surface")
            output.append(surface)
        else:
            output.append(piece)
    return {
        "status": "generated",
        "law_id": law.law_id,
        "text": " ".join(output),
        "slots": list(slots),
    }


def evaluator_law_mapping(
    runtime: TransitionRuntime,
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        law, _ = _match_transition_law(runtime, str(row["text"]))
        label = str(truth[str(row["case_id"])]["semantic_label"])
        votes[label][law.law_id] += 1
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for label in sorted(votes):
        ranked = votes[label].most_common()
        if len(ranked) != 1:
            raise ValueError("evaluator label did not settle on one emergent law")
        law_id = ranked[0][0]
        if law_id in used:
            raise ValueError("multiple evaluator labels settled on the same law")
        mapping[label] = law_id
        used.add(law_id)
    return mapping


def evaluate_transition_rows(
    runtime: TransitionRuntime,
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    evaluator_mapping: Mapping[str, str],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        target = truth[str(row["case_id"])]
        try:
            result = apply_transition_text(runtime, str(row["text"]), row["before"])
            generated = generate_transition_text(
                runtime,
                str(result["law_id"]),
                result["slots"],
            )
            expected_law = evaluator_mapping[str(target["semantic_label"])]
            execution_correct = result["world_after"] == target["expected_after"]
            law_correct = result["law_id"] == expected_law
            generation_correct = generated["text"] == row["text"]
            error = None
        except (KeyError, TypeError, ValueError) as exc:
            result = None
            generated = None
            execution_correct = False
            law_correct = False
            generation_correct = False
            error = str(exc)
        predictions.append(
            {
                "case_id": row["case_id"],
                "semantic_label": target["semantic_label"],
                "predicted_law_id": None if result is None else result["law_id"],
                "execution_correct": execution_correct,
                "law_correct": law_correct,
                "generation_correct": generation_correct,
                "generated": None if generated is None else generated["text"],
                "error": error,
            }
        )
    cases = len(predictions)
    return {
        "cases": cases,
        "execution_correct": sum(row["execution_correct"] for row in predictions),
        "execution_accuracy": sum(row["execution_correct"] for row in predictions)
        / max(1, cases),
        "law_correct": sum(row["law_correct"] for row in predictions),
        "law_accuracy": sum(row["law_correct"] for row in predictions) / max(1, cases),
        "generation_correct": sum(row["generation_correct"] for row in predictions),
        "generation_accuracy": sum(row["generation_correct"] for row in predictions)
        / max(1, cases),
        "predictions": predictions,
    }


def _strict_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _surface_law_from_payload(payload: Mapping[str, Any]) -> SurfaceLaw:
    expected = {
        "surface",
        "concept",
        "evidence_count",
        "mass",
        "support",
        "coherence",
        "active",
    }
    if set(payload) != expected:
        raise ValueError("surface law shape is invalid")
    law = SurfaceLaw(
        surface=str(payload["surface"]),
        concept=str(payload["concept"]),
        evidence_count=int(payload["evidence_count"]),
        mass=_strict_finite(payload["mass"], "surface law mass"),
        support=_strict_finite(payload["support"], "surface law support"),
        coherence=_strict_finite(payload["coherence"], "surface law coherence"),
        active=payload["active"],
    )
    if not law.surface or transition_tokens(law.surface) != (law.surface,):
        raise ValueError("surface law must contain one normalized token")
    transition_concept_kind(law.concept)
    if law.evidence_count < 2 or law.mass <= 0.0:
        raise ValueError("surface law evidence is invalid")
    if not 0.0 <= law.support <= 1.0 or not 0.0 <= law.coherence <= 1.0:
        raise ValueError("surface law scores must be within [0, 1]")
    if not isinstance(law.active, bool) or not law.active:
        raise ValueError("serialized surface laws must be active")
    return law


def _transition_law_from_payload(payload: Mapping[str, Any]) -> TransitionLaw:
    expected = {
        "law_id",
        "signature",
        "pattern",
        "slot_types",
        "effects",
        "evidence_count",
        "mass",
        "support",
        "active",
    }
    if set(payload) != expected:
        raise ValueError("transition law shape is invalid")
    if not isinstance(payload["pattern"], list) or not isinstance(
        payload["slot_types"], list
    ) or not isinstance(payload["effects"], list):
        raise ValueError("transition law sequences must be arrays")
    slot_types = tuple(str(value) for value in payload["slot_types"])
    if not slot_types or any(
        value not in {"agent", "object", "location"} for value in slot_types
    ):
        raise ValueError("transition law slot types are invalid")
    effects = tuple(
        EffectAtom.from_payload(effect, slot_types) for effect in payload["effects"]
    )
    if not effects:
        raise ValueError("transition law requires at least one effect")
    signature = transition_hash(
        {
            "slot_types": list(slot_types),
            "effects": [effect.payload() for effect in effects],
        }
    )
    law = TransitionLaw(
        law_id=str(payload["law_id"]),
        signature=str(payload["signature"]),
        pattern=tuple(str(piece) for piece in payload["pattern"]),
        slot_types=slot_types,
        effects=effects,
        evidence_count=int(payload["evidence_count"]),
        mass=_strict_finite(payload["mass"], "transition law mass"),
        support=_strict_finite(payload["support"], "transition law support"),
        active=payload["active"],
    )
    if law.signature != signature or law.law_id != "law-" + signature[:16]:
        raise ValueError("transition law identity does not match its effect program")
    if law.evidence_count < 2 or law.mass <= 0.0:
        raise ValueError("transition law evidence is invalid")
    if not 0.0 <= law.support <= 1.0:
        raise ValueError("transition law support must be within [0, 1]")
    if not isinstance(law.active, bool) or not law.active:
        raise ValueError("serialized transition laws must be active")
    for piece in law.pattern:
        if piece.startswith("{") and piece.endswith("}"):
            if _strict_index(piece[1:-1], "pattern slot") >= len(slot_types):
                raise ValueError("transition pattern slot is out of range")
        elif transition_tokens(piece) != (piece,):
            raise ValueError("transition pattern literals must be normalized tokens")
    return law


def transition_model_payload(runtime: TransitionRuntime) -> dict[str, Any]:
    if runtime.state.traces or runtime.state.raw_evidence_count:
        raise ValueError("abstract raw transition evidence before serialization")
    core = {
        "schema_version": TRANSITION_MODEL_SCHEMA,
        "runtime": ATOM_TRANSITION_RUNTIME,
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "knowledge_graph_hash": transition_hash(runtime.knowledge.manifest()),
        "config": asdict(runtime.kernel.config),
        "surface_laws": [law.payload() for law in runtime.state.surface_laws],
        "transition_laws": [law.payload() for law in runtime.state.transition_laws],
        "raw_episode_count": len(runtime.state.traces),
        "raw_evidence_count": runtime.state.raw_evidence_count,
        "temperature": runtime.state.temperature,
        "energy": runtime.state.energy,
        "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
        "maximum_phase_energy": runtime.state.maximum_phase_energy,
        "accepted_improving_moves": runtime.state.accepted_improving_moves,
        "accepted_worse_moves": runtime.state.accepted_worse_moves,
        "observations": runtime.state.observations,
        "forgotten": runtime.state.forgotten,
        "conservation_applications": runtime.state.conservation_applications,
        "operator_counts": dict(runtime.state.operator_counts),
        "outcome_counts": dict(runtime.state.outcome_counts),
        "transition_hash": runtime.state.transition_hash,
        "transitions": runtime.state.transitions,
    }
    return {**core, "model_hash": transition_hash(core)}


def runtime_from_transition_model(payload: Mapping[str, Any]) -> TransitionRuntime:
    expected = {
        "schema_version",
        "runtime",
        "wiki_runtime",
        "rag_runtime",
        "knowledge_graph_hash",
        "config",
        "surface_laws",
        "transition_laws",
        "raw_episode_count",
        "raw_evidence_count",
        "temperature",
        "energy",
        "cumulative_phase_energy",
        "maximum_phase_energy",
        "accepted_improving_moves",
        "accepted_worse_moves",
        "observations",
        "forgotten",
        "conservation_applications",
        "operator_counts",
        "outcome_counts",
        "transition_hash",
        "transitions",
        "model_hash",
    }
    if set(payload) != expected:
        raise ValueError("transition model fields are invalid")
    if payload["schema_version"] != TRANSITION_MODEL_SCHEMA:
        raise ValueError("unsupported transition model schema")
    if payload["runtime"] != ATOM_TRANSITION_RUNTIME:
        raise ValueError("transition runtime marker mismatch")
    if payload["wiki_runtime"] != ATOM_WIKI_GRAPH_RUNTIME:
        raise ValueError("transition wiki runtime marker mismatch")
    if payload["rag_runtime"] != ATOM_RAG_RUNTIME:
        raise ValueError("transition RAG runtime marker mismatch")
    core = {key: payload[key] for key in payload if key != "model_hash"}
    if transition_hash(core) != payload["model_hash"]:
        raise ValueError("transition model hash mismatch")
    if not isinstance(payload["config"], dict):
        raise ValueError("transition config must be an object")
    config = TransitionConfig(**payload["config"])
    config.validate()
    if not isinstance(payload["surface_laws"], list) or not isinstance(
        payload["transition_laws"], list
    ):
        raise ValueError("transition laws must be arrays")
    surface_laws = tuple(
        _surface_law_from_payload(row) for row in payload["surface_laws"]
    )
    transition_laws = tuple(
        _transition_law_from_payload(row) for row in payload["transition_laws"]
    )
    if len({row.surface for row in surface_laws}) != len(surface_laws):
        raise ValueError("serialized surfaces must be unique")
    if len({row.concept for row in surface_laws}) != len(surface_laws):
        raise ValueError("serialized concepts must be unique")
    if len({(row.law_id, row.pattern) for row in transition_laws}) != len(
        transition_laws
    ):
        raise ValueError("serialized transition laws must be unique")
    if payload["raw_episode_count"] != 0 or payload["raw_evidence_count"] != 0:
        raise ValueError("serialized transition model must not retain raw evidence")
    graph = build_language_graph()
    if transition_hash(graph.manifest()) != payload["knowledge_graph_hash"]:
        raise ValueError("transition knowledge graph hash mismatch")
    for field in ("operator_counts", "outcome_counts"):
        if not isinstance(payload[field], dict) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in payload[field].values()
        ):
            raise ValueError(f"{field} must contain non-negative integers")
    integer_fields = (
        "accepted_improving_moves",
        "accepted_worse_moves",
        "observations",
        "forgotten",
        "conservation_applications",
        "transitions",
    )
    for field in integer_fields:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    state = TransitionState(
        exposures=(),
        association_evidence=(),
        transition_evidence=(),
        surface_laws=surface_laws,
        transition_laws=transition_laws,
        traces=(),
        temperature=_strict_finite(payload["temperature"], "temperature"),
        energy=_strict_finite(payload["energy"], "energy"),
        phase_energy=0.0,
        cumulative_phase_energy=_strict_finite(
            payload["cumulative_phase_energy"], "cumulative phase energy"
        ),
        maximum_phase_energy=_strict_finite(
            payload["maximum_phase_energy"], "maximum phase energy"
        ),
        accepted_improving_moves=payload["accepted_improving_moves"],
        accepted_worse_moves=payload["accepted_worse_moves"],
        observations=payload["observations"],
        forgotten=payload["forgotten"],
        conservation_applications=payload["conservation_applications"],
        radiated_event="",
        gravitated_event="",
        bound_event="",
        operator_counts=_counter_payload(payload["operator_counts"]),
        outcome_counts=_counter_payload(payload["outcome_counts"]),
        last_outcome="restored",
        transition_hash=str(payload["transition_hash"]),
        transitions=payload["transitions"],
    )
    return TransitionRuntime(
        kernel=UniverseTransitionKernel(config=config),
        state=state,
        knowledge=graph,
    )


def write_transition_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_transition_workflow_request(payload: Mapping[str, Any]) -> None:
    expected = {"schema_version", "request_id", "world", "turns"}
    if set(payload) != expected:
        raise ValueError("transition workflow request fields are invalid")
    if payload["schema_version"] != TRANSITION_MODEL_SCHEMA:
        raise ValueError("unsupported transition workflow schema")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("workflow request_id must be non-empty text")
    validate_transition_world(payload["world"])
    turns = payload["turns"]
    if not isinstance(turns, list) or not 1 <= len(turns) <= 100:
        raise ValueError("workflow turns must contain 1 to 100 entries")
    seen: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"turn_id", "text"}:
            raise ValueError("workflow turn fields are invalid")
        turn_id = turn["turn_id"]
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen:
            raise ValueError("workflow turn identifiers must be unique text")
        seen.add(turn_id)
        transition_tokens(str(turn["text"]))


def run_transition_workflow(
    model_path: Path,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict) or not isinstance(request, dict):
        raise ValueError("workflow model and request roots must be objects")
    runtime = runtime_from_transition_model(model)
    validate_transition_workflow_request(request)
    world = copy_transition_world(request["world"])
    turns: list[dict[str, Any]] = []
    for turn in request["turns"]:
        result = apply_transition_text(runtime, str(turn["text"]), world)
        generated = generate_transition_text(
            runtime,
            str(result["law_id"]),
            result["slots"],
        )
        world = result["world_after"]
        turns.append(
            {
                "turn_id": turn["turn_id"],
                "text": turn["text"],
                "law_id": result["law_id"],
                "slots": result["slots"],
                "effects": result["effects"],
                "generated": generated["text"],
                "world_after": world,
                "knowledge_context": result["knowledge_context"],
            }
        )
    core = {
        "schema_version": TRANSITION_MODEL_SCHEMA,
        "request_id": request["request_id"],
        "runtime": {
            "transition_runtime": ATOM_TRANSITION_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "model_hash": model["model_hash"],
        },
        "turns": turns,
        "final_world": world,
    }
    response = {**core, "workflow_hash": transition_hash(core)}
    write_transition_json(response_path, response)
    return response


def transition_architecture_audit(source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    graph = build_language_graph()
    graph.assert_all_leaves_are_universe_primitives()
    checks = {
        "primitive_enum_matches_universe_core": {
            value.value for value in TransitionPrimitive
        }
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "learning_recipe_resolves_to_all_seven": set(graph.expand("language_learn"))
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "fixed_predicate_inventory_is_not_imported": (
            "atom_language_field" not in imported_modules
        ),
        "wiki_and_rag_are_runtime_wired": "retrieve_atom_context" in source
        and "build_language_graph" in source,
        "model_loader_is_fail_closed": "transition model fields are invalid" in source
        and "transition model hash mismatch" in source,
        "effect_program_is_executable": "apply_transition_text" in source
        and "_resolved_after_value" in source,
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_transition_self_tests() -> dict[str, Any]:
    program = build_transition_discovery_program()
    row = program["train"][0]
    participants = delta_participants(row)
    corrupt = dict(row)
    corrupt["unexpected"] = True
    corrupt_rejected = False
    try:
        validate_transition_row(corrupt)
    except ValueError:
        corrupt_rejected = True
    graph = build_language_graph()
    retrieval = retrieve_atom_context(
        graph,
        "ground learn remember forget abstract transition consequence",
        limit=12,
    )
    retrieved = {row["name"] for row in retrieval}
    checks = {
        "program_is_50_15_25": program["manifest"]["counts"]
        == {"train": 50, "validation": 15, "heldout": 25},
        "observations_hide_evaluator_semantics": all(
            not {"semantic_label", "participants", "predicate", "roles"} & set(row)
            for split in ("train", "validation", "heldout")
            for row in program[split]
        ),
        "heldout_surfaces_are_unseen": program["manifest"][
            "heldout_surface_overlap"
        ]
        == 0,
        "delta_participants_are_observable": len(participants) >= 2,
        "strict_row_validation_rejects_unknown_fields": corrupt_rejected,
        "graph_retrieval_reaches_cognitive_atoms": {
            "ground",
            "learn",
            "remember",
            "forget",
            "abstract",
        }
        <= retrieved,
        "architecture_audit_passes": transition_architecture_audit()["passed"],
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }
