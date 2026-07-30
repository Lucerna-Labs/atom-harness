"""Grounded, bidirectional Atom Language Field.

This module contains no neural network, gradient path, pretrained model, or
trainable weight matrix.  Surface pulses crystallize into lexical, frame, and
reference laws through the seven universe primitives.  The immutable language
state may be replaced only by :class:`UniverseLanguageKernel`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
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
)


LANGUAGE_MODEL_SCHEMA = 1
LANGUAGE_SEED = 20260721
ATOM_LANGUAGE_RUNTIME = "atom-grounded-language-field-v1"


class Primitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        json.dumps([seed, *parts], sort_keys=True, default=str).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def tokenize_word_pulses(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        raise ValueError("text must contain 1 to 256 characters")
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def character_pulses(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        raise ValueError("text must contain 1 to 256 characters")
    return tuple(text.lower())


def spans_from_character_pulses(pulses: Sequence[str]) -> tuple[str, ...]:
    joined = "".join(pulses)
    return tuple(re.findall(r"[a-z0-9]+", joined))


def pulses_for_stage(text: str, stage: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if stage == "word":
        words = tokenize_word_pulses(text)
        return words, ()
    if stage == "character":
        characters = character_pulses(text)
        return spans_from_character_pulses(characters), characters
    raise ValueError("stage must be 'word' or 'character'")


def concept_kind(concept: str) -> str:
    if concept.startswith("agent-"):
        return "agent"
    if concept.startswith("object-"):
        return "patient"
    if concept.startswith("location-"):
        return "destination"
    raise ValueError(f"Unknown grounded concept: {concept}")


ROLE_KINDS = {
    "agent": "agent",
    "patient": "patient",
    "destination": "destination",
    "recipient": "agent",
}


PREDICATE_ROLES = {
    "MOVE": ("agent", "destination"),
    "TAKE": ("agent", "patient"),
    "GIVE": ("agent", "patient", "recipient"),
    "AT": ("agent", "destination"),
    "HAS": ("agent", "patient"),
    "WHERE": ("agent",),
    "WHAT_HAS": ("agent",),
    "WHO_HAS": ("patient",),
    "HAS_QUERY": ("agent", "patient"),
    "AT_QUERY": ("agent", "destination"),
    "YES": (),
    "NO": (),
}


@dataclass(frozen=True)
class MeaningFrame:
    speech_act: str
    predicate: str
    polarity: bool
    roles: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if self.speech_act not in {"command", "assertion", "question", "answer"}:
            raise ValueError("Unknown speech act")
        if self.predicate not in PREDICATE_ROLES:
            raise ValueError("Unknown predicate")
        if not isinstance(self.polarity, bool):
            raise ValueError("polarity must be boolean")
        role_map = dict(self.roles)
        if len(role_map) != len(self.roles):
            raise ValueError("meaning roles must be unique")
        expected = set(PREDICATE_ROLES[self.predicate])
        if set(role_map) != expected:
            raise ValueError(
                f"{self.predicate} roles mismatch; expected={sorted(expected)}"
            )
        for role, concept in self.roles:
            if role not in ROLE_KINDS:
                raise ValueError(f"Unknown semantic role: {role}")
            if concept_kind(concept) != ROLE_KINDS[role]:
                raise ValueError(f"Concept {concept} cannot fill role {role}")

    @property
    def key(self) -> str:
        return f"{self.speech_act}:{self.predicate}:{int(self.polarity)}"

    @property
    def role_map(self) -> dict[str, str]:
        return dict(self.roles)

    def payload(self) -> dict[str, Any]:
        return {
            "speech_act": self.speech_act,
            "predicate": self.predicate,
            "polarity": self.polarity,
            "roles": dict(self.roles),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MeaningFrame:
        expected = {"speech_act", "predicate", "polarity", "roles"}
        if set(payload) != expected or not isinstance(payload["roles"], dict):
            raise ValueError("meaning frame shape is invalid")
        frame = cls(
            speech_act=str(payload["speech_act"]),
            predicate=str(payload["predicate"]),
            polarity=payload["polarity"],
            roles=tuple(
                sorted(
                    (str(role), str(value)) for role, value in payload["roles"].items()
                )
            ),
        )
        frame.validate()
        return frame


def make_frame(
    speech_act: str, predicate: str, polarity: bool = True, **roles: str
) -> MeaningFrame:
    frame = MeaningFrame(
        speech_act=speech_act,
        predicate=predicate,
        polarity=polarity,
        roles=tuple(sorted(roles.items())),
    )
    frame.validate()
    return frame


@dataclass(frozen=True)
class LanguageConfig:
    initial_temperature: float = 1.40
    temperature_floor: float = 0.20
    cooling_rate: float = 0.96
    phase_mix_strength: float = 0.045
    lexeme_min_hits: int = 2
    lexeme_min_selectivity: float = 0.30
    lexeme_margin: float = 1.10
    frame_min_hits: int = 2
    reference_min_hits: int = 2
    character_span_min_hits: int = 2
    information_mass_budget: float = 96.0
    trace_dissipation: float = 0.08
    trace_expiration: float = 0.10
    anneal_trials: int = 18
    context_ttl: int = 3
    chaos_seed: int = LANGUAGE_SEED

    def validate(self) -> None:
        integer_ranges = {
            "lexeme_min_hits": (2, 100),
            "frame_min_hits": (2, 100),
            "reference_min_hits": (2, 100),
            "character_span_min_hits": (2, 100),
            "anneal_trials": (1, 1_000),
            "context_ttl": (1, 32),
            "chaos_seed": (0, 2**63 - 1),
        }
        for name, (minimum, maximum) in integer_ranges.items():
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
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_floor > self.initial_temperature:
            raise ValueError("temperature_floor cannot exceed initial_temperature")
        for name in (
            "cooling_rate",
            "phase_mix_strength",
            "lexeme_min_selectivity",
            "trace_dissipation",
            "trace_expiration",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not 1.0 <= self.lexeme_margin <= 10.0:
            raise ValueError("lexeme_margin must be within [1, 10]")


@dataclass(frozen=True)
class AssociationObservation:
    surface: str
    concept: str
    position: int
    token_count: int
    weight: float = 1.0


@dataclass(frozen=True)
class TemplateObservation:
    direction: str
    frame: MeaningFrame
    pattern: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class ReferenceObservation:
    surface: str
    concept_type: str
    recency: int
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
        return clamp(
            math.hypot(self.phase_real, self.phase_imaginary) / max(self.weight, 1e-12),
            0.0,
            1.0,
        )


@dataclass(frozen=True)
class TemplateEvidence:
    direction: str
    frame_key: str
    speech_act: str
    predicate: str
    polarity: bool
    roles: tuple[str, ...]
    pattern: tuple[str, ...]
    hits: int
    weight: float


@dataclass(frozen=True)
class ReferenceEvidence:
    surface: str
    concept_type: str
    recency: int
    hits: int
    weight: float


@dataclass(frozen=True)
class LexemeLaw:
    surface: str
    concept: str
    evidence_count: int
    mass: float
    support: float
    coherence: float
    active: bool = True


@dataclass(frozen=True)
class FrameLaw:
    direction: str
    frame_key: str
    speech_act: str
    predicate: str
    polarity: bool
    roles: tuple[str, ...]
    pattern: tuple[str, ...]
    evidence_count: int
    mass: float
    support: float
    active: bool = True


@dataclass(frozen=True)
class ReferenceLaw:
    surface: str
    concept_type: str
    recency: int
    evidence_count: int
    mass: float
    support: float
    active: bool = True


@dataclass(frozen=True)
class CharacterSpanLaw:
    surface: str
    evidence_count: int
    support: float
    active: bool = True


@dataclass(frozen=True)
class EpisodeTrace:
    event_id: str
    tokens: tuple[str, ...]
    frame_keys: tuple[str, ...]
    salience: float
    support: float
    age: int


@dataclass(frozen=True)
class LanguageStimulus:
    mode: str
    event_id: str
    tokens: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()
    associations: tuple[AssociationObservation, ...] = ()
    templates: tuple[TemplateObservation, ...] = ()
    references: tuple[ReferenceObservation, ...] = ()
    salience: float = 1.0

    def validate(self) -> None:
        if self.mode not in {"observe", "anneal", "abstract", "idle"}:
            raise ValueError("Unknown language stimulus mode")
        if (
            not isinstance(self.event_id, str)
            or not self.event_id
            or len(self.event_id) > 200
        ):
            raise ValueError("event_id must contain 1 to 200 characters")
        if (
            isinstance(self.salience, bool)
            or not isinstance(self.salience, (int, float))
            or not math.isfinite(self.salience)
            or not 0.1 <= self.salience <= 2.0
        ):
            raise ValueError("salience must be finite and within [0.1, 2.0]")


@dataclass(frozen=True)
class LanguageState:
    stage: str
    association_evidence: tuple[AssociationEvidence, ...]
    template_evidence: tuple[TemplateEvidence, ...]
    reference_evidence: tuple[ReferenceEvidence, ...]
    lexeme_laws: tuple[LexemeLaw, ...]
    frame_laws: tuple[FrameLaw, ...]
    reference_laws: tuple[ReferenceLaw, ...]
    character_span_laws: tuple[CharacterSpanLaw, ...]
    traces: tuple[EpisodeTrace, ...]
    temperature: float
    energy: float
    phase_energy: float
    cumulative_phase_energy: float
    maximum_phase_energy: float
    accepted_improving_moves: int
    accepted_worse_moves: int
    observations: int
    forgotten: int
    conservation_excess: float
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
        return (
            sum(row.mass for row in self.lexeme_laws if row.active)
            + sum(row.mass for row in self.frame_laws if row.active)
            + sum(row.mass for row in self.reference_laws if row.active)
        )


@dataclass(frozen=True)
class ExecutionRecord:
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


def _counter_tuple(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(name), int(count)) for name, count in values.items()))


def _bump_counter(
    values: Sequence[tuple[str, int]], name: str
) -> tuple[tuple[str, int], ...]:
    counts = Counter(dict(values))
    counts[name] += 1
    return _counter_tuple(counts)


class UniverseLanguageKernel:
    """Sole replacement authority for the immutable language substrate."""

    def __init__(
        self,
        config: LanguageConfig | None = None,
        disabled: Iterable[Primitive] = (),
    ) -> None:
        self.config = config or LanguageConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self, stage: str) -> LanguageState:
        if stage not in {"word", "character"}:
            raise ValueError("stage must be 'word' or 'character'")
        return LanguageState(
            stage=stage,
            association_evidence=(),
            template_evidence=(),
            reference_evidence=(),
            lexeme_laws=(),
            frame_laws=(),
            reference_laws=(),
            character_span_laws=(),
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
            conservation_excess=0.0,
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
        self, state: LanguageState, primitive: Primitive, stimulus: LanguageStimulus
    ) -> LanguageState:
        stimulus.validate()
        if primitive in self.disabled:
            return state
        handlers = {
            Primitive.RADIATION: self._radiation,
            Primitive.DISSIPATION: self._dissipation,
            Primitive.GRAVITATION: self._gravitation,
            Primitive.ATTRACTION_REPULSION: self._attraction_repulsion,
            Primitive.NUCLEATION: self._nucleation,
            Primitive.CONSERVATION: self._conservation,
            Primitive.DECAY: self._decay,
        }
        transitioned = handlers[primitive](state, stimulus)
        digest = stable_hash(
            {
                "previous": state.transition_hash,
                "primitive": primitive.value,
                "event": stimulus.event_id,
                "mode": stimulus.mode,
                "laws": {
                    "lexeme": [
                        (row.surface, row.concept, round(row.support, 9))
                        for row in transitioned.lexeme_laws
                    ],
                    "frame": [
                        (row.direction, row.frame_key, row.pattern)
                        for row in transitioned.frame_laws
                    ],
                    "reference": [
                        (row.surface, row.concept_type, row.recency)
                        for row in transitioned.reference_laws
                    ],
                },
                "temperature": round(transitioned.temperature, 12),
                "energy": round(transitioned.energy, 12),
                "outcome": transitioned.last_outcome,
            }
        )
        return replace(
            transitioned,
            operator_counts=_bump_counter(
                transitioned.operator_counts, primitive.value
            ),
            transition_hash=digest,
            transitions=state.transitions + 1,
        )

    def _radiation(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        if stimulus.mode == "observe":
            trace = EpisodeTrace(
                event_id=stimulus.event_id,
                tokens=stimulus.tokens,
                frame_keys=tuple(sorted({row.frame.key for row in stimulus.templates})),
                salience=stimulus.salience,
                support=stimulus.salience,
                age=0,
            )
            traces = (*state.traces, trace)
            span_counts = Counter(
                {row.surface: row.evidence_count for row in state.character_span_laws}
            )
            if state.stage == "character":
                span_counts.update(stimulus.tokens)
            span_laws = tuple(
                CharacterSpanLaw(
                    surface=surface,
                    evidence_count=hits,
                    support=clamp(hits / self.config.character_span_min_hits, 0.0, 1.0),
                    active=hits >= self.config.character_span_min_hits,
                )
                for surface, hits in sorted(span_counts.items())
            )
            return replace(
                state,
                traces=traces,
                character_span_laws=span_laws,
                radiated_event=stimulus.event_id,
                observations=state.observations + 1,
                last_outcome="surface_pulse_radiated",
            )
        if stimulus.mode == "anneal":
            phase = self.config.phase_mix_strength * (
                2.0
                * deterministic_fraction(
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
        return replace(state, radiated_event=stimulus.event_id)

    def _gravitation(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        if stimulus.mode == "observe" and state.radiated_event == stimulus.event_id:
            associations = {
                (row.surface, row.concept): row for row in state.association_evidence
            }
            phase_energy = 0.0
            for observation in stimulus.associations:
                key = (observation.surface, observation.concept)
                existing = associations.get(key)
                weight = stimulus.salience * observation.weight
                angle = (
                    2.0
                    * math.pi
                    * observation.position
                    / max(1, observation.token_count)
                )
                interference = self.config.phase_mix_strength * (
                    deterministic_fraction(
                        self.config.chaos_seed,
                        stimulus.event_id,
                        observation.surface,
                        observation.concept,
                    )
                    - 0.5
                )
                phase_energy += abs(interference)
                real = weight * math.cos(angle + interference)
                imaginary = weight * math.sin(angle + interference)
                associations[key] = AssociationEvidence(
                    surface=observation.surface,
                    concept=observation.concept,
                    hits=(0 if existing is None else existing.hits) + 1,
                    weight=(0.0 if existing is None else existing.weight) + weight,
                    phase_real=(0.0 if existing is None else existing.phase_real)
                    + real,
                    phase_imaginary=(
                        0.0 if existing is None else existing.phase_imaginary
                    )
                    + imaginary,
                )

            templates = {
                (row.direction, row.frame_key, row.pattern): row
                for row in state.template_evidence
            }
            for observation in stimulus.templates:
                key = (
                    observation.direction,
                    observation.frame.key,
                    observation.pattern,
                )
                existing = templates.get(key)
                templates[key] = TemplateEvidence(
                    direction=observation.direction,
                    frame_key=observation.frame.key,
                    speech_act=observation.frame.speech_act,
                    predicate=observation.frame.predicate,
                    polarity=observation.frame.polarity,
                    roles=tuple(role for role, _ in observation.frame.roles),
                    pattern=observation.pattern,
                    hits=(0 if existing is None else existing.hits) + 1,
                    weight=(0.0 if existing is None else existing.weight)
                    + stimulus.salience * observation.weight,
                )

            references = {
                (row.surface, row.concept_type, row.recency): row
                for row in state.reference_evidence
            }
            for observation in stimulus.references:
                key = (
                    observation.surface,
                    observation.concept_type,
                    observation.recency,
                )
                existing = references.get(key)
                references[key] = ReferenceEvidence(
                    surface=observation.surface,
                    concept_type=observation.concept_type,
                    recency=observation.recency,
                    hits=(0 if existing is None else existing.hits) + 1,
                    weight=(0.0 if existing is None else existing.weight)
                    + stimulus.salience * observation.weight,
                )
            return replace(
                state,
                association_evidence=tuple(
                    associations[key] for key in sorted(associations)
                ),
                template_evidence=tuple(templates[key] for key in sorted(templates)),
                reference_evidence=tuple(references[key] for key in sorted(references)),
                gravitated_event=stimulus.event_id,
                phase_energy=phase_energy,
                cumulative_phase_energy=state.cumulative_phase_energy + phase_energy,
                maximum_phase_energy=max(state.maximum_phase_energy, phase_energy),
                last_outcome="grounding_attractor_formed",
            )
        if stimulus.mode == "anneal" and state.radiated_event == stimulus.event_id:
            return replace(
                state,
                gravitated_event=stimulus.event_id,
                last_outcome="anneal_attractor_formed",
            )
        return state

    def _attraction_repulsion(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        if state.gravitated_event != stimulus.event_id:
            return state
        if stimulus.mode == "anneal":
            energy = state.energy
            improving = state.accepted_improving_moves
            worse = state.accepted_worse_moves
            for trial in range(self.config.anneal_trials):
                fraction = deterministic_fraction(
                    self.config.chaos_seed,
                    stimulus.event_id,
                    state.transitions,
                    trial,
                )
                delta = (fraction - 0.47) * 0.08
                if delta <= 0.0:
                    energy = max(0.0, energy + delta)
                    improving += 1
                    continue
                probability = math.exp(-delta / max(state.temperature, 1e-9))
                draw = deterministic_fraction(
                    self.config.chaos_seed,
                    "accept",
                    stimulus.event_id,
                    trial,
                )
                if draw < probability:
                    energy += delta
                    worse += 1
            return replace(
                state,
                energy=energy,
                accepted_improving_moves=improving,
                accepted_worse_moves=worse,
                bound_event=stimulus.event_id,
                last_outcome="annealed_binding_candidates",
            )
        ambiguity = 0.0
        by_surface: defaultdict[str, list[AssociationEvidence]] = defaultdict(list)
        for row in state.association_evidence:
            by_surface[row.surface].append(row)
        for rows in by_surface.values():
            weights = sorted((row.weight for row in rows), reverse=True)
            if len(weights) > 1 and weights[0] > 0.0:
                ambiguity += weights[1] / weights[0]
        return replace(
            state,
            bound_event=stimulus.event_id,
            energy=ambiguity / max(1, len(by_surface)),
            last_outcome="compatible_roles_bound",
        )

    def _candidate_lexeme_laws(
        self, evidence: Sequence[AssociationEvidence]
    ) -> tuple[LexemeLaw, ...]:
        by_surface: defaultdict[str, list[AssociationEvidence]] = defaultdict(list)
        for row in evidence:
            by_surface[row.surface].append(row)
        candidates: list[LexemeLaw] = []
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
            second_weight = ranked[1].weight if len(ranked) > 1 else 0.0
            total_weight = sum(row.weight for row in rows)
            selectivity = top.weight / max(total_weight, 1e-12)
            if (
                top.hits < self.config.lexeme_min_hits
                or selectivity < self.config.lexeme_min_selectivity
                or top.weight + 1e-12 < second_weight * self.config.lexeme_margin
            ):
                continue
            candidates.append(
                LexemeLaw(
                    surface=surface,
                    concept=top.concept,
                    evidence_count=top.hits,
                    mass=top.weight,
                    support=clamp(selectivity, 0.0, 1.0),
                    coherence=round(top.coherence, 12),
                )
            )
        return tuple(sorted(candidates, key=lambda row: row.surface))

    def _candidate_frame_laws(
        self, evidence: Sequence[TemplateEvidence]
    ) -> tuple[FrameLaw, ...]:
        candidates = []
        for row in evidence:
            if row.hits < self.config.frame_min_hits:
                continue
            candidates.append(
                FrameLaw(
                    direction=row.direction,
                    frame_key=row.frame_key,
                    speech_act=row.speech_act,
                    predicate=row.predicate,
                    polarity=row.polarity,
                    roles=row.roles,
                    pattern=row.pattern,
                    evidence_count=row.hits,
                    mass=row.weight,
                    support=clamp(
                        row.hits / (self.config.frame_min_hits + 1.0), 0.0, 1.0
                    ),
                )
            )
        return tuple(
            sorted(
                candidates, key=lambda row: (row.direction, row.frame_key, row.pattern)
            )
        )

    def _candidate_reference_laws(
        self, evidence: Sequence[ReferenceEvidence]
    ) -> tuple[ReferenceLaw, ...]:
        return tuple(
            ReferenceLaw(
                surface=row.surface,
                concept_type=row.concept_type,
                recency=row.recency,
                evidence_count=row.hits,
                mass=row.weight,
                support=clamp(
                    row.hits / (self.config.reference_min_hits + 1.0), 0.0, 1.0
                ),
            )
            for row in sorted(
                evidence,
                key=lambda item: (item.surface, item.concept_type, item.recency),
            )
            if row.hits >= self.config.reference_min_hits
        )

    @staticmethod
    def _merge_lexeme_laws(
        existing: Sequence[LexemeLaw], candidates: Sequence[LexemeLaw]
    ) -> tuple[LexemeLaw, ...]:
        by_surface = {row.surface: row for row in existing if row.active}
        for candidate in candidates:
            current = by_surface.get(candidate.surface)
            if current is None or current.concept == candidate.concept:
                by_surface[candidate.surface] = candidate
        return tuple(by_surface[surface] for surface in sorted(by_surface))

    @staticmethod
    def _merge_frame_laws(
        existing: Sequence[FrameLaw], candidates: Sequence[FrameLaw]
    ) -> tuple[FrameLaw, ...]:
        by_key = {
            (row.direction, row.frame_key, row.pattern): row
            for row in existing
            if row.active
        }
        for candidate in candidates:
            by_key[(candidate.direction, candidate.frame_key, candidate.pattern)] = (
                candidate
            )
        return tuple(
            by_key[key]
            for key in sorted(by_key, key=lambda item: (item[0], item[1], item[2]))
        )

    @staticmethod
    def _merge_reference_laws(
        existing: Sequence[ReferenceLaw], candidates: Sequence[ReferenceLaw]
    ) -> tuple[ReferenceLaw, ...]:
        by_key = {
            (row.surface, row.concept_type): row for row in existing if row.active
        }
        for candidate in candidates:
            by_key[(candidate.surface, candidate.concept_type)] = candidate
        return tuple(by_key[key] for key in sorted(by_key))

    def _nucleation(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        if stimulus.mode not in {"observe", "abstract"}:
            return state
        if stimulus.mode == "observe" and state.bound_event != stimulus.event_id:
            return state
        lexemes = self._merge_lexeme_laws(
            state.lexeme_laws,
            self._candidate_lexeme_laws(state.association_evidence),
        )
        frames = self._merge_frame_laws(
            state.frame_laws,
            self._candidate_frame_laws(state.template_evidence),
        )
        references = self._merge_reference_laws(
            state.reference_laws,
            self._candidate_reference_laws(state.reference_evidence),
        )
        return replace(
            state,
            lexeme_laws=lexemes,
            frame_laws=frames,
            reference_laws=references,
            last_outcome="language_laws_nucleated",
        )

    def _conservation(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        concept_winners: dict[str, LexemeLaw] = {}
        for law in sorted(
            state.lexeme_laws,
            key=lambda row: (-row.support, -row.evidence_count, row.surface),
        ):
            if law.concept not in concept_winners:
                concept_winners[law.concept] = law
        lexemes = tuple(sorted(concept_winners.values(), key=lambda row: row.surface))

        frame_winners: dict[tuple[str, str, tuple[str, ...]], FrameLaw] = {}
        for law in sorted(
            state.frame_laws,
            key=lambda row: (-row.support, -row.evidence_count, row.pattern),
        ):
            frame_winners.setdefault((law.direction, law.frame_key, law.pattern), law)
        frames = tuple(
            sorted(
                frame_winners.values(),
                key=lambda row: (row.direction, row.frame_key),
            )
        )

        reference_winners: dict[tuple[str, str], ReferenceLaw] = {}
        for law in sorted(
            state.reference_laws,
            key=lambda row: (-row.support, -row.evidence_count, row.surface),
        ):
            reference_winners.setdefault((law.surface, law.concept_type), law)
        references = tuple(
            sorted(
                reference_winners.values(),
                key=lambda row: (row.surface, row.concept_type),
            )
        )
        mass = (
            sum(row.mass for row in lexemes)
            + sum(row.mass for row in frames)
            + sum(row.mass for row in references)
        )
        excess = max(0.0, mass - self.config.information_mass_budget)
        if excess > 0.0:
            scale = self.config.information_mass_budget / mass
            lexemes = tuple(replace(row, mass=row.mass * scale) for row in lexemes)
            frames = tuple(replace(row, mass=row.mass * scale) for row in frames)
            references = tuple(
                replace(row, mass=row.mass * scale) for row in references
            )
        return replace(
            state,
            lexeme_laws=lexemes,
            frame_laws=frames,
            reference_laws=references,
            conservation_excess=0.0 if excess > 0.0 else excess,
            conservation_applications=state.conservation_applications + 1,
            last_outcome="semantic_mass_conserved",
        )

    def _dissipation(
        self, state: LanguageState, stimulus: LanguageStimulus
    ) -> LanguageState:
        temperature = max(
            self.config.temperature_floor,
            state.temperature * self.config.cooling_rate,
        )
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
            temperature=temperature,
            last_outcome="transients_dissipated",
        )

    def _decay(self, state: LanguageState, stimulus: LanguageStimulus) -> LanguageState:
        if stimulus.mode == "abstract":
            forgotten = state.forgotten + len(state.traces)
            return replace(
                state,
                traces=(),
                association_evidence=(),
                template_evidence=(),
                reference_evidence=(),
                forgotten=forgotten,
                last_outcome="raw_episodes_coarse_grained",
            )
        kept = tuple(
            row for row in state.traces if row.support >= self.config.trace_expiration
        )
        return replace(
            state,
            traces=kept,
            forgotten=state.forgotten + len(state.traces) - len(kept),
            last_outcome="unsupported_traces_decayed",
        )


class LanguageRuntime:
    def __init__(
        self,
        stage: str,
        kernel: UniverseLanguageKernel | None = None,
        state: LanguageState | None = None,
        knowledge: AtomWikiGraph | None = None,
    ) -> None:
        self.kernel = kernel or UniverseLanguageKernel()
        self.knowledge = knowledge or build_language_graph()
        self.knowledge.assert_all_leaves_are_universe_primitives()
        self.state = state or self.kernel.initial_state(stage)
        if self.state.stage != stage:
            raise ValueError("runtime stage does not match restored state")
        self.records: list[ExecutionRecord] = []

    def execute(self, recipe: str, stimulus: LanguageStimulus) -> LanguageState:
        primitives = self.knowledge.expand(recipe)
        before = self.state.transition_hash
        for name in primitives:
            self.state = self.kernel.apply(self.state, Primitive(name), stimulus)
        self.records.append(
            ExecutionRecord(
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

    def observe(self, stimulus: LanguageStimulus) -> LanguageState:
        if stimulus.mode != "observe":
            raise ValueError("observe requires an observe stimulus")
        return self.execute("language_learn", stimulus)

    def anneal(self, event_id: str) -> LanguageState:
        return self.execute(
            "thermal_anneal",
            LanguageStimulus(mode="anneal", event_id=event_id),
        )

    def abstract(self, event_id: str = "language-abstract") -> LanguageState:
        return self.execute(
            "language_abstract",
            LanguageStimulus(mode="abstract", event_id=event_id),
        )


def config_with(base: LanguageConfig | None = None, **overrides: Any) -> LanguageConfig:
    values = asdict(base or LanguageConfig())
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f"Unknown language config values: {sorted(unknown)}")
    values.update(overrides)
    config = LanguageConfig(**values)
    config.validate()
    return config


def validate_world_state(world: Mapping[str, Any]) -> None:
    if set(world) != {"locations", "holders"}:
        raise ValueError("world must contain exactly locations and holders")
    if not isinstance(world["locations"], dict) or not isinstance(
        world["holders"], dict
    ):
        raise ValueError("world locations and holders must be objects")
    agents = set(world["locations"])
    if not agents or any(concept_kind(agent) != "agent" for agent in agents):
        raise ValueError("world locations must be keyed by agent concepts")
    for location in world["locations"].values():
        if concept_kind(str(location)) != "destination":
            raise ValueError("agent locations must be location concepts")
    if not world["holders"]:
        raise ValueError("world holders cannot be empty")
    for item, holder in world["holders"].items():
        if concept_kind(str(item)) != "patient":
            raise ValueError("world holders must be keyed by object concepts")
        if holder is not None and holder not in agents:
            raise ValueError("object holder must be null or a known agent")


def copy_world_state(world: Mapping[str, Any]) -> dict[str, Any]:
    validate_world_state(world)
    return {
        "locations": dict(sorted(world["locations"].items())),
        "holders": dict(sorted(world["holders"].items())),
    }


def frame_from_world_delta(row: Mapping[str, Any]) -> MeaningFrame | None:
    before = row["before"]
    after = row["after"]
    validate_world_state(before)
    validate_world_state(after)
    location_changes = [
        agent
        for agent in before["locations"]
        if before["locations"][agent] != after["locations"][agent]
    ]
    holder_changes = [
        item
        for item in before["holders"]
        if before["holders"][item] != after["holders"][item]
    ]
    if len(location_changes) == 1 and not holder_changes:
        agent = location_changes[0]
        return make_frame(
            "command",
            "MOVE",
            agent=agent,
            destination=str(after["locations"][agent]),
        )
    if len(holder_changes) == 1 and not location_changes:
        item = holder_changes[0]
        old_holder = before["holders"][item]
        new_holder = after["holders"][item]
        if old_holder is None and new_holder is not None:
            return make_frame("command", "TAKE", agent=str(new_holder), patient=item)
        if (
            old_holder is not None
            and new_holder is not None
            and old_holder != new_holder
        ):
            return make_frame(
                "command",
                "GIVE",
                agent=str(old_holder),
                patient=item,
                recipient=str(new_holder),
            )
    return None


def lexeme_maps(
    laws: Sequence[LexemeLaw],
) -> tuple[dict[str, str], dict[str, str]]:
    surface_to_concept = {row.surface: row.concept for row in laws if row.active}
    concept_to_surface = {}
    for row in sorted(
        (law for law in laws if law.active),
        key=lambda item: (-item.support, -item.evidence_count, item.surface),
    ):
        concept_to_surface.setdefault(row.concept, row.surface)
    return surface_to_concept, concept_to_surface


def _mapped_concepts(
    tokens: Sequence[str], surface_to_concept: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(
        surface_to_concept[token] for token in tokens if token in surface_to_concept
    )


def _context_concepts(
    context_text: str | None,
    stage: str,
    surface_to_concept: Mapping[str, str],
) -> dict[str, str]:
    if not context_text:
        return {}
    tokens, _ = pulses_for_stage(context_text, stage)
    context: dict[str, str] = {}
    for concept in _mapped_concepts(tokens, surface_to_concept):
        kind = concept_kind(concept)
        context[kind] = concept
    return context


def frame_from_grounding(
    row: Mapping[str, Any],
    stage: str,
    surface_to_concept: Mapping[str, str],
) -> tuple[MeaningFrame | None, dict[str, str]]:
    changed = frame_from_world_delta(row)
    context = _context_concepts(row.get("context_text"), stage, surface_to_concept)
    if changed is not None:
        return changed, context

    tokens, _ = pulses_for_stage(str(row["text"]), stage)
    current = _mapped_concepts(tokens, surface_to_concept)
    answer_text = row.get("answer_text")
    answer_tokens = ()
    if answer_text:
        answer_tokens, _ = pulses_for_stage(str(answer_text), stage)
    answer_concepts = _mapped_concepts(answer_tokens, surface_to_concept)

    agents = [value for value in current if concept_kind(value) == "agent"]
    objects = [value for value in current if concept_kind(value) == "patient"]
    locations = [value for value in current if concept_kind(value) == "destination"]
    if "they" in tokens and "agent" in context:
        agents.append(context["agent"])
    if "it" in tokens and "patient" in context:
        objects.append(context["patient"])

    answer_agents = [
        value for value in answer_concepts if concept_kind(value) == "agent"
    ]
    answer_objects = [
        value for value in answer_concepts if concept_kind(value) == "patient"
    ]
    answer_locations = [
        value for value in answer_concepts if concept_kind(value) == "destination"
    ]

    if answer_text is not None and len(answer_tokens) == 1 and not answer_concepts:
        if agents and objects:
            return make_frame(
                "question", "HAS_QUERY", agent=agents[0], patient=objects[0]
            ), context
        if agents and locations:
            return make_frame(
                "question",
                "AT_QUERY",
                agent=agents[0],
                destination=locations[0],
            ), context
    if answer_locations and agents:
        return make_frame("question", "WHERE", agent=agents[0]), context
    if answer_objects and agents:
        return make_frame("question", "WHAT_HAS", agent=agents[0]), context
    if answer_agents and objects and not agents:
        return make_frame("question", "WHO_HAS", patient=objects[0]), context
    if answer_text is None and agents and locations:
        frame = make_frame("assertion", "AT", agent=agents[0], destination=locations[0])
        if world_satisfies_frame(frame, row["before"]):
            return frame, context
    if answer_text is None and agents and objects:
        frame = make_frame("assertion", "HAS", agent=agents[0], patient=objects[0])
        if world_satisfies_frame(frame, row["before"]):
            return frame, context
    return None, context


def answer_frame_from_text(
    answer_text: str | None,
    stage: str,
    surface_to_concept: Mapping[str, str],
) -> MeaningFrame | None:
    if answer_text is None:
        return None
    if answer_text == "yes":
        return make_frame("answer", "YES")
    if answer_text == "no":
        return make_frame("answer", "NO")
    tokens, _ = pulses_for_stage(answer_text, stage)
    concepts = _mapped_concepts(tokens, surface_to_concept)
    agents = [value for value in concepts if concept_kind(value) == "agent"]
    objects = [value for value in concepts if concept_kind(value) == "patient"]
    locations = [value for value in concepts if concept_kind(value) == "destination"]
    if agents and locations:
        return make_frame("assertion", "AT", agent=agents[0], destination=locations[0])
    if agents and objects:
        return make_frame("assertion", "HAS", agent=agents[0], patient=objects[0])
    return None


def answer_frame_from_grounding(
    row: Mapping[str, Any],
    question_frame: MeaningFrame,
    stage: str,
    surface_to_concept: Mapping[str, str],
) -> MeaningFrame | None:
    answer_text = row.get("answer_text")
    if answer_text is None:
        return None
    if question_frame.speech_act == "question":
        consequence = apply_meaning_to_world(question_frame, row["before"])
        answer_payload = consequence.get("answer_frame")
        if isinstance(answer_payload, Mapping):
            return MeaningFrame.from_payload(answer_payload)
    return answer_frame_from_text(
        str(answer_text),
        stage,
        surface_to_concept,
    )


def world_satisfies_frame(frame: MeaningFrame, world: Mapping[str, Any]) -> bool:
    validate_world_state(world)
    roles = frame.role_map
    if frame.predicate == "AT":
        return world["locations"].get(roles["agent"]) == roles["destination"]
    if frame.predicate == "HAS":
        return world["holders"].get(roles["patient"]) == roles["agent"]
    return True


def _pattern_for_frame(
    tokens: Sequence[str],
    frame: MeaningFrame,
    surface_to_concept: Mapping[str, str],
    context: Mapping[str, str],
) -> tuple[str, ...] | None:
    roles = frame.role_map
    concept_to_role = {concept: role for role, concept in roles.items()}
    pattern: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        concept = surface_to_concept.get(token)
        role = concept_to_role.get(concept) if concept is not None else None
        if role is None and token == "they" and "agent" in roles:
            if context.get("agent") == roles["agent"]:
                role = "agent"
        if role is None and token == "it" and "patient" in roles:
            if context.get("patient") == roles["patient"]:
                role = "patient"
        if role is None:
            pattern.append(token)
        else:
            pattern.append("{" + role + "}")
            seen.add(role)
    if seen != set(roles):
        return None
    return tuple(pattern)


def _cross_product_associations(
    tokens: Sequence[str], frame: MeaningFrame
) -> tuple[AssociationObservation, ...]:
    concepts = tuple(dict.fromkeys(frame.role_map.values()))
    return tuple(
        AssociationObservation(
            surface=token,
            concept=concept,
            position=position,
            token_count=len(tokens),
        )
        for position, token in enumerate(tokens)
        for concept in concepts
    )


def _direct_associations(
    tokens: Sequence[str],
    frame: MeaningFrame,
    surface_to_concept: Mapping[str, str],
) -> tuple[AssociationObservation, ...]:
    allowed = set(frame.role_map.values())
    return tuple(
        AssociationObservation(
            surface=token,
            concept=surface_to_concept[token],
            position=position,
            token_count=len(tokens),
            weight=1.35,
        )
        for position, token in enumerate(tokens)
        if surface_to_concept.get(token) in allowed
    )


def _stimulus_for_row(
    row: Mapping[str, Any],
    stage: str,
    frame: MeaningFrame,
    surface_to_concept: Mapping[str, str],
    *,
    bootstrap: bool,
    epoch: int,
) -> LanguageStimulus:
    tokens, characters = pulses_for_stage(str(row["text"]), stage)
    context = _context_concepts(row.get("context_text"), stage, surface_to_concept)
    associations = (
        _cross_product_associations(tokens, frame)
        if bootstrap
        else _direct_associations(tokens, frame, surface_to_concept)
    )
    templates: list[TemplateObservation] = []
    references: list[ReferenceObservation] = []
    if not bootstrap:
        pattern = _pattern_for_frame(tokens, frame, surface_to_concept, context)
        if pattern is not None:
            templates.append(TemplateObservation("parse", frame, pattern))
            if not row.get("context_text"):
                templates.append(TemplateObservation("speak", frame, pattern))
        paraphrase = row.get("paraphrase_text")
        if paraphrase:
            paraphrase_tokens, _ = pulses_for_stage(str(paraphrase), stage)
            paraphrase_pattern = _pattern_for_frame(
                paraphrase_tokens,
                frame,
                surface_to_concept,
                {},
            )
            if paraphrase_pattern is not None:
                templates.extend(
                    (
                        TemplateObservation("parse", frame, paraphrase_pattern),
                        TemplateObservation("speak", frame, paraphrase_pattern),
                    )
                )
        for surface, kind in (("they", "agent"), ("it", "patient")):
            if surface in tokens and kind in context:
                references.append(ReferenceObservation(surface, kind, 0))

        answer_frame = answer_frame_from_grounding(
            row,
            frame,
            stage,
            surface_to_concept,
        )
        if answer_frame is not None:
            answer_tokens, _ = pulses_for_stage(str(row["answer_text"]), stage)
            answer_pattern = _pattern_for_frame(
                answer_tokens, answer_frame, surface_to_concept, context
            )
            if answer_pattern is not None:
                templates.extend(
                    (
                        TemplateObservation("parse", answer_frame, answer_pattern),
                        TemplateObservation("speak", answer_frame, answer_pattern),
                    )
                )
    return LanguageStimulus(
        mode="observe",
        event_id=f"{stage}-{epoch}-{row['case_id']}",
        tokens=tokens,
        characters=characters,
        associations=associations,
        templates=tuple(templates),
        references=tuple(references),
        salience=float(row["salience"]),
    )


def language_training_snapshot(runtime: LanguageRuntime, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "temperature": runtime.state.temperature,
        "energy": runtime.state.energy,
        "phase_energy": runtime.state.phase_energy,
        "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
        "lexeme_laws": len(runtime.state.lexeme_laws),
        "frame_laws": len(runtime.state.frame_laws),
        "reference_laws": len(runtime.state.reference_laws),
        "character_span_laws": len(runtime.state.character_span_laws),
        "raw_traces": len(runtime.state.traces),
        "transition_hash": runtime.state.transition_hash,
    }


def train_language_field(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    config: LanguageConfig | None = None,
    disabled: Iterable[Primitive] = (),
) -> tuple[LanguageRuntime, list[dict[str, Any]], dict[str, Any]]:
    runtime = LanguageRuntime(
        stage,
        kernel=UniverseLanguageKernel(config or LanguageConfig(), disabled=disabled),
    )
    history = [language_training_snapshot(runtime, "initial")]
    action_rows = [row for row in rows if frame_from_world_delta(row) is not None]
    for epoch in range(3):
        for row in action_rows:
            frame = frame_from_world_delta(row)
            if frame is None:
                raise AssertionError("Action curriculum row lost its world delta")
            runtime.observe(
                _stimulus_for_row(
                    row,
                    stage,
                    frame,
                    {},
                    bootstrap=True,
                    epoch=epoch,
                )
            )
        runtime.anneal(f"{stage}-bootstrap-anneal-{epoch}")
        history.append(language_training_snapshot(runtime, f"bootstrap-{epoch}"))

    unresolved: set[str] = set()
    for epoch in range(4):
        surface_to_concept, _ = lexeme_maps(runtime.state.lexeme_laws)
        for row in rows:
            frame, _ = frame_from_grounding(row, stage, surface_to_concept)
            if frame is None:
                unresolved.add(str(row["case_id"]))
                continue
            unresolved.discard(str(row["case_id"]))
            runtime.observe(
                _stimulus_for_row(
                    row,
                    stage,
                    frame,
                    surface_to_concept,
                    bootstrap=False,
                    epoch=epoch + 3,
                )
            )
        runtime.anneal(f"{stage}-grounded-anneal-{epoch}")
        history.append(language_training_snapshot(runtime, f"grounded-{epoch}"))
    diagnostics = {
        "training_rows": len(rows),
        "action_curriculum_rows": len(action_rows),
        "unresolved_case_ids": sorted(unresolved),
        "stage": stage,
    }
    return runtime, history, diagnostics


def adapt_language_field(
    runtime: LanguageRuntime,
    rows: Sequence[Mapping[str, Any]],
    *,
    transient_rows: Sequence[Mapping[str, Any]] = (),
    bootstrap_epochs: int = 2,
    grounded_epochs: int = 3,
    adaptation_id: str = "transfer",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("adaptation rows cannot be empty")
    for name, value in (
        ("bootstrap_epochs", bootstrap_epochs),
        ("grounded_epochs", grounded_epochs),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 10
        ):
            raise ValueError(f"{name} must be an integer within [1, 10]")
    if not isinstance(adaptation_id, str) or not adaptation_id.strip():
        raise ValueError("adaptation_id must be non-empty text")
    if runtime.state.traces or runtime.state.association_evidence:
        raise ValueError("adaptation requires an abstracted or restored language model")
    if runtime.state.template_evidence or runtime.state.reference_evidence:
        raise ValueError("adaptation cannot start with retained raw evidence")
    if not runtime.state.lexeme_laws or not runtime.state.frame_laws:
        raise ValueError("adaptation requires persistent lexical and frame laws")

    stage = runtime.state.stage
    base_lexemes = {
        (row.surface, row.concept) for row in runtime.state.lexeme_laws if row.active
    }
    base_frames = {
        (row.direction, row.frame_key, row.pattern)
        for row in runtime.state.frame_laws
        if row.active
    }
    history = [language_training_snapshot(runtime, f"{adaptation_id}-initial")]

    for index, row in enumerate(transient_rows):
        frame = frame_from_world_delta(row)
        if frame is None:
            raise ValueError(
                "transient adaptation rows require an observable world delta"
            )
        runtime.observe(
            _stimulus_for_row(
                row,
                stage,
                frame,
                {},
                bootstrap=True,
                epoch=-(index + 1),
            )
        )
    if transient_rows:
        runtime.execute(
            "forget",
            LanguageStimulus(
                mode="idle",
                event_id=f"{adaptation_id}-transient-forget",
            ),
        )
        history.append(
            language_training_snapshot(runtime, f"{adaptation_id}-transient")
        )

    action_rows = [row for row in rows if frame_from_world_delta(row) is not None]
    if not action_rows:
        raise ValueError("adaptation requires grounded action demonstrations")
    for epoch in range(bootstrap_epochs):
        for row in action_rows:
            frame = frame_from_world_delta(row)
            if frame is None:
                raise AssertionError("adaptation action lost its world delta")
            runtime.observe(
                _stimulus_for_row(
                    row,
                    stage,
                    frame,
                    {},
                    bootstrap=True,
                    epoch=epoch,
                )
            )
        runtime.anneal(f"{adaptation_id}-{stage}-bootstrap-anneal-{epoch}")
        history.append(
            language_training_snapshot(runtime, f"{adaptation_id}-bootstrap-{epoch}")
        )

    unresolved: set[str] = set()
    for epoch in range(grounded_epochs):
        surface_to_concept, _ = lexeme_maps(runtime.state.lexeme_laws)
        for row in rows:
            frame, _ = frame_from_grounding(row, stage, surface_to_concept)
            if frame is None:
                unresolved.add(str(row["case_id"]))
                continue
            unresolved.discard(str(row["case_id"]))
            runtime.observe(
                _stimulus_for_row(
                    row,
                    stage,
                    frame,
                    surface_to_concept,
                    bootstrap=False,
                    epoch=epoch + bootstrap_epochs,
                )
            )
        runtime.anneal(f"{adaptation_id}-{stage}-grounded-anneal-{epoch}")
        history.append(
            language_training_snapshot(runtime, f"{adaptation_id}-grounded-{epoch}")
        )

    adapted_lexemes = {
        (row.surface, row.concept) for row in runtime.state.lexeme_laws if row.active
    }
    adapted_frames = {
        (row.direction, row.frame_key, row.pattern)
        for row in runtime.state.frame_laws
        if row.active
    }
    diagnostics = {
        "adaptation_id": adaptation_id,
        "stage": stage,
        "grounding_rows": len(rows),
        "transient_rows": len(transient_rows),
        "bootstrap_epochs": bootstrap_epochs,
        "grounded_epochs": grounded_epochs,
        "unresolved_case_ids": sorted(unresolved),
        "retained_base_lexemes": len(base_lexemes & adapted_lexemes),
        "base_lexemes": len(base_lexemes),
        "retained_base_frames": len(base_frames & adapted_frames),
        "base_frames": len(base_frames),
        "new_lexemes": len(adapted_lexemes - base_lexemes),
    }
    return history, diagnostics


def _reference_lookup(
    state: LanguageState,
) -> dict[tuple[str, str], ReferenceLaw]:
    return {
        (row.surface, row.concept_type): row
        for row in state.reference_laws
        if row.active
    }


def _active_context_concept(
    context: Mapping[str, Mapping[str, Any]], kind: str
) -> str | None:
    row = context.get(kind)
    if not isinstance(row, Mapping):
        return None
    ttl = row.get("ttl")
    concept = row.get("concept")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, int)
        or ttl <= 0
        or not isinstance(concept, str)
    ):
        return None
    if concept_kind(concept) != kind:
        return None
    return concept


def _match_frame_law(
    law: FrameLaw,
    tokens: Sequence[str],
    surface_to_concept: Mapping[str, str],
    references: Mapping[tuple[str, str], ReferenceLaw],
    context: Mapping[str, Mapping[str, Any]],
) -> tuple[MeaningFrame, float] | None:
    if law.direction != "parse" or len(law.pattern) != len(tokens):
        return None
    bindings: dict[str, str] = {}
    strengths: list[float] = [law.support]
    for expected, token in zip(law.pattern, tokens, strict=True):
        if not (expected.startswith("{") and expected.endswith("}")):
            if token != expected:
                return None
            continue
        role = expected[1:-1]
        if role not in ROLE_KINDS:
            return None
        required_kind = ROLE_KINDS[role]
        concept = surface_to_concept.get(token)
        if concept is not None and concept_kind(concept) == required_kind:
            strengths.append(1.0)
        else:
            reference = references.get((token, required_kind))
            concept = _active_context_concept(context, required_kind)
            if reference is None or concept is None:
                return None
            strengths.append(reference.support)
        if role in bindings and bindings[role] != concept:
            return None
        bindings[role] = concept
    if set(bindings) != set(law.roles):
        return None
    frame = make_frame(
        law.speech_act,
        law.predicate,
        polarity=law.polarity,
        **bindings,
    )
    return frame, round(sum(strengths) / len(strengths), 12)


def interpret_text(
    runtime: LanguageRuntime,
    text: str,
    context: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    tokens, _ = pulses_for_stage(text, runtime.state.stage)
    surface_to_concept, _ = lexeme_maps(runtime.state.lexeme_laws)
    references = _reference_lookup(runtime.state)
    context = context or {}
    matches: list[tuple[MeaningFrame, float, FrameLaw]] = []
    for law in runtime.state.frame_laws:
        match = _match_frame_law(law, tokens, surface_to_concept, references, context)
        if match is not None:
            matches.append((match[0], match[1], law))
    if not matches:
        unknown = sorted(
            {
                token
                for token in tokens
                if token not in surface_to_concept
                and not any(token in law.pattern for law in runtime.state.frame_laws)
                and not any(
                    token == row.surface for row in runtime.state.reference_laws
                )
            }
        )
        return {
            "status": "unknown_token" if unknown else "unknown_structure",
            "unknown_tokens": unknown,
            "meaning": None,
            "confidence": 0.0,
            "matched_pattern": None,
        }
    matches.sort(
        key=lambda row: (
            -row[1],
            -row[2].evidence_count,
            row[0].key,
            row[2].pattern,
        )
    )
    best = matches[0]
    distinct = {stable_hash(row[0].payload()) for row in matches if row[1] == best[1]}
    if len(distinct) > 1:
        return {
            "status": "ambiguous_meaning",
            "unknown_tokens": [],
            "meaning": None,
            "confidence": 0.0,
            "matched_pattern": None,
        }
    return {
        "status": "understood",
        "unknown_tokens": [],
        "meaning": best[0].payload(),
        "confidence": round(best[1], 12),
        "matched_pattern": list(best[2].pattern),
    }


def generate_text(runtime: LanguageRuntime, frame: MeaningFrame) -> dict[str, Any]:
    frame.validate()
    _, concept_to_surface = lexeme_maps(runtime.state.lexeme_laws)
    candidates = [
        row
        for row in runtime.state.frame_laws
        if row.active and row.direction == "speak" and row.frame_key == frame.key
    ]
    if not candidates:
        return {
            "status": "unknown_generation",
            "text": None,
            "confidence": 0.0,
            "semantic_mass_total": len(frame.roles),
            "semantic_mass_remaining": len(frame.roles),
        }
    candidates.sort(key=lambda row: (-row.support, -row.evidence_count, row.pattern))
    law = candidates[0]
    roles = frame.role_map
    output: list[str] = []
    expressed: set[str] = set()
    for piece in law.pattern:
        if piece.startswith("{") and piece.endswith("}"):
            role = piece[1:-1]
            concept = roles.get(role)
            surface = concept_to_surface.get(concept) if concept is not None else None
            if surface is None:
                return {
                    "status": "unknown_generation",
                    "text": None,
                    "confidence": 0.0,
                    "semantic_mass_total": len(frame.roles),
                    "semantic_mass_remaining": len(frame.roles) - len(expressed),
                }
            output.append(surface)
            expressed.add(role)
        else:
            output.append(piece)
    remaining = set(roles) - expressed
    if remaining:
        return {
            "status": "conservation_failure",
            "text": None,
            "confidence": 0.0,
            "semantic_mass_total": len(frame.roles),
            "semantic_mass_remaining": len(remaining),
        }
    return {
        "status": "generated",
        "text": " ".join(output),
        "confidence": round(law.support, 12),
        "semantic_mass_total": len(frame.roles),
        "semantic_mass_remaining": 0,
    }


def apply_meaning_to_world(
    frame: MeaningFrame, world: Mapping[str, Any]
) -> dict[str, Any]:
    frame.validate()
    updated = copy_world_state(world)
    roles = frame.role_map
    answer: MeaningFrame | None = None
    action: dict[str, Any] | None = None
    status = "understood"
    if frame.predicate == "MOVE":
        updated["locations"][roles["agent"]] = roles["destination"]
        action = {
            "predicate": "MOVE",
            "agent": roles["agent"],
            "destination": roles["destination"],
        }
    elif frame.predicate == "TAKE":
        holder = updated["holders"].get(roles["patient"])
        if holder not in {None, roles["agent"]}:
            status = "invalid_world_action"
        else:
            updated["holders"][roles["patient"]] = roles["agent"]
            action = {
                "predicate": "TAKE",
                "agent": roles["agent"],
                "patient": roles["patient"],
            }
    elif frame.predicate == "GIVE":
        if updated["holders"].get(roles["patient"]) != roles["agent"]:
            status = "invalid_world_action"
        else:
            updated["holders"][roles["patient"]] = roles["recipient"]
            action = {
                "predicate": "GIVE",
                "agent": roles["agent"],
                "patient": roles["patient"],
                "recipient": roles["recipient"],
            }
    elif frame.predicate in {"AT", "HAS"}:
        status = (
            "understood" if world_satisfies_frame(frame, updated) else "contradiction"
        )
    elif frame.predicate == "WHERE":
        answer = make_frame(
            "assertion",
            "AT",
            agent=roles["agent"],
            destination=str(updated["locations"][roles["agent"]]),
        )
    elif frame.predicate == "WHAT_HAS":
        items = sorted(
            item
            for item, holder in updated["holders"].items()
            if holder == roles["agent"]
        )
        if not items:
            status = "unknown_world_answer"
        else:
            answer = make_frame(
                "assertion",
                "HAS",
                agent=roles["agent"],
                patient=items[0],
            )
    elif frame.predicate == "WHO_HAS":
        holder = updated["holders"].get(roles["patient"])
        if holder is None:
            status = "unknown_world_answer"
        else:
            answer = make_frame(
                "assertion",
                "HAS",
                agent=str(holder),
                patient=roles["patient"],
            )
    elif frame.predicate == "HAS_QUERY":
        predicate = (
            "YES"
            if updated["holders"].get(roles["patient"]) == roles["agent"]
            else "NO"
        )
        answer = make_frame("answer", predicate)
    elif frame.predicate == "AT_QUERY":
        predicate = (
            "YES"
            if updated["locations"].get(roles["agent"]) == roles["destination"]
            else "NO"
        )
        answer = make_frame("answer", predicate)
    return {
        "status": status,
        "world_after": updated,
        "action": action,
        "answer_frame": None if answer is None else answer.payload(),
    }


def _age_context(context: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    aged: dict[str, dict[str, Any]] = {}
    for kind, row in context.items():
        if not isinstance(row, Mapping):
            continue
        ttl = row.get("ttl")
        concept = row.get("concept")
        if isinstance(ttl, int) and not isinstance(ttl, bool) and ttl > 1:
            aged[kind] = {"concept": concept, "ttl": ttl - 1}
    return aged


def update_context(
    context: Mapping[str, Mapping[str, Any]],
    frames: Sequence[MeaningFrame],
    ttl: int,
) -> dict[str, dict[str, Any]]:
    updated = _age_context(context)
    for frame in frames:
        for _, concept in frame.roles:
            updated[concept_kind(concept)] = {"concept": concept, "ttl": ttl}
    return dict(sorted(updated.items()))


def interact_text(
    runtime: LanguageRuntime,
    text: str,
    world: Mapping[str, Any],
    context: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    context = context or {}
    interpreted = interpret_text(runtime, text, context)
    if interpreted["meaning"] is None:
        return {
            **interpreted,
            "action": None,
            "answer": None,
            "world_after": copy_world_state(world),
            "context_after": _age_context(context),
        }
    frame = MeaningFrame.from_payload(interpreted["meaning"])
    applied = apply_meaning_to_world(frame, world)
    answer_frame = (
        None
        if applied["answer_frame"] is None
        else MeaningFrame.from_payload(applied["answer_frame"])
    )
    answer = None
    if answer_frame is not None:
        answer = generate_text(runtime, answer_frame)
    frames = [frame]
    if answer_frame is not None:
        frames.append(answer_frame)
    return {
        "status": applied["status"],
        "unknown_tokens": interpreted["unknown_tokens"],
        "meaning": frame.payload(),
        "confidence": interpreted["confidence"],
        "matched_pattern": interpreted["matched_pattern"],
        "action": applied["action"],
        "answer": answer,
        "world_after": applied["world_after"],
        "context_after": update_context(
            context, frames, runtime.kernel.config.context_ttl
        ),
    }


def evaluate_language_rows(
    runtime: LanguageRuntime,
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = []
    frame_correct = 0
    world_correct = 0
    answer_correct = 0
    generation_correct = 0
    grammar_valid = 0
    reference_cases = 0
    reference_correct = 0
    family_scores: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        expected = truth[str(row["case_id"])]
        context: dict[str, dict[str, Any]] = {}
        if row.get("context_text"):
            context_result = interact_text(
                runtime, str(row["context_text"]), row["before"], context
            )
            context = context_result["context_after"]
        result = interact_text(runtime, str(row["text"]), row["before"], context)
        expected_frame = MeaningFrame.from_payload(expected["frame"])
        parsed_correct = result["meaning"] == expected_frame.payload()
        actual_world = result["world_after"]
        expected_world = expected["expected_after"]
        changed = frame_from_world_delta(row) is not None
        state_correct = actual_world == expected_world if changed else True
        expected_answer = expected.get("expected_answer")
        actual_answer = None
        if result.get("answer") and result["answer"].get("status") == "generated":
            actual_answer = result["answer"]["text"]
        response_correct = (
            actual_answer == expected_answer if expected_answer is not None else True
        )
        generated = generate_text(runtime, expected_frame)
        generated_parse = None
        if generated["status"] == "generated":
            generated_parse = interpret_text(runtime, generated["text"], {})
        semantic_roundtrip = bool(
            generated_parse and generated_parse["meaning"] == expected_frame.payload()
        )
        valid_grammar = generated["status"] == "generated" and semantic_roundtrip
        frame_correct += int(parsed_correct)
        world_correct += int(state_correct)
        answer_correct += int(response_correct)
        generation_correct += int(semantic_roundtrip)
        grammar_valid += int(valid_grammar)
        if row.get("context_text"):
            reference_cases += 1
            reference_correct += int(parsed_correct)
        combined = parsed_correct and state_correct and response_correct
        family_scores[str(expected["family"])].append(float(combined))
        predictions.append(
            {
                "case_id": row["case_id"],
                "family": expected["family"],
                "expected_meaning": expected_frame.payload(),
                "predicted_meaning": result["meaning"],
                "frame_correct": parsed_correct,
                "world_correct": state_correct,
                "answer_correct": response_correct,
                "generated": generated["text"],
                "semantic_roundtrip": semantic_roundtrip,
                "status": result["status"],
            }
        )
    total = len(rows)
    return {
        "cases": total,
        "frame_accuracy": frame_correct / total if total else 0.0,
        "world_accuracy": world_correct / total if total else 0.0,
        "answer_accuracy": answer_correct / total if total else 0.0,
        "grounded_accuracy": sum(
            row["frame_correct"] and row["world_correct"] and row["answer_correct"]
            for row in predictions
        )
        / total
        if total
        else 0.0,
        "generation_roundtrip_accuracy": generation_correct / total if total else 0.0,
        "grammar_validity": grammar_valid / total if total else 0.0,
        "reference_cases": reference_cases,
        "reference_accuracy": reference_correct / reference_cases
        if reference_cases
        else 1.0,
        "family_accuracy": {
            family: sum(values) / len(values)
            for family, values in sorted(family_scores.items())
        },
        "predictions": predictions,
    }


def character_span_f1(
    runtime: LanguageRuntime, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if runtime.state.stage != "character":
        return {"true_positive": 0, "predicted": 0, "gold": 0, "f1": 1.0}
    known = {row.surface for row in runtime.state.character_span_laws if row.active}
    true_positive = 0
    predicted = 0
    gold = 0
    for row in rows:
        for field in ("text", "context_text", "answer_text"):
            value = row.get(field)
            if not value:
                continue
            gold_spans = tokenize_word_pulses(str(value))
            character_spans = spans_from_character_pulses(character_pulses(str(value)))
            gold += len(gold_spans)
            predicted += sum(span in known for span in character_spans)
            true_positive += sum(
                left == right and left in known
                for left, right in zip(gold_spans, character_spans, strict=True)
            )
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _strict_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    keys = set(value)
    if keys != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - keys)}, "
            f"unknown={sorted(keys - expected)}"
        )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def language_model_payload(runtime: LanguageRuntime) -> dict[str, Any]:
    if runtime.state.traces:
        raise ValueError("Language model export requires raw episodes to be abstracted")
    if (
        runtime.state.association_evidence
        or runtime.state.template_evidence
        or runtime.state.reference_evidence
    ):
        raise ValueError("Language model export cannot retain raw evidence tables")
    payload: dict[str, Any] = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "model_type": "atom_grounded_language_field",
        "language_runtime": ATOM_LANGUAGE_RUNTIME,
        "stage": runtime.state.stage,
        "lexeme_laws": [asdict(row) for row in runtime.state.lexeme_laws],
        "frame_laws": [asdict(row) for row in runtime.state.frame_laws],
        "reference_laws": [asdict(row) for row in runtime.state.reference_laws],
        "character_span_laws": [
            asdict(row) for row in runtime.state.character_span_laws if row.active
        ],
        "temperature": round(runtime.state.temperature, 12),
        "transition_hash": runtime.state.transition_hash,
        "knowledge_graph_hash": stable_hash(runtime.knowledge.manifest()),
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "config": asdict(runtime.kernel.config),
        "raw_episode_count": 0,
        "raw_evidence_count": 0,
    }
    payload["model_hash"] = stable_hash(payload)
    return payload


def _law_bool(value: Any, name: str) -> bool:
    if value is not True:
        raise ValueError(f"{name} must be true")
    return True


def runtime_from_language_model(payload: Mapping[str, Any]) -> LanguageRuntime:
    required = {
        "schema_version",
        "model_type",
        "language_runtime",
        "stage",
        "lexeme_laws",
        "frame_laws",
        "reference_laws",
        "character_span_laws",
        "temperature",
        "transition_hash",
        "knowledge_graph_hash",
        "wiki_runtime",
        "rag_runtime",
        "config",
        "raw_episode_count",
        "raw_evidence_count",
        "model_hash",
    }
    _strict_keys(payload, required, "language model")
    if payload["schema_version"] != LANGUAGE_MODEL_SCHEMA:
        raise ValueError("Unsupported language model schema")
    if payload["model_type"] != "atom_grounded_language_field":
        raise ValueError("Unexpected language model type")
    if payload["language_runtime"] != ATOM_LANGUAGE_RUNTIME:
        raise ValueError("Language runtime marker mismatch")
    check = dict(payload)
    supplied_hash = check.pop("model_hash")
    if not isinstance(supplied_hash, str) or supplied_hash != stable_hash(check):
        raise ValueError("Language model hash mismatch")
    if payload["wiki_runtime"] != ATOM_WIKI_GRAPH_RUNTIME:
        raise ValueError("Wiki runtime marker mismatch")
    if payload["rag_runtime"] != ATOM_RAG_RUNTIME:
        raise ValueError("RAG runtime marker mismatch")
    if payload["raw_episode_count"] != 0 or payload["raw_evidence_count"] != 0:
        raise ValueError("Exported language model cannot retain raw observations")
    if not isinstance(payload["config"], dict):
        raise ValueError("Language model config must be an object")
    config = LanguageConfig(**payload["config"])
    config.validate()
    stage = payload["stage"]
    if stage not in {"word", "character"}:
        raise ValueError("Language model stage is invalid")

    if not isinstance(payload["lexeme_laws"], list) or not payload["lexeme_laws"]:
        raise ValueError("Language model requires lexeme laws")
    lexemes = []
    for index, row in enumerate(payload["lexeme_laws"]):
        if not isinstance(row, dict):
            raise ValueError("Lexeme laws must be objects")
        _strict_keys(
            row,
            {
                "surface",
                "concept",
                "evidence_count",
                "mass",
                "support",
                "coherence",
                "active",
            },
            f"lexeme_law[{index}]",
        )
        surface = row["surface"]
        concept = row["concept"]
        if not isinstance(surface, str) or not tokenize_word_pulses(surface):
            raise ValueError("Lexeme surface must be text")
        if not isinstance(concept, str):
            raise ValueError("Lexeme concept must be text")
        concept_kind(concept)
        evidence_count = row["evidence_count"]
        if (
            isinstance(evidence_count, bool)
            or not isinstance(evidence_count, int)
            or evidence_count <= 0
        ):
            raise ValueError("Lexeme evidence_count must be positive")
        support = _finite_number(row["support"], "lexeme support")
        coherence = _finite_number(row["coherence"], "lexeme coherence")
        if not 0.0 <= support <= 1.0 or not 0.0 <= coherence <= 1.0:
            raise ValueError("Lexeme support and coherence must be within [0, 1]")
        lexemes.append(
            LexemeLaw(
                surface=surface,
                concept=concept,
                evidence_count=evidence_count,
                mass=_finite_number(row["mass"], "lexeme mass"),
                support=support,
                coherence=coherence,
                active=_law_bool(row["active"], "lexeme active"),
            )
        )
    if len({row.surface for row in lexemes}) != len(lexemes):
        raise ValueError("Lexeme surfaces must be unique")
    if len({row.concept for row in lexemes}) != len(lexemes):
        raise ValueError("Conservation requires one lexeme per concept")

    if not isinstance(payload["frame_laws"], list) or not payload["frame_laws"]:
        raise ValueError("Language model requires frame laws")
    frames = []
    for index, row in enumerate(payload["frame_laws"]):
        if not isinstance(row, dict):
            raise ValueError("Frame laws must be objects")
        _strict_keys(
            row,
            {
                "direction",
                "frame_key",
                "speech_act",
                "predicate",
                "polarity",
                "roles",
                "pattern",
                "evidence_count",
                "mass",
                "support",
                "active",
            },
            f"frame_law[{index}]",
        )
        if row["direction"] not in {"parse", "speak"}:
            raise ValueError("Frame direction is invalid")
        if not isinstance(row["roles"], list) or not isinstance(row["pattern"], list):
            raise ValueError("Frame roles and pattern must be lists")
        roles = tuple(str(value) for value in row["roles"])
        pattern = tuple(str(value) for value in row["pattern"])
        if set(roles) != set(PREDICATE_ROLES.get(str(row["predicate"]), ())):
            raise ValueError("Frame role basis is invalid")
        if any(not value for value in pattern):
            raise ValueError("Frame pattern cannot contain empty values")
        support = _finite_number(row["support"], "frame support")
        if not 0.0 <= support <= 1.0:
            raise ValueError("Frame support must be within [0, 1]")
        evidence_count = row["evidence_count"]
        if (
            isinstance(evidence_count, bool)
            or not isinstance(evidence_count, int)
            or evidence_count <= 0
        ):
            raise ValueError("Frame evidence_count must be positive")
        frame = FrameLaw(
            direction=str(row["direction"]),
            frame_key=str(row["frame_key"]),
            speech_act=str(row["speech_act"]),
            predicate=str(row["predicate"]),
            polarity=row["polarity"],
            roles=roles,
            pattern=pattern,
            evidence_count=evidence_count,
            mass=_finite_number(row["mass"], "frame mass"),
            support=support,
            active=_law_bool(row["active"], "frame active"),
        )
        expected_key = make_frame(
            frame.speech_act,
            frame.predicate,
            polarity=frame.polarity,
            **{
                role: (
                    "agent-0"
                    if ROLE_KINDS[role] == "agent"
                    else "object-0"
                    if ROLE_KINDS[role] == "patient"
                    else "location-0"
                )
                for role in frame.roles
            },
        ).key
        if frame.frame_key != expected_key:
            raise ValueError("Frame key does not match its semantic fields")
        frames.append(frame)

    references = []
    if not isinstance(payload["reference_laws"], list):
        raise ValueError("Reference laws must be a list")
    for row in payload["reference_laws"]:
        _strict_keys(
            row,
            {
                "surface",
                "concept_type",
                "recency",
                "evidence_count",
                "mass",
                "support",
                "active",
            },
            "reference law",
        )
        if row["concept_type"] not in {"agent", "patient"}:
            raise ValueError("Reference concept type is invalid")
        references.append(
            ReferenceLaw(
                surface=str(row["surface"]),
                concept_type=str(row["concept_type"]),
                recency=int(row["recency"]),
                evidence_count=int(row["evidence_count"]),
                mass=_finite_number(row["mass"], "reference mass"),
                support=_finite_number(row["support"], "reference support"),
                active=_law_bool(row["active"], "reference active"),
            )
        )

    spans = []
    if not isinstance(payload["character_span_laws"], list):
        raise ValueError("Character span laws must be a list")
    for row in payload["character_span_laws"]:
        _strict_keys(
            row,
            {"surface", "evidence_count", "support", "active"},
            "character span law",
        )
        spans.append(
            CharacterSpanLaw(
                surface=str(row["surface"]),
                evidence_count=int(row["evidence_count"]),
                support=_finite_number(row["support"], "span support"),
                active=_law_bool(row["active"], "span active"),
            )
        )
    if stage == "character" and not spans:
        raise ValueError("Character-stage model requires span laws")

    temperature = _finite_number(payload["temperature"], "temperature")
    if not config.temperature_floor <= temperature <= config.initial_temperature:
        raise ValueError("Restored temperature is outside configured bounds")
    state = LanguageState(
        stage=stage,
        association_evidence=(),
        template_evidence=(),
        reference_evidence=(),
        lexeme_laws=tuple(lexemes),
        frame_laws=tuple(frames),
        reference_laws=tuple(references),
        character_span_laws=tuple(spans),
        traces=(),
        temperature=temperature,
        energy=0.0,
        phase_energy=0.0,
        cumulative_phase_energy=0.0,
        maximum_phase_energy=0.0,
        accepted_improving_moves=0,
        accepted_worse_moves=0,
        observations=0,
        forgotten=0,
        conservation_excess=0.0,
        conservation_applications=0,
        radiated_event="",
        gravitated_event="",
        bound_event="",
        operator_counts=(),
        outcome_counts=(),
        last_outcome="restored",
        transition_hash=str(payload["transition_hash"]),
        transitions=0,
    )
    runtime = LanguageRuntime(
        stage,
        kernel=UniverseLanguageKernel(config),
        state=state,
    )
    if stable_hash(runtime.knowledge.manifest()) != payload["knowledge_graph_hash"]:
        raise ValueError("Language knowledge graph hash mismatch")
    return runtime


def validate_language_request(
    payload: Mapping[str, Any], runtime: LanguageRuntime
) -> None:
    _strict_keys(
        payload,
        {"schema_version", "request_id", "stage", "world", "turns"},
        "language request",
    )
    if payload["schema_version"] != LANGUAGE_MODEL_SCHEMA:
        raise ValueError("Unsupported language request schema")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be non-empty text")
    if payload["stage"] != runtime.state.stage:
        raise ValueError("request stage does not match model stage")
    if not isinstance(payload["world"], dict):
        raise ValueError("request world must be an object")
    validate_world_state(payload["world"])
    turns = payload["turns"]
    if not isinstance(turns, list) or not 1 <= len(turns) <= 64:
        raise ValueError("turns must contain 1 to 64 rows")
    ids: set[str] = set()
    for index, row in enumerate(turns):
        if not isinstance(row, dict):
            raise ValueError("turn rows must be objects")
        mode = row.get("mode")
        expected = (
            {"turn_id", "mode", "text"}
            if mode == "interact"
            else {"turn_id", "mode", "meaning"}
            if mode == "generate"
            else set()
        )
        if not expected:
            raise ValueError("turn mode must be interact or generate")
        _strict_keys(row, expected, f"turn[{index}]")
        turn_id = row["turn_id"]
        if not isinstance(turn_id, str) or not turn_id or turn_id in ids:
            raise ValueError("turn_id values must be unique non-empty text")
        ids.add(turn_id)
        if mode == "interact":
            pulses_for_stage(row["text"], runtime.state.stage)
        else:
            if not isinstance(row["meaning"], dict):
                raise ValueError("generate meaning must be an object")
            MeaningFrame.from_payload(row["meaning"])


def run_language_workflow(
    model_path: Path, request_path: Path, response_path: Path
) -> dict[str, Any]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime = runtime_from_language_model(model)
    validate_language_request(request, runtime)
    world = copy_world_state(request["world"])
    context: dict[str, dict[str, Any]] = {}
    results = []
    for row in request["turns"]:
        if row["mode"] == "interact":
            result = interact_text(runtime, row["text"], world, context)
            world = result["world_after"]
            context = result["context_after"]
            results.append(
                {
                    "turn_id": row["turn_id"],
                    "mode": "interact",
                    **result,
                }
            )
        else:
            frame = MeaningFrame.from_payload(row["meaning"])
            generated = generate_text(runtime, frame)
            results.append(
                {
                    "turn_id": row["turn_id"],
                    "mode": "generate",
                    "meaning": frame.payload(),
                    **generated,
                }
            )
    response = {
        "schema_version": LANGUAGE_MODEL_SCHEMA,
        "request_id": request["request_id"],
        "model_hash": model["model_hash"],
        "stage": runtime.state.stage,
        "turns": results,
        "world_after": world,
        "context_after": context,
        "runtime": {
            "language_runtime": ATOM_LANGUAGE_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "transition_hash": runtime.state.transition_hash,
        },
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return response


def architecture_audit(source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UniverseLanguageKernel"
    )
    replace_calls_outside_kernel = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name == "replace" and not kernel.lineno <= node.lineno <= kernel.end_lineno:
            replace_calls_outside_kernel.append(node.lineno)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    graph = build_language_graph()
    language_nodes = {
        "ground",
        "lexical_nucleation",
        "role_bind",
        "understand",
        "resolve_reference",
        "speak",
        "language_learn",
        "language_abstract",
    }
    checks = {
        "seven_universe_primitives": {row.value for row in Primitive}
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "only_kernel_replaces_state": not replace_calls_outside_kernel,
        "no_neural_framework_import": not (
            imported_roots & {"torch", "tensorflow", "jax", "keras"}
        ),
        "runtime_does_not_import_evaluator_dataset": "atom_language_dataset"
        not in imported_roots,
        "language_graph_nodes_present": language_nodes <= set(graph.node_names),
        "language_graph_resolves_to_seven_primitives": all(
            set(graph.expand(name)) <= set(UNIVERSE_PRIMITIVE_NAMES)
            and bool(graph.expand(name))
            for name in language_nodes
        ),
        "wiki_and_rag_runtime_wired": "ATOM_WIKI_GRAPH_RUNTIME" in source
        and "ATOM_RAG_RUNTIME" in source,
        "strict_serialized_workflow_wired": "validate_language_request" in source
        and "run_language_workflow" in source,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "replace_calls_outside_kernel": replace_calls_outside_kernel,
        "resolved_language_recipes": {
            name: list(graph.expand(name)) for name in sorted(language_nodes)
        },
    }


def run_language_field_self_tests() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    graph = build_language_graph()
    graph.assert_all_leaves_are_universe_primitives()
    checks["knowledge_graph_language_compositions_resolve"] = all(
        graph.expand(name)
        for name in (
            "ground",
            "lexical_nucleation",
            "role_bind",
            "understand",
            "resolve_reference",
            "speak",
        )
    )
    checks["architecture_mutation_boundary"] = architecture_audit()["passed"]
    checks["word_pulses_are_deterministic"] = tokenize_word_pulses(
        "Move Mira to Pond!"
    ) == ("move", "mira", "to", "pond")
    characters = character_pulses("mira holds key")
    checks["character_spans_nucleate_from_boundary_pulses"] = (
        spans_from_character_pulses(characters) == ("mira", "holds", "key")
    )
    character_runtime = LanguageRuntime("character")
    for index in range(2):
        character_runtime.observe(
            LanguageStimulus(
                mode="observe",
                event_id=f"character-span-self-test-{index}",
                tokens=("mira", "holds", "key"),
                characters=characters,
            )
        )
    checks["character_span_evidence_accumulates_across_observations"] = {
        row.surface for row in character_runtime.state.character_span_laws if row.active
    } == {"mira", "holds", "key"}
    frame = make_frame("command", "MOVE", agent="agent-0", destination="location-1")
    world = {
        "locations": {
            "agent-0": "location-0",
            "agent-1": "location-1",
        },
        "holders": {"object-0": None},
    }
    applied = apply_meaning_to_world(frame, world)
    checks["meaning_frame_executes_against_world"] = (
        applied["world_after"]["locations"]["agent-0"] == "location-1"
        and world["locations"]["agent-0"] == "location-0"
    )
    state = UniverseLanguageKernel().initial_state("word")
    frozen = False
    try:
        state.temperature = 0.0  # type: ignore[misc]
    except Exception as error:
        frozen = error.__class__.__name__ == "FrozenInstanceError"
    checks["language_state_is_immutable"] = frozen
    malformed_rejected = False
    try:
        MeaningFrame.from_payload(
            {
                "speech_act": "command",
                "predicate": "MOVE",
                "polarity": True,
                "roles": {"agent": "agent-0"},
            }
        )
    except ValueError:
        malformed_rejected = True
    checks["malformed_meaning_fails_closed"] = malformed_rejected
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }
