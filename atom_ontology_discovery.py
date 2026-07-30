"""Alias-invariant ontology and executable law discovery.

This learner sees opaque relation tables and opaque entity identifiers.  It
derives type and relation atoms from graph position, nullability, and overlap;
then learns lexemes and simultaneous effect programs over those atoms.  Raw
relation aliases are local bindings and are never serialized into the model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_ontology_dataset import build_ontology_discovery_program
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    AtomWikiGraph,
    build_language_graph,
    retrieve_atom_context,
)


ONTOLOGY_MODEL_SCHEMA = 1
ONTOLOGY_DISCOVERY_SEED = 7_071_344
ATOM_ONTOLOGY_RUNTIME = "atom-emergent-ontology-v1"
_RELATION_ALIAS = re.compile(r"[a-z][a-z0-9]{1,15}")
_ENTITY_ID = re.compile(r"n[0-9]{2,5}")


class OntologyPrimitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


def ontology_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ontology_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        json.dumps([seed, *parts], sort_keys=True, default=str).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def ontology_clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ontology_tokens(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or len(text) > 256:
        raise ValueError("text must contain 1 to 256 characters")
    tokens = tuple(re.findall(r"[a-z0-9]+", text.lower()))
    if not tokens:
        raise ValueError("text does not contain any surface pulses")
    return tokens


@dataclass(frozen=True)
class OntologyTypeAtom:
    type_id: str
    roles: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {"type_id": self.type_id, "roles": list(self.roles)}


@dataclass(frozen=True)
class OntologyRelationAtom:
    relation_id: str
    domain_type: str
    range_type: str
    nullable: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveredOntology:
    signature: str
    types: tuple[OntologyTypeAtom, ...]
    relations: tuple[OntologyRelationAtom, ...]
    alias_to_relation: Mapping[str, str] = field(compare=False, repr=False)
    entity_types: Mapping[str, str] = field(compare=False, repr=False)

    def payload(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "types": [row.payload() for row in self.types],
            "relations": [row.payload() for row in self.relations],
        }

    @property
    def relation_to_alias(self) -> dict[str, str]:
        return {relation: alias for alias, relation in self.alias_to_relation.items()}


def _type_atom(roles: Sequence[str]) -> OntologyTypeAtom:
    normalized = tuple(sorted(str(role) for role in roles))
    return OntologyTypeAtom(
        type_id="type-" + ontology_hash({"roles": list(normalized)})[:12],
        roles=normalized,
    )


def _relation_atom(
    domain_type: str,
    range_type: str,
    nullable: bool,
) -> OntologyRelationAtom:
    core = {
        "domain_type": domain_type,
        "range_type": range_type,
        "nullable": nullable,
    }
    return OntologyRelationAtom(
        relation_id="relation-" + ontology_hash(core)[:12],
        **core,
    )


def discover_ontology(world: Mapping[str, Any]) -> DiscoveredOntology:
    """Infer a stable structural schema without using relation or type names."""

    if not isinstance(world, dict) or len(world) != 2:
        raise ValueError("world must contain exactly two opaque relation tables")
    normalized: dict[str, dict[str, str | None]] = {}
    for alias, table in world.items():
        if not isinstance(alias, str) or _RELATION_ALIAS.fullmatch(alias) is None:
            raise ValueError("relation aliases must be short opaque identifiers")
        if not isinstance(table, dict) or not table or len(table) > 128:
            raise ValueError("each relation must be a non-empty bounded object")
        clean: dict[str, str | None] = {}
        for key, value in table.items():
            if not isinstance(key, str) or _ENTITY_ID.fullmatch(key) is None:
                raise ValueError("relation keys must be opaque entity identifiers")
            if value is not None and (
                not isinstance(value, str) or _ENTITY_ID.fullmatch(value) is None
            ):
                raise ValueError("relation values must be opaque entities or null")
            clean[key] = value
        normalized[alias] = clean

    total_aliases = [
        alias
        for alias, table in normalized.items()
        if all(v is not None for v in table.values())
    ]
    null_bearing_aliases = [
        alias
        for alias, table in normalized.items()
        if any(v is None for v in table.values())
    ]
    if len(total_aliases) != 1 or len(null_bearing_aliases) != 1:
        raise ValueError("world must expose one total and one null-bearing relation")
    total_alias = total_aliases[0]
    null_alias = null_bearing_aliases[0]
    total_table = normalized[total_alias]
    null_table = normalized[null_alias]
    total_keys = set(total_table)
    total_values = {str(value) for value in total_table.values()}
    null_keys = set(null_table)
    null_values = {str(value) for value in null_table.values() if value is not None}
    if null_values - total_keys:
        raise ValueError("non-null bridge values must inhabit the total key position")
    if total_keys & total_values or total_keys & null_keys or total_values & null_keys:
        raise ValueError("structural entity positions must be disjoint")

    bridge_type = _type_atom(("total:key", "null-bearing:value"))
    terminal_type = _type_atom(("total:value",))
    source_type = _type_atom(("null-bearing:key",))
    total_relation = _relation_atom(
        bridge_type.type_id,
        terminal_type.type_id,
        False,
    )
    null_relation = _relation_atom(
        source_type.type_id,
        bridge_type.type_id,
        True,
    )
    types = tuple(
        sorted((bridge_type, terminal_type, source_type), key=lambda row: row.type_id)
    )
    relations = tuple(
        sorted((total_relation, null_relation), key=lambda row: row.relation_id)
    )
    core = {
        "types": [row.payload() for row in types],
        "relations": [row.payload() for row in relations],
    }
    entity_types = {
        **{entity: bridge_type.type_id for entity in total_keys},
        **{entity: terminal_type.type_id for entity in total_values},
        **{entity: source_type.type_id for entity in null_keys},
    }
    return DiscoveredOntology(
        signature=ontology_hash(core),
        types=types,
        relations=relations,
        alias_to_relation={
            total_alias: total_relation.relation_id,
            null_alias: null_relation.relation_id,
        },
        entity_types=dict(sorted(entity_types.items())),
    )


def copy_opaque_world(world: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    discover_ontology(world)
    return {
        alias: dict(sorted(table.items())) for alias, table in sorted(world.items())
    }


def validate_ontology_row(row: Mapping[str, Any]) -> None:
    expected = {"case_id", "text", "before", "after", "salience"}
    if set(row) != expected:
        raise ValueError(f"ontology row fields must be {sorted(expected)}")
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise ValueError("case_id must be non-empty text")
    ontology_tokens(str(row["text"]))
    before_schema = discover_ontology(row["before"])
    after_schema = discover_ontology(row["after"])
    if before_schema.signature != after_schema.signature:
        raise ValueError("before and after worlds must share a structural ontology")
    if set(row["before"]) != set(row["after"]):
        raise ValueError("before and after relation aliases must match")
    for alias in row["before"]:
        if set(row["before"][alias]) != set(row["after"][alias]):
            raise ValueError("before and after relation keys must match")
    salience = row["salience"]
    if (
        isinstance(salience, bool)
        or not isinstance(salience, (int, float))
        or not math.isfinite(float(salience))
        or not 0.1 <= float(salience) <= 2.0
    ):
        raise ValueError("salience must be finite and within [0.1, 2.0]")
    if row["before"] == row["after"]:
        raise ValueError("ontology row must change the world")


def ontology_delta_participants(row: Mapping[str, Any]) -> tuple[str, ...]:
    validate_ontology_row(row)
    participants: set[str] = set()
    for alias, table in row["before"].items():
        for key, old in table.items():
            new = row["after"][alias][key]
            if old == new:
                continue
            participants.add(str(key))
            for value in (old, new):
                if value is not None:
                    participants.add(str(value))
    return tuple(sorted(participants))


@dataclass(frozen=True)
class OntologyConfig:
    initial_temperature: float = 1.45
    temperature_floor: float = 0.22
    cooling_rate: float = 0.94
    phase_mix_strength: float = 0.052
    surface_min_hits: int = 2
    surface_min_support: float = 0.70
    surface_margin: float = 1.50
    law_min_hits: int = 2
    anneal_trials: int = 20
    chaos_seed: int = ONTOLOGY_DISCOVERY_SEED

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
        for name in ("initial_temperature", "temperature_floor"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_floor > self.initial_temperature:
            raise ValueError("temperature floor cannot exceed initial temperature")
        for name in ("cooling_rate", "phase_mix_strength", "surface_min_support"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not 1.0 <= self.surface_margin <= 10.0:
            raise ValueError("surface_margin must be within [1, 10]")


@dataclass(frozen=True)
class SurfaceLaw:
    surface: str
    entity: str
    evidence_count: int
    mass: float
    support: float
    coherence: float

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyEffectAtom:
    relation_id: str
    key_slot: int
    before: str
    after: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OntologyLaw:
    law_id: str
    signature: str
    pattern: tuple[str, ...]
    slot_types: tuple[str, ...]
    effects: tuple[OntologyEffectAtom, ...]
    evidence_count: int
    mass: float
    support: float

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
        }


@dataclass(frozen=True)
class LawObservation:
    signature: str
    pattern: tuple[str, ...]
    slot_types: tuple[str, ...]
    effects: tuple[OntologyEffectAtom, ...]


@dataclass
class OntologyTrainingState:
    ontology: DiscoveredOntology | None = None
    token_exposure: Counter[str] = field(default_factory=Counter)
    association_hits: Counter[tuple[str, str]] = field(default_factory=Counter)
    association_mass: defaultdict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    bound_surfaces: dict[str, str] = field(default_factory=dict)
    surface_laws: dict[str, SurfaceLaw] = field(default_factory=dict)
    law_hits: Counter[str] = field(default_factory=Counter)
    law_mass: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    law_templates: dict[str, LawObservation] = field(default_factory=dict)
    bound_laws: set[str] = field(default_factory=set)
    laws: dict[str, OntologyLaw] = field(default_factory=dict)
    traces: list[str] = field(default_factory=list)
    temperature: float = 1.45
    energy: float = 0.0
    phase_energy: float = 0.0
    cumulative_phase_energy: float = 0.0
    maximum_phase_energy: float = 0.0
    accepted_improving_moves: int = 0
    accepted_worse_moves: int = 0
    observations: int = 0
    forgotten: int = 0
    conservation_applications: int = 0
    operator_counts: Counter[str] = field(default_factory=Counter)
    outcome_counts: Counter[str] = field(default_factory=Counter)
    transition_hash: str = "0" * 64
    transitions: int = 0
    pending_energy: float | None = None

    @property
    def raw_evidence_count(self) -> int:
        return (
            len(self.token_exposure)
            + len(self.association_hits)
            + len(self.law_hits)
            + len(self.traces)
        )


class UniverseOntologyKernel:
    """Sole mutation boundary for learned ontology and law state."""

    def __init__(
        self,
        config: OntologyConfig | None = None,
        disabled: Iterable[OntologyPrimitive] = (),
    ) -> None:
        self.config = config or OntologyConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self) -> OntologyTrainingState:
        return OntologyTrainingState(temperature=self.config.initial_temperature)

    def apply(
        self,
        state: OntologyTrainingState,
        primitive: OntologyPrimitive,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if primitive in self.disabled:
            return
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty text")
        handlers = {
            OntologyPrimitive.RADIATION: self._radiation,
            OntologyPrimitive.DISSIPATION: self._dissipation,
            OntologyPrimitive.GRAVITATION: self._gravitation,
            OntologyPrimitive.ATTRACTION_REPULSION: self._attraction_repulsion,
            OntologyPrimitive.NUCLEATION: self._nucleation,
            OntologyPrimitive.CONSERVATION: self._conservation,
            OntologyPrimitive.DECAY: self._decay,
        }
        before = state.transition_hash
        handlers[primitive](state, mode, event_id, payload or {})
        state.operator_counts[primitive.value] += 1
        outcome = f"{mode}:{primitive.value}"
        state.outcome_counts[outcome] += 1
        state.transitions += 1
        state.transition_hash = ontology_hash(
            {
                "previous": before,
                "primitive": primitive.value,
                "mode": mode,
                "event": event_id,
                "surface_count": len(state.surface_laws),
                "law_count": len(state.laws),
                "temperature": round(state.temperature, 12),
                "energy": round(state.energy, 12),
            }
        )

    def _radiation(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "lexical_observe":
            tokens = tuple(str(token) for token in payload["tokens"])
            state.token_exposure.update(tokens)
            state.traces.append(event_id)
            state.observations += 1
        elif mode == "law_observe":
            state.traces.append(event_id)
        elif mode == "anneal":
            phase = self.config.phase_mix_strength * (
                2.0
                * ontology_fraction(
                    self.config.chaos_seed,
                    event_id,
                    state.transitions,
                    "radiation",
                )
                - 1.0
            )
            state.phase_energy = abs(phase)
            state.cumulative_phase_energy += abs(phase)
            state.maximum_phase_energy = max(state.maximum_phase_energy, abs(phase))

    def _gravitation(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "lexical_observe":
            salience = float(payload["salience"])
            for surface, entity in payload["associations"]:
                key = (str(surface), str(entity))
                noise = self.config.phase_mix_strength * (
                    ontology_fraction(
                        self.config.chaos_seed,
                        event_id,
                        surface,
                        entity,
                    )
                    - 0.5
                )
                mass = salience * (1.0 + noise)
                state.association_hits[key] += 1
                state.association_mass[key] += mass
                state.cumulative_phase_energy += abs(noise)
                state.maximum_phase_energy = max(state.maximum_phase_energy, abs(noise))
        elif mode == "law_observe":
            observation = payload["observation"]
            if not isinstance(observation, LawObservation):
                raise ValueError("law observation payload is invalid")
            existing = state.law_templates.get(observation.signature)
            if existing is not None and existing != observation:
                raise ValueError("one law signature received conflicting programs")
            state.law_templates[observation.signature] = observation
            state.law_hits[observation.signature] += 1
            noise = self.config.phase_mix_strength * (
                ontology_fraction(
                    self.config.chaos_seed,
                    event_id,
                    observation.signature,
                )
                - 0.5
            )
            state.law_mass[observation.signature] += 1.0 + noise
            state.cumulative_phase_energy += abs(noise)
            state.maximum_phase_energy = max(state.maximum_phase_energy, abs(noise))
        elif mode == "anneal":
            proposal = self.config.phase_mix_strength * (
                2.0
                * ontology_fraction(
                    self.config.chaos_seed,
                    event_id,
                    state.transitions,
                    "proposal",
                )
                - 1.0
            )
            state.pending_energy = abs(state.energy + proposal)

    def _attraction_repulsion(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "lexical_finalize":
            candidates: defaultdict[str, list[tuple[float, float, int, str]]] = (
                defaultdict(list)
            )
            for (surface, entity), hits in state.association_hits.items():
                exposure = state.token_exposure.get(surface, 0)
                if exposure == 0:
                    continue
                support = hits / exposure
                candidates[surface].append(
                    (support, state.association_mass[(surface, entity)], hits, entity)
                )
            proposed: list[tuple[float, float, str, str]] = []
            for surface, rows in candidates.items():
                ranked = sorted(rows, reverse=True)
                best_support, best_mass, best_hits, best_entity = ranked[0]
                runner_mass = ranked[1][1] if len(ranked) > 1 else 0.0
                margin = best_mass / max(runner_mass, 1e-12)
                if (
                    best_hits >= self.config.surface_min_hits
                    and best_support >= self.config.surface_min_support
                    and margin >= self.config.surface_margin
                ):
                    proposed.append((best_support, best_mass, surface, best_entity))
            state.bound_surfaces.clear()
            claimed: set[str] = set()
            for _, _, surface, entity in sorted(proposed, reverse=True):
                if entity not in claimed:
                    state.bound_surfaces[surface] = entity
                    claimed.add(entity)
        elif mode == "law_finalize":
            state.bound_laws = {
                signature
                for signature, hits in state.law_hits.items()
                if hits >= self.config.law_min_hits and signature in state.law_templates
            }
        elif mode == "anneal" and state.pending_energy is not None:
            delta = state.pending_energy - state.energy
            threshold = math.exp(-max(delta, 0.0) / max(state.temperature, 1e-12))
            draw = ontology_fraction(
                self.config.chaos_seed,
                event_id,
                state.transitions,
                "accept",
            )
            if delta <= 0.0 or draw < threshold:
                state.energy = state.pending_energy
                if delta <= 0.0:
                    state.accepted_improving_moves += 1
                else:
                    state.accepted_worse_moves += 1
            state.pending_energy = None

    def _nucleation(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "ontology":
            schema = payload.get("ontology")
            if not isinstance(schema, DiscoveredOntology):
                raise ValueError("ontology nucleation requires a discovered schema")
            state.ontology = schema
        elif mode == "lexical_finalize":
            laws: dict[str, SurfaceLaw] = {}
            for surface, entity in sorted(state.bound_surfaces.items()):
                hits = state.association_hits[(surface, entity)]
                exposure = state.token_exposure[surface]
                mass = state.association_mass[(surface, entity)]
                laws[surface] = SurfaceLaw(
                    surface=surface,
                    entity=entity,
                    evidence_count=hits,
                    mass=mass,
                    support=hits / exposure,
                    coherence=ontology_clamp(mass / max(hits, 1), 0.0, 1.0),
                )
            state.surface_laws = laws
        elif mode == "law_finalize":
            laws: dict[str, OntologyLaw] = {}
            total = sum(state.law_hits.values())
            for signature in sorted(state.bound_laws):
                observation = state.law_templates[signature]
                hits = state.law_hits[signature]
                laws[signature] = OntologyLaw(
                    law_id="law-" + signature[:16],
                    signature=signature,
                    pattern=observation.pattern,
                    slot_types=observation.slot_types,
                    effects=observation.effects,
                    evidence_count=hits,
                    mass=state.law_mass[signature],
                    support=hits / max(total, 1),
                )
            state.laws = laws

    def _conservation(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        state.conservation_applications += 1

    def _dissipation(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        state.temperature = max(
            self.config.temperature_floor,
            state.temperature * self.config.cooling_rate,
        )
        if mode == "forget":
            state.forgotten += (
                len(state.token_exposure)
                + len(state.association_hits)
                + len(state.law_hits)
            )
            state.token_exposure.clear()
            state.association_hits.clear()
            state.association_mass.clear()
            state.law_hits.clear()
            state.law_mass.clear()
            state.law_templates.clear()
            state.bound_surfaces.clear()
            state.bound_laws.clear()

    def _decay(
        self,
        state: OntologyTrainingState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "forget":
            state.forgotten += len(state.traces)
            state.traces.clear()


class OntologyRuntime:
    def __init__(
        self,
        kernel: UniverseOntologyKernel,
        state: OntologyTrainingState,
        reference_ontology: DiscoveredOntology,
        knowledge: AtomWikiGraph | None = None,
    ) -> None:
        self.kernel = kernel
        self.state = state
        self.reference_ontology = reference_ontology
        self.knowledge = knowledge or build_language_graph()

    @property
    def surface_laws(self) -> tuple[SurfaceLaw, ...]:
        return tuple(
            self.state.surface_laws[key] for key in sorted(self.state.surface_laws)
        )

    @property
    def transition_laws(self) -> tuple[OntologyLaw, ...]:
        return tuple(self.state.laws[key] for key in sorted(self.state.laws))

    def retrieve(self, query: str) -> list[dict[str, object]]:
        return retrieve_atom_context(self.knowledge, query, limit=5)


def _apply_recipe(
    runtime: OntologyRuntime,
    recipe: str,
    mode: str,
    event_id: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    for primitive_name in runtime.knowledge.expand(recipe):
        runtime.kernel.apply(
            runtime.state,
            OntologyPrimitive(primitive_name),
            mode,
            event_id,
            payload,
        )


def _surface_map(runtime: OntologyRuntime) -> dict[str, str]:
    return {law.surface: law.entity for law in runtime.surface_laws}


def _surface_reverse(runtime: OntologyRuntime) -> dict[str, str]:
    return {law.entity: law.surface for law in runtime.surface_laws}


def lexical_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    validate_ontology_row(row)
    tokens = ontology_tokens(str(row["text"]))
    participants = ontology_delta_participants(row)
    return {
        "tokens": tokens,
        "associations": tuple(
            (surface, entity) for surface in tokens for entity in participants
        ),
        "salience": float(row["salience"]),
    }


def _slot_expression(value: str, slot_map: Mapping[str, int]) -> str | None:
    index = slot_map.get(value)
    return None if index is None else f"slot:{index}"


def _encode_before_expression(
    value: str | None,
    slot_map: Mapping[str, int],
    ontology: DiscoveredOntology,
) -> str:
    if value is None:
        return "none"
    slot = _slot_expression(value, slot_map)
    if slot is not None:
        return slot
    type_id = ontology.entity_types.get(value)
    if type_id is None:
        raise ValueError("before value has no structural type")
    return f"any:{type_id}"


def _encode_after_expression(
    value: str | None,
    slot_map: Mapping[str, int],
    before: Mapping[str, Mapping[str, str | None]],
    ontology: DiscoveredOntology,
) -> str:
    if value is None:
        return "none"
    slot = _slot_expression(value, slot_map)
    if slot is not None:
        return slot
    for alias, table in sorted(before.items()):
        relation_id = ontology.alias_to_relation[alias]
        for entity, index in sorted(slot_map.items(), key=lambda row: row[1]):
            if entity in table and table[entity] == value:
                return f"old:{relation_id}:{index}"
    raise ValueError("after value cannot be expressed from transition participants")


def _validate_expression(
    expression: str,
    *,
    position: str,
    slot_types: Sequence[str],
    ontology: DiscoveredOntology,
) -> None:
    type_ids = {row.type_id for row in ontology.types}
    relation_ids = {row.relation_id for row in ontology.relations}
    if expression == "none":
        return
    pieces = expression.split(":")
    if len(pieces) == 2 and pieces[0] == "slot":
        if not pieces[1].isdigit() or not 0 <= int(pieces[1]) < len(slot_types):
            raise ValueError("effect slot expression is out of range")
        return
    if len(pieces) == 2 and pieces[0] == "any" and position == "before":
        if pieces[1] not in type_ids:
            raise ValueError("effect wildcard type is unknown")
        return
    if len(pieces) == 3 and pieces[0] == "old" and position == "after":
        if pieces[1] not in relation_ids:
            raise ValueError("old-cell relation is unknown")
        if not pieces[2].isdigit() or not 0 <= int(pieces[2]) < len(slot_types):
            raise ValueError("old-cell slot is out of range")
        relation = next(
            row for row in ontology.relations if row.relation_id == pieces[1]
        )
        if slot_types[int(pieces[2])] != relation.domain_type:
            raise ValueError("old-cell key slot has an incompatible structural type")
        return
    raise ValueError(f"unsupported {position} effect expression")


def validate_effect_atom(
    effect: OntologyEffectAtom,
    slot_types: Sequence[str],
    ontology: DiscoveredOntology,
) -> None:
    relations = {row.relation_id: row for row in ontology.relations}
    relation = relations.get(effect.relation_id)
    if relation is None:
        raise ValueError("effect relation is not part of the learned ontology")
    if not 0 <= effect.key_slot < len(slot_types):
        raise ValueError("effect key slot is out of range")
    if slot_types[effect.key_slot] != relation.domain_type:
        raise ValueError("effect key slot has an incompatible structural type")
    _validate_expression(
        effect.before,
        position="before",
        slot_types=slot_types,
        ontology=ontology,
    )
    _validate_expression(
        effect.after,
        position="after",
        slot_types=slot_types,
        ontology=ontology,
    )


def law_observation_from_row(
    row: Mapping[str, Any],
    surface_map: Mapping[str, str],
) -> LawObservation:
    validate_ontology_row(row)
    before = row["before"]
    after = row["after"]
    ontology = discover_ontology(before)
    tokens = ontology_tokens(str(row["text"]))
    slots: list[str] = []
    pattern: list[str] = []
    for token in tokens:
        entity = surface_map.get(token)
        if entity is None:
            pattern.append(token)
            continue
        if entity in slots:
            index = slots.index(entity)
        else:
            index = len(slots)
            slots.append(entity)
        pattern.append(f"{{{index}}}")
    participants = set(ontology_delta_participants(row))
    if not slots or not set(slots) <= participants:
        raise ValueError("learned lexemes are not grounded in causal participants")
    slot_map = {entity: index for index, entity in enumerate(slots)}
    slot_types = tuple(ontology.entity_types[entity] for entity in slots)
    effects: list[OntologyEffectAtom] = []
    for alias, table in before.items():
        relation_id = ontology.alias_to_relation[alias]
        for key, old in table.items():
            new = after[alias][key]
            if old == new:
                continue
            if key not in slot_map:
                raise ValueError("changed relation key is absent from utterance slots")
            effect = OntologyEffectAtom(
                relation_id=relation_id,
                key_slot=slot_map[key],
                before=_encode_before_expression(old, slot_map, ontology),
                after=_encode_after_expression(new, slot_map, before, ontology),
            )
            validate_effect_atom(effect, slot_types, ontology)
            effects.append(effect)
    effects.sort(key=lambda row: (row.relation_id, row.key_slot, row.before, row.after))
    if not effects:
        raise ValueError("law observation requires at least one effect")
    signature = ontology_hash(
        {
            "slot_types": list(slot_types),
            "effects": [effect.payload() for effect in effects],
        }
    )
    return LawObservation(
        signature=signature,
        pattern=tuple(pattern),
        slot_types=slot_types,
        effects=tuple(effects),
    )


def train_ontology_field(
    rows: Sequence[Mapping[str, Any]],
    config: OntologyConfig | None = None,
    disabled: Iterable[OntologyPrimitive] = (),
) -> OntologyRuntime:
    if not rows:
        raise ValueError("ontology training requires observations")
    kernel = UniverseOntologyKernel(config=config, disabled=disabled)
    knowledge = build_language_graph()
    reference = discover_ontology(rows[0]["before"])
    state = kernel.initial_state()
    runtime = OntologyRuntime(kernel, state, reference, knowledge)
    _apply_recipe(
        runtime,
        "learn",
        "ontology",
        "ontology-nucleation",
        {"ontology": reference},
    )
    for row in rows:
        validate_ontology_row(row)
        schema = discover_ontology(row["before"])
        if schema.signature != reference.signature:
            raise ValueError("training rows do not share one structural ontology")
        _apply_recipe(
            runtime,
            "language_learn",
            "lexical_observe",
            str(row["case_id"]),
            lexical_payload_from_row(row),
        )
    for index in range(kernel.config.anneal_trials):
        _apply_recipe(
            runtime,
            "thermal_anneal",
            "anneal",
            f"ontology-anneal-{index:03d}",
        )
    _apply_recipe(
        runtime,
        "attention",
        "lexical_finalize",
        "bind-opaque-lexemes",
    )
    _apply_recipe(
        runtime,
        "remember",
        "lexical_finalize",
        "remember-opaque-lexemes",
    )
    surfaces = _surface_map(runtime)
    if surfaces:
        for row in rows:
            try:
                observation = law_observation_from_row(row, surfaces)
            except ValueError:
                if not kernel.disabled:
                    raise
                continue
            _apply_recipe(
                runtime,
                "language_learn",
                "law_observe",
                "law-" + str(row["case_id"]),
                {"observation": observation},
            )
        _apply_recipe(
            runtime,
            "attention",
            "law_finalize",
            "bind-structural-laws",
        )
        _apply_recipe(
            runtime,
            "remember",
            "law_finalize",
            "remember-structural-laws",
        )
    _apply_recipe(runtime, "forget", "forget", "forget-raw-episodes")
    return runtime


def _match_ontology_law(
    runtime: OntologyRuntime,
    text: str,
    world: Mapping[str, Any],
) -> tuple[OntologyLaw, tuple[str, ...], DiscoveredOntology]:
    tokens = ontology_tokens(text)
    ontology = discover_ontology(world)
    if ontology.signature != runtime.reference_ontology.signature:
        raise ValueError("world ontology does not match the learned structural schema")
    lexicon = _surface_map(runtime)
    matches: list[tuple[OntologyLaw, tuple[str, ...]]] = []
    for law in runtime.transition_laws:
        if len(law.pattern) != len(tokens):
            continue
        slots: list[str | None] = [None] * len(law.slot_types)
        valid = True
        for token, piece in zip(tokens, law.pattern, strict=True):
            if piece.startswith("{") and piece.endswith("}"):
                raw = piece[1:-1]
                if not raw.isdigit() or int(raw) >= len(slots):
                    valid = False
                    break
                entity = lexicon.get(token)
                index = int(raw)
                if (
                    entity is None
                    or ontology.entity_types.get(entity) != law.slot_types[index]
                ):
                    valid = False
                    break
                if slots[index] is not None and slots[index] != entity:
                    valid = False
                    break
                slots[index] = entity
            elif piece != token:
                valid = False
                break
        if (
            valid
            and all(slot is not None for slot in slots)
            and len(set(slots)) == len(slots)
        ):
            matches.append((law, tuple(str(slot) for slot in slots)))
    if len(matches) != 1:
        raise ValueError("utterance did not settle on exactly one structural law")
    law, slots = matches[0]
    return law, slots, ontology


def _resolve_expected_before(
    expression: str,
    slots: Sequence[str],
    ontology: DiscoveredOntology,
) -> tuple[str | None, str | None]:
    if expression == "none":
        return None, None
    pieces = expression.split(":")
    if pieces[0] == "slot" and len(pieces) == 2:
        return slots[int(pieces[1])], None
    if pieces[0] == "any" and len(pieces) == 2:
        return None, pieces[1]
    raise ValueError("invalid before expression")


def _resolve_after(
    expression: str,
    slots: Sequence[str],
    before: Mapping[str, Mapping[str, str | None]],
    ontology: DiscoveredOntology,
) -> str | None:
    if expression == "none":
        return None
    pieces = expression.split(":")
    if pieces[0] == "slot" and len(pieces) == 2:
        return slots[int(pieces[1])]
    if pieces[0] == "old" and len(pieces) == 3:
        alias = ontology.relation_to_alias[pieces[1]]
        key = slots[int(pieces[2])]
        return before[alias][key]
    raise ValueError("invalid after expression")


def apply_ontology_text(
    runtime: OntologyRuntime,
    text: str,
    world: Mapping[str, Any],
) -> dict[str, Any]:
    before = copy_opaque_world(world)
    law, slots, ontology = _match_ontology_law(runtime, text, before)
    after = deepcopy(before)
    resolved: list[dict[str, Any]] = []
    for effect in law.effects:
        validate_effect_atom(effect, law.slot_types, ontology)
        alias = ontology.relation_to_alias[effect.relation_id]
        key = slots[effect.key_slot]
        if key not in before[alias]:
            raise ValueError("effect key is absent from the local relation binding")
        actual = before[alias][key]
        exact, wildcard_type = _resolve_expected_before(effect.before, slots, ontology)
        if wildcard_type is None:
            if actual != exact:
                raise ValueError("effect precondition rejected the current world")
        elif actual is None or ontology.entity_types.get(actual) != wildcard_type:
            raise ValueError("effect wildcard precondition rejected the current world")
        value = _resolve_after(effect.after, slots, before, ontology)
        after[alias][key] = value
        resolved.append(
            {
                "relation_id": effect.relation_id,
                "local_alias": alias,
                "key": key,
                "before": actual,
                "after": value,
            }
        )
    return {
        "law_id": law.law_id,
        "slots": list(slots),
        "effects": resolved,
        "world_after": after,
        "ontology_binding": dict(sorted(ontology.alias_to_relation.items())),
        "knowledge_context": runtime.retrieve(
            f"retrieve ontology effect law {law.law_id} from opaque world"
        ),
    }


def generate_ontology_text(
    runtime: OntologyRuntime,
    law_id: str,
    slots: Sequence[str],
) -> dict[str, Any]:
    matches = [law for law in runtime.transition_laws if law.law_id == law_id]
    if len(matches) != 1:
        raise ValueError("generation requires one known law identifier")
    law = matches[0]
    if len(slots) != len(law.slot_types):
        raise ValueError("generation slot count is invalid")
    surfaces = _surface_reverse(runtime)
    output: list[str] = []
    for piece in law.pattern:
        if piece.startswith("{") and piece.endswith("}"):
            index = int(piece[1:-1])
            surface = surfaces.get(str(slots[index]))
            if surface is None:
                raise ValueError("generation slot has no learned surface")
            output.append(surface)
        else:
            output.append(piece)
    return {"text": " ".join(output), "law_id": law_id, "slots": list(slots)}


def evaluator_ontology_law_mapping(
    runtime: OntologyRuntime,
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    votes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        law, _, _ = _match_ontology_law(runtime, str(row["text"]), row["before"])
        label = str(truth[str(row["case_id"])]["semantic_label"])
        votes[label][law.law_id] += 1
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for label in sorted(votes):
        ranked = votes[label].most_common()
        if len(ranked) != 1:
            raise ValueError("evaluator label did not settle on one structural law")
        law_id = ranked[0][0]
        if law_id in used:
            raise ValueError("evaluator labels collapsed onto one structural law")
        mapping[label] = law_id
        used.add(law_id)
    return mapping


def evaluate_ontology_rows(
    runtime: OntologyRuntime,
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    evaluator_mapping: Mapping[str, str],
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        target = truth[str(row["case_id"])]
        try:
            result = apply_ontology_text(runtime, str(row["text"]), row["before"])
            generated = generate_ontology_text(
                runtime, result["law_id"], result["slots"]
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
        / max(cases, 1),
        "law_correct": sum(row["law_correct"] for row in predictions),
        "law_accuracy": sum(row["law_correct"] for row in predictions) / max(cases, 1),
        "generation_correct": sum(row["generation_correct"] for row in predictions),
        "generation_accuracy": sum(row["generation_correct"] for row in predictions)
        / max(cases, 1),
        "predictions": predictions,
    }


def _strict_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _ontology_from_payload(payload: Mapping[str, Any]) -> DiscoveredOntology:
    if set(payload) != {"signature", "types", "relations"}:
        raise ValueError("ontology payload fields are invalid")
    if not isinstance(payload["types"], list) or not isinstance(
        payload["relations"], list
    ):
        raise ValueError("ontology atoms must be arrays")
    types: list[OntologyTypeAtom] = []
    for row in payload["types"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"type_id", "roles"}
            or not isinstance(row["roles"], list)
        ):
            raise ValueError("ontology type atom is invalid")
        atom = _type_atom(tuple(str(role) for role in row["roles"]))
        if atom.type_id != row["type_id"]:
            raise ValueError("ontology type identity does not match its roles")
        types.append(atom)
    if len(types) != 3 or len({row.type_id for row in types}) != 3:
        raise ValueError("ontology must contain three unique structural types")
    type_ids = {row.type_id for row in types}
    relations: list[OntologyRelationAtom] = []
    for row in payload["relations"]:
        expected = {"relation_id", "domain_type", "range_type", "nullable"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("ontology relation atom is invalid")
        if not isinstance(row["nullable"], bool):
            raise ValueError("ontology relation nullability must be boolean")
        if row["domain_type"] not in type_ids or row["range_type"] not in type_ids:
            raise ValueError("ontology relation references an unknown type")
        atom = _relation_atom(
            str(row["domain_type"]),
            str(row["range_type"]),
            row["nullable"],
        )
        if atom.relation_id != row["relation_id"]:
            raise ValueError("ontology relation identity does not match its structure")
        relations.append(atom)
    if len(relations) != 2 or len({row.relation_id for row in relations}) != 2:
        raise ValueError("ontology must contain two unique relation atoms")
    types_tuple = tuple(sorted(types, key=lambda row: row.type_id))
    relations_tuple = tuple(sorted(relations, key=lambda row: row.relation_id))
    core = {
        "types": [row.payload() for row in types_tuple],
        "relations": [row.payload() for row in relations_tuple],
    }
    signature = ontology_hash(core)
    if payload["signature"] != signature:
        raise ValueError("ontology signature does not match its atoms")
    return DiscoveredOntology(
        signature=signature,
        types=types_tuple,
        relations=relations_tuple,
        alias_to_relation={},
        entity_types={},
    )


def _surface_law_from_payload(payload: Mapping[str, Any]) -> SurfaceLaw:
    expected = {"surface", "entity", "evidence_count", "mass", "support", "coherence"}
    if set(payload) != expected:
        raise ValueError("surface law shape is invalid")
    law = SurfaceLaw(
        surface=str(payload["surface"]),
        entity=str(payload["entity"]),
        evidence_count=int(payload["evidence_count"]),
        mass=_strict_finite(payload["mass"], "surface law mass"),
        support=_strict_finite(payload["support"], "surface law support"),
        coherence=_strict_finite(payload["coherence"], "surface law coherence"),
    )
    if ontology_tokens(law.surface) != (law.surface,):
        raise ValueError("surface law must contain one normalized token")
    if _ENTITY_ID.fullmatch(law.entity) is None:
        raise ValueError("surface law entity must remain opaque")
    if law.evidence_count < 2 or law.mass <= 0.0:
        raise ValueError("surface law evidence is invalid")
    if not 0.0 <= law.support <= 1.0 or not 0.0 <= law.coherence <= 1.0:
        raise ValueError("surface law scores must be within [0, 1]")
    return law


def _law_from_payload(
    payload: Mapping[str, Any],
    ontology: DiscoveredOntology,
) -> OntologyLaw:
    expected = {
        "law_id",
        "signature",
        "pattern",
        "slot_types",
        "effects",
        "evidence_count",
        "mass",
        "support",
    }
    if set(payload) != expected:
        raise ValueError("structural law shape is invalid")
    if (
        not isinstance(payload["pattern"], list)
        or not isinstance(payload["slot_types"], list)
        or not isinstance(payload["effects"], list)
    ):
        raise ValueError("structural law sequences must be arrays")
    type_ids = {row.type_id for row in ontology.types}
    slot_types = tuple(str(value) for value in payload["slot_types"])
    if not slot_types or any(value not in type_ids for value in slot_types):
        raise ValueError("structural law slot type is unknown")
    effects: list[OntologyEffectAtom] = []
    for row in payload["effects"]:
        if not isinstance(row, dict) or set(row) != {
            "relation_id",
            "key_slot",
            "before",
            "after",
        }:
            raise ValueError("effect atom shape is invalid")
        if isinstance(row["key_slot"], bool) or not isinstance(row["key_slot"], int):
            raise ValueError("effect key slot must be an integer")
        effect = OntologyEffectAtom(
            relation_id=str(row["relation_id"]),
            key_slot=row["key_slot"],
            before=str(row["before"]),
            after=str(row["after"]),
        )
        validate_effect_atom(effect, slot_types, ontology)
        effects.append(effect)
    if not effects:
        raise ValueError("structural law requires an effect program")
    signature = ontology_hash(
        {
            "slot_types": list(slot_types),
            "effects": [effect.payload() for effect in effects],
        }
    )
    law = OntologyLaw(
        law_id=str(payload["law_id"]),
        signature=str(payload["signature"]),
        pattern=tuple(str(piece) for piece in payload["pattern"]),
        slot_types=slot_types,
        effects=tuple(effects),
        evidence_count=int(payload["evidence_count"]),
        mass=_strict_finite(payload["mass"], "law mass"),
        support=_strict_finite(payload["support"], "law support"),
    )
    if law.signature != signature or law.law_id != "law-" + signature[:16]:
        raise ValueError("structural law identity does not match its effect program")
    if law.evidence_count < 2 or law.mass <= 0.0 or not 0.0 <= law.support <= 1.0:
        raise ValueError("structural law evidence is invalid")
    for piece in law.pattern:
        if piece.startswith("{") and piece.endswith("}"):
            raw = piece[1:-1]
            if not raw.isdigit() or int(raw) >= len(slot_types):
                raise ValueError("structural law pattern slot is invalid")
        elif ontology_tokens(piece) != (piece,):
            raise ValueError("structural law literal is invalid")
    return law


def ontology_model_payload(runtime: OntologyRuntime) -> dict[str, Any]:
    state = runtime.state
    if (
        state.ontology is None
        or state.ontology.signature != runtime.reference_ontology.signature
    ):
        raise ValueError("ontology was not nucleated into learned state")
    if state.raw_evidence_count != 0:
        raise ValueError("forget raw ontology evidence before serialization")
    if state.conservation_applications <= 0:
        raise ValueError("learned atoms were not conserved")
    if not state.surface_laws or not state.laws:
        raise ValueError("ontology model requires learned lexemes and laws")
    core = {
        "schema_version": ONTOLOGY_MODEL_SCHEMA,
        "runtime": ATOM_ONTOLOGY_RUNTIME,
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "knowledge_graph_hash": ontology_hash(runtime.knowledge.manifest()),
        "config": asdict(runtime.kernel.config),
        "ontology": state.ontology.payload(),
        "surface_laws": [law.payload() for law in runtime.surface_laws],
        "transition_laws": [law.payload() for law in runtime.transition_laws],
        "raw_episode_count": len(state.traces),
        "raw_evidence_count": state.raw_evidence_count,
        "temperature": state.temperature,
        "energy": state.energy,
        "cumulative_phase_energy": state.cumulative_phase_energy,
        "maximum_phase_energy": state.maximum_phase_energy,
        "accepted_improving_moves": state.accepted_improving_moves,
        "accepted_worse_moves": state.accepted_worse_moves,
        "observations": state.observations,
        "forgotten": state.forgotten,
        "conservation_applications": state.conservation_applications,
        "operator_counts": dict(sorted(state.operator_counts.items())),
        "outcome_counts": dict(sorted(state.outcome_counts.items())),
        "transition_hash": state.transition_hash,
        "transitions": state.transitions,
    }
    return {**core, "model_hash": ontology_hash(core)}


def runtime_from_ontology_model(payload: Mapping[str, Any]) -> OntologyRuntime:
    expected = {
        "schema_version",
        "runtime",
        "wiki_runtime",
        "rag_runtime",
        "knowledge_graph_hash",
        "config",
        "ontology",
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
        raise ValueError("ontology model fields are invalid")
    if (
        payload["schema_version"] != ONTOLOGY_MODEL_SCHEMA
        or payload["runtime"] != ATOM_ONTOLOGY_RUNTIME
    ):
        raise ValueError("unsupported ontology model runtime")
    if (
        payload["wiki_runtime"] != ATOM_WIKI_GRAPH_RUNTIME
        or payload["rag_runtime"] != ATOM_RAG_RUNTIME
    ):
        raise ValueError("ontology knowledge runtime marker mismatch")
    core = {key: payload[key] for key in payload if key != "model_hash"}
    if ontology_hash(core) != payload["model_hash"]:
        raise ValueError("ontology model hash mismatch")
    if not isinstance(payload["config"], dict) or not isinstance(
        payload["ontology"], dict
    ):
        raise ValueError("ontology model configuration is invalid")
    config = OntologyConfig(**payload["config"])
    config.validate()
    ontology = _ontology_from_payload(payload["ontology"])
    if not isinstance(payload["surface_laws"], list) or not isinstance(
        payload["transition_laws"], list
    ):
        raise ValueError("ontology learned laws must be arrays")
    surfaces = tuple(_surface_law_from_payload(row) for row in payload["surface_laws"])
    laws = tuple(_law_from_payload(row, ontology) for row in payload["transition_laws"])
    if len({row.surface for row in surfaces}) != len(surfaces) or len(
        {row.entity for row in surfaces}
    ) != len(surfaces):
        raise ValueError("serialized surface laws must be one-to-one")
    if len({row.signature for row in laws}) != len(laws):
        raise ValueError("serialized structural laws must be unique")
    if payload["raw_episode_count"] != 0 or payload["raw_evidence_count"] != 0:
        raise ValueError("serialized ontology model must not retain raw evidence")
    graph = build_language_graph()
    if ontology_hash(graph.manifest()) != payload["knowledge_graph_hash"]:
        raise ValueError("ontology knowledge graph hash mismatch")
    if not isinstance(payload["operator_counts"], dict) or set(
        payload["operator_counts"]
    ) != set(UNIVERSE_PRIMITIVE_NAMES):
        raise ValueError("operator counts must cover all universe primitives")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in payload["operator_counts"].values()
    ):
        raise ValueError("operator counts must be positive integers")
    if not isinstance(payload["outcome_counts"], dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in payload["outcome_counts"].values()
    ):
        raise ValueError("outcome counts must be non-negative integers")
    integer_fields = (
        "accepted_improving_moves",
        "accepted_worse_moves",
        "observations",
        "forgotten",
        "conservation_applications",
        "transitions",
    )
    for name in integer_fields:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    state = OntologyTrainingState(
        ontology=ontology,
        surface_laws={row.surface: row for row in surfaces},
        laws={row.signature: row for row in laws},
        temperature=_strict_finite(payload["temperature"], "temperature"),
        energy=_strict_finite(payload["energy"], "energy"),
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
        operator_counts=Counter(
            {str(key): int(value) for key, value in payload["operator_counts"].items()}
        ),
        outcome_counts=Counter(
            {str(key): int(value) for key, value in payload["outcome_counts"].items()}
        ),
        transition_hash=str(payload["transition_hash"]),
        transitions=payload["transitions"],
    )
    return OntologyRuntime(
        UniverseOntologyKernel(config=config),
        state,
        ontology,
        graph,
    )


def write_ontology_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_ontology_workflow_request(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"schema_version", "request_id", "world", "turns"}:
        raise ValueError("ontology workflow request fields are invalid")
    if payload["schema_version"] != ONTOLOGY_MODEL_SCHEMA:
        raise ValueError("unsupported ontology workflow schema")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("workflow request_id must be non-empty text")
    discover_ontology(payload["world"])
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
        ontology_tokens(str(turn["text"]))


def run_ontology_workflow(
    model_path: Path,
    request_path: Path,
    response_path: Path,
) -> dict[str, Any]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict) or not isinstance(request, dict):
        raise ValueError("workflow model and request roots must be objects")
    runtime = runtime_from_ontology_model(model)
    validate_ontology_workflow_request(request)
    world = copy_opaque_world(request["world"])
    turns: list[dict[str, Any]] = []
    for turn in request["turns"]:
        result = apply_ontology_text(runtime, str(turn["text"]), world)
        generated = generate_ontology_text(runtime, result["law_id"], result["slots"])
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
                "ontology_binding": result["ontology_binding"],
                "knowledge_context": result["knowledge_context"],
            }
        )
    core = {
        "schema_version": ONTOLOGY_MODEL_SCHEMA,
        "request_id": request["request_id"],
        "runtime": {
            "ontology_runtime": ATOM_ONTOLOGY_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "model_hash": model["model_hash"],
        },
        "turns": turns,
        "final_world": world,
    }
    response = {**core, "workflow_hash": ontology_hash(core)}
    write_ontology_json(response_path, response)
    return response


def ontology_architecture_audit(source_path: Path | None = None) -> dict[str, Any]:
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
            value.value for value in OntologyPrimitive
        }
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "learning_recipe_resolves_to_all_seven": set(graph.expand("language_learn"))
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "fixed_schema_runtime_is_not_imported": "atom_transition_discovery"
        not in imported_modules,
        "wiki_and_rag_are_runtime_wired": "retrieve_atom_context" in source
        and "build_language_graph" in source,
        "model_loader_is_fail_closed": "ontology model fields are invalid" in source
        and "ontology model hash mismatch" in source,
        "structural_ontology_is_executable": "discover_ontology" in source
        and "apply_ontology_text" in source,
        "raw_aliases_are_not_serialized": "alias_to_relation"
        not in ontology_model_payload.__code__.co_consts,
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_ontology_self_tests() -> dict[str, Any]:
    program = build_ontology_discovery_program()
    train_schema = discover_ontology(program["train"][0]["before"])
    validation_schema = discover_ontology(program["validation"][0]["before"])
    heldout_schema = discover_ontology(program["heldout"][0]["before"])
    corrupt = dict(program["train"][0])
    corrupt["unexpected"] = True
    rejected = False
    try:
        validate_ontology_row(corrupt)
    except ValueError:
        rejected = True
    checks = {
        "split_schemas_are_identical": train_schema.signature
        == validation_schema.signature
        == heldout_schema.signature,
        "three_structural_types": len(train_schema.types) == 3,
        "two_structural_relations": len(train_schema.relations) == 2,
        "opaque_row_validation_fails_closed": rejected,
        "architecture": ontology_architecture_audit()["passed"],
    }
    return {
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }
