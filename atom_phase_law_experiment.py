"""From-scratch Atom phase-law learning experiment.

Opaque symbols self-arrange on a conserved phase lattice. Repeated transition
examples phase-mix into small operator laws. The system has no neural network,
gradient, loss backpropagation, pretrained model, or trainable weight matrix.
Only the seven universe primitives may replace substrate state; cognitive atoms
are graph-resolved compositions of those primitives.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_phase_side_view import (
    ATOM_ARTIFACT_BINDING,
    ATOM_SIDE_VIEW_RUNTIME,
    render_phase_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    AtomWikiGraph,
    retrieve_atom_context,
)


SCHEMA_VERSION = 3
SEED = 20260721
TAU = 2.0 * math.pi


class Primitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


if tuple(primitive.value for primitive in Primitive) != UNIVERSE_PRIMITIVE_NAMES:
    raise RuntimeError("Runtime primitive order diverges from the Atom wiki graph")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def cyclic_distance(left: float, right: float, size: int) -> float:
    raw = abs((left - right) % size)
    return min(raw, size - raw)


def phase_vector(slot: float, size: int, weight: float) -> tuple[float, float]:
    angle = TAU * (slot % size) / size
    return weight * math.cos(angle), weight * math.sin(angle)


def slot_from_vector(real: float, imaginary: float, size: int) -> float:
    if math.hypot(real, imaginary) <= 1e-15:
        return 0.0
    return (math.atan2(imaginary, real) % TAU) * size / TAU


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def counter_tuple(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(name), int(value)) for name, value in values.items()))


def bump_counter(
    values: tuple[tuple[str, int], ...], name: str, amount: int = 1
) -> tuple[tuple[str, int], ...]:
    counter = Counter(dict(values))
    counter[name] += amount
    return counter_tuple(counter)


@dataclass(frozen=True)
class PhaseConfig:
    lattice_size: int = 8
    epochs: int = 52
    nucleation_hits: int = 2
    law_min_active_traces: int = 2
    initial_trace_mass: float = 0.20
    trace_mass_gain: float = 0.10
    trace_support_gain: float = 0.10
    initial_trace_support: float = 0.20
    initial_trace_ttl: float = 3.0
    law_initial_mass: float = 0.45
    law_mass_gain: float = 0.035
    information_mass_budget: float = 12.0
    initial_temperature: float = 1.35
    temperature_floor: float = 0.18
    cooling_rate: float = 0.955
    phase_mix_strength: float = 0.055
    trace_dissipation: float = 0.025
    law_dissipation: float = 0.004
    expiration_support: float = 0.11
    crystallization_coherence: float = 0.58
    observe_swap_trials: int = 2
    anneal_swap_trials: int = 18
    chaos_seed: int = SEED

    def validate(self) -> None:
        integer_ranges = {
            "lattice_size": (4, 64),
            "epochs": (2, 1_000),
            "nucleation_hits": (2, 100),
            "law_min_active_traces": (2, 100),
            "observe_swap_trials": (0, 100),
            "anneal_swap_trials": (1, 1_000),
            "chaos_seed": (0, 2**63 - 1),
        }
        for name, (minimum, maximum) in integer_ranges.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{name} must be an integer within [{minimum}, {maximum}]"
                )
        positive = (
            "initial_trace_mass",
            "trace_mass_gain",
            "trace_support_gain",
            "initial_trace_support",
            "initial_trace_ttl",
            "law_initial_mass",
            "law_mass_gain",
            "information_mass_budget",
            "initial_temperature",
            "temperature_floor",
        )
        for name in positive:
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
        unit_interval = (
            "cooling_rate",
            "phase_mix_strength",
            "trace_dissipation",
            "law_dissipation",
            "expiration_support",
            "crystallization_coherence",
        )
        for name in unit_interval:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.cooling_rate <= 0.0:
            raise ValueError("cooling_rate must be greater than zero")
        if self.phase_mix_strength > 0.25:
            raise ValueError("phase_mix_strength must not exceed 0.25")


@dataclass(frozen=True)
class SymbolPhase:
    name: str
    slot: int
    mass: float = 1.0
    support: float = 1.0


@dataclass(frozen=True)
class OperatorLaw:
    operator: str
    shift: int
    mass: float
    support: float
    coherence: float
    evidence_count: int
    evidence_digest: str
    active: bool = True


@dataclass(frozen=True)
class TransitionTrace:
    trace_id: int
    source: str
    operator: str
    target: str
    mass: float
    support: float
    ttl: float
    hits: int
    created_tick: int
    last_tick: int
    active: bool = False


@dataclass(frozen=True)
class PhaseField:
    operator: str
    phase_slot: float
    quantized_shift: int
    coherence: float
    total_weight: float
    phase_energy: float


@dataclass(frozen=True)
class Stimulus:
    mode: str
    source: str | None = None
    operators: tuple[str, ...] = ()
    target: str | None = None
    salience: float = 1.0
    event_id: str = ""

    def validate(self) -> None:
        if self.mode not in {"observe", "predict", "anneal", "idle", "consolidate"}:
            raise ValueError("Unknown stimulus mode")
        if not isinstance(self.event_id, str) or len(self.event_id) > 200:
            raise ValueError("event_id must be text no longer than 200 characters")
        if (
            isinstance(self.salience, bool)
            or not isinstance(self.salience, (int, float))
            or not math.isfinite(self.salience)
            or not 0.1 <= self.salience <= 2.0
        ):
            raise ValueError("salience must be finite and within [0.1, 2.0]")
        if self.mode == "observe":
            if not self.source or not self.target or len(self.operators) != 1:
                raise ValueError("observe requires source, one operator, and target")
        elif self.mode == "predict":
            if not self.source or not self.operators or self.target is not None:
                raise ValueError("predict requires source and one or more operators")
        elif self.source is not None or self.target is not None:
            raise ValueError(f"{self.mode} cannot carry source or target")


@dataclass(frozen=True)
class PhaseState:
    symbols: tuple[SymbolPhase, ...]
    laws: tuple[OperatorLaw, ...]
    traces: tuple[TransitionTrace, ...]
    tick: int
    temperature: float
    phase_fields: tuple[PhaseField, ...]
    field_ready: bool
    wave_slot: float | None
    wave_strength: float
    candidate_scores: tuple[tuple[str, float], ...]
    prediction: str | None
    confidence: float
    energy: float
    phase_energy: float
    cumulative_phase_energy: float
    maximum_phase_energy: float
    accepted_improving_moves: int
    accepted_worse_moves: int
    observations: int
    predictions: int
    forgotten: int
    operator_counts: tuple[tuple[str, int], ...]
    outcome_counts: tuple[tuple[str, int], ...]
    last_outcome: str
    transition_hash: str
    transitions: int

    @property
    def information_mass(self) -> float:
        return sum(trace.mass for trace in self.traces) + sum(
            law.mass for law in self.laws if law.active
        )


@dataclass(frozen=True)
class ExecutionRecord:
    recipe: str
    event_id: str
    mode: str
    primitives: tuple[str, ...]
    before_hash: str
    after_hash: str
    energy: float
    temperature: float
    phase_energy: float
    prediction: str | None
    outcome: str


class UniversePhaseKernel:
    """Sole replacement authority for the immutable phase substrate."""

    def __init__(
        self,
        config: PhaseConfig | None = None,
        disabled: Iterable[Primitive] = (),
    ) -> None:
        self.config = config or PhaseConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self, symbol_names: Sequence[str]) -> PhaseState:
        names = tuple(str(name) for name in symbol_names)
        if len(names) != self.config.lattice_size or len(set(names)) != len(names):
            raise ValueError("symbol_names must uniquely cover the configured lattice")
        slots = list(range(self.config.lattice_size))
        random.Random(self.config.chaos_seed).shuffle(slots)
        symbols = tuple(
            SymbolPhase(name=name, slot=slot)
            for name, slot in zip(names, slots, strict=True)
        )
        return PhaseState(
            symbols=symbols,
            laws=(),
            traces=(),
            tick=0,
            temperature=self.config.initial_temperature,
            phase_fields=(),
            field_ready=False,
            wave_slot=None,
            wave_strength=0.0,
            candidate_scores=(),
            prediction=None,
            confidence=0.0,
            energy=0.0,
            phase_energy=0.0,
            cumulative_phase_energy=0.0,
            maximum_phase_energy=0.0,
            accepted_improving_moves=0,
            accepted_worse_moves=0,
            observations=0,
            predictions=0,
            forgotten=0,
            operator_counts=(),
            outcome_counts=(),
            last_outcome="initial",
            transition_hash="0" * 64,
            transitions=0,
        )

    def apply(
        self, state: PhaseState, primitive: Primitive, stimulus: Stimulus
    ) -> PhaseState:
        stimulus.validate()
        self._validate_membership(state, stimulus)
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
        counts = bump_counter(transitioned.operator_counts, primitive.value)
        digest = stable_hash(
            {
                "previous": state.transition_hash,
                "primitive": primitive.value,
                "mode": stimulus.mode,
                "event": stimulus.event_id,
                "tick": transitioned.tick,
                "symbols": [(row.name, row.slot) for row in transitioned.symbols],
                "laws": [
                    (
                        row.operator,
                        row.shift,
                        row.evidence_count,
                        round(row.coherence, 9),
                    )
                    for row in transitioned.laws
                ],
                "traces": [
                    (row.source, row.operator, row.target, row.hits, row.active)
                    for row in transitioned.traces
                ],
                "temperature": round(transitioned.temperature, 12),
                "energy": round(transitioned.energy, 12),
                "prediction": transitioned.prediction,
                "outcome": transitioned.last_outcome,
            }
        )
        return replace(
            transitioned,
            operator_counts=counts,
            transition_hash=digest,
            transitions=state.transitions + 1,
        )

    def _validate_membership(self, state: PhaseState, stimulus: Stimulus) -> None:
        symbols = {row.name for row in state.symbols}
        operators = {row.operator for row in state.laws} | {
            row.operator for row in state.traces
        }
        if stimulus.source is not None and stimulus.source not in symbols:
            raise ValueError(f"Unknown source symbol: {stimulus.source}")
        if stimulus.target is not None and stimulus.target not in symbols:
            raise ValueError(f"Unknown target symbol: {stimulus.target}")
        if stimulus.mode == "predict":
            unknown = set(stimulus.operators) - operators
            if unknown:
                raise ValueError(f"Unknown operator laws: {sorted(unknown)}")

    def _symbol_slots(self, symbols: Sequence[SymbolPhase]) -> dict[str, int]:
        return {row.name: row.slot for row in symbols}

    def _law_map(self, laws: Sequence[OperatorLaw]) -> dict[str, OperatorLaw]:
        return {row.operator: row for row in laws if row.active}

    def _trace_weight(self, trace: TransitionTrace) -> float:
        return max(0.05, trace.support) * (1.0 + math.log1p(trace.hits))

    def _base_phase_field(
        self,
        symbols: Sequence[SymbolPhase],
        laws: Sequence[OperatorLaw],
        traces: Sequence[TransitionTrace],
        operator: str,
    ) -> tuple[float, float, float]:
        slots = self._symbol_slots(symbols)
        real = 0.0
        imaginary = 0.0
        total = 0.0
        for trace in traces:
            if not trace.active or trace.operator != operator:
                continue
            shift = (
                slots[trace.target] - slots[trace.source]
            ) % self.config.lattice_size
            weight = self._trace_weight(trace)
            x, y = phase_vector(shift, self.config.lattice_size, weight)
            real += x
            imaginary += y
            total += weight
        law = self._law_map(laws).get(operator)
        if law is not None:
            weight = max(0.05, law.mass * law.support) * (
                0.35 + 0.20 * math.sqrt(max(1, law.evidence_count))
            )
            x, y = phase_vector(law.shift, self.config.lattice_size, weight)
            real += x
            imaginary += y
            total += weight
        return real, imaginary, total

    def _energy_for(
        self,
        symbols: Sequence[SymbolPhase],
        laws: Sequence[OperatorLaw],
        traces: Sequence[TransitionTrace],
    ) -> float:
        slots = self._symbol_slots(symbols)
        law_map = self._law_map(laws)
        weighted_error = 0.0
        total_weight = 0.0
        for trace in traces:
            if not trace.active:
                continue
            weight = self._trace_weight(trace)
            law = law_map.get(trace.operator)
            if law is None:
                distance = self.config.lattice_size / 2.0
            else:
                predicted = (slots[trace.source] + law.shift) % self.config.lattice_size
                distance = cyclic_distance(
                    predicted, slots[trace.target], self.config.lattice_size
                )
            weighted_error += weight * distance * distance
            total_weight += weight
        return weighted_error / total_weight if total_weight > 1e-12 else 0.0

    def _law_evidence(
        self, operator: str, traces: Sequence[TransitionTrace]
    ) -> tuple[int, str]:
        rows = [
            {
                "source": trace.source,
                "operator": trace.operator,
                "target": trace.target,
                "hits": trace.hits,
            }
            for trace in traces
            if trace.active and trace.operator == operator
        ]
        rows.sort(key=lambda row: (row["source"], row["target"]))
        return sum(int(row["hits"]) for row in rows), stable_hash(rows)

    def _refresh_one_law(
        self,
        laws: Sequence[OperatorLaw],
        traces: Sequence[TransitionTrace],
        symbols: Sequence[SymbolPhase],
        operator: str,
        force_shift: bool,
    ) -> tuple[OperatorLaw, ...]:
        active = [row for row in traces if row.active and row.operator == operator]
        if len(active) < self.config.law_min_active_traces:
            return tuple(laws)
        real, imaginary, total = self._base_phase_field(symbols, (), active, operator)
        coherence = math.hypot(real, imaginary) / max(total, 1e-12)
        candidate = int(
            round(slot_from_vector(real, imaginary, self.config.lattice_size))
        )
        candidate %= self.config.lattice_size
        evidence_count, evidence_digest = self._law_evidence(operator, active)
        rows = list(laws)
        for index, law in enumerate(rows):
            if law.operator != operator:
                continue
            rows[index] = replace(
                law,
                shift=candidate if force_shift else law.shift,
                mass=law.mass + self.config.law_mass_gain,
                support=min(1.0, law.support + 0.02),
                coherence=coherence,
                evidence_count=evidence_count,
                evidence_digest=evidence_digest,
                active=True,
            )
            return tuple(rows)
        rows.append(
            OperatorLaw(
                operator=operator,
                shift=candidate,
                mass=self.config.law_initial_mass,
                support=0.45,
                coherence=coherence,
                evidence_count=evidence_count,
                evidence_digest=evidence_digest,
            )
        )
        rows.sort(key=lambda row: row.operator)
        return tuple(rows)

    def _nucleation(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        if stimulus.mode == "observe":
            assert stimulus.source is not None and stimulus.target is not None
            operator = stimulus.operators[0]
            traces = list(state.traces)
            matched = None
            for index, trace in enumerate(traces):
                if (
                    trace.source == stimulus.source
                    and trace.operator == operator
                    and trace.target == stimulus.target
                ):
                    matched = index
                    break
            if matched is None:
                trace = TransitionTrace(
                    trace_id=max((row.trace_id for row in traces), default=-1) + 1,
                    source=stimulus.source,
                    operator=operator,
                    target=stimulus.target,
                    mass=self.config.initial_trace_mass * stimulus.salience,
                    support=self.config.initial_trace_support * stimulus.salience,
                    ttl=self.config.initial_trace_ttl,
                    hits=1,
                    created_tick=state.tick,
                    last_tick=state.tick,
                    active=False,
                )
                traces.append(trace)
                outcome = "seed"
            else:
                previous = traces[matched]
                hits = previous.hits + 1
                traces[matched] = replace(
                    previous,
                    mass=previous.mass
                    + self.config.trace_mass_gain * stimulus.salience,
                    support=min(
                        1.0,
                        previous.support
                        + self.config.trace_support_gain * stimulus.salience,
                    ),
                    ttl=self.config.initial_trace_ttl + min(24.0, hits * 0.75),
                    hits=hits,
                    last_tick=state.tick,
                    active=hits >= self.config.nucleation_hits,
                )
                outcome = "crystallize" if not previous.active else "reinforce"
            laws = self._refresh_one_law(
                state.laws,
                traces,
                state.symbols,
                operator,
                force_shift=False,
            )
            transitioned = replace(
                state,
                traces=tuple(traces),
                laws=laws,
                observations=state.observations + 1,
                outcome_counts=bump_counter(state.outcome_counts, outcome),
                last_outcome=outcome,
            )
            return replace(
                transitioned,
                energy=self._energy_for(
                    transitioned.symbols, transitioned.laws, transitioned.traces
                ),
            )
        if stimulus.mode == "consolidate":
            operators = sorted(
                {row.operator for row in state.traces if row.active}
                | {row.operator for row in state.laws if row.active}
            )
            laws = state.laws
            for operator in operators:
                laws = self._refresh_one_law(
                    laws,
                    state.traces,
                    state.symbols,
                    operator,
                    force_shift=True,
                )
            transitioned = replace(
                state,
                laws=laws,
                last_outcome="nucleate_laws",
                outcome_counts=bump_counter(state.outcome_counts, "nucleate_laws"),
            )
            return replace(
                transitioned,
                energy=self._energy_for(
                    transitioned.symbols, transitioned.laws, transitioned.traces
                ),
            )
        return state

    def _radiation(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        operators = list(stimulus.operators)
        if not operators and stimulus.mode in {"anneal", "consolidate"}:
            operators = sorted(
                {row.operator for row in state.traces if row.active}
                | {row.operator for row in state.laws if row.active}
            )
        temperature_span = (
            self.config.initial_temperature - self.config.temperature_floor
        )
        thermal_ratio = (
            clamp(
                (state.temperature - self.config.temperature_floor) / temperature_span,
                0.0,
                1.0,
            )
            if temperature_span > 1e-12
            else 0.0
        )
        fields = []
        for index, operator in enumerate(operators):
            real, imaginary, total = self._base_phase_field(
                state.symbols, state.laws, state.traces, operator
            )
            if total <= 1e-12:
                fields.append(PhaseField(operator, 0.0, 0, 0.0, 0.0, 0.0))
                continue
            base_slot = slot_from_vector(real, imaginary, self.config.lattice_size)
            base_magnitude = math.hypot(real, imaginary)
            base_coherence = base_magnitude / total
            chaos_weight = (
                self.config.phase_mix_strength
                * (0.10 + 0.90 * thermal_ratio)
                * total
                * (0.05 + 1.0 - min(1.0, base_coherence))
            )
            chaos_slot = (
                deterministic_fraction(
                    self.config.chaos_seed,
                    stimulus.event_id,
                    state.tick,
                    state.transitions,
                    operator,
                    index,
                )
                * self.config.lattice_size
            )
            chaos_real, chaos_imaginary = phase_vector(
                chaos_slot, self.config.lattice_size, chaos_weight
            )
            mixed_real = real + chaos_real
            mixed_imaginary = imaginary + chaos_imaginary
            mixed_slot = slot_from_vector(
                mixed_real, mixed_imaginary, self.config.lattice_size
            )
            phase_energy = (
                cyclic_distance(base_slot, mixed_slot, self.config.lattice_size)
                / self.config.lattice_size
            )
            coherence = math.hypot(mixed_real, mixed_imaginary) / max(
                total + chaos_weight, 1e-12
            )
            quantized = int(round(mixed_slot)) % self.config.lattice_size
            fields.append(
                PhaseField(
                    operator=operator,
                    phase_slot=mixed_slot,
                    quantized_shift=quantized,
                    coherence=clamp(coherence, 0.0, 1.0),
                    total_weight=total + chaos_weight,
                    phase_energy=phase_energy,
                )
            )
        wave_slot = None
        wave_strength = 0.0
        if stimulus.source is not None and fields:
            wave_slot = float(self._symbol_slots(state.symbols)[stimulus.source])
            strengths = []
            for field in fields:
                if field.total_weight <= 1e-12:
                    wave_slot = None
                    strengths = []
                    break
                wave_slot = (wave_slot + field.phase_slot) % self.config.lattice_size
                strengths.append(field.coherence)
            wave_strength = math.prod(strengths) if strengths else 0.0
        phase_energy = (
            sum(field.phase_energy for field in fields) / len(fields) if fields else 0.0
        )
        return replace(
            state,
            phase_fields=tuple(fields),
            field_ready=False,
            wave_slot=wave_slot,
            wave_strength=wave_strength,
            candidate_scores=(),
            prediction=None,
            confidence=0.0,
            phase_energy=phase_energy,
            cumulative_phase_energy=state.cumulative_phase_energy + phase_energy,
            maximum_phase_energy=max(state.maximum_phase_energy, phase_energy),
            last_outcome="radiate",
        )

    def _gravitation(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        if not state.phase_fields or any(
            field.total_weight <= 1e-12 for field in state.phase_fields
        ):
            return replace(
                state,
                field_ready=False,
                candidate_scores=(),
                last_outcome="no_attractor",
            )
        scores: tuple[tuple[str, float], ...] = ()
        if stimulus.mode == "predict":
            if state.wave_slot is None:
                return replace(
                    state,
                    field_ready=False,
                    candidate_scores=(),
                    last_outcome="no_wave",
                )
            scores = tuple(
                (
                    symbol.name,
                    state.wave_strength
                    * math.exp(
                        -1.8
                        * cyclic_distance(
                            state.wave_slot,
                            symbol.slot,
                            self.config.lattice_size,
                        )
                        ** 2
                    ),
                )
                for symbol in state.symbols
            )
        return replace(
            state,
            field_ready=True,
            candidate_scores=scores,
            last_outcome="gravitate",
        )

    def _accept_move(
        self,
        delta: float,
        temperature: float,
        event_id: str,
        state: PhaseState,
        trial: int,
    ) -> bool:
        if delta < -1e-12:
            return True
        if abs(delta) <= 1e-12:
            thermal_span = max(
                1e-12,
                self.config.initial_temperature - self.config.temperature_floor,
            )
            ratio = clamp(
                (temperature - self.config.temperature_floor) / thermal_span,
                0.0,
                1.0,
            )
            return (
                deterministic_fraction(
                    self.config.chaos_seed,
                    "equal",
                    event_id,
                    state.transitions,
                    trial,
                )
                < 0.06 * ratio
            )
        probability = math.exp(-delta / max(temperature, 1e-9))
        return (
            deterministic_fraction(
                self.config.chaos_seed,
                "worse",
                event_id,
                state.transitions,
                trial,
            )
            < probability
        )

    def _swap_symbols(
        self, symbols: Sequence[SymbolPhase], left: int, right: int
    ) -> tuple[SymbolPhase, ...]:
        rows = list(symbols)
        left_slot = rows[left].slot
        right_slot = rows[right].slot
        rows[left] = replace(rows[left], slot=right_slot)
        rows[right] = replace(rows[right], slot=left_slot)
        return tuple(rows)

    def _attraction_repulsion(
        self, state: PhaseState, stimulus: Stimulus
    ) -> PhaseState:
        if not state.field_ready:
            return state
        if stimulus.mode == "predict":
            if not state.candidate_scores:
                return replace(
                    state,
                    prediction=None,
                    confidence=0.0,
                    predictions=state.predictions + 1,
                    outcome_counts=bump_counter(state.outcome_counts, "reject"),
                    last_outcome="reject",
                )
            prediction, score = max(
                state.candidate_scores, key=lambda row: (row[1], row[0])
            )
            accepted = score >= 0.28
            outcome = "retrieve" if accepted else "reject"
            return replace(
                state,
                prediction=prediction if accepted else None,
                confidence=clamp(score, 0.0, 1.0) if accepted else 0.0,
                predictions=state.predictions + 1,
                outcome_counts=bump_counter(state.outcome_counts, outcome),
                last_outcome=outcome,
            )
        if stimulus.mode not in {"observe", "anneal"}:
            return state

        laws = list(state.laws)
        current_energy = self._energy_for(state.symbols, laws, state.traces)
        improving = state.accepted_improving_moves
        worse = state.accepted_worse_moves
        for field_index, field in enumerate(state.phase_fields):
            law_index = next(
                (
                    index
                    for index, law in enumerate(laws)
                    if law.operator == field.operator and law.active
                ),
                None,
            )
            if law_index is None:
                continue
            previous = laws[law_index]
            candidate = replace(
                previous,
                shift=field.quantized_shift,
                coherence=field.coherence,
            )
            candidate_laws = list(laws)
            candidate_laws[law_index] = candidate
            candidate_energy = self._energy_for(
                state.symbols, candidate_laws, state.traces
            )
            delta = candidate_energy - current_energy
            if self._accept_move(
                delta,
                state.temperature,
                stimulus.event_id,
                state,
                10_000 + field_index,
            ):
                laws = candidate_laws
                current_energy = candidate_energy
                if delta < -1e-12:
                    improving += 1
                elif delta > 1e-12:
                    worse += 1

        symbols = state.symbols
        trials = (
            self.config.observe_swap_trials
            if stimulus.mode == "observe"
            else self.config.anneal_swap_trials
        )
        thermal_span = max(
            1e-12, self.config.initial_temperature - self.config.temperature_floor
        )
        thermal_ratio = clamp(
            (state.temperature - self.config.temperature_floor) / thermal_span,
            0.0,
            1.0,
        )
        for trial in range(trials):
            candidates = []
            for left in range(len(symbols)):
                for right in range(left + 1, len(symbols)):
                    swapped = self._swap_symbols(symbols, left, right)
                    energy = self._energy_for(swapped, laws, state.traces)
                    candidates.append((energy, left, right, swapped))
            candidates.sort(key=lambda row: (row[0], row[1], row[2]))
            if not candidates:
                break
            explore = (
                deterministic_fraction(
                    self.config.chaos_seed,
                    "rank",
                    stimulus.event_id,
                    state.transitions,
                    trial,
                )
                < 0.28 * thermal_ratio
            )
            rank_limit = min(6, len(candidates))
            rank = (
                int(
                    deterministic_fraction(
                        self.config.chaos_seed,
                        "rank-choice",
                        stimulus.event_id,
                        state.transitions,
                        trial,
                    )
                    * rank_limit
                )
                if explore
                else 0
            )
            candidate_energy, _, _, candidate_symbols = candidates[rank]
            delta = candidate_energy - current_energy
            if not self._accept_move(
                delta,
                state.temperature,
                stimulus.event_id,
                state,
                trial,
            ):
                continue
            symbols = candidate_symbols
            current_energy = candidate_energy
            if delta < -1e-12:
                improving += 1
            elif delta > 1e-12:
                worse += 1

        outcome = "anneal_move" if stimulus.mode == "anneal" else "bind"
        return replace(
            state,
            symbols=tuple(symbols),
            laws=tuple(laws),
            energy=current_energy,
            accepted_improving_moves=improving,
            accepted_worse_moves=worse,
            outcome_counts=bump_counter(state.outcome_counts, outcome),
            last_outcome=outcome,
        )

    def _conservation(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        del stimulus
        slots = sorted(symbol.slot for symbol in state.symbols)
        if slots != list(range(self.config.lattice_size)):
            raise RuntimeError("Phase lattice lost its one-symbol-per-slot invariant")
        if len({law.operator for law in state.laws}) != len(state.laws):
            raise RuntimeError("Operator laws lost uniqueness")
        total = state.information_mass
        if total <= self.config.information_mass_budget or total <= 1e-12:
            return replace(state, last_outcome="conserve")
        scale = self.config.information_mass_budget / total
        traces = tuple(
            replace(trace, mass=trace.mass * scale) for trace in state.traces
        )
        laws = tuple(replace(law, mass=law.mass * scale) for law in state.laws)
        return replace(
            state,
            traces=traces,
            laws=laws,
            outcome_counts=bump_counter(state.outcome_counts, "mass_projection"),
            last_outcome="mass_projection",
        )

    def _dissipation(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        del stimulus
        traces = tuple(
            replace(
                trace,
                ttl=trace.ttl - 1.0 / max(1.0, math.sqrt(trace.hits)),
                support=trace.support
                * (
                    1.0
                    - self.config.trace_dissipation / max(1.0, math.sqrt(trace.hits))
                ),
                mass=trace.mass
                * (1.0 - 0.30 * self.config.trace_dissipation / max(1, trace.hits)),
            )
            for trace in state.traces
        )
        laws = tuple(
            replace(
                law,
                support=law.support * (1.0 - self.config.law_dissipation),
                mass=law.mass * (1.0 - 0.25 * self.config.law_dissipation),
            )
            for law in state.laws
        )
        temperature = max(
            self.config.temperature_floor,
            self.config.temperature_floor
            + (state.temperature - self.config.temperature_floor)
            * self.config.cooling_rate,
        )
        return replace(
            state,
            traces=traces,
            laws=laws,
            tick=state.tick + 1,
            temperature=temperature,
            phase_fields=(),
            field_ready=False,
            wave_slot=None,
            wave_strength=0.0,
            candidate_scores=(),
            prediction=None,
            confidence=0.0,
            phase_energy=0.0,
            last_outcome="dissipate",
        )

    def _decay(self, state: PhaseState, stimulus: Stimulus) -> PhaseState:
        law_map = self._law_map(state.laws)
        if stimulus.mode == "consolidate":
            retained = []
            for trace in state.traces:
                law = law_map.get(trace.operator)
                crystallized = (
                    law is not None
                    and law.coherence >= self.config.crystallization_coherence
                    and law.evidence_count >= self.config.law_min_active_traces
                )
                if not crystallized:
                    retained.append(trace)
            forgotten = len(state.traces) - len(retained)
        else:
            retained = [
                trace
                for trace in state.traces
                if not (
                    trace.hits < self.config.nucleation_hits
                    and (
                        trace.ttl <= 0.0
                        or trace.support < self.config.expiration_support
                    )
                )
            ]
            forgotten = len(state.traces) - len(retained)
        if forgotten == 0:
            return state
        outcome = "coarse_grain" if stimulus.mode == "consolidate" else "decay"
        transitioned = replace(
            state,
            traces=tuple(retained),
            forgotten=state.forgotten + forgotten,
            outcome_counts=bump_counter(state.outcome_counts, outcome, forgotten),
            last_outcome=outcome,
        )
        return replace(
            transitioned,
            energy=self._energy_for(
                transitioned.symbols, transitioned.laws, transitioned.traces
            ),
        )


class PhaseRuntime:
    def __init__(
        self,
        symbol_names: Sequence[str],
        kernel: UniversePhaseKernel | None = None,
        knowledge: AtomWikiGraph | None = None,
        state: PhaseState | None = None,
    ) -> None:
        self.kernel = kernel or UniversePhaseKernel()
        self.knowledge = knowledge or AtomWikiGraph()
        self.knowledge.assert_all_leaves_are_universe_primitives()
        self.state = state or self.kernel.initial_state(symbol_names)
        self.records: list[ExecutionRecord] = []
        self.recipe_counts: Counter[str] = Counter()

    def execute(self, recipe: str, stimulus: Stimulus) -> PhaseState:
        primitive_names = self.knowledge.expand(recipe)
        before = self.state.transition_hash
        current = self.state
        for name in primitive_names:
            current = self.kernel.apply(current, Primitive(name), stimulus)
        self.state = current
        self.recipe_counts[recipe] += 1
        self.records.append(
            ExecutionRecord(
                recipe=recipe,
                event_id=stimulus.event_id,
                mode=stimulus.mode,
                primitives=primitive_names,
                before_hash=before,
                after_hash=current.transition_hash,
                energy=current.energy,
                temperature=current.temperature,
                phase_energy=current.phase_energy,
                prediction=current.prediction,
                outcome=current.last_outcome,
            )
        )
        return current

    def observe(
        self,
        source: str,
        operator: str,
        target: str,
        event_id: str,
        salience: float = 1.0,
        atom: str = "remember",
    ) -> PhaseState:
        return self.execute(
            atom,
            Stimulus(
                mode="observe",
                source=source,
                operators=(operator,),
                target=target,
                salience=salience,
                event_id=event_id,
            ),
        )

    def anneal(self, event_id: str) -> PhaseState:
        return self.execute(
            "thermal_anneal", Stimulus(mode="anneal", event_id=event_id)
        )

    def forget(self, steps: int = 1) -> PhaseState:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 0 <= steps <= 100
        ):
            raise ValueError("steps must be an integer within [0, 100]")
        for index in range(steps):
            self.execute(
                "forget",
                Stimulus(mode="idle", event_id=f"forget-{self.state.tick}-{index}"),
            )
        return self.state

    def consolidate(self, event_id: str = "abstract-final") -> PhaseState:
        return self.execute("abstract", Stimulus(mode="consolidate", event_id=event_id))

    def predict(
        self, source: str, operators: Sequence[str], event_id: str
    ) -> Mapping[str, Any] | None:
        state = self.execute(
            "retrieve",
            Stimulus(
                mode="predict",
                source=source,
                operators=tuple(operators),
                event_id=event_id,
            ),
        )
        if state.prediction is None:
            return None
        return {
            "symbol": state.prediction,
            # Canonicalize public floating-point values so a serialized request
            # produces byte-stable JSON across libm implementations.
            "confidence": round(state.confidence, 12),
            "wave_slot": state.wave_slot,
            "wave_strength": round(state.wave_strength, 12),
        }


def build_tiny_world(seed: int = SEED) -> dict[str, Any]:
    opaque_symbols = ["aru", "bex", "cai", "dun", "evo", "fyn", "gim", "hex"]
    opaque_operators = ["kora", "mavi", "senu", "tela"]
    rng = random.Random(seed)
    rng.shuffle(opaque_symbols)
    rng.shuffle(opaque_operators)
    latent_to_symbol = dict(enumerate(opaque_symbols))
    operator_specs = [
        (opaque_operators[0], 1, (0, 1, 2, 3, 4, 5, 6), (), (7,)),
        (opaque_operators[1], -1, (0, 2, 3, 5, 6), (1,), (4, 7)),
        (opaque_operators[2], 2, (0, 1, 3, 4, 6), (2,), (5, 7)),
        (opaque_operators[3], 0, (0, 2, 4, 6), (1, 5), (3, 7)),
    ]

    def make_row(
        split: str, source_index: int, operator: str, shift: int
    ) -> dict[str, Any]:
        target_index = (source_index + shift) % len(opaque_symbols)
        return {
            "case_id": f"{split}-{operator}-{source_index}",
            "source": latent_to_symbol[source_index],
            "operators": [operator],
            "target": latent_to_symbol[target_index],
        }

    train = []
    validation = []
    heldout = []
    truth_shifts = {}
    for (
        operator,
        shift,
        train_indices,
        validation_indices,
        heldout_indices,
    ) in operator_specs:
        truth_shifts[operator] = shift % len(opaque_symbols)
        train.extend(
            make_row("train", index, operator, shift) for index in train_indices
        )
        validation.extend(
            make_row("validation", index, operator, shift)
            for index in validation_indices
        )
        heldout.extend(
            make_row("heldout", index, operator, shift) for index in heldout_indices
        )

    two_step = []
    three_step = []
    for source_index in range(len(opaque_symbols)):
        for first in opaque_operators:
            for second in opaque_operators:
                target_index = (
                    source_index + truth_shifts[first] + truth_shifts[second]
                ) % len(opaque_symbols)
                two_step.append(
                    {
                        "case_id": f"two-{source_index}-{first}-{second}",
                        "source": latent_to_symbol[source_index],
                        "operators": [first, second],
                        "target": latent_to_symbol[target_index],
                    }
                )
                for third in opaque_operators:
                    final_index = (target_index + truth_shifts[third]) % len(
                        opaque_symbols
                    )
                    three_step.append(
                        {
                            "case_id": f"three-{source_index}-{first}-{second}-{third}",
                            "source": latent_to_symbol[source_index],
                            "operators": [first, second, third],
                            "target": latent_to_symbol[final_index],
                        }
                    )

    noise = []
    for index, base in enumerate(train[:3]):
        correct_index = opaque_symbols.index(base["target"])
        wrong_target = opaque_symbols[(correct_index + 3 + index) % len(opaque_symbols)]
        noise.append(
            {
                "case_id": f"noise-{index}",
                "source": base["source"],
                "operators": list(base["operators"]),
                "target": wrong_target,
            }
        )
    return {
        "seed": seed,
        "symbols": sorted(opaque_symbols),
        "operators": sorted(opaque_operators),
        "train": train,
        "validation": validation,
        "heldout_single_step": heldout,
        "unseen_two_step": two_step,
        "unseen_three_step": three_step,
        "one_off_noise": noise,
        "hidden_truth": {
            "latent_to_symbol": latent_to_symbol,
            "operator_shifts": truth_shifts,
        },
    }


def structural_snapshot(runtime: PhaseRuntime) -> dict[str, Any]:
    return {
        "symbols": [asdict(row) for row in runtime.state.symbols],
        "laws": [asdict(row) for row in runtime.state.laws],
        "traces": [asdict(row) for row in runtime.state.traces],
        "temperature": runtime.state.temperature,
        "energy": runtime.state.energy,
        "accepted_improving_moves": runtime.state.accepted_improving_moves,
        "accepted_worse_moves": runtime.state.accepted_worse_moves,
    }


def train_phase_model(
    program: Mapping[str, Any],
    config: PhaseConfig | None = None,
    disabled: Iterable[Primitive] = (),
) -> tuple[PhaseRuntime, list[dict[str, Any]]]:
    resolved = config or PhaseConfig()
    kernel = UniversePhaseKernel(resolved, disabled=disabled)
    runtime = PhaseRuntime(program["symbols"], kernel=kernel)
    history = []
    for epoch in range(resolved.epochs):
        rows = list(program["train"])
        random.Random(resolved.chaos_seed + 10_007 * epoch).shuffle(rows)
        for index, row in enumerate(rows):
            runtime.observe(
                row["source"],
                row["operators"][0],
                row["target"],
                event_id=f"epoch-{epoch}-{index}-{row['case_id']}",
            )
        if epoch == 0:
            for row in program["one_off_noise"]:
                runtime.observe(
                    row["source"],
                    row["operators"][0],
                    row["target"],
                    event_id=row["case_id"],
                    salience=0.55,
                )
        runtime.anneal(event_id=f"anneal-{epoch}")
        runtime.forget(1)
        history.append(
            {
                "epoch": epoch,
                "energy": runtime.state.energy,
                "temperature": runtime.state.temperature,
                "laws": len(runtime.state.laws),
                "active_traces": sum(row.active for row in runtime.state.traces),
                "raw_traces": len(runtime.state.traces),
                "information_mass": runtime.state.information_mass,
                "phase_energy": runtime.state.cumulative_phase_energy,
                "accepted_improving_moves": runtime.state.accepted_improving_moves,
                "accepted_worse_moves": runtime.state.accepted_worse_moves,
            }
        )
    return runtime, history


def evaluate_rows(
    runtime: PhaseRuntime,
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
    allow_unknown_laws: bool = False,
) -> dict[str, Any]:
    predictions = []
    correct = 0
    covered = 0
    for index, row in enumerate(rows):
        known_laws = {law.operator for law in runtime.state.laws if law.active}
        unknown_laws = set(row["operators"]) - known_laws
        if unknown_laws and allow_unknown_laws:
            result = None
        else:
            result = runtime.predict(
                row["source"],
                row["operators"],
                event_id=f"{prefix}-{index}-{row['case_id']}",
            )
        predicted = result["symbol"] if result is not None else None
        is_correct = predicted == row["target"]
        correct += int(is_correct)
        covered += int(predicted is not None)
        predictions.append(
            {
                "case_id": row["case_id"],
                "source": row["source"],
                "operators": list(row["operators"]),
                "expected": row["target"],
                "predicted": predicted,
                "correct": is_correct,
                "confidence": result["confidence"] if result is not None else 0.0,
            }
        )
    total = len(rows)
    return {
        "cases": total,
        "correct": correct,
        "covered": covered,
        "accuracy": correct / total if total else 0.0,
        "coverage": covered / total if total else 0.0,
        "predictions": predictions,
    }


class ExactTransitionTable:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._table = {
            (row["source"], row["operators"][0]): row["target"] for row in rows
        }

    def predict(self, source: str, operators: Sequence[str]) -> str | None:
        current = source
        for operator in operators:
            current = self._table.get((current, operator))
            if current is None:
                return None
        return current


def evaluate_table(
    table: ExactTransitionTable, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    predictions = [table.predict(row["source"], row["operators"]) for row in rows]
    correct = sum(
        prediction == row["target"]
        for prediction, row in zip(predictions, rows, strict=True)
    )
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "coverage": sum(value is not None for value in predictions) / len(rows)
        if rows
        else 0.0,
    }


def model_payload(runtime: PhaseRuntime, program: Mapping[str, Any]) -> dict[str, Any]:
    if runtime.state.traces:
        raise ValueError(
            "Model export requires raw transition traces to be coarse-grained"
        )
    structural_transition_hash = stable_hash(
        {
            "symbols": [asdict(row) for row in runtime.state.symbols],
            "laws": [asdict(row) for row in runtime.state.laws if row.active],
            "training_hash": stable_hash(program["train"]),
            "config": asdict(runtime.kernel.config),
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "atom_emergent_phase_law",
        "lattice_size": runtime.kernel.config.lattice_size,
        "symbols": [asdict(row) for row in runtime.state.symbols],
        "laws": [asdict(row) for row in runtime.state.laws if row.active],
        "temperature": runtime.state.temperature,
        "transition_hash": structural_transition_hash,
        "training_hash": stable_hash(program["train"]),
        "knowledge_graph_hash": stable_hash(runtime.knowledge.manifest()),
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "config": asdict(runtime.kernel.config),
        "raw_trace_count": 0,
    }
    payload["model_hash"] = stable_hash(payload)
    return payload


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


def runtime_from_model(payload: Mapping[str, Any]) -> PhaseRuntime:
    required = {
        "schema_version",
        "model_type",
        "lattice_size",
        "symbols",
        "laws",
        "temperature",
        "transition_hash",
        "training_hash",
        "knowledge_graph_hash",
        "wiki_runtime",
        "rag_runtime",
        "config",
        "raw_trace_count",
        "model_hash",
    }
    _strict_keys(payload, required, "model")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported model schema")
    if payload["model_type"] != "atom_emergent_phase_law":
        raise ValueError("Unexpected model type")
    check = dict(payload)
    supplied_hash = check.pop("model_hash")
    if supplied_hash != stable_hash(check):
        raise ValueError("Model hash mismatch")
    if payload["wiki_runtime"] != ATOM_WIKI_GRAPH_RUNTIME:
        raise ValueError("Wiki runtime marker mismatch")
    if payload["rag_runtime"] != ATOM_RAG_RUNTIME:
        raise ValueError("RAG runtime marker mismatch")
    if payload["raw_trace_count"] != 0:
        raise ValueError("Exported model must not retain raw transition traces")
    if not isinstance(payload["config"], dict):
        raise ValueError("config must be an object")
    config = PhaseConfig(**payload["config"])
    config.validate()
    if payload["lattice_size"] != config.lattice_size:
        raise ValueError("lattice size diverges from config")
    if (
        not isinstance(payload["symbols"], list)
        or len(payload["symbols"]) != config.lattice_size
    ):
        raise ValueError("symbols must cover the phase lattice")
    symbols = []
    for index, row in enumerate(payload["symbols"]):
        if not isinstance(row, dict):
            raise ValueError("symbol rows must be objects")
        _strict_keys(row, {"name", "slot", "mass", "support"}, f"symbol[{index}]")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError("symbol name must be non-empty text")
        if isinstance(row["slot"], bool) or not isinstance(row["slot"], int):
            raise ValueError("symbol slot must be an integer")
        symbols.append(
            SymbolPhase(
                name=row["name"],
                slot=row["slot"],
                mass=_finite_number(row["mass"], "symbol mass"),
                support=_finite_number(row["support"], "symbol support"),
            )
        )
    if sorted(row.slot for row in symbols) != list(range(config.lattice_size)):
        raise ValueError("symbol slots must be a permutation of the lattice")
    if len({row.name for row in symbols}) != len(symbols):
        raise ValueError("symbol names must be unique")
    if not isinstance(payload["laws"], list) or not payload["laws"]:
        raise ValueError("laws must be a non-empty list")
    laws = []
    law_keys = {
        "operator",
        "shift",
        "mass",
        "support",
        "coherence",
        "evidence_count",
        "evidence_digest",
        "active",
    }
    for index, row in enumerate(payload["laws"]):
        if not isinstance(row, dict):
            raise ValueError("law rows must be objects")
        _strict_keys(row, law_keys, f"law[{index}]")
        if not isinstance(row["operator"], str) or not row["operator"]:
            raise ValueError("law operator must be non-empty text")
        if (
            isinstance(row["shift"], bool)
            or not isinstance(row["shift"], int)
            or not 0 <= row["shift"] < config.lattice_size
        ):
            raise ValueError("law shift must be within the lattice")
        if (
            isinstance(row["evidence_count"], bool)
            or not isinstance(row["evidence_count"], int)
            or row["evidence_count"] <= 0
        ):
            raise ValueError("law evidence_count must be positive")
        if (
            not isinstance(row["evidence_digest"], str)
            or len(row["evidence_digest"]) != 64
        ):
            raise ValueError("law evidence_digest must be SHA-256 text")
        if row["active"] is not True:
            raise ValueError("exported laws must be active")
        laws.append(
            OperatorLaw(
                operator=row["operator"],
                shift=row["shift"],
                mass=_finite_number(row["mass"], "law mass"),
                support=_finite_number(row["support"], "law support"),
                coherence=_finite_number(row["coherence"], "law coherence"),
                evidence_count=row["evidence_count"],
                evidence_digest=row["evidence_digest"],
                active=True,
            )
        )
    if len({row.operator for row in laws}) != len(laws):
        raise ValueError("operator laws must be unique")
    state = PhaseState(
        symbols=tuple(symbols),
        laws=tuple(laws),
        traces=(),
        tick=0,
        temperature=_finite_number(payload["temperature"], "temperature"),
        phase_fields=(),
        field_ready=False,
        wave_slot=None,
        wave_strength=0.0,
        candidate_scores=(),
        prediction=None,
        confidence=0.0,
        energy=0.0,
        phase_energy=0.0,
        cumulative_phase_energy=0.0,
        maximum_phase_energy=0.0,
        accepted_improving_moves=0,
        accepted_worse_moves=0,
        observations=0,
        predictions=0,
        forgotten=0,
        operator_counts=(),
        outcome_counts=(),
        last_outcome="restored",
        transition_hash=str(payload["transition_hash"]),
        transitions=0,
    )
    return PhaseRuntime(
        [row.name for row in symbols],
        kernel=UniversePhaseKernel(config),
        state=state,
    )


def validate_prediction_request(
    payload: Mapping[str, Any], symbols: set[str], operators: set[str]
) -> None:
    _strict_keys(payload, {"request_id", "queries"}, "request")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be non-empty text")
    if (
        not isinstance(payload["queries"], list)
        or not 1 <= len(payload["queries"]) <= 10_000
    ):
        raise ValueError("queries must be a list containing 1 to 10000 rows")
    query_ids = set()
    for index, row in enumerate(payload["queries"]):
        if not isinstance(row, dict):
            raise ValueError("query rows must be objects")
        _strict_keys(row, {"query_id", "source", "operators"}, f"query[{index}]")
        if not isinstance(row["query_id"], str) or not row["query_id"]:
            raise ValueError("query_id must be non-empty text")
        if row["query_id"] in query_ids:
            raise ValueError("query_id values must be unique")
        query_ids.add(row["query_id"])
        if row["source"] not in symbols:
            raise ValueError(f"Unknown query source: {row['source']}")
        if (
            not isinstance(row["operators"], list)
            or not 1 <= len(row["operators"]) <= 32
        ):
            raise ValueError("query operators must contain 1 to 32 values")
        unknown = set(row["operators"]) - operators
        if unknown:
            raise ValueError(f"Unknown query operators: {sorted(unknown)}")


def run_serialized_workflow(
    model_path: Path, request_path: Path, response_path: Path
) -> dict[str, Any]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime = runtime_from_model(model)
    symbols = {row.name for row in runtime.state.symbols}
    operators = {row.operator for row in runtime.state.laws}
    validate_prediction_request(request, symbols, operators)
    predictions = []
    for row in request["queries"]:
        result = runtime.predict(
            row["source"], row["operators"], event_id=row["query_id"]
        )
        predictions.append(
            {
                "query_id": row["query_id"],
                "prediction": result["symbol"] if result is not None else None,
                "confidence": result["confidence"] if result is not None else 0.0,
            }
        )
    response = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "model_hash": model["model_hash"],
        "predictions": predictions,
        "runtime": {
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "transition_hash": runtime.state.transition_hash,
            "operator_counts": dict(runtime.state.operator_counts),
        },
    }
    write_json(response_path, response)
    return response


def score_workflow(
    response: Mapping[str, Any], expected_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {row["case_id"]: row["target"] for row in expected_rows}
    predictions = {
        row["query_id"]: row["prediction"] for row in response["predictions"]
    }
    correct = sum(
        predictions.get(case_id) == target for case_id, target in expected.items()
    )
    return {
        "cases": len(expected),
        "correct": correct,
        "accuracy": correct / len(expected) if expected else 0.0,
        "passed": correct == len(expected),
    }


def architecture_audit(source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or Path(__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UniversePhaseKernel"
    )
    outside_replace = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_state_replace = isinstance(node.func, ast.Name) and node.func.id == "replace"
        if is_state_replace and not kernel.lineno <= node.lineno <= kernel.end_lineno:
            outside_replace.append(node.lineno)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    runtime_calls = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    graph = AtomWikiGraph()
    expanded = {
        name: list(graph.expand(name))
        for name in (
            "phase_mix",
            "thermal_anneal",
            "attention",
            "learn",
            "remember",
            "forget",
            "retrieve",
            "revise",
            "abstract",
        )
    }
    checks = {
        "seven_universe_primitives": len(tuple(Primitive)) == 7,
        "only_kernel_replaces_state": not outside_replace,
        "all_recipe_leaves_are_universe_primitives": all(
            leaves and set(leaves) <= set(UNIVERSE_PRIMITIVE_NAMES)
            for leaves in expanded.values()
        ),
        "no_neural_framework_import": not {
            "torch",
            "tensorflow",
            "jax",
        }
        & imported_roots,
        "no_gradient_or_backprop_runtime": not {
            "backward",
            "gradient",
            "grad",
        }
        & runtime_calls,
        "wiki_graph_runtime_wired": "ATOM_WIKI_GRAPH_RUNTIME" in source,
        "rag_runtime_wired": "ATOM_RAG_RUNTIME" in source,
        "artifact_side_view_wired": "ATOM_SIDE_VIEW_RUNTIME" in source
        and "ATOM_ARTIFACT_BINDING" in source,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "replace_calls_outside_kernel": outside_replace,
        "resolved_recipes": expanded,
    }


def run_self_tests() -> dict[str, Any]:
    checks = {}
    graph = AtomWikiGraph()
    graph.assert_all_leaves_are_universe_primitives()
    checks["knowledge_graph_resolves_to_seven_primitives"] = True
    retrieved = retrieve_atom_context(
        graph, "phase interference cool thermal chaos abstract law", limit=8
    )
    retrieved_names = {row["name"] for row in retrieved}
    checks["graph_rag_retrieves_composition_context"] = {
        "phase_mix",
        "thermal_anneal",
        "abstract",
    } <= retrieved_names
    program = build_tiny_world()
    train_keys = {(row["source"], row["operators"][0]) for row in program["train"]}
    heldout_keys = {
        (row["source"], row["operators"][0]) for row in program["heldout_single_step"]
    }
    checks["heldout_pairs_are_unseen"] = not train_keys & heldout_keys
    checks["multi_step_compositions_are_unseen"] = all(
        len(row["operators"]) > 1
        for row in (*program["unseen_two_step"], *program["unseen_three_step"])
    )
    kernel = UniversePhaseKernel()
    state = kernel.initial_state(program["symbols"])
    checks["initial_lattice_is_bijective"] = sorted(
        row.slot for row in state.symbols
    ) == list(range(kernel.config.lattice_size))
    checks["substrate_is_frozen"] = PhaseState.__dataclass_params__.frozen
    checks["architecture_mutation_boundary"] = architecture_audit()["passed"]
    malformed = {
        "request_id": "bad",
        "queries": [
            {
                "query_id": "q",
                "source": program["symbols"][0],
                "operators": [],
                "extra": 1,
            }
        ],
    }
    try:
        validate_prediction_request(
            malformed, set(program["symbols"]), set(program["operators"])
        )
    except ValueError:
        checks["strict_request_rejects_unknown_shape"] = True
    else:
        checks["strict_request_rejects_unknown_shape"] = False
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failed, "failed": failed, "checks": checks}


def config_with(base: PhaseConfig | None = None, **overrides: Any) -> PhaseConfig:
    values = asdict(base or PhaseConfig())
    values.update(overrides)
    config = PhaseConfig(**values)
    config.validate()
    return config


def run_ablation(
    program: Mapping[str, Any], primitive: Primitive, config: PhaseConfig
) -> dict[str, Any]:
    runtime, history = train_phase_model(program, config=config, disabled=(primitive,))
    pre = evaluate_rows(
        runtime,
        program["heldout_single_step"],
        f"ablate-{primitive.value}",
        allow_unknown_laws=True,
    )
    mass_excess = max(
        0.0, runtime.state.information_mass - config.information_mass_budget
    )
    temperature_drop = config.initial_temperature - runtime.state.temperature
    runtime.consolidate(event_id=f"ablate-{primitive.value}-abstract")
    raw_traces = len(runtime.state.traces)
    signals = {
        Primitive.RADIATION: pre["accuracy"] < 0.70,
        Primitive.GRAVITATION: pre["accuracy"] < 0.70,
        Primitive.ATTRACTION_REPULSION: pre["accuracy"] < 0.70,
        Primitive.NUCLEATION: pre["accuracy"] < 0.70,
        Primitive.DISSIPATION: temperature_drop <= 1e-12,
        Primitive.CONSERVATION: mass_excess > 1e-6,
        Primitive.DECAY: raw_traces > 0,
    }
    return {
        "primitive": primitive.value,
        "heldout_accuracy": pre["accuracy"],
        "final_energy": history[-1]["energy"],
        "temperature_drop": temperature_drop,
        "information_mass_excess": mass_excess,
        "raw_traces_after_abstraction": raw_traces,
        "causal_effect_observed": bool(signals[primitive]),
    }


def experiment_gates(report: Mapping[str, Any], config: PhaseConfig) -> dict[str, Any]:
    evaluation = report["evaluation"]
    baselines = report["baselines"]
    chaos = report["controlled_chaos"]
    gates = {
        "seven_primitives_are_sole_mutators": bool(
            report["architecture_audit"]["passed"]
        ),
        "wiki_graph_and_graph_rag_are_runtime_wired": bool(
            report["knowledge_runtime"]["passed"]
        ),
        "all_seven_primitives_exercised": set(report["training"]["operator_counts"])
        == set(UNIVERSE_PRIMITIVE_NAMES)
        and all(report["training"]["operator_counts"].values()),
        "every_primitive_has_causal_ablation": all(
            row["causal_effect_observed"]
            for row in report["primitive_ablations"].values()
        ),
        "training_field_settles": report["training"]["final_energy"] <= 0.05,
        "learned_observed_transitions": evaluation["train"]["accuracy"] >= 0.95,
        "generalized_unseen_single_steps": evaluation["heldout_single_step"]["accuracy"]
        >= 0.70,
        "composed_unseen_two_steps": evaluation["unseen_two_step"]["accuracy"] >= 0.70,
        "composed_unseen_three_steps": evaluation["unseen_three_step"]["accuracy"]
        >= 0.60,
        "beats_exact_table_on_heldout": evaluation["heldout_single_step"]["accuracy"]
        > baselines["exact_table"]["heldout_single_step"]["accuracy"],
        "beats_exact_table_on_two_step": evaluation["unseen_two_step"]["accuracy"]
        > baselines["exact_table"]["unseen_two_step"]["accuracy"],
        "raw_examples_coarse_grained_into_laws": report["training"][
            "raw_traces_after_abstraction"
        ]
        == 0
        and report["training"]["compression_ratio"] >= 4.0,
        "conservation_budget_holds": report["training"]["information_mass_excess"]
        <= 1e-9,
        "phase_mixing_is_active_bounded_and_causal": chaos["phase_active"]
        and chaos["phase_bounded"]
        and chaos["phase_changes_trajectory"],
        "thermal_annealing_cools_and_changes_trajectory": chaos["temperature_monotonic"]
        and chaos["temperature_drop"] > 0.0
        and chaos["thermal_changes_trajectory"],
        "training_is_deterministic": chaos["deterministic_replay"],
        "serialized_model_reloads": bool(report["serialized_model_reloads"]),
        "serialized_workflow_runs": bool(report["serialized_workflow"]["passed"]),
    }
    return {"gates": gates, "passed": all(gates.values())}


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    self_tests = run_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Self-tests failed: {self_tests['failed']}")
    config = PhaseConfig()
    program = build_tiny_world()
    write_jsonl(output_dir / "phase_law_train.jsonl", program["train"])
    write_jsonl(output_dir / "phase_law_validation.jsonl", program["validation"])
    write_jsonl(
        output_dir / "phase_law_heldout_single_step.jsonl",
        program["heldout_single_step"],
    )
    write_jsonl(
        output_dir / "phase_law_unseen_two_step.jsonl", program["unseen_two_step"]
    )
    write_jsonl(
        output_dir / "phase_law_unseen_three_step.jsonl", program["unseen_three_step"]
    )

    runtime, history = train_phase_model(program, config=config)
    temperatures = [row["temperature"] for row in history]
    pre_abstraction = {
        "train": evaluate_rows(runtime, program["train"], "train"),
        "validation": evaluate_rows(runtime, program["validation"], "validation"),
        "heldout_single_step": evaluate_rows(
            runtime, program["heldout_single_step"], "heldout"
        ),
        "unseen_two_step": evaluate_rows(
            runtime, program["unseen_two_step"], "two-step"
        ),
        "unseen_three_step": evaluate_rows(
            runtime, program["unseen_three_step"], "three-step"
        ),
    }
    raw_before = len(runtime.state.traces)
    phase_energy_before = runtime.state.cumulative_phase_energy
    maximum_phase_energy = runtime.state.maximum_phase_energy
    accepted_improving = runtime.state.accepted_improving_moves
    accepted_worse = runtime.state.accepted_worse_moves
    runtime.consolidate()
    raw_after = len(runtime.state.traces)
    post_abstraction = {
        "heldout_single_step": evaluate_rows(
            runtime, program["heldout_single_step"], "post-heldout"
        ),
        "unseen_two_step": evaluate_rows(
            runtime, program["unseen_two_step"], "post-two-step"
        ),
        "unseen_three_step": evaluate_rows(
            runtime, program["unseen_three_step"], "post-three-step"
        ),
    }
    model = model_payload(runtime, program)
    model_path = output_dir / "atom_phase_law_model.json"
    write_json(model_path, model)
    restored = runtime_from_model(json.loads(model_path.read_text(encoding="utf-8")))
    restored_model = model_payload(restored, program)
    serialized_model_reloads = restored_model["model_hash"] == model["model_hash"]

    workflow_rows = [
        *program["heldout_single_step"],
        *program["unseen_two_step"][:16],
        *program["unseen_three_step"][:16],
    ]
    request = {
        "request_id": "atom-phase-law-real-workflow-003",
        "queries": [
            {
                "query_id": row["case_id"],
                "source": row["source"],
                "operators": row["operators"],
            }
            for row in workflow_rows
        ],
    }
    request_path = output_dir / "phase_law_workflow_request.json"
    response_path = output_dir / "phase_law_workflow_response.json"
    write_json(request_path, request)
    workflow_response = run_serialized_workflow(model_path, request_path, response_path)
    workflow_score = score_workflow(workflow_response, workflow_rows)

    table = ExactTransitionTable(program["train"])
    table_metrics = {
        name: evaluate_table(table, program[name])
        for name in (
            "heldout_single_step",
            "unseen_two_step",
            "unseen_three_step",
        )
    }

    replay, replay_history = train_phase_model(program, config=config)
    replay.consolidate(event_id="abstract-final")
    replay_model = model_payload(replay, program)
    zero_phase_config = config_with(config, phase_mix_strength=0.0)
    zero_phase, zero_phase_history = train_phase_model(
        program, config=zero_phase_config
    )
    no_cooling_config = config_with(
        config,
        temperature_floor=config.initial_temperature,
        cooling_rate=1.0,
    )
    no_cooling, no_cooling_history = train_phase_model(
        program, config=no_cooling_config
    )
    trajectory_hash = stable_hash(history)
    phase_zero_hash = stable_hash(zero_phase_history)
    no_cooling_hash = stable_hash(no_cooling_history)

    ablations = {
        primitive.value: run_ablation(program, primitive, config)
        for primitive in Primitive
    }
    graph = runtime.knowledge
    retrieval = retrieve_atom_context(
        graph,
        "phase interference controlled chaos cooling abstraction law memory",
        limit=9,
    )
    retrieval_names = {row["name"] for row in retrieval}
    knowledge_runtime = {
        "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
        "rag_runtime": ATOM_RAG_RUNTIME,
        "graph_hash": stable_hash(graph.manifest()),
        "retrieval_query": "phase interference controlled chaos cooling abstraction law memory",
        "retrieved": retrieval,
        "passed": {"phase_mix", "thermal_anneal", "abstract"} <= retrieval_names,
    }

    initial_nonzero_energy = next(
        (row["energy"] for row in history if row["energy"] > 1e-12), 0.0
    )
    information_mass_excess = max(
        0.0, runtime.state.information_mass - config.information_mass_budget
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "atom_emergent_phase_law_v3",
        "model_hash": model["model_hash"],
        "manifest": {
            "seed": SEED,
            "standard_neural_network": False,
            "gradient_descent": False,
            "backpropagation": False,
            "pretrained_model": False,
            "trainable_weight_matrix": False,
            "state_transition_authority": "UniversePhaseKernel only",
            "universe_primitives": list(UNIVERSE_PRIMITIVE_NAMES),
            "cognitive_composition": graph.manifest()["composition"],
            "dataset_hashes": {
                "train": stable_hash(program["train"]),
                "validation": stable_hash(program["validation"]),
                "heldout": stable_hash(program["heldout_single_step"]),
                "two_step": stable_hash(program["unseen_two_step"]),
                "three_step": stable_hash(program["unseen_three_step"]),
            },
            "artifact_side_view_runtime": ATOM_SIDE_VIEW_RUNTIME,
            "artifact_binding": ATOM_ARTIFACT_BINDING,
        },
        "self_tests": self_tests,
        "architecture_audit": architecture_audit(),
        "knowledge_runtime": knowledge_runtime,
        "training": {
            "epochs": config.epochs,
            "examples": len(program["train"]),
            "one_off_noise_examples": len(program["one_off_noise"]),
            "initial_nonzero_energy": initial_nonzero_energy,
            "final_energy": history[-1]["energy"],
            "energy_reduction": initial_nonzero_energy - history[-1]["energy"],
            "temperature_history": temperatures,
            "accepted_improving_moves": accepted_improving,
            "accepted_worse_moves": accepted_worse,
            "raw_traces_before_abstraction": raw_before,
            "raw_traces_after_abstraction": raw_after,
            "persistent_laws": len(model["laws"]),
            "compression_ratio": len(program["train"]) / max(1, len(model["laws"])),
            "information_mass": runtime.state.information_mass,
            "information_mass_excess": information_mass_excess,
            "operator_counts": dict(runtime.state.operator_counts),
            "outcome_counts": dict(runtime.state.outcome_counts),
            "history_hash": trajectory_hash,
        },
        "evaluation": pre_abstraction,
        "post_abstraction_evaluation": post_abstraction,
        "baselines": {"exact_table": table_metrics},
        "controlled_chaos": {
            "initial_temperature": config.initial_temperature,
            "final_temperature": history[-1]["temperature"],
            "temperature_drop": config.initial_temperature - history[-1]["temperature"],
            "temperature_monotonic": all(
                later <= earlier + 1e-12
                for earlier, later in zip(temperatures, temperatures[1:])
            ),
            "cumulative_phase_energy": phase_energy_before,
            "maximum_phase_energy": maximum_phase_energy,
            "phase_active": phase_energy_before > 0.0,
            "phase_bounded": maximum_phase_energy <= config.phase_mix_strength + 1e-9,
            "phase_changes_trajectory": trajectory_hash != phase_zero_hash,
            "thermal_changes_trajectory": trajectory_hash != no_cooling_hash,
            "accepted_worse_moves": accepted_worse,
            "deterministic_replay": replay_model["model_hash"] == model["model_hash"]
            and stable_hash(replay_history) == trajectory_hash,
            "zero_phase_history_hash": phase_zero_hash,
            "no_cooling_history_hash": no_cooling_hash,
        },
        "primitive_ablations": ablations,
        "serialized_model_reloads": serialized_model_reloads,
        "serialized_workflow": workflow_score,
        "model": model,
        "execution_tail": [asdict(row) for row in runtime.records[-16:]],
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["experiment_gates"] = experiment_gates(report, config)
    report_path = output_dir / "atom_phase_law_report.json"
    write_json(report_path, report)
    side_view_path = render_phase_artifact(
        report, model, output_dir / "atom_phase_law_side_view.html"
    )
    report["artifact_side_view"] = {
        "path": side_view_path.name,
        "sha256": hashlib.sha256(side_view_path.read_bytes()).hexdigest(),
        "runtime_marker": ATOM_SIDE_VIEW_RUNTIME,
        "artifact_binding_marker": ATOM_ARTIFACT_BINDING,
        "model_hash_bound": model["model_hash"],
    }
    write_json(report_path, report)
    write_json(output_dir / "atom_phase_law_manifest.json", report["manifest"])
    write_json(output_dir / "atom_phase_law_training_history.json", history)
    write_json(output_dir / "atom_phase_law_knowledge_graph.json", graph.manifest())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/kaggle/working")
            if Path("/kaggle/working").is_dir()
            else Path("phase_law_outputs")
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request is not None:
        if args.model is None or args.response is None:
            raise ValueError("--request requires --model and --response")
        response = run_serialized_workflow(args.model, args.request, args.response)
        print(json.dumps(response, indent=2, sort_keys=True))
        return
    if args.model is not None or args.response is not None:
        raise ValueError("--model and --response are only valid with --request")
    if args.self_test:
        result = run_self_tests()
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    report = run_experiment(args.output_dir)
    summary = {
        "experiment": report["experiment"],
        "model_hash": report["model_hash"],
        "training": {
            key: report["training"][key]
            for key in (
                "initial_nonzero_energy",
                "final_energy",
                "accepted_improving_moves",
                "accepted_worse_moves",
                "raw_traces_after_abstraction",
                "persistent_laws",
                "compression_ratio",
            )
        },
        "evaluation": {
            name: {
                "accuracy": row["accuracy"],
                "coverage": row["coverage"],
                "cases": row["cases"],
            }
            for name, row in report["evaluation"].items()
        },
        "controlled_chaos": report["controlled_chaos"],
        "ablations": {
            name: row["causal_effect_observed"]
            for name, row in report["primitive_ablations"].items()
        },
        "experiment_gates": report["experiment_gates"],
        "artifact_side_view": report["artifact_side_view"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
