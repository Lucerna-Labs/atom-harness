"""Universe-core Atom experiment.

The substrate is immutable. Only seven universe primitives can produce a new
substrate state. Cognitive atoms are declarative recipes that expand into those
seven operations; they never mutate traces directly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
SEED = 20260721
SYMBOL_DIM = 8
CONTEXT_DIM = 4
CUE_DIM = SYMBOL_DIM + CONTEXT_DIM
VALUE_COUNT = 4


class Primitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


UNIVERSE_PRIMITIVES = tuple(primitive.value for primitive in Primitive)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_unit(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    return 2.0 * (integer / ((1 << 64) - 1)) - 1.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(value: Sequence[float]) -> float:
    return math.sqrt(dot(value, value))


def normalized(value: Sequence[float]) -> tuple[float, ...]:
    magnitude = norm(value)
    if magnitude <= 1e-12:
        raise ValueError("Cannot normalize an all-zero vector")
    return tuple(component / magnitude for component in value)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator > 1e-12 else 0.0


def softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def one_hot(index: int, size: int = VALUE_COUNT) -> tuple[float, ...]:
    if not 0 <= index < size:
        raise ValueError(f"Value index must be within [0, {size})")
    return tuple(1.0 if position == index else 0.0 for position in range(size))


def hadamard(order: int) -> list[list[float]]:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = [[1.0]]
    while len(matrix) < order:
        top = [row + row for row in matrix]
        bottom = [row + [-value for value in row] for row in matrix]
        matrix = top + bottom
    return matrix


HADAMARD_SYMBOLS = hadamard(SYMBOL_DIM)
CONTEXT_CODES = (
    [1.0, 1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0, 1.0],
)


def encode_cue(symbol: int, context: int) -> list[float]:
    if not 0 <= symbol < SYMBOL_DIM:
        raise ValueError(f"symbol must be within [0, {SYMBOL_DIM})")
    if context not in (0, 1):
        raise ValueError("context must be 0 or 1")
    return HADAMARD_SYMBOLS[symbol] + CONTEXT_CODES[context]


def masked_cue(cue: Sequence[float], positions: Sequence[int]) -> list[float]:
    result = list(cue)
    for position in positions:
        if not 0 <= position < SYMBOL_DIM:
            raise ValueError("Only symbol dimensions may be masked")
        result[position] = 0.0
    return result


def validate_cue(cue: Sequence[float]) -> None:
    if not isinstance(cue, (list, tuple)):
        raise ValueError("cue must be a numeric list")
    if len(cue) != CUE_DIM:
        raise ValueError(f"cue must contain exactly {CUE_DIM} values")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in cue
    ):
        raise ValueError("cue values must be numeric")
    if any(not math.isfinite(float(value)) for value in cue):
        raise ValueError("cue values must be finite")
    if any(not -1.0 <= float(value) <= 1.0 for value in cue):
        raise ValueError("cue values must be within [-1, 1]")
    if norm(cue) <= 1e-12:
        raise ValueError("cue cannot be all zero")


@dataclass(frozen=True)
class AtomRecipe:
    name: str
    steps: tuple[str, ...]
    purpose: str


DEFAULT_RECIPES = (
    AtomRecipe(
        "phase_mix",
        (Primitive.RADIATION.value, Primitive.GRAVITATION.value),
        "mix bounded signal phases, then let mass shape the attractor field",
    ),
    AtomRecipe(
        "attention",
        ("phase_mix", Primitive.ATTRACTION_REPULSION.value),
        "propagate a signal, form an attractor field, then bind or reject",
    ),
    AtomRecipe(
        "thermal_anneal",
        (Primitive.DISSIPATION.value, Primitive.CONSERVATION.value),
        "cool exploratory variation while preserving the mass invariant",
    ),
    AtomRecipe(
        "forget",
        ("thermal_anneal", Primitive.DECAY.value),
        "let time remove unsupported structure while preserving the mass invariant",
    ),
    AtomRecipe(
        "learn",
        (
            "attention",
            Primitive.NUCLEATION.value,
            Primitive.CONSERVATION.value,
        ),
        "bind recurring evidence or form a new persistent structure",
    ),
    AtomRecipe(
        "remember",
        ("forget", "learn"),
        "apply time, then reshape the substrate around an experience",
    ),
    AtomRecipe(
        "retrieve",
        ("attention",),
        "allow a cue to settle into a learned attractor",
    ),
    AtomRecipe(
        "revise",
        ("remember",),
        "let contradictory evidence reshape the same persistent structure",
    ),
    AtomRecipe(
        "abstract",
        ("remember",),
        "collapse repeated related experiences into a shared structure",
    ),
)


class RecipeBook:
    def __init__(self, recipes: Sequence[AtomRecipe] = DEFAULT_RECIPES) -> None:
        self._recipes = {recipe.name: recipe for recipe in recipes}
        if len(self._recipes) != len(recipes):
            raise ValueError("Recipe names must be unique")
        if not self._recipes:
            raise ValueError("At least one recipe is required")
        for name in self._recipes:
            self.expand(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._recipes))

    def expand(self, name: str) -> tuple[Primitive, ...]:
        if name not in self._recipes:
            raise ValueError(f"Unknown cognitive atom: {name}")

        def visit(item: str, ancestry: tuple[str, ...]) -> tuple[Primitive, ...]:
            if item in UNIVERSE_PRIMITIVES:
                return (Primitive(item),)
            if item not in self._recipes:
                raise ValueError(f"Unknown recipe step: {item}")
            if item in ancestry:
                chain = " -> ".join((*ancestry, item))
                raise ValueError(f"Cyclic Atom composition: {chain}")
            expanded: list[Primitive] = []
            for step in self._recipes[item].steps:
                expanded.extend(visit(step, (*ancestry, item)))
            return tuple(expanded)

        return visit(name, ())

    def manifest(self) -> dict[str, Any]:
        return {
            name: {
                "declared_steps": list(self._recipes[name].steps),
                "expanded_primitives": [step.value for step in self.expand(name)],
                "purpose": self._recipes[name].purpose,
            }
            for name in self.names
        }


@dataclass(frozen=True)
class UniverseConfig:
    capacity: int = 18
    match_threshold: float = 0.72
    query_threshold: float = 0.68
    attention_temperature: float = 11.0
    cue_learning_rate: float = 0.24
    evidence_rate: float = 1.0
    reinforcement_mass: float = 0.45
    reinforcement_support: float = 0.22
    initial_mass: float = 0.50
    initial_support: float = 0.40
    initial_ttl: float = 16.0
    ttl_per_reinforcement: float = 2.0
    dissipation_rate: float = 0.030
    mass_decay_rate: float = 0.008
    expiration_support: float = 0.10
    conservation_budget: float = 12.0
    minimum_reliability: float = 0.16
    phase_mix_strength: float = 0.035
    anneal_start_temperature: float = 1.35
    anneal_floor_temperature: float = 0.72
    anneal_cooling_rate: float = 0.94
    chaos_seed: int = SEED

    def validate(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        for name in (
            "match_threshold",
            "query_threshold",
            "cue_learning_rate",
            "initial_support",
            "expiration_support",
            "minimum_reliability",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in (
            "attention_temperature",
            "evidence_rate",
            "reinforcement_mass",
            "reinforcement_support",
            "initial_mass",
            "initial_ttl",
            "ttl_per_reinforcement",
            "conservation_budget",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("dissipation_rate", "mass_decay_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value < 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1)")
        if (
            isinstance(self.phase_mix_strength, bool)
            or not isinstance(self.phase_mix_strength, (int, float))
            or not math.isfinite(self.phase_mix_strength)
            or not 0.0 <= self.phase_mix_strength <= 0.25
        ):
            raise ValueError("phase_mix_strength must be finite and within [0, 0.25]")
        for name in ("anneal_start_temperature", "anneal_floor_temperature"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.anneal_floor_temperature > self.anneal_start_temperature:
            raise ValueError("anneal temperature floor cannot exceed its start")
        if (
            isinstance(self.anneal_cooling_rate, bool)
            or not isinstance(self.anneal_cooling_rate, (int, float))
            or not math.isfinite(self.anneal_cooling_rate)
            or not 0.0 < self.anneal_cooling_rate <= 1.0
        ):
            raise ValueError("anneal_cooling_rate must be within (0, 1]")
        if (
            isinstance(self.chaos_seed, bool)
            or not isinstance(self.chaos_seed, int)
            or self.chaos_seed < 0
        ):
            raise ValueError("chaos_seed must be a non-negative integer")


@dataclass(frozen=True)
class Trace:
    trace_id: int
    cue: tuple[float, ...]
    evidence: tuple[float, ...]
    mass: float
    support: float
    ttl: float
    hits: int
    contradictions: int
    created_tick: int
    last_tick: int
    active: bool = True

    @property
    def value(self) -> int:
        return max(range(len(self.evidence)), key=self.evidence.__getitem__)

    @property
    def distribution(self) -> tuple[float, ...]:
        total = sum(self.evidence)
        if total <= 1e-12:
            return tuple(1.0 / len(self.evidence) for _ in self.evidence)
        return tuple(value / total for value in self.evidence)

    @property
    def reliability(self) -> float:
        hit_strength = 1.0 - math.exp(-0.55 * self.hits)
        mass_strength = min(1.0, self.mass)
        return max(
            0.0,
            min(
                1.0,
                0.45 * self.support + 0.30 * mass_strength + 0.25 * hit_strength,
            ),
        )


@dataclass(frozen=True)
class Stimulus:
    mode: str
    cue: tuple[float, ...] | None = None
    value: int | None = None
    salience: float = 1.0
    event_id: str = ""

    def validate(self) -> None:
        if self.mode not in {"observe", "query", "idle"}:
            raise ValueError("Stimulus mode must be observe, query, or idle")
        if self.mode == "idle":
            if self.cue is not None or self.value is not None:
                raise ValueError("Idle stimuli cannot carry a cue or value")
            return
        if self.cue is None:
            raise ValueError("Observe and query stimuli require a cue")
        validate_cue(self.cue)
        if self.mode == "observe":
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, int)
                or not 0 <= self.value < VALUE_COUNT
            ):
                raise ValueError("Observe stimuli require a valid value")
            if (
                isinstance(self.salience, bool)
                or not isinstance(self.salience, (int, float))
                or not math.isfinite(self.salience)
                or not 0.1 <= self.salience <= 2.0
            ):
                raise ValueError("salience must be finite and within [0.1, 2.0]")
        elif self.value is not None:
            raise ValueError("Query stimuli cannot carry a value")


@dataclass(frozen=True)
class FieldState:
    traces: tuple[Trace, ...]
    tick: int
    activations: tuple[float, ...]
    attractor_weights: tuple[float, ...]
    candidate_id: int | None
    prediction: int | None
    confidence: float
    resonance: float
    temperature: float
    phase_energy: float
    cumulative_phase_energy: float
    observations: int
    queries: int
    forgotten: int
    operator_counts: tuple[tuple[str, int], ...]
    outcome_counts: tuple[tuple[str, int], ...]
    last_outcome: str
    transition_hash: str
    transitions: int

    @property
    def active_traces(self) -> tuple[Trace, ...]:
        return tuple(trace for trace in self.traces if trace.active)

    @property
    def total_active_mass(self) -> float:
        return sum(trace.mass for trace in self.active_traces)


def counter_tuple(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(name), int(value)) for name, value in values.items()))


def bump_counter(
    values: tuple[tuple[str, int], ...], name: str, amount: int = 1
) -> tuple[tuple[str, int], ...]:
    counter = Counter(dict(values))
    counter[name] += amount
    return counter_tuple(counter)


class UniverseKernel:
    """The sole state-transition authority for the immutable substrate."""

    def __init__(
        self,
        config: UniverseConfig | None = None,
        disabled: Iterable[Primitive] = (),
    ) -> None:
        self.config = config or UniverseConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self) -> FieldState:
        return FieldState(
            traces=(),
            tick=0,
            activations=(),
            attractor_weights=(),
            candidate_id=None,
            prediction=None,
            confidence=0.0,
            resonance=0.0,
            temperature=self.config.anneal_start_temperature,
            phase_energy=0.0,
            cumulative_phase_energy=0.0,
            observations=0,
            queries=0,
            forgotten=0,
            operator_counts=(),
            outcome_counts=(),
            last_outcome="initial",
            transition_hash="0" * 64,
            transitions=0,
        )

    def apply(
        self, state: FieldState, primitive: Primitive, stimulus: Stimulus
    ) -> FieldState:
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
        counts = bump_counter(transitioned.operator_counts, primitive.value)
        digest = stable_hash(
            {
                "previous": state.transition_hash,
                "primitive": primitive.value,
                "tick": transitioned.tick,
                "traces": [
                    {
                        "id": trace.trace_id,
                        "value": trace.value,
                        "mass": trace.mass,
                        "support": trace.support,
                        "ttl": trace.ttl,
                        "hits": trace.hits,
                        "active": trace.active,
                    }
                    for trace in transitioned.traces
                ],
                "prediction": transitioned.prediction,
                "outcome": transitioned.last_outcome,
                "temperature": transitioned.temperature,
                "phase_energy": transitioned.phase_energy,
            }
        )
        return replace(
            transitioned,
            operator_counts=counts,
            transition_hash=digest,
            transitions=state.transitions + 1,
        )

    def _radiation(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        if stimulus.mode == "idle":
            return replace(
                state,
                activations=tuple(0.0 for _ in state.traces),
                attractor_weights=tuple(0.0 for _ in state.traces),
                candidate_id=None,
                prediction=None,
                confidence=0.0,
                resonance=0.0,
                phase_energy=0.0,
            )
        assert stimulus.cue is not None
        raw_activations = tuple(
            cosine_similarity(stimulus.cue, trace.cue) if trace.active else 0.0
            for trace in state.traces
        )
        active_indices = [
            index for index, trace in enumerate(state.traces) if trace.active
        ]
        active_mean = (
            sum(raw_activations[index] for index in active_indices)
            / len(active_indices)
            if active_indices
            else 0.0
        )
        temperature_span = (
            self.config.anneal_start_temperature - self.config.anneal_floor_temperature
        )
        thermal_ratio = (
            clamp(
                (state.temperature - self.config.anneal_floor_temperature)
                / temperature_span,
                0.0,
                1.0,
            )
            if temperature_span > 1e-12
            else 0.0
        )
        effective_strength = self.config.phase_mix_strength * (
            0.20 + 0.80 * thermal_ratio
        )
        activations = []
        phase_deltas = []
        for index, (raw, trace) in enumerate(
            zip(raw_activations, state.traces, strict=True)
        ):
            if not trace.active:
                activations.append(0.0)
                continue
            phase = deterministic_unit(
                self.config.chaos_seed,
                stimulus.event_id,
                state.tick,
                trace.trace_id,
            )
            interference = 0.55 * (active_mean - raw) + 0.45 * phase * (1.0 - abs(raw))
            delta = clamp(
                effective_strength * interference,
                -effective_strength,
                effective_strength,
            )
            activations.append(clamp(raw + delta, -1.0, 1.0))
            phase_deltas.append(abs(delta))
        phase_energy = sum(phase_deltas) / len(phase_deltas) if phase_deltas else 0.0
        return replace(
            state,
            activations=tuple(activations),
            attractor_weights=(),
            candidate_id=None,
            prediction=None,
            confidence=0.0,
            resonance=max(activations, default=0.0),
            phase_energy=phase_energy,
            cumulative_phase_energy=state.cumulative_phase_energy + phase_energy,
            last_outcome="radiate",
        )

    def _gravitation(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        del stimulus
        if len(state.activations) != len(state.traces):
            return replace(state, attractor_weights=(), last_outcome="no_field")
        active_indices = [
            index for index, trace in enumerate(state.traces) if trace.active
        ]
        if not active_indices:
            return replace(
                state,
                attractor_weights=tuple(0.0 for _ in state.traces),
                last_outcome="empty_field",
            )
        logits = [
            self.config.attention_temperature
            * state.activations[index]
            / state.temperature
            + math.log(max(1e-8, state.traces[index].reliability))
            + 0.15 * math.log(max(1e-8, state.traces[index].mass))
            for index in active_indices
        ]
        active_weights = softmax(logits)
        weights = [0.0] * len(state.traces)
        for index, weight in zip(active_indices, active_weights, strict=True):
            weights[index] = weight
        return replace(
            state,
            attractor_weights=tuple(weights),
            last_outcome="gravitate",
        )

    def _attraction_repulsion(
        self, state: FieldState, stimulus: Stimulus
    ) -> FieldState:
        if stimulus.mode == "idle":
            return state
        observation_delta = 1 if stimulus.mode == "observe" else 0
        query_delta = 1 if stimulus.mode == "query" else 0
        field_ready = (
            len(state.activations) == len(state.traces)
            and len(state.attractor_weights) == len(state.traces)
            and any(trace.active for trace in state.traces)
        )
        threshold = (
            self.config.match_threshold
            if stimulus.mode == "observe"
            else self.config.query_threshold
        )
        eligible = (
            [
                index
                for index, trace in enumerate(state.traces)
                if trace.active
                and state.activations[index] >= threshold
                and trace.reliability >= self.config.minimum_reliability
            ]
            if field_ready
            else []
        )
        if not eligible:
            outcome = "repel" if stimulus.mode == "observe" else "reject"
            return replace(
                state,
                candidate_id=None,
                prediction=None,
                confidence=0.0,
                observations=state.observations + observation_delta,
                queries=state.queries + query_delta,
                outcome_counts=bump_counter(state.outcome_counts, outcome),
                last_outcome=outcome,
            )

        best_index = max(
            eligible,
            key=lambda index: (
                state.attractor_weights[index],
                state.activations[index],
                -state.traces[index].trace_id,
            ),
        )
        best_trace = state.traces[best_index]
        resonance = state.activations[best_index]
        if stimulus.mode == "query":
            distribution = [0.0] * VALUE_COUNT
            eligible_weight = sum(state.attractor_weights[index] for index in eligible)
            for index in eligible:
                weight = state.attractor_weights[index] / max(1e-12, eligible_weight)
                for value_index, probability in enumerate(
                    state.traces[index].distribution
                ):
                    distribution[value_index] += weight * probability
            prediction = max(range(VALUE_COUNT), key=distribution.__getitem__)
            return replace(
                state,
                candidate_id=best_trace.trace_id,
                prediction=prediction,
                confidence=distribution[prediction],
                resonance=resonance,
                queries=state.queries + 1,
                outcome_counts=bump_counter(state.outcome_counts, "retrieve"),
                last_outcome="retrieve",
            )

        assert stimulus.cue is not None and stimulus.value is not None
        previous_value = best_trace.value
        cue_rate = self.config.cue_learning_rate * min(1.0, stimulus.salience)
        learned_cue = normalized(
            tuple(
                (1.0 - cue_rate) * old + cue_rate * new
                for old, new in zip(
                    best_trace.cue,
                    normalized(stimulus.cue),
                    strict=True,
                )
            )
        )
        evidence = list(best_trace.evidence)
        evidence[stimulus.value] += self.config.evidence_rate * stimulus.salience
        hits = best_trace.hits + 1
        contradiction = previous_value != stimulus.value
        learned_trace = replace(
            best_trace,
            cue=learned_cue,
            evidence=tuple(evidence),
            mass=best_trace.mass + self.config.reinforcement_mass * stimulus.salience,
            support=min(
                1.0,
                best_trace.support
                + self.config.reinforcement_support * stimulus.salience,
            )
            * (0.94 if contradiction else 1.0),
            ttl=self.config.initial_ttl + self.config.ttl_per_reinforcement * hits,
            hits=hits,
            contradictions=best_trace.contradictions + int(contradiction),
            last_tick=state.tick,
        )
        traces = list(state.traces)
        traces[best_index] = learned_trace
        outcome = "revise" if contradiction else "reinforce"
        return replace(
            state,
            traces=tuple(traces),
            candidate_id=learned_trace.trace_id,
            resonance=resonance,
            observations=state.observations + 1,
            outcome_counts=bump_counter(state.outcome_counts, outcome),
            last_outcome=outcome,
        )

    def _nucleation(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        if stimulus.mode != "observe" or state.candidate_id is not None:
            return state
        if len(state.active_traces) >= self.config.capacity:
            return replace(
                state,
                outcome_counts=bump_counter(state.outcome_counts, "capacity_reject"),
                last_outcome="capacity_reject",
            )
        assert stimulus.cue is not None and stimulus.value is not None
        trace = Trace(
            trace_id=len(state.traces),
            cue=normalized(stimulus.cue),
            evidence=tuple(
                value * self.config.evidence_rate * stimulus.salience
                for value in one_hot(stimulus.value)
            ),
            mass=self.config.initial_mass * stimulus.salience,
            support=min(1.0, self.config.initial_support * stimulus.salience),
            ttl=self.config.initial_ttl,
            hits=1,
            contradictions=0,
            created_tick=state.tick,
            last_tick=state.tick,
        )
        return replace(
            state,
            traces=(*state.traces, trace),
            activations=(*state.activations, 0.0),
            attractor_weights=(*state.attractor_weights, 0.0),
            candidate_id=trace.trace_id,
            outcome_counts=bump_counter(state.outcome_counts, "nucleate"),
            last_outcome="nucleate",
        )

    def _conservation(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        del stimulus
        total = state.total_active_mass
        if total <= self.config.conservation_budget or total <= 1e-12:
            return state
        scale = self.config.conservation_budget / total
        traces = tuple(
            replace(trace, mass=trace.mass * scale) if trace.active else trace
            for trace in state.traces
        )
        return replace(
            state,
            traces=traces,
            outcome_counts=bump_counter(state.outcome_counts, "conserve_projection"),
            last_outcome="conserve_projection",
        )

    def _dissipation(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        del stimulus
        traces = []
        for trace in state.traces:
            if not trace.active:
                traces.append(trace)
                continue
            protection = 1.0 + 0.75 * max(0, trace.hits - 1)
            traces.append(
                replace(
                    trace,
                    ttl=trace.ttl - 1.0 / protection,
                    support=trace.support
                    * (
                        1.0
                        - self.config.dissipation_rate / math.sqrt(max(1, trace.hits))
                    ),
                    mass=trace.mass
                    * (1.0 - self.config.mass_decay_rate / max(1, trace.hits)),
                )
            )
        cooled_temperature = max(
            self.config.anneal_floor_temperature,
            self.config.anneal_floor_temperature
            + (state.temperature - self.config.anneal_floor_temperature)
            * self.config.anneal_cooling_rate,
        )
        return replace(
            state,
            traces=tuple(traces),
            tick=state.tick + 1,
            activations=tuple(0.0 for _ in traces),
            attractor_weights=tuple(0.0 for _ in traces),
            candidate_id=None,
            prediction=None,
            confidence=0.0,
            resonance=0.0,
            temperature=cooled_temperature,
            phase_energy=0.0,
            last_outcome="dissipate",
        )

    def _decay(self, state: FieldState, stimulus: Stimulus) -> FieldState:
        del stimulus
        forgotten = 0
        traces = []
        for trace in state.traces:
            should_decay = trace.active and (
                (trace.ttl <= 0.0 and trace.support < 0.55)
                or (
                    trace.support < self.config.expiration_support
                    and trace.mass < self.config.initial_mass
                )
            )
            if should_decay:
                traces.append(replace(trace, active=False))
                forgotten += 1
            else:
                traces.append(trace)
        if not forgotten:
            return state
        return replace(
            state,
            traces=tuple(traces),
            forgotten=state.forgotten + forgotten,
            outcome_counts=bump_counter(state.outcome_counts, "forget", forgotten),
            last_outcome="forget",
        )


@dataclass(frozen=True)
class ExecutionRecord:
    recipe: str
    event_id: str
    mode: str
    primitives: tuple[str, ...]
    before_hash: str
    after_hash: str
    outcome: str
    prediction: int | None
    active_traces: int
    total_mass: float
    temperature: float
    phase_energy: float
    cumulative_phase_energy: float


class CompositionRuntime:
    def __init__(
        self,
        kernel: UniverseKernel | None = None,
        recipes: RecipeBook | None = None,
        state: FieldState | None = None,
    ) -> None:
        self.kernel = kernel or UniverseKernel()
        self.recipes = recipes or RecipeBook()
        self.state = state or self.kernel.initial_state()
        self.records: list[ExecutionRecord] = []
        self.recipe_counts: Counter[str] = Counter()

    def execute(self, recipe: str, stimulus: Stimulus) -> FieldState:
        steps = self.recipes.expand(recipe)
        before = self.state.transition_hash
        current = self.state
        for primitive in steps:
            current = self.kernel.apply(current, primitive, stimulus)
        self.state = current
        self.recipe_counts[recipe] += 1
        self.records.append(
            ExecutionRecord(
                recipe=recipe,
                event_id=stimulus.event_id,
                mode=stimulus.mode,
                primitives=tuple(primitive.value for primitive in steps),
                before_hash=before,
                after_hash=current.transition_hash,
                outcome=current.last_outcome,
                prediction=current.prediction,
                active_traces=len(current.active_traces),
                total_mass=current.total_active_mass,
                temperature=current.temperature,
                phase_energy=current.phase_energy,
                cumulative_phase_energy=current.cumulative_phase_energy,
            )
        )
        return current

    def observe(
        self,
        cue: Sequence[float],
        value: int,
        atom: str = "remember",
        salience: float = 1.0,
        event_id: str = "",
    ) -> Mapping[str, Any]:
        state = self.execute(
            atom,
            Stimulus(
                mode="observe",
                cue=tuple(cue),
                value=value,
                salience=salience,
                event_id=event_id,
            ),
        )
        return {
            "outcome": state.last_outcome,
            "candidate_id": state.candidate_id,
            "resonance": state.resonance,
        }

    def idle(self, steps: int) -> None:
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not 0 <= steps <= 10_000
        ):
            raise ValueError("idle steps must be an integer within [0, 10000]")
        for index in range(steps):
            self.execute(
                "forget",
                Stimulus(mode="idle", event_id=f"idle-{self.state.tick}-{index}"),
            )

    def retrieve(
        self, cue: Sequence[float], event_id: str = "query"
    ) -> Mapping[str, Any] | None:
        state = self.execute(
            "retrieve",
            Stimulus(mode="query", cue=tuple(cue), event_id=event_id),
        )
        if state.prediction is None:
            return None
        return {
            "value": state.prediction,
            "confidence": state.confidence,
            "resonance": state.resonance,
            "source_trace": state.candidate_id,
        }

    def active_count(self) -> int:
        return len(self.state.active_traces)


def trace_payload(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "cue": list(trace.cue),
        "evidence": list(trace.evidence),
        "mass": trace.mass,
        "support": trace.support,
        "ttl": trace.ttl,
        "hits": trace.hits,
        "contradictions": trace.contradictions,
        "created_tick": trace.created_tick,
        "last_tick": trace.last_tick,
        "active": trace.active,
    }


def state_payload(state: FieldState) -> dict[str, Any]:
    return {
        "traces": [trace_payload(trace) for trace in state.traces],
        "tick": state.tick,
        "activations": list(state.activations),
        "attractor_weights": list(state.attractor_weights),
        "candidate_id": state.candidate_id,
        "prediction": state.prediction,
        "confidence": state.confidence,
        "resonance": state.resonance,
        "temperature": state.temperature,
        "phase_energy": state.phase_energy,
        "cumulative_phase_energy": state.cumulative_phase_energy,
        "observations": state.observations,
        "queries": state.queries,
        "forgotten": state.forgotten,
        "operator_counts": dict(state.operator_counts),
        "outcome_counts": dict(state.outcome_counts),
        "last_outcome": state.last_outcome,
        "transition_hash": state.transition_hash,
        "transitions": state.transitions,
    }


def runtime_payload(runtime: CompositionRuntime) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(runtime.kernel.config),
        "recipes": runtime.recipes.manifest(),
        "recipe_hash": stable_hash(runtime.recipes.manifest()),
        "state": state_payload(runtime.state),
        "recipe_counts": dict(sorted(runtime.recipe_counts.items())),
    }


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def runtime_from_payload(payload: Mapping[str, Any]) -> CompositionRuntime:
    required = {
        "schema_version",
        "config",
        "recipes",
        "recipe_hash",
        "state",
        "recipe_counts",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported or malformed composition state")
    if not isinstance(payload["config"], Mapping):
        raise ValueError("Malformed universe configuration")
    try:
        config = UniverseConfig(**dict(payload["config"]))
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("Malformed universe configuration") from error
    recipes = RecipeBook()
    expected_manifest = recipes.manifest()
    if payload["recipes"] != expected_manifest:
        raise ValueError("Serialized cognitive recipes do not match the runtime")
    if payload["recipe_hash"] != stable_hash(expected_manifest):
        raise ValueError("Serialized recipe hash is invalid")

    state_data = payload["state"]
    state_fields = {
        "traces",
        "tick",
        "activations",
        "attractor_weights",
        "candidate_id",
        "prediction",
        "confidence",
        "resonance",
        "temperature",
        "phase_energy",
        "cumulative_phase_energy",
        "observations",
        "queries",
        "forgotten",
        "operator_counts",
        "outcome_counts",
        "last_outcome",
        "transition_hash",
        "transitions",
    }
    if not isinstance(state_data, Mapping) or set(state_data) != state_fields:
        raise ValueError("Malformed universe field state")
    tick = _strict_nonnegative_int(state_data["tick"], "tick")
    observations = _strict_nonnegative_int(state_data["observations"], "observations")
    queries = _strict_nonnegative_int(state_data["queries"], "queries")
    forgotten = _strict_nonnegative_int(state_data["forgotten"], "forgotten")
    transitions = _strict_nonnegative_int(state_data["transitions"], "transitions")

    trace_fields = {
        "trace_id",
        "cue",
        "evidence",
        "mass",
        "support",
        "ttl",
        "hits",
        "contradictions",
        "created_tick",
        "last_tick",
        "active",
    }
    trace_rows = state_data["traces"]
    if not isinstance(trace_rows, list) or len(trace_rows) > observations:
        raise ValueError("Malformed trace collection")
    traces: list[Trace] = []
    for index, row in enumerate(trace_rows):
        if not isinstance(row, Mapping) or set(row) != trace_fields:
            raise ValueError("Malformed trace row")
        trace_id = _strict_nonnegative_int(row["trace_id"], "trace_id")
        if trace_id != index:
            raise ValueError("Trace identifiers must be sequential")
        validate_cue(row["cue"])
        evidence = row["evidence"]
        if not isinstance(evidence, list) or len(evidence) != VALUE_COUNT:
            raise ValueError("Malformed trace evidence")
        evidence_values = tuple(
            _strict_finite_number(value, "evidence") for value in evidence
        )
        if any(value < 0.0 for value in evidence_values) or sum(evidence_values) <= 0.0:
            raise ValueError("Trace evidence must be non-negative and non-empty")
        mass = _strict_finite_number(row["mass"], "mass")
        support = _strict_finite_number(row["support"], "support")
        ttl = _strict_finite_number(row["ttl"], "ttl")
        if mass < 0.0 or not 0.0 <= support <= 1.0:
            raise ValueError("Trace mass or support is outside its invariant")
        hits = _strict_nonnegative_int(row["hits"], "hits")
        contradictions = _strict_nonnegative_int(
            row["contradictions"], "contradictions"
        )
        if hits < 1 or contradictions >= hits:
            raise ValueError("Trace learning counters are inconsistent")
        created_tick = _strict_nonnegative_int(row["created_tick"], "created_tick")
        last_tick = _strict_nonnegative_int(row["last_tick"], "last_tick")
        if not 0 <= created_tick <= last_tick <= tick:
            raise ValueError("Trace timing is inconsistent")
        if not isinstance(row["active"], bool):
            raise ValueError("Trace activity must be Boolean")
        traces.append(
            Trace(
                trace_id=trace_id,
                cue=tuple(float(value) for value in row["cue"]),
                evidence=evidence_values,
                mass=mass,
                support=support,
                ttl=ttl,
                hits=hits,
                contradictions=contradictions,
                created_tick=created_tick,
                last_tick=last_tick,
                active=row["active"],
            )
        )

    def numeric_vector(name: str) -> tuple[float, ...]:
        values = state_data[name]
        if not isinstance(values, list) or len(values) != len(traces):
            raise ValueError(f"{name} must align with the trace collection")
        return tuple(_strict_finite_number(value, name) for value in values)

    activations = numeric_vector("activations")
    weights = numeric_vector("attractor_weights")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Attractor weights must remain within [0, 1]")
    candidate_id = state_data["candidate_id"]
    if candidate_id is not None:
        candidate_id = _strict_nonnegative_int(candidate_id, "candidate_id")
        if candidate_id >= len(traces):
            raise ValueError("candidate_id does not identify a trace")
    prediction = state_data["prediction"]
    if prediction is not None:
        prediction = _strict_nonnegative_int(prediction, "prediction")
        if prediction >= VALUE_COUNT:
            raise ValueError("prediction is outside the value range")
    confidence = _strict_finite_number(state_data["confidence"], "confidence")
    resonance = _strict_finite_number(state_data["resonance"], "resonance")
    temperature = _strict_finite_number(state_data["temperature"], "temperature")
    phase_energy = _strict_finite_number(state_data["phase_energy"], "phase_energy")
    cumulative_phase_energy = _strict_finite_number(
        state_data["cumulative_phase_energy"], "cumulative_phase_energy"
    )
    if not 0.0 <= confidence <= 1.0 or not -1.0 <= resonance <= 1.0:
        raise ValueError("Response values are outside their invariants")
    if not (
        config.anneal_floor_temperature - 1e-12
        <= temperature
        <= config.anneal_start_temperature + 1e-12
    ):
        raise ValueError("Temperature is outside the configured annealing range")
    if (
        phase_energy < 0.0
        or phase_energy > config.phase_mix_strength + 1e-12
        or cumulative_phase_energy < phase_energy
    ):
        raise ValueError("Phase energy is outside its configured bound")

    def validated_counter(name: str, allowed: set[str] | None = None) -> Counter[str]:
        values = state_data[name]
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be an object")
        result: Counter[str] = Counter()
        for key, value in values.items():
            if not isinstance(key, str) or (allowed is not None and key not in allowed):
                raise ValueError(f"{name} contains an unknown key")
            result[key] = _strict_nonnegative_int(value, f"{name}.{key}")
        return result

    operator_counts = validated_counter("operator_counts", set(UNIVERSE_PRIMITIVES))
    outcome_counts = validated_counter("outcome_counts")
    last_outcome = state_data["last_outcome"]
    if not isinstance(last_outcome, str) or not last_outcome:
        raise ValueError("last_outcome must be a non-empty string")
    transition_hash = state_data["transition_hash"]
    if (
        not isinstance(transition_hash, str)
        or len(transition_hash) != 64
        or any(character not in "0123456789abcdef" for character in transition_hash)
    ):
        raise ValueError("transition_hash must be a lowercase SHA-256 digest")
    if sum(operator_counts.values()) != transitions:
        raise ValueError("Operator counts do not match the transition count")
    if sum(not trace.active for trace in traces) != forgotten:
        raise ValueError("Forgotten count does not match inactive traces")
    if len([trace for trace in traces if trace.active]) > config.capacity:
        raise ValueError("Active traces exceed configured capacity")
    if (
        sum(trace.mass for trace in traces if trace.active)
        > config.conservation_budget + 1e-9
    ):
        raise ValueError("Active mass exceeds the conservation budget")
    if observations != sum(
        outcome_counts[name] for name in ("reinforce", "revise", "repel")
    ):
        raise ValueError("Observation count is inconsistent with outcomes")
    if queries != outcome_counts["retrieve"] + outcome_counts["reject"]:
        raise ValueError("Query count is inconsistent with outcomes")

    recipe_counts_data = payload["recipe_counts"]
    if not isinstance(recipe_counts_data, Mapping):
        raise ValueError("recipe_counts must be an object")
    recipe_counts: Counter[str] = Counter()
    for name, value in recipe_counts_data.items():
        if name not in recipes.names:
            raise ValueError("recipe_counts contains an unknown cognitive atom")
        recipe_counts[name] = _strict_nonnegative_int(value, f"recipe_counts.{name}")

    state = FieldState(
        traces=tuple(traces),
        tick=tick,
        activations=activations,
        attractor_weights=weights,
        candidate_id=candidate_id,
        prediction=prediction,
        confidence=confidence,
        resonance=resonance,
        temperature=temperature,
        phase_energy=phase_energy,
        cumulative_phase_energy=cumulative_phase_energy,
        observations=observations,
        queries=queries,
        forgotten=forgotten,
        operator_counts=counter_tuple(operator_counts),
        outcome_counts=counter_tuple(outcome_counts),
        last_outcome=last_outcome,
        transition_hash=transition_hash,
        transitions=transitions,
    )
    runtime = CompositionRuntime(
        kernel=UniverseKernel(config), recipes=recipes, state=state
    )
    runtime.recipe_counts = recipe_counts
    return runtime


def architecture_audit(source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or Path(__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    replace_calls = []
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "replace":
            continue
        replace_calls.append(node.lineno)
        current: ast.AST | None = node
        containing_class = None
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.ClassDef):
                containing_class = current.name
                break
        if containing_class != "UniverseKernel":
            violations.append(node.lineno)

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    banned_ml_imports = sorted(
        imported_roots.intersection({"torch", "tensorflow", "jax", "sklearn"})
    )
    recipes = RecipeBook()
    resolved = {
        name: [primitive.value for primitive in recipes.expand(name)]
        for name in recipes.names
    }
    return {
        "passed": bool(replace_calls)
        and not violations
        and not banned_ml_imports
        and len(UNIVERSE_PRIMITIVES) == 7
        and all(steps for steps in resolved.values()),
        "field_state_frozen": FieldState.__dataclass_params__.frozen,
        "trace_frozen": Trace.__dataclass_params__.frozen,
        "replace_call_lines": replace_calls,
        "replace_calls_outside_universe_kernel": violations,
        "universe_primitive_count": len(UNIVERSE_PRIMITIVES),
        "universe_primitives": list(UNIVERSE_PRIMITIVES),
        "resolved_recipes": resolved,
        "ml_framework_imports": banned_ml_imports,
    }


class ExactAddressMemory:
    def __init__(self) -> None:
        self.rows: dict[tuple[float, ...], int] = {}
        self.observations = 0

    def observe(self, cue: Sequence[float], value: int) -> None:
        self.rows[tuple(cue)] = value
        self.observations += 1

    def idle(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")

    def retrieve(self, cue: Sequence[float]) -> Mapping[str, Any] | None:
        value = self.rows.get(tuple(cue))
        return None if value is None else {"value": value, "confidence": 1.0}

    def active_count(self) -> int:
        return len(self.rows)


class RawNearestMemory:
    def __init__(self, query_threshold: float = 0.68) -> None:
        self.rows: list[tuple[list[float], int]] = []
        self.query_threshold = query_threshold
        self.observations = 0

    def observe(self, cue: Sequence[float], value: int) -> None:
        self.rows.append((list(cue), value))
        self.observations += 1

    def idle(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")

    def retrieve(self, cue: Sequence[float]) -> Mapping[str, Any] | None:
        if not self.rows:
            return None
        similarities = [cosine_similarity(cue, row[0]) for row in self.rows]
        best = max(similarities)
        if best < self.query_threshold:
            return None
        eligible = [
            index
            for index, similarity in enumerate(similarities)
            if similarity >= best - 0.03
        ]
        votes: Counter[int] = Counter(self.rows[index][1] for index in eligible)
        value = max(votes, key=lambda item: (votes[item], -item))
        return {"value": value, "confidence": votes[value] / len(eligible)}

    def active_count(self) -> int:
        return len(self.rows)


def build_tiny_program(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    associations = [
        {"symbol": 0, "context": 0, "value": 0},
        {"symbol": 1, "context": 0, "value": 1},
        {"symbol": 2, "context": 0, "value": 2},
        {"symbol": 3, "context": 0, "value": 3},
        {"symbol": 4, "context": 0, "value": 0},
        {"symbol": 5, "context": 0, "value": 1},
        {"symbol": 0, "context": 1, "value": 2},
        {"symbol": 1, "context": 1, "value": 3},
        {"symbol": 2, "context": 1, "value": 0},
        {"symbol": 3, "context": 1, "value": 1},
    ]
    experiences: list[dict[str, Any]] = []
    for repetition in range(3):
        rows = list(associations)
        rng.shuffle(rows)
        atom = "remember" if repetition == 0 else "abstract"
        for row in rows:
            experiences.append(
                {
                    "event_id": f"stable-r{repetition}-{row['symbol']}-{row['context']}",
                    "atom": atom,
                    "cue": encode_cue(row["symbol"], row["context"]),
                    "value": row["value"],
                    "salience": 1.0,
                    "kind": "stable",
                }
            )

    correction = {"symbol": 2, "context": 0, "old_value": 2, "value": 3}
    for repetition in range(6):
        experiences.append(
            {
                "event_id": f"correction-{repetition}",
                "atom": "revise",
                "cue": encode_cue(correction["symbol"], correction["context"]),
                "value": correction["value"],
                "salience": 1.15,
                "kind": "correction",
            }
        )

    noise = [
        {"symbol": 6, "context": 1, "value": 1},
        {"symbol": 7, "context": 1, "value": 3},
    ]
    for index, row in enumerate(noise):
        experiences.append(
            {
                "event_id": f"noise-{index}",
                "atom": "remember",
                "cue": encode_cue(row["symbol"], row["context"]),
                "value": row["value"],
                "salience": 0.75,
                "kind": "noise",
            }
        )

    final_associations = [dict(row) for row in associations]
    for row in final_associations:
        if (
            row["symbol"] == correction["symbol"]
            and row["context"] == correction["context"]
        ):
            row["value"] = correction["value"]

    queries: list[dict[str, Any]] = []
    for row in final_associations:
        cue = encode_cue(row["symbol"], row["context"])
        queries.append(
            {
                "query_id": f"full-{row['symbol']}-{row['context']}",
                "atom": "retrieve",
                "category": "full",
                "cue": cue,
                "target": row["value"],
            }
        )
        mask = sorted(rng.sample(range(SYMBOL_DIM), k=3))
        queries.append(
            {
                "query_id": f"masked-{row['symbol']}-{row['context']}",
                "atom": "retrieve",
                "category": "masked",
                "cue": masked_cue(cue, mask),
                "target": row["value"],
                "masked_positions": mask,
            }
        )
        if row["symbol"] < 4:
            queries.append(
                {
                    "query_id": f"context-{row['symbol']}-{row['context']}",
                    "atom": "retrieve",
                    "category": "context",
                    "cue": cue,
                    "target": row["value"],
                }
            )
    queries.append(
        {
            "query_id": "correction-final",
            "atom": "retrieve",
            "category": "correction",
            "cue": encode_cue(correction["symbol"], correction["context"]),
            "target": correction["value"],
        }
    )
    for index, row in enumerate(noise):
        queries.append(
            {
                "query_id": f"noise-rejection-{index}",
                "atom": "retrieve",
                "category": "noise_rejection",
                "cue": encode_cue(row["symbol"], row["context"]),
                "target": None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "experiences": experiences,
        "queries": queries,
        "idle_steps": 18,
        "associations": final_associations,
        "correction": correction,
        "noise": noise,
    }


def _score_predictions(
    predictions: Sequence[Mapping[str, Any]], observations: int, active: int
) -> dict[str, Any]:
    categories: defaultdict[str, list[float]] = defaultdict(list)
    for row in predictions:
        categories[str(row["category"])].append(float(row["correct"]))
    accuracy = {
        category: sum(values) / len(values)
        for category, values in sorted(categories.items())
    }
    behavior = sum(accuracy.values()) / len(accuracy)
    compression_ratio = observations / max(1, active)
    compression_score = min(1.0, compression_ratio / 3.0)
    return {
        "category_accuracy": accuracy,
        "behavior_score": behavior,
        "compression_ratio": compression_ratio,
        "active_memory_units": active,
        "observations": observations,
        "total_score": 0.80 * behavior + 0.20 * compression_score,
        "predictions": list(predictions),
    }


def evaluate_atom_runtime(
    runtime: CompositionRuntime, program: Mapping[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    for row in program["experiences"]:
        runtime.observe(
            row["cue"],
            int(row["value"]),
            atom=str(row["atom"]),
            salience=float(row["salience"]),
            event_id=str(row["event_id"]),
        )
    runtime.idle(int(program["idle_steps"]))
    predictions = []
    for row in program["queries"]:
        result = runtime.retrieve(row["cue"], event_id=str(row["query_id"]))
        prediction = None if result is None else int(result["value"])
        target = row["target"]
        correct = prediction is None if target is None else prediction == int(target)
        predictions.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "target": target,
                "prediction": prediction,
                "correct": correct,
                "confidence": 0.0 if result is None else float(result["confidence"]),
                "resonance": None if result is None else float(result["resonance"]),
            }
        )
    metrics = _score_predictions(
        predictions, runtime.state.observations, runtime.active_count()
    )
    metrics.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "total_active_mass": runtime.state.total_active_mass,
            "mass_excess": max(
                0.0,
                runtime.state.total_active_mass
                - runtime.kernel.config.conservation_budget,
            ),
            "operator_counts": dict(runtime.state.operator_counts),
            "outcome_counts": dict(runtime.state.outcome_counts),
            "recipe_counts": dict(sorted(runtime.recipe_counts.items())),
            "transition_hash": runtime.state.transition_hash,
            "transitions": runtime.state.transitions,
            "initial_temperature": runtime.kernel.config.anneal_start_temperature,
            "final_temperature": runtime.state.temperature,
            "temperature_drop": runtime.kernel.config.anneal_start_temperature
            - runtime.state.temperature,
            "phase_mix_strength": runtime.kernel.config.phase_mix_strength,
            "current_phase_energy": runtime.state.phase_energy,
            "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
            "mean_phase_energy": runtime.state.cumulative_phase_energy
            / max(1, dict(runtime.state.operator_counts).get("radiation", 0)),
            "max_recorded_phase_energy": max(
                (record.phase_energy for record in runtime.records), default=0.0
            ),
            "temperature_monotonic": all(
                later.temperature <= earlier.temperature + 1e-12
                for earlier, later in zip(
                    runtime.records, runtime.records[1:], strict=False
                )
            ),
        }
    )
    return metrics


def evaluate_baseline(system: Any, program: Mapping[str, Any]) -> dict[str, Any]:
    for row in program["experiences"]:
        system.observe(row["cue"], int(row["value"]))
    system.idle(int(program["idle_steps"]))
    predictions = []
    for row in program["queries"]:
        result = system.retrieve(row["cue"])
        prediction = None if result is None else int(result["value"])
        target = row["target"]
        correct = prediction is None if target is None else prediction == int(target)
        predictions.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "target": target,
                "prediction": prediction,
                "correct": correct,
                "confidence": 0.0 if result is None else float(result["confidence"]),
            }
        )
    return _score_predictions(predictions, system.observations, system.active_count())


def run_ablations(program: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for primitive in Primitive:
        runtime = CompositionRuntime(kernel=UniverseKernel(disabled=(primitive,)))
        metrics = evaluate_atom_runtime(runtime, program)
        category = metrics["category_accuracy"]
        if primitive is Primitive.CONSERVATION:
            causal = metrics["mass_excess"] > 1e-6
            signal = {"mass_excess": metrics["mass_excess"]}
        elif primitive in {Primitive.DISSIPATION, Primitive.DECAY}:
            causal = category.get("noise_rejection", 0.0) < 1.0
            signal = {"noise_rejection": category.get("noise_rejection", 0.0)}
        else:
            causal = category.get("full", 0.0) < 0.90
            signal = {"full": category.get("full", 0.0)}
        results[primitive.value] = {
            "causal_effect_observed": causal,
            "signal": signal,
            "behavior_score": metrics["behavior_score"],
            "active_memory_units": metrics["active_memory_units"],
            "total_active_mass": metrics["total_active_mass"],
            "operator_counts": metrics["operator_counts"],
        }
    return results


def config_with(**overrides: Any) -> UniverseConfig:
    values = asdict(UniverseConfig())
    values.update(overrides)
    return UniverseConfig(**values)


def run_chaos_controls(program: Mapping[str, Any]) -> dict[str, Any]:
    default_runtime = CompositionRuntime()
    default_metrics = evaluate_atom_runtime(default_runtime, program)
    repeat_runtime = CompositionRuntime()
    repeat_metrics = evaluate_atom_runtime(repeat_runtime, program)

    zero_phase_runtime = CompositionRuntime(
        kernel=UniverseKernel(config_with(phase_mix_strength=0.0))
    )
    zero_phase_metrics = evaluate_atom_runtime(zero_phase_runtime, program)
    isothermal_runtime = CompositionRuntime(
        kernel=UniverseKernel(
            config_with(
                anneal_start_temperature=1.0,
                anneal_floor_temperature=1.0,
                anneal_cooling_rate=1.0,
            )
        )
    )
    isothermal_metrics = evaluate_atom_runtime(isothermal_runtime, program)

    sweep = {}
    for strength in (0.0, 0.02, 0.035, 0.06):
        runtime = CompositionRuntime(
            kernel=UniverseKernel(config_with(phase_mix_strength=strength))
        )
        metrics = evaluate_atom_runtime(runtime, program)
        sweep[f"{strength:.3f}"] = {
            "behavior_score": metrics["behavior_score"],
            "total_score": metrics["total_score"],
            "cumulative_phase_energy": metrics["cumulative_phase_energy"],
            "max_recorded_phase_energy": metrics["max_recorded_phase_energy"],
            "final_temperature": metrics["final_temperature"],
            "transition_hash": metrics["transition_hash"],
        }

    default_predictions = [row["prediction"] for row in default_metrics["predictions"]]
    repeat_predictions = [row["prediction"] for row in repeat_metrics["predictions"]]
    return {
        "phase_mixing": {
            "enabled_strength": default_metrics["phase_mix_strength"],
            "cumulative_phase_energy": default_metrics["cumulative_phase_energy"],
            "max_recorded_phase_energy": default_metrics["max_recorded_phase_energy"],
            "zero_phase_energy": zero_phase_metrics["cumulative_phase_energy"],
            "bounded": default_metrics["max_recorded_phase_energy"]
            <= default_metrics["phase_mix_strength"] + 1e-12,
            "changes_trajectory": default_metrics["transition_hash"]
            != zero_phase_metrics["transition_hash"],
        },
        "thermal_annealing": {
            "initial_temperature": default_metrics["initial_temperature"],
            "final_temperature": default_metrics["final_temperature"],
            "temperature_drop": default_metrics["temperature_drop"],
            "monotonic": default_metrics["temperature_monotonic"],
            "isothermal_final_temperature": isothermal_metrics["final_temperature"],
            "changes_trajectory": default_metrics["transition_hash"]
            != isothermal_metrics["transition_hash"],
        },
        "determinism": {
            "predictions_match": default_predictions == repeat_predictions,
            "transition_hash_matches": default_metrics["transition_hash"]
            == repeat_metrics["transition_hash"],
        },
        "behavior_preserved": all(
            row["behavior_score"] >= 0.99 for row in sweep.values()
        ),
        "sweep": sweep,
    }


EXPERIENCE_ATOMS = {"remember", "learn", "revise", "abstract"}


def validate_workflow_request(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Workflow request must be a JSON object")
    required = {"request_id", "experiences", "idle_steps", "queries"}
    if set(payload) != required:
        raise ValueError(f"Workflow keys must be exactly {sorted(required)}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"].strip():
        raise ValueError("request_id must be a non-empty string")
    experiences = payload["experiences"]
    queries = payload["queries"]
    if not isinstance(experiences, list) or not 1 <= len(experiences) <= 512:
        raise ValueError("experiences must contain between 1 and 512 rows")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 128:
        raise ValueError("queries must contain between 1 and 128 rows")
    idle_steps = payload["idle_steps"]
    if (
        isinstance(idle_steps, bool)
        or not isinstance(idle_steps, int)
        or not 0 <= idle_steps <= 10_000
    ):
        raise ValueError("idle_steps must be an integer within [0, 10000]")
    event_ids: set[str] = set()
    for row in experiences:
        expected = {"event_id", "atom", "cue", "value", "salience"}
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError("Malformed experience row")
        event_id = row["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")
        if event_id in event_ids:
            raise ValueError("event_id values must be unique")
        event_ids.add(event_id)
        if row["atom"] not in EXPERIENCE_ATOMS:
            raise ValueError("Experience atom is not allowed")
        validate_cue(row["cue"])
        if (
            isinstance(row["value"], bool)
            or not isinstance(row["value"], int)
            or not 0 <= row["value"] < VALUE_COUNT
        ):
            raise ValueError("Experience value is out of range")
        salience = row["salience"]
        if (
            isinstance(salience, bool)
            or not isinstance(salience, (int, float))
            or not math.isfinite(salience)
            or not 0.1 <= float(salience) <= 2.0
        ):
            raise ValueError("Experience salience is invalid")
    query_ids: set[str] = set()
    for row in queries:
        if not isinstance(row, Mapping) or set(row) != {"query_id", "atom", "cue"}:
            raise ValueError("Malformed query row")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query_id must be a non-empty string")
        if query_id in query_ids:
            raise ValueError("query_id values must be unique")
        query_ids.add(query_id)
        if row["atom"] != "retrieve":
            raise ValueError("Query atom must be retrieve")
        validate_cue(row["cue"])


def run_serialized_workflow(request_path: Path, response_path: Path) -> dict[str, Any]:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    validate_workflow_request(payload)
    runtime = CompositionRuntime()
    for row in payload["experiences"]:
        runtime.observe(
            row["cue"],
            int(row["value"]),
            atom=str(row["atom"]),
            salience=float(row["salience"]),
            event_id=str(row["event_id"]),
        )
    runtime.idle(int(payload["idle_steps"]))
    predictions = []
    for row in payload["queries"]:
        result = runtime.retrieve(row["cue"], event_id=str(row["query_id"]))
        predictions.append(
            {
                "query_id": row["query_id"],
                "prediction": None if result is None else int(result["value"]),
                "confidence": 0.0 if result is None else float(result["confidence"]),
                "resonance": None if result is None else float(result["resonance"]),
            }
        )
    response = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "request_id": payload["request_id"],
        "request_hash": stable_hash(payload),
        "predictions": predictions,
        "runtime": {
            "observations": runtime.state.observations,
            "active_traces": runtime.active_count(),
            "forgotten_traces": runtime.state.forgotten,
            "compression_ratio": runtime.state.observations
            / max(1, runtime.active_count()),
            "operator_counts": dict(runtime.state.operator_counts),
            "outcome_counts": dict(runtime.state.outcome_counts),
            "recipe_counts": dict(sorted(runtime.recipe_counts.items())),
            "transition_hash": runtime.state.transition_hash,
            "state_hash": stable_hash(runtime_payload(runtime)),
            "temperature": runtime.state.temperature,
            "phase_energy": runtime.state.phase_energy,
            "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
        },
        "execution_tail": [asdict(record) for record in runtime.records[-8:]],
    }
    write_json(response_path, response)
    return response


def score_workflow_response(
    response: Mapping[str, Any], queries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {str(row["query_id"]): row for row in queries}
    category_values: defaultdict[str, list[float]] = defaultdict(list)
    correct = 0
    for row in response["predictions"]:
        query = expected[str(row["query_id"])]
        prediction = row["prediction"]
        target = query["target"]
        matches = prediction is None if target is None else prediction == target
        correct += int(matches)
        category_values[str(query["category"])].append(float(matches))
    category_accuracy = {
        name: sum(values) / len(values)
        for name, values in sorted(category_values.items())
    }
    return {
        "status": response["status"],
        "queries": len(response["predictions"]),
        "accuracy": correct / len(queries),
        "category_accuracy": category_accuracy,
        "passed": response["status"] == "ok" and correct == len(queries),
    }


def run_self_tests() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    audit = architecture_audit()
    checks["sole_mutation_boundary"] = bool(audit["passed"])
    checks["immutable_substrate"] = bool(
        audit["field_state_frozen"] and audit["trace_frozen"]
    )
    checks["seven_core_primitives"] = audit["universe_primitive_count"] == 7
    checks["no_ml_framework"] = not audit["ml_framework_imports"]
    checks["recipe_graph_resolves"] = all(audit["resolved_recipes"].values())

    try:
        RecipeBook(
            (
                AtomRecipe("cycle_a", ("cycle_b",), "cycle test"),
                AtomRecipe("cycle_b", ("cycle_a",), "cycle test"),
            )
        )
    except ValueError:
        checks["recipe_cycles_rejected"] = True
    else:
        checks["recipe_cycles_rejected"] = False

    program = build_tiny_program()
    checks["program_deterministic"] = stable_hash(program) == stable_hash(
        build_tiny_program()
    )
    runtime = CompositionRuntime()
    metrics = evaluate_atom_runtime(runtime, program)
    category = metrics["category_accuracy"]
    checks["learns_associations"] = category.get("full", 0.0) >= 0.90
    checks["reconstructs_masked_cues"] = category.get("masked", 0.0) >= 0.80
    checks["separates_context"] = category.get("context", 0.0) >= 0.90
    checks["revises_conflicts"] = category.get("correction", 0.0) >= 1.0
    checks["forgets_noise"] = category.get("noise_rejection", 0.0) >= 0.80
    checks["all_primitives_exercised"] = set(metrics["operator_counts"]) == set(
        UNIVERSE_PRIMITIVES
    ) and all(metrics["operator_counts"].values())
    checks["transition_ledger_consistent"] = metrics["transitions"] == sum(
        metrics["operator_counts"].values()
    )
    checks["conservation_holds"] = metrics["mass_excess"] <= 1e-9

    serialized = runtime_payload(runtime)
    restored = runtime_from_payload(json.loads(json.dumps(serialized)))
    checks["state_round_trip"] = stable_hash(serialized) == stable_hash(
        runtime_payload(restored)
    )
    corrupt = json.loads(json.dumps(serialized))
    corrupt["state"]["traces"][0]["mass"] = float("nan")
    try:
        runtime_from_payload(corrupt)
    except ValueError:
        checks["corrupt_state_rejected"] = True
    else:
        checks["corrupt_state_rejected"] = False

    ablations = run_ablations(program)
    checks["all_primitive_ablations_causal"] = all(
        row["causal_effect_observed"] for row in ablations.values()
    )
    chaos = run_chaos_controls(program)
    checks["phase_mixing_active"] = (
        chaos["phase_mixing"]["cumulative_phase_energy"] > 0.0
        and chaos["phase_mixing"]["zero_phase_energy"] == 0.0
    )
    checks["phase_mixing_bounded"] = bool(chaos["phase_mixing"]["bounded"])
    checks["phase_mixing_changes_trajectory"] = bool(
        chaos["phase_mixing"]["changes_trajectory"]
    )
    checks["annealing_cools_monotonically"] = bool(
        chaos["thermal_annealing"]["temperature_drop"] > 0.0
        and chaos["thermal_annealing"]["monotonic"]
    )
    checks["annealing_changes_trajectory"] = bool(
        chaos["thermal_annealing"]["changes_trajectory"]
    )
    checks["controlled_chaos_deterministic"] = all(chaos["determinism"].values())
    checks["controlled_chaos_preserves_behavior"] = bool(chaos["behavior_preserved"])

    cue = encode_cue(0, 0)
    malformed = {
        "request_id": "self-test",
        "experiences": [
            {
                "event_id": "event",
                "atom": "direct_write",
                "cue": cue,
                "value": 0,
                "salience": 1.0,
            }
        ],
        "idle_steps": 0,
        "queries": [{"query_id": "query", "atom": "retrieve", "cue": cue}],
    }
    try:
        validate_workflow_request(malformed)
    except ValueError:
        checks["direct_cognitive_write_rejected"] = True
    else:
        checks["direct_cognitive_write_rejected"] = False

    failed = sorted(name for name, value in checks.items() if not value)
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not failed,
        "failed": failed,
        "checks": checks,
    }


def experiment_gates(
    atom: Mapping[str, Any],
    exact: Mapping[str, Any],
    nearest: Mapping[str, Any],
    workflow: Mapping[str, Any],
    state_round_trip: bool,
    audit: Mapping[str, Any],
    ablations: Mapping[str, Any],
    chaos: Mapping[str, Any],
) -> dict[str, Any]:
    category = atom["category_accuracy"]
    gates = {
        "seven_primitives_are_sole_mutators": bool(audit["passed"]),
        "cognitive_atoms_are_compositions": all(audit["resolved_recipes"].values()),
        "all_seven_primitives_exercised": set(atom["operator_counts"])
        == set(UNIVERSE_PRIMITIVES)
        and all(atom["operator_counts"].values()),
        "all_seven_ablations_have_effect": all(
            row["causal_effect_observed"] for row in ablations.values()
        ),
        "phase_mixing_is_active_bounded_and_causal": bool(
            chaos["phase_mixing"]["cumulative_phase_energy"] > 0.0
            and chaos["phase_mixing"]["bounded"]
            and chaos["phase_mixing"]["changes_trajectory"]
        ),
        "thermal_annealing_cools_and_is_causal": bool(
            chaos["thermal_annealing"]["temperature_drop"] > 0.0
            and chaos["thermal_annealing"]["monotonic"]
            and chaos["thermal_annealing"]["changes_trajectory"]
        ),
        "controlled_chaos_is_deterministic": all(chaos["determinism"].values()),
        "controlled_chaos_preserves_behavior": bool(chaos["behavior_preserved"]),
        "learns_associations": category.get("full", 0.0) >= 0.90,
        "reconstructs_masked_cues": category.get("masked", 0.0) >= 0.80,
        "separates_context": category.get("context", 0.0) >= 0.90,
        "revises_conflicts": category.get("correction", 0.0) >= 1.0,
        "forgets_unsupported_traces": category.get("noise_rejection", 0.0) >= 0.80,
        "conservation_holds": atom["mass_excess"] <= 1e-9,
        "compresses_repetition": atom["compression_ratio"] >= 3.0,
        "beats_exact_addressing": atom["total_score"] > exact["total_score"],
        "beats_raw_nearest_memory": atom["total_score"] > nearest["total_score"],
        "serialized_state_reloads": state_round_trip,
        "serialized_workflow_runs": bool(workflow["passed"]),
    }
    return {"gates": gates, "passed": all(gates.values())}


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    self_tests = run_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Self-tests failed: {self_tests['failed']}")

    program = build_tiny_program()
    write_jsonl(output_dir / "universe_core_train.jsonl", program["experiences"])
    write_jsonl(output_dir / "universe_core_queries.jsonl", program["queries"])

    runtime = CompositionRuntime()
    atom_metrics = evaluate_atom_runtime(runtime, program)
    exact_metrics = evaluate_baseline(ExactAddressMemory(), program)
    nearest_metrics = evaluate_baseline(RawNearestMemory(), program)
    audit = architecture_audit()
    ablations = run_ablations(program)
    chaos = run_chaos_controls(program)

    state_path = output_dir / "universe_core_state.json"
    serialized_state = runtime_payload(runtime)
    write_json(state_path, serialized_state)
    restored = runtime_from_payload(json.loads(state_path.read_text(encoding="utf-8")))
    state_round_trip = stable_hash(serialized_state) == stable_hash(
        runtime_payload(restored)
    )

    workflow_request = {
        "request_id": "universe-core-tiny-002",
        "experiences": [
            {
                "event_id": row["event_id"],
                "atom": row["atom"],
                "cue": row["cue"],
                "value": row["value"],
                "salience": row["salience"],
            }
            for row in program["experiences"]
        ],
        "idle_steps": program["idle_steps"],
        "queries": [
            {"query_id": row["query_id"], "atom": row["atom"], "cue": row["cue"]}
            for row in program["queries"]
        ],
    }
    request_path = output_dir / "universe_core_workflow_request.json"
    response_path = output_dir / "universe_core_workflow_response.json"
    write_json(request_path, workflow_request)
    workflow_response = run_serialized_workflow(request_path, response_path)
    workflow_metrics = score_workflow_response(workflow_response, program["queries"])

    gates = experiment_gates(
        atom_metrics,
        exact_metrics,
        nearest_metrics,
        workflow_metrics,
        state_round_trip,
        audit,
        ablations,
        chaos,
    )
    recipe_book = RecipeBook()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "atom_universe_core_composition_v2",
        "seed": SEED,
        "standard_neural_network": False,
        "gradient_descent": False,
        "trainable_weights": 0,
        "state_transition_authority": "UniverseKernel only",
        "universe_primitives": list(UNIVERSE_PRIMITIVES),
        "cognitive_recipes": recipe_book.manifest(),
        "recipe_hash": stable_hash(recipe_book.manifest()),
        "chaos_controls": {
            "phase_mix_strength": runtime.kernel.config.phase_mix_strength,
            "anneal_start_temperature": runtime.kernel.config.anneal_start_temperature,
            "anneal_floor_temperature": runtime.kernel.config.anneal_floor_temperature,
            "anneal_cooling_rate": runtime.kernel.config.anneal_cooling_rate,
            "chaos_seed": runtime.kernel.config.chaos_seed,
            "deterministic": True,
        },
        "dataset": {
            "experiences": len(program["experiences"]),
            "queries": len(program["queries"]),
            "train_hash": stable_hash(program["experiences"]),
            "query_hash": stable_hash(program["queries"]),
        },
        "workflow": {
            "request": request_path.name,
            "response": response_path.name,
            "strict_validation": True,
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": manifest["experiment"],
        "manifest": manifest,
        "self_tests": self_tests,
        "architecture_audit": audit,
        "systems": {
            "atom_universe_core": atom_metrics,
            "exact_address_memory": exact_metrics,
            "raw_nearest_memory": nearest_metrics,
        },
        "primitive_ablations": ablations,
        "chaos_controls": chaos,
        "state_round_trip": state_round_trip,
        "serialized_workflow": workflow_metrics,
        "execution_tail": [asdict(record) for record in runtime.records[-12:]],
        "experiment_gates": gates,
    }
    write_json(output_dir / "universe_core_manifest.json", manifest)
    write_json(output_dir / "universe_core_chaos_controls.json", chaos)
    write_json(output_dir / "universe_core_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/kaggle/working")
            if Path("/kaggle/working").is_dir()
            else Path("universe_core_outputs")
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-request", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_request is not None:
        if args.response is None:
            raise ValueError("--run-request requires --response")
        response = run_serialized_workflow(args.run_request, args.response)
        print(json.dumps(response, indent=2, sort_keys=True))
        return
    if args.response is not None:
        raise ValueError("--response requires --run-request")
    if args.self_test:
        report = run_self_tests()
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    report = run_experiment(args.output_dir)
    summary = {
        "experiment_gates": report["experiment_gates"],
        "atom": {
            key: report["systems"]["atom_universe_core"][key]
            for key in (
                "category_accuracy",
                "compression_ratio",
                "active_memory_units",
                "total_score",
                "total_active_mass",
                "operator_counts",
                "outcome_counts",
            )
        },
        "ablations": {
            name: row["causal_effect_observed"]
            for name, row in report["primitive_ablations"].items()
        },
        "chaos_controls": report["chaos_controls"],
        "workflow": report["serialized_workflow"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
