"""Tiny Atom cognitive-memory experiment with local, non-gradient learning.

The system does not train a neural network. Experiences physically reshape a
small field of traces through attention, binding, reinforcement, conservation,
dissipation, decay, contradiction, and attractor-style retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
SEED = 20260721
SYMBOL_DIM = 8
CONTEXT_DIM = 4
CUE_DIM = SYMBOL_DIM + CONTEXT_DIM
VALUE_COUNT = 4

COGNITIVE_ATOMS = (
    "attention",
    "association",
    "local_learning",
    "reinforcement",
    "remember",
    "retrieve",
    "forget",
    "contradiction_revision",
    "abstraction",
)

UNIVERSE_ATOMS = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = norm(left) * norm(right)
    return dot(left, right) / denominator if denominator > 1e-12 else 0.0


def normalized(value: Sequence[float]) -> list[float]:
    magnitude = norm(value)
    if magnitude <= 1e-12:
        raise ValueError("Cannot normalize an all-zero vector")
    return [component / magnitude for component in value]


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def one_hot(index: int, size: int = VALUE_COUNT) -> list[float]:
    if not 0 <= index < size:
        raise ValueError(f"Value index must be within [0, {size})")
    return [1.0 if position == index else 0.0 for position in range(size)]


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


def partial_cue(cue: Sequence[float], mask_positions: Sequence[int]) -> list[float]:
    result = list(cue)
    for position in mask_positions:
        if not 0 <= position < SYMBOL_DIM:
            raise ValueError("Only symbol dimensions may be masked")
        result[position] = 0.0
    return result


@dataclass(frozen=True)
class AtomMemoryConfig:
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

    def validate(self) -> None:
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
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
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
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dissipation_rate < 1.0:
            raise ValueError("dissipation_rate must be within [0, 1)")
        if not 0.0 <= self.mass_decay_rate < 1.0:
            raise ValueError("mass_decay_rate must be within [0, 1)")


@dataclass
class MemoryTrace:
    trace_id: int
    cue: list[float]
    evidence: list[float]
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
    def value_distribution(self) -> list[float]:
        total = sum(self.evidence)
        return (
            [value / total for value in self.evidence]
            if total > 1e-12
            else [1.0 / len(self.evidence)] * len(self.evidence)
        )

    @property
    def reliability(self) -> float:
        hit_strength = 1.0 - math.exp(-0.55 * self.hits)
        mass_strength = min(1.0, self.mass)
        return max(
            0.0,
            min(1.0, 0.45 * self.support + 0.30 * mass_strength + 0.25 * hit_strength),
        )


class MemorySystem(Protocol):
    observation_count: int

    def observe(
        self, cue: Sequence[float], value: int, salience: float = 1.0
    ) -> Mapping[str, Any]: ...

    def idle(self, steps: int) -> None: ...

    def retrieve(self, cue: Sequence[float]) -> Mapping[str, Any] | None: ...

    def active_count(self) -> int: ...


class AtomCognitiveMemory:
    def __init__(self, config: AtomMemoryConfig | None = None) -> None:
        self.config = config or AtomMemoryConfig()
        self.config.validate()
        self.traces: list[MemoryTrace] = []
        self.tick = 0
        self.observation_count = 0
        self.forgotten_count = 0
        self.action_counts: Counter[str] = Counter()
        self.transition_log: list[dict[str, Any]] = []

    def _active(self) -> list[MemoryTrace]:
        return [trace for trace in self.traces if trace.active]

    def _advance_one(self) -> None:
        self.tick += 1
        for trace in self._active():
            hit_protection = 1.0 + 0.75 * max(0, trace.hits - 1)
            trace.ttl -= 1.0 / hit_protection
            trace.support *= 1.0 - self.config.dissipation_rate / math.sqrt(
                max(1, trace.hits)
            )
            trace.mass *= 1.0 - self.config.mass_decay_rate / max(1, trace.hits)
            if trace.ttl <= 0.0 and trace.support < 0.55:
                trace.active = False
                self.forgotten_count += 1
                self.action_counts["forget"] += 1
            elif (
                trace.support < self.config.expiration_support
                and trace.mass < self.config.initial_mass
            ):
                trace.active = False
                self.forgotten_count += 1
                self.action_counts["forget"] += 1

    def idle(self, steps: int) -> None:
        if not 0 <= steps <= 10_000:
            raise ValueError("idle steps must be within [0, 10000]")
        for _ in range(steps):
            self._advance_one()
        self._conserve()

    def _conserve(self) -> None:
        active = self._active()
        total_mass = sum(trace.mass for trace in active)
        if total_mass <= self.config.conservation_budget or total_mass <= 1e-12:
            return
        scale = self.config.conservation_budget / total_mass
        for trace in active:
            trace.mass *= scale
        self.action_counts["conservation"] += 1

    def _retire_weakest_if_needed(self) -> None:
        active = self._active()
        if len(active) < self.config.capacity:
            return
        weakest = min(
            active,
            key=lambda trace: (
                trace.reliability + 0.02 * trace.ttl,
                trace.last_tick,
                -trace.trace_id,
            ),
        )
        weakest.active = False
        self.forgotten_count += 1
        self.action_counts["capacity_decay"] += 1

    def _nucleate(
        self, cue: Sequence[float], value: int, salience: float
    ) -> MemoryTrace:
        self._retire_weakest_if_needed()
        trace = MemoryTrace(
            trace_id=len(self.traces),
            cue=normalized(cue),
            evidence=[
                component * self.config.evidence_rate * salience
                for component in one_hot(value)
            ],
            mass=self.config.initial_mass * salience,
            support=min(1.0, self.config.initial_support * salience),
            ttl=self.config.initial_ttl,
            hits=1,
            contradictions=0,
            created_tick=self.tick,
            last_tick=self.tick,
        )
        self.traces.append(trace)
        self.action_counts["nucleate"] += 1
        return trace

    def _bind(
        self, trace: MemoryTrace, cue: Sequence[float], value: int, salience: float
    ) -> str:
        previous_value = trace.value
        cue_rate = self.config.cue_learning_rate * min(1.0, salience)
        trace.cue = normalized(
            [
                (1.0 - cue_rate) * old + cue_rate * new
                for old, new in zip(trace.cue, normalized(cue), strict=True)
            ]
        )
        trace.evidence[value] += self.config.evidence_rate * salience
        trace.hits += 1
        trace.mass += self.config.reinforcement_mass * salience
        trace.support = min(
            1.0, trace.support + self.config.reinforcement_support * salience
        )
        trace.ttl = self.config.initial_ttl + (
            self.config.ttl_per_reinforcement * trace.hits
        )
        trace.last_tick = self.tick
        if previous_value != value:
            trace.contradictions += 1
            trace.support *= 0.94
            action = "revise"
        else:
            action = "reinforce"
        self.action_counts[action] += 1
        return action

    def observe(
        self, cue: Sequence[float], value: int, salience: float = 1.0
    ) -> Mapping[str, Any]:
        validate_cue(cue)
        if not 0 <= value < VALUE_COUNT:
            raise ValueError(f"value must be within [0, {VALUE_COUNT})")
        if not 0.1 <= salience <= 2.0 or not math.isfinite(salience):
            raise ValueError("salience must be finite and within [0.1, 2.0]")
        self._advance_one()
        self.observation_count += 1
        active = self._active()
        similarities = [cosine_similarity(cue, trace.cue) for trace in active]
        attention = softmax(
            [
                self.config.attention_temperature
                * (similarity + 0.08 * trace.reliability)
                for similarity, trace in zip(similarities, active, strict=True)
            ]
        )
        if active and max(similarities) >= self.config.match_threshold:
            best_index = max(
                range(len(active)),
                key=lambda index: (similarities[index], attention[index]),
            )
            trace = active[best_index]
            action = self._bind(trace, cue, value, salience)
            similarity = similarities[best_index]
        else:
            trace = self._nucleate(cue, value, salience)
            action = "nucleate"
            similarity = max(similarities, default=0.0)
        self._conserve()
        transition = {
            "tick": self.tick,
            "action": action,
            "trace_id": trace.trace_id,
            "value": value,
            "resonance": similarity,
            "active_traces": self.active_count(),
            "total_mass": sum(item.mass for item in self._active()),
        }
        self.transition_log.append(transition)
        return transition

    def retrieve(self, cue: Sequence[float]) -> Mapping[str, Any] | None:
        validate_cue(cue)
        active = self._active()
        if not active:
            return None
        similarities = [cosine_similarity(cue, trace.cue) for trace in active]
        eligible = [
            index
            for index, (similarity, trace) in enumerate(
                zip(similarities, active, strict=True)
            )
            if similarity >= self.config.query_threshold
            and trace.reliability >= self.config.minimum_reliability
        ]
        if not eligible:
            return None
        logits = [
            self.config.attention_temperature * similarities[index]
            + math.log(max(1e-8, active[index].reliability))
            for index in eligible
        ]
        weights = softmax(logits)
        distribution = [0.0] * VALUE_COUNT
        for weight, index in zip(weights, eligible, strict=True):
            for value_index, probability in enumerate(
                active[index].value_distribution
            ):
                distribution[value_index] += weight * probability
        prediction = max(range(VALUE_COUNT), key=distribution.__getitem__)
        best_index = max(eligible, key=similarities.__getitem__)
        self.action_counts["retrieve"] += 1
        return {
            "value": prediction,
            "confidence": distribution[prediction],
            "distribution": distribution,
            "resonance": similarities[best_index],
            "source_trace": active[best_index].trace_id,
            "source_reliability": active[best_index].reliability,
        }

    def active_count(self) -> int:
        return len(self._active())

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(self.config),
            "tick": self.tick,
            "observation_count": self.observation_count,
            "forgotten_count": self.forgotten_count,
            "action_counts": dict(self.action_counts),
            "traces": [asdict(trace) for trace in self.traces],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> AtomCognitiveMemory:
        required = {
            "schema_version",
            "config",
            "tick",
            "observation_count",
            "forgotten_count",
            "action_counts",
            "traces",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != required
            or state["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("Unsupported or malformed Atom cognitive state")
        config_payload = state["config"]
        if not isinstance(config_payload, Mapping):
            raise ValueError("Malformed memory configuration")
        try:
            memory = cls(AtomMemoryConfig(**dict(config_payload)))
        except (TypeError, ValueError) as error:
            raise ValueError("Malformed memory configuration") from error
        for name in ("tick", "observation_count", "forgotten_count"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        memory.tick = state["tick"]
        memory.observation_count = state["observation_count"]
        memory.forgotten_count = state["forgotten_count"]

        action_counts = state["action_counts"]
        if not isinstance(action_counts, Mapping) or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in action_counts.items()
        ):
            raise ValueError("Malformed action counts in cognitive state")
        memory.action_counts = Counter(dict(action_counts))

        trace_payloads = state["traces"]
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
        if not isinstance(trace_payloads, list) or len(trace_payloads) > memory.observation_count:
            raise ValueError("Malformed trace collection in cognitive state")
        memory.traces = []
        for index, payload in enumerate(trace_payloads):
            if not isinstance(payload, Mapping) or set(payload) != trace_fields:
                raise ValueError("Malformed trace in cognitive state")
            trace = MemoryTrace(**dict(payload))
            if (
                isinstance(trace.trace_id, bool)
                or not isinstance(trace.trace_id, int)
                or trace.trace_id != index
            ):
                raise ValueError("Trace identifiers must be sequential integers")
            validate_cue(trace.cue)
            if (
                not isinstance(trace.evidence, list)
                or len(trace.evidence) != VALUE_COUNT
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                    for value in trace.evidence
                )
                or sum(trace.evidence) <= 0.0
            ):
                raise ValueError("Malformed value evidence in cognitive state")
            if (
                isinstance(trace.mass, bool)
                or not isinstance(trace.mass, (int, float))
                or not math.isfinite(trace.mass)
                or trace.mass < 0.0
            ):
                raise ValueError("Malformed mass in cognitive state")
            if (
                isinstance(trace.support, bool)
                or not isinstance(trace.support, (int, float))
                or not math.isfinite(trace.support)
                or not 0.0 <= trace.support <= 1.0
            ):
                raise ValueError("Malformed support in cognitive state")
            if (
                isinstance(trace.ttl, bool)
                or not isinstance(trace.ttl, (int, float))
                or not math.isfinite(trace.ttl)
            ):
                raise ValueError("Malformed lifetime in cognitive state")
            if (
                isinstance(trace.hits, bool)
                or not isinstance(trace.hits, int)
                or trace.hits < 1
                or isinstance(trace.contradictions, bool)
                or not isinstance(trace.contradictions, int)
                or not 0 <= trace.contradictions < trace.hits
            ):
                raise ValueError("Malformed learning counts in cognitive state")
            if (
                isinstance(trace.created_tick, bool)
                or not isinstance(trace.created_tick, int)
                or isinstance(trace.last_tick, bool)
                or not isinstance(trace.last_tick, int)
                or not 0 <= trace.created_tick <= trace.last_tick <= memory.tick
            ):
                raise ValueError("Malformed trace timing in cognitive state")
            if not isinstance(trace.active, bool):
                raise ValueError("Malformed activity flag in cognitive state")
            memory.traces.append(trace)

        inactive_count = sum(not trace.active for trace in memory.traces)
        active_traces = [trace for trace in memory.traces if trace.active]
        observed_actions = sum(
            memory.action_counts[name] for name in ("nucleate", "reinforce", "revise")
        )
        if inactive_count != memory.forgotten_count:
            raise ValueError("Forgotten trace count is inconsistent")
        if len(active_traces) > memory.config.capacity:
            raise ValueError("Active trace count exceeds configured capacity")
        if observed_actions != memory.observation_count:
            raise ValueError("Observation count is inconsistent with action counts")
        if (
            sum(trace.mass for trace in active_traces)
            > memory.config.conservation_budget + 1e-9
        ):
            raise ValueError("Active trace mass exceeds the conservation budget")
        return memory


class ExactAddressMemory:
    def __init__(self) -> None:
        self.rows: dict[tuple[float, ...], int] = {}
        self.observation_count = 0

    def observe(
        self, cue: Sequence[float], value: int, salience: float = 1.0
    ) -> Mapping[str, Any]:
        del salience
        validate_cue(cue)
        self.rows[tuple(cue)] = value
        self.observation_count += 1
        return {"action": "write"}

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
        self.observation_count = 0

    def observe(
        self, cue: Sequence[float], value: int, salience: float = 1.0
    ) -> Mapping[str, Any]:
        del salience
        validate_cue(cue)
        self.rows.append((list(cue), value))
        self.observation_count += 1
        return {"action": "append"}

    def idle(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")

    def retrieve(self, cue: Sequence[float]) -> Mapping[str, Any] | None:
        validate_cue(cue)
        if not self.rows:
            return None
        similarities = [cosine_similarity(cue, row[0]) for row in self.rows]
        best = max(similarities)
        if best < self.query_threshold:
            return None
        eligible = [
            index for index, similarity in enumerate(similarities) if similarity >= best - 0.03
        ]
        votes: Counter[int] = Counter(self.rows[index][1] for index in eligible)
        value = max(votes, key=lambda item: (votes[item], -item))
        return {"value": value, "confidence": votes[value] / len(eligible)}

    def active_count(self) -> int:
        return len(self.rows)


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
        round_rows = list(associations)
        rng.shuffle(round_rows)
        for row in round_rows:
            experiences.append(
                {
                    "event_id": f"stable-r{repetition}-{row['symbol']}-{row['context']}",
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
                "cue": encode_cue(row["symbol"], row["context"]),
                "value": row["value"],
                "salience": 0.75,
                "kind": "noise",
            }
        )

    final_associations = [dict(row) for row in associations]
    for row in final_associations:
        if row["symbol"] == correction["symbol"] and row["context"] == correction["context"]:
            row["value"] = correction["value"]

    queries: list[dict[str, Any]] = []
    for row in final_associations:
        cue = encode_cue(row["symbol"], row["context"])
        queries.append(
            {
                "query_id": f"full-{row['symbol']}-{row['context']}",
                "category": "full",
                "cue": cue,
                "target": row["value"],
            }
        )
        mask = sorted(rng.sample(range(SYMBOL_DIM), k=3))
        queries.append(
            {
                "query_id": f"partial-{row['symbol']}-{row['context']}",
                "category": "partial",
                "cue": partial_cue(cue, mask),
                "target": row["value"],
                "masked_positions": mask,
            }
        )
        if row["symbol"] < 4:
            queries.append(
                {
                    "query_id": f"context-{row['symbol']}-{row['context']}",
                    "category": "context",
                    "cue": cue,
                    "target": row["value"],
                }
            )
    queries.append(
        {
            "query_id": "correction-final",
            "category": "correction",
            "cue": encode_cue(correction["symbol"], correction["context"]),
            "target": correction["value"],
        }
    )
    for index, row in enumerate(noise):
        queries.append(
            {
                "query_id": f"noise-rejection-{index}",
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


def evaluate_trained_system(
    system: MemorySystem, program: Mapping[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    for row in program["experiences"]:
        system.observe(row["cue"], int(row["value"]), float(row["salience"]))
    system.idle(int(program["idle_steps"]))
    category_results: defaultdict[str, list[float]] = defaultdict(list)
    predictions: list[dict[str, Any]] = []
    for row in program["queries"]:
        result = system.retrieve(row["cue"])
        prediction = None if result is None else int(result["value"])
        target = row["target"]
        correct = prediction is None if target is None else prediction == int(target)
        category_results[str(row["category"])].append(float(correct))
        predictions.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "target": target,
                "prediction": prediction,
                "correct": correct,
                "confidence": 0.0 if result is None else float(result["confidence"]),
                "resonance": None if result is None else result.get("resonance"),
            }
        )
    accuracies = {
        category: sum(values) / len(values)
        for category, values in sorted(category_results.items())
    }
    behavior = sum(accuracies.values()) / len(accuracies)
    active = system.active_count()
    compression_ratio = system.observation_count / max(1, active)
    compression_score = min(1.0, compression_ratio / 3.0)
    total_score = 0.80 * behavior + 0.20 * compression_score
    return {
        "category_accuracy": accuracies,
        "behavior_score": behavior,
        "compression_ratio": compression_ratio,
        "active_memory_units": active,
        "observations": system.observation_count,
        "total_score": total_score,
        "runtime_seconds": time.perf_counter() - started,
        "predictions": predictions,
    }


def validate_workflow_request(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Workflow request must be a JSON object")
    required = {"request_id", "experiences", "idle_steps", "queries"}
    if set(payload) != required:
        raise ValueError(
            f"Workflow keys must be exactly {sorted(required)}"
        )
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
        if not isinstance(row, Mapping) or set(row) != {
            "event_id",
            "cue",
            "value",
            "salience",
        }:
            raise ValueError("Malformed experience row")
        if not isinstance(row["event_id"], str) or not row["event_id"]:
            raise ValueError("event_id must be a non-empty string")
        if row["event_id"] in event_ids:
            raise ValueError("event_id values must be unique")
        event_ids.add(row["event_id"])
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
        ):
            raise ValueError("Experience salience must be finite")
        if not 0.1 <= float(salience) <= 2.0:
            raise ValueError("Experience salience is out of range")
    query_ids: set[str] = set()
    for row in queries:
        if not isinstance(row, Mapping) or set(row) != {"query_id", "cue"}:
            raise ValueError("Malformed query row")
        if not isinstance(row["query_id"], str) or not row["query_id"]:
            raise ValueError("query_id must be a non-empty string")
        if row["query_id"] in query_ids:
            raise ValueError("query_id values must be unique")
        query_ids.add(row["query_id"])
        validate_cue(row["cue"])


def run_serialized_workflow(
    request_path: Path, response_path: Path
) -> dict[str, Any]:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    validate_workflow_request(payload)
    memory = AtomCognitiveMemory()
    transitions = [
        dict(
            memory.observe(
                row["cue"], int(row["value"]), float(row["salience"])
            )
        )
        for row in payload["experiences"]
    ]
    memory.idle(int(payload["idle_steps"]))
    predictions = []
    for row in payload["queries"]:
        result = memory.retrieve(row["cue"])
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
        "memory": {
            "observations": memory.observation_count,
            "active_traces": memory.active_count(),
            "forgotten_traces": memory.forgotten_count,
            "compression_ratio": memory.observation_count
            / max(1, memory.active_count()),
            "action_counts": dict(memory.action_counts),
            "state_hash": stable_hash(memory.to_state()),
        },
        "transition_tail": transitions[-8:],
    }
    write_json(response_path, response)
    return response


def score_workflow_response(
    response: Mapping[str, Any], queries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {str(row["query_id"]): row for row in queries}
    correct = 0
    by_category: defaultdict[str, list[float]] = defaultdict(list)
    for prediction in response["predictions"]:
        row = expected[str(prediction["query_id"])]
        target = row["target"]
        actual = prediction["prediction"]
        is_correct = actual is None if target is None else actual == target
        correct += int(is_correct)
        by_category[str(row["category"])].append(float(is_correct))
    category_accuracy = {
        category: sum(values) / len(values)
        for category, values in sorted(by_category.items())
    }
    return {
        "queries": len(queries),
        "accuracy": correct / len(queries),
        "category_accuracy": category_accuracy,
        "response_status": response["status"],
        "passed": bool(
            response["status"] == "ok"
            and category_accuracy.get("full", 0.0) >= 0.90
            and category_accuracy.get("partial", 0.0) >= 0.80
            and category_accuracy.get("context", 0.0) >= 0.90
            and category_accuracy.get("correction", 0.0) >= 1.0
            and category_accuracy.get("noise_rejection", 0.0) >= 0.80
        ),
    }


def run_self_tests() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    program_a = build_tiny_program()
    program_b = build_tiny_program()
    checks["program_deterministic"] = stable_hash(program_a) == stable_hash(program_b)
    checks["cue_orthogonality"] = all(
        abs(dot(HADAMARD_SYMBOLS[left], HADAMARD_SYMBOLS[right])) < 1e-9
        for left in range(SYMBOL_DIM)
        for right in range(left + 1, SYMBOL_DIM)
    )
    memory = AtomCognitiveMemory()
    cue = encode_cue(0, 0)
    for _ in range(3):
        memory.observe(cue, 2)
    full = memory.retrieve(cue)
    partial = memory.retrieve(partial_cue(cue, (1, 3, 5)))
    checks["local_learning"] = full is not None and full["value"] == 2
    checks["partial_retrieval"] = partial is not None and partial["value"] == 2
    noise_cue = encode_cue(7, 1)
    memory.observe(noise_cue, 1, salience=0.75)
    memory.idle(18)
    checks["reinforced_memory_survives"] = memory.retrieve(cue) is not None
    checks["unsupported_memory_forgotten"] = memory.retrieve(noise_cue) is None
    state = memory.to_state()
    reloaded = AtomCognitiveMemory.from_state(state)
    checks["state_round_trip"] = stable_hash(state) == stable_hash(reloaded.to_state())
    malformed = {
        "request_id": "bad",
        "experiences": [
            {"event_id": "bad", "cue": cue, "value": 99, "salience": 1.0}
        ],
        "idle_steps": 0,
        "queries": [{"query_id": "q", "cue": cue}],
    }
    try:
        validate_workflow_request(malformed)
    except ValueError:
        checks["malformed_request_rejected"] = True
    else:
        checks["malformed_request_rejected"] = False
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
) -> dict[str, Any]:
    category = atom["category_accuracy"]
    gates = {
        "learns_associations": category.get("full", 0.0) >= 0.90,
        "retrieves_partial_cues": category.get("partial", 0.0) >= 0.80,
        "separates_context": category.get("context", 0.0) >= 0.90,
        "revises_contradictions": category.get("correction", 0.0) >= 1.0,
        "forgets_unsupported_traces": category.get("noise_rejection", 0.0) >= 0.80,
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
    write_jsonl(output_dir / "tiny_cognitive_train.jsonl", program["experiences"])
    write_jsonl(output_dir / "tiny_cognitive_queries.jsonl", program["queries"])

    atom_memory = AtomCognitiveMemory()
    exact_memory = ExactAddressMemory()
    nearest_memory = RawNearestMemory()
    atom_metrics = evaluate_trained_system(atom_memory, program)
    exact_metrics = evaluate_trained_system(exact_memory, program)
    nearest_metrics = evaluate_trained_system(nearest_memory, program)

    state_path = output_dir / "atom_cognitive_state.json"
    write_json(state_path, atom_memory.to_state())
    reloaded = AtomCognitiveMemory.from_state(
        json.loads(state_path.read_text(encoding="utf-8"))
    )
    reloaded_predictions = [
        None
        if (result := reloaded.retrieve(row["cue"])) is None
        else int(result["value"])
        for row in program["queries"]
    ]
    atom_predictions = [row["prediction"] for row in atom_metrics["predictions"]]
    state_round_trip = reloaded_predictions == atom_predictions

    workflow_request = {
        "request_id": "atom-cognitive-tiny-002",
        "experiences": [
            {
                "event_id": row["event_id"],
                "cue": row["cue"],
                "value": row["value"],
                "salience": row["salience"],
            }
            for row in program["experiences"]
        ],
        "idle_steps": program["idle_steps"],
        "queries": [
            {"query_id": row["query_id"], "cue": row["cue"]}
            for row in program["queries"]
        ],
    }
    request_path = output_dir / "atom_cognitive_workflow_request.json"
    response_path = output_dir / "atom_cognitive_workflow_response.json"
    write_json(request_path, workflow_request)
    workflow_response = run_serialized_workflow(request_path, response_path)
    workflow_metrics = score_workflow_response(workflow_response, program["queries"])

    gates = experiment_gates(
        atom_metrics,
        exact_metrics,
        nearest_metrics,
        workflow_metrics,
        state_round_trip,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "atom_cognitive_memory_v2",
        "seed": SEED,
        "standard_neural_network": False,
        "gradient_descent": False,
        "trainable_weights": 0,
        "cognitive_atoms": COGNITIVE_ATOMS,
        "universe_atoms": UNIVERSE_ATOMS,
        "memory_representation": "persistent causal traces shaped by local interactions",
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
        "experiment": "atom_cognitive_memory_v2",
        "manifest": manifest,
        "self_tests": self_tests,
        "systems": {
            "atom_cognitive_memory": atom_metrics,
            "exact_address_memory": exact_metrics,
            "raw_nearest_memory": nearest_metrics,
        },
        "atom_dynamics": {
            "action_counts": dict(atom_memory.action_counts),
            "forgotten_traces": atom_memory.forgotten_count,
            "all_traces": len(atom_memory.traces),
            "active_traces": atom_memory.active_count(),
            "transition_tail": atom_memory.transition_log[-12:],
        },
        "state_round_trip": state_round_trip,
        "serialized_workflow": workflow_metrics,
        "experiment_gates": gates,
    }
    write_json(output_dir / "atom_cognitive_manifest.json", manifest)
    write_json(output_dir / "atom_cognitive_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/kaggle/working")
            if Path("/kaggle/working").is_dir()
            else Path("cognitive_outputs")
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
            key: report["systems"]["atom_cognitive_memory"][key]
            for key in (
                "category_accuracy",
                "compression_ratio",
                "active_memory_units",
                "total_score",
            )
        },
        "baselines": {
            name: {
                "category_accuracy": value["category_accuracy"],
                "compression_ratio": value["compression_ratio"],
                "total_score": value["total_score"],
            }
            for name, value in report["systems"].items()
            if name != "atom_cognitive_memory"
        },
        "workflow": report["serialized_workflow"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
