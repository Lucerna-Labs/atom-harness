"""Atom-composed homeostatic control for an online law-forming field.

The controller never receives regime labels or evaluator truth.  It regulates
temperature, phase strength, and nucleation threshold from field observables.
All learned-state and controller-state replacement occurs inside the seven
primitive handlers of :class:`HomeostaticUniverseKernel`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_homeostatic_dataset import validate_homeostatic_observation
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    UNIVERSE_PRIMITIVE_NAMES,
    AtomWikiGraph,
    build_homeostatic_graph,
    retrieve_atom_context,
)


HOMEOSTATIC_MODEL_SCHEMA = 1
HOMEOSTATIC_REQUEST_SCHEMA = 1
HOMEOSTATIC_SEED = 8_240_611
ATOM_HOMEOSTATIC_RUNTIME = "atom-homeostatic-governor-v1"


class HomeostaticPrimitive(str, Enum):
    RADIATION = "radiation"
    DISSIPATION = "dissipation"
    GRAVITATION = "gravitation"
    ATTRACTION_REPULSION = "attraction_repulsion"
    NUCLEATION = "nucleation"
    CONSERVATION = "conservation"
    DECAY = "decay"


def homeostatic_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def homeostatic_fraction(seed: int, *parts: Any) -> float:
    digest = hashlib.sha256(
        json.dumps([seed, *parts], sort_keys=True, default=str).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def homeostatic_clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _strict_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class HomeostaticConfig:
    adaptive: bool = True
    window_size: int = 8
    min_hits: int = 3
    initial_temperature: float = 0.28
    minimum_temperature: float = 0.04
    maximum_temperature: float = 0.95
    nominal_temperature: float = 0.26
    fixed_cooling_rate: float = 0.91
    initial_phase_strength: float = 0.055
    minimum_phase_strength: float = 0.01
    maximum_phase_strength: float = 0.24
    nominal_phase_strength: float = 0.06
    initial_nucleation_threshold: float = 0.78
    minimum_nucleation_threshold: float = 0.56
    maximum_nucleation_threshold: float = 0.91
    nominal_nucleation_threshold: float = 0.78
    acceptance_low: float = 0.20
    acceptance_high: float = 0.48
    order_low: float = 0.64
    order_high: float = 0.94
    evidence_decay: float = 0.996
    evidence_budget_per_cue: float = 18.0
    chaos_budget: float = 1.25
    temperature_step_limit: float = 0.34
    phase_step_limit: float = 0.075
    threshold_step_limit: float = 0.08
    chaos_seed: int = HOMEOSTATIC_SEED

    def validate(self) -> None:
        if not isinstance(self.adaptive, bool):
            raise ValueError("adaptive must be boolean")
        for name, minimum, maximum in (
            ("window_size", 4, 256),
            ("min_hits", 2, 100),
            ("chaos_seed", 0, 2**63 - 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
        numeric = (
            "initial_temperature",
            "minimum_temperature",
            "maximum_temperature",
            "nominal_temperature",
            "fixed_cooling_rate",
            "initial_phase_strength",
            "minimum_phase_strength",
            "maximum_phase_strength",
            "nominal_phase_strength",
            "initial_nucleation_threshold",
            "minimum_nucleation_threshold",
            "maximum_nucleation_threshold",
            "nominal_nucleation_threshold",
            "acceptance_low",
            "acceptance_high",
            "order_low",
            "order_high",
            "evidence_decay",
            "evidence_budget_per_cue",
            "chaos_budget",
            "temperature_step_limit",
            "phase_step_limit",
            "threshold_step_limit",
        )
        for name in numeric:
            _strict_number(getattr(self, name), name)
        if not (
            0.0 < self.minimum_temperature
            <= self.initial_temperature
            <= self.maximum_temperature
            <= 2.0
        ):
            raise ValueError("temperature bounds are inconsistent")
        if not self.minimum_temperature <= self.nominal_temperature <= self.maximum_temperature:
            raise ValueError("nominal temperature is outside its bounds")
        if not 0.0 < self.fixed_cooling_rate <= 1.0:
            raise ValueError("fixed_cooling_rate must be within (0, 1]")
        if not (
            0.0 <= self.minimum_phase_strength
            <= self.initial_phase_strength
            <= self.maximum_phase_strength
            <= 1.0
        ):
            raise ValueError("phase-strength bounds are inconsistent")
        if not self.minimum_phase_strength <= self.nominal_phase_strength <= self.maximum_phase_strength:
            raise ValueError("nominal phase strength is outside its bounds")
        if not (
            0.5 <= self.minimum_nucleation_threshold
            <= self.initial_nucleation_threshold
            <= self.maximum_nucleation_threshold
            < 1.0
        ):
            raise ValueError("nucleation-threshold bounds are inconsistent")
        if not (
            self.minimum_nucleation_threshold
            <= self.nominal_nucleation_threshold
            <= self.maximum_nucleation_threshold
        ):
            raise ValueError("nominal nucleation threshold is outside its bounds")
        if not 0.0 <= self.acceptance_low < self.acceptance_high <= 1.0:
            raise ValueError("acceptance band is invalid")
        if not 0.0 <= self.order_low < self.order_high <= 1.0:
            raise ValueError("order band is invalid")
        if not 0.0 < self.evidence_decay <= 1.0:
            raise ValueError("evidence_decay must be within (0, 1]")
        if self.evidence_budget_per_cue <= 0.0 or self.chaos_budget <= 0.0:
            raise ValueError("conservation budgets must be positive")
        if min(
            self.temperature_step_limit,
            self.phase_step_limit,
            self.threshold_step_limit,
        ) <= 0.0:
            raise ValueError("control step limits must be positive")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlSnapshot:
    window: int
    events: int
    action: str
    temperature_before: float
    temperature_after: float
    phase_before: float
    phase_after: float
    threshold_before: float
    threshold_after: float
    acceptance_ratio: float
    surprise_rate: float
    coherence: float
    order_parameter: float
    free_mass_ratio: float
    nucleation_rate: float
    churn_rate: float
    energy: float
    chaos_load: float

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in tuple(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 12)
        return data


@dataclass
class HomeostaticState:
    evidence_mass: defaultdict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    evidence_hits: Counter[tuple[str, str]] = field(default_factory=Counter)
    committed: dict[str, str] = field(default_factory=dict)
    committed_strength: dict[str, float] = field(default_factory=dict)
    raw_events: list[str] = field(default_factory=list)
    temperature: float = 0.28
    phase_strength: float = 0.055
    nucleation_threshold: float = 0.78
    energy: float = 0.0
    phase_energy: float = 0.0
    cumulative_phase_energy: float = 0.0
    maximum_phase_energy: float = 0.0
    accepted_improving_moves: int = 0
    accepted_uphill_moves: int = 0
    proposed_uphill_moves: int = 0
    rejected_uphill_moves: int = 0
    observations: int = 0
    commits: int = 0
    replacements: int = 0
    reheats: int = 0
    cools: int = 0
    forgotten: int = 0
    conservation_applications: int = 0
    dissipation_steps: int = 0
    operator_counts: Counter[str] = field(default_factory=Counter)
    outcome_counts: Counter[str] = field(default_factory=Counter)
    transition_hash: str = "0" * 64
    transitions: int = 0
    control_history: list[ControlSnapshot] = field(default_factory=list)
    window_events: int = 0
    window_surprises: int = 0
    window_uphill_proposals: int = 0
    window_uphill_accepts: int = 0
    window_commits: int = 0
    window_replacements: int = 0
    window_conflicts: Counter[tuple[str, str]] = field(default_factory=Counter)
    pending_observation: dict[str, Any] | None = None
    pending_assimilation: dict[str, Any] | None = None
    pending_observables: dict[str, float] | None = None
    pending_controls: dict[str, Any] | None = None
    pending_projection: dict[str, Any] | None = None

    @property
    def raw_evidence_count(self) -> int:
        return (
            len(self.evidence_mass)
            + len(self.evidence_hits)
            + len(self.raw_events)
            + len(self.window_conflicts)
            + self.window_events
        )


def _field_order(state: HomeostaticState) -> tuple[float, float]:
    cues = sorted({cue for cue, _ in state.evidence_mass} | set(state.committed))
    if not cues:
        return 0.0, 1.0
    stable = 0.0
    free = 0.0
    for cue in cues:
        committed = state.committed.get(cue)
        for (candidate_cue, effect), mass in state.evidence_mass.items():
            if candidate_cue != cue:
                continue
            if effect == committed:
                stable += mass
            else:
                free += mass
    total = stable + free
    if total <= 1e-12:
        return 0.0, 1.0
    return stable / total, free / total


def _chaos_load(config: HomeostaticConfig, temperature: float, phase: float) -> float:
    temperature_span = config.maximum_temperature - config.minimum_temperature
    phase_span = config.maximum_phase_strength - config.minimum_phase_strength
    temperature_load = (temperature - config.minimum_temperature) / temperature_span
    phase_load = (phase - config.minimum_phase_strength) / phase_span
    return max(0.0, temperature_load) + max(0.0, phase_load)


class HomeostaticUniverseKernel:
    """Sole mutation boundary for field evidence, learned laws, and controls."""

    def __init__(
        self,
        config: HomeostaticConfig | None = None,
        disabled: Iterable[HomeostaticPrimitive] = (),
    ) -> None:
        self.config = config or HomeostaticConfig()
        self.config.validate()
        self.disabled = frozenset(disabled)

    def initial_state(self) -> HomeostaticState:
        return HomeostaticState(
            temperature=self.config.initial_temperature,
            phase_strength=self.config.initial_phase_strength,
            nucleation_threshold=self.config.initial_nucleation_threshold,
        )

    def apply(
        self,
        state: HomeostaticState,
        primitive: HomeostaticPrimitive,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if primitive in self.disabled:
            return
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be non-empty text")
        handlers = {
            HomeostaticPrimitive.RADIATION: self._radiation,
            HomeostaticPrimitive.DISSIPATION: self._dissipation,
            HomeostaticPrimitive.GRAVITATION: self._gravitation,
            HomeostaticPrimitive.ATTRACTION_REPULSION: self._attraction_repulsion,
            HomeostaticPrimitive.NUCLEATION: self._nucleation,
            HomeostaticPrimitive.CONSERVATION: self._conservation,
            HomeostaticPrimitive.DECAY: self._decay,
        }
        previous = state.transition_hash
        handlers[primitive](state, mode, event_id, payload or {})
        state.operator_counts[primitive.value] += 1
        state.outcome_counts[f"{mode}:{primitive.value}"] += 1
        state.transitions += 1
        state.transition_hash = homeostatic_hash(
            {
                "previous": previous,
                "primitive": primitive.value,
                "mode": mode,
                "event": event_id,
                "temperature": round(state.temperature, 12),
                "phase": round(state.phase_strength, 12),
                "threshold": round(state.nucleation_threshold, 12),
                "laws": dict(sorted(state.committed.items())),
                "raw": state.raw_evidence_count,
            }
        )

    def _radiation(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "observe":
            validate_homeostatic_observation(payload)
            phase = state.phase_strength * (
                2.0
                * homeostatic_fraction(
                    self.config.chaos_seed,
                    event_id,
                    state.transitions,
                    "phase",
                )
                - 1.0
            )
            state.pending_observation = dict(payload)
            state.phase_energy = abs(phase)
            state.cumulative_phase_energy += abs(phase)
            state.maximum_phase_energy = max(state.maximum_phase_energy, abs(phase))
            state.raw_events.append(event_id)
            state.window_events += 1
            state.observations += 1
        elif mode == "govern" and self.config.adaptive:
            if state.pending_controls is None:
                return
            candidate = {
                "temperature": state.temperature
                + max(float(state.pending_controls["temperature_delta"]), 0.0),
                "phase_strength": state.phase_strength
                + max(float(state.pending_controls["phase_delta"]), 0.0),
                "nucleation_threshold": float(
                    state.pending_controls["nucleation_threshold"]
                ),
                "action": str(state.pending_controls["action"]),
            }
            state.pending_projection = candidate

    def _gravitation(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "observe":
            if state.pending_observation is None:
                return
            row = state.pending_observation
            cue = str(row["cue"])
            effect = str(row["effect"])
            current = state.committed.get(cue)
            surprise = current is not None and current != effect
            if surprise:
                state.window_surprises += 1
            noise = state.phase_energy * (
                1.0
                if homeostatic_fraction(
                    self.config.chaos_seed,
                    event_id,
                    state.transitions,
                    "mass-sign",
                )
                >= 0.5
                else -1.0
            )
            state.pending_assimilation = {
                "cue": cue,
                "effect": effect,
                "current": current,
                "surprise": surprise,
                "mass": float(row["salience"]) * (1.0 + noise),
            }
        elif mode == "govern":
            events = max(state.window_events, 1)
            proposals = state.window_uphill_proposals
            acceptance = (
                state.window_uphill_accepts / proposals if proposals else 0.0
            )
            surprise = state.window_surprises / events
            if proposals:
                maxima: dict[str, int] = {}
                totals: Counter[str] = Counter()
                for (cue, _), count in state.window_conflicts.items():
                    totals[cue] += count
                for cue in totals:
                    maxima[cue] = max(
                        count
                        for (candidate, _), count in state.window_conflicts.items()
                        if candidate == cue
                    )
                coherence = sum(maxima.values()) / proposals
            else:
                coherence = 0.0
            order, free = _field_order(state)
            nucleation_rate = state.window_commits / events
            churn_rate = state.window_replacements / events
            energy = (
                0.45 * surprise
                + 0.25 * free
                + 0.20 * churn_rate
                + 0.10 * abs(order - self.config.order_high)
            )
            state.energy = energy
            state.pending_observables = {
                "acceptance_ratio": acceptance,
                "surprise_rate": surprise,
                "coherence": coherence,
                "order_parameter": order,
                "free_mass_ratio": free,
                "nucleation_rate": nucleation_rate,
                "churn_rate": churn_rate,
                "energy": energy,
            }

    def _attraction_repulsion(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "observe":
            item = state.pending_assimilation
            if item is None:
                return
            cue = str(item["cue"])
            effect = str(item["effect"])
            current = item["current"]
            accepted = current is None or current == effect
            if current is not None and current != effect:
                state.proposed_uphill_moves += 1
                state.window_uphill_proposals += 1
                state.window_conflicts[(cue, effect)] += 1
                order, _ = _field_order(state)
                barrier = 0.30 + 0.58 * order
                probability = min(
                    1.0,
                    math.exp(-barrier / max(state.temperature, 1e-12))
                    + state.phase_strength,
                )
                draw = homeostatic_fraction(
                    self.config.chaos_seed,
                    event_id,
                    state.transitions,
                    "accept",
                )
                accepted = draw < probability
                if accepted:
                    state.accepted_uphill_moves += 1
                    state.window_uphill_accepts += 1
                else:
                    state.rejected_uphill_moves += 1
            elif accepted:
                state.accepted_improving_moves += 1
            if accepted:
                state.evidence_mass[(cue, effect)] += max(float(item["mass"]), 0.01)
                state.evidence_hits[(cue, effect)] += 1
                if current is not None and current != effect:
                    repulsion = 0.06 + 0.62 * state.phase_strength
                    state.evidence_mass[(cue, str(current))] *= max(
                        0.0, 1.0 - repulsion
                    )
            item["accepted"] = accepted
        elif mode == "govern":
            observed = state.pending_observables
            if observed is None:
                return
            if not self.config.adaptive:
                state.pending_controls = {
                    "temperature_delta": 0.0,
                    "phase_delta": 0.0,
                    "nucleation_threshold": state.nucleation_threshold,
                    "action": "fixed_schedule",
                }
                return
            surprise = observed["surprise_rate"]
            coherence = observed["coherence"]
            acceptance = observed["acceptance_ratio"]
            churn = observed["churn_rate"]
            free = observed["free_mass_ratio"]
            coherent_pressure = surprise * coherence
            incoherent_pressure = surprise * (1.0 - coherence)
            temperature_delta = 0.0
            phase_delta = 0.0
            threshold_delta = 0.0
            action = "hold_target_band"
            if coherent_pressure >= 0.42 and coherence >= 0.72:
                acceptance_deficit = max(
                    self.config.acceptance_low - acceptance,
                    0.0,
                )
                temperature_delta = 0.24 + 0.55 * acceptance_deficit
                phase_delta = 0.045 + 0.06 * coherent_pressure
                threshold_delta = -0.055
                action = "reheat_coherent_shift"
            elif incoherent_pressure >= 0.22 or (
                surprise >= 0.45 and coherence < 0.65
            ):
                temperature_delta = -0.16 - 0.12 * incoherent_pressure
                phase_delta = -0.035 - 0.035 * incoherent_pressure
                threshold_delta = 0.055
                action = "cool_incoherent_disturbance"
            elif churn >= 0.08 or acceptance > self.config.acceptance_high:
                temperature_delta = -0.12
                phase_delta = -0.025
                threshold_delta = 0.045
                action = "damp_churn"
            elif (
                state.window_uphill_proposals > 0
                and acceptance < self.config.acceptance_low
                and coherence >= 0.72
            ):
                temperature_delta = 0.16
                phase_delta = 0.035
                threshold_delta = -0.035
                action = "escape_stagnation"
            else:
                temperature_delta = 0.18 * (
                    self.config.nominal_temperature - state.temperature
                )
                phase_delta = 0.18 * (
                    self.config.nominal_phase_strength - state.phase_strength
                )
                threshold_delta = 0.18 * (
                    self.config.nominal_nucleation_threshold
                    - state.nucleation_threshold
                )
            if observed["nucleation_rate"] > 0.25:
                threshold_delta += 0.025
            if free > 0.38 and coherence >= 0.75:
                threshold_delta -= 0.025
            state.pending_controls = {
                "temperature_delta": temperature_delta,
                "phase_delta": phase_delta,
                "nucleation_threshold": state.nucleation_threshold
                + threshold_delta,
                "action": action,
            }

    def _nucleation(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "observe":
            item = state.pending_assimilation
            if item is None or not item.get("accepted"):
                return
            cue = str(item["cue"])
            candidates = [
                (mass, effect)
                for (candidate_cue, effect), mass in state.evidence_mass.items()
                if candidate_cue == cue and mass > 0.0
            ]
            if not candidates:
                return
            best_mass, best_effect = max(candidates, key=lambda row: (row[0], row[1]))
            total = sum(mass for mass, _ in candidates)
            hits = state.evidence_hits[(cue, best_effect)]
            support = best_mass / max(total, 1e-12)
            if hits < self.config.min_hits or support < state.nucleation_threshold:
                return
            previous = state.committed.get(cue)
            state.committed[cue] = best_effect
            state.committed_strength[cue] = support
            if previous != best_effect:
                state.commits += 1
                state.window_commits += 1
                if previous is not None:
                    state.replacements += 1
                    state.window_replacements += 1
        elif mode == "govern" and state.pending_projection is not None:
            proposal = state.pending_projection
            before_temperature = state.temperature
            before_phase = state.phase_strength
            before_threshold = state.nucleation_threshold
            state.temperature = float(proposal["temperature"])
            state.phase_strength = float(proposal["phase_strength"])
            state.nucleation_threshold = float(proposal["nucleation_threshold"])
            if state.temperature > before_temperature + 1e-12:
                state.reheats += 1
            elif state.temperature < before_temperature - 1e-12:
                state.cools += 1
            observed = state.pending_observables or {}
            state.control_history.append(
                ControlSnapshot(
                    window=len(state.control_history),
                    events=state.window_events,
                    action=str(proposal["action"]),
                    temperature_before=before_temperature,
                    temperature_after=state.temperature,
                    phase_before=before_phase,
                    phase_after=state.phase_strength,
                    threshold_before=before_threshold,
                    threshold_after=state.nucleation_threshold,
                    acceptance_ratio=float(observed.get("acceptance_ratio", 0.0)),
                    surprise_rate=float(observed.get("surprise_rate", 0.0)),
                    coherence=float(observed.get("coherence", 0.0)),
                    order_parameter=float(observed.get("order_parameter", 0.0)),
                    free_mass_ratio=float(observed.get("free_mass_ratio", 1.0)),
                    nucleation_rate=float(observed.get("nucleation_rate", 0.0)),
                    churn_rate=float(observed.get("churn_rate", 0.0)),
                    energy=float(observed.get("energy", 0.0)),
                    chaos_load=_chaos_load(
                        self.config,
                        state.temperature,
                        state.phase_strength,
                    ),
                )
            )

    def _conservation(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        state.conservation_applications += 1
        if mode == "observe":
            cues = {cue for cue, _ in state.evidence_mass}
            for cue in cues:
                keys = [key for key in state.evidence_mass if key[0] == cue]
                total = sum(state.evidence_mass[key] for key in keys)
                if total <= self.config.evidence_budget_per_cue or total <= 1e-12:
                    continue
                scale = self.config.evidence_budget_per_cue / total
                for key in keys:
                    state.evidence_mass[key] *= scale
        elif mode == "govern":
            proposal = state.pending_projection
            if proposal is None:
                proposal = {
                    "temperature": state.temperature,
                    "phase_strength": state.phase_strength,
                    "nucleation_threshold": state.nucleation_threshold,
                    "action": "fixed_schedule",
                }
            temperature = homeostatic_clamp(
                float(proposal["temperature"]),
                max(
                    self.config.minimum_temperature,
                    state.temperature - self.config.temperature_step_limit,
                ),
                min(
                    self.config.maximum_temperature,
                    state.temperature + self.config.temperature_step_limit,
                ),
            )
            phase = homeostatic_clamp(
                float(proposal["phase_strength"]),
                max(
                    self.config.minimum_phase_strength,
                    state.phase_strength - self.config.phase_step_limit,
                ),
                min(
                    self.config.maximum_phase_strength,
                    state.phase_strength + self.config.phase_step_limit,
                ),
            )
            threshold = homeostatic_clamp(
                float(proposal["nucleation_threshold"]),
                max(
                    self.config.minimum_nucleation_threshold,
                    state.nucleation_threshold - self.config.threshold_step_limit,
                ),
                min(
                    self.config.maximum_nucleation_threshold,
                    state.nucleation_threshold + self.config.threshold_step_limit,
                ),
            )
            load = _chaos_load(self.config, temperature, phase)
            if load > self.config.chaos_budget:
                scale = self.config.chaos_budget / load
                temperature_fraction = (
                    temperature - self.config.minimum_temperature
                ) / (
                    self.config.maximum_temperature - self.config.minimum_temperature
                )
                phase_fraction = (
                    phase - self.config.minimum_phase_strength
                ) / (
                    self.config.maximum_phase_strength
                    - self.config.minimum_phase_strength
                )
                temperature = self.config.minimum_temperature + scale * temperature_fraction * (
                    self.config.maximum_temperature - self.config.minimum_temperature
                )
                phase = self.config.minimum_phase_strength + scale * phase_fraction * (
                    self.config.maximum_phase_strength
                    - self.config.minimum_phase_strength
                )
            state.pending_projection = {
                "temperature": temperature,
                "phase_strength": phase,
                "nucleation_threshold": threshold,
                "action": str(proposal["action"]),
            }
        elif mode == "forget":
            if _chaos_load(self.config, state.temperature, state.phase_strength) > (
                self.config.chaos_budget + 1e-12
            ):
                raise ValueError("final chaos load exceeds conservation budget")

    def _dissipation(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        state.dissipation_steps += 1
        if mode == "observe":
            for key in tuple(state.evidence_mass):
                state.evidence_mass[key] *= self.config.evidence_decay
            if not self.config.adaptive:
                state.temperature = max(
                    self.config.minimum_temperature,
                    state.temperature * self.config.fixed_cooling_rate,
                )
        elif mode == "govern" and self.config.adaptive:
            if state.pending_controls is None:
                return
            if state.pending_projection is None:
                state.pending_projection = {
                    "temperature": state.temperature,
                    "phase_strength": state.phase_strength,
                    "nucleation_threshold": float(
                        state.pending_controls["nucleation_threshold"]
                    ),
                    "action": str(state.pending_controls["action"]),
                }
            state.pending_projection["temperature"] += min(
                float(state.pending_controls["temperature_delta"]), 0.0
            )
            state.pending_projection["phase_strength"] += min(
                float(state.pending_controls["phase_delta"]), 0.0
            )

    def _decay(
        self,
        state: HomeostaticState,
        mode: str,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        if mode == "observe":
            for key in tuple(state.evidence_mass):
                if state.evidence_mass[key] < 1e-9:
                    del state.evidence_mass[key]
                    state.evidence_hits.pop(key, None)
            state.pending_observation = None
            state.pending_assimilation = None
            state.phase_energy = 0.0
        elif mode == "govern":
            state.window_events = 0
            state.window_surprises = 0
            state.window_uphill_proposals = 0
            state.window_uphill_accepts = 0
            state.window_commits = 0
            state.window_replacements = 0
            state.window_conflicts.clear()
            state.pending_observables = None
            state.pending_controls = None
            state.pending_projection = None
        elif mode == "forget":
            state.forgotten += state.raw_evidence_count
            state.evidence_mass.clear()
            state.evidence_hits.clear()
            state.raw_events.clear()
            state.window_conflicts.clear()
            state.window_events = 0
            state.window_surprises = 0
            state.window_uphill_proposals = 0
            state.window_uphill_accepts = 0
            state.window_commits = 0
            state.window_replacements = 0
            state.pending_observation = None
            state.pending_assimilation = None
            state.pending_observables = None
            state.pending_controls = None
            state.pending_projection = None


@dataclass
class HomeostaticRuntime:
    kernel: HomeostaticUniverseKernel
    state: HomeostaticState
    knowledge: AtomWikiGraph

    def predict(self, cue: str) -> str | None:
        if not isinstance(cue, str) or not cue:
            raise ValueError("cue must be non-empty text")
        return self.state.committed.get(cue)


@dataclass(frozen=True)
class TrainingTrace:
    event_id: str
    predicted_before: str | None
    committed_after: str | None


@dataclass
class HomeostaticTrainingResult:
    runtime: HomeostaticRuntime
    traces: tuple[TrainingTrace, ...]
    commitment_events: tuple[dict[str, Any], ...]


def _apply_homeostatic_recipe(
    runtime: HomeostaticRuntime,
    recipe: str,
    mode: str,
    event_id: str,
    payload: Mapping[str, Any] | None = None,
) -> None:
    for primitive_name in runtime.knowledge.expand(recipe):
        runtime.kernel.apply(
            runtime.state,
            HomeostaticPrimitive(primitive_name),
            mode,
            event_id,
            payload,
        )


def train_homeostatic_field(
    observations: Sequence[Mapping[str, Any]],
    *,
    adaptive: bool,
    disabled: Iterable[HomeostaticPrimitive] = (),
    config: HomeostaticConfig | None = None,
) -> HomeostaticTrainingResult:
    if not observations:
        raise ValueError("homeostatic training requires observations")
    chosen = config or HomeostaticConfig(adaptive=adaptive)
    if chosen.adaptive != adaptive:
        raise ValueError("config adaptive mode does not match requested mode")
    kernel = HomeostaticUniverseKernel(chosen, disabled=disabled)
    knowledge = build_homeostatic_graph()
    runtime = HomeostaticRuntime(kernel, kernel.initial_state(), knowledge)
    traces: list[TrainingTrace] = []
    commitments: list[dict[str, Any]] = []
    for index, row in enumerate(observations):
        validate_homeostatic_observation(row)
        cue = str(row["cue"])
        before = runtime.predict(cue)
        previous = dict(runtime.state.committed)
        _apply_homeostatic_recipe(
            runtime,
            "homeostatic_observe",
            "observe",
            str(row["event_id"]),
            row,
        )
        after = runtime.predict(cue)
        traces.append(TrainingTrace(str(row["event_id"]), before, after))
        if previous != runtime.state.committed:
            commitments.append(
                {
                    "event_id": str(row["event_id"]),
                    "cue": cue,
                    "previous": previous.get(cue),
                    "current": runtime.state.committed.get(cue),
                }
            )
        if (index + 1) % chosen.window_size == 0:
            _apply_homeostatic_recipe(
                runtime,
                "homeostatic_govern",
                "govern",
                f"govern-window-{(index + 1) // chosen.window_size:03d}",
            )
    if runtime.state.window_events:
        _apply_homeostatic_recipe(
            runtime,
            "homeostatic_govern",
            "govern",
            "govern-final-window",
        )
    _apply_homeostatic_recipe(
        runtime,
        "homeostatic_forget",
        "forget",
        "forget-raw-evidence",
    )
    return HomeostaticTrainingResult(
        runtime=runtime,
        traces=tuple(traces),
        commitment_events=tuple(commitments),
    )


def evaluate_final_laws(
    runtime: HomeostaticRuntime,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    predictions = {cue: runtime.predict(cue) for cue in sorted(expected)}
    correct = sum(predictions[cue] == effect for cue, effect in expected.items())
    return {
        "cases": len(expected),
        "correct": correct,
        "accuracy": correct / max(len(expected), 1),
        "predictions": predictions,
    }


def evaluate_prequential(
    result: HomeostaticTrainingResult,
    evaluator_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: defaultdict[str, list[bool]] = defaultdict(list)
    for trace in result.traces:
        truth = evaluator_truth[trace.event_id]
        predicted = trace.predicted_before
        grouped[str(truth["regime"])].append(
            predicted == str(truth["expected_effect"])
        )
    return {
        regime: {
            "cases": len(values),
            "correct": sum(values),
            "accuracy": sum(values) / len(values),
        }
        for regime, values in sorted(grouped.items())
    }


def homeostatic_model_payload(runtime: HomeostaticRuntime) -> dict[str, Any]:
    state = runtime.state
    config = runtime.kernel.config
    if state.raw_evidence_count != 0:
        raise ValueError("raw evidence must be forgotten before serialization")
    if set(state.operator_counts) != set(UNIVERSE_PRIMITIVE_NAMES):
        raise ValueError("all seven primitives must be exercised")
    if not all(state.operator_counts.values()):
        raise ValueError("primitive execution counts must be positive")
    if state.conservation_applications <= 0:
        raise ValueError("conservation was not applied")
    if state.dissipation_steps <= 0:
        raise ValueError("dissipation was not applied")
    load = _chaos_load(config, state.temperature, state.phase_strength)
    if load > config.chaos_budget + 1e-12:
        raise ValueError("chaos budget is violated")
    payload: dict[str, Any] = {
        "schema_version": HOMEOSTATIC_MODEL_SCHEMA,
        "runtime": ATOM_HOMEOSTATIC_RUNTIME,
        "knowledge_runtime": {
            "wiki": ATOM_WIKI_GRAPH_RUNTIME,
            "rag": ATOM_RAG_RUNTIME,
            "observe_leaves": list(runtime.knowledge.expand("homeostatic_observe")),
            "govern_leaves": list(runtime.knowledge.expand("homeostatic_govern")),
        },
        "config": config.payload(),
        "laws": [
            {
                "cue": cue,
                "effect": effect,
                "strength": round(state.committed_strength.get(cue, 0.0), 12),
            }
            for cue, effect in sorted(state.committed.items())
        ],
        "controller": {
            "temperature": round(state.temperature, 12),
            "phase_strength": round(state.phase_strength, 12),
            "nucleation_threshold": round(state.nucleation_threshold, 12),
            "chaos_load": round(load, 12),
            "energy": round(state.energy, 12),
            "history": [snapshot.payload() for snapshot in state.control_history],
        },
        "training": {
            "observations": state.observations,
            "commits": state.commits,
            "replacements": state.replacements,
            "accepted_improving_moves": state.accepted_improving_moves,
            "accepted_uphill_moves": state.accepted_uphill_moves,
            "proposed_uphill_moves": state.proposed_uphill_moves,
            "rejected_uphill_moves": state.rejected_uphill_moves,
            "reheats": state.reheats,
            "cools": state.cools,
            "forgotten": state.forgotten,
            "raw_event_count": len(state.raw_events),
            "raw_evidence_count": state.raw_evidence_count,
            "cumulative_phase_energy": round(state.cumulative_phase_energy, 12),
            "maximum_phase_energy": round(state.maximum_phase_energy, 12),
            "conservation_applications": state.conservation_applications,
            "dissipation_steps": state.dissipation_steps,
            "operator_counts": dict(sorted(state.operator_counts.items())),
            "transition_hash": state.transition_hash,
        },
    }
    payload["model_hash"] = homeostatic_hash(payload)
    return payload


@dataclass(frozen=True)
class LoadedHomeostaticModel:
    payload: Mapping[str, Any]
    knowledge: AtomWikiGraph

    def predict(self, cue: str) -> str | None:
        if not isinstance(cue, str) or not cue:
            raise ValueError("cue must be non-empty text")
        mapping = {str(row["cue"]): str(row["effect"]) for row in self.payload["laws"]}
        return mapping.get(cue)


def load_homeostatic_model(payload: Mapping[str, Any]) -> LoadedHomeostaticModel:
    expected = {
        "schema_version",
        "runtime",
        "knowledge_runtime",
        "config",
        "laws",
        "controller",
        "training",
        "model_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"model fields must be {sorted(expected)}")
    if payload["schema_version"] != HOMEOSTATIC_MODEL_SCHEMA:
        raise ValueError("unsupported homeostatic model schema")
    if payload["runtime"] != ATOM_HOMEOSTATIC_RUNTIME:
        raise ValueError("homeostatic runtime marker mismatch")
    supplied_hash = payload["model_hash"]
    core = deepcopy(dict(payload))
    del core["model_hash"]
    if not isinstance(supplied_hash, str) or homeostatic_hash(core) != supplied_hash:
        raise ValueError("homeostatic model hash mismatch")
    knowledge = payload["knowledge_runtime"]
    if not isinstance(knowledge, dict) or set(knowledge) != {
        "wiki",
        "rag",
        "observe_leaves",
        "govern_leaves",
    }:
        raise ValueError("knowledge runtime fields are invalid")
    if knowledge["wiki"] != ATOM_WIKI_GRAPH_RUNTIME or knowledge["rag"] != ATOM_RAG_RUNTIME:
        raise ValueError("knowledge runtime marker mismatch")
    graph = build_homeostatic_graph()
    if knowledge["observe_leaves"] != list(graph.expand("homeostatic_observe")):
        raise ValueError("observe recipe mismatch")
    if knowledge["govern_leaves"] != list(graph.expand("homeostatic_govern")):
        raise ValueError("govern recipe mismatch")
    config_data = payload["config"]
    if not isinstance(config_data, dict) or set(config_data) != set(asdict(HomeostaticConfig())):
        raise ValueError("homeostatic config fields are invalid")
    config = HomeostaticConfig(**config_data)
    config.validate()
    laws = payload["laws"]
    if not isinstance(laws, list) or not laws:
        raise ValueError("model must contain learned laws")
    cues: set[str] = set()
    for row in laws:
        if not isinstance(row, dict) or set(row) != {"cue", "effect", "strength"}:
            raise ValueError("law fields are invalid")
        if not isinstance(row["cue"], str) or not isinstance(row["effect"], str):
            raise ValueError("law cue and effect must be text")
        if row["cue"] in cues:
            raise ValueError("law cues must be unique")
        cues.add(row["cue"])
        strength = _strict_number(row["strength"], "law strength")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("law strength is outside [0, 1]")
    controller = payload["controller"]
    if not isinstance(controller, dict) or set(controller) != {
        "temperature",
        "phase_strength",
        "nucleation_threshold",
        "chaos_load",
        "energy",
        "history",
    }:
        raise ValueError("controller fields are invalid")
    temperature = _strict_number(controller["temperature"], "temperature")
    phase = _strict_number(controller["phase_strength"], "phase strength")
    threshold = _strict_number(
        controller["nucleation_threshold"], "nucleation threshold"
    )
    load = _strict_number(controller["chaos_load"], "chaos load")
    _strict_number(controller["energy"], "controller energy")
    if not config.minimum_temperature <= temperature <= config.maximum_temperature:
        raise ValueError("temperature is outside configured bounds")
    if not config.minimum_phase_strength <= phase <= config.maximum_phase_strength:
        raise ValueError("phase strength is outside configured bounds")
    if not (
        config.minimum_nucleation_threshold
        <= threshold
        <= config.maximum_nucleation_threshold
    ):
        raise ValueError("nucleation threshold is outside configured bounds")
    if abs(load - _chaos_load(config, temperature, phase)) > 1e-9:
        raise ValueError("serialized chaos load is inconsistent")
    if load > config.chaos_budget + 1e-12:
        raise ValueError("serialized chaos load exceeds budget")
    history = controller["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("controller history is required")
    snapshot_fields = {item.name for item in fields(ControlSnapshot)}
    for index, row in enumerate(history):
        if not isinstance(row, dict) or set(row) != snapshot_fields:
            raise ValueError("controller snapshot fields are invalid")
        if row["window"] != index:
            raise ValueError("controller windows must be contiguous")
        _strict_int(row["events"], "snapshot events", 1)
        if not isinstance(row["action"], str) or not row["action"]:
            raise ValueError("snapshot action must be text")
        for name in snapshot_fields - {"window", "events", "action"}:
            _strict_number(row[name], f"snapshot {name}")
        if float(row["chaos_load"]) > config.chaos_budget + 1e-9:
            raise ValueError("snapshot chaos load exceeds budget")
    training = payload["training"]
    expected_training = {
        "observations",
        "commits",
        "replacements",
        "accepted_improving_moves",
        "accepted_uphill_moves",
        "proposed_uphill_moves",
        "rejected_uphill_moves",
        "reheats",
        "cools",
        "forgotten",
        "raw_event_count",
        "raw_evidence_count",
        "cumulative_phase_energy",
        "maximum_phase_energy",
        "conservation_applications",
        "dissipation_steps",
        "operator_counts",
        "transition_hash",
    }
    if not isinstance(training, dict) or set(training) != expected_training:
        raise ValueError("training fields are invalid")
    for name in expected_training - {
        "cumulative_phase_energy",
        "maximum_phase_energy",
        "operator_counts",
        "transition_hash",
    }:
        _strict_int(training[name], f"training {name}")
    if training["raw_event_count"] != 0 or training["raw_evidence_count"] != 0:
        raise ValueError("serialized model contains raw evidence")
    _strict_number(training["cumulative_phase_energy"], "cumulative phase energy")
    _strict_number(training["maximum_phase_energy"], "maximum phase energy")
    counts = training["operator_counts"]
    if not isinstance(counts, dict) or set(counts) != set(UNIVERSE_PRIMITIVE_NAMES):
        raise ValueError("operator counts do not cover the seven primitives")
    if not all(_strict_int(value, f"operator {name}", 1) for name, value in counts.items()):
        raise ValueError("operator counts must be positive")
    if not isinstance(training["transition_hash"], str) or len(training["transition_hash"]) != 64:
        raise ValueError("transition hash is invalid")
    return LoadedHomeostaticModel(deepcopy(dict(payload)), graph)


def validate_homeostatic_request(request: Mapping[str, Any]) -> None:
    expected = {"schema_version", "request_id", "queries"}
    if not isinstance(request, dict) or set(request) != expected:
        raise ValueError(f"request fields must be {sorted(expected)}")
    if request["schema_version"] != HOMEOSTATIC_REQUEST_SCHEMA:
        raise ValueError("unsupported request schema")
    if not isinstance(request["request_id"], str) or not request["request_id"]:
        raise ValueError("request_id must be non-empty text")
    queries = request["queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= 64:
        raise ValueError("queries must contain 1 to 64 items")
    turn_ids: set[str] = set()
    for query in queries:
        if not isinstance(query, dict) or set(query) != {"turn_id", "cue"}:
            raise ValueError("query fields must be cue and turn_id")
        if not isinstance(query["turn_id"], str) or not query["turn_id"]:
            raise ValueError("turn_id must be non-empty text")
        if query["turn_id"] in turn_ids:
            raise ValueError("turn_id values must be unique")
        turn_ids.add(query["turn_id"])
        if not isinstance(query["cue"], str) or not query["cue"]:
            raise ValueError("cue must be non-empty text")


def run_homeostatic_request(
    model: LoadedHomeostaticModel,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    validate_homeostatic_request(request)
    turns = []
    for query in request["queries"]:
        cue = str(query["cue"])
        turns.append(
            {
                "turn_id": str(query["turn_id"]),
                "cue": cue,
                "effect": model.predict(cue),
                "knowledge_context": [
                    {
                        **item,
                        "neighbors": list(item["neighbors"]),
                    }
                    for item in retrieve_atom_context(
                        model.knowledge,
                        "homeostatic retrieve law metaplasticity feedback",
                        limit=6,
                    )
                ],
            }
        )
    return {
        "schema_version": HOMEOSTATIC_REQUEST_SCHEMA,
        "request_id": request["request_id"],
        "runtime": {
            "homeostatic_runtime": ATOM_HOMEOSTATIC_RUNTIME,
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "model_hash": model.payload["model_hash"],
        },
        "turns": turns,
    }


def write_homeostatic_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def homeostatic_architecture_audit() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HomeostaticUniverseKernel"
    )
    kernel_methods = {
        node.name
        for node in kernel.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    graph = build_homeostatic_graph()
    graph.assert_all_leaves_are_universe_primitives()
    checks = {
        "primitive_enum_matches_universe_core": {item.value for item in HomeostaticPrimitive}
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "kernel_exposes_all_primitive_handlers": {
            "_radiation",
            "_dissipation",
            "_gravitation",
            "_attraction_repulsion",
            "_nucleation",
            "_conservation",
            "_decay",
        }
        <= kernel_methods,
        "observe_recipe_uses_all_seven": set(graph.expand("homeostatic_observe"))
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "govern_recipe_uses_all_seven": set(graph.expand("homeostatic_govern"))
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "strict_loader_present": "def load_homeostatic_model" in source,
        "wiki_and_rag_runtime_wired": "retrieve_atom_context" in source,
        "fixed_and_adaptive_share_kernel": "if not self.config.adaptive" in source,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def run_homeostatic_self_tests() -> dict[str, Any]:
    graph = build_homeostatic_graph()
    checks: dict[str, bool] = {
        "governor_resolves_to_all_seven": set(graph.expand("homeostatic_govern"))
        == set(UNIVERSE_PRIMITIVE_NAMES),
        "counter_noise_is_deterministic": homeostatic_fraction(5, "a", 3)
        == homeostatic_fraction(5, "a", 3),
        "counter_noise_changes_with_event": homeostatic_fraction(5, "a", 3)
        != homeostatic_fraction(5, "b", 3),
    }
    try:
        HomeostaticConfig(chaos_budget=-1.0).validate()
    except ValueError:
        checks["invalid_budget_fails_closed"] = True
    else:
        checks["invalid_budget_fails_closed"] = False
    try:
        validate_homeostatic_request(
            {"schema_version": 1, "request_id": "x", "queries": [], "extra": 1}
        )
    except ValueError:
        checks["invalid_request_fails_closed"] = True
    else:
        checks["invalid_request_fails_closed"] = False
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }
