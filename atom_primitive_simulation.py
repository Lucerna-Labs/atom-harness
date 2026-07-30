"""Executable scalar-field world for testing Primitive Forge compositions.

This is a bounded mathematical simulation, not a claim that the transforms are
complete physical laws.  The reference path works from a root expansion while
the graph path recursively executes stored composition recipes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from atom_primitive_forge import PrimitiveForge


PRIMITIVE_SIMULATION_RUNTIME = "atom-primitive-scalar-simulation-v1"


@dataclass(frozen=True)
class SimulationWorld:
    """Counterfactual controls for a bounded one-dimensional field."""

    propagation: float
    drive: float
    loss: float
    gravity: float
    attractor: float
    coupling: float
    polarity: float
    threshold: float
    crystallization: float
    budget: float
    retention: float
    parallel_gain: float = 1.0
    feedback_gain: float = 0.5

    def __post_init__(self) -> None:
        values = vars(self)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("simulation controls must be finite numbers")
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError("loss must be within [0, 1]")
        if not 0.0 <= self.gravity <= 1.0:
            raise ValueError("gravity must be within [0, 1]")
        if not 0.0 <= self.crystallization <= 1.0:
            raise ValueError("crystallization must be within [0, 1]")
        if not 0.0 <= self.retention <= 1.0:
            raise ValueError("retention must be within [0, 1]")
        if self.threshold < 0.0 or self.budget <= 0.0:
            raise ValueError("threshold and budget bounds are invalid")
        if not 0.0 <= self.parallel_gain <= 1.0:
            raise ValueError("parallel gain must be within [0, 1]")
        if not 0.0 <= self.feedback_gain <= 1.0:
            raise ValueError("feedback gain must be within [0, 1]")

    def to_payload(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}


def training_world(index: int) -> SimulationWorld:
    """Generate repeatable calibration worlds without a named outcome table."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("training world index must be a non-negative integer")
    phase = index + 1
    return SimulationWorld(
        propagation=0.22 + 0.025 * (phase % 4),
        drive=-0.18 + 0.11 * (phase % 5),
        loss=0.08 + 0.025 * (phase % 3),
        gravity=0.16 + 0.035 * (phase % 4),
        attractor=-0.42 + 0.19 * (phase % 5),
        coupling=0.07 + 0.025 * (phase % 4),
        polarity=-1.0 if phase % 2 else 1.0,
        threshold=0.20 + 0.025 * (phase % 4),
        crystallization=0.22 + 0.04 * (phase % 3),
        budget=1.05 + 0.08 * (phase % 4),
        retention=0.84 - 0.025 * (phase % 3),
        parallel_gain=0.78 + 0.04 * (phase % 3),
        feedback_gain=0.34 + 0.05 * (phase % 4),
    )


def counterfactual_world(index: int) -> SimulationWorld:
    """Generate held-out controls outside the calibration schedule."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(
            "counterfactual world index must be a non-negative integer"
        )
    phase = index + 7
    return SimulationWorld(
        propagation=0.37 + 0.02 * (phase % 3),
        drive=0.31 - 0.14 * (phase % 5),
        loss=0.19 + 0.02 * (phase % 4),
        gravity=0.31 + 0.025 * (phase % 3),
        attractor=0.58 - 0.23 * (phase % 5),
        coupling=0.16 + 0.02 * (phase % 4),
        polarity=1.0 if phase % 2 else -1.0,
        threshold=0.31 + 0.02 * (phase % 3),
        crystallization=0.38 + 0.035 * (phase % 4),
        budget=0.82 + 0.07 * (phase % 4),
        retention=0.71 - 0.02 * (phase % 3),
        parallel_gain=0.64 + 0.05 * (phase % 4),
        feedback_gain=0.52 + 0.04 * (phase % 3),
    )


def _bounded(value: float, budget: float) -> float:
    return max(-budget, min(budget, value))


def apply_root(root: str, state: float, world: SimulationWorld) -> float:
    """Graph executor for one root transition."""

    value = _bounded(float(state), world.budget)
    if root == "radiation":
        return _bounded(
            value + world.propagation * world.drive,
            world.budget,
        )
    if root == "dissipation":
        return _bounded(value * (1.0 - world.loss), world.budget)
    if root == "gravitation":
        return _bounded(
            value + world.gravity * (world.attractor - value),
            world.budget,
        )
    if root == "attraction_repulsion":
        open_fraction = max(0.0, 1.0 - abs(value) / world.budget)
        return _bounded(
            value + world.coupling * world.polarity * open_fraction,
            world.budget,
        )
    if root == "nucleation":
        if abs(value) >= world.threshold:
            direction = -1.0 if value < 0.0 else 1.0
            magnitude = abs(value) + world.crystallization * (
                world.budget - abs(value)
            )
            return _bounded(direction * magnitude, world.budget)
        return _bounded(
            value * (1.0 - 0.25 * world.crystallization),
            world.budget,
        )
    if root == "conservation":
        return _bounded(value, world.budget)
    if root == "decay":
        return _bounded(value * world.retention, world.budget)
    raise ValueError(f"unknown root transition: {root}")


def reference_root_step(
    root: str,
    state: float,
    world: SimulationWorld,
) -> float:
    """Independent flat-oracle implementation used by held-out evaluation."""

    value = max(-world.budget, min(world.budget, float(state)))
    if root == "radiation":
        result = value + world.drive * world.propagation
    elif root == "dissipation":
        result = value - value * world.loss
    elif root == "gravitation":
        result = (1.0 - world.gravity) * value + world.gravity * world.attractor
    elif root == "attraction_repulsion":
        space = max(0.0, world.budget - abs(value)) / world.budget
        result = value + world.polarity * world.coupling * space
    elif root == "nucleation":
        if abs(value) < world.threshold:
            result = value - value * 0.25 * world.crystallization
        else:
            sign = -1.0 if value < 0.0 else 1.0
            result = sign * (
                (1.0 - world.crystallization) * abs(value)
                + world.crystallization * world.budget
            )
    elif root == "conservation":
        result = value
    elif root == "decay":
        result = world.retention * value
    else:
        raise ValueError(f"unknown reference root transition: {root}")
    return max(-world.budget, min(world.budget, result))


def evaluate_primitive(
    forge: PrimitiveForge,
    primitive_id: str,
    state: float,
    world: SimulationWorld,
) -> float:
    """Execute a stored graph recipe recursively."""

    def execute(item: str, value: float, ancestry: tuple[str, ...]) -> float:
        if item in ancestry:
            raise ValueError("cyclic recipe reached the simulation executor")
        record = forge.get(item)
        if record.root:
            return apply_root(item, value, world)
        if record.recipe is None:
            raise ValueError(f"derived primitive lacks a recipe: {item}")
        parameters = dict(record.recipe.parameters)
        if record.recipe.mode == "serial":
            result = value
            for component in record.recipe.components:
                result = execute(component, result, (*ancestry, item))
            return result
        if record.recipe.mode == "parallel":
            component_values = [
                execute(component, value, (*ancestry, item))
                for component in record.recipe.components
            ]
            mean_value = sum(component_values) / len(component_values)
            gain = parameters.get("gain", world.parallel_gain)
            return _bounded(
                value + gain * (mean_value - value),
                world.budget,
            )
        if record.recipe.mode == "feedback":
            plant, controller = record.recipe.components
            plant_value = execute(plant, value, (*ancestry, item))
            correction = execute(
                controller,
                plant_value,
                (*ancestry, item),
            )
            gain = parameters.get("gain", world.feedback_gain)
            return _bounded(
                plant_value + gain * (correction - plant_value),
                world.budget,
            )
        raise ValueError(f"unsupported stored recipe mode: {record.recipe.mode}")

    if not math.isfinite(float(state)):
        raise ValueError("simulation state must be finite")
    return execute(primitive_id, float(state), ())


def evaluate_root_expansion(
    roots: tuple[str, ...],
    state: float,
    world: SimulationWorld,
) -> float:
    """Evaluate a flat serial expansion through the independent oracle."""

    if not roots:
        raise ValueError("root expansion cannot be empty")
    result = float(state)
    for root in roots:
        result = reference_root_step(root, result, world)
    return result
