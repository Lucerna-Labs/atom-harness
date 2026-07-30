"""Compositional world programs for the Atom causal-world curriculum.

The dense simulator remains intentionally small.  Semantic scale comes from
composing the seven root mechanics with independent environmental axes.  A
world program is decoded on demand, so millions of possible regimes exist
without allocating a table of worlds or adding a second knowledge authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash


CAUSAL_WORLD_CURRICULUM_RUNTIME = "atom-causal-world-curriculum-v1"

WORLD_PROGRAM_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "scale",
        ("microscopic", "mesoscopic", "macroscopic", "planetary", "networked"),
    ),
    (
        "resources",
        ("abundant", "balanced", "scarce", "pulsed", "competitive"),
    ),
    ("signal", ("clear", "noisy", "delayed", "sparse", "saturated")),
    (
        "relations",
        ("cooperative", "competitive", "asymmetric", "modular", "fluid"),
    ),
    (
        "time",
        ("stable", "oscillatory", "volatile", "aging", "regenerative"),
    ),
    (
        "topology",
        ("lattice", "small_world", "hierarchical", "clustered", "sparse"),
    ),
    ("phase_regime", ("locked", "drifting", "turbulent", "resonant")),
    ("energy_regime", ("low", "balanced", "high", "pulsed", "cascading")),
    ("boundary", ("open", "closed", "porous", "reflective")),
)

WORLD_PROGRAM_AXIS_VALUES: Mapping[str, tuple[str, ...]] = dict(WORLD_PROGRAM_AXES)
TPU_PROGRAMS_PER_SHARD = 4


@dataclass(frozen=True)
class WorldProgram:
    """One lazily decoded causal-world regime."""

    program_id: int
    primary_root: str
    secondary_root: str
    scale: str
    resources: str
    signal: str
    relations: str
    time: str
    topology: str
    phase_regime: str
    energy_regime: str
    boundary: str

    def validate(self) -> None:
        if isinstance(self.program_id, bool) or not isinstance(self.program_id, int):
            raise TypeError("world program ID must be an integer")
        if not 0 <= self.program_id < world_program_space_size():
            raise ValueError("world program ID is outside the procedural space")
        if self.primary_root not in ROOT_MECHANICS:
            raise ValueError("world program primary root is unknown")
        if self.secondary_root not in ROOT_MECHANICS:
            raise ValueError("world program secondary root is unknown")
        if self.primary_root == self.secondary_root:
            raise ValueError("world program roots must be distinct")
        for axis, values in WORLD_PROGRAM_AXES:
            if getattr(self, axis) not in values:
                raise ValueError(f"world program {axis} value is unknown")

    def condition_signature(self) -> tuple[str, ...]:
        self.validate()
        return (
            f"scale:{self.scale}",
            f"resources:{self.resources}",
            f"signal:{self.signal}",
            f"relations:{self.relations}",
            f"time:{self.time}",
            f"topology:{self.topology}",
            f"phase_regime:{self.phase_regime}",
            f"energy_regime:{self.energy_regime}",
            f"boundary:{self.boundary}",
            f"primary_root:{self.primary_root}",
            f"secondary_root:{self.secondary_root}",
        )

    def manifest(self) -> dict[str, Any]:
        self.validate()
        core = {
            **asdict(self),
            "condition_signature": list(self.condition_signature()),
            "root_gains": list(world_program_root_gains(self)),
        }
        return {**core, "program_hash": canonical_hash(core)}


def world_program_space_size() -> int:
    root_pairs = len(ROOT_MECHANICS) * (len(ROOT_MECHANICS) - 1)
    return root_pairs * math.prod(len(values) for _, values in WORLD_PROGRAM_AXES)


def decode_world_program(program_id: int) -> WorldProgram:
    if isinstance(program_id, bool) or not isinstance(program_id, int):
        raise TypeError("world program ID must be an integer")
    if not 0 <= program_id < world_program_space_size():
        raise ValueError("world program ID is outside the procedural space")
    value = program_id
    reversed_axis_values: dict[str, str] = {}
    for axis, values in reversed(WORLD_PROGRAM_AXES):
        value, digit = divmod(value, len(values))
        reversed_axis_values[axis] = values[digit]
    value, secondary_digit = divmod(value, len(ROOT_MECHANICS) - 1)
    value, primary_digit = divmod(value, len(ROOT_MECHANICS))
    if value != 0:
        raise AssertionError("world program mixed-radix decode did not terminate")
    primary_root = ROOT_MECHANICS[primary_digit]
    secondary_roots = tuple(root for root in ROOT_MECHANICS if root != primary_root)
    program = WorldProgram(
        program_id=program_id,
        primary_root=primary_root,
        secondary_root=secondary_roots[secondary_digit],
        **reversed_axis_values,
    )
    program.validate()
    return program


def encode_world_program(program: WorldProgram) -> int:
    if program.primary_root not in ROOT_MECHANICS:
        raise ValueError("world program primary root is unknown")
    if program.secondary_root not in ROOT_MECHANICS:
        raise ValueError("world program secondary root is unknown")
    if program.primary_root == program.secondary_root:
        raise ValueError("world program roots must be distinct")
    primary_digit = ROOT_MECHANICS.index(program.primary_root)
    secondary_roots = tuple(
        root for root in ROOT_MECHANICS if root != program.primary_root
    )
    digits = [primary_digit, secondary_roots.index(program.secondary_root)]
    radices = [len(ROOT_MECHANICS), len(ROOT_MECHANICS) - 1]
    for axis, values in WORLD_PROGRAM_AXES:
        value = getattr(program, axis)
        if value not in values:
            raise ValueError(f"world program {axis} value is unknown")
        digits.append(values.index(value))
        radices.append(len(values))
    encoded = 0
    for digit, radix in zip(digits, radices, strict=True):
        encoded = encoded * radix + digit
    return encoded


def curriculum_program_ids(
    shard_index: int,
    *,
    programs_per_shard: int = TPU_PROGRAMS_PER_SHARD,
) -> tuple[int, ...]:
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("curriculum shard index must be an integer")
    if shard_index < 0:
        raise ValueError("curriculum shard index cannot be negative")
    if isinstance(programs_per_shard, bool) or not isinstance(programs_per_shard, int):
        raise TypeError("programs per shard must be an integer")
    if programs_per_shard <= 0:
        raise ValueError("programs per shard must be positive")
    start = shard_index * programs_per_shard
    program_ids: list[int] = []
    for slot in range(programs_per_shard):
        serial = start + slot
        primary_root = ROOT_MECHANICS[serial % len(ROOT_MECHANICS)]
        secondary_roots = tuple(root for root in ROOT_MECHANICS if root != primary_root)
        axis_digits = {
            "scale": (2 * serial + serial // 7) % 5,
            "resources": (3 * serial + 1) % 5,
            "signal": (2 * serial + 2) % 5,
            "relations": (4 * serial + 3) % 5,
            "time": (serial + 4) % 5,
            "topology": (3 * serial + serial // 5) % 5,
            "phase_regime": (3 * serial + 1) % 4,
            "energy_regime": (2 * serial + serial // 7) % 5,
            "boundary": (2 * serial + serial // 3) % 4,
        }
        values = {
            axis: WORLD_PROGRAM_AXIS_VALUES[axis][digit]
            for axis, digit in axis_digits.items()
        }
        provisional = WorldProgram(
            program_id=0,
            primary_root=primary_root,
            secondary_root=secondary_roots[(serial // 7) % len(secondary_roots)],
            **values,
        )
        program_id = encode_world_program(provisional)
        program_ids.append(program_id)
    if len(program_ids) != len(set(program_ids)):
        raise RuntimeError("curriculum schedule produced duplicate world programs")
    return tuple(program_ids)


def curriculum_programs(
    shard_index: int,
    *,
    programs_per_shard: int = TPU_PROGRAMS_PER_SHARD,
) -> tuple[WorldProgram, ...]:
    return tuple(
        decode_world_program(program_id)
        for program_id in curriculum_program_ids(
            shard_index,
            programs_per_shard=programs_per_shard,
        )
    )


def world_program_root_gains(program: WorldProgram) -> tuple[float, ...]:
    program.validate()
    gains = {root: 1.0 for root in ROOT_MECHANICS}
    gains[program.primary_root] *= 1.38
    gains[program.secondary_root] *= 1.19

    if program.scale == "microscopic":
        gains["radiation"] *= 1.10
        gains["dissipation"] *= 1.08
    elif program.scale == "planetary":
        gains["gravitation"] *= 1.22
        gains["conservation"] *= 1.08
    elif program.scale == "networked":
        gains["radiation"] *= 1.18
        gains["attraction_repulsion"] *= 1.10

    if program.resources == "abundant":
        gains["nucleation"] *= 1.16
    elif program.resources == "scarce":
        gains["conservation"] *= 1.20
        gains["decay"] *= 1.10
    elif program.resources == "pulsed":
        gains["radiation"] *= 1.12
        gains["nucleation"] *= 1.08
    elif program.resources == "competitive":
        gains["attraction_repulsion"] *= 1.16

    if program.signal == "clear":
        gains["radiation"] *= 1.12
    elif program.signal == "noisy":
        gains["radiation"] *= 0.86
        gains["dissipation"] *= 1.14
    elif program.signal == "delayed":
        gains["conservation"] *= 1.08
    elif program.signal == "sparse":
        gains["radiation"] *= 0.78
    elif program.signal == "saturated":
        gains["dissipation"] *= 1.12

    if program.relations == "cooperative":
        gains["attraction_repulsion"] *= 1.20
        gains["nucleation"] *= 1.08
    elif program.relations == "competitive":
        gains["attraction_repulsion"] *= 1.18
        gains["decay"] *= 1.08
    elif program.relations == "modular":
        gains["nucleation"] *= 1.14
        gains["conservation"] *= 1.08
    elif program.relations == "fluid":
        gains["dissipation"] *= 1.10

    if program.time == "stable":
        gains["decay"] *= 0.78
        gains["conservation"] *= 1.10
    elif program.time == "oscillatory":
        gains["radiation"] *= 1.10
    elif program.time == "volatile":
        gains["radiation"] *= 1.18
        gains["dissipation"] *= 1.16
    elif program.time == "aging":
        gains["decay"] *= 1.30
    elif program.time == "regenerative":
        gains["nucleation"] *= 1.26

    if program.phase_regime == "locked":
        gains["attraction_repulsion"] *= 1.10
        gains["dissipation"] *= 1.06
    elif program.phase_regime == "turbulent":
        gains["radiation"] *= 1.18
        gains["nucleation"] *= 1.10
    elif program.phase_regime == "resonant":
        gains["radiation"] *= 1.12
        gains["attraction_repulsion"] *= 1.08

    if program.energy_regime == "low":
        gains["conservation"] *= 1.12
        gains["decay"] *= 0.90
    elif program.energy_regime == "high":
        gains["radiation"] *= 1.14
        gains["dissipation"] *= 1.12
    elif program.energy_regime == "pulsed":
        gains["radiation"] *= 1.18
        gains["nucleation"] *= 1.08
    elif program.energy_regime == "cascading":
        gains["radiation"] *= 1.22
        gains["decay"] *= 1.12

    if program.boundary == "closed":
        gains["conservation"] *= 1.18
        gains["radiation"] *= 0.90
    elif program.boundary == "porous":
        gains["radiation"] *= 1.08
        gains["dissipation"] *= 1.06
    elif program.boundary == "reflective":
        gains["attraction_repulsion"] *= 1.08
        gains["conservation"] *= 1.10

    return tuple(float(min(1.75, max(0.45, gains[root]))) for root in ROOT_MECHANICS)


def world_program_feature_transform(
    program: WorldProgram,
) -> dict[str, tuple[float, float]]:
    program.validate()
    transforms: dict[str, list[float]] = {}

    def adjust(feature: str, *, scale: float = 1.0, shift: float = 0.0) -> None:
        current = transforms.setdefault(feature, [1.0, 0.0])
        current[0] *= scale
        current[1] = current[1] * scale + shift

    if program.scale == "microscopic":
        adjust("mass", scale=0.64)
        adjust("velocity_x", scale=1.30)
        adjust("velocity_y", scale=1.30)
        adjust("velocity_z", scale=1.30)
    elif program.scale == "mesoscopic":
        adjust("mass", scale=0.84)
        adjust("velocity_x", scale=1.12)
        adjust("velocity_y", scale=1.12)
        adjust("velocity_z", scale=1.12)
    elif program.scale == "macroscopic":
        adjust("mass", scale=1.14)
        adjust("structure", shift=0.08)
    elif program.scale == "planetary":
        adjust("mass", scale=1.34)
        adjust("pressure", scale=1.16)
        adjust("velocity_x", scale=0.58)
        adjust("velocity_y", scale=0.58)
        adjust("velocity_z", scale=0.58)
    elif program.scale == "networked":
        adjust("signal", scale=1.30)
        adjust("trust", scale=1.10)
        adjust("language_alignment", scale=1.10)

    if program.resources == "abundant":
        adjust("resource", scale=1.36, shift=0.08)
        adjust("health", scale=1.08)
    elif program.resources == "scarce":
        adjust("resource", scale=0.54)
        adjust("health", scale=0.86)
        adjust("uncertainty", shift=0.10)
    elif program.resources == "pulsed":
        adjust("resource", scale=1.14)
        adjust("phase", scale=1.16)
    elif program.resources == "competitive":
        adjust("resource", scale=0.76)
        adjust("goal", scale=1.14)
        adjust("uncertainty", scale=1.10)

    if program.signal == "clear":
        adjust("signal", scale=1.16)
        adjust("uncertainty", scale=0.70)
    elif program.signal == "noisy":
        adjust("signal", scale=0.76)
        adjust("uncertainty", scale=1.24)
    elif program.signal == "delayed":
        adjust("signal", scale=0.86)
        adjust("memory_strength", scale=0.90)
    elif program.signal == "sparse":
        adjust("signal", scale=0.42)
        adjust("activation", scale=0.84)
    elif program.signal == "saturated":
        adjust("signal", scale=1.48, shift=0.14)
        adjust("uncertainty", scale=0.88)

    if program.relations == "cooperative":
        adjust("trust", scale=1.24)
        adjust("cohesion", shift=0.12)
        adjust("memory_strength", scale=1.08)
    elif program.relations == "competitive":
        adjust("trust", scale=0.66)
        adjust("cohesion", shift=-0.12)
        adjust("goal", scale=1.10)
    elif program.relations == "asymmetric":
        adjust("ownership", scale=1.28)
        adjust("trust", scale=0.86)
    elif program.relations == "modular":
        adjust("structure", shift=0.14)
        adjust("cohesion", shift=0.10)
    elif program.relations == "fluid":
        adjust("cohesion", scale=0.70)
        adjust("novelty", scale=1.20)

    if program.time == "stable":
        adjust("lifetime", scale=1.24)
        adjust("integrity", scale=1.08)
    elif program.time == "oscillatory":
        adjust("phase", scale=1.26)
    elif program.time == "volatile":
        adjust("uncertainty", scale=1.24)
        adjust("novelty", scale=1.20)
        adjust("integrity", scale=0.86)
    elif program.time == "aging":
        adjust("lifetime", scale=0.66)
        adjust("health", scale=0.80)
    elif program.time == "regenerative":
        adjust("health", scale=1.20)
        adjust("lifetime", scale=1.10)
        adjust("memory_strength", scale=1.10)

    if program.topology == "lattice":
        adjust("structure", shift=0.10)
    elif program.topology == "small_world":
        adjust("signal", scale=1.14)
    elif program.topology == "hierarchical":
        adjust("ownership", scale=1.20)
        adjust("trust", scale=0.90)
    elif program.topology == "clustered":
        adjust("cohesion", shift=0.12)
        adjust("trust", scale=1.08)
    elif program.topology == "sparse":
        adjust("support", scale=0.70)
        adjust("signal", scale=0.80)

    if program.phase_regime == "locked":
        adjust("phase", scale=0.44)
        adjust("cohesion", shift=0.08)
    elif program.phase_regime == "drifting":
        adjust("phase", scale=0.82)
    elif program.phase_regime == "turbulent":
        adjust("phase", scale=1.42)
        adjust("uncertainty", shift=0.10)
    elif program.phase_regime == "resonant":
        adjust("phase", scale=1.10)
        adjust("activation", scale=1.10)

    if program.energy_regime == "low":
        adjust("energy", scale=0.58)
        adjust("activation", scale=0.82)
    elif program.energy_regime == "high":
        adjust("energy", scale=1.34, shift=0.06)
        adjust("temperature", scale=1.14)
    elif program.energy_regime == "pulsed":
        adjust("energy", scale=1.16)
        adjust("phase", scale=1.18)
    elif program.energy_regime == "cascading":
        adjust("energy", scale=1.28)
        adjust("signal", scale=1.16)
        adjust("uncertainty", shift=0.08)

    if program.boundary == "closed":
        adjust("support", scale=1.14)
        adjust("signal", scale=0.84)
    elif program.boundary == "porous":
        adjust("signal", scale=1.12)
        adjust("resource", scale=1.06)
    elif program.boundary == "reflective":
        adjust("cohesion", shift=0.08)
        adjust("memory_strength", scale=1.10)

    return {
        feature: (float(values[0]), float(values[1]))
        for feature, values in transforms.items()
    }


def world_program_relation_transform(
    program: WorldProgram,
) -> dict[str, tuple[float, float]]:
    program.validate()
    transforms: dict[str, list[float]] = {}

    def adjust(relation: str, *, scale: float = 1.0, shift: float = 0.0) -> None:
        current = transforms.setdefault(relation, [1.0, 0.0])
        current[0] *= scale
        current[1] = current[1] * scale + shift

    if program.signal == "clear":
        adjust("visibility", scale=1.20)
        adjust("relation_uncertainty", scale=0.68)
    elif program.signal == "noisy":
        adjust("visibility", scale=0.76)
        adjust("relation_uncertainty", scale=1.26)
    elif program.signal == "delayed":
        adjust("causal_delay", scale=1.48)
    elif program.signal == "sparse":
        adjust("visibility", scale=0.54)
        adjust("language_channel", scale=0.62)
    elif program.signal == "saturated":
        adjust("conductivity", scale=1.20)
        adjust("relation_uncertainty", scale=1.10)

    if program.relations == "cooperative":
        adjust("affinity", scale=1.22)
        adjust("bond_strength", scale=1.18)
        adjust("trust_channel", scale=1.30)
    elif program.relations == "competitive":
        adjust("affinity", scale=0.72)
        adjust("trust_channel", scale=0.60)
        adjust("resource_channel", scale=1.22)
    elif program.relations == "asymmetric":
        adjust("permeability", scale=1.22)
        adjust("trust_channel", scale=0.78)
    elif program.relations == "modular":
        adjust("bond_strength", scale=1.32)
        adjust("symbolic_match", scale=1.22)
        adjust("permeability", scale=0.62)
    elif program.relations == "fluid":
        adjust("permeability", scale=1.34)
        adjust("bond_strength", scale=0.64)

    if program.topology == "small_world":
        adjust("conductivity", scale=1.18)
    elif program.topology == "hierarchical":
        adjust("causal_delay", scale=1.16)
    elif program.topology == "clustered":
        adjust("affinity", scale=1.18)
        adjust("bond_strength", scale=1.16)
    elif program.topology == "sparse":
        adjust("conductivity", scale=0.54)
        adjust("visibility", scale=0.68)

    if program.boundary == "closed":
        adjust("permeability", scale=0.48)
        adjust("visibility", scale=0.82)
    elif program.boundary == "porous":
        adjust("permeability", scale=1.34)
        adjust("resource_channel", scale=1.16)
    elif program.boundary == "reflective":
        adjust("permeability", scale=0.72)
        adjust("bond_strength", scale=1.10)

    return {
        relation: (float(values[0]), float(values[1]))
        for relation, values in transforms.items()
    }


def world_program_random_link_probability(program: WorldProgram) -> float:
    program.validate()
    return {
        "lattice": 0.14,
        "small_world": 0.44,
        "hierarchical": 0.34,
        "clustered": 0.24,
        "sparse": 0.72,
    }[program.topology]


def curriculum_manifest(
    *,
    shard_count: int,
    programs_per_shard: int = TPU_PROGRAMS_PER_SHARD,
) -> dict[str, Any]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("curriculum shard count must be an integer")
    if shard_count <= 0:
        raise ValueError("curriculum shard count must be positive")
    program_ids = tuple(
        program_id
        for shard_index in range(shard_count)
        for program_id in curriculum_program_ids(
            shard_index,
            programs_per_shard=programs_per_shard,
        )
    )
    programs = tuple(decode_world_program(program_id) for program_id in program_ids)
    schedule = [program.manifest() for program in programs]
    axis_values_exercised = {
        axis: sorted({getattr(program, axis) for program in programs})
        for axis, _ in WORLD_PROGRAM_AXES
    }
    return {
        "runtime": CAUSAL_WORLD_CURRICULUM_RUNTIME,
        "procedural_program_space": world_program_space_size(),
        "contextual_domain_world_space": world_program_space_size() * 8,
        "axis_count": len(WORLD_PROGRAM_AXES),
        "axes": {axis: list(values) for axis, values in WORLD_PROGRAM_AXES},
        "root_pair_count": len(ROOT_MECHANICS) * (len(ROOT_MECHANICS) - 1),
        "shards": shard_count,
        "programs_per_shard": programs_per_shard,
        "exercised_programs": len(programs),
        "unique_exercised_programs": len(set(program_ids)),
        "primary_roots_exercised": sorted(
            {program.primary_root for program in programs}
        ),
        "secondary_roots_exercised": sorted(
            {program.secondary_root for program in programs}
        ),
        "axis_values_exercised": axis_values_exercised,
        "schedule": schedule,
        "schedule_hash": canonical_hash(schedule),
    }


def causal_world_curriculum_self_test() -> dict[str, bool]:
    manifest = curriculum_manifest(shard_count=16)
    first = curriculum_program_ids(0)
    replay = curriculum_program_ids(0)
    programs = [decode_world_program(program_id) for program_id in first]
    signatures = [program.condition_signature() for program in programs]
    checks = {
        "program_space_exceeds_twenty_million": world_program_space_size() > 20_000_000,
        "contextual_space_exceeds_hundred_million": manifest[
            "contextual_domain_world_space"
        ]
        > 100_000_000,
        "sixty_four_real_curriculum_programs": manifest["exercised_programs"] == 64,
        "curriculum_programs_are_unique": manifest["unique_exercised_programs"] == 64,
        "all_primary_roots_exercised": set(manifest["primary_roots_exercised"])
        == set(ROOT_MECHANICS),
        "all_secondary_roots_exercised": set(manifest["secondary_roots_exercised"])
        == set(ROOT_MECHANICS),
        "all_axis_values_exercised": all(
            set(manifest["axis_values_exercised"][axis]) == set(values)
            for axis, values in WORLD_PROGRAM_AXES
        ),
        "deterministic_schedule": first == replay,
        "distinct_condition_signatures": len(set(signatures)) == len(signatures),
        "root_gains_are_bounded": all(
            0.45 <= gain <= 1.75
            for program in programs
            for gain in world_program_root_gains(program)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"causal-world curriculum self-test failed: {checks}")
    return checks


causal_world_curriculum_self_test()
