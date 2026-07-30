"""Shared contracts for the graph-native Atom causal world.

The simulator, causal learner, language shell, local runtime, and Kaggle TPU
runtime all use these exact contracts.  The schema is deliberately larger
than any one experiment: small local runs and massive accelerator shards are
different executions of the same world, not different architectures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CAUSAL_WORLD_SCHEMA = 1
CAUSAL_WORLD_RUNTIME = "atom-causal-world-v1"
CAUSAL_WORLD_SEED = 2026072207

DOMAIN_NAMES = (
    "physical",
    "chemical",
    "biological",
    "ecological",
    "agent",
    "social",
    "symbolic",
    "language",
)

ROOT_MECHANICS = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
)

ARCHITECTURE_COMPONENTS = (
    "causal_graph",
    "phase_locked_loop",
    "phase_mixer",
    "molecular_recognition",
    "topological_persistence",
    "thermal_annealing",
    "projective_measurement",
)

FEATURE_NAMES = (
    "existence",
    "energy",
    "mass",
    "charge",
    "temperature",
    "pressure",
    "cohesion",
    "integrity",
    "support",
    "lifetime",
    "resource",
    "health",
    "signal",
    "trust",
    "belief",
    "goal",
    "ownership",
    "value",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "polarity",
    "phase",
    "activation",
    "structure",
    "uncertainty",
    "novelty",
    "memory_strength",
    "language_alignment",
)

RELATION_NAMES = (
    "distance",
    "conductivity",
    "permeability",
    "affinity",
    "bond_strength",
    "visibility",
    "trust_channel",
    "resource_channel",
    "symbolic_match",
    "language_channel",
    "causal_delay",
    "relation_uncertainty",
)

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
RELATION_INDEX = {name: index for index, name in enumerate(RELATION_NAMES)}
DOMAIN_INDEX = {name: index for index, name in enumerate(DOMAIN_NAMES)}

INTERVENTION_FEATURES = (
    "energy",
    "temperature",
    "pressure",
    "charge",
    "resource",
    "signal",
    "trust",
    "belief",
    "goal",
    "value",
    "polarity",
    "language_alignment",
)

OBSERVABLE_FEATURES = (
    "energy",
    "mass",
    "temperature",
    "pressure",
    "cohesion",
    "integrity",
    "resource",
    "health",
    "signal",
    "trust",
    "belief",
    "value",
    "activation",
    "structure",
    "uncertainty",
    "memory_strength",
    "language_alignment",
)

CAUSAL_EFFECTS_PER_INTERVENTION = 8

DOMAIN_MECHANISMS: Mapping[str, tuple[str, ...]] = {
    "physical": (
        "propagation",
        "conduction",
        "momentum_transfer",
        "aggregation",
        "barrier_attenuation",
    ),
    "chemical": (
        "catalysis",
        "reaction_threshold",
        "bond_formation",
        "bond_breaking",
        "equilibration",
    ),
    "biological": (
        "metabolism",
        "repair",
        "competition",
        "reproduction_threshold",
        "senescence",
    ),
    "ecological": (
        "resource_flow",
        "predation",
        "symbiosis",
        "population_pressure",
        "niche_collapse",
    ),
    "agent": (
        "goal_pursuit",
        "attention_shift",
        "belief_revision",
        "planning",
        "action_failure",
    ),
    "social": (
        "trust_transfer",
        "cooperation",
        "exchange",
        "promise_keeping",
        "deception_detection",
    ),
    "symbolic": (
        "rewrite",
        "substitution",
        "constraint_propagation",
        "equivalence",
        "contradiction",
    ),
    "language": (
        "reference_binding",
        "semantic_alignment",
        "message_propagation",
        "paraphrase",
        "correction",
    ),
}


@dataclass(frozen=True)
class CausalWorldConfig:
    """One execution profile over the shared causal-world schema."""

    profile: str
    worlds_per_shard: int
    shard_count: int
    entity_count: int
    neighbor_count: int
    time_steps: int
    intervention_candidates: int
    active_experiments: int
    phase_dimensions: int
    maximum_laws: int
    seed: int = CAUSAL_WORLD_SEED

    def validate(self) -> None:
        if self.profile not in {"test", "local", "tpu-massive"}:
            raise ValueError("unsupported causal-world profile")
        integer_fields = {
            "worlds_per_shard": self.worlds_per_shard,
            "shard_count": self.shard_count,
            "entity_count": self.entity_count,
            "neighbor_count": self.neighbor_count,
            "time_steps": self.time_steps,
            "intervention_candidates": self.intervention_candidates,
            "active_experiments": self.active_experiments,
            "phase_dimensions": self.phase_dimensions,
            "maximum_laws": self.maximum_laws,
            "seed": self.seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.neighbor_count >= self.entity_count:
            raise ValueError("neighbor_count must be smaller than entity_count")
        if self.phase_dimensions < 4:
            raise ValueError("phase_dimensions must be at least four")
        if self.maximum_laws < self.intervention_candidates:
            raise ValueError("maximum_laws cannot be smaller than candidate count")

    def scale_manifest(self) -> dict[str, Any]:
        self.validate()
        worlds = self.worlds_per_shard * self.shard_count
        world_steps = worlds * self.time_steps
        entity_updates = world_steps * self.entity_count
        relation_updates = entity_updates * self.neighbor_count
        counterfactual_rollouts = worlds * self.intervention_candidates
        return {
            "profile": self.profile,
            "worlds": worlds,
            "world_steps": world_steps,
            "entity_updates": entity_updates,
            "relation_updates": relation_updates,
            "counterfactual_rollouts": counterfactual_rollouts,
            "seed_space_bits": 128,
            "domain_count": len(DOMAIN_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "relation_feature_count": len(RELATION_NAMES),
            "root_mechanic_count": len(ROOT_MECHANICS),
            "composite_mechanism_count": sum(
                len(values) for values in DOMAIN_MECHANISMS.values()
            ),
        }


PROFILES: Mapping[str, CausalWorldConfig] = {
    "test": CausalWorldConfig(
        profile="test",
        worlds_per_shard=8,
        shard_count=3,
        entity_count=24,
        neighbor_count=4,
        time_steps=5,
        intervention_candidates=12,
        active_experiments=8,
        phase_dimensions=8,
        maximum_laws=512,
    ),
    "local": CausalWorldConfig(
        profile="local",
        worlds_per_shard=48,
        shard_count=8,
        entity_count=96,
        neighbor_count=8,
        time_steps=16,
        intervention_candidates=32,
        active_experiments=24,
        phase_dimensions=12,
        maximum_laws=8192,
    ),
    "tpu-massive": CausalWorldConfig(
        profile="tpu-massive",
        worlds_per_shard=8192,
        shard_count=16,
        entity_count=256,
        neighbor_count=12,
        time_steps=64,
        intervention_candidates=64,
        active_experiments=48,
        phase_dimensions=16,
        maximum_laws=1_000_000,
    ),
}


@dataclass
class WorldBatch:
    """Dense working set compiled from a sparse persistent world."""

    state: np.ndarray
    neighbor_indices: np.ndarray
    relations: np.ndarray
    active_mask: np.ndarray
    domain_ids: np.ndarray
    program_ids: np.ndarray
    world_seeds: np.ndarray
    initial_budgets: np.ndarray
    tick: int = 0

    def validate(self) -> None:
        if self.state.ndim != 3 or self.state.shape[2] != len(FEATURE_NAMES):
            raise ValueError("state must have shape [world, entity, feature]")
        worlds, entities, _ = self.state.shape
        if self.neighbor_indices.ndim != 3:
            raise ValueError("neighbor indices must have three dimensions")
        if self.neighbor_indices.shape[:2] != (worlds, entities):
            raise ValueError("neighbor indices do not match state")
        neighbors = self.neighbor_indices.shape[2]
        if self.relations.shape != (
            worlds,
            entities,
            neighbors,
            len(RELATION_NAMES),
        ):
            raise ValueError("relation tensor does not match state")
        if self.active_mask.shape != (worlds, entities):
            raise ValueError("active mask does not match state")
        if self.domain_ids.shape != (worlds,):
            raise ValueError("domain IDs do not match world count")
        if self.program_ids.shape != (worlds,):
            raise ValueError("program IDs do not match world count")
        if not np.issubdtype(self.program_ids.dtype, np.integer):
            raise ValueError("program IDs must be integers")
        if self.program_ids.min(initial=0) < 0:
            raise ValueError("program IDs cannot be negative")
        if self.world_seeds.shape != (worlds,):
            raise ValueError("world seeds do not match world count")
        if self.initial_budgets.shape != (worlds, 3):
            raise ValueError("initial budgets must hold mass, energy, and resources")
        if self.neighbor_indices.min(initial=0) < 0:
            raise ValueError("neighbor index cannot be negative")
        if self.neighbor_indices.max(initial=0) >= entities:
            raise ValueError("neighbor index exceeds entity count")
        arrays = (
            self.state,
            self.relations,
            self.active_mask,
            self.initial_budgets,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("world tensors must be finite")

    def copy(self) -> "WorldBatch":
        return WorldBatch(
            state=self.state.copy(),
            neighbor_indices=self.neighbor_indices.copy(),
            relations=self.relations.copy(),
            active_mask=self.active_mask.copy(),
            domain_ids=self.domain_ids.copy(),
            program_ids=self.program_ids.copy(),
            world_seeds=self.world_seeds.copy(),
            initial_budgets=self.initial_budgets.copy(),
            tick=self.tick,
        )


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    feature: str
    magnitude: float
    polarity: int
    target_rule: str
    start_tick: int = 0
    duration: int = 1

    def validate(self) -> None:
        if not self.intervention_id:
            raise ValueError("intervention ID cannot be empty")
        if self.feature not in INTERVENTION_FEATURES:
            raise ValueError("intervention feature is not supported")
        if not np.isfinite(self.magnitude) or self.magnitude <= 0.0:
            raise ValueError("intervention magnitude must be positive and finite")
        if self.polarity not in {-1, 1}:
            raise ValueError("intervention polarity must be -1 or 1")
        if self.target_rule not in {
            "most_active",
            "least_active",
            "highest_uncertainty",
        }:
            raise ValueError("unsupported intervention target rule")
        if self.start_tick < 0 or self.duration <= 0:
            raise ValueError("invalid intervention timing")


@dataclass(frozen=True)
class CausalEvidence:
    evidence_id: str
    domain: str
    cause_feature: str
    effect_feature: str
    direction: int
    magnitude: float
    delay: int
    context_signature: tuple[str, ...]
    treated_worlds: int
    baseline_worlds: int
    variance: float
    invariant_error: float
    provenance_hash: str

    def validate(self) -> None:
        if self.domain not in DOMAIN_NAMES:
            raise ValueError("unknown evidence domain")
        if self.cause_feature not in FEATURE_INDEX:
            raise ValueError("unknown cause feature")
        if self.effect_feature not in FEATURE_INDEX:
            raise ValueError("unknown effect feature")
        if self.direction not in {-1, 1}:
            raise ValueError("effect direction must be -1 or 1")
        if self.delay < 0:
            raise ValueError("effect delay cannot be negative")
        if self.treated_worlds <= 0 or self.baseline_worlds <= 0:
            raise ValueError("causal evidence requires treated and baseline worlds")
        numeric = (self.magnitude, self.variance, self.invariant_error)
        if not all(np.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError("causal evidence metrics must be finite and nonnegative")
        if len(self.provenance_hash) != 64:
            raise ValueError("causal evidence provenance hash is invalid")


def get_profile(name: str) -> CausalWorldConfig:
    try:
        profile = PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown causal-world profile: {name}") from error
    profile.validate()
    return profile


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ndarray_digest(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def config_manifest(config: CausalWorldConfig) -> dict[str, Any]:
    config.validate()
    return {
        "schema": CAUSAL_WORLD_SCHEMA,
        "runtime": CAUSAL_WORLD_RUNTIME,
        "config": asdict(config),
        "scale": config.scale_manifest(),
        "domains": list(DOMAIN_NAMES),
        "root_mechanics": list(ROOT_MECHANICS),
        "architecture_components": list(ARCHITECTURE_COMPONENTS),
        "feature_names": list(FEATURE_NAMES),
        "relation_names": list(RELATION_NAMES),
        "intervention_features": list(INTERVENTION_FEATURES),
        "observable_features": list(OBSERVABLE_FEATURES),
        "causal_effects_per_intervention": CAUSAL_EFFECTS_PER_INTERVENTION,
        "domain_mechanisms": {
            name: list(values) for name, values in DOMAIN_MECHANISMS.items()
        },
    }


def ensure_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique")


def causal_world_schema_self_test() -> dict[str, bool]:
    ensure_unique(DOMAIN_NAMES, "domain")
    ensure_unique(ROOT_MECHANICS, "root mechanic")
    ensure_unique(ARCHITECTURE_COMPONENTS, "architecture component")
    ensure_unique(FEATURE_NAMES, "feature")
    ensure_unique(RELATION_NAMES, "relation")
    checks = {
        "eight_world_domains": len(DOMAIN_NAMES) == 8,
        "seven_root_mechanics": len(ROOT_MECHANICS) == 7,
        "seven_architecture_components": len(ARCHITECTURE_COMPONENTS) == 7,
        "massive_profile_exceeds_billion_entity_updates": (
            PROFILES["tpu-massive"].scale_manifest()["entity_updates"] > 1_000_000_000
        ),
        "massive_profile_exceeds_billion_relation_updates": (
            PROFILES["tpu-massive"].scale_manifest()["relation_updates"] > 1_000_000_000
        ),
        "all_domains_have_multiple_mechanisms": all(
            len(values) >= 5 for values in DOMAIN_MECHANISMS.values()
        ),
        "multiple_effects_retained_per_intervention": (
            2 <= CAUSAL_EFFECTS_PER_INTERVENTION <= len(OBSERVABLE_FEATURES)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"causal-world schema self-test failed: {checks}")
    return checks


causal_world_schema_self_test()
