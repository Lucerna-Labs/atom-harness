"""Deterministic procedural simulator for the graph-native causal world."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from atom_causal_world_curriculum import (
    curriculum_programs,
    decode_world_program,
    world_program_feature_transform,
    world_program_random_link_probability,
    world_program_relation_transform,
    world_program_root_gains,
)
from atom_causal_world_schema import (
    CAUSAL_EFFECTS_PER_INTERVENTION,
    DOMAIN_INDEX,
    DOMAIN_NAMES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    INTERVENTION_FEATURES,
    OBSERVABLE_FEATURES,
    RELATION_INDEX,
    RELATION_NAMES,
    CausalEvidence,
    CausalWorldConfig,
    Intervention,
    WorldBatch,
    canonical_hash,
    get_profile,
    ndarray_digest,
)


CAUSAL_WORLD_SIMULATOR_RUNTIME = "atom-causal-world-simulator-v1"

# Domain-specific strengths over the seven root mechanics in canonical order:
# radiation, dissipation, gravitation, attraction/repulsion, nucleation,
# conservation, and decay.  These are simulator mechanics, not labels exposed
# to the causal learner.
_DOMAIN_ROOT_STRENGTHS = np.asarray(
    (
        (0.76, 0.38, 0.70, 0.58, 0.40, 0.96, 0.18),
        (0.58, 0.62, 0.28, 0.78, 0.74, 0.92, 0.32),
        (0.44, 0.30, 0.34, 0.62, 0.68, 0.88, 0.38),
        (0.64, 0.46, 0.52, 0.72, 0.54, 0.86, 0.44),
        (0.70, 0.28, 0.46, 0.64, 0.58, 0.82, 0.24),
        (0.72, 0.36, 0.40, 0.82, 0.56, 0.80, 0.34),
        (0.66, 0.24, 0.30, 0.74, 0.86, 0.98, 0.18),
        (0.88, 0.32, 0.36, 0.76, 0.62, 0.90, 0.28),
    ),
    dtype=np.float32,
)


def _clip01(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-6)


def _domain_mask(domain_ids: np.ndarray, name: str) -> np.ndarray:
    return (domain_ids == DOMAIN_INDEX[name]).astype(np.float32)[:, None]


def _clip_programmed_feature(feature: str, value: np.ndarray) -> np.ndarray:
    if feature in {
        "charge",
        "cohesion",
        "belief",
        "goal",
        "value",
        "polarity",
        "language_alignment",
    }:
        return np.clip(value, -1.0, 1.0)
    if feature == "mass":
        return np.clip(value, 0.0, 3.0)
    if feature in {"energy", "temperature", "pressure", "resource", "signal"}:
        return np.clip(value, 0.0, 2.5)
    if feature == "phase":
        return np.arctan2(np.sin(value), np.cos(value))
    return np.clip(value, 0.0, 1.0)


def _apply_programmed_state(state: np.ndarray, program_ids: np.ndarray) -> None:
    for program_id_value in np.unique(program_ids):
        program_id = int(program_id_value)
        program = decode_world_program(program_id)
        world_mask = program_ids == program_id
        for feature, (scale, shift) in world_program_feature_transform(program).items():
            feature_index = FEATURE_INDEX[feature]
            transformed = state[world_mask, :, feature_index] * scale + shift
            state[world_mask, :, feature_index] = _clip_programmed_feature(
                feature, transformed
            )


def _apply_programmed_relations(relations: np.ndarray, program_ids: np.ndarray) -> None:
    for program_id_value in np.unique(program_ids):
        program_id = int(program_id_value)
        program = decode_world_program(program_id)
        world_mask = program_ids == program_id
        for relation, (scale, shift) in world_program_relation_transform(
            program
        ).items():
            relation_index = RELATION_INDEX[relation]
            transformed = relations[world_mask, :, :, relation_index] * scale + shift
            relations[world_mask, :, :, relation_index] = np.clip(transformed, 0.0, 1.0)


def _program_root_gain_matrix(program_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            world_program_root_gains(decode_world_program(int(program_id)))
            for program_id in program_ids
        ],
        dtype=np.float32,
    )


class ProceduralWorldCompiler:
    """Compile seed ranges into persistent, typed, multi-domain worlds."""

    def __init__(self, config: CausalWorldConfig) -> None:
        config.validate()
        self.config = config

    def compile_shard(
        self, shard_index: int, *, program_id: int | None = None
    ) -> WorldBatch:
        if isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise TypeError("shard index must be an integer")
        if not 0 <= shard_index < self.config.shard_count:
            raise ValueError("shard index is outside configured shard count")
        seed = self.config.seed + 1_000_003 * shard_index
        rng = np.random.default_rng(seed)
        worlds = self.config.worlds_per_shard
        entities = self.config.entity_count
        neighbors = self.config.neighbor_count

        world_seeds = rng.integers(
            1,
            np.iinfo(np.int64).max,
            size=worlds,
            dtype=np.int64,
        )
        domain_ids = np.arange(worlds, dtype=np.int16) % len(DOMAIN_NAMES)
        rng.shuffle(domain_ids)
        # Three NumPy validation shards share one curriculum regime so repeated
        # local interventions can crystallize.  The massive XLA path exercises
        # four distinct programs inside every shard.
        if program_id is None:
            program = curriculum_programs(shard_index // 3, programs_per_shard=1)[0]
        else:
            if isinstance(program_id, bool) or not isinstance(program_id, int):
                raise TypeError("program ID override must be an integer")
            program = decode_world_program(program_id)
        program_ids = np.full(worlds, program.program_id, dtype=np.int64)

        active_probability = rng.uniform(0.78, 0.99, size=(worlds, 1))
        active_mask = (rng.random((worlds, entities)) <= active_probability).astype(
            np.float32
        )
        active_mask[:, 0] = 1.0

        state = np.zeros((worlds, entities, len(FEATURE_NAMES)), dtype=np.float32)
        state[..., FEATURE_INDEX["existence"]] = active_mask
        state[..., FEATURE_INDEX["energy"]] = rng.uniform(
            0.15, 1.15, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["mass"]] = rng.uniform(
            0.20, 1.40, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["charge"]] = rng.choice(
            np.asarray((-1.0, 1.0), dtype=np.float32), size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["temperature"]] = rng.uniform(
            0.05, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["pressure"]] = rng.uniform(
            0.05, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["cohesion"]] = rng.uniform(
            -0.6, 0.8, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["integrity"]] = rng.uniform(
            0.35, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["support"]] = rng.uniform(
            0.1, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["lifetime"]] = rng.uniform(
            0.35, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["resource"]] = rng.uniform(
            0.05, 1.25, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["health"]] = rng.uniform(
            0.35, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["signal"]] = rng.uniform(
            0.0, 0.7, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["trust"]] = rng.uniform(
            0.1, 0.9, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["belief"]] = rng.uniform(
            -1.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["goal"]] = rng.uniform(
            -1.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["ownership"]] = rng.uniform(
            0.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["value"]] = rng.uniform(
            -1.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1] = (
            rng.uniform(-1.0, 1.0, size=(worlds, entities, 3))
        )
        state[..., FEATURE_INDEX["velocity_x"] : FEATURE_INDEX["velocity_z"] + 1] = (
            rng.uniform(-0.08, 0.08, size=(worlds, entities, 3))
        )
        state[..., FEATURE_INDEX["polarity"]] = rng.uniform(
            -1.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["phase"]] = rng.uniform(
            -np.pi, np.pi, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["activation"]] = rng.uniform(
            0.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["structure"]] = (
            rng.random((worlds, entities)) > 0.88
        ).astype(np.float32)
        state[..., FEATURE_INDEX["uncertainty"]] = rng.uniform(
            0.15, 0.95, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["novelty"]] = rng.uniform(
            0.0, 1.0, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["memory_strength"]] = rng.uniform(
            0.0, 0.65, size=(worlds, entities)
        )
        state[..., FEATURE_INDEX["language_alignment"]] = rng.uniform(
            -1.0, 1.0, size=(worlds, entities)
        )

        self._specialize_domains(state, domain_ids, rng)
        _apply_programmed_state(state, program_ids)
        state *= active_mask[..., None]

        entity_ids = np.arange(entities, dtype=np.int32)[None, :, None]
        offsets = np.arange(1, neighbors + 1, dtype=np.int32)[None, None, :]
        ring = (entity_ids + offsets) % entities
        ring = np.broadcast_to(ring, (worlds, entities, neighbors)).copy()
        random_links = rng.integers(
            1, entities, size=(worlds, entities, neighbors), dtype=np.int32
        )
        random_links = (random_links + entity_ids) % entities
        random_link_probabilities = np.asarray(
            [
                world_program_random_link_probability(
                    decode_world_program(int(program_id))
                )
                for program_id in program_ids
            ],
            dtype=np.float32,
        )[:, None, None]
        use_random = (
            rng.random((worlds, entities, neighbors)) < random_link_probabilities
        )
        neighbor_indices = np.where(use_random, random_links, ring).astype(np.int32)

        relations = rng.uniform(
            0.05,
            1.0,
            size=(worlds, entities, neighbors, len(RELATION_NAMES)),
        ).astype(np.float32)
        relations[..., RELATION_INDEX["causal_delay"]] = (
            rng.integers(1, 5, size=(worlds, entities, neighbors)) / 4.0
        )
        relations[..., RELATION_INDEX["relation_uncertainty"]] = rng.uniform(
            0.05, 0.85, size=(worlds, entities, neighbors)
        )
        _apply_programmed_relations(relations, program_ids)
        self._refresh_distances(state, neighbor_indices, relations)

        budgets = np.stack(
            (
                (state[..., FEATURE_INDEX["mass"]] * active_mask).sum(axis=1),
                (state[..., FEATURE_INDEX["energy"]] * active_mask).sum(axis=1),
                (state[..., FEATURE_INDEX["resource"]] * active_mask).sum(axis=1),
            ),
            axis=1,
        ).astype(np.float32)
        batch = WorldBatch(
            state=state,
            neighbor_indices=neighbor_indices,
            relations=relations,
            active_mask=active_mask,
            domain_ids=domain_ids,
            program_ids=program_ids,
            world_seeds=world_seeds,
            initial_budgets=budgets,
        )
        batch.validate()
        return batch

    @staticmethod
    def _specialize_domains(
        state: np.ndarray, domain_ids: np.ndarray, rng: np.random.Generator
    ) -> None:
        worlds, entities, _ = state.shape
        biological = domain_ids == DOMAIN_INDEX["biological"]
        ecological = domain_ids == DOMAIN_INDEX["ecological"]
        agent = domain_ids == DOMAIN_INDEX["agent"]
        social = domain_ids == DOMAIN_INDEX["social"]
        symbolic = domain_ids == DOMAIN_INDEX["symbolic"]
        language = domain_ids == DOMAIN_INDEX["language"]
        chemical = domain_ids == DOMAIN_INDEX["chemical"]
        state[chemical, :, FEATURE_INDEX["pressure"]] *= 1.35
        state[chemical, :, FEATURE_INDEX["temperature"]] *= 1.25
        state[biological | ecological, :, FEATURE_INDEX["resource"]] *= 1.4
        state[biological | ecological, :, FEATURE_INDEX["health"]] = rng.uniform(
            0.55, 1.0, size=((biological | ecological).sum(), entities)
        )
        state[agent | social, :, FEATURE_INDEX["goal"]] *= 1.35
        state[agent | social, :, FEATURE_INDEX["trust"]] *= 1.1
        state[symbolic, :, FEATURE_INDEX["value"]] = (
            rng.integers(-3, 4, size=(symbolic.sum(), entities)) / 3.0
        )
        state[symbolic, :, FEATURE_INDEX["structure"]] = (
            rng.random((symbolic.sum(), entities)) > 0.5
        ).astype(np.float32)
        state[language, :, FEATURE_INDEX["signal"]] *= 1.45
        state[language, :, FEATURE_INDEX["language_alignment"]] = rng.uniform(
            -1.0, 1.0, size=(language.sum(), entities)
        )

    @staticmethod
    def _refresh_distances(
        state: np.ndarray,
        neighbor_indices: np.ndarray,
        relations: np.ndarray,
    ) -> None:
        worlds = state.shape[0]
        batch_index = np.arange(worlds)[:, None, None]
        positions = state[
            ...,
            FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1,
        ]
        neighbor_positions = positions[batch_index, neighbor_indices]
        distances = np.linalg.norm(
            neighbor_positions - positions[:, :, None, :], axis=-1
        )
        relations[..., RELATION_INDEX["distance"]] = np.clip(distances / 3.5, 0.0, 1.0)


def generate_interventions(count: int) -> tuple[Intervention, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("intervention count must be a positive integer")
    target_rules = ("most_active", "highest_uncertainty", "least_active")
    interventions: list[Intervention] = []
    for index in range(count):
        feature = INTERVENTION_FEATURES[index % len(INTERVENTION_FEATURES)]
        polarity = 1 if (index // len(INTERVENTION_FEATURES)) % 2 == 0 else -1
        magnitude = 0.18 + 0.07 * (index % 5)
        intervention = Intervention(
            intervention_id=f"iv-{index:04d}-{feature}-{polarity:+d}",
            feature=feature,
            magnitude=magnitude,
            polarity=polarity,
            target_rule=target_rules[index % len(target_rules)],
            duration=1 + index % 3,
        )
        intervention.validate()
        interventions.append(intervention)
    return tuple(interventions)


def normalize_causal_response(
    signed_effect: float, intervention_delta: float
) -> tuple[int, float]:
    """Convert a raw treated-minus-baseline effect into a causal derivative."""

    if not np.isfinite(signed_effect) or not np.isfinite(intervention_delta):
        raise ValueError("causal response inputs must be finite")
    if abs(intervention_delta) < 1e-12:
        raise ValueError("causal response requires a nonzero intervention")
    response = signed_effect / intervention_delta
    if abs(response) < 1e-7:
        return 1, 0.0
    return (1 if response > 0.0 else -1), abs(float(response))


def _intervention_targets(batch: WorldBatch, intervention: Intervention) -> np.ndarray:
    state = batch.state
    active = batch.active_mask
    if intervention.target_rule == "most_active":
        score = state[..., FEATURE_INDEX["activation"]] - 10.0 * (1.0 - active)
        return score.argmax(axis=1)
    if intervention.target_rule == "least_active":
        score = state[..., FEATURE_INDEX["activation"]] + 10.0 * (1.0 - active)
        return score.argmin(axis=1)
    score = state[..., FEATURE_INDEX["uncertainty"]] - 10.0 * (1.0 - active)
    return score.argmax(axis=1)


def apply_intervention(batch: WorldBatch, intervention: Intervention) -> np.ndarray:
    intervention.validate()
    targets = _intervention_targets(batch, intervention)
    feature_index = FEATURE_INDEX[intervention.feature]
    world_index = np.arange(batch.state.shape[0])
    batch.state[world_index, targets, feature_index] += (
        intervention.polarity * intervention.magnitude
    )
    if intervention.feature in {
        "energy",
        "temperature",
        "pressure",
        "resource",
        "signal",
        "trust",
    }:
        batch.state[world_index, targets, feature_index] = np.clip(
            batch.state[world_index, targets, feature_index], 0.0, 2.5
        )
    else:
        batch.state[world_index, targets, feature_index] = np.clip(
            batch.state[world_index, targets, feature_index], -1.0, 1.0
        )
    return targets


def _weighted_neighbor_mean(
    neighbor_values: np.ndarray,
    neighbor_active: np.ndarray,
    relation_weight: np.ndarray,
) -> np.ndarray:
    weights = neighbor_active * relation_weight
    return _safe_ratio(
        (neighbor_values * weights).sum(axis=2),
        weights.sum(axis=2),
    )


def advance_world(batch: WorldBatch) -> dict[str, float]:
    """Advance every compiled world one tick through all seven mechanics."""

    batch.validate()
    state = batch.state
    worlds = state.shape[0]
    batch_index = np.arange(worlds)[:, None, None]
    neighbor_state = state[batch_index, batch.neighbor_indices]
    neighbor_active = batch.active_mask[batch_index, batch.neighbor_indices]
    roots = _DOMAIN_ROOT_STRENGTHS[batch.domain_ids] * _program_root_gain_matrix(
        batch.program_ids
    )

    conductivity = batch.relations[..., RELATION_INDEX["conductivity"]]
    permeability = batch.relations[..., RELATION_INDEX["permeability"]]
    affinity = batch.relations[..., RELATION_INDEX["affinity"]]
    trust_channel = batch.relations[..., RELATION_INDEX["trust_channel"]]
    resource_channel = batch.relations[..., RELATION_INDEX["resource_channel"]]
    symbolic_match = batch.relations[..., RELATION_INDEX["symbolic_match"]]
    language_channel = batch.relations[..., RELATION_INDEX["language_channel"]]
    distance_weight = 1.0 - batch.relations[..., RELATION_INDEX["distance"]]

    signal_i = FEATURE_INDEX["signal"]
    energy_i = FEATURE_INDEX["energy"]
    temperature_i = FEATURE_INDEX["temperature"]
    pressure_i = FEATURE_INDEX["pressure"]
    mass_i = FEATURE_INDEX["mass"]
    charge_i = FEATURE_INDEX["charge"]
    cohesion_i = FEATURE_INDEX["cohesion"]
    integrity_i = FEATURE_INDEX["integrity"]
    support_i = FEATURE_INDEX["support"]
    lifetime_i = FEATURE_INDEX["lifetime"]
    resource_i = FEATURE_INDEX["resource"]
    health_i = FEATURE_INDEX["health"]
    trust_i = FEATURE_INDEX["trust"]
    belief_i = FEATURE_INDEX["belief"]
    value_i = FEATURE_INDEX["value"]
    polarity_i = FEATURE_INDEX["polarity"]
    phase_i = FEATURE_INDEX["phase"]
    structure_i = FEATURE_INDEX["structure"]
    uncertainty_i = FEATURE_INDEX["uncertainty"]
    memory_i = FEATURE_INDEX["memory_strength"]
    language_i = FEATURE_INDEX["language_alignment"]

    signal_mean = _weighted_neighbor_mean(
        neighbor_state[..., signal_i], neighbor_active, permeability
    )
    energy_mean = _weighted_neighbor_mean(
        neighbor_state[..., energy_i], neighbor_active, conductivity
    )
    temperature_mean = _weighted_neighbor_mean(
        neighbor_state[..., temperature_i], neighbor_active, conductivity
    )
    pressure_mean = _weighted_neighbor_mean(
        neighbor_state[..., pressure_i], neighbor_active, permeability
    )
    resource_mean = _weighted_neighbor_mean(
        neighbor_state[..., resource_i], neighbor_active, resource_channel
    )
    trust_mean = _weighted_neighbor_mean(
        neighbor_state[..., trust_i], neighbor_active, trust_channel
    )
    belief_mean = _weighted_neighbor_mean(
        neighbor_state[..., belief_i], neighbor_active, trust_channel
    )
    value_mean = _weighted_neighbor_mean(
        neighbor_state[..., value_i], neighbor_active, symbolic_match
    )
    language_mean = _weighted_neighbor_mean(
        neighbor_state[..., language_i], neighbor_active, language_channel
    )

    radiation = roots[:, 0:1]
    dissipation = roots[:, 1:2]
    gravitation = roots[:, 2:3]
    attraction = roots[:, 3:4]
    nucleation = roots[:, 4:5]
    conservation = roots[:, 5:6]
    decay = roots[:, 6:7]

    next_state = state.copy()
    next_state[..., signal_i] += 0.17 * radiation * (signal_mean - state[..., signal_i])
    next_state[..., energy_i] += 0.10 * radiation * (energy_mean - state[..., energy_i])
    next_state[..., temperature_i] += (
        0.13 * dissipation * (temperature_mean - state[..., temperature_i])
    )
    next_state[..., pressure_i] += (
        0.09 * dissipation * (pressure_mean - state[..., pressure_i])
    )

    positions = state[
        ..., FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1
    ]
    velocities = state[
        ..., FEATURE_INDEX["velocity_x"] : FEATURE_INDEX["velocity_z"] + 1
    ]
    neighbor_positions = neighbor_state[
        ..., FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1
    ]
    neighbor_mass = neighbor_state[..., mass_i]
    positional_weights = neighbor_active * distance_weight * neighbor_mass
    center = _safe_ratio(
        (neighbor_positions * positional_weights[..., None]).sum(axis=2),
        positional_weights.sum(axis=2)[..., None],
    )
    acceleration = 0.025 * gravitation[..., None] * (center - positions)

    signed_affinity = (
        -state[..., charge_i, None] * neighbor_state[..., charge_i]
        + state[..., polarity_i, None] * neighbor_state[..., polarity_i]
    )
    relation_force = np.tanh(signed_affinity) * affinity * neighbor_active
    directional = neighbor_positions - positions[:, :, None, :]
    directional /= np.maximum(np.linalg.norm(directional, axis=-1, keepdims=True), 1e-5)
    attraction_force = (
        (relation_force[..., None] * directional).mean(axis=2)
        * 0.018
        * attraction[..., None]
    )
    velocities = np.clip(
        0.94 * velocities + acceleration + attraction_force, -0.35, 0.35
    )
    positions = np.clip(positions + velocities, -4.0, 4.0)
    next_state[..., FEATURE_INDEX["velocity_x"] : FEATURE_INDEX["velocity_z"] + 1] = (
        velocities
    )
    next_state[..., FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1] = (
        positions
    )
    next_state[..., cohesion_i] = np.tanh(
        state[..., cohesion_i] + 0.08 * attraction * relation_force.mean(axis=2)
    )

    phase_neighbors = neighbor_state[..., phase_i]
    phase_delta = np.sin(phase_neighbors - state[..., phase_i, None])
    phase_coupling = (phase_delta * affinity * neighbor_active).mean(axis=2)
    next_state[..., phase_i] = np.arctan2(
        np.sin(state[..., phase_i] + 0.12 * phase_coupling),
        np.cos(state[..., phase_i] + 0.12 * phase_coupling),
    )

    potential = (
        0.24 * _clip01(next_state[..., energy_i])
        + 0.18 * _clip01(next_state[..., pressure_i])
        + 0.20 * _clip01(next_state[..., cohesion_i] * 0.5 + 0.5)
        + 0.18 * _clip01(state[..., support_i])
        + 0.20 * _clip01(state[..., resource_i])
    )
    formed = potential * nucleation >= (0.56 + 0.18 * state[..., uncertainty_i])
    next_state[..., structure_i] = np.maximum(
        state[..., structure_i], formed.astype(np.float32)
    )
    next_state[..., integrity_i] += 0.08 * formed * (1.0 - state[..., integrity_i])
    next_state[..., memory_i] += 0.06 * formed * (1.0 - state[..., memory_i])

    _apply_domain_dynamics(
        next_state,
        neighbor_state,
        neighbor_active,
        batch.domain_ids,
        resource_mean,
        trust_mean,
        belief_mean,
        value_mean,
        language_mean,
    )

    next_state[..., lifetime_i] -= 0.008 + 0.018 * decay * (
        1.0 - _clip01(state[..., support_i])
    )
    next_state[..., integrity_i] -= (
        0.006 * decay * (1.0 - _clip01(state[..., cohesion_i] * 0.5 + 0.5))
    )
    next_state[..., health_i] -= 0.004 * decay * (1.0 - _clip01(state[..., resource_i]))
    expired = (
        (next_state[..., lifetime_i] <= 0.0)
        | (next_state[..., integrity_i] <= 0.0)
        | (next_state[..., health_i] <= 0.0)
    ) & (next_state[..., structure_i] < 0.5)
    next_active = batch.active_mask * (~expired).astype(np.float32)
    no_survivor = next_active.sum(axis=1) <= 0.0
    if no_survivor.any():
        keeper = state[..., support_i].argmax(axis=1)
        next_active[no_survivor, keeper[no_survivor]] = 1.0

    for feature, lower, upper in (
        ("energy", 0.0, 2.5),
        ("mass", 0.0, 3.0),
        ("temperature", 0.0, 2.5),
        ("pressure", 0.0, 2.5),
        ("integrity", 0.0, 1.0),
        ("support", 0.0, 1.0),
        ("lifetime", -0.2, 1.5),
        ("resource", 0.0, 2.5),
        ("health", 0.0, 1.0),
        ("signal", 0.0, 2.5),
        ("trust", 0.0, 1.0),
        ("activation", 0.0, 1.0),
        ("structure", 0.0, 1.0),
        ("uncertainty", 0.0, 1.0),
        ("novelty", 0.0, 1.0),
        ("memory_strength", 0.0, 1.0),
    ):
        index = FEATURE_INDEX[feature]
        next_state[..., index] = np.clip(next_state[..., index], lower, upper)
    for feature in (
        "charge",
        "cohesion",
        "belief",
        "goal",
        "value",
        "polarity",
        "language_alignment",
    ):
        index = FEATURE_INDEX[feature]
        next_state[..., index] = np.clip(next_state[..., index], -1.0, 1.0)

    next_state *= next_active[..., None]
    next_state[..., FEATURE_INDEX["existence"]] = next_active
    _conserve_budgets(
        next_state,
        state,
        next_active,
        batch.initial_budgets,
        conservation,
    )
    batch.state = next_state.astype(np.float32)
    batch.active_mask = next_active.astype(np.float32)
    batch.tick += 1
    ProceduralWorldCompiler._refresh_distances(
        batch.state, batch.neighbor_indices, batch.relations
    )

    current = np.stack(
        (
            (batch.state[..., mass_i] * next_active).sum(axis=1),
            (batch.state[..., energy_i] * next_active).sum(axis=1),
            (batch.state[..., resource_i] * next_active).sum(axis=1),
        ),
        axis=1,
    )
    invariant_error = np.abs(
        _safe_ratio(current - batch.initial_budgets, batch.initial_budgets)
    )
    return {
        "maximum_invariant_error": float(invariant_error.max(initial=0.0)),
        "mean_invariant_error": float(invariant_error.mean()),
        "active_fraction": float(next_active.mean()),
        "mean_structure": float(batch.state[..., structure_i].mean()),
    }


def _apply_domain_dynamics(
    state: np.ndarray,
    neighbor_state: np.ndarray,
    neighbor_active: np.ndarray,
    domain_ids: np.ndarray,
    resource_mean: np.ndarray,
    trust_mean: np.ndarray,
    belief_mean: np.ndarray,
    value_mean: np.ndarray,
    language_mean: np.ndarray,
) -> None:
    physical = _domain_mask(domain_ids, "physical")
    chemical = _domain_mask(domain_ids, "chemical")
    biological = _domain_mask(domain_ids, "biological")
    ecological = _domain_mask(domain_ids, "ecological")
    agent = _domain_mask(domain_ids, "agent")
    social = _domain_mask(domain_ids, "social")
    symbolic = _domain_mask(domain_ids, "symbolic")
    language = _domain_mask(domain_ids, "language")

    energy = state[..., FEATURE_INDEX["energy"]]
    temperature = state[..., FEATURE_INDEX["temperature"]]
    pressure = state[..., FEATURE_INDEX["pressure"]]
    cohesion = state[..., FEATURE_INDEX["cohesion"]]
    integrity = state[..., FEATURE_INDEX["integrity"]]
    resource = state[..., FEATURE_INDEX["resource"]]
    health = state[..., FEATURE_INDEX["health"]]
    signal = state[..., FEATURE_INDEX["signal"]]
    trust = state[..., FEATURE_INDEX["trust"]]
    belief = state[..., FEATURE_INDEX["belief"]]
    goal = state[..., FEATURE_INDEX["goal"]]
    value = state[..., FEATURE_INDEX["value"]]
    activation = state[..., FEATURE_INDEX["activation"]]
    structure = state[..., FEATURE_INDEX["structure"]]
    uncertainty = state[..., FEATURE_INDEX["uncertainty"]]
    memory = state[..., FEATURE_INDEX["memory_strength"]]
    alignment = state[..., FEATURE_INDEX["language_alignment"]]

    # Give the physical domain an explicit thermodynamic cascade.  Combined
    # with the domain programs below, this makes compositional causal paths a
    # property of the simulated world rather than an evaluator convenience.
    temperature += physical * 0.018 * np.tanh(energy - temperature)
    pressure += physical * 0.014 * np.tanh(temperature - pressure)
    activation += physical * 0.010 * _clip01(pressure) * (1.0 - activation)

    reaction = _clip01((temperature + pressure + energy - 1.45) / 1.4)
    energy -= chemical * 0.025 * reaction
    cohesion += chemical * 0.05 * reaction * (1.0 - cohesion)
    structure[:] = np.maximum(structure, chemical * (reaction > 0.58))

    metabolism = biological * np.minimum(resource, 0.026 + 0.018 * activation)
    resource -= metabolism
    energy += 0.72 * metabolism
    health += 0.035 * metabolism - biological * 0.006
    integrity += biological * 0.012 * health * (1.0 - integrity)

    resource_flow = ecological * 0.045 * (resource_mean - resource)
    resource += resource_flow
    health += ecological * 0.018 * np.tanh(resource - 0.35)
    cohesion += ecological * 0.014 * np.tanh(trust_mean - 0.5)

    goal_error = goal - belief
    activation += agent * 0.06 * np.abs(goal_error) * (1.0 - activation)
    belief += agent * 0.035 * np.tanh(signal - belief)
    activation += agent * 0.025 * np.tanh(belief) * (1.0 - activation)
    uncertainty -= agent * 0.025 * np.abs(signal - belief)
    energy -= agent * 0.009 * activation

    trust += social * 0.05 * (trust_mean - trust)
    belief += social * 0.035 * trust * (belief_mean - belief)
    cohesion += social * 0.03 * (trust - 0.5)
    memory += social * 0.018 * trust * (1.0 - memory)

    symbolic_error = value_mean - value
    value += symbolic * 0.11 * symbolic_error
    belief += symbolic * 0.08 * np.tanh(symbolic_error)
    activation += symbolic * 0.04 * (1.0 - np.abs(symbolic_error))
    uncertainty -= symbolic * 0.045 * (1.0 - np.abs(symbolic_error))

    alignment_error = language_mean - alignment
    alignment += language * 0.10 * alignment_error
    signal += language * 0.055 * _clip01(1.0 - np.abs(alignment_error))
    belief += language * 0.06 * trust * alignment_error
    memory += language * 0.035 * _clip01(signal) * (1.0 - memory)
    uncertainty -= language * 0.03 * _clip01(memory)

    # Neighbor competition introduces context-dependent failures rather than
    # making every domain monotonic and trivially predictable.
    neighbor_pressure = (
        neighbor_state[..., FEATURE_INDEX["activation"]] * neighbor_active
    ).mean(axis=2)
    health -= (biological + ecological) * 0.008 * neighbor_pressure


def _conserve_budgets(
    state: np.ndarray,
    previous_state: np.ndarray,
    active: np.ndarray,
    initial_budgets: np.ndarray,
    conservation: np.ndarray,
) -> None:
    active_distribution = _safe_ratio(active, active.sum(axis=1, keepdims=True))
    retention = np.clip(conservation, 0.0, 1.0)
    for budget_index, feature in enumerate(("mass", "energy", "resource")):
        feature_index = FEATURE_INDEX[feature]
        target = initial_budgets[:, budget_index : budget_index + 1]
        proposal = np.maximum(state[..., feature_index], 0.0) * active
        proposal_total = proposal.sum(axis=1, keepdims=True)
        proposal_normalized = np.where(
            proposal_total > 1e-6,
            proposal * _safe_ratio(target, proposal_total),
            active_distribution * target,
        )
        previous = np.maximum(previous_state[..., feature_index], 0.0) * active
        previous_total = previous.sum(axis=1, keepdims=True)
        previous_normalized = np.where(
            previous_total > 1e-6,
            previous * _safe_ratio(target, previous_total),
            active_distribution * target,
        )
        # Universal conservation protects the total budget.  The root strength
        # controls how much of the previous local distribution survives versus
        # how much the current dynamics may redistribute among living entities.
        state[..., feature_index] = (
            retention * previous_normalized
            + (1.0 - retention) * proposal_normalized
        ) * active


def rollout_counterfactual_pair(
    initial: WorldBatch,
    intervention: Intervention,
    steps: int,
) -> tuple[tuple[CausalEvidence, ...], dict[str, Any]]:
    """Compare matched treated and untreated worlds and emit causal evidence."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("rollout steps must be a positive integer")
    intervention.validate()
    baseline = initial.copy()
    treated = initial.copy()
    effects: list[np.ndarray] = []
    invariant_errors: list[float] = []
    targets: np.ndarray | None = None
    for tick in range(steps):
        if (
            intervention.start_tick
            <= tick
            < (intervention.start_tick + intervention.duration)
        ):
            targets = apply_intervention(treated, intervention)
        baseline_diagnostics = advance_world(baseline)
        treated_diagnostics = advance_world(treated)
        invariant_errors.extend(
            (
                baseline_diagnostics["maximum_invariant_error"],
                treated_diagnostics["maximum_invariant_error"],
            )
        )
        effects.append((treated.state - baseline.state).mean(axis=1))
    if targets is None:
        raise AssertionError("intervention was not applied during rollout")

    effect_trace = np.stack(effects, axis=0)
    observable_indices = np.asarray(
        [FEATURE_INDEX[name] for name in OBSERVABLE_FEATURES], dtype=np.int32
    )
    evidence: list[CausalEvidence] = []
    for domain_index, domain in enumerate(DOMAIN_NAMES):
        world_mask = initial.domain_ids == domain_index
        if not world_mask.any():
            continue
        domain_program_ids = np.unique(initial.program_ids[world_mask])
        if len(domain_program_ids) != 1:
            raise ValueError(
                "NumPy evidence reduction requires one world program per shard"
            )
        program = decode_world_program(int(domain_program_ids[0]))
        domain_trace = effect_trace[:, world_mask][:, :, observable_indices]
        mean_by_tick = domain_trace.mean(axis=1)
        effect_strength = np.max(np.abs(mean_by_tick), axis=0)
        observable_positions = np.argsort(-effect_strength, kind="stable")[
            :CAUSAL_EFFECTS_PER_INTERVENTION
        ]
        for observable_position_value in observable_positions:
            observable_position = int(observable_position_value)
            peak_tick = int(np.abs(mean_by_tick[:, observable_position]).argmax())
            effect_feature = OBSERVABLE_FEATURES[observable_position]
            world_effect = domain_trace[peak_tick, :, observable_position]
            signed_mean = float(world_effect.mean())
            direction, magnitude = normalize_causal_response(
                signed_mean,
                intervention.polarity * intervention.magnitude,
            )
            context = (
                f"domain:{domain}",
                f"cause:{intervention.feature}",
                f"target:{intervention.target_rule}",
                f"polarity:{intervention.polarity:+d}",
                f"world_tick:{initial.tick}",
                *program.condition_signature(),
            )
            provenance_payload = {
                "intervention": asdict(intervention),
                "domain": domain,
                "world_program": program.manifest(),
                "effect_feature": effect_feature,
                "peak_tick": peak_tick,
                "world_seeds": initial.world_seeds[world_mask].tolist(),
                "initial_state": ndarray_digest(initial.state[world_mask]),
                "final_baseline": ndarray_digest(baseline.state[world_mask]),
                "final_treated": ndarray_digest(treated.state[world_mask]),
            }
            item = CausalEvidence(
                evidence_id=(
                    f"evidence-{intervention.intervention_id}-{domain}-{effect_feature}"
                ),
                domain=domain,
                cause_feature=intervention.feature,
                effect_feature=effect_feature,
                direction=direction,
                magnitude=magnitude,
                delay=peak_tick + 1,
                context_signature=context,
                treated_worlds=int(world_mask.sum()),
                baseline_worlds=int(world_mask.sum()),
                variance=float(world_effect.var()),
                invariant_error=float(max(invariant_errors, default=0.0)),
                provenance_hash=canonical_hash(provenance_payload),
            )
            item.validate()
            evidence.append(item)
    diagnostics = {
        "runtime": CAUSAL_WORLD_SIMULATOR_RUNTIME,
        "intervention": asdict(intervention),
        "steps": steps,
        "target_count": int(len(targets)),
        "evidence_count": len(evidence),
        "maximum_invariant_error": float(max(invariant_errors, default=0.0)),
        "effect_trace_digest": ndarray_digest(effect_trace),
        "baseline_final_digest": ndarray_digest(baseline.state),
        "treated_final_digest": ndarray_digest(treated.state),
    }
    return tuple(evidence), diagnostics


def summarize_world(batch: WorldBatch) -> dict[str, Any]:
    batch.validate()
    domain_counts = {
        domain: int((batch.domain_ids == index).sum())
        for index, domain in enumerate(DOMAIN_NAMES)
    }
    return {
        "runtime": CAUSAL_WORLD_SIMULATOR_RUNTIME,
        "worlds": int(batch.state.shape[0]),
        "entities_per_world": int(batch.state.shape[1]),
        "neighbors_per_entity": int(batch.neighbor_indices.shape[2]),
        "tick": batch.tick,
        "active_fraction": float(batch.active_mask.mean()),
        "domain_counts": domain_counts,
        "program_ids": sorted({int(value) for value in batch.program_ids}),
        "programs": [
            decode_world_program(int(value)).manifest()
            for value in sorted(set(batch.program_ids.tolist()))
        ],
        "state_digest": ndarray_digest(batch.state),
        "relation_digest": ndarray_digest(batch.relations),
        "seed_digest": ndarray_digest(batch.world_seeds),
    }


def causal_world_simulator_self_test() -> dict[str, bool]:
    config = get_profile("test")
    compiler = ProceduralWorldCompiler(config)
    first = compiler.compile_shard(0)
    replay = compiler.compile_shard(0)
    interventions = generate_interventions(config.intervention_candidates)
    evidence, diagnostics = rollout_counterfactual_pair(
        first, interventions[0], config.time_steps
    )
    checks = {
        "deterministic_compilation": np.array_equal(first.state, replay.state),
        "all_domains_represented": set(first.domain_ids.tolist())
        == set(range(len(DOMAIN_NAMES))),
        "intervention_produces_multiple_effects": len(evidence)
        == len(DOMAIN_NAMES) * CAUSAL_EFFECTS_PER_INTERVENTION,
        "evidence_has_provenance": all(
            len(item.provenance_hash) == 64 for item in evidence
        ),
        "state_is_finite": bool(np.isfinite(first.state).all()),
        "relations_are_finite": bool(np.isfinite(first.relations).all()),
        "conservation_is_bounded": diagnostics["maximum_invariant_error"] < 0.25,
        "opposite_interventions_preserve_derivative_direction": (
            normalize_causal_response(0.2, 0.1)
            == normalize_causal_response(-0.2, -0.1)
            == (1, 2.0)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"causal-world simulator self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(causal_world_simulator_self_test())
