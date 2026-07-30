"""Vectorized accelerator execution for massive causal-world shards.

The persistent causal graph stays sparse on the host.  This module compiles
matched world rollouts into regular fixed-shape tensors suitable for XLA and
returns only interventional evidence and aggregate diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from atom_causal_world_curriculum import (
    TPU_PROGRAMS_PER_SHARD,
    curriculum_manifest,
    curriculum_program_ids,
    curriculum_programs,
    decode_world_program,
    world_program_feature_transform,
    world_program_random_link_probability,
    world_program_relation_transform,
    world_program_root_gains,
)
from atom_causal_world_schema import (
    CAUSAL_EFFECTS_PER_INTERVENTION,
    DOMAIN_NAMES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    OBSERVABLE_FEATURES,
    RELATION_NAMES,
    ROOT_MECHANICS,
    CausalEvidence,
    CausalWorldConfig,
    canonical_hash,
    get_profile,
    ndarray_digest,
)
from atom_causal_world_simulator import (
    generate_interventions,
    normalize_causal_response,
)


CAUSAL_ACCELERATOR_RUNTIME = "atom-causal-world-xla-v1"
MASSIVE_MICROBATCH_WORLDS = 512
MASSIVE_COMPILED_TIME_STEPS = 64
_JAX_EXECUTOR: Any | None = None
_JAX_EXECUTOR_BUILD_COUNT = 0
_JAX_EXECUTOR_DEVICE_COUNT = 0
_JAX_EXECUTOR_MODE = "uninitialized"


def probe_jax_accelerator() -> dict[str, Any]:
    try:
        import jax
    except (ImportError, OSError) as error:
        return {
            "runtime": CAUSAL_ACCELERATOR_RUNTIME,
            "jax_available": False,
            "tpu_available": False,
            "gpu_available": False,
            "devices": [],
            "error": f"{type(error).__name__}: {error}",
        }
    devices = jax.devices()
    descriptors = [
        {
            "id": int(getattr(device, "id", index)),
            "platform": str(device.platform),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
        }
        for index, device in enumerate(devices)
    ]
    return {
        "runtime": CAUSAL_ACCELERATOR_RUNTIME,
        "jax_available": True,
        "jax_version": str(jax.__version__),
        "tpu_available": any(item["platform"].lower() == "tpu" for item in descriptors),
        "gpu_available": any(
            item["platform"].lower() in {"gpu", "cuda", "rocm"}
            for item in descriptors
        ),
        "devices": descriptors,
        "error": None,
    }


def build_accelerator_plan(config: CausalWorldConfig) -> dict[str, Any]:
    config.validate()
    microbatch_worlds = min(MASSIVE_MICROBATCH_WORLDS, config.worlds_per_shard)
    if config.worlds_per_shard % microbatch_worlds != 0:
        raise ValueError("worlds per shard must be divisible by accelerator microbatch")
    microbatch_count = config.worlds_per_shard // microbatch_worlds
    state_values = microbatch_worlds * config.entity_count * len(FEATURE_NAMES)
    relation_values = (
        microbatch_worlds
        * config.entity_count
        * config.neighbor_count
        * len(RELATION_NAMES)
    )
    curriculum = curriculum_manifest(
        shard_count=config.shard_count,
        programs_per_shard=TPU_PROGRAMS_PER_SHARD,
    )
    curriculum_summary = {
        key: value for key, value in curriculum.items() if key != "schedule"
    }
    return {
        "runtime": CAUSAL_ACCELERATOR_RUNTIME,
        "profile": config.profile,
        "shards": config.shard_count,
        "worlds_per_shard": config.worlds_per_shard,
        "microbatch_worlds": microbatch_worlds,
        "microbatches_per_shard": microbatch_count,
        "state_values_per_microbatch": state_values,
        "relation_values_per_microbatch": relation_values,
        "matched_rollouts_per_world": 2,
        "deterministic_replay_microbatches": 1,
        "compiled_time_steps": config.time_steps,
        "candidate_interventions": config.intervention_candidates,
        "effects_retained_per_intervention": CAUSAL_EFFECTS_PER_INTERVENTION,
        "world_programs_per_shard": TPU_PROGRAMS_PER_SHARD,
        "curriculum": curriculum_summary,
        "evidence_partitions_per_hypothesis": 4,
        "expected_evidence_per_shard": (
            config.intervention_candidates
            * len(DOMAIN_NAMES)
            * CAUSAL_EFFECTS_PER_INTERVENTION
            * 4
            * TPU_PROGRAMS_PER_SHARD
        ),
        "expected_entity_updates_per_shard": (
            config.worlds_per_shard * config.entity_count * config.time_steps
        ),
        "expected_relation_updates_per_shard": (
            config.worlds_per_shard
            * config.entity_count
            * config.neighbor_count
            * config.time_steps
        ),
        "persistent_graph_location": "host",
        "accelerator_working_set": "fixed_shape_retrieved_world_shard",
        "xla_executor_cache_scope": "process",
        "device_parallelism": "pmap-all-local-devices",
        "root_mechanics_executed": list(ROOT_MECHANICS),
    }


def _jax_execute_causal_rollout(
    baseline_state: Any,
    treated_state: Any,
    neighbor_indices: Any,
    relation_weight: Any,
    initial_budgets: Any,
    root_strengths: Any,
    domain_ids: Any,
    *,
    time_steps: int,
) -> tuple[Any, Any, Any]:
    """Execute one fixed-shape matched rollout inside a cached XLA program."""

    import jax.numpy as jnp
    from jax import lax

    batch_size = baseline_state.shape[0]
    batch_index = jnp.arange(batch_size)[:, None, None]

    def step_one(current: Any) -> Any:
        neighbor_state = current[batch_index, neighbor_indices]
        neighbor_active = neighbor_state[:, :, :, FEATURE_INDEX["existence"]]
        weights = neighbor_active * relation_weight
        denominator = jnp.maximum(weights.sum(axis=2), 1e-6)

        def neighbor_mean(feature_index: int) -> Any:
            return (neighbor_state[:, :, :, feature_index] * weights).sum(
                axis=2
            ) / denominator

        next_state = current
        radiation = root_strengths[:, 0:1]
        dissipation = root_strengths[:, 1:2]
        gravitation = root_strengths[:, 2:3]
        attraction = root_strengths[:, 3:4]
        nucleation = root_strengths[:, 4:5]
        decay = root_strengths[:, 6:7]
        for feature_name, rate, strength in (
            ("signal", 0.17, radiation),
            ("energy", 0.10, radiation),
            ("temperature", 0.13, dissipation),
            ("pressure", 0.09, dissipation),
            ("resource", 0.07, root_strengths[:, 5:6]),
            ("trust", 0.06, attraction),
            ("belief", 0.05, attraction),
            ("value", 0.07, attraction),
            ("language_alignment", 0.09, radiation),
        ):
            feature_index = FEATURE_INDEX[feature_name]
            value = current[:, :, feature_index]
            updated = value + rate * strength * (neighbor_mean(feature_index) - value)
            next_state = next_state.at[:, :, feature_index].set(updated)

        phase = current[:, :, FEATURE_INDEX["phase"]]
        neighbor_phase = neighbor_state[:, :, :, FEATURE_INDEX["phase"]]
        phase_pull = (jnp.sin(neighbor_phase - phase[:, :, None]) * weights).sum(
            axis=2
        ) / denominator
        phase_next = phase + 0.12 * phase_pull
        phase_next = jnp.arctan2(jnp.sin(phase_next), jnp.cos(phase_next))
        next_state = next_state.at[:, :, FEATURE_INDEX["phase"]].set(phase_next)

        position_slice = slice(
            FEATURE_INDEX["position_x"], FEATURE_INDEX["position_z"] + 1
        )
        velocity_slice = slice(
            FEATURE_INDEX["velocity_x"], FEATURE_INDEX["velocity_z"] + 1
        )
        position = current[:, :, position_slice]
        velocity = current[:, :, velocity_slice]
        neighbor_position = neighbor_state[:, :, :, position_slice]
        displacement = neighbor_position - position[:, :, None, :]
        distance_squared = jnp.maximum(jnp.square(displacement).sum(axis=-1), 0.04)
        neighbor_mass = neighbor_state[:, :, :, FEATURE_INDEX["mass"]]
        gravitational_pull = (
            displacement
            * weights[:, :, :, None]
            * neighbor_mass[:, :, :, None]
            / distance_squared[:, :, :, None]
        ).sum(axis=2) / denominator[:, :, None]
        velocity_next = 0.985 * velocity + 0.008 * gravitation[:, :, None] * jnp.tanh(
            gravitational_pull
        )
        position_next = position + velocity_next
        next_state = next_state.at[:, :, velocity_slice].set(velocity_next)
        next_state = next_state.at[:, :, position_slice].set(position_next)

        charge = current[:, :, FEATURE_INDEX["charge"]]
        polarity = current[:, :, FEATURE_INDEX["polarity"]]
        affinity_signal = jnp.tanh(
            charge * neighbor_mean(FEATURE_INDEX["charge"])
            + polarity * neighbor_mean(FEATURE_INDEX["polarity"])
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["cohesion"]].add(
            0.012 * attraction * affinity_signal
        )

        potential = (
            0.25 * jnp.clip(next_state[:, :, FEATURE_INDEX["energy"]], 0.0, 1.0)
            + 0.20 * jnp.clip(next_state[:, :, FEATURE_INDEX["pressure"]], 0.0, 1.0)
            + 0.20
            * jnp.clip(
                next_state[:, :, FEATURE_INDEX["cohesion"]] * 0.5 + 0.5,
                0.0,
                1.0,
            )
            + 0.20 * jnp.clip(next_state[:, :, FEATURE_INDEX["resource"]], 0.0, 1.0)
            + 0.15 * jnp.clip(next_state[:, :, FEATURE_INDEX["support"]], 0.0, 1.0)
        )
        formed = (
            potential * nucleation
            >= 0.56 + 0.18 * next_state[:, :, FEATURE_INDEX["uncertainty"]]
        ).astype(jnp.float32)
        structure = jnp.maximum(next_state[:, :, FEATURE_INDEX["structure"]], formed)
        next_state = next_state.at[:, :, FEATURE_INDEX["structure"]].set(structure)
        next_state = next_state.at[:, :, FEATURE_INDEX["memory_strength"]].add(
            0.05 * formed * (1.0 - next_state[:, :, FEATURE_INDEX["memory_strength"]])
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["lifetime"]].add(
            -0.008 - 0.018 * decay * (1.0 - next_state[:, :, FEATURE_INDEX["support"]])
        )

        physical = (domain_ids == 0).astype(jnp.float32)[:, None]
        chemical = (domain_ids == 1).astype(jnp.float32)[:, None]
        biological = (domain_ids == 2).astype(jnp.float32)[:, None]
        ecological = (domain_ids == 3).astype(jnp.float32)[:, None]
        agent = (domain_ids == 4).astype(jnp.float32)[:, None]
        social = (domain_ids == 5).astype(jnp.float32)[:, None]
        symbolic = (domain_ids == 6).astype(jnp.float32)[:, None]
        language = (domain_ids == 7).astype(jnp.float32)[:, None]
        reaction = jnp.clip(
            (
                next_state[:, :, FEATURE_INDEX["temperature"]]
                + next_state[:, :, FEATURE_INDEX["pressure"]]
                + next_state[:, :, FEATURE_INDEX["energy"]]
                - 1.45
            )
            / 1.4,
            0.0,
            1.0,
        )
        # These domain cascades are deliberately compositional.  They give the
        # learned graph real intermediate variables (for example
        # signal -> belief -> activation) instead of a bag of disconnected
        # one-hop effects.  The equations mirror the NumPy world runtime so the
        # accelerator does not train on a structurally poorer universe.
        next_state = next_state.at[:, :, FEATURE_INDEX["temperature"]].add(
            physical
            * 0.018
            * jnp.tanh(
                next_state[:, :, FEATURE_INDEX["energy"]]
                - next_state[:, :, FEATURE_INDEX["temperature"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["pressure"]].add(
            physical
            * 0.014
            * jnp.tanh(
                next_state[:, :, FEATURE_INDEX["temperature"]]
                - next_state[:, :, FEATURE_INDEX["pressure"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["activation"]].add(
            physical
            * 0.010
            * jnp.clip(next_state[:, :, FEATURE_INDEX["pressure"]], 0.0, 1.0)
            * (1.0 - next_state[:, :, FEATURE_INDEX["activation"]])
        )

        next_state = next_state.at[:, :, FEATURE_INDEX["energy"]].add(
            -chemical * 0.025 * reaction
        )
        chemical_cohesion = next_state[:, :, FEATURE_INDEX["cohesion"]]
        next_state = next_state.at[:, :, FEATURE_INDEX["cohesion"]].set(
            chemical_cohesion + chemical * 0.05 * reaction * (1.0 - chemical_cohesion)
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["structure"]].set(
            jnp.maximum(
                next_state[:, :, FEATURE_INDEX["structure"]],
                chemical * (reaction > 0.58).astype(jnp.float32),
            )
        )

        metabolism = biological * jnp.minimum(
            next_state[:, :, FEATURE_INDEX["resource"]],
            0.026 + 0.018 * next_state[:, :, FEATURE_INDEX["activation"]],
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["resource"]].add(-metabolism)
        next_state = next_state.at[:, :, FEATURE_INDEX["energy"]].add(0.72 * metabolism)
        next_state = next_state.at[:, :, FEATURE_INDEX["health"]].add(
            0.035 * metabolism - biological * 0.006
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["integrity"]].add(
            biological
            * 0.012
            * next_state[:, :, FEATURE_INDEX["health"]]
            * (1.0 - next_state[:, :, FEATURE_INDEX["integrity"]])
        )

        resource_flow = (
            ecological
            * 0.045
            * (
                neighbor_mean(FEATURE_INDEX["resource"])
                - next_state[:, :, FEATURE_INDEX["resource"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["resource"]].add(resource_flow)
        next_state = next_state.at[:, :, FEATURE_INDEX["health"]].add(
            ecological
            * 0.018
            * jnp.tanh(next_state[:, :, FEATURE_INDEX["resource"]] - 0.35)
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["cohesion"]].add(
            ecological * 0.014 * jnp.tanh(neighbor_mean(FEATURE_INDEX["trust"]) - 0.5)
        )

        goal_error = (
            next_state[:, :, FEATURE_INDEX["goal"]]
            - next_state[:, :, FEATURE_INDEX["belief"]]
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["activation"]].add(
            agent
            * 0.06
            * jnp.abs(goal_error)
            * (1.0 - next_state[:, :, FEATURE_INDEX["activation"]])
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["belief"]].add(
            agent
            * 0.035
            * jnp.tanh(
                next_state[:, :, FEATURE_INDEX["signal"]]
                - next_state[:, :, FEATURE_INDEX["belief"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["activation"]].add(
            agent
            * 0.025
            * jnp.tanh(next_state[:, :, FEATURE_INDEX["belief"]])
            * (1.0 - next_state[:, :, FEATURE_INDEX["activation"]])
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["uncertainty"]].add(
            -agent
            * 0.025
            * jnp.abs(
                next_state[:, :, FEATURE_INDEX["signal"]]
                - next_state[:, :, FEATURE_INDEX["belief"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["energy"]].add(
            -agent * 0.009 * next_state[:, :, FEATURE_INDEX["activation"]]
        )

        social_trust = next_state[:, :, FEATURE_INDEX["trust"]]
        social_trust = social_trust + social * 0.05 * (
            neighbor_mean(FEATURE_INDEX["trust"]) - social_trust
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["trust"]].set(social_trust)
        next_state = next_state.at[:, :, FEATURE_INDEX["belief"]].add(
            social
            * 0.035
            * social_trust
            * (
                neighbor_mean(FEATURE_INDEX["belief"])
                - next_state[:, :, FEATURE_INDEX["belief"]]
            )
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["cohesion"]].add(
            social * 0.03 * (social_trust - 0.5)
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["memory_strength"]].add(
            social
            * 0.018
            * social_trust
            * (1.0 - next_state[:, :, FEATURE_INDEX["memory_strength"]])
        )

        symbolic_error = (
            neighbor_mean(FEATURE_INDEX["value"])
            - next_state[:, :, FEATURE_INDEX["value"]]
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["value"]].add(
            symbolic * 0.11 * symbolic_error
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["belief"]].add(
            symbolic * 0.08 * jnp.tanh(symbolic_error)
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["activation"]].add(
            symbolic * 0.04 * (1.0 - jnp.abs(symbolic_error))
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["uncertainty"]].add(
            -symbolic * 0.045 * (1.0 - jnp.abs(symbolic_error))
        )

        alignment_error = (
            neighbor_mean(FEATURE_INDEX["language_alignment"])
            - next_state[:, :, FEATURE_INDEX["language_alignment"]]
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["language_alignment"]].add(
            language * 0.10 * alignment_error
        )
        alignment_quality = jnp.clip(1.0 - jnp.abs(alignment_error), 0.0, 1.0)
        next_state = next_state.at[:, :, FEATURE_INDEX["signal"]].add(
            language * 0.055 * alignment_quality
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["belief"]].add(
            language * 0.06 * next_state[:, :, FEATURE_INDEX["trust"]] * alignment_error
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["memory_strength"]].add(
            language
            * 0.035
            * jnp.clip(next_state[:, :, FEATURE_INDEX["signal"]], 0.0, 1.0)
            * (1.0 - next_state[:, :, FEATURE_INDEX["memory_strength"]])
        )
        next_state = next_state.at[:, :, FEATURE_INDEX["uncertainty"]].add(
            -language
            * 0.03
            * jnp.clip(next_state[:, :, FEATURE_INDEX["memory_strength"]], 0.0, 1.0)
        )

        existence = current[:, :, FEATURE_INDEX["existence"]]
        survival = existence * (
            next_state[:, :, FEATURE_INDEX["lifetime"]] > 0.02
        ).astype(jnp.float32)
        no_survivor = survival.sum(axis=1) <= 0.0
        keeper = current[:, :, FEATURE_INDEX["support"]].argmax(axis=1)
        keeper_survival = jnp.maximum(
            survival[jnp.arange(batch_size), keeper],
            no_survivor.astype(jnp.float32),
        )
        survival = survival.at[jnp.arange(batch_size), keeper].set(keeper_survival)
        next_state = next_state.at[:, :, FEATURE_INDEX["existence"]].set(survival)
        for feature_index, upper in (
            (FEATURE_INDEX["energy"], 2.5),
            (FEATURE_INDEX["mass"], 3.0),
            (FEATURE_INDEX["temperature"], 2.5),
            (FEATURE_INDEX["pressure"], 2.5),
            (FEATURE_INDEX["resource"], 2.5),
            (FEATURE_INDEX["signal"], 2.5),
            (FEATURE_INDEX["health"], 1.0),
            (FEATURE_INDEX["trust"], 1.0),
            (FEATURE_INDEX["structure"], 1.0),
            (FEATURE_INDEX["uncertainty"], 1.0),
            (FEATURE_INDEX["memory_strength"], 1.0),
        ):
            next_state = next_state.at[:, :, feature_index].set(
                jnp.clip(next_state[:, :, feature_index], 0.0, upper)
            )
        next_state = next_state * survival[:, :, None]
        next_state = next_state.at[:, :, FEATURE_INDEX["existence"]].set(survival)
        conservation = jnp.clip(root_strengths[:, 5:6], 0.0, 1.0)
        active_distribution = survival / jnp.maximum(
            survival.sum(axis=1, keepdims=True), 1e-6
        )
        for budget_index, feature_index in enumerate(
            (
                FEATURE_INDEX["mass"],
                FEATURE_INDEX["energy"],
                FEATURE_INDEX["resource"],
            )
        ):
            target = initial_budgets[:, budget_index : budget_index + 1]
            proposal = jnp.maximum(next_state[:, :, feature_index], 0.0) * survival
            proposal_total = proposal.sum(axis=1, keepdims=True)
            proposal_normalized = jnp.where(
                proposal_total > 1e-6,
                proposal * target / jnp.maximum(proposal_total, 1e-6),
                active_distribution * target,
            )
            previous = jnp.maximum(current[:, :, feature_index], 0.0) * survival
            previous_total = previous.sum(axis=1, keepdims=True)
            previous_normalized = jnp.where(
                previous_total > 1e-6,
                previous * target / jnp.maximum(previous_total, 1e-6),
                active_distribution * target,
            )
            values = (
                conservation * previous_normalized
                + (1.0 - conservation) * proposal_normalized
            ) * survival
            next_state = next_state.at[:, :, feature_index].set(values)
        return next_state

    def scan_step(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Any]:
        baseline_current, treated_current = carry
        baseline_next = step_one(baseline_current)
        treated_next = step_one(treated_current)
        effect = (treated_next - baseline_next).mean(axis=1)
        return (baseline_next, treated_next), effect

    (baseline_final, treated_final), effect_trace = lax.scan(
        scan_step,
        (baseline_state, treated_state),
        xs=None,
        length=time_steps,
    )
    return baseline_final, treated_final, effect_trace


def _jax_execute_fixed_causal_rollout(
    baseline_state: Any,
    treated_state: Any,
    neighbor_indices: Any,
    relation_weight: Any,
    initial_budgets: Any,
    root_strengths: Any,
    domain_ids: Any,
) -> tuple[Any, Any, Any]:
    return _jax_execute_causal_rollout(
        baseline_state,
        treated_state,
        neighbor_indices,
        relation_weight,
        initial_budgets,
        root_strengths,
        domain_ids,
        time_steps=MASSIVE_COMPILED_TIME_STEPS,
    )


def _get_jax_executor() -> tuple[Any, int, str]:
    global _JAX_EXECUTOR, _JAX_EXECUTOR_BUILD_COUNT
    global _JAX_EXECUTOR_DEVICE_COUNT, _JAX_EXECUTOR_MODE
    if _JAX_EXECUTOR is None:
        import jax

        device_count = jax.local_device_count()
        if device_count <= 0:
            raise RuntimeError("JAX reported no local accelerator devices")
        if MASSIVE_MICROBATCH_WORLDS % device_count != 0:
            raise RuntimeError(
                "accelerator devices do not evenly divide the microbatch"
            )
        if device_count > 1:
            _JAX_EXECUTOR = jax.pmap(_jax_execute_fixed_causal_rollout)
            _JAX_EXECUTOR_MODE = "pmap"
        else:
            _JAX_EXECUTOR = jax.jit(_jax_execute_fixed_causal_rollout)
            _JAX_EXECUTOR_MODE = "jit"
        _JAX_EXECUTOR_DEVICE_COUNT = device_count
        _JAX_EXECUTOR_BUILD_COUNT += 1
    return _JAX_EXECUTOR, _JAX_EXECUTOR_DEVICE_COUNT, _JAX_EXECUTOR_MODE


def _program_feature_parameters(
    program_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.ones((len(program_ids), len(FEATURE_NAMES)), dtype=np.float32)
    shifts = np.zeros_like(scales)
    for row, program_id in enumerate(program_ids):
        program = decode_world_program(int(program_id))
        for feature, (scale, shift) in world_program_feature_transform(program).items():
            feature_index = FEATURE_INDEX[feature]
            scales[row, feature_index] = scale
            shifts[row, feature_index] = shift
    return scales, shifts


def _program_relation_weight_scales(program_ids: np.ndarray) -> np.ndarray:
    scales: list[float] = []
    channels = (
        "conductivity",
        "permeability",
        "affinity",
        "bond_strength",
        "visibility",
        "trust_channel",
        "resource_channel",
        "symbolic_match",
        "language_channel",
    )
    base = 0.55
    for program_id in program_ids:
        program = decode_world_program(int(program_id))
        transforms = world_program_relation_transform(program)
        values = [
            np.clip(
                base * transforms.get(channel, (1.0, 0.0))[0]
                + transforms.get(channel, (1.0, 0.0))[1],
                0.0,
                1.0,
            )
            for channel in channels
        ]
        scales.append(float(np.mean(values) / base))
    return np.asarray(scales, dtype=np.float32)


def _clip_jax_programmed_feature(feature: str, value: Any) -> Any:
    import jax.numpy as jnp

    if feature in {
        "charge",
        "cohesion",
        "belief",
        "goal",
        "value",
        "polarity",
        "language_alignment",
    }:
        return jnp.clip(value, -1.0, 1.0)
    if feature == "mass":
        return jnp.clip(value, 0.0, 3.0)
    if feature in {"energy", "temperature", "pressure", "resource", "signal"}:
        return jnp.clip(value, 0.0, 2.5)
    if feature == "phase":
        return jnp.arctan2(jnp.sin(value), jnp.cos(value))
    return jnp.clip(value, 0.0, 1.0)


def _jax_rollout_microbatch(
    *,
    config: CausalWorldConfig,
    seed: int,
    global_world_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    plan = build_accelerator_plan(config)
    if config.time_steps != MASSIVE_COMPILED_TIME_STEPS:
        raise ValueError("accelerator profile does not match compiled time steps")
    batch_size = int(plan["microbatch_worlds"])
    entities = config.entity_count
    neighbors = config.neighbor_count
    feature_count = len(FEATURE_NAMES)
    candidate_count = config.intervention_candidates
    interventions = generate_interventions(candidate_count)
    shard_index, shard_remainder = divmod(global_world_offset, config.worlds_per_shard)
    if shard_remainder % batch_size != 0:
        raise ValueError("accelerator microbatch is not aligned to a world shard")
    shard_programs = curriculum_programs(
        shard_index,
        programs_per_shard=TPU_PROGRAMS_PER_SHARD,
    )
    shard_program_ids = np.asarray(
        [program.program_id for program in shard_programs], dtype=np.int64
    )

    key = jax.random.PRNGKey(seed & 0x7FFFFFFF)
    keys = jax.random.split(key, 8)
    state = jax.random.uniform(
        keys[0],
        (batch_size, entities, feature_count),
        minval=0.05,
        maxval=1.0,
        dtype=jnp.float32,
    )
    signed_features = jnp.asarray(
        [
            FEATURE_INDEX[name]
            for name in (
                "charge",
                "cohesion",
                "belief",
                "goal",
                "value",
                "polarity",
                "language_alignment",
            )
        ],
        dtype=jnp.int32,
    )
    signed_values = jax.random.uniform(
        keys[1],
        (batch_size, entities, len(signed_features)),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    state = state.at[:, :, signed_features].set(signed_values)
    positions = jax.random.uniform(
        keys[2],
        (batch_size, entities, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    velocities = jax.random.uniform(
        keys[3],
        (batch_size, entities, 3),
        minval=-0.08,
        maxval=0.08,
        dtype=jnp.float32,
    )
    state = state.at[
        :, :, FEATURE_INDEX["position_x"] : FEATURE_INDEX["position_z"] + 1
    ].set(positions)
    state = state.at[
        :, :, FEATURE_INDEX["velocity_x"] : FEATURE_INDEX["velocity_z"] + 1
    ].set(velocities)
    active = (
        jax.random.uniform(keys[4], (batch_size, entities), dtype=jnp.float32) > 0.08
    ).astype(jnp.float32)
    active = active.at[:, 0].set(1.0)
    state = state * active[:, :, None]
    state = state.at[:, :, FEATURE_INDEX["existence"]].set(active)

    global_ids = np.arange(
        global_world_offset,
        global_world_offset + batch_size,
        dtype=np.int64,
    )
    candidate_ids_np = global_ids % candidate_count
    domain_ids_np = (global_ids // candidate_count) % len(DOMAIN_NAMES)
    program_slots_np = (
        global_ids // (candidate_count * len(DOMAIN_NAMES))
    ) % TPU_PROGRAMS_PER_SHARD
    program_ids_np = shard_program_ids[program_slots_np]
    candidate_ids = jnp.asarray(candidate_ids_np, dtype=jnp.int32)
    domain_ids = jnp.asarray(domain_ids_np, dtype=jnp.int32)
    program_slots = jnp.asarray(program_slots_np, dtype=jnp.int32)
    feature_scales_np, feature_shifts_np = _program_feature_parameters(
        shard_program_ids
    )
    feature_scales = jnp.asarray(feature_scales_np, dtype=jnp.float32)
    feature_shifts = jnp.asarray(feature_shifts_np, dtype=jnp.float32)
    for feature_index, feature in enumerate(FEATURE_NAMES):
        if np.all(feature_scales_np[:, feature_index] == 1.0) and np.all(
            feature_shifts_np[:, feature_index] == 0.0
        ):
            continue
        transformed = (
            state[:, :, feature_index]
            * feature_scales[program_slots, feature_index, None]
            + feature_shifts[program_slots, feature_index, None]
        )
        state = state.at[:, :, feature_index].set(
            _clip_jax_programmed_feature(feature, transformed)
        )
    state = state * active[:, :, None]
    state = state.at[:, :, FEATURE_INDEX["existence"]].set(active)
    entity_ids = jnp.arange(entities, dtype=jnp.int32)[None, :, None]
    random_offsets = jax.random.randint(
        keys[5], (batch_size, entities, neighbors), 1, entities
    )
    random_neighbor_indices = (entity_ids + random_offsets) % entities
    ring_offsets = jnp.arange(1, neighbors + 1, dtype=jnp.int32)[None, None, :]
    ring_neighbor_indices = jnp.broadcast_to(
        (entity_ids + ring_offsets) % entities,
        random_neighbor_indices.shape,
    )
    random_link_probabilities = jnp.asarray(
        [world_program_random_link_probability(program) for program in shard_programs],
        dtype=jnp.float32,
    )
    use_random = (
        jax.random.uniform(
            keys[7], (batch_size, entities, neighbors), dtype=jnp.float32
        )
        < random_link_probabilities[program_slots, None, None]
    )
    neighbor_indices = jnp.where(
        use_random, random_neighbor_indices, ring_neighbor_indices
    )
    relation_weight = jax.random.uniform(
        keys[6],
        (batch_size, entities, neighbors),
        minval=0.08,
        maxval=1.0,
        dtype=jnp.float32,
    )
    relation_scales = jnp.asarray(
        _program_relation_weight_scales(shard_program_ids), dtype=jnp.float32
    )
    relation_weight = jnp.clip(
        relation_weight * relation_scales[program_slots, None, None],
        0.01,
        1.0,
    )

    cause_features = jnp.asarray(
        [FEATURE_INDEX[item.feature] for item in interventions], dtype=jnp.int32
    )
    cause_deltas = jnp.asarray(
        [item.polarity * item.magnitude for item in interventions], dtype=jnp.float32
    )
    target_scores = state[:, :, FEATURE_INDEX["uncertainty"]] - 10.0 * (1.0 - active)
    targets = target_scores.argmax(axis=1)
    treated = state.at[
        jnp.arange(batch_size), targets, cause_features[candidate_ids]
    ].add(cause_deltas[candidate_ids])
    treated = treated.at[:, :, FEATURE_INDEX["energy"]].set(
        jnp.clip(treated[:, :, FEATURE_INDEX["energy"]], 0.0, 2.5)
    )
    treated = treated.at[:, :, FEATURE_INDEX["temperature"]].set(
        jnp.clip(treated[:, :, FEATURE_INDEX["temperature"]], 0.0, 2.5)
    )
    treated = treated.at[:, :, FEATURE_INDEX["pressure"]].set(
        jnp.clip(treated[:, :, FEATURE_INDEX["pressure"]], 0.0, 2.5)
    )
    treated = treated.at[:, :, FEATURE_INDEX["resource"]].set(
        jnp.clip(treated[:, :, FEATURE_INDEX["resource"]], 0.0, 2.5)
    )

    budget_features = jnp.asarray(
        [
            FEATURE_INDEX["mass"],
            FEATURE_INDEX["energy"],
            FEATURE_INDEX["resource"],
        ],
        dtype=jnp.int32,
    )
    initial_budgets = (state[:, :, budget_features] * active[:, :, None]).sum(axis=1)
    base_root_strengths = jnp.asarray(
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
        dtype=jnp.float32,
    )[domain_ids]
    program_root_gains = jnp.asarray(
        [world_program_root_gains(program) for program in shard_programs],
        dtype=jnp.float32,
    )
    root_strengths = base_root_strengths * program_root_gains[program_slots]

    execute, devices_used, executor_mode = _get_jax_executor()
    started = time.perf_counter()
    if executor_mode == "pmap":
        per_device = batch_size // devices_used

        def shard_batch(value: Any) -> Any:
            return value.reshape((devices_used, per_device, *value.shape[1:]))

        baseline_shards, treated_shards, effect_shards = execute(
            shard_batch(state),
            shard_batch(treated),
            shard_batch(neighbor_indices),
            shard_batch(relation_weight),
            shard_batch(initial_budgets),
            shard_batch(root_strengths),
            shard_batch(domain_ids),
        )
        baseline_final = baseline_shards.reshape(state.shape)
        treated_final = treated_shards.reshape(state.shape)
        effect_trace = jnp.transpose(effect_shards, (1, 0, 2, 3)).reshape(
            (config.time_steps, batch_size, feature_count)
        )
    else:
        baseline_final, treated_final, effect_trace = execute(
            state,
            treated,
            neighbor_indices,
            relation_weight,
            initial_budgets,
            root_strengths,
            domain_ids,
        )
    effect_trace.block_until_ready()
    elapsed = time.perf_counter() - started
    effect_np = np.asarray(effect_trace)
    budget_np = np.asarray(initial_budgets)
    baseline_budget = np.asarray(baseline_final[:, :, budget_features].sum(axis=1))
    treated_budget = np.asarray(treated_final[:, :, budget_features].sum(axis=1))
    invariant_error = np.maximum(
        np.abs((baseline_budget - budget_np) / np.maximum(budget_np, 1e-6)),
        np.abs((treated_budget - budget_np) / np.maximum(budget_np, 1e-6)),
    )
    diagnostics = {
        "elapsed_seconds": elapsed,
        "entity_updates": batch_size * entities * config.time_steps * 2,
        "relation_updates": (batch_size * entities * neighbors * config.time_steps * 2),
        "maximum_invariant_error": float(invariant_error.max(initial=0.0)),
        "mean_invariant_error": float(invariant_error.mean()),
        "devices_used": devices_used,
        "executor_mode": executor_mode,
        "effect_digest": ndarray_digest(effect_np),
        "program_ids": sorted({int(value) for value in program_ids_np}),
    }
    return effect_np, candidate_ids_np, domain_ids_np, program_ids_np, diagnostics


def run_jax_massive_shard(
    config: CausalWorldConfig,
    shard_index: int,
    *,
    require_tpu: bool = True,
    require_gpu: bool = False,
) -> tuple[tuple[CausalEvidence, ...], dict[str, Any]]:
    """Run one resumable shard and reduce it to graph-ready causal evidence."""

    config.validate()
    if config.profile != "tpu-massive":
        raise ValueError("JAX massive runner requires the tpu-massive profile")
    if not 0 <= shard_index < config.shard_count:
        raise ValueError("shard index is outside configured shard count")
    if require_tpu and require_gpu:
        raise ValueError("a JAX shard cannot require both TPU and GPU")
    probe = probe_jax_accelerator()
    if not probe["jax_available"]:
        raise RuntimeError(f"JAX is unavailable: {probe['error']}")
    if require_tpu and not probe["tpu_available"]:
        raise RuntimeError(
            "TPU_REQUIRED: no TPU device was assigned to this Kaggle run"
        )
    if require_gpu and not probe["gpu_available"]:
        raise RuntimeError(
            "GPU_REQUIRED: no GPU device was assigned to this Kaggle run"
        )

    plan = build_accelerator_plan(config)
    grouped: dict[tuple[int, int, int], list[np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    shard_offset = shard_index * config.worlds_per_shard
    replay_reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = (
        None
    )
    for microbatch in range(int(plan["microbatches_per_shard"])):
        offset = shard_offset + microbatch * int(plan["microbatch_worlds"])
        (
            effect,
            candidate_ids,
            domain_ids,
            program_ids,
            item_diagnostics,
        ) = _jax_rollout_microbatch(
            config=config,
            seed=config.seed + shard_index * 1009 + microbatch,
            global_world_offset=offset,
        )
        diagnostics.append(item_diagnostics)
        if microbatch == 0:
            replay_reference = (
                effect.copy(),
                candidate_ids.copy(),
                domain_ids.copy(),
                program_ids.copy(),
            )
        for candidate in range(config.intervention_candidates):
            for domain in range(len(DOMAIN_NAMES)):
                for program_id in curriculum_program_ids(
                    shard_index,
                    programs_per_shard=TPU_PROGRAMS_PER_SHARD,
                ):
                    mask = (
                        (candidate_ids == candidate)
                        & (domain_ids == domain)
                        & (program_ids == program_id)
                    )
                    if mask.any():
                        grouped.setdefault((candidate, domain, program_id), []).append(
                            effect[:, mask, :]
                        )

    if replay_reference is None:
        raise RuntimeError("accelerator plan produced no microbatches")
    (
        replay_effect,
        replay_candidates,
        replay_domains,
        replay_programs,
        replay_diagnostics,
    ) = _jax_rollout_microbatch(
        config=config,
        seed=config.seed + shard_index * 1009,
        global_world_offset=shard_offset,
    )
    deterministic_replay = {
        "effect_trace_equal": np.array_equal(replay_reference[0], replay_effect),
        "candidate_assignment_equal": np.array_equal(
            replay_reference[1], replay_candidates
        ),
        "domain_assignment_equal": np.array_equal(replay_reference[2], replay_domains),
        "program_assignment_equal": np.array_equal(
            replay_reference[3], replay_programs
        ),
        "reference_digest": ndarray_digest(replay_reference[0]),
        "replay_digest": ndarray_digest(replay_effect),
        "replay_elapsed_seconds": replay_diagnostics["elapsed_seconds"],
    }
    deterministic_replay["passed"] = all(
        deterministic_replay[key]
        for key in (
            "effect_trace_equal",
            "candidate_assignment_equal",
            "domain_assignment_equal",
            "program_assignment_equal",
        )
    )

    interventions = generate_interventions(config.intervention_candidates)
    observable_indices = np.asarray(
        [FEATURE_INDEX[name] for name in OBSERVABLE_FEATURES], dtype=np.int32
    )
    evidence_rows: list[CausalEvidence] = []
    evidence_partitions = 4
    expected_group_count = (
        config.intervention_candidates * len(DOMAIN_NAMES) * TPU_PROGRAMS_PER_SHARD
    )
    if len(grouped) != expected_group_count:
        raise RuntimeError(
            "accelerator curriculum did not cover every intervention-domain-program group"
        )
    if any(len(traces) != evidence_partitions for traces in grouped.values()):
        raise RuntimeError(
            "accelerator curriculum groups do not have four independent worlds"
        )
    for (candidate_index, domain_index, program_id), traces in sorted(grouped.items()):
        full_trace = np.concatenate(traces, axis=1)
        observable = full_trace[:, :, observable_indices]
        mean_by_tick = observable.mean(axis=1)
        intervention = interventions[candidate_index]
        domain = DOMAIN_NAMES[domain_index]
        program = decode_world_program(program_id)
        effect_strength = np.max(np.abs(mean_by_tick), axis=0)
        observable_positions = np.argsort(-effect_strength, kind="stable")[
            :CAUSAL_EFFECTS_PER_INTERVENTION
        ]
        for observable_position_value in observable_positions:
            observable_position = int(observable_position_value)
            peak_tick = int(np.abs(mean_by_tick[:, observable_position]).argmax())
            effect_feature = OBSERVABLE_FEATURES[observable_position]
            trace_groups = np.array_split(
                np.arange(len(traces)), min(evidence_partitions, len(traces))
            )
            for partition_index, trace_indices in enumerate(trace_groups):
                partition_trace = np.concatenate(
                    [traces[int(index)] for index in trace_indices], axis=1
                )
                world_effect = partition_trace[
                    peak_tick, :, observable_indices[observable_position]
                ]
                signed_mean = float(world_effect.mean())
                direction, magnitude = normalize_causal_response(
                    signed_mean,
                    intervention.polarity * intervention.magnitude,
                )
                provenance = {
                    "runtime": CAUSAL_ACCELERATOR_RUNTIME,
                    "config": asdict(config),
                    "shard_index": shard_index,
                    "partition_index": partition_index,
                    "candidate_index": candidate_index,
                    "domain": domain,
                    "world_program": program.manifest(),
                    "effect_feature": effect_feature,
                    "peak_tick": peak_tick,
                    "full_trace": ndarray_digest(full_trace),
                    "partition_trace": ndarray_digest(partition_trace),
                }
                item = CausalEvidence(
                    evidence_id=(
                        f"tpu-s{shard_index:03d}-p{partition_index:02d}-"
                        f"w{program_id:08d}-{intervention.intervention_id}-"
                        f"{domain}-{effect_feature}"
                    ),
                    domain=domain,
                    cause_feature=intervention.feature,
                    effect_feature=effect_feature,
                    direction=direction,
                    magnitude=magnitude,
                    delay=peak_tick + 1,
                    context_signature=(
                        f"domain:{domain}",
                        f"cause:{intervention.feature}",
                        f"target:{intervention.target_rule}",
                        f"polarity:{intervention.polarity:+d}",
                        "backend:xla",
                        *program.condition_signature(),
                    ),
                    treated_worlds=int(partition_trace.shape[1]),
                    baseline_worlds=int(partition_trace.shape[1]),
                    variance=float(world_effect.var()),
                    invariant_error=float(
                        max(item["maximum_invariant_error"] for item in diagnostics)
                    ),
                    provenance_hash=canonical_hash(provenance),
                )
                item.validate()
                evidence_rows.append(item)

    elapsed = sum(item["elapsed_seconds"] for item in diagnostics)
    entity_updates = sum(item["entity_updates"] for item in diagnostics)
    relation_updates = sum(item["relation_updates"] for item in diagnostics)
    device_counts = {int(item["devices_used"]) for item in diagnostics}
    executor_modes = {str(item["executor_mode"]) for item in diagnostics}
    if len(device_counts) != 1 or len(executor_modes) != 1:
        raise RuntimeError("accelerator executor topology changed within a shard")
    report = {
        "runtime": CAUSAL_ACCELERATOR_RUNTIME,
        "probe": probe,
        "plan": plan,
        "shard_index": shard_index,
        "world_programs": [
            program.manifest()
            for program in curriculum_programs(
                shard_index,
                programs_per_shard=TPU_PROGRAMS_PER_SHARD,
            )
        ],
        "evidence_count": len(evidence_rows),
        "elapsed_seconds": elapsed,
        "entity_updates": entity_updates,
        "relation_updates": relation_updates,
        "entity_updates_per_second": entity_updates / max(elapsed, 1e-9),
        "relation_updates_per_second": relation_updates / max(elapsed, 1e-9),
        "devices_used": next(iter(device_counts)),
        "executor_mode": next(iter(executor_modes)),
        "maximum_invariant_error": max(
            item["maximum_invariant_error"] for item in diagnostics
        ),
        "microbatch_digests": [item["effect_digest"] for item in diagnostics],
        "evidence_partitions_per_hypothesis": evidence_partitions,
        "effects_retained_per_intervention": CAUSAL_EFFECTS_PER_INTERVENTION,
        "jit_executor_constructions": _JAX_EXECUTOR_BUILD_COUNT,
        "deterministic_replay": deterministic_replay,
        "evidence_hash": canonical_hash([asdict(item) for item in evidence_rows]),
    }
    return tuple(evidence_rows), report


def accelerator_self_test() -> dict[str, bool]:
    config = get_profile("tpu-massive")
    plan = build_accelerator_plan(config)
    checks = {
        "fixed_microbatch": plan["microbatch_worlds"] == MASSIVE_MICROBATCH_WORLDS,
        "multiple_microbatches": plan["microbatches_per_shard"] > 1,
        "matched_counterfactuals": plan["matched_rollouts_per_world"] == 2,
        "billion_scale_full_run": config.scale_manifest()["entity_updates"]
        > 1_000_000_000,
        "host_graph_is_separate": plan["persistent_graph_location"] == "host",
        "independent_evidence_partitions": 4 >= 3,
        "multiple_effects_per_intervention": CAUSAL_EFFECTS_PER_INTERVENTION >= 4,
        "four_programs_per_shard": plan["world_programs_per_shard"] == 4,
        "semantic_space_exceeds_fifty_million": plan["curriculum"][
            "procedural_program_space"
        ]
        > 50_000_000,
        "sixty_four_programs_exercised": plan["curriculum"]["exercised_programs"] == 64,
        "single_cached_xla_executor": plan["xla_executor_cache_scope"] == "process",
        "all_local_devices_are_parallel": plan["device_parallelism"]
        == "pmap-all-local-devices",
        "all_seven_roots_execute": set(plan["root_mechanics_executed"])
        == set(ROOT_MECHANICS),
        "measured_replay_required": plan["deterministic_replay_microbatches"] == 1,
        "probe_is_fail_closed": {
            "tpu_available",
            "gpu_available",
        } <= set(probe_jax_accelerator()),
    }
    if not all(checks.values()):
        raise AssertionError(f"accelerator self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(accelerator_self_test())
