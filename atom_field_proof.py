"""Tiny from-scratch proof for a universe-first recurrent neural field.

The executable simulator and the neural architecture are intentionally small.
No pretrained model, tokenizer, downloaded weights, or LoRA code is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = 4
SEED = 20260721
NODE_COUNT = 6
NODE_FEATURE_NAMES = (
    "signal",
    "mass",
    "charge",
    "support",
    "ttl",
    "active",
    "structure",
    "cohesion",
)
GLOBAL_FEATURE_NAMES = (
    "radiation_rate",
    "dissipation_rate",
    "gravitation_rate",
    "attraction_repulsion_rate",
    "nucleation_rate",
    "nucleation_threshold",
    "decay_rate",
    "closed_system",
)
PROCESS_NAMES = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
)
CONTINUOUS_TARGET_NAMES = ("signal", "mass", "cohesion", "ttl")
BINARY_TARGET_NAMES = ("active", "structure")

NODE_DIM = len(NODE_FEATURE_NAMES)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
PROCESS_COUNT = len(PROCESS_NAMES)
CONTINUOUS_DIM = len(CONTINUOUS_TARGET_NAMES)
BINARY_DIM = len(BINARY_TARGET_NAMES)
FIELD_TICKS = 4

TRAIN_FAMILIES = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
    "signal_threshold",
    "cluster_bind",
    "forget_preserve",
)
HELDOUT_FAMILIES = (
    "broadcast_bind_crystallize",
    "aggregate_preserve_expire",
    "full_cycle",
)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_process_target(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0:
        raise ValueError("Every case must activate at least one process.")
    return [value / total for value in clipped]


def _row_normalize(adjacency: Tensor) -> Tensor:
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def simulate_field(
    node_features: Tensor,
    adjacency: Tensor,
    global_features: Tensor,
    ticks: int = FIELD_TICKS,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Apply all enabled universe operators from a shared prior state per tick."""

    state = node_features.clone().to(dtype=torch.float32)
    adjacency = adjacency.clone().to(dtype=torch.float32)
    controls = global_features.clone().to(dtype=torch.float32)
    initial_budget = float((state[:, 1] * state[:, 5]).sum().item())
    transition_log: list[dict[str, Any]] = []

    for tick in range(ticks):
        signal = state[:, 0]
        mass = state[:, 1].clamp_min(0.0)
        charge = state[:, 2].clamp(-1.0, 1.0)
        support = state[:, 3].clamp(0.0, 1.0)
        ttl = state[:, 4]
        active = state[:, 5].clamp(0.0, 1.0)
        structure = state[:, 6].clamp(0.0, 1.0)
        cohesion = state[:, 7].clamp(-1.0, 1.0)

        radiation_rate = controls[0].clamp(0.0, 1.0)
        dissipation_rate = controls[1].clamp(0.0, 1.0)
        gravitation_rate = controls[2].clamp(0.0, 1.0)
        attraction_rate = controls[3].clamp(0.0, 1.0)
        nucleation_rate = controls[4].clamp(0.0, 1.0)
        threshold = controls[5].clamp(0.1, 3.0)
        decay_rate = controls[6].clamp(0.0, 1.0)
        closed_system = controls[7].clamp(0.0, 1.0)

        # All proposals below read the same prior state.
        normalized_edges = _row_normalize(adjacency)
        incoming_signal = normalized_edges.transpose(0, 1) @ (signal * active)
        radiation_delta = radiation_rate * incoming_signal

        attenuation = 1.0 - dissipation_rate * (1.0 - support)
        signal_proposal = (signal + radiation_delta) * attenuation.clamp(0.0, 1.0)

        attractor_logits = (2.0 * mass + support).masked_fill(active < 0.5, -1e4)
        attractor_weights = torch.softmax(attractor_logits, dim=0)
        live_budget = (mass * active).sum()
        gravitational_mass = (
            (1.0 - gravitation_rate) * mass
            + gravitation_rate * live_budget * attractor_weights
        )

        signed_relation = -torch.outer(charge, charge) * adjacency
        relation_degree = adjacency.transpose(0, 1).sum(dim=-1).clamp_min(1.0)
        relation_field = (
            signed_relation.transpose(0, 1) @ active
        ) / relation_degree
        cohesion_proposal = torch.tanh(
            cohesion + attraction_rate * relation_field
        )

        mean_mass = gravitational_mass[active > 0.5].mean() if bool((active > 0.5).any()) else torch.tensor(1.0)
        normalized_mass = gravitational_mass / mean_mass.clamp_min(1e-6)
        potential = (
            0.55 * signal_proposal.relu()
            + 0.25 * normalized_mass
            + 0.20 * cohesion_proposal.relu()
        )
        nucleated = (nucleation_rate * potential >= threshold).to(torch.float32)
        structure_proposal = torch.maximum(structure, nucleated)

        ttl_proposal = ttl - decay_rate * (1.0 - 0.25 * support)
        expired = (
            (ttl_proposal <= 0.0)
            & (structure_proposal < 0.5)
            & (support < 0.6)
        )
        active_proposal = active * (~expired).to(torch.float32)

        mass_proposal = gravitational_mass * active_proposal
        if float(closed_system.item()) >= 0.5 and bool((active_proposal > 0.5).any()):
            surviving = mass_proposal.sum()
            if float(surviving.item()) <= 1e-8:
                keeper = int(torch.argmax(support).item())
                active_proposal[keeper] = 1.0
                mass_proposal[keeper] = initial_budget
            else:
                mass_proposal = mass_proposal * (initial_budget / surviving)

        signal_next = signal_proposal.clamp(-2.0, 2.0) * active_proposal
        # Conservation is an invariant, not a soft preference.  An upper clamp
        # here would destroy budget whenever gravitation concentrates more than
        # the cap on one node, so only the physically invalid negative side is
        # projected away.
        mass_next = mass_proposal.clamp_min(0.0)
        cohesion_next = cohesion_proposal * active_proposal
        ttl_next = ttl_proposal.clamp(-1.0, 1.0)
        structure_next = structure_proposal * active_proposal

        state = torch.stack(
            (
                signal_next,
                mass_next,
                charge,
                support,
                ttl_next,
                active_proposal,
                structure_next,
                cohesion_next,
            ),
            dim=-1,
        )
        transition_log.append(
            {
                "tick": tick + 1,
                "active_count": int((active_proposal > 0.5).sum().item()),
                "structure_count": int((structure_next > 0.5).sum().item()),
                "mass_total": round(float(mass_next.sum().item()), 6),
                "signal_total": round(float(signal_next.sum().item()), 6),
            }
        )

    continuous = state[:, [0, 1, 7, 4]]
    binary = state[:, [5, 6]]
    diagnostics = {
        "initial_mass_budget": round(initial_budget, 6),
        "final_mass_total": round(float(continuous[:, 1].sum().item()), 6),
        "transitions": transition_log,
    }
    return continuous, binary, diagnostics


@dataclass(frozen=True)
class FamilySpec:
    rates: tuple[float, float, float, float, float, float, float, float]
    signature: tuple[str, ...]


FAMILY_SPECS: dict[str, FamilySpec] = {
    "radiation": FamilySpec((0.65, 0.0, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0), ("radiation",)),
    "dissipation": FamilySpec((0.0, 0.60, 0.0, 0.0, 0.0, 2.4, 0.0, 0.0), ("dissipation",)),
    "gravitation": FamilySpec((0.0, 0.0, 0.55, 0.0, 0.0, 2.4, 0.0, 0.0), ("gravitation",)),
    "attraction_repulsion": FamilySpec((0.0, 0.0, 0.0, 0.65, 0.0, 2.4, 0.0, 0.0), ("attraction_repulsion",)),
    "nucleation": FamilySpec((0.0, 0.0, 0.0, 0.0, 0.90, 0.62, 0.0, 0.0), ("nucleation",)),
    "conservation": FamilySpec((0.0, 0.0, 0.25, 0.0, 0.0, 2.4, 0.0, 1.0), ("conservation",)),
    "decay": FamilySpec((0.0, 0.0, 0.0, 0.0, 0.0, 2.4, 0.55, 0.0), ("decay",)),
    "signal_threshold": FamilySpec((0.55, 0.10, 0.0, 0.0, 0.85, 0.70, 0.0, 0.0), ("radiation", "dissipation", "nucleation")),
    "cluster_bind": FamilySpec((0.0, 0.0, 0.50, 0.60, 0.35, 0.82, 0.0, 0.0), ("gravitation", "attraction_repulsion", "nucleation")),
    "forget_preserve": FamilySpec((0.0, 0.45, 0.25, 0.0, 0.0, 2.4, 0.50, 1.0), ("dissipation", "gravitation", "conservation", "decay")),
    "broadcast_bind_crystallize": FamilySpec((0.55, 0.10, 0.0, 0.55, 0.85, 0.72, 0.0, 0.0), ("radiation", "dissipation", "attraction_repulsion", "nucleation")),
    "aggregate_preserve_expire": FamilySpec((0.0, 0.15, 0.50, 0.25, 0.0, 2.4, 0.45, 1.0), ("dissipation", "gravitation", "attraction_repulsion", "conservation", "decay")),
    "full_cycle": FamilySpec((0.45, 0.30, 0.40, 0.45, 0.75, 0.78, 0.35, 1.0), PROCESS_NAMES),
}


def _jitter(value: float, rng: random.Random, amount: float = 0.06) -> float:
    if value in (0.0, 1.0):
        return value
    return max(0.0, min(1.0, value + rng.uniform(-amount, amount)))


def build_case(family: str, case_seed: int, split: str, index: int) -> dict[str, Any]:
    if family not in FAMILY_SPECS:
        raise KeyError(f"Unknown family: {family}")
    rng = random.Random(case_seed)
    spec = FAMILY_SPECS[family]

    adjacency = torch.zeros(NODE_COUNT, NODE_COUNT, dtype=torch.float32)
    for source in range(NODE_COUNT - 1):
        adjacency[source, source + 1] = 1.0
    for source in range(NODE_COUNT):
        for target in range(NODE_COUNT):
            if source != target and rng.random() < 0.18:
                adjacency[source, target] = 1.0

    signal = torch.tensor([rng.uniform(0.0, 0.18) for _ in range(NODE_COUNT)])
    mass = torch.tensor([rng.uniform(0.25, 1.0) for _ in range(NODE_COUNT)])
    charge = torch.tensor([rng.choice((-1.0, 1.0)) for _ in range(NODE_COUNT)])
    support = torch.tensor([rng.uniform(0.15, 0.95) for _ in range(NODE_COUNT)])
    ttl = torch.tensor([rng.uniform(0.35, 1.0) for _ in range(NODE_COUNT)])
    active = torch.ones(NODE_COUNT)
    structure = torch.zeros(NODE_COUNT)
    cohesion = torch.zeros(NODE_COUNT)

    if "radiation" in spec.signature:
        signal.zero_()
        signal[rng.randrange(0, 2)] = rng.uniform(0.8, 1.4)
    if "dissipation" in spec.signature:
        for node in rng.sample(range(NODE_COUNT), k=2):
            support[node] = rng.uniform(0.0, 0.18)
            signal[node] = max(float(signal[node]), rng.uniform(0.5, 1.0))
    if "gravitation" in spec.signature:
        heavy = rng.randrange(NODE_COUNT)
        mass[heavy] = rng.uniform(1.2, 1.8)
    if "attraction_repulsion" in spec.signature:
        charge = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        if rng.random() < 0.5:
            charge = charge.roll(rng.randrange(NODE_COUNT))
    if "nucleation" in spec.signature:
        candidate = rng.randrange(NODE_COUNT)
        signal[candidate] = max(float(signal[candidate]), rng.uniform(0.75, 1.35))
        mass[candidate] = max(float(mass[candidate]), rng.uniform(0.9, 1.5))
    if "decay" in spec.signature:
        for node in rng.sample(range(NODE_COUNT), k=2):
            ttl[node] = rng.uniform(0.02, 0.20)
            support[node] = rng.uniform(0.0, 0.30)
    if family == "conservation":
        mass[rng.randrange(NODE_COUNT)] = rng.uniform(1.4, 1.9)

    controls = list(spec.rates)
    for position in (0, 1, 2, 3, 4, 6):
        controls[position] = _jitter(controls[position], rng)
    if controls[5] < 2.0:
        controls[5] = max(0.45, controls[5] + rng.uniform(-0.08, 0.08))
    global_features = torch.tensor(controls, dtype=torch.float32)
    nodes = torch.stack(
        (signal, mass, charge, support, ttl, active, structure, cohesion),
        dim=-1,
    )

    target_continuous, target_binary, diagnostics = simulate_field(
        nodes, adjacency, global_features
    )
    process_values = [
        controls[0],
        controls[1],
        controls[2],
        controls[3],
        controls[4],
        controls[7],
        controls[6],
    ]
    process_target = normalize_process_target(process_values)
    transition_signature = {
        "family": family,
        "enabled": list(spec.signature),
        "edge_count": int(adjacency.sum().item()),
        "closed": int(controls[7] >= 0.5),
        "ticks": FIELD_TICKS,
    }
    return {
        "id": f"field-{split}-{family}-{index:03d}",
        "split": split,
        "family": family,
        "seed": case_seed,
        "node_features": nodes.tolist(),
        "adjacency": adjacency.tolist(),
        "global_features": global_features.tolist(),
        "process_target": process_target,
        "target_continuous": target_continuous.tolist(),
        "target_binary": target_binary.tolist(),
        "transition_signature": transition_signature,
        "transition_hash": stable_hash(transition_signature),
        "diagnostics": diagnostics,
    }


def generate_splits() -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "heldout": []}
    for family_index, family in enumerate(TRAIN_FAMILIES):
        for index in range(14):
            splits["train"].append(build_case(family, SEED + family_index * 1000 + index, "train", index))
        for index in range(4):
            splits["validation"].append(build_case(family, SEED + 50_000 + family_index * 1000 + index, "validation", index))
    for family_index, family in enumerate(HELDOUT_FAMILIES):
        for index in range(12):
            splits["heldout"].append(build_case(family, SEED + 90_000 + family_index * 1000 + index, "heldout", index))
    return splits


def audit_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    expected = {"train": 140, "validation": 40, "heldout": 36}
    failures: list[str] = []
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts != expected:
        failures.append(f"split counts {counts} != {expected}")

    ids: set[str] = set()
    for split, rows in splits.items():
        for row in rows:
            if row["id"] in ids:
                failures.append(f"duplicate id {row['id']}")
            ids.add(str(row["id"]))
            target = row["process_target"]
            if len(target) != PROCESS_COUNT or abs(sum(target) - 1.0) > 1e-6:
                failures.append(f"bad process target {row['id']}")
            if len(row["node_features"]) != NODE_COUNT:
                failures.append(f"bad node count {row['id']}")
            if split == "heldout" and row["family"] not in HELDOUT_FAMILIES:
                failures.append(f"heldout family leak {row['id']}")

    train_families = {row["family"] for row in splits["train"]}
    heldout_families = {row["family"] for row in splits["heldout"]}
    overlap = sorted(train_families & heldout_families)
    if overlap:
        failures.append(f"heldout family overlap: {overlap}")

    hashes = {
        split: stable_hash(rows)
        for split, rows in splits.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "counts": counts,
        "families": {
            split: dict(sorted(Counter(row["family"] for row in rows).items()))
            for split, rows in splits.items()
        },
        "split_hashes": hashes,
        "heldout_family_overlap": overlap,
        "failures": failures,
        "passed": not failures,
    }


class FieldCaseDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "id": str(row["id"]),
            "family": str(row["family"]),
            "node_features": torch.tensor(row["node_features"], dtype=torch.float32),
            "adjacency": torch.tensor(row["adjacency"], dtype=torch.float32),
            "global_features": torch.tensor(row["global_features"], dtype=torch.float32),
            "process_target": torch.tensor(row["process_target"], dtype=torch.float32),
            "target_continuous": torch.tensor(row["target_continuous"], dtype=torch.float32),
            "target_binary": torch.tensor(row["target_binary"], dtype=torch.float32),
        }


def make_loader(
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FieldCaseDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def _batch_row_normalize(adjacency: Tensor) -> Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)


def project_outputs(
    raw_continuous: Tensor,
    binary_logits: Tensor,
    node_features: Tensor,
    global_features: Tensor,
) -> Tensor:
    signal = 2.0 * torch.tanh(raw_continuous[..., 0])
    mass = F.softplus(raw_continuous[..., 1])
    cohesion = torch.tanh(raw_continuous[..., 2])
    ttl = torch.tanh(raw_continuous[..., 3])

    active_probability = torch.sigmoid(binary_logits[..., 0])
    live_mass = mass * active_probability
    initial_budget = (
        node_features[..., 1] * node_features[..., 5]
    ).sum(dim=1, keepdim=True)
    normalized_mass = live_mass * (
        initial_budget / live_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    )
    closed = global_features[:, 7].view(-1, 1) >= 0.5
    projected_mass = torch.where(closed, normalized_mass, live_mass)
    return torch.stack((signal, projected_mass, cohesion, ttl), dim=-1)


class AtomFieldCell(nn.Module):
    """A simultaneous, state-native update over the seven root operators.

    The operators do not compete for a unit attention budget. Each receives its
    own non-negative control and acts on the same prior field. The few learned
    values calibrate rates and thresholds; the topology of each process is part
    of the architecture itself.
    """

    def __init__(self) -> None:
        super().__init__()
        # The topology is supplied, but the numerical laws begin deliberately
        # miscalibrated so the tiny dataset must do real work.
        self.log_rate_gain = nn.Parameter(
            torch.log(torch.tensor((0.72, 1.20, 0.78, 1.18, 0.70, 0.82, 1.22)))
        )
        self.log_gravitation_mix = nn.Parameter(torch.log(torch.tensor((1.40, 1.50))))
        self.potential_logits = nn.Parameter(torch.log(torch.tensor((0.35, 0.40, 0.25))))
        self.nucleation_bias = nn.Parameter(torch.tensor(0.08))
        self.log_nucleation_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))
        self.decay_support_logit = nn.Parameter(torch.logit(torch.tensor(0.45)))
        self.log_decay_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))

    @staticmethod
    def _straight_through_binary(margin: Tensor, temperature: Tensor) -> Tensor:
        probability = torch.sigmoid(temperature * margin)
        hard = (margin >= 0.0).to(probability.dtype)
        return hard + probability - probability.detach()

    def forward(
        self,
        state: Tensor,
        adjacency: Tensor,
        global_features: Tensor,
        initial_budget: Tensor,
        ablate: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, node_count, _ = state.shape
        signal = state[..., 0]
        mass = state[..., 1].clamp_min(0.0)
        charge = state[..., 2].clamp(-1.0, 1.0)
        support = state[..., 3].clamp(0.0, 1.0)
        ttl = state[..., 4]
        active = state[..., 5].clamp(0.0, 1.0)
        structure = state[..., 6].clamp(0.0, 1.0)
        cohesion = state[..., 7].clamp(-1.0, 1.0)

        raw_rates = torch.stack(
            (
                global_features[:, 0],
                global_features[:, 1],
                global_features[:, 2],
                global_features[:, 3],
                global_features[:, 4],
                global_features[:, 7],
                global_features[:, 6],
            ),
            dim=-1,
        ).clamp_min(0.0)
        rates = (raw_rates * self.log_rate_gain.exp().unsqueeze(0)).clamp(0.0, 1.0)
        if ablate is not None:
            if not 0 <= ablate < PROCESS_COUNT:
                raise ValueError(f"Bad ablation index: {ablate}")
            rates = rates.clone()
            rates[:, ablate] = 0.0

        radiation_rate = rates[:, 0].unsqueeze(1)
        dissipation_rate = rates[:, 1].unsqueeze(1)
        gravitation_rate = rates[:, 2].unsqueeze(1)
        attraction_rate = rates[:, 3].unsqueeze(1)
        nucleation_rate = rates[:, 4].unsqueeze(1)
        conservation_rate = rates[:, 5].unsqueeze(1)
        decay_rate = rates[:, 6].unsqueeze(1)

        # Every proposal reads the same state supplied to this tick.
        normalized_edges = _batch_row_normalize(adjacency)
        incoming_signal = torch.bmm(
            normalized_edges.transpose(1, 2),
            (signal * active).unsqueeze(-1),
        ).squeeze(-1)
        attenuation = (
            1.0 - dissipation_rate * (1.0 - support)
        ).clamp(0.0, 1.0)
        signal_proposal = (signal + radiation_rate * incoming_signal) * attenuation

        gravitation_mix = self.log_gravitation_mix.exp()
        attractor_logits = (
            gravitation_mix[0] * mass + gravitation_mix[1] * support
        ).masked_fill(active < 0.5, -1e4)
        attractor_weights = torch.softmax(attractor_logits, dim=1)
        live_budget = (mass * active).sum(dim=1, keepdim=True)
        gravitational_mass = (
            (1.0 - gravitation_rate) * mass
            + gravitation_rate * live_budget * attractor_weights
        )

        signed_relation = -charge.unsqueeze(2) * charge.unsqueeze(1) * adjacency
        relation_degree = (
            adjacency.transpose(1, 2).sum(dim=-1).clamp_min(1.0)
        )
        relation_field = torch.bmm(
            signed_relation.transpose(1, 2), active.unsqueeze(-1)
        ).squeeze(-1) / relation_degree
        cohesion_proposal = torch.tanh(
            cohesion + attraction_rate * relation_field
        )

        live_mask = (active > 0.5).to(mass.dtype)
        mean_mass = (gravitational_mass * live_mask).sum(
            dim=1, keepdim=True
        ) / live_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized_mass = gravitational_mass / mean_mass.clamp_min(1e-6)
        potential_weights = torch.softmax(self.potential_logits, dim=0)
        potential = (
            potential_weights[0] * signal_proposal.relu()
            + potential_weights[1] * normalized_mass
            + potential_weights[2] * cohesion_proposal.relu()
        )
        threshold = global_features[:, 5].unsqueeze(1).clamp(0.1, 3.0)
        nucleation_margin = (
            nucleation_rate * potential - threshold - self.nucleation_bias
        )
        nucleated = self._straight_through_binary(
            nucleation_margin, self.log_nucleation_temperature.exp()
        )
        structure_proposal = torch.maximum(structure, nucleated)

        decay_support_fraction = torch.sigmoid(self.decay_support_logit)
        ttl_proposal = ttl - decay_rate * (
            1.0 - decay_support_fraction * support
        )
        expiry_soft = (
            torch.sigmoid(self.log_decay_temperature.exp() * -ttl_proposal)
            * torch.sigmoid(self.log_decay_temperature.exp() * (0.5 - structure_proposal))
            * torch.sigmoid(self.log_decay_temperature.exp() * (0.6 - support))
        )
        expiry_hard = (
            (ttl_proposal <= 0.0)
            & (structure_proposal < 0.5)
            & (support < 0.6)
        ).to(state.dtype)
        expired = expiry_hard + expiry_soft - expiry_soft.detach()
        active_proposal = (active * (1.0 - expired)).clamp(0.0, 1.0)

        mass_proposal = gravitational_mass * active_proposal
        surviving = mass_proposal.sum(dim=1, keepdim=True)
        closed = conservation_rate >= 0.5
        no_survivor = closed & (surviving <= 1e-8)
        keeper = F.one_hot(
            support.argmax(dim=1), num_classes=node_count
        ).to(state.dtype)
        active_proposal = torch.where(no_survivor, keeper, active_proposal)
        mass_proposal = torch.where(
            no_survivor, keeper * initial_budget, mass_proposal
        )
        surviving = mass_proposal.sum(dim=1, keepdim=True)
        conserved_mass = mass_proposal * (
            initial_budget / surviving.clamp_min(1e-8)
        )
        mass_next = torch.where(closed, conserved_mass, mass_proposal).clamp_min(0.0)

        signal_next = signal_proposal.clamp(-2.0, 2.0) * active_proposal
        cohesion_next = cohesion_proposal * active_proposal
        ttl_next = ttl_proposal.clamp(-1.0, 1.0)
        structure_next = structure_proposal * active_proposal
        next_state = torch.stack(
            (
                signal_next,
                mass_next,
                charge,
                support,
                ttl_next,
                active_proposal,
                structure_next,
                cohesion_next,
            ),
            dim=-1,
        )

        # Reporting/supervision is normalized, while the physical operators above
        # remain independent and simultaneous.
        route_total = rates.sum(dim=-1, keepdim=True)
        normalized_route = torch.where(
            route_total > 1e-8,
            rates / route_total.clamp_min(1e-8),
            torch.zeros_like(rates),
        )
        return next_state, normalized_route.unsqueeze(1).expand(
            batch_size, node_count, PROCESS_COUNT
        )


class AtomFieldNet(nn.Module):
    def __init__(self, ticks: int = FIELD_TICKS) -> None:
        super().__init__()
        self.ticks = ticks
        self.cell = AtomFieldCell()

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        global_features: Tensor,
        ablate: int | None = None,
    ) -> dict[str, Tensor]:
        state = node_features
        initial_budget = (
            node_features[..., 1] * node_features[..., 5]
        ).sum(dim=1, keepdim=True)
        routes: list[Tensor] = []
        for _ in range(self.ticks):
            state, route = self.cell(
                state,
                adjacency,
                global_features,
                initial_budget,
                ablate=ablate,
            )
            routes.append(route)
        continuous = state[..., [0, 1, 7, 4]]
        binary_probability = state[..., [5, 6]].clamp(0.0, 1.0)
        binary_logits = 12.0 * (binary_probability - 0.5)
        route_tensor = torch.stack(routes, dim=1)
        return {
            "continuous": continuous,
            "binary_logits": binary_logits,
            "route_mean": route_tensor.mean(dim=(1, 2)),
            "route_by_tick": route_tensor.mean(dim=2),
        }


class FlatFieldBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = NODE_COUNT * NODE_DIM + NODE_COUNT * NODE_COUNT + GLOBAL_DIM
        output_dim = NODE_COUNT * (CONTINUOUS_DIM + BINARY_DIM)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        global_features: Tensor,
        ablate: int | None = None,
    ) -> dict[str, Tensor]:
        if ablate is not None:
            raise ValueError("The flat baseline has no operator branches to ablate.")
        batch_size = node_features.shape[0]
        flat = torch.cat(
            (
                node_features.reshape(batch_size, -1),
                adjacency.reshape(batch_size, -1),
                global_features,
            ),
            dim=-1,
        )
        raw = self.network(flat).reshape(
            batch_size, NODE_COUNT, CONTINUOUS_DIM + BINARY_DIM
        )
        raw_continuous = raw[..., :CONTINUOUS_DIM]
        binary_logits = raw[..., CONTINUOUS_DIM:]
        continuous = project_outputs(
            raw_continuous, binary_logits, node_features, global_features
        )
        return {
            "continuous": continuous,
            "binary_logits": binary_logits,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved


def compute_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    route_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    continuous_target = batch["target_continuous"]
    binary_target = batch["target_binary"]
    dimension_weights = torch.tensor(
        (1.0, 1.0, 0.75, 0.50),
        device=continuous_target.device,
    ).view(1, 1, -1)
    continuous_loss = (
        (outputs["continuous"] - continuous_target).square() * dimension_weights
    ).mean()
    binary_loss = F.binary_cross_entropy_with_logits(
        outputs["binary_logits"], binary_target
    )
    total = continuous_loss + 0.75 * binary_loss
    route_loss = torch.zeros((), device=total.device)
    if "route_mean" in outputs:
        route = outputs["route_mean"].clamp_min(1e-8)
        route_target = batch["process_target"]
        route_loss = F.kl_div(route.log(), route_target, reduction="batchmean")
        total = total + route_weight * route_loss

    closed = batch["global_features"][:, 7] >= 0.5
    conservation_loss = torch.zeros((), device=total.device)
    if bool(closed.any()):
        predicted_total = outputs["continuous"][closed, :, 1].sum(dim=1)
        initial_total = (
            batch["node_features"][closed, :, 1]
            * batch["node_features"][closed, :, 5]
        ).sum(dim=1)
        conservation_loss = (predicted_total - initial_total).abs().mean()
        total = total + 0.05 * conservation_loss
    return total, {
        "continuous_loss": float(continuous_loss.detach().item()),
        "binary_loss": float(binary_loss.detach().item()),
        "route_loss": float(route_loss.detach().item()),
        "conservation_loss": float(conservation_loss.detach().item()),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = (
        "continuous_mae",
        "signal_mae",
        "mass_mae",
        "cohesion_mae",
        "ttl_mae",
        "active_accuracy",
        "structure_accuracy",
        "exact_state",
        "conservation_error",
        "route_l1",
    )
    return {
        key: _mean([float(row[key]) for row in rows if key in row])
        for key in keys
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int = 64,
    ablate: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = make_loader(rows, batch_size, shuffle=False, seed=SEED)
    metric_rows: list[dict[str, Any]] = []
    route_sums = torch.zeros(PROCESS_COUNT, dtype=torch.float64)
    route_count = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        outputs = model(
            batch["node_features"],
            batch["adjacency"],
            batch["global_features"],
            ablate=ablate,
        )
        absolute = (outputs["continuous"] - batch["target_continuous"]).abs()
        binary_prediction = (torch.sigmoid(outputs["binary_logits"]) >= 0.5).to(torch.float32)
        binary_correct = binary_prediction == batch["target_binary"]
        continuous_close = absolute.amax(dim=(1, 2)) <= 0.20
        binary_exact = binary_correct.all(dim=(1, 2))
        closed = batch["global_features"][:, 7] >= 0.5
        predicted_mass = outputs["continuous"][..., 1].sum(dim=1)
        initial_mass = (
            batch["node_features"][..., 1] * batch["node_features"][..., 5]
        ).sum(dim=1)
        conservation = torch.where(
            closed,
            (predicted_mass - initial_mass).abs(),
            torch.zeros_like(predicted_mass),
        )

        route_l1 = torch.zeros(absolute.shape[0], device=device)
        if "route_mean" in outputs:
            route_l1 = (
                outputs["route_mean"] - batch["process_target"]
            ).abs().mean(dim=1)
            route_sums += outputs["route_mean"].detach().double().sum(dim=0).cpu()
            route_count += int(outputs["route_mean"].shape[0])

        for index in range(absolute.shape[0]):
            metric_rows.append(
                {
                    "id": raw_batch["id"][index],
                    "family": raw_batch["family"][index],
                    "continuous_mae": float(absolute[index].mean().item()),
                    "signal_mae": float(absolute[index, :, 0].mean().item()),
                    "mass_mae": float(absolute[index, :, 1].mean().item()),
                    "cohesion_mae": float(absolute[index, :, 2].mean().item()),
                    "ttl_mae": float(absolute[index, :, 3].mean().item()),
                    "active_accuracy": float(binary_correct[index, :, 0].float().mean().item()),
                    "structure_accuracy": float(binary_correct[index, :, 1].float().mean().item()),
                    "exact_state": float((continuous_close[index] & binary_exact[index]).item()),
                    "conservation_error": float(conservation[index].item()) if bool(closed[index]) else 0.0,
                    "route_l1": float(route_l1[index].item()),
                }
            )

    by_family: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for row in metric_rows:
        by_family[str(row["family"])].append(row)
    route_mean = (
        (route_sums / route_count).tolist()
        if route_count
        else [0.0] * PROCESS_COUNT
    )
    return {
        "aggregate": aggregate_metric_rows(metric_rows),
        "per_family": {
            family: aggregate_metric_rows(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
        "route_mean": {
            name: float(route_mean[index])
            for index, name in enumerate(PROCESS_NAMES)
        },
        "examples": metric_rows,
        "ablation": PROCESS_NAMES[ablate] if ablate is not None else None,
    }


def validation_objective(metrics: Mapping[str, float]) -> float:
    return (
        float(metrics["continuous_mae"])
        + 0.35 * (1.0 - float(metrics["active_accuracy"]))
        + 0.35 * (1.0 - float(metrics["structure_accuracy"]))
        + 0.15 * float(metrics["route_l1"])
    )


def train_model(
    model: nn.Module,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    route_weight: float,
    label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    model.to(device)
    loader = make_loader(train_rows, batch_size=28, shuffle=True, seed=SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )
    best_state = copy.deepcopy(model.state_dict())
    best_objective = math.inf
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale_checks = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["node_features"],
                batch["adjacency"],
                batch["global_features"],
            )
            loss, _ = compute_loss(outputs, batch, route_weight=route_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{label} produced non-finite loss at epoch {epoch}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            validation = evaluate_model(model, validation_rows, device)
            objective = validation_objective(validation["aggregate"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": _mean(losses),
                    "validation_objective": objective,
                    "validation_continuous_mae": float(validation["aggregate"]["continuous_mae"]),
                    "validation_active_accuracy": float(validation["aggregate"]["active_accuracy"]),
                    "validation_structure_accuracy": float(validation["aggregate"]["structure_accuracy"]),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            if objective < best_objective - 1e-5:
                best_objective = objective
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 10 and epoch >= 100:
                break

    model.load_state_dict(best_state, strict=True)
    runtime = time.perf_counter() - started
    return model, {
        "label": label,
        "configured_epochs": epochs,
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "runtime_seconds": runtime,
        "history": history,
        "trainable_parameters": parameter_count(model),
    }


def ablation_report(
    model: AtomFieldNet,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    full_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    full = full_metrics["aggregate"]
    report: dict[str, Any] = {}
    for process_index, process_name in enumerate(PROCESS_NAMES):
        ablated = evaluate_model(
            model, rows, device, ablate=process_index
        )
        aggregate = ablated["aggregate"]
        report[process_name] = {
            "continuous_mae": aggregate["continuous_mae"],
            "continuous_mae_delta": aggregate["continuous_mae"] - full["continuous_mae"],
            "exact_state": aggregate["exact_state"],
            "exact_state_delta": aggregate["exact_state"] - full["exact_state"],
            "active_accuracy_delta": aggregate["active_accuracy"] - full["active_accuracy"],
            "structure_accuracy_delta": aggregate["structure_accuracy"] - full["structure_accuracy"],
        }
    return report


def run_self_tests() -> dict[str, Any]:
    set_deterministic(SEED)
    splits_a = generate_splits()
    splits_b = generate_splits()
    audit = audit_splits(splits_a)
    checks: dict[str, bool] = {
        "dataset_audit": bool(audit["passed"]),
        "dataset_deterministic": stable_hash(splits_a) == stable_hash(splits_b),
        "heldout_compositions_disjoint": not bool(audit["heldout_family_overlap"]),
    }

    conservation_case = next(
        row for row in splits_a["train"] if row["family"] == "conservation"
    )
    initial_budget = sum(
        node[1] * node[5] for node in conservation_case["node_features"]
    )
    final_budget = sum(node[1] for node in conservation_case["target_continuous"])
    checks["simulator_conservation"] = abs(initial_budget - final_budget) < 1e-5

    tiny_rows = [
        next(row for row in splits_a["validation"] if row["family"] == family)
        for family in ("radiation", "conservation", "nucleation")
    ]
    tiny_batch = next(iter(make_loader(tiny_rows, 3, False, SEED)))
    atom = AtomFieldNet(ticks=2)
    outputs = atom(
        tiny_batch["node_features"],
        tiny_batch["adjacency"],
        tiny_batch["global_features"],
    )
    checks["atom_continuous_shape"] = tuple(outputs["continuous"].shape) == (
        3,
        NODE_COUNT,
        CONTINUOUS_DIM,
    )
    checks["atom_binary_shape"] = tuple(outputs["binary_logits"].shape) == (
        3,
        NODE_COUNT,
        BINARY_DIM,
    )
    checks["atom_route_shape"] = tuple(outputs["route_mean"].shape) == (
        3,
        PROCESS_COUNT,
    )
    route_sums = outputs["route_mean"].sum(dim=-1)
    checks["router_normalized"] = bool(torch.allclose(route_sums, torch.ones_like(route_sums), atol=1e-5))

    closed = tiny_batch["global_features"][:, 7] >= 0.5
    if bool(closed.any()):
        predicted_total = outputs["continuous"][closed, :, 1].sum(dim=1)
        source_total = (
            tiny_batch["node_features"][closed, :, 1]
            * tiny_batch["node_features"][closed, :, 5]
        ).sum(dim=1)
        checks["model_conservation_projection"] = bool(
            torch.allclose(predicted_total, source_total, atol=1e-5)
        )
    else:
        checks["model_conservation_projection"] = False

    ablated = atom(
        tiny_batch["node_features"],
        tiny_batch["adjacency"],
        tiny_batch["global_features"],
        ablate=0,
    )
    checks["ablation_changes_field"] = bool(
        (outputs["continuous"] - ablated["continuous"]).abs().max().item() > 1e-8
    )

    loss, _ = compute_loss(outputs, tiny_batch, route_weight=0.15)
    loss.backward()
    gradient_total = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in atom.parameters()
        if parameter.grad is not None
    )
    checks["finite_training_loss"] = bool(torch.isfinite(loss).item())
    checks["gradient_flow"] = gradient_total > 0.0 and math.isfinite(gradient_total)

    baseline = FlatFieldBaseline(hidden_dim=64)
    baseline_outputs = baseline(
        tiny_batch["node_features"],
        tiny_batch["adjacency"],
        tiny_batch["global_features"],
    )
    checks["baseline_shapes"] = (
        tuple(baseline_outputs["continuous"].shape)
        == (3, NODE_COUNT, CONTINUOUS_DIM)
        and tuple(baseline_outputs["binary_logits"].shape)
        == (3, NODE_COUNT, BINARY_DIM)
    )
    request_payload = {
        "request_id": "self-test-valid",
        "node_features": tiny_rows[0]["node_features"],
        "adjacency": tiny_rows[0]["adjacency"],
        "global_features": tiny_rows[0]["global_features"],
    }
    request_tensors = validate_field_request(request_payload)
    checks["serialized_request_contract"] = tuple(request_tensors[0].shape) == (
        NODE_COUNT,
        NODE_DIM,
    )
    invalid_request = copy.deepcopy(request_payload)
    invalid_request["node_features"][0][1] = -1.0
    try:
        validate_field_request(invalid_request)
    except ValueError:
        checks["invalid_request_rejected"] = True
    else:
        checks["invalid_request_rejected"] = False
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "failed": failed,
        "passed": not failed,
        "dataset_audit": audit,
        "parameter_counts": {
            "atom_test_configuration": parameter_count(atom),
            "flat_test_configuration": parameter_count(baseline),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def validate_field_request(payload: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
    required = {"request_id", "node_features", "adjacency", "global_features"}
    missing = required - set(payload)
    unknown = set(payload) - required
    if missing or unknown:
        raise ValueError(
            f"Invalid request keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if not isinstance(payload["request_id"], str) or not payload["request_id"].strip():
        raise ValueError("request_id must be a non-empty string")
    try:
        nodes = torch.tensor(payload["node_features"], dtype=torch.float32)
        adjacency = torch.tensor(payload["adjacency"], dtype=torch.float32)
        globals_ = torch.tensor(payload["global_features"], dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Field request contains non-numeric tensor data") from error
    if tuple(nodes.shape) != (NODE_COUNT, NODE_DIM):
        raise ValueError(f"node_features must have shape {(NODE_COUNT, NODE_DIM)}")
    if tuple(adjacency.shape) != (NODE_COUNT, NODE_COUNT):
        raise ValueError(f"adjacency must have shape {(NODE_COUNT, NODE_COUNT)}")
    if tuple(globals_.shape) != (GLOBAL_DIM,):
        raise ValueError(f"global_features must have shape {(GLOBAL_DIM,)}")
    if not all(torch.isfinite(value).all() for value in (nodes, adjacency, globals_)):
        raise ValueError("Field request must contain only finite values")
    if bool((nodes[:, 1] < 0.0).any()):
        raise ValueError("Node mass cannot be negative")
    for column, name, lower, upper in (
        (2, "charge", -1.0, 1.0),
        (3, "support", 0.0, 1.0),
        (5, "active", 0.0, 1.0),
        (6, "structure", 0.0, 1.0),
        (7, "cohesion", -1.0, 1.0),
    ):
        if bool(((nodes[:, column] < lower) | (nodes[:, column] > upper)).any()):
            raise ValueError(f"Node {name} values must be within [{lower}, {upper}]")
    if bool(((adjacency < 0.0) | (adjacency > 1.0)).any()):
        raise ValueError("Adjacency values must be within [0, 1]")
    if not torch.allclose(torch.diagonal(adjacency), torch.zeros(NODE_COUNT)):
        raise ValueError("Adjacency self-loops are not accepted")
    rate_positions = (0, 1, 2, 3, 4, 6, 7)
    if any(not 0.0 <= float(globals_[index]) <= 1.0 for index in rate_positions):
        raise ValueError("Process rates and closed_system must be within [0, 1]")
    if not 0.1 <= float(globals_[5]) <= 3.0:
        raise ValueError("nucleation_threshold must be within [0.1, 3.0]")
    return nodes, adjacency, globals_


def run_serialized_field_workflow(
    request_path: Path,
    weights_path: Path,
    response_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    nodes, adjacency, globals_ = validate_field_request(payload)
    model = AtomFieldNet(ticks=FIELD_TICKS).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        output = model(
            nodes.unsqueeze(0).to(device),
            adjacency.unsqueeze(0).to(device),
            globals_.unsqueeze(0).to(device),
        )
    continuous = output["continuous"][0].cpu()
    binary = (
        torch.sigmoid(output["binary_logits"][0]).cpu() >= 0.5
    ).to(torch.int64)
    route = output["route_mean"][0].cpu()
    initial_mass = float((nodes[:, 1] * nodes[:, 5]).sum().item())
    final_mass = float(continuous[:, 1].sum().item())
    closed = bool(float(globals_[7]) >= 0.5)
    response = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "request_id": payload["request_id"],
        "request_hash": stable_hash(payload),
        "field_ticks": FIELD_TICKS,
        "continuous_names": CONTINUOUS_TARGET_NAMES,
        "binary_names": BINARY_TARGET_NAMES,
        "predicted_continuous": continuous.tolist(),
        "predicted_binary": binary.tolist(),
        "route": {
            name: float(route[index].item())
            for index, name in enumerate(PROCESS_NAMES)
        },
        "invariant": {
            "closed_system": closed,
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "absolute_error": abs(final_mass - initial_mass) if closed else 0.0,
        },
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(response_path, response)
    return response


def proof_gates(
    atom_initial_validation: Mapping[str, Any],
    atom_validation: Mapping[str, Any],
    atom_heldout: Mapping[str, Any],
    baseline_heldout: Mapping[str, Any],
    ablations: Mapping[str, Mapping[str, float]],
    serialized_workflow: Mapping[str, Any],
) -> dict[str, Any]:
    initial_validation_objective = validation_objective(
        atom_initial_validation["aggregate"]
    )
    trained_validation_objective = validation_objective(
        atom_validation["aggregate"]
    )
    atom = atom_heldout["aggregate"]
    baseline = baseline_heldout["aggregate"]
    sensitive = [
        name
        for name, result in ablations.items()
        if result["continuous_mae_delta"] > 0.002
        or result["exact_state_delta"] < -0.02
        or result["active_accuracy_delta"] < -0.02
        or result["structure_accuracy_delta"] < -0.02
    ]
    gates = {
        "training_improves_calibration": (
            trained_validation_objective <= 0.90 * initial_validation_objective
        ),
        "learned_continuous_state": atom["continuous_mae"] <= 0.22,
        "learned_activity_state": atom["active_accuracy"] >= 0.80,
        "learned_structure_state": atom["structure_accuracy"] >= 0.80,
        "closed_invariant_preserved": atom["conservation_error"] <= 1e-5,
        "beats_flat_on_unseen_compositions": (
            atom["continuous_mae"] < baseline["continuous_mae"]
            or atom["exact_state"] > baseline["exact_state"]
        ),
        "operator_ablation_sensitivity": len(sensitive) >= 3,
        "serialized_workflow_exercised": bool(serialized_workflow["passed"]),
    }
    return {
        "gates": gates,
        "initial_validation_objective": initial_validation_objective,
        "trained_validation_objective": trained_validation_objective,
        "sensitive_operators": sensitive,
        "passed": all(gates.values()),
    }


def run_experiment(output_dir: Path, epochs: int) -> dict[str, Any]:
    set_deterministic(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    self_tests = run_self_tests()
    if not self_tests["passed"]:
        raise RuntimeError(f"Self-tests failed: {self_tests['failed']}")

    splits = generate_splits()
    audit = audit_splits(splits)
    if not audit["passed"]:
        raise RuntimeError(f"Dataset audit failed: {audit['failures']}")
    for split, rows in splits.items():
        write_jsonl(output_dir / f"tiny_{split}.jsonl", rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atom = AtomFieldNet(ticks=FIELD_TICKS).to(device)
    baseline = FlatFieldBaseline(hidden_dim=128)
    initial_atom_state = copy.deepcopy(atom.state_dict())
    atom_initial_validation = evaluate_model(atom, splits["validation"], device)
    atom_initial_heldout = evaluate_model(atom, splits["heldout"], device)
    atom, atom_training = train_model(
        atom,
        splits["train"],
        splits["validation"],
        device,
        epochs=epochs,
        learning_rate=1.0e-2,
        route_weight=0.15,
        label="atom_field",
    )
    set_deterministic(SEED + 1)
    baseline, baseline_training = train_model(
        baseline,
        splits["train"],
        splits["validation"],
        device,
        epochs=epochs,
        learning_rate=2.5e-3,
        route_weight=0.0,
        label="flat_baseline",
    )

    atom_validation = evaluate_model(atom, splits["validation"], device)
    atom_heldout = evaluate_model(atom, splits["heldout"], device)
    baseline_validation = evaluate_model(baseline, splits["validation"], device)
    baseline_heldout = evaluate_model(baseline, splits["heldout"], device)
    ablations = ablation_report(atom, splits["heldout"], device, atom_heldout)
    parameter_change = {
        name: float((atom.state_dict()[name] - initial).abs().max().item())
        for name, initial in initial_atom_state.items()
    }

    torch.save(atom.state_dict(), output_dir / "atom_field_state.pt")
    torch.save(baseline.state_dict(), output_dir / "flat_baseline_state.pt")
    workflow_case = build_case(
        "full_cycle", SEED + 900_001, "workflow", 0
    )
    workflow_request = {
        "request_id": "fresh-full-cycle-001",
        "node_features": workflow_case["node_features"],
        "adjacency": workflow_case["adjacency"],
        "global_features": workflow_case["global_features"],
    }
    workflow_request_path = output_dir / "atom_field_workflow_request.json"
    workflow_response_path = output_dir / "atom_field_workflow_response.json"
    write_json(workflow_request_path, workflow_request)
    workflow_response = run_serialized_field_workflow(
        workflow_request_path,
        output_dir / "atom_field_state.pt",
        workflow_response_path,
        device,
    )
    workflow_continuous = torch.tensor(
        workflow_response["predicted_continuous"], dtype=torch.float32
    )
    workflow_binary = torch.tensor(
        workflow_response["predicted_binary"], dtype=torch.float32
    )
    workflow_target_continuous = torch.tensor(
        workflow_case["target_continuous"], dtype=torch.float32
    )
    workflow_target_binary = torch.tensor(
        workflow_case["target_binary"], dtype=torch.float32
    )
    workflow_absolute = (workflow_continuous - workflow_target_continuous).abs()
    workflow_metrics = {
        "request_id": workflow_response["request_id"],
        "response_status": workflow_response["status"],
        "continuous_mae": float(workflow_absolute.mean().item()),
        "active_accuracy": float(
            (workflow_binary[:, 0] == workflow_target_binary[:, 0])
            .to(torch.float32)
            .mean()
            .item()
        ),
        "structure_accuracy": float(
            (workflow_binary[:, 1] == workflow_target_binary[:, 1])
            .to(torch.float32)
            .mean()
            .item()
        ),
        "exact_state": bool(
            workflow_absolute.max().item() <= 0.20
            and torch.equal(workflow_binary, workflow_target_binary)
        ),
        "conservation_error": float(
            workflow_response["invariant"]["absolute_error"]
        ),
    }
    workflow_metrics["passed"] = bool(
        workflow_metrics["response_status"] == "ok"
        and workflow_metrics["continuous_mae"] <= 0.22
        and workflow_metrics["active_accuracy"] >= 0.80
        and workflow_metrics["structure_accuracy"] >= 0.80
        and workflow_metrics["conservation_error"] <= 1e-5
    )
    gates = proof_gates(
        atom_initial_validation,
        atom_validation,
        atom_heldout,
        baseline_heldout,
        ablations,
        workflow_metrics,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "atom_field_proof_v4",
        "seed": SEED,
        "architecture": {
            "node_count": NODE_COUNT,
            "node_features": NODE_FEATURE_NAMES,
            "global_features": GLOBAL_FEATURE_NAMES,
            "processes": PROCESS_NAMES,
            "field_ticks": FIELD_TICKS,
            "atom_state_dim": NODE_DIM,
            "operator_semantics": "simultaneous independent state-native branches",
            "flat_hidden_dim": 128,
            "atom_parameters": parameter_count(atom),
            "flat_parameters": parameter_count(baseline),
            "pretrained_weights": False,
            "initialization": "deliberately miscalibrated operator coefficients",
            "lora": False,
        },
        "runtime_interface": {
            "request": "atom_field_workflow_request.json",
            "response": "atom_field_workflow_response.json",
            "validation": "strict shape, range, finiteness, and key checks",
        },
        "dataset": audit,
        "dataset_policy": {
            "gold_source": "deterministic executable simulator",
            "heldout_families_unseen_in_training": True,
            "process_names_visible_to_model": False,
            "process_targets_used_only_for_auxiliary_routing_loss": True,
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "atom_field_proof_v4",
        "manifest": manifest,
        "self_tests": self_tests,
        "training": {
            "atom": atom_training,
            "atom_parameter_max_abs_change": parameter_change,
            "flat_baseline": baseline_training,
        },
        "initial": {
            "atom_validation": atom_initial_validation,
            "atom_heldout": atom_initial_heldout,
        },
        "validation": {
            "atom": atom_validation,
            "flat_baseline": baseline_validation,
        },
        "heldout": {
            "atom": atom_heldout,
            "flat_baseline": baseline_heldout,
        },
        "operator_ablations": ablations,
        "serialized_workflow": workflow_metrics,
        "proof_of_life": gates,
    }
    write_json(output_dir / "atom_field_manifest.json", manifest)
    write_json(output_dir / "atom_field_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs"),
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--infer-request", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--response", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.infer_request is not None:
        if args.weights is None or args.response is None:
            raise ValueError("--infer-request requires --weights and --response")
        response = run_serialized_field_workflow(
            args.infer_request,
            args.weights,
            args.response,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print(json.dumps(response, indent=2, sort_keys=True))
        return
    if args.weights is not None or args.response is not None:
        raise ValueError("--weights and --response require --infer-request")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.self_test:
        result = run_self_tests()
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    report = run_experiment(args.output_dir, args.epochs)
    summary = {
        "proof_of_life": report["proof_of_life"],
        "atom_heldout": report["heldout"]["atom"]["aggregate"],
        "flat_heldout": report["heldout"]["flat_baseline"]["aggregate"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
