"""Multilingual consequence-grounded curriculum for the Atom neural field.

The runtime rows never expose family names, primitive names, query meanings,
language identities, or the simulator controls.  They contain an opaque
utterance, a before-field, an observed after-field, and an opaque response.
Evaluator truth is stored separately and is never passed to the model.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from atom_field_proof import (
    NODE_COUNT,
    PROCESS_NAMES,
    simulate_field,
)


NEURAL_LANGUAGE_SCHEMA = 1
NEURAL_LANGUAGE_SEED = 2026072117
BASE_LANGUAGE_KEYS = ("field-a", "field-b")
TRANSFER_LANGUAGE_KEY = "field-c"
ZERO_SHOT_LANGUAGE_KEY = "field-d"

QUERY_TYPES = (
    "signal_peak",
    "mass_peak",
    "cohesion_peak",
    "ttl_peak",
    "active_count",
    "structure_count",
)

TRAIN_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "u00": ("radiation",),
    "u01": ("dissipation",),
    "u02": ("gravitation",),
    "u03": ("attraction_repulsion",),
    "u04": ("nucleation",),
    "u05": ("conservation",),
    "u06": ("decay",),
    "u07": ("radiation", "dissipation"),
    "u08": ("radiation", "nucleation"),
    "u09": ("gravitation", "conservation"),
    "u10": ("attraction_repulsion", "nucleation"),
    "u11": ("dissipation", "decay"),
    "u12": ("gravitation", "attraction_repulsion"),
    "u13": ("conservation", "decay"),
    "u14": ("radiation", "dissipation", "nucleation"),
    "u15": ("gravitation", "attraction_repulsion", "nucleation"),
    "u16": ("dissipation", "gravitation", "conservation"),
    "u17": ("radiation", "attraction_repulsion", "decay"),
    "u18": ("nucleation", "conservation", "decay"),
    "u19": ("radiation", "gravitation", "conservation"),
    "u20": ("dissipation", "attraction_repulsion", "decay"),
}

HELDOUT_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "h00": ("radiation", "dissipation", "attraction_repulsion", "nucleation"),
    "h01": (
        "dissipation",
        "gravitation",
        "attraction_repulsion",
        "conservation",
        "decay",
    ),
    "h02": ("radiation", "gravitation", "nucleation", "conservation", "decay"),
    "h03": tuple(PROCESS_NAMES),
}

RUNTIME_ROW_KEYS = {
    "adjacency",
    "event_id",
    "node_features",
    "response",
    "salience",
    "target_binary",
    "target_continuous",
    "utterance",
}


@dataclass(frozen=True)
class OpaqueLanguage:
    marker: str
    start: tuple[str, ...]
    joiners: tuple[str, ...]
    ask: str
    low_phase: str
    high_phase: str
    operators: Mapping[str, str]
    queries: Mapping[str, str]
    node_answers: tuple[str, ...]
    count_answers: tuple[str, ...]


@dataclass(frozen=True)
class NeuralLanguageProgram:
    stages: Mapping[str, tuple[dict[str, Any], ...]]
    evaluator_truth: Mapping[str, Mapping[str, Any]]
    vocabulary: tuple[str, ...]
    response_vocabulary: tuple[str, ...]
    manifest: Mapping[str, Any]


LANGUAGES: dict[str, OpaqueLanguage] = {
    "field-a": OpaqueLanguage(
        marker="zaq",
        start=("mira", "sovel"),
        joiners=("en", "tor"),
        ask="vex",
        low_phase="liri",
        high_phase="drom",
        operators={
            "radiation": "pava",
            "dissipation": "nemi",
            "gravitation": "goru",
            "attraction_repulsion": "tasi",
            "nucleation": "kelo",
            "conservation": "vanu",
            "decay": "rith",
        },
        queries={
            "signal_peak": "sai",
            "mass_peak": "moro",
            "cohesion_peak": "bela",
            "ttl_peak": "tiru",
            "active_count": "naka",
            "structure_count": "kiri",
        },
        node_answers=("az0", "az1", "az2", "az3", "az4", "az5"),
        count_answers=("ac0", "ac1", "ac2", "ac3", "ac4", "ac5", "ac6"),
    ),
    "field-b": OpaqueLanguage(
        marker="qim",
        start=("uln", "cavo"),
        joiners=("esh", "dra"),
        ask="phex",
        low_phase="yori",
        high_phase="brak",
        operators={
            "radiation": "wexa",
            "dissipation": "ciri",
            "gravitation": "humo",
            "attraction_repulsion": "jasi",
            "nucleation": "feno",
            "conservation": "qalu",
            "decay": "zeth",
        },
        queries={
            "signal_peak": "wii",
            "mass_peak": "jaro",
            "cohesion_peak": "dela",
            "ttl_peak": "qiru",
            "active_count": "faka",
            "structure_count": "xiri",
        },
        node_answers=("bz0", "bz1", "bz2", "bz3", "bz4", "bz5"),
        count_answers=("bc0", "bc1", "bc2", "bc3", "bc4", "bc5", "bc6"),
    ),
    "field-c": OpaqueLanguage(
        marker="ryx",
        start=("ona", "tevu"),
        joiners=("isa", "plo"),
        ask="khex",
        low_phase="sumi",
        high_phase="gral",
        operators={
            "radiation": "luxa",
            "dissipation": "peri",
            "gravitation": "dumo",
            "attraction_repulsion": "kasi",
            "nucleation": "jeno",
            "conservation": "salu",
            "decay": "meth",
        },
        queries={
            "signal_peak": "lii",
            "mass_peak": "karo",
            "cohesion_peak": "pela",
            "ttl_peak": "siru",
            "active_count": "gaka",
            "structure_count": "miri",
        },
        node_answers=("cz0", "cz1", "cz2", "cz3", "cz4", "cz5"),
        count_answers=("cc0", "cc1", "cc2", "cc3", "cc4", "cc5", "cc6"),
    ),
    "field-d": OpaqueLanguage(
        marker="vop",
        start=("ari", "nuvu"),
        joiners=("eko", "sha"),
        ask="zhex",
        low_phase="tumi",
        high_phase="fral",
        operators={
            "radiation": "muxa",
            "dissipation": "leri",
            "gravitation": "sumo",
            "attraction_repulsion": "pasi",
            "nucleation": "reno",
            "conservation": "talu",
            "decay": "weth",
        },
        queries={
            "signal_peak": "mii",
            "mass_peak": "taro",
            "cohesion_peak": "sela",
            "ttl_peak": "viru",
            "active_count": "haka",
            "structure_count": "riri",
        },
        node_answers=("dz0", "dz1", "dz2", "dz3", "dz4", "dz5"),
        count_answers=("dc0", "dc1", "dc2", "dc3", "dc4", "dc5", "dc6"),
    ),
}


def neural_language_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize_neural_utterance(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("utterance must be text")
    tokens = tuple(text.strip().split())
    if not 2 <= len(tokens) <= 28:
        raise ValueError("utterance token count must be between 2 and 28")
    if any(not token.isascii() or not token.isalnum() for token in tokens):
        raise ValueError("utterance tokens must be opaque ASCII alphanumerics")
    return tokens


def _argmax(values: Sequence[Sequence[float]], column: int) -> int:
    return max(
        range(len(values)), key=lambda index: (float(values[index][column]), -index)
    )


def _semantic_answer(
    query_type: str,
    target_continuous: Sequence[Sequence[float]],
    target_binary: Sequence[Sequence[float]],
) -> tuple[str, int]:
    if query_type == "signal_peak":
        return "node", _argmax(target_continuous, 0)
    if query_type == "mass_peak":
        return "node", _argmax(target_continuous, 1)
    if query_type == "cohesion_peak":
        return "node", _argmax(target_continuous, 2)
    if query_type == "ttl_peak":
        return "node", _argmax(target_continuous, 3)
    if query_type == "active_count":
        return "count", sum(float(row[0]) >= 0.5 for row in target_binary)
    if query_type == "structure_count":
        return "count", sum(float(row[1]) >= 0.5 for row in target_binary)
    raise KeyError(f"unknown query type: {query_type}")


def _surface_answer(
    language: OpaqueLanguage,
    query_type: str,
    target_continuous: Sequence[Sequence[float]],
    target_binary: Sequence[Sequence[float]],
) -> tuple[str, str]:
    kind, value = _semantic_answer(query_type, target_continuous, target_binary)
    if kind == "node":
        return language.node_answers[value], f"node:{value}"
    return language.count_answers[value], f"count:{value}"


def _render_utterance(
    language: OpaqueLanguage,
    signature: Sequence[str],
    query_type: str,
    threshold: float,
    seed: int,
) -> str:
    rng = random.Random(seed)
    operators = [language.operators[name] for name in signature]
    rng.shuffle(operators)
    process_tokens: list[str] = []
    for index, operator in enumerate(operators):
        if index:
            process_tokens.append(
                language.joiners[(seed + index) % len(language.joiners)]
            )
        process_tokens.append(operator)
    phase = language.low_phase if threshold < 1.0 else language.high_phase
    start = language.start[seed % len(language.start)]
    if seed % 2:
        tokens = [
            language.marker,
            start,
            phase,
            *process_tokens,
            language.ask,
            language.queries[query_type],
        ]
    else:
        tokens = [
            language.marker,
            start,
            *process_tokens,
            phase,
            language.ask,
            language.queries[query_type],
        ]
    while len(tokens) < 6:
        tokens.insert(-2, language.joiners[len(tokens) % len(language.joiners)])
    return " ".join(tokens)


def _composition_controls(signature: Sequence[str]) -> torch.Tensor:
    enabled = set(signature)
    values = [
        0.58 if "radiation" in enabled else 0.0,
        0.48 if "dissipation" in enabled else 0.0,
        0.46 if "gravitation" in enabled else 0.0,
        0.58 if "attraction_repulsion" in enabled else 0.0,
        0.86 if "nucleation" in enabled else 0.0,
        0.72 if "nucleation" in enabled else 2.4,
        0.48 if "decay" in enabled else 0.0,
        1.0 if "conservation" in enabled else 0.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _build_composition_case(
    composition: str,
    signature: Sequence[str],
    seed: int,
    stage: str,
    index: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    enabled = set(signature)
    adjacency = torch.zeros(NODE_COUNT, NODE_COUNT, dtype=torch.float32)
    for source in range(NODE_COUNT - 1):
        adjacency[source, source + 1] = 1.0
    for source in range(NODE_COUNT):
        for target in range(NODE_COUNT):
            if source != target and rng.random() < 0.22:
                adjacency[source, target] = 1.0

    signal = torch.tensor([rng.uniform(0.0, 0.16) for _ in range(NODE_COUNT)])
    mass = torch.tensor([rng.uniform(0.30, 1.0) for _ in range(NODE_COUNT)])
    charge = torch.tensor([rng.choice((-1.0, 1.0)) for _ in range(NODE_COUNT)])
    support = torch.tensor([rng.uniform(0.20, 0.95) for _ in range(NODE_COUNT)])
    ttl = torch.tensor([rng.uniform(0.38, 1.0) for _ in range(NODE_COUNT)])
    active = torch.ones(NODE_COUNT)
    structure = torch.zeros(NODE_COUNT)
    cohesion = torch.zeros(NODE_COUNT)

    if "radiation" in enabled:
        signal.zero_()
        signal[rng.randrange(0, 2)] = rng.uniform(0.9, 1.35)
    if "dissipation" in enabled:
        for node in rng.sample(range(NODE_COUNT), k=2):
            support[node] = rng.uniform(0.0, 0.16)
            signal[node] = max(float(signal[node]), rng.uniform(0.55, 0.95))
    if "gravitation" in enabled:
        heavy = rng.randrange(NODE_COUNT)
        mass[heavy] = rng.uniform(1.35, 1.75)
    if "attraction_repulsion" in enabled:
        charge = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).roll(
            rng.randrange(NODE_COUNT)
        )
    if "nucleation" in enabled:
        candidate = rng.randrange(NODE_COUNT)
        signal[candidate] = max(float(signal[candidate]), rng.uniform(0.9, 1.3))
        mass[candidate] = max(float(mass[candidate]), rng.uniform(1.0, 1.45))
    if "decay" in enabled:
        for node in rng.sample(range(NODE_COUNT), k=2):
            ttl[node] = rng.uniform(0.03, 0.18)
            support[node] = rng.uniform(0.0, 0.25)

    node_features = torch.stack(
        (signal, mass, charge, support, ttl, active, structure, cohesion), dim=-1
    )
    controls = _composition_controls(signature)
    target_continuous, target_binary, diagnostics = simulate_field(
        node_features, adjacency, controls
    )
    return {
        "adjacency": adjacency.tolist(),
        "composition": composition,
        "diagnostics": diagnostics,
        "global_features": controls.tolist(),
        "id": f"field-{stage}-{composition}-{index:05d}",
        "node_features": node_features.tolist(),
        "target_binary": target_binary.tolist(),
        "target_continuous": target_continuous.tolist(),
    }


def validate_neural_runtime_row(row: Mapping[str, Any]) -> None:
    if set(row) != RUNTIME_ROW_KEYS:
        raise ValueError(f"runtime row fields must be {sorted(RUNTIME_ROW_KEYS)}")
    if not isinstance(row["event_id"], str) or not row["event_id"]:
        raise ValueError("event_id must be non-empty text")
    tokenize_neural_utterance(row["utterance"])
    if not isinstance(row["response"], str) or not row["response"].isalnum():
        raise ValueError("response must be one opaque token")
    if isinstance(row["salience"], bool) or not isinstance(
        row["salience"], (int, float)
    ):
        raise ValueError("salience must be numeric")
    if not 0.0 < float(row["salience"]) <= 1.0:
        raise ValueError("salience must be in (0, 1]")
    if len(row["node_features"]) != 6 or any(
        len(values) != 8 for values in row["node_features"]
    ):
        raise ValueError("node_features must have shape [6, 8]")
    if len(row["adjacency"]) != 6 or any(
        len(values) != 6 for values in row["adjacency"]
    ):
        raise ValueError("adjacency must have shape [6, 6]")
    if len(row["target_continuous"]) != 6 or any(
        len(values) != 4 for values in row["target_continuous"]
    ):
        raise ValueError("target_continuous must have shape [6, 4]")
    if len(row["target_binary"]) != 6 or any(
        len(values) != 2 for values in row["target_binary"]
    ):
        raise ValueError("target_binary must have shape [6, 2]")


_INDUCED_CONTROL_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}


def _simulate_control_candidates(
    node_features: torch.Tensor,
    adjacency: torch.Tensor,
    controls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = controls.shape[0]
    state = node_features.unsqueeze(0).expand(candidate_count, -1, -1).clone()
    edges = adjacency.unsqueeze(0).expand(candidate_count, -1, -1)
    initial_budget = (state[..., 1] * state[..., 5]).sum(dim=1, keepdim=True)
    for _ in range(4):
        signal = state[..., 0]
        mass = state[..., 1].clamp_min(0.0)
        charge = state[..., 2].clamp(-1.0, 1.0)
        support = state[..., 3].clamp(0.0, 1.0)
        ttl = state[..., 4]
        active = state[..., 5].clamp(0.0, 1.0)
        structure = state[..., 6].clamp(0.0, 1.0)
        cohesion = state[..., 7].clamp(-1.0, 1.0)

        normalized_edges = edges / edges.sum(dim=-1, keepdim=True).clamp_min(1.0)
        incoming_signal = torch.bmm(
            normalized_edges.transpose(1, 2), (signal * active).unsqueeze(-1)
        ).squeeze(-1)
        signal_proposal = (signal + controls[:, 0:1] * incoming_signal) * (
            1.0 - controls[:, 1:2] * (1.0 - support)
        ).clamp(0.0, 1.0)

        attractor_logits = (2.0 * mass + support).masked_fill(active < 0.5, -1e4)
        attractor_weights = torch.softmax(attractor_logits, dim=1)
        live_budget = (mass * active).sum(dim=1, keepdim=True)
        gravitational_mass = (1.0 - controls[:, 2:3]) * mass + controls[
            :, 2:3
        ] * live_budget * attractor_weights

        signed_relation = -charge.unsqueeze(2) * charge.unsqueeze(1) * edges
        relation_degree = edges.transpose(1, 2).sum(dim=-1).clamp_min(1.0)
        relation_field = (
            torch.bmm(signed_relation.transpose(1, 2), active.unsqueeze(-1)).squeeze(-1)
            / relation_degree
        )
        cohesion_proposal = torch.tanh(cohesion + controls[:, 3:4] * relation_field)

        live_mask = (active > 0.5).to(mass.dtype)
        mean_mass = (gravitational_mass * live_mask).sum(
            dim=1, keepdim=True
        ) / live_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized_mass = gravitational_mass / mean_mass.clamp_min(1e-6)
        potential = (
            0.55 * signal_proposal.relu()
            + 0.25 * normalized_mass
            + 0.20 * cohesion_proposal.relu()
        )
        nucleated = (controls[:, 4:5] * potential >= controls[:, 5:6]).to(torch.float32)
        structure_proposal = torch.maximum(structure, nucleated)

        ttl_proposal = ttl - controls[:, 6:7] * (1.0 - 0.25 * support)
        expired = (ttl_proposal <= 0.0) & (structure_proposal < 0.5) & (support < 0.6)
        active_proposal = active * (~expired).to(torch.float32)
        mass_proposal = gravitational_mass * active_proposal
        surviving = mass_proposal.sum(dim=1, keepdim=True)
        closed = controls[:, 7:8] >= 0.5
        no_survivor = closed & (surviving <= 1e-8)
        keeper = F.one_hot(support.argmax(dim=1), num_classes=NODE_COUNT).to(
            torch.float32
        )
        active_proposal = torch.where(no_survivor, keeper, active_proposal)
        mass_proposal = torch.where(no_survivor, keeper * initial_budget, mass_proposal)
        surviving = mass_proposal.sum(dim=1, keepdim=True)
        conserved_mass = mass_proposal * (initial_budget / surviving.clamp_min(1e-8))
        mass_next = torch.where(closed, conserved_mass, mass_proposal).clamp_min(0.0)

        state = torch.stack(
            (
                signal_proposal.clamp(-2.0, 2.0) * active_proposal,
                mass_next,
                charge,
                support,
                ttl_proposal.clamp(-1.0, 1.0),
                active_proposal,
                structure_proposal * active_proposal,
                cohesion_proposal * active_proposal,
            ),
            dim=-1,
        )
    return state[..., [0, 1, 7, 4]], state[..., [5, 6]]


def induce_control_from_consequence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the latent operator composition using only before/after fields.

    This is the non-linguistic inverse problem.  The inducer tries every known
    composition of the seven root mechanics against the observed consequence;
    it never reads the opaque utterance, evaluator truth, or family identity.
    """

    validate_neural_runtime_row(row)
    event_id = str(row["event_id"])
    consequence_payload = {
        "adjacency": row["adjacency"],
        "node_features": row["node_features"],
        "target_binary": row["target_binary"],
        "target_continuous": row["target_continuous"],
    }
    consequence_hash = neural_language_hash(consequence_payload)
    cached = _INDUCED_CONTROL_CACHE.get(event_id)
    if cached is not None and cached[0] == consequence_hash:
        return dict(cached[1])

    node_features = torch.tensor(row["node_features"], dtype=torch.float32)
    adjacency = torch.tensor(row["adjacency"], dtype=torch.float32)
    target_continuous = torch.tensor(row["target_continuous"], dtype=torch.float32)
    target_binary = torch.tensor(row["target_binary"], dtype=torch.float32)
    candidates = {**TRAIN_COMPOSITIONS, **HELDOUT_COMPOSITIONS}
    candidate_rows = list(candidates.items())
    control_batch = torch.stack(
        [_composition_controls(signature) for _, signature in candidate_rows]
    )
    predicted_continuous, predicted_binary = _simulate_control_candidates(
        node_features, adjacency, control_batch
    )
    residuals = (predicted_continuous - target_continuous.unsqueeze(0)).abs().mean(
        dim=(1, 2)
    ) + 2.0 * (predicted_binary - target_binary.unsqueeze(0)).abs().mean(dim=(1, 2))
    scored: list[tuple[float, str, tuple[str, ...], torch.Tensor]] = []
    for candidate_index, (composition, signature) in enumerate(candidate_rows):
        scored.append(
            (
                float(residuals[candidate_index].item()),
                composition,
                tuple(signature),
                control_batch[candidate_index],
            )
        )
    residual, composition, signature, controls = min(
        scored, key=lambda item: (item[0], item[1])
    )
    result = {
        "composition": composition,
        "controls": controls.tolist(),
        "residual": residual,
        "signature": list(signature),
    }
    _INDUCED_CONTROL_CACHE[event_id] = (consequence_hash, dict(result))
    return result


def _runtime_and_truth(
    *,
    stage: str,
    language_key: str,
    composition: str,
    signature: Sequence[str],
    index: int,
    seed: int,
    noise_utterance_composition: tuple[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = _build_composition_case(composition, signature, seed, stage, index)
    query_type = QUERY_TYPES[index % len(QUERY_TYPES)]
    language = LANGUAGES[language_key]
    if noise_utterance_composition is None:
        utterance_composition = composition
        utterance_signature = signature
    else:
        utterance_composition, utterance_signature = noise_utterance_composition
    utterance_controls = _composition_controls(utterance_signature)
    utterance = _render_utterance(
        language,
        utterance_signature,
        query_type,
        float(utterance_controls[5]),
        seed,
    )
    response, semantic_answer = _surface_answer(
        language,
        query_type,
        case["target_continuous"],
        case["target_binary"],
    )
    event_id = f"nl-{stage}-{language.marker}-{index:05d}"
    runtime = {
        "adjacency": case["adjacency"],
        "event_id": event_id,
        "node_features": case["node_features"],
        "response": response,
        "salience": 0.55 if noise_utterance_composition else 1.0,
        "target_binary": case["target_binary"],
        "target_continuous": case["target_continuous"],
        "utterance": utterance,
    }
    validate_neural_runtime_row(runtime)
    truth = {
        "expected_response": response,
        "family": composition,
        "global_features": case["global_features"],
        "is_noise": noise_utterance_composition is not None,
        "language": language_key,
        "process_signature": list(signature),
        "query_type": query_type,
        "semantic_answer": semantic_answer,
        "stage": stage,
        "utterance_family": utterance_composition,
    }
    return runtime, truth


def _append_stage(
    stages: dict[str, list[dict[str, Any]]],
    truth: dict[str, dict[str, Any]],
    stage: str,
    language_keys: Iterable[str],
    compositions: Mapping[str, Sequence[str]],
    cases_per_composition: int,
    seed_offset: int,
) -> None:
    stage_rows = stages.setdefault(stage, [])
    for language_index, language_key in enumerate(language_keys):
        for composition_index, (composition, signature) in enumerate(
            compositions.items()
        ):
            for local_index in range(cases_per_composition):
                index = len(stage_rows)
                seed = (
                    NEURAL_LANGUAGE_SEED
                    + seed_offset
                    + language_index * 1_000_000
                    + composition_index * 10_000
                    + local_index
                )
                runtime, evaluator = _runtime_and_truth(
                    stage=stage,
                    language_key=language_key,
                    composition=composition,
                    signature=signature,
                    index=index,
                    seed=seed,
                )
                stage_rows.append(runtime)
                truth[runtime["event_id"]] = evaluator


def _append_noise_stage(
    stages: dict[str, list[dict[str, Any]]],
    truth: dict[str, dict[str, Any]],
    count: int,
) -> None:
    stage = "transfer_noise"
    stage_rows = stages.setdefault(stage, [])
    compositions = tuple(TRAIN_COMPOSITIONS.items())
    for index in range(count):
        composition, signature = compositions[index % len(compositions)]
        mismatch = compositions[(index * 3 + 4) % len(compositions)]
        if mismatch[0] == composition:
            mismatch = compositions[(index + 1) % len(compositions)]
        seed = NEURAL_LANGUAGE_SEED + 6_000_000 + index
        runtime, evaluator = _runtime_and_truth(
            stage=stage,
            language_key=TRANSFER_LANGUAGE_KEY,
            composition=composition,
            signature=signature,
            index=index,
            seed=seed,
            noise_utterance_composition=mismatch,
        )
        stage_rows.append(runtime)
        truth[runtime["event_id"]] = evaluator


def _program_audit(
    stages: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluator_truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    ids: set[str] = set()
    runtime_tokens: set[str] = set()
    response_tokens: set[str] = set()
    for stage, rows in stages.items():
        for row in rows:
            try:
                validate_neural_runtime_row(row)
            except (TypeError, ValueError) as error:
                failures.append(f"{stage}:{row.get('event_id')}: {error}")
            event_id = str(row["event_id"])
            if event_id in ids:
                failures.append(f"duplicate event_id {event_id}")
            ids.add(event_id)
            runtime_tokens.update(tokenize_neural_utterance(str(row["utterance"])))
            response_tokens.add(str(row["response"]))
            if event_id not in evaluator_truth:
                failures.append(f"missing evaluator truth for {event_id}")
            leaked = set(row) & {
                "family",
                "global_features",
                "is_noise",
                "language",
                "process_signature",
                "query_type",
                "semantic_answer",
                "stage",
                "utterance_family",
            }
            if leaked:
                failures.append(
                    f"evaluator fields leaked into {event_id}: {sorted(leaked)}"
                )

    expected_counts = {
        "base_train": 1008,
        "base_validation": 252,
        "base_composition": 192,
        "transfer_adaptation": 252,
        "transfer_noise": 256,
        "transfer_recovery": 126,
        "transfer_composition": 96,
        "zero_shot_composition": 48,
    }
    counts = {stage: len(rows) for stage, rows in stages.items()}
    if counts != expected_counts:
        failures.append(f"stage counts {counts} != {expected_counts}")

    base_tokens = {
        token
        for stage in ("base_train", "base_validation", "base_composition")
        for row in stages[stage]
        for token in tokenize_neural_utterance(str(row["utterance"]))
    }
    transfer_tokens = {
        token
        for stage in (
            "transfer_adaptation",
            "transfer_noise",
            "transfer_recovery",
            "transfer_composition",
        )
        for row in stages[stage]
        for token in tokenize_neural_utterance(str(row["utterance"]))
    }
    zero_tokens = {
        token
        for row in stages["zero_shot_composition"]
        for token in tokenize_neural_utterance(str(row["utterance"]))
    }
    if base_tokens & transfer_tokens:
        failures.append("base and transfer utterance vocabularies overlap")
    if (base_tokens | transfer_tokens) & zero_tokens:
        failures.append("zero-shot utterance vocabulary overlaps learned languages")

    adaptation_responses = {
        str(row["response"]) for row in stages["transfer_adaptation"]
    }
    adaptation_responses.update(
        str(row["response"]) for row in stages["transfer_recovery"]
    )
    transfer_eval_responses = {
        str(row["response"]) for row in stages["transfer_composition"]
    }
    missing_transfer_responses = sorted(transfer_eval_responses - adaptation_responses)
    if missing_transfer_responses:
        failures.append(
            "transfer evaluation uses unseen response surfaces: "
            + ", ".join(missing_transfer_responses)
        )

    return {
        "checks": {
            "base_transfer_vocab_disjoint": not bool(base_tokens & transfer_tokens),
            "evaluator_truth_separate": not any(
                "leaked" in failure for failure in failures
            ),
            "response_surfaces_covered": not missing_transfer_responses,
            "runtime_rows_valid": not any(
                ":" in failure and "fields leaked" not in failure
                for failure in failures
            ),
            "zero_shot_vocab_disjoint": not bool(
                (base_tokens | transfer_tokens) & zero_tokens
            ),
        },
        "counts": counts,
        "failed": failures,
        "passed": not failures,
        "runtime_token_count": len(runtime_tokens),
        "response_token_count": len(response_tokens),
    }


def build_neural_language_program() -> NeuralLanguageProgram:
    stages: dict[str, list[dict[str, Any]]] = {}
    evaluator_truth: dict[str, dict[str, Any]] = {}

    _append_stage(
        stages,
        evaluator_truth,
        "base_train",
        BASE_LANGUAGE_KEYS,
        TRAIN_COMPOSITIONS,
        24,
        0,
    )
    _append_stage(
        stages,
        evaluator_truth,
        "base_validation",
        BASE_LANGUAGE_KEYS,
        TRAIN_COMPOSITIONS,
        6,
        1_000_000,
    )
    _append_stage(
        stages,
        evaluator_truth,
        "base_composition",
        BASE_LANGUAGE_KEYS,
        HELDOUT_COMPOSITIONS,
        24,
        2_000_000,
    )
    _append_stage(
        stages,
        evaluator_truth,
        "transfer_adaptation",
        (TRANSFER_LANGUAGE_KEY,),
        TRAIN_COMPOSITIONS,
        12,
        3_000_000,
    )
    _append_noise_stage(stages, evaluator_truth, 256)
    _append_stage(
        stages,
        evaluator_truth,
        "transfer_recovery",
        (TRANSFER_LANGUAGE_KEY,),
        TRAIN_COMPOSITIONS,
        6,
        4_000_000,
    )
    _append_stage(
        stages,
        evaluator_truth,
        "transfer_composition",
        (TRANSFER_LANGUAGE_KEY,),
        HELDOUT_COMPOSITIONS,
        24,
        5_000_000,
    )
    _append_stage(
        stages,
        evaluator_truth,
        "zero_shot_composition",
        (ZERO_SHOT_LANGUAGE_KEY,),
        HELDOUT_COMPOSITIONS,
        12,
        7_000_000,
    )

    audit = _program_audit(stages, evaluator_truth)
    if not audit["passed"]:
        raise AssertionError("; ".join(audit["failed"]))

    vocabulary = sorted(
        {
            token
            for rows in stages.values()
            for row in rows
            for token in tokenize_neural_utterance(str(row["utterance"]))
        }
    )
    response_vocabulary = sorted(
        {str(row["response"]) for rows in stages.values() for row in rows}
    )
    frozen_stages = {name: tuple(rows) for name, rows in stages.items()}
    runtime_hashes = {
        stage: neural_language_hash(rows) for stage, rows in frozen_stages.items()
    }
    evaluator_hash = neural_language_hash(evaluator_truth)
    manifest = {
        "audit": audit,
        "base_languages": len(BASE_LANGUAGE_KEYS),
        "evaluator_sha256": evaluator_hash,
        "heldout_composition_families": len(HELDOUT_COMPOSITIONS),
        "operator_count": len(PROCESS_NAMES),
        "response_vocabulary_size": len(response_vocabulary),
        "runtime_fields": sorted(RUNTIME_ROW_KEYS),
        "runtime_sha256": runtime_hashes,
        "schema_version": NEURAL_LANGUAGE_SCHEMA,
        "seed": NEURAL_LANGUAGE_SEED,
        "stages": {name: len(rows) for name, rows in frozen_stages.items()},
        "train_family_count": len(TRAIN_COMPOSITIONS),
        "transfer_language_count": 1,
        "vocabulary_size": len(vocabulary),
        "zero_shot_language_count": 1,
    }
    return NeuralLanguageProgram(
        stages=frozen_stages,
        evaluator_truth=evaluator_truth,
        vocabulary=tuple(vocabulary),
        response_vocabulary=tuple(response_vocabulary),
        manifest=manifest,
    )


def neural_language_self_tests() -> dict[str, Any]:
    program = build_neural_language_program()
    first = build_neural_language_program()
    checks = {
        "deterministic_manifest": program.manifest == first.manifest,
        "evaluator_truth_is_separate": all(
            set(row) == RUNTIME_ROW_KEYS
            for rows in program.stages.values()
            for row in rows
        ),
        "seven_operator_vocabulary": all(
            set(language.operators) == set(PROCESS_NAMES)
            for language in LANGUAGES.values()
        ),
        "transfer_and_zero_shot_are_disjoint": not (
            set(LANGUAGES[TRANSFER_LANGUAGE_KEY].operators.values())
            & set(LANGUAGES[ZERO_SHOT_LANGUAGE_KEY].operators.values())
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}
