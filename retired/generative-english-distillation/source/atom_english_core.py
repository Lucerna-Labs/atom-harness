"""Generative English core built from causal Atom graph dynamics.

Tokens are nodes in a directed temporal graph. A bounded predecessor set and
persistent graph landmarks form the edges. Training is parallel over complete
token blocks; generation uses equivalent recurrent graph state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ATOM_ENGLISH_CORE_RUNTIME = "atom-generative-english-core-v1"
ATOM_LONG_CONTEXT_TARGET = 524_288
ATOM_LONG_CONTEXT_MILESTONES = (
    32_768,
    65_536,
    131_072,
    264_000,
    ATOM_LONG_CONTEXT_TARGET,
)
ATOM_ROOT_PRIMITIVES = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
)


def _unique_positive(values: Iterable[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        number = int(value)
        if number <= 0:
            raise ValueError("causal graph offsets must be positive")
        if number not in result:
            result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class AtomEnglishConfig:
    """Serializable shape and dynamics for an Atom English model."""

    vocab_size: int
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    ffn_multiplier: float = 3.0
    local_radius: int = 12
    dilation_offsets: tuple[int, ...] = (
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
    )
    graph_neighbors: int = 12
    persistence_slots: int = 4
    persistence_landmarks_per_slot: int = 4
    persistence_chunk: int = 32
    symbolic_copy_orders: tuple[int, ...] = (4, 8, 16)
    symbolic_copy_logit_gain: float = 18.0
    symbolic_copy_neural_margin: float = 4.0
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_base: float = 10_000.0
    rope_native_context: int = 8_192
    criticality_target: float = 0.58
    criticality_weight: float = 0.01
    update_ratio_limit: float = 1.25
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    schema_version: int = 1
    architecture: str = ATOM_ENGLISH_CORE_RUNTIME

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Atom English config schema")
        if self.architecture != ATOM_ENGLISH_CORE_RUNTIME:
            raise ValueError("unsupported Atom English architecture")
        if self.vocab_size < 256:
            raise ValueError("vocabulary must contain at least 256 tokens")
        if self.d_model < 64 or self.d_model % self.n_heads:
            raise ValueError("d_model must be >= 64 and divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("each phase head must have an even width")
        if self.n_layers < 1 or self.n_heads < 1:
            raise ValueError("the model requires at least one layer and head")
        if not 1.0 <= self.ffn_multiplier <= 8.0:
            raise ValueError("ffn_multiplier must be inside [1, 8]")
        if not 1 <= self.local_radius <= 64:
            raise ValueError("local_radius must be inside [1, 64]")
        offsets = _unique_positive(self.dilation_offsets)
        if offsets != self.dilation_offsets:
            raise ValueError("dilation offsets must be unique positive values")
        candidate_count = (
            self.local_radius
            + len(self.dilation_offsets)
            + self.persistence_slots * (1 + self.persistence_landmarks_per_slot)
        )
        if not 1 <= self.graph_neighbors <= candidate_count:
            raise ValueError("graph_neighbors exceeds the candidate graph")
        if not 1 <= self.persistence_slots <= 16:
            raise ValueError("persistence_slots must be inside [1, 16]")
        if not 1 <= self.persistence_landmarks_per_slot <= 8:
            raise ValueError("persistence_landmarks_per_slot must be inside [1, 8]")
        if not 4 <= self.persistence_chunk <= self.max_seq_len:
            raise ValueError("persistence_chunk is outside the context")
        if (
            self.persistence_slots * self.persistence_landmarks_per_slot
            > self.persistence_chunk
        ):
            raise ValueError(
                "persistence chunk must hold every slot landmark partition"
            )
        copy_orders = _unique_positive(self.symbolic_copy_orders)
        if copy_orders != self.symbolic_copy_orders:
            raise ValueError("symbolic copy orders must be unique positive values")
        if copy_orders[0] < 2 or copy_orders[-1] > 32:
            raise ValueError("symbolic copy orders must be inside [2, 32]")
        if not 1.0 <= self.symbolic_copy_logit_gain <= 32.0:
            raise ValueError("symbolic copy logit gain must be inside [1, 32]")
        if not 0.0 <= self.symbolic_copy_neural_margin <= 8.0:
            raise ValueError("symbolic copy neural margin must be inside [0, 8]")
        if self.max_seq_len < 64:
            raise ValueError("max_seq_len must be at least 64")
        if not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be inside [0, 0.5)")
        if self.rope_base <= 100.0:
            raise ValueError("rope_base is too small")
        if self.rope_native_context < 64:
            raise ValueError("rope_native_context must be at least 64")
        if not 0.0 < self.criticality_target < 1.0:
            raise ValueError("criticality_target must be inside (0, 1)")
        if not 0.0 <= self.criticality_weight <= 1.0:
            raise ValueError("criticality_weight must be inside [0, 1]")
        if not 0.1 <= self.update_ratio_limit <= 4.0:
            raise ValueError("update_ratio_limit must be inside [0.1, 4]")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = int(getattr(self, name))
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} is outside the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def graph_offsets(self) -> tuple[int, ...]:
        local = tuple(range(1, self.local_radius + 1))
        return _unique_positive(
            value
            for value in (*local, *self.dilation_offsets)
            if value <= self.max_seq_len
        )

    @property
    def exact_cache_tokens(self) -> int:
        return max(self.graph_offsets)

    @property
    def persistence_levels(self) -> int:
        chunk_count = math.ceil(self.max_seq_len / self.persistence_chunk)
        return max(1, math.ceil(math.log2(chunk_count + 1)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dilation_offsets"] = list(self.dilation_offsets)
        payload["symbolic_copy_orders"] = list(self.symbolic_copy_orders)
        payload["root_primitives"] = list(ATOM_ROOT_PRIMITIVES)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomEnglishConfig":
        data = dict(payload)
        roots = data.pop("root_primitives", list(ATOM_ROOT_PRIMITIVES))
        if tuple(roots) != ATOM_ROOT_PRIMITIVES:
            raise ValueError("Atom English root primitive contract changed")
        data["dilation_offsets"] = tuple(data["dilation_offsets"])
        if "symbolic_copy_orders" in data:
            data["symbolic_copy_orders"] = tuple(data["symbolic_copy_orders"])
        return cls(**data)


def atom_english_profile(
    name: str,
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> AtomEnglishConfig:
    common = {
        "vocab_size": vocab_size,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    if name == "verification":
        return AtomEnglishConfig(
            **common,
            d_model=64,
            n_layers=2,
            n_heads=4,
            ffn_multiplier=2.0,
            local_radius=4,
            dilation_offsets=(8, 16, 32, 64),
            graph_neighbors=4,
            persistence_slots=2,
            persistence_chunk=8,
            max_seq_len=64,
        )
    if name == "kaggle-40m":
        return AtomEnglishConfig(**common)
    if name == "kaggle-82m":
        return AtomEnglishConfig(
            **common,
            d_model=512,
            n_layers=12,
            n_heads=8,
            local_radius=16,
            graph_neighbors=16,
            persistence_slots=6,
            max_seq_len=1536,
        )
    if name == "scale-227m":
        return AtomEnglishConfig(
            **common,
            d_model=768,
            n_layers=16,
            n_heads=12,
            ffn_multiplier=3.5,
            local_radius=20,
            dilation_offsets=(
                24,
                32,
                48,
                64,
                96,
                128,
                192,
                256,
                384,
                512,
                768,
                1024,
                1536,
                2048,
                3072,
                4096,
                6144,
            ),
            graph_neighbors=20,
            persistence_slots=8,
            max_seq_len=ATOM_LONG_CONTEXT_TARGET,
            criticality_weight=0.006,
        )
    raise ValueError(f"unknown Atom English profile: {name}")


def atom_english_context_expansion_plan(
    source_config: AtomEnglishConfig,
    target_config: AtomEnglishConfig,
) -> dict[str, Any]:
    """Validate and describe a context-only model migration."""

    immutable_fields = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "ffn_multiplier",
        "local_radius",
        "graph_neighbors",
        "persistence_slots",
        "persistence_landmarks_per_slot",
        "persistence_chunk",
        "symbolic_copy_orders",
        "symbolic_copy_logit_gain",
        "symbolic_copy_neural_margin",
        "dropout",
        "rope_base",
        "rope_native_context",
        "criticality_target",
        "criticality_weight",
        "update_ratio_limit",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "schema_version",
        "architecture",
    )
    changed = [
        name
        for name in immutable_fields
        if getattr(source_config, name) != getattr(target_config, name)
    ]
    if changed:
        raise ValueError(
            "context expansion cannot change model semantics: " + ", ".join(changed)
        )
    if target_config.max_seq_len < source_config.max_seq_len:
        raise ValueError("context expansion cannot shorten the context")
    source_offsets = source_config.graph_offsets
    target_offsets = target_config.graph_offsets
    if not set(source_offsets).issubset(target_offsets):
        raise ValueError("target context omits learned causal distances")
    return {
        "source_context_tokens": source_config.max_seq_len,
        "target_context_tokens": target_config.max_seq_len,
        "preserved_causal_distances": list(source_offsets),
        "initialized_causal_distances": sorted(
            set(target_offsets) - set(source_offsets)
        ),
    }


def expand_atom_english_context(
    model: "AtomCausalLanguageModel",
    target_config: AtomEnglishConfig,
) -> "AtomCausalLanguageModel":
    """Expand causal distances while preserving every compatible learned value."""

    source_config = model.config
    atom_english_context_expansion_plan(source_config, target_config)
    source_offsets = source_config.graph_offsets
    target_offsets = target_config.graph_offsets

    device = model.token_embedding.weight.device
    dtype = model.token_embedding.weight.dtype
    expanded = AtomCausalLanguageModel(target_config).to(
        device=device,
        dtype=dtype,
    )
    source_state = model.state_dict()
    target_state = expanded.state_dict()
    source_indices = {offset: index for index, offset in enumerate(source_offsets)}
    target_indices = {offset: index for index, offset in enumerate(target_offsets)}
    with torch.no_grad():
        for name, source_value in source_state.items():
            target_value = target_state.get(name)
            if target_value is None:
                raise ValueError(f"expanded model is missing parameter: {name}")
            if source_value.shape == target_value.shape:
                target_value.copy_(source_value)
                continue
            if not name.endswith(".graph.relation_bias"):
                raise ValueError(f"unsupported context expansion tensor: {name}")
            if source_value.shape[0] != target_value.shape[0]:
                raise ValueError("causal relation head count changed")
            for offset, source_index in source_indices.items():
                target_value[:, target_indices[offset]].copy_(
                    source_value[:, source_index]
                )
    missing, unexpected = expanded.load_state_dict(target_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"expanded model state mismatch: missing={missing}, unexpected={unexpected}"
        )
    return expanded


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, value: Tensor) -> Tensor:
        scale = value.float().pow(2).mean(dim=-1, keepdim=True)
        inverse = torch.reciprocal(torch.sqrt(scale + self.epsilon))
        return value * inverse.to(value.dtype) * self.weight


class ConservativeResidual(nn.Module):
    """Bound update energy before adding it to the residual stream."""

    def __init__(self, width: int, ratio_limit: float) -> None:
        super().__init__()
        self.ratio_limit = float(ratio_limit)
        self.gain = nn.Parameter(torch.zeros(width))

    def forward(self, residual: Tensor, update: Tensor) -> Tensor:
        residual_energy = (
            residual.float().pow(2).mean(dim=-1, keepdim=True) + 1e-12
        ).sqrt()
        update_energy = (
            update.float().pow(2).mean(dim=-1, keepdim=True) + 1e-12
        ).sqrt()
        limit = self.ratio_limit * residual_energy.clamp_min(1e-4)
        ratio = (limit / update_energy.clamp_min(1e-6)).clamp(max=1.0)
        gain = torch.sigmoid(self.gain).to(update.dtype)
        return residual + update * ratio.to(update.dtype) * gain


class PhaseMixer(nn.Module):
    """Nonlinear amplitude and phase composition."""

    def __init__(self, width: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        hidden = int(math.ceil(width * multiplier / 8.0) * 8)
        self.in_projection = nn.Linear(width, hidden * 3, bias=False)
        self.out_projection = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.phase_bias = nn.Parameter(torch.zeros(hidden))

    def forward(self, value: Tensor) -> Tensor:
        amplitude, carrier, phase = self.in_projection(value).chunk(3, dim=-1)
        angle = math.pi * torch.tanh(phase + self.phase_bias)
        mixed = F.silu(amplitude) * torch.cos(angle)
        mixed = mixed + F.silu(carrier) * torch.sin(angle)
        return self.dropout(self.out_projection(mixed))


@dataclass
class AtomGraphStepState:
    exact_cache_capacity: int
    persistence_levels: int
    rotated_keys: deque[Tensor] = field(default_factory=deque)
    values: deque[Tensor] = field(default_factory=deque)
    hierarchy_numerators: list[Tensor | None] = field(default_factory=list)
    hierarchy_masses: list[Tensor | None] = field(default_factory=list)
    hierarchy_landmarks: list[Tensor | None] = field(default_factory=list)
    hierarchy_landmark_scores: list[Tensor | None] = field(default_factory=list)
    hierarchy_landmark_positions: list[Tensor | None] = field(default_factory=list)
    current_numerator: Tensor | None = None
    current_mass: Tensor | None = None
    current_values: list[Tensor] = field(default_factory=list)
    current_gates: list[Tensor] = field(default_factory=list)
    current_positions: list[Tensor] = field(default_factory=list)
    symbolic_histories: list[list[int]] = field(default_factory=list)
    symbolic_transitions: list[dict[int, int | dict[int, int]]] = field(
        default_factory=list
    )
    current_count: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        if self.exact_cache_capacity < 1:
            raise ValueError("exact cache capacity must be positive")
        if self.persistence_levels < 1:
            raise ValueError("persistence level count must be positive")
        self.rotated_keys = deque(
            self.rotated_keys,
            maxlen=self.exact_cache_capacity,
        )
        self.values = deque(
            self.values,
            maxlen=self.exact_cache_capacity,
        )
        if not self.hierarchy_numerators:
            self.hierarchy_numerators = [None] * self.persistence_levels
        if not self.hierarchy_masses:
            self.hierarchy_masses = [None] * self.persistence_levels
        if not self.hierarchy_landmarks:
            self.hierarchy_landmarks = [None] * self.persistence_levels
        if not self.hierarchy_landmark_scores:
            self.hierarchy_landmark_scores = [None] * self.persistence_levels
        if not self.hierarchy_landmark_positions:
            self.hierarchy_landmark_positions = [None] * self.persistence_levels
        if len(self.hierarchy_numerators) != self.persistence_levels:
            raise ValueError("persistence numerator hierarchy has the wrong size")
        if len(self.hierarchy_masses) != self.persistence_levels:
            raise ValueError("persistence mass hierarchy has the wrong size")
        hierarchy_fields = (
            self.hierarchy_landmarks,
            self.hierarchy_landmark_scores,
            self.hierarchy_landmark_positions,
        )
        if any(len(values) != self.persistence_levels for values in hierarchy_fields):
            raise ValueError("persistence landmark hierarchy has the wrong size")

    @property
    def resident_exact_tokens(self) -> int:
        return len(self.rotated_keys)


@dataclass(frozen=True)
class AtomGraphDiagnostics:
    edge_entropy: Tensor
    active_edges: Tensor
    temperature: Tensor


def _phase_positions(positions: Tensor, native_context: int) -> Tensor:
    """Preserve native phases, then extend them without periodic alias collapse."""

    position = positions.float()
    native = torch.tensor(
        float(native_context),
        dtype=torch.float32,
        device=positions.device,
    )
    overflow = (position - native).clamp_min(0.0)
    extended = native + native * torch.log1p(overflow / native)
    return torch.where(position <= native, position, extended)


def _apply_rope(
    value: Tensor,
    positions: Tensor,
    base: float,
    native_context: int,
) -> Tensor:
    width = value.shape[-1]
    half = width // 2
    frequency = torch.arange(half, dtype=torch.float32, device=value.device)
    frequency = torch.pow(
        torch.tensor(base, dtype=torch.float32, device=value.device),
        -frequency / max(half, 1),
    )
    phase_position = _phase_positions(positions, native_context)
    angle = phase_position.unsqueeze(-1) * frequency.unsqueeze(0)
    cosine = torch.cos(angle).to(value.dtype)[None, None, :, :]
    sine = torch.sin(angle).to(value.dtype)[None, None, :, :]
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _apply_rope_step(
    value: Tensor,
    position: int,
    base: float,
    native_context: int,
) -> Tensor:
    positions = torch.tensor([position], device=value.device)
    return _apply_rope(
        value.unsqueeze(2),
        positions,
        base,
        native_context,
    ).squeeze(2)


def _apply_rope_candidates(
    value: Tensor,
    positions: Tensor,
    base: float,
    native_context: int,
) -> Tensor:
    """Apply phase to independently positioned memory candidates."""

    if value.ndim != 5:
        raise ValueError("candidate phase input must have five dimensions")
    batch, _, time, candidates, width = value.shape
    if positions.shape != (batch, time, candidates):
        raise ValueError("candidate positions do not match projected memory")
    half = width // 2
    frequency = torch.arange(half, dtype=torch.float32, device=value.device)
    frequency = torch.pow(
        torch.tensor(base, dtype=torch.float32, device=value.device),
        -frequency / max(half, 1),
    )
    valid = positions >= 0
    safe_positions = positions.clamp_min(0)
    phase_position = _phase_positions(safe_positions, native_context)
    angle = phase_position.unsqueeze(-1) * frequency.view(1, 1, 1, -1)
    cosine = torch.cos(angle).to(value.dtype)[:, None]
    sine = torch.sin(angle).to(value.dtype)[:, None]
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    rotated = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)
    return torch.where(valid[:, None, :, :, None], rotated, value)


class TopologicalPersistence(nn.Module):
    """Query-recognized logarithmic landmarks from completed text regions."""

    def __init__(
        self,
        width: int,
        slots: int,
        landmarks_per_slot: int,
        chunk_size: int,
        maximum_context: int,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.slots = int(slots)
        self.landmarks_per_slot = int(landmarks_per_slot)
        self.chunk_size = int(chunk_size)
        chunk_count = math.ceil(maximum_context / chunk_size)
        self.max_levels = max(1, math.ceil(math.log2(chunk_count + 1)))
        self.persistence_gate = nn.Linear(width, slots, bias=True)
        nn.init.constant_(self.persistence_gate.bias, -0.5)

    def _recognized_summary(
        self,
        query: Tensor,
        numerators: Sequence[Tensor | None],
        masses: Sequence[Tensor | None],
    ) -> tuple[Tensor, Tensor]:
        batch, time, _ = query.shape
        occupied = [
            (level, numerator, masses[level])
            for level, numerator in enumerate(numerators)
            if numerator is not None and masses[level] is not None
        ]
        if not occupied:
            return (
                query.new_zeros((batch, time, self.slots, self.width)),
                torch.zeros(
                    (batch, time, self.slots),
                    dtype=torch.bool,
                    device=query.device,
                ),
            )
        level_ids = [level for level, _, _ in occupied]
        stacked_numerator = torch.stack(
            [numerator for _, numerator, _ in occupied],
            dim=1,
        )
        stacked_mass = torch.stack(
            [mass for _, _, mass in occupied],
            dim=1,
        )
        summaries = stacked_numerator / stacked_mass.clamp_min(1e-6).unsqueeze(-1)
        valid = stacked_mass > 1e-6
        normalized_query = F.normalize(query.float(), dim=-1)
        normalized_summary = F.normalize(summaries.float(), dim=-1)
        scores = torch.einsum(
            "btd,blsd->btls",
            normalized_query,
            normalized_summary,
        )
        level_penalty = torch.tensor(
            level_ids,
            dtype=scores.dtype,
            device=scores.device,
        ).view(1, 1, -1, 1)
        scores = scores - 0.025 * level_penalty
        expanded_valid = valid[:, None].expand(batch, time, -1, -1)
        scores = scores.masked_fill(~expanded_valid, -1.0e4)
        weights = torch.softmax(scores, dim=2) * expanded_valid.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-9)
        recognized = torch.einsum(
            "btls,blsd->btsd",
            weights.to(summaries.dtype),
            summaries,
        )
        return recognized.to(query.dtype), valid.any(dim=1)[:, None].expand(
            batch,
            time,
            self.slots,
        )

    def _empty_landmarks(
        self,
        query: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = query.shape
        count = self.slots * self.landmarks_per_slot
        return (
            query.new_zeros((batch, time, count, self.width)),
            torch.zeros(
                (batch, time, count),
                dtype=torch.bool,
                device=query.device,
            ),
            torch.full(
                (batch, time, count),
                -1,
                dtype=torch.long,
                device=query.device,
            ),
        )

    def _recognized_landmarks(
        self,
        query: Tensor,
        landmarks: Sequence[Tensor | None],
        landmark_scores: Sequence[Tensor | None],
        landmark_positions: Sequence[Tensor | None],
    ) -> tuple[Tensor, Tensor, Tensor]:
        occupied = [
            (
                landmark,
                landmark_scores[level],
                landmark_positions[level],
            )
            for level, landmark in enumerate(landmarks)
            if (
                landmark is not None
                and landmark_scores[level] is not None
                and landmark_positions[level] is not None
            )
        ]
        if not occupied:
            return self._empty_landmarks(query)

        batch, time, _ = query.shape
        values = torch.cat(
            [landmark for landmark, _, _ in occupied],
            dim=2,
        )
        salience = torch.cat(
            [score for _, score, _ in occupied],
            dim=2,
        )
        positions = torch.cat(
            [position for _, _, position in occupied],
            dim=2,
        )
        normalized_query = F.normalize(query.float(), dim=-1)
        normalized_values = F.normalize(values.float(), dim=-1)
        recognition = torch.einsum(
            "btd,bsnd->btsn",
            normalized_query,
            normalized_values,
        )
        positive_salience = salience.float().clamp_min(0.0)
        salience_scale = positive_salience.amax(dim=2, keepdim=True).clamp_min(1e-6)
        recognition = recognition + 0.05 * (positive_salience / salience_scale)[:, None]
        maximum_position = positions.amax(dim=(1, 2), keepdim=True).clamp_min(1)
        recency = positions.float() / maximum_position.float()
        recognition = recognition + 0.01 * recency[:, None]

        keep = min(self.landmarks_per_slot, values.shape[2])
        selected_scores, selected_indices = torch.topk(
            recognition,
            keep,
            dim=-1,
        )
        expanded_values = values[:, None].expand(
            batch,
            time,
            self.slots,
            values.shape[2],
            self.width,
        )
        selected = torch.gather(
            expanded_values,
            3,
            selected_indices.unsqueeze(-1).expand(
                batch,
                time,
                self.slots,
                keep,
                self.width,
            ),
        )
        selected_position = torch.gather(
            positions[:, None].expand(
                batch,
                time,
                self.slots,
                values.shape[2],
            ),
            3,
            selected_indices,
        )
        straight_through_scale = 1.0 + 0.1 * (
            selected_scores - selected_scores.detach()
        )
        selected = selected * straight_through_scale.unsqueeze(-1).to(selected.dtype)
        valid = torch.ones(
            (batch, time, self.slots, keep),
            dtype=torch.bool,
            device=query.device,
        )
        if keep < self.landmarks_per_slot:
            missing = self.landmarks_per_slot - keep
            selected = torch.cat(
                (
                    selected,
                    selected.new_zeros((batch, time, self.slots, missing, self.width)),
                ),
                dim=3,
            )
            valid = torch.cat(
                (
                    valid,
                    torch.zeros(
                        (batch, time, self.slots, missing),
                        dtype=torch.bool,
                        device=query.device,
                    ),
                ),
                dim=3,
            )
            selected_position = torch.cat(
                (
                    selected_position,
                    torch.full(
                        (batch, time, self.slots, missing),
                        -1,
                        dtype=torch.long,
                        device=query.device,
                    ),
                ),
                dim=3,
            )
        return (
            selected.reshape(
                batch,
                time,
                self.slots * self.landmarks_per_slot,
                self.width,
            ).to(query.dtype),
            valid.reshape(
                batch,
                time,
                self.slots * self.landmarks_per_slot,
            ),
            selected_position.reshape(
                batch,
                time,
                self.slots * self.landmarks_per_slot,
            ),
        )

    def _recognized_memory(
        self,
        query: Tensor,
        numerators: Sequence[Tensor | None],
        masses: Sequence[Tensor | None],
        landmarks: Sequence[Tensor | None],
        landmark_scores: Sequence[Tensor | None],
        landmark_positions: Sequence[Tensor | None],
    ) -> tuple[Tensor, Tensor, Tensor]:
        summary, summary_valid = self._recognized_summary(
            query,
            numerators,
            masses,
        )
        episodic, episodic_valid, episodic_positions = self._recognized_landmarks(
            query,
            landmarks,
            landmark_scores,
            landmark_positions,
        )
        occupied_positions = [
            position for position in landmark_positions if position is not None
        ]
        if occupied_positions:
            summary_positions = torch.cat(
                occupied_positions,
                dim=2,
            ).amax(dim=2)
            summary_positions = summary_positions[:, None].expand(
                query.shape[0],
                query.shape[1],
                self.slots,
            )
        else:
            summary_positions = torch.full(
                summary_valid.shape,
                -1,
                dtype=torch.long,
                device=query.device,
            )
        return (
            torch.cat((summary, episodic), dim=2),
            torch.cat((summary_valid, episodic_valid), dim=2),
            torch.cat((summary_positions, episodic_positions), dim=2),
        )

    def _chunk_landmarks(
        self,
        value: Tensor,
        gate: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = value.shape
        if time != self.chunk_size:
            raise ValueError("landmarks require a completed persistence chunk")
        if positions.ndim == 1:
            positions = positions[None].expand(batch, -1)
        if positions.shape != (batch, time):
            raise ValueError("landmark positions do not match the chunk")

        novelty = (
            (value.float() - value.float().mean(dim=1, keepdim=True))
            .square()
            .mean(dim=-1)
            .sqrt()
        )
        novelty = novelty / novelty.mean(dim=1, keepdim=True).clamp_min(1e-6)
        salience = gate.float() * (1.0 + 0.25 * novelty.unsqueeze(-1))
        selected_values: list[Tensor] = []
        selected_scores: list[Tensor] = []
        selected_positions: list[Tensor] = []
        for slot in range(self.slots):
            partition = torch.arange(
                slot,
                time,
                self.slots,
                device=value.device,
            )
            if partition.numel() < self.landmarks_per_slot:
                raise ValueError("persistence landmark partition is undersized")
            slot_scores = salience[:, partition, slot]
            top_scores, relative_indices = torch.topk(
                slot_scores,
                self.landmarks_per_slot,
                dim=1,
            )
            absolute_indices = partition[relative_indices]
            selected_values.append(
                torch.gather(
                    value,
                    1,
                    absolute_indices.unsqueeze(-1).expand(
                        batch,
                        self.landmarks_per_slot,
                        self.width,
                    ),
                )
            )
            selected_scores.append(top_scores.to(value.dtype))
            selected_positions.append(torch.gather(positions, 1, absolute_indices))
        return (
            torch.stack(selected_values, dim=1),
            torch.stack(selected_scores, dim=1),
            torch.stack(selected_positions, dim=1),
        )

    def _merge_landmark_bank(
        self,
        existing_landmarks: Tensor,
        existing_scores: Tensor,
        existing_positions: Tensor,
        incoming_landmarks: Tensor,
        incoming_scores: Tensor,
        incoming_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        candidates = torch.cat((existing_landmarks, incoming_landmarks), dim=2)
        candidate_scores = torch.cat(
            (existing_scores, incoming_scores),
            dim=2,
        )
        candidate_positions = torch.cat(
            (existing_positions, incoming_positions),
            dim=2,
        )
        tie_break = candidate_positions.float()
        tie_break = tie_break / tie_break.amax(
            dim=2,
            keepdim=True,
        ).clamp_min(1.0)
        ranked_scores = candidate_scores.float() + 1e-6 * tie_break
        _, selected_indices = torch.topk(
            ranked_scores,
            self.landmarks_per_slot,
            dim=2,
        )
        selected_landmarks = torch.gather(
            candidates,
            2,
            selected_indices.unsqueeze(-1).expand(
                candidates.shape[0],
                self.slots,
                self.landmarks_per_slot,
                self.width,
            ),
        )
        return (
            selected_landmarks,
            torch.gather(candidate_scores, 2, selected_indices),
            torch.gather(candidate_positions, 2, selected_indices),
        )

    def _merge_completed_chunk(
        self,
        numerator: Tensor,
        mass: Tensor,
        landmarks: Tensor,
        landmark_scores: Tensor,
        landmark_positions: Tensor,
        hierarchy_numerators: list[Tensor | None],
        hierarchy_masses: list[Tensor | None],
        hierarchy_landmarks: list[Tensor | None],
        hierarchy_landmark_scores: list[Tensor | None],
        hierarchy_landmark_positions: list[Tensor | None],
    ) -> None:
        carry_numerator = numerator
        carry_mass = mass
        carry_landmarks = landmarks
        carry_landmark_scores = landmark_scores
        carry_landmark_positions = landmark_positions
        for level in range(self.max_levels):
            existing_numerator = hierarchy_numerators[level]
            existing_mass = hierarchy_masses[level]
            existing_landmarks = hierarchy_landmarks[level]
            existing_landmark_scores = hierarchy_landmark_scores[level]
            existing_landmark_positions = hierarchy_landmark_positions[level]
            occupied = (
                existing_numerator is not None
                and existing_mass is not None
                and existing_landmarks is not None
                and existing_landmark_scores is not None
                and existing_landmark_positions is not None
            )
            if not occupied:
                if any(
                    item is not None
                    for item in (
                        existing_numerator,
                        existing_mass,
                        existing_landmarks,
                        existing_landmark_scores,
                        existing_landmark_positions,
                    )
                ):
                    raise ValueError("persistence hierarchy is internally inconsistent")
                hierarchy_numerators[level] = carry_numerator
                hierarchy_masses[level] = carry_mass
                hierarchy_landmarks[level] = carry_landmarks
                hierarchy_landmark_scores[level] = carry_landmark_scores
                hierarchy_landmark_positions[level] = carry_landmark_positions
                return
            assert existing_numerator is not None
            assert existing_mass is not None
            assert existing_landmarks is not None
            assert existing_landmark_scores is not None
            assert existing_landmark_positions is not None
            carry_numerator = existing_numerator + carry_numerator
            carry_mass = existing_mass + carry_mass
            (
                carry_landmarks,
                carry_landmark_scores,
                carry_landmark_positions,
            ) = self._merge_landmark_bank(
                existing_landmarks,
                existing_landmark_scores,
                existing_landmark_positions,
                carry_landmarks,
                carry_landmark_scores,
                carry_landmark_positions,
            )
            hierarchy_numerators[level] = None
            hierarchy_masses[level] = None
            hierarchy_landmarks[level] = None
            hierarchy_landmark_scores[level] = None
            hierarchy_landmark_positions[level] = None
        raise ValueError("persistence hierarchy exceeded configured context")

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _, time, _ = value.shape
        hierarchy_numerators: list[Tensor | None] = [None] * self.max_levels
        hierarchy_masses: list[Tensor | None] = [None] * self.max_levels
        hierarchy_landmarks: list[Tensor | None] = [None] * self.max_levels
        hierarchy_landmark_scores: list[Tensor | None] = [None] * self.max_levels
        hierarchy_landmark_positions: list[Tensor | None] = [None] * self.max_levels
        summaries: list[Tensor] = []
        validity: list[Tensor] = []
        positions: list[Tensor] = []
        for start in range(0, time, self.chunk_size):
            chunk = value[:, start : start + self.chunk_size]
            summary, valid, memory_positions = self._recognized_memory(
                chunk,
                hierarchy_numerators,
                hierarchy_masses,
                hierarchy_landmarks,
                hierarchy_landmark_scores,
                hierarchy_landmark_positions,
            )
            summaries.append(summary)
            validity.append(valid)
            positions.append(memory_positions)
            if chunk.shape[1] == self.chunk_size:
                gate = torch.sigmoid(self.persistence_gate(chunk))
                numerator = torch.einsum("bts,btd->bsd", gate, chunk)
                mass = gate.sum(dim=1)
                landmarks, landmark_scores, landmark_positions = self._chunk_landmarks(
                    chunk,
                    gate,
                    torch.arange(
                        start,
                        start + self.chunk_size,
                        device=value.device,
                    ),
                )
                self._merge_completed_chunk(
                    numerator,
                    mass,
                    landmarks,
                    landmark_scores,
                    landmark_positions,
                    hierarchy_numerators,
                    hierarchy_masses,
                    hierarchy_landmarks,
                    hierarchy_landmark_scores,
                    hierarchy_landmark_positions,
                )
        return (
            torch.cat(summaries, dim=1),
            torch.cat(validity, dim=1),
            torch.cat(positions, dim=1),
        )

    def step_summary(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = value.shape[0]
        gate = torch.sigmoid(self.persistence_gate(value))
        if state.persistence_levels != self.max_levels:
            raise ValueError("generation state persistence hierarchy is incompatible")
        if state.current_numerator is None:
            zeros = value.new_zeros((batch, self.slots, self.width))
            state.current_numerator = zeros.clone()
            state.current_mass = value.new_zeros((batch, self.slots))
        summary, valid, positions = self._recognized_memory(
            value.unsqueeze(1),
            state.hierarchy_numerators,
            state.hierarchy_masses,
            state.hierarchy_landmarks,
            state.hierarchy_landmark_scores,
            state.hierarchy_landmark_positions,
        )
        return summary[:, 0], valid[:, 0], positions[:, 0], gate

    def commit_step(
        self,
        value: Tensor,
        gate: Tensor,
        state: AtomGraphStepState,
        *,
        position: int | None = None,
    ) -> None:
        assert state.current_numerator is not None
        assert state.current_mass is not None
        state.current_numerator = state.current_numerator + gate.unsqueeze(
            -1
        ) * value.unsqueeze(1)
        state.current_mass = state.current_mass + gate
        state.current_values.append(value)
        state.current_gates.append(gate)
        state.current_positions.append(
            torch.full(
                (value.shape[0],),
                state.position if position is None else position,
                dtype=torch.long,
                device=value.device,
            )
        )
        state.current_count += 1
        if state.current_count == self.chunk_size:
            chunk = torch.stack(state.current_values, dim=1)
            chunk_gate = torch.stack(state.current_gates, dim=1)
            chunk_positions = torch.stack(state.current_positions, dim=1)
            landmarks, landmark_scores, landmark_positions = self._chunk_landmarks(
                chunk,
                chunk_gate,
                chunk_positions,
            )
            self._merge_completed_chunk(
                state.current_numerator,
                state.current_mass,
                landmarks,
                landmark_scores,
                landmark_positions,
                state.hierarchy_numerators,
                state.hierarchy_masses,
                state.hierarchy_landmarks,
                state.hierarchy_landmark_scores,
                state.hierarchy_landmark_positions,
            )
            state.current_numerator = torch.zeros_like(state.current_numerator)
            state.current_mass = torch.zeros_like(state.current_mass)
            state.current_values.clear()
            state.current_gates.clear()
            state.current_positions.clear()
            state.current_count = 0

    def chunk_summary(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if value.shape[1] > self.chunk_size - state.current_count:
            raise ValueError("persistence chunk crosses a landmark boundary")
        if state.persistence_levels != self.max_levels:
            raise ValueError("generation state persistence hierarchy is incompatible")
        if state.current_numerator is None:
            batch = value.shape[0]
            state.current_numerator = value.new_zeros((batch, self.slots, self.width))
            state.current_mass = value.new_zeros((batch, self.slots))
        summary, valid, positions = self._recognized_memory(
            value,
            state.hierarchy_numerators,
            state.hierarchy_masses,
            state.hierarchy_landmarks,
            state.hierarchy_landmark_scores,
            state.hierarchy_landmark_positions,
        )
        gate = torch.sigmoid(self.persistence_gate(value))
        return summary, valid, positions, gate

    def commit_chunk(
        self,
        value: Tensor,
        gate: Tensor,
        state: AtomGraphStepState,
    ) -> None:
        for index in range(value.shape[1]):
            self.commit_step(
                value[:, index],
                gate[:, index],
                state,
                position=state.position + index,
            )


class CausalAtomGraph(nn.Module):
    """Sparse causal graph propagation with homeostatic edge entropy."""

    def __init__(self, config: AtomEnglishConfig) -> None:
        super().__init__()
        self.config = config
        self.offsets = config.graph_offsets
        width = config.d_model
        heads = config.n_heads
        self.qkv_projection = nn.Linear(width, width * 3, bias=False)
        self.out_projection = nn.Linear(width, width, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(heads, len(self.offsets)))
        self.memory_bias = nn.Parameter(torch.zeros(heads, config.persistence_slots))
        self.decay_rate = nn.Parameter(torch.full((heads,), -2.0))
        self.temperature_raw = nn.Parameter(torch.full((heads,), 0.3))
        self.attraction_gate = nn.Linear(width, heads, bias=True)
        self.dissipation_gate = nn.Linear(width, width, bias=True)
        self.nucleation_gate = nn.Linear(width, width, bias=True)
        self.radiation_gain = nn.Parameter(torch.zeros(width))
        self.persistence = TopologicalPersistence(
            width,
            config.persistence_slots,
            config.persistence_landmarks_per_slot,
            config.persistence_chunk,
            config.max_seq_len,
        )
        self.dropout = nn.Dropout(config.dropout)

    def _project(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = value.shape
        projected = self.qkv_projection(value).view(
            batch,
            time,
            3,
            self.config.n_heads,
            self.config.head_dim,
        )
        projected = projected.permute(2, 0, 3, 1, 4)
        return projected[0], projected[1], projected[2]

    def _memory_project(
        self,
        summary: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, time, slots, _ = summary.shape
        projected = self.qkv_projection(summary).view(
            batch,
            time,
            slots,
            3,
            self.config.n_heads,
            self.config.head_dim,
        )
        key = projected[:, :, :, 1].permute(0, 3, 1, 2, 4)
        value = projected[:, :, :, 2].permute(0, 3, 1, 2, 4)
        key = _apply_rope_candidates(
            key,
            positions,
            self.config.rope_base,
            self.config.rope_native_context,
        )
        return key, value

    def _memory_candidate_bias(self) -> Tensor:
        landmark_bias = self.memory_bias.repeat_interleave(
            self.config.persistence_landmarks_per_slot,
            dim=-1,
        )
        return torch.cat((self.memory_bias, landmark_bias), dim=-1)

    def _temperature(self) -> Tensor:
        return (F.softplus(self.temperature_raw) + 0.25).clamp(0.3, 2.5)

    def _select_edges(
        self,
        scores: Tensor,
        candidates: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, AtomGraphDiagnostics]:
        batch, heads, time, candidate_count = scores.shape
        expanded_mask = mask.expand(batch, heads, time, candidate_count)
        temperature = self._temperature().view(1, heads, 1, 1)
        annealed = (scores / temperature).masked_fill(~expanded_mask, -1.0e4)
        selected_count = min(self.config.graph_neighbors, candidate_count)
        selected_scores, selected_indices = torch.topk(annealed, selected_count, dim=-1)
        selected_mask = torch.gather(expanded_mask, -1, selected_indices)
        selected_values = torch.gather(
            candidates,
            3,
            selected_indices.unsqueeze(-1).expand(
                batch,
                heads,
                time,
                selected_count,
                self.config.head_dim,
            ),
        )
        weights = torch.softmax(selected_scores.float(), dim=-1)
        weights = weights * selected_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mixed = (weights.to(selected_values.dtype).unsqueeze(-1) * selected_values).sum(
            dim=3
        )
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1)
        entropy = entropy / math.log(max(selected_count, 2))
        diagnostics = AtomGraphDiagnostics(
            edge_entropy=entropy.mean(),
            active_edges=selected_mask.float().sum(dim=-1).mean(),
            temperature=self._temperature().mean(),
        )
        return mixed, diagnostics

    def forward(self, value: Tensor) -> tuple[Tensor, AtomGraphDiagnostics]:
        batch, time, width = value.shape
        if time > self.config.max_seq_len:
            raise ValueError("token sequence exceeds Atom graph context")
        query, key, projected_value = self._project(value)
        positions = torch.arange(time, device=value.device)
        query = _apply_rope(
            query,
            positions,
            self.config.rope_base,
            self.config.rope_native_context,
        )
        key = _apply_rope(
            key,
            positions,
            self.config.rope_base,
            self.config.rope_native_context,
        )

        local_keys: list[Tensor] = []
        local_values: list[Tensor] = []
        local_masks: list[Tensor] = []
        for offset in self.offsets:
            local_keys.append(F.pad(key, (0, 0, offset, 0))[:, :, :time])
            local_values.append(F.pad(projected_value, (0, 0, offset, 0))[:, :, :time])
            local_masks.append(positions >= offset)
        candidate_key = torch.stack(local_keys, dim=3)
        candidate_value = torch.stack(local_values, dim=3)
        local_mask = torch.stack(local_masks, dim=-1)[None, None]

        summary, memory_valid, memory_positions = self.persistence(value)
        memory_key, memory_value = self._memory_project(
            summary,
            memory_positions,
        )
        candidate_key = torch.cat((candidate_key, memory_key), dim=3)
        candidate_value = torch.cat((candidate_value, memory_value), dim=3)
        mask = torch.cat(
            (
                local_mask.expand(batch, 1, time, -1),
                memory_valid[:, None],
            ),
            dim=-1,
        )

        scores = torch.einsum("bhtd,bhtcd->bhtc", query, candidate_key) / math.sqrt(
            self.config.head_dim
        )
        distance = torch.tensor(
            self.offsets,
            dtype=scores.dtype,
            device=scores.device,
        )
        decay = F.softplus(self.decay_rate).view(1, -1, 1, 1)
        local_decay = -decay * torch.log1p(distance).view(1, 1, 1, -1)
        local_bias = self.relation_bias.view(1, self.config.n_heads, 1, -1)
        memory_bias = self._memory_candidate_bias().view(
            1,
            self.config.n_heads,
            1,
            -1,
        )
        scores = scores + torch.cat(
            (
                local_bias + local_decay,
                memory_bias.expand(1, -1, 1, -1),
            ),
            dim=-1,
        )
        mixed, diagnostics = self._select_edges(scores, candidate_value, mask)
        attraction = torch.tanh(self.attraction_gate(value))
        attraction = attraction.permute(0, 2, 1).unsqueeze(-1)
        mixed = mixed * attraction
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, time, width)
        update = self.out_projection(mixed)
        radiation = torch.sigmoid(self.radiation_gain)
        retention = torch.sigmoid(self.dissipation_gate(value))
        nucleation = torch.sigmoid(self.nucleation_gate(value))
        update = update * radiation * retention * (0.5 + nucleation)
        return self.dropout(update), diagnostics

    def forward_step(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, AtomGraphDiagnostics]:
        batch, width = value.shape
        if state.position >= self.config.max_seq_len:
            raise ValueError("generation exceeded Atom graph context")
        projected = self.qkv_projection(value).view(
            batch,
            3,
            self.config.n_heads,
            self.config.head_dim,
        )
        query = projected[:, 0]
        key = projected[:, 1]
        projected_value = projected[:, 2]
        query = _apply_rope_step(
            query,
            state.position,
            self.config.rope_base,
            self.config.rope_native_context,
        )
        key = _apply_rope_step(
            key,
            state.position,
            self.config.rope_base,
            self.config.rope_native_context,
        )

        keys: list[Tensor] = []
        values: list[Tensor] = []
        masks: list[bool] = []
        zero = key.new_zeros(key.shape)
        for offset in self.offsets:
            available = len(state.rotated_keys) >= offset
            keys.append(state.rotated_keys[-offset] if available else zero)
            values.append(state.values[-offset] if available else zero)
            masks.append(available)

        (
            summary,
            memory_valid,
            memory_positions,
            persistence_gate,
        ) = self.persistence.step_summary(value, state)
        memory_projected = self.qkv_projection(summary).view(
            batch,
            summary.shape[1],
            3,
            self.config.n_heads,
            self.config.head_dim,
        )
        memory_key = memory_projected[:, :, 1].permute(0, 2, 1, 3)
        memory_value = memory_projected[:, :, 2].permute(0, 2, 1, 3)
        memory_key = _apply_rope_candidates(
            memory_key.unsqueeze(2),
            memory_positions.unsqueeze(1),
            self.config.rope_base,
            self.config.rope_native_context,
        ).squeeze(2)
        candidate_key = torch.cat((torch.stack(keys, dim=2), memory_key), dim=2)
        candidate_value = torch.cat((torch.stack(values, dim=2), memory_value), dim=2)
        local_mask = torch.tensor(masks, device=value.device)[None, None, :]
        mask = torch.cat(
            (
                local_mask.expand(batch, 1, -1),
                memory_valid[:, None],
            ),
            dim=-1,
        )

        scores = torch.einsum("bhd,bhcd->bhc", query, candidate_key) / math.sqrt(
            self.config.head_dim
        )
        distance = torch.tensor(
            self.offsets,
            dtype=scores.dtype,
            device=scores.device,
        )
        decay = F.softplus(self.decay_rate).view(1, -1, 1)
        local_bias = self.relation_bias[None] - decay * torch.log1p(distance).view(
            1, 1, -1
        )
        local_bias = local_bias.expand(batch, -1, -1)
        memory_bias = self._memory_candidate_bias()[None].expand(
            batch,
            -1,
            -1,
        )
        scores = scores + torch.cat((local_bias, memory_bias), dim=-1)
        expanded_mask = mask.expand(batch, self.config.n_heads, -1)
        temperature = self._temperature().view(1, -1, 1)
        annealed = (scores / temperature).masked_fill(~expanded_mask, -1.0e4)
        count = min(self.config.graph_neighbors, annealed.shape[-1])
        selected_scores, selected_indices = torch.topk(annealed, count, dim=-1)
        selected_mask = torch.gather(expanded_mask, -1, selected_indices)
        selected_values = torch.gather(
            candidate_value,
            2,
            selected_indices.unsqueeze(-1).expand(
                batch,
                self.config.n_heads,
                count,
                self.config.head_dim,
            ),
        )
        weights = torch.softmax(selected_scores.float(), dim=-1)
        weights = weights * selected_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        mixed = (weights.to(selected_values.dtype).unsqueeze(-1) * selected_values).sum(
            dim=2
        )
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1) / math.log(
            max(count, 2)
        )
        diagnostics = AtomGraphDiagnostics(
            edge_entropy=entropy.mean(),
            active_edges=selected_mask.float().sum(dim=-1).mean(),
            temperature=self._temperature().mean(),
        )
        attraction = torch.tanh(self.attraction_gate(value)).unsqueeze(-1)
        mixed = mixed * attraction
        update = self.out_projection(mixed.reshape(batch, width))
        radiation = torch.sigmoid(self.radiation_gain)
        retention = torch.sigmoid(self.dissipation_gate(value))
        nucleation = torch.sigmoid(self.nucleation_gate(value))
        update = update * radiation * retention * (0.5 + nucleation)

        self.persistence.commit_step(value, persistence_gate, state)
        state.rotated_keys.append(key)
        state.values.append(projected_value)
        state.position += 1
        return self.dropout(update), diagnostics

    def forward_chunk(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, AtomGraphDiagnostics]:
        batch, time, width = value.shape
        if time < 1:
            raise ValueError("graph chunk cannot be empty")
        if state.position + time > self.config.max_seq_len:
            raise ValueError("graph chunk exceeds Atom graph context")
        if time > self.config.persistence_chunk - state.current_count:
            raise ValueError("graph chunk crosses a persistence boundary")
        query, key, projected_value = self._project(value)
        positions = torch.arange(
            state.position,
            state.position + time,
            device=value.device,
        )
        query = _apply_rope(
            query,
            positions,
            self.config.rope_base,
            self.config.rope_native_context,
        )
        key = _apply_rope(
            key,
            positions,
            self.config.rope_base,
            self.config.rope_native_context,
        )

        history_length = len(state.rotated_keys)
        if history_length:
            history_key = torch.stack(tuple(state.rotated_keys), dim=2)
            history_value = torch.stack(tuple(state.values), dim=2)
            combined_key = torch.cat((history_key, key), dim=2)
            combined_value = torch.cat(
                (history_value, projected_value),
                dim=2,
            )
        else:
            combined_key = key
            combined_value = projected_value
        relative_positions = history_length + torch.arange(
            time,
            device=value.device,
        )
        local_keys: list[Tensor] = []
        local_values: list[Tensor] = []
        local_masks: list[Tensor] = []
        for offset in self.offsets:
            indices = relative_positions - offset
            valid = indices >= 0
            safe_indices = indices.clamp_min(0)
            local_keys.append(combined_key[:, :, safe_indices])
            local_values.append(combined_value[:, :, safe_indices])
            local_masks.append(valid)
        candidate_key = torch.stack(local_keys, dim=3)
        candidate_value = torch.stack(local_values, dim=3)
        local_mask = torch.stack(local_masks, dim=-1)[None, None]

        (
            summary,
            memory_valid,
            memory_positions,
            persistence_gate,
        ) = self.persistence.chunk_summary(value, state)
        memory_key, memory_value = self._memory_project(
            summary,
            memory_positions,
        )
        candidate_key = torch.cat((candidate_key, memory_key), dim=3)
        candidate_value = torch.cat((candidate_value, memory_value), dim=3)
        mask = torch.cat(
            (
                local_mask.expand(batch, 1, time, -1),
                memory_valid[:, None],
            ),
            dim=-1,
        )
        scores = torch.einsum(
            "bhtd,bhtcd->bhtc",
            query,
            candidate_key,
        ) / math.sqrt(self.config.head_dim)
        distance = torch.tensor(
            self.offsets,
            dtype=scores.dtype,
            device=scores.device,
        )
        decay = F.softplus(self.decay_rate).view(1, -1, 1, 1)
        local_decay = -decay * torch.log1p(distance).view(1, 1, 1, -1)
        local_bias = self.relation_bias.view(1, self.config.n_heads, 1, -1)
        memory_bias = self._memory_candidate_bias().view(
            1,
            self.config.n_heads,
            1,
            -1,
        )
        scores = scores + torch.cat(
            (
                local_bias + local_decay,
                memory_bias.expand(1, -1, 1, -1),
            ),
            dim=-1,
        )
        mixed, diagnostics = self._select_edges(scores, candidate_value, mask)
        attraction = torch.tanh(self.attraction_gate(value))
        attraction = attraction.permute(0, 2, 1).unsqueeze(-1)
        mixed = mixed * attraction
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, time, width)
        update = self.out_projection(mixed)
        radiation = torch.sigmoid(self.radiation_gain)
        retention = torch.sigmoid(self.dissipation_gate(value))
        nucleation = torch.sigmoid(self.nucleation_gate(value))
        update = update * radiation * retention * (0.5 + nucleation)

        self.persistence.commit_chunk(
            value,
            persistence_gate,
            state,
        )
        for index in range(time):
            state.rotated_keys.append(key[:, :, index])
            state.values.append(projected_value[:, :, index])
        state.position += time
        return self.dropout(update), diagnostics


class AtomCausalLanguageBlock(nn.Module):
    def __init__(self, config: AtomEnglishConfig) -> None:
        super().__init__()
        self.graph_norm = RMSNorm(config.d_model)
        self.graph = CausalAtomGraph(config)
        self.graph_residual = ConservativeResidual(
            config.d_model, config.update_ratio_limit
        )
        self.phase_norm = RMSNorm(config.d_model)
        self.phase_mixer = PhaseMixer(
            config.d_model,
            config.ffn_multiplier,
            config.dropout,
        )
        self.phase_residual = ConservativeResidual(
            config.d_model, config.update_ratio_limit
        )

    def forward(self, value: Tensor) -> tuple[Tensor, AtomGraphDiagnostics]:
        graph_update, diagnostics = self.graph(self.graph_norm(value))
        value = self.graph_residual(value, graph_update)
        phase_update = self.phase_mixer(self.phase_norm(value))
        value = self.phase_residual(value, phase_update)
        return value, diagnostics

    def forward_chunk(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, AtomGraphDiagnostics]:
        graph_update, diagnostics = self.graph.forward_chunk(
            self.graph_norm(value),
            state,
        )
        value = self.graph_residual(value, graph_update)
        phase_update = self.phase_mixer(self.phase_norm(value))
        value = self.phase_residual(value, phase_update)
        return value, diagnostics

    def forward_step(
        self,
        value: Tensor,
        state: AtomGraphStepState,
    ) -> tuple[Tensor, AtomGraphDiagnostics]:
        graph_update, diagnostics = self.graph.forward_step(
            self.graph_norm(value), state
        )
        value = self.graph_residual(value, graph_update)
        phase_update = self.phase_mixer(self.phase_norm(value))
        value = self.phase_residual(value, phase_update)
        return value, diagnostics


@dataclass(frozen=True)
class AtomLanguageOutput:
    logits: Tensor
    loss: Tensor | None
    cross_entropy: Tensor | None
    criticality_loss: Tensor
    mean_edge_entropy: Tensor
    mean_active_edges: Tensor
    mean_temperature: Tensor


class AtomCausalLanguageModel(nn.Module):
    """Autoregressive language model with a causal graph sequence engine."""

    def __init__(self, config: AtomEnglishConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            AtomCausalLanguageBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for block in self.blocks:
            block.graph.out_projection.weight.data.mul_(residual_scale)
            block.phase_mixer.out_projection.weight.data.mul_(residual_scale)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _symbolic_transition_key(
        history: Sequence[int],
        order: int,
    ) -> int:
        mask = (1 << 64) - 1
        value = 0xCBF29CE484222325
        for token in history[-order:]:
            value ^= (int(token) + 0x9E3779B97F4A7C15) & mask
            value = (value * 0x100000001B3) & mask
            value ^= value >> 32
        return (order << 64) | value

    def _ensure_symbolic_state(
        self,
        state: AtomGraphStepState,
        batch: int,
    ) -> None:
        if not state.symbolic_histories and not state.symbolic_transitions:
            state.symbolic_histories = [[] for _ in range(batch)]
            state.symbolic_transitions = [{} for _ in range(batch)]
            return
        if (
            len(state.symbolic_histories) != batch
            or len(state.symbolic_transitions) != batch
        ):
            raise ValueError("symbolic transition state batch changed")

    @staticmethod
    def _record_symbolic_transition(
        table: dict[int, int | dict[int, int]],
        key: int,
        token: int,
    ) -> None:
        existing = table.get(key)
        if existing is None:
            table[key] = token
        elif isinstance(existing, int):
            if existing != token:
                table[key] = {existing: 1, token: 1}
        else:
            existing[token] = existing.get(token, 0) + 1

    @staticmethod
    def _symbolic_votes(
        value: int | dict[int, int],
    ) -> dict[int, int]:
        return {value: 1} if isinstance(value, int) else value

    def _observe_symbolic_transitions(
        self,
        input_ids: Tensor,
        state: AtomGraphStepState,
    ) -> list[list[dict[int, int] | None]]:
        batch, time = input_ids.shape
        self._ensure_symbolic_state(state, batch)
        token_rows = input_ids.detach().to("cpu").tolist()
        observations: list[list[dict[int, int] | None]] = []
        maximum_entries = self.config.max_seq_len * len(
            self.config.symbolic_copy_orders
        )
        for batch_index in range(batch):
            history = state.symbolic_histories[batch_index]
            table = state.symbolic_transitions[batch_index]
            row: list[dict[int, int] | None] = []
            for raw_token in token_rows[batch_index]:
                token = int(raw_token)
                for order in self.config.symbolic_copy_orders:
                    if len(history) < order:
                        continue
                    key = self._symbolic_transition_key(history, order)
                    if key in table or len(table) < maximum_entries:
                        self._record_symbolic_transition(
                            table,
                            key,
                            token,
                        )
                history.append(token)
                votes: dict[int, int] | None = None
                for order in reversed(self.config.symbolic_copy_orders):
                    if len(history) < order:
                        continue
                    key = self._symbolic_transition_key(history, order)
                    observed = table.get(key)
                    if observed is not None:
                        votes = self._symbolic_votes(observed)
                        break
                row.append(votes)
            if len(row) != time:
                raise AssertionError("symbolic observation length changed")
            observations.append(row)
        return observations

    def _apply_symbolic_copy(
        self,
        logits: Tensor,
        input_ids: Tensor,
        state: AtomGraphStepState,
    ) -> Tensor:
        observations = self._observe_symbolic_transitions(input_ids, state)
        bonus = torch.zeros_like(logits)
        if logits.ndim == 2:
            maximum_logits = logits.amax(dim=-1)
            selected = [
                observations[batch_index][-1]
                for batch_index in range(input_ids.shape[0])
            ]
            for batch_index, votes in enumerate(selected):
                if not votes:
                    continue
                total = max(sum(votes.values()), 1)
                for token, count in votes.items():
                    if 0 <= token < self.config.vocab_size:
                        margin = (
                            maximum_logits[batch_index]
                            - logits[
                                batch_index,
                                token,
                            ]
                        )
                        proposed = logits.new_tensor(
                            self.config.symbolic_copy_logit_gain
                            + math.log(count / total)
                        )
                        bonus[batch_index, token] = torch.where(
                            margin <= self.config.symbolic_copy_neural_margin,
                            proposed,
                            proposed.new_zeros(()),
                        )
        elif logits.ndim == 3:
            if logits.shape[:2] != input_ids.shape:
                raise ValueError("symbolic copy logits do not match input tokens")
            maximum_logits = logits.amax(dim=-1)
            for batch_index, row in enumerate(observations):
                for time_index, votes in enumerate(row):
                    if not votes:
                        continue
                    total = max(sum(votes.values()), 1)
                    for token, count in votes.items():
                        if 0 <= token < self.config.vocab_size:
                            margin = (
                                maximum_logits[
                                    batch_index,
                                    time_index,
                                ]
                                - logits[
                                    batch_index,
                                    time_index,
                                    token,
                                ]
                            )
                            proposed = logits.new_tensor(
                                self.config.symbolic_copy_logit_gain
                                + math.log(count / total)
                            )
                            bonus[
                                batch_index,
                                time_index,
                                token,
                            ] = torch.where(
                                margin <= self.config.symbolic_copy_neural_margin,
                                proposed,
                                proposed.new_zeros(()),
                            )
        else:
            raise ValueError("symbolic copy logits have an invalid rank")
        return logits + bonus

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> AtomLanguageOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input_ids exceed the configured context")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        value = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        value = self.embedding_dropout(value)
        diagnostics: list[AtomGraphDiagnostics] = []
        for block in self.blocks:
            value, layer_diagnostics = block(value)
            diagnostics.append(layer_diagnostics)
        value = self.final_norm(value)
        logits = F.linear(value, self.token_embedding.weight)
        if not self.training:
            symbolic_state = AtomGraphStepState(
                exact_cache_capacity=1,
                persistence_levels=1,
            )
            logits = self._apply_symbolic_copy(
                logits,
                input_ids,
                symbolic_state,
            )
        entropy = torch.stack([item.edge_entropy for item in diagnostics]).mean()
        active_edges = torch.stack([item.active_edges for item in diagnostics]).mean()
        temperature = torch.stack([item.temperature for item in diagnostics]).mean()
        criticality = (entropy - self.config.criticality_target).square()
        cross_entropy: Tensor | None = None
        loss: Tensor | None = None
        if labels is not None:
            cross_entropy = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss = cross_entropy + self.config.criticality_weight * criticality
        return AtomLanguageOutput(
            logits=logits,
            loss=loss,
            cross_entropy=cross_entropy,
            criticality_loss=criticality,
            mean_edge_entropy=entropy,
            mean_active_edges=active_edges,
            mean_temperature=temperature,
        )

    def initial_generation_state(
        self,
    ) -> list[AtomGraphStepState]:
        return [
            AtomGraphStepState(
                exact_cache_capacity=block.graph.config.exact_cache_tokens,
                persistence_levels=block.graph.persistence.max_levels,
            )
            for block in self.blocks
        ]

    def forward_step(
        self,
        token_ids: Tensor,
        states: Sequence[AtomGraphStepState],
    ) -> tuple[Tensor, tuple[AtomGraphDiagnostics, ...]]:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [batch]")
        if len(states) != len(self.blocks):
            raise ValueError("generation state does not match layer count")
        value = self.token_embedding(token_ids) * math.sqrt(self.config.d_model)
        value = self.embedding_dropout(value)
        diagnostics: list[AtomGraphDiagnostics] = []
        for block, state in zip(self.blocks, states, strict=True):
            value, layer_diagnostics = block.forward_step(value, state)
            diagnostics.append(layer_diagnostics)
        value = self.final_norm(value)
        logits = F.linear(value, self.token_embedding.weight)
        logits = self._apply_symbolic_copy(
            logits,
            token_ids.unsqueeze(1),
            states[0],
        )
        return logits, tuple(diagnostics)

    def prefill(
        self,
        input_ids: Tensor,
        states: Sequence[AtomGraphStepState] | None = None,
    ) -> tuple[
        Tensor,
        list[AtomGraphStepState],
        tuple[AtomGraphDiagnostics, ...],
    ]:
        """Consume a prompt in vectorized landmark chunks and return final logits."""

        logits, active_states, diagnostics = self._stream_input(
            input_ids,
            states,
            collect_all_logits=False,
        )
        return logits, active_states, diagnostics

    def forward_stream(
        self,
        input_ids: Tensor,
        states: Sequence[AtomGraphStepState] | None = None,
    ) -> tuple[
        Tensor,
        list[AtomGraphStepState],
        tuple[AtomGraphDiagnostics, ...],
    ]:
        """Return every logit for a bounded trainable tail over prior graph state."""

        return self._stream_input(
            input_ids,
            states,
            collect_all_logits=True,
        )

    def _stream_input(
        self,
        input_ids: Tensor,
        states: Sequence[AtomGraphStepState] | None,
        *,
        collect_all_logits: bool,
    ) -> tuple[
        Tensor,
        list[AtomGraphStepState],
        tuple[AtomGraphDiagnostics, ...],
    ]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.shape[1] < 1:
            raise ValueError("prefill input cannot be empty")
        active_states = (
            self.initial_generation_state() if states is None else list(states)
        )
        if len(active_states) != len(self.blocks):
            raise ValueError("prefill state does not match layer count")
        positions = {state.position for state in active_states}
        counts = {state.current_count for state in active_states}
        if len(positions) != 1 or len(counts) != 1:
            raise ValueError("prefill layer states are not synchronized")
        start_position = next(iter(positions))
        if start_position + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("prefill exceeds the configured context")

        logit_chunks: list[Tensor] = []
        final_logits: Tensor | None = None
        final_diagnostics: tuple[AtomGraphDiagnostics, ...] = ()
        offset = 0
        while offset < input_ids.shape[1]:
            current_count = active_states[0].current_count
            remaining = self.config.persistence_chunk - current_count
            length = min(remaining, input_ids.shape[1] - offset)
            token_chunk = input_ids[:, offset : offset + length]
            value = self.token_embedding(token_chunk) * math.sqrt(self.config.d_model)
            value = self.embedding_dropout(value)
            diagnostics: list[AtomGraphDiagnostics] = []
            for block, state in zip(
                self.blocks,
                active_states,
                strict=True,
            ):
                value, layer_diagnostics = block.forward_chunk(value, state)
                diagnostics.append(layer_diagnostics)
            normalized = self.final_norm(value if collect_all_logits else value[:, -1])
            chunk_logits = F.linear(normalized, self.token_embedding.weight)
            chunk_logits = self._apply_symbolic_copy(
                chunk_logits,
                token_chunk,
                active_states[0],
            )
            if collect_all_logits:
                logit_chunks.append(chunk_logits)
            else:
                final_logits = chunk_logits
            final_diagnostics = tuple(diagnostics)
            offset += length
        if collect_all_logits:
            final_logits = torch.cat(logit_chunks, dim=1)
        assert final_logits is not None
        return final_logits, active_states, final_diagnostics

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        eos_token_id: int | None = None,
        seed: int = 0,
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("cached generation currently accepts one prompt")
        if input_ids.shape[1] < 1:
            raise ValueError("generation prompt cannot be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("requested generation exceeds model context")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be inside (0, 1]")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1")
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)
        logits, states, _ = self.prefill(input_ids)
        generated = input_ids.clone()
        stop_id = (
            self.config.eos_token_id if eos_token_id is None else int(eos_token_id)
        )
        for step in range(max_new_tokens):
            next_logits = logits[0].float().clone()
            if repetition_penalty > 1.0:
                used = torch.unique(generated[0])
                used_logits = next_logits[used]
                next_logits[used] = torch.where(
                    used_logits < 0,
                    used_logits * repetition_penalty,
                    used_logits / repetition_penalty,
                )
            next_logits = next_logits / temperature
            keep = min(top_k, next_logits.shape[-1])
            top_values, top_indices = torch.topk(next_logits, keep)
            probabilities = torch.softmax(top_values, dim=-1)
            sorted_probabilities, order = probabilities.sort(descending=True)
            cumulative = sorted_probabilities.cumsum(dim=-1)
            remove = cumulative - sorted_probabilities >= top_p
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
            sorted_probabilities = sorted_probabilities / (
                sorted_probabilities.sum().clamp_min(1e-9)
            )
            sampled_order = torch.multinomial(
                sorted_probabilities,
                num_samples=1,
                generator=generator,
            )
            selected = order[sampled_order]
            next_token = top_indices[selected].view(1)
            generated = torch.cat((generated, next_token.view(1, 1)), dim=1)
            if int(next_token.item()) == stop_id:
                break
            if step + 1 < max_new_tokens:
                logits, _ = self.forward_step(next_token, states)
        return generated


def atom_english_architecture_manifest(
    config: AtomEnglishConfig,
    model: AtomCausalLanguageModel | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_CORE_RUNTIME,
        "config": config.to_dict(),
        "parameter_count": (model.parameter_count() if model is not None else None),
        "sequence_engine": "sparse-directed-temporal-causal-graph",
        "training_execution": "parallel-causal-blocks",
        "generation_execution": "recurrent-cached-graph-state",
        "context_system": {
            "maximum_tokens": config.max_seq_len,
            "native_phase_tokens": config.rope_native_context,
            "exact_recent_tokens": config.exact_cache_tokens,
            "persistent_landmark_levels": config.persistence_levels,
            "persistent_summary_slots": config.persistence_slots,
            "episodic_landmarks_per_slot": (config.persistence_landmarks_per_slot),
            "episodic_candidates_per_level": (
                config.persistence_slots * config.persistence_landmarks_per_slot
            ),
            "landmark_selection": (
                "position-partitioned learned salience with exact ordered values"
            ),
            "persistent_memory_growth": "logarithmic",
            "symbolic_transition_orders": list(config.symbolic_copy_orders),
            "symbolic_transition_capacity": (
                config.max_seq_len * len(config.symbolic_copy_orders)
            ),
            "symbolic_transition_growth": (
                "bounded linear in the declared context, never quadratic"
            ),
            "symbolic_copy_neural_margin": (config.symbolic_copy_neural_margin),
            "prefill": "vectorized-landmark-chunks",
            "required_evaluation_lengths": list(ATOM_LONG_CONTEXT_MILESTONES),
        },
        "root_bindings": {
            "radiation": "edge-value propagation",
            "dissipation": "learned retention and bounded forgetting",
            "gravitation": "content-weighted predecessor selection",
            "attraction_repulsion": "signed recognition gate",
            "nucleation": "sparse update and landmark binding",
            "conservation": "bounded residual energy",
            "decay": "distance-dependent edge suppression",
        },
        "composed_mechanisms": {
            "causal_graph": "explicit predecessor and landmark edges",
            "phase_locked_loop": "rotational relative phase",
            "phase_mixer": "amplitude carrier phase composition",
            "molecular_recognition": "annealed top-k edge selection",
            "topological_persistence": "completed-region graph landmarks",
            "symbolic_transition_memory": (
                "observed suffix-to-continuation causal edges"
            ),
            "thermal_annealing": "learned bounded edge temperature",
            "projective_measurement": "tied token projection",
        },
    }


def atom_english_core_self_test() -> dict[str, bool]:
    config = atom_english_profile(
        "verification",
        vocab_size=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(7)
    model = AtomCausalLanguageModel(config).eval()
    tokens = torch.randint(3, config.vocab_size, (2, 24))
    with torch.no_grad():
        parallel = model(tokens)
        states = model.initial_generation_state()
        recurrent: list[Tensor] = []
        for index in range(tokens.shape[1]):
            logits, _ = model.forward_step(tokens[:, index], states)
            recurrent.append(logits)
        recurrent_logits = torch.stack(recurrent, dim=1)
        symbolic_state = AtomGraphStepState(
            exact_cache_capacity=1,
            persistence_levels=1,
        )
        copy_input = torch.tensor(
            [[41, 42, 43, 44, 207, 41, 42, 43, 44]],
            dtype=torch.long,
        )
        copy_logits = torch.zeros((1, config.vocab_size))
        copy_logits[0, 206] = 2.0
        copied = model._apply_symbolic_copy(
            copy_logits,
            copy_input,
            symbolic_state,
        )
    return {
        "all_seven_roots_bound": tuple(
            atom_english_architecture_manifest(config, model)["root_bindings"]
        )
        == ATOM_ROOT_PRIMITIVES,
        "parallel_shape": parallel.logits.shape == (2, 24, config.vocab_size),
        "finite_logits": bool(torch.isfinite(parallel.logits).all()),
        "finite_criticality": bool(torch.isfinite(parallel.criticality_loss)),
        "cached_parallel_equivalence": bool(
            torch.allclose(
                parallel.logits,
                recurrent_logits,
                atol=2e-5,
                rtol=2e-4,
            )
        ),
        "parameter_count_real": model.parameter_count() > config.vocab_size,
        "symbolic_transition_recall": int(copied.argmax(dim=-1).item()) == 207,
    }
