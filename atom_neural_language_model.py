"""Neural language-to-field architecture built from the seven Atom operators."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from atom_field_proof import (
    FIELD_TICKS,
    NODE_COUNT,
    NODE_DIM,
    PROCESS_COUNT,
    PROCESS_NAMES,
    AtomFieldCell,
)
from atom_neural_language_dataset import (
    induce_control_from_consequence,
    tokenize_neural_utterance,
)


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


NEURAL_MODEL_SCHEMA = 2
PAD_TOKEN = "0pad"
TEXT_OPERATOR_NAMES = tuple(PROCESS_NAMES)


def set_neural_deterministic(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def neural_model_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NeuralArchitectureConfig:
    hidden_dim: int = 48
    text_ticks: int = 3
    field_ticks: int = FIELD_TICKS
    maximum_tokens: int = 28
    minimum_temperature: float = 0.20
    maximum_temperature: float = 2.00
    minimum_phase_strength: float = 0.0
    maximum_phase_strength: float = 0.35
    norm_budget: float = 1.0

    def validate(self) -> None:
        if self.hidden_dim < 16 or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and at least 16")
        if not 1 <= self.text_ticks <= 8:
            raise ValueError("text_ticks must be in [1, 8]")
        if not 1 <= self.field_ticks <= 8:
            raise ValueError("field_ticks must be in [1, 8]")
        if not 6 <= self.maximum_tokens <= 64:
            raise ValueError("maximum_tokens must be in [6, 64]")
        if not 0.0 < self.minimum_temperature <= self.maximum_temperature:
            raise ValueError("temperature bounds are invalid")
        if not 0.0 <= self.minimum_phase_strength <= self.maximum_phase_strength:
            raise ValueError("phase bounds are invalid")
        if not 0.1 <= self.norm_budget <= 4.0:
            raise ValueError("norm_budget must be in [0.1, 4.0]")


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """Fail-closed claim policy and fast-path recurrent budget."""

    minimum_operator_support: float = 0.80
    minimum_query_support: float = 0.80
    fast_text_ticks: int = 1

    def validate(self, architecture: NeuralArchitectureConfig) -> None:
        if not 0.5 <= self.minimum_operator_support <= 1.0:
            raise ValueError("minimum_operator_support must be in [0.5, 1.0]")
        if not 0.5 <= self.minimum_query_support <= 1.0:
            raise ValueError("minimum_query_support must be in [0.5, 1.0]")
        if not 1 <= self.fast_text_ticks <= architecture.text_ticks:
            raise ValueError("fast_text_ticks must fit the architecture text budget")


@dataclass(frozen=True)
class NeuralVocabulary:
    tokens: tuple[str, ...]
    responses: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != PAD_TOKEN:
            raise ValueError("tokens must begin with the padding token")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique")
        if not self.responses or len(set(self.responses)) != len(self.responses):
            raise ValueError("responses must be unique and non-empty")
        if any(not token.isascii() or not token.isalnum() for token in self.tokens):
            raise ValueError("vocabulary tokens must be opaque ASCII alphanumerics")
        if any(not token.isascii() or not token.isalnum() for token in self.responses):
            raise ValueError("response tokens must be opaque ASCII alphanumerics")

    @classmethod
    def build(
        cls,
        tokens: Iterable[str],
        responses: Iterable[str],
    ) -> "NeuralVocabulary":
        return cls(
            tokens=(PAD_TOKEN, *tuple(sorted(set(tokens)))),
            responses=tuple(sorted(set(responses))),
        )

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def response_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.responses)}

    def encode(
        self, utterance: str, maximum_tokens: int
    ) -> tuple[list[int], list[bool]]:
        mapping = self.token_to_id
        tokens = tokenize_neural_utterance(utterance)
        if len(tokens) > maximum_tokens:
            raise ValueError("utterance exceeds configured maximum_tokens")
        try:
            ids = [mapping[token] for token in tokens]
        except KeyError as error:
            raise ValueError(f"unknown utterance token: {error.args[0]}") from error
        mask = [True] * len(ids)
        padding = maximum_tokens - len(ids)
        return ids + [0] * padding, mask + [False] * padding


class RuntimeRowDataset(Dataset[Mapping[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def collate_runtime_rows(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
) -> dict[str, Any]:
    token_ids: list[list[int]] = []
    token_masks: list[list[bool]] = []
    response_mapping = vocabulary.response_to_id
    response_ids: list[int] = []
    marker_ids: list[int] = []
    for row in rows:
        encoded, mask = vocabulary.encode(str(row["utterance"]), config.maximum_tokens)
        token_ids.append(encoded)
        token_masks.append(mask)
        response = str(row["response"])
        if response not in response_mapping:
            raise ValueError(f"unknown response token: {response}")
        response_ids.append(response_mapping[response])
        marker_ids.append(encoded[0])
    induced = [induce_control_from_consequence(row) for row in rows]
    return {
        "adjacency": torch.tensor(
            [row["adjacency"] for row in rows], dtype=torch.float32
        ),
        "event_ids": [str(row["event_id"]) for row in rows],
        "induced_controls": torch.tensor(
            [entry["controls"] for entry in induced], dtype=torch.float32
        ),
        "induced_operators": torch.tensor(
            [
                [float(name in entry["signature"]) for name in PROCESS_NAMES]
                for entry in induced
            ],
            dtype=torch.float32,
        ),
        "marker_ids": torch.tensor(marker_ids, dtype=torch.long),
        "node_features": torch.tensor(
            [row["node_features"] for row in rows], dtype=torch.float32
        ),
        "response_ids": torch.tensor(response_ids, dtype=torch.long),
        "salience": torch.tensor(
            [row["salience"] for row in rows], dtype=torch.float32
        ),
        "target_binary": torch.tensor(
            [row["target_binary"] for row in rows], dtype=torch.float32
        ),
        "target_continuous": torch.tensor(
            [row["target_continuous"] for row in rows], dtype=torch.float32
        ),
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "token_mask": torch.tensor(token_masks, dtype=torch.bool),
    }


def make_runtime_loader(
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        RuntimeRowDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_runtime_rows(batch, vocabulary, config),
        num_workers=0,
    )


def move_neural_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


class AtomSequenceCell(nn.Module):
    """Simultaneous language-field update with one branch per root primitive."""

    def __init__(self, hidden_dim: int, norm_budget: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_budget = norm_budget
        self.radiation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gravitation_score = nn.Linear(hidden_dim, 1, bias=False)
        self.gravitation_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.charge = nn.Linear(hidden_dim, 1, bias=False)
        self.attraction_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dissipation_gate = nn.Linear(hidden_dim, hidden_dim)
        self.nucleation_value = nn.Linear(hidden_dim, hidden_dim)
        self.nucleation_score = nn.Linear(hidden_dim, 1)
        self.decay_gate = nn.Linear(hidden_dim, hidden_dim)
        self.log_gains = nn.Parameter(
            torch.log(torch.tensor((0.42, 0.30, 0.24, 0.18, 0.30, 1.0, 0.12)))
        )
        self.phase_frequency = nn.Parameter(torch.linspace(0.4, 1.6, hidden_dim // 2))

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
        masked = logits.masked_fill(~mask, -1e4)
        values = torch.softmax(masked, dim=dim)
        return values * mask.to(values.dtype)

    def _phase_mix(self, values: Tensor, phase_strength: float) -> Tensor:
        if phase_strength <= 0.0:
            return values
        pairs = values.view(*values.shape[:-1], self.hidden_dim // 2, 2)
        angle = phase_strength * self.phase_frequency.view(1, 1, -1)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = torch.stack(
            (cosine * even - sine * odd, sine * even + cosine * odd),
            dim=-1,
        )
        return rotated.reshape_as(values)

    def forward(
        self,
        state: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        phase_strength: float,
        ablate: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if ablate is not None and not 0 <= ablate < PROCESS_COUNT:
            raise ValueError(f"bad text primitive ablation: {ablate}")
        valid = mask.unsqueeze(-1).to(state.dtype)
        prior = state * valid
        gains = self.log_gains.exp()
        if ablate is not None:
            gains = gains.clone()
            gains[ablate] = 0.0

        left = F.pad(prior[:, :-1], (0, 0, 1, 0))
        right = F.pad(prior[:, 1:], (0, 0, 0, 1))
        radiation = self.radiation(0.5 * (left + right)) * valid

        gravity_weights = self._masked_softmax(
            self.gravitation_score(prior).squeeze(-1), mask, dim=1
        )
        attractor = torch.sum(
            gravity_weights.unsqueeze(-1) * self.gravitation_value(prior),
            dim=1,
            keepdim=True,
        )
        gravitation = attractor.expand_as(prior) * valid

        charge = torch.tanh(self.charge(prior)).squeeze(-1) * mask.to(prior.dtype)
        signed = -charge.unsqueeze(2) * charge.unsqueeze(1)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        signed = signed * pair_mask.to(signed.dtype)
        attraction = torch.bmm(signed, self.attraction_value(prior))
        attraction = attraction / mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(
            -1
        )

        dissipation = torch.sigmoid(self.dissipation_gate(prior)) * prior

        threshold = 0.45 + 0.20 / max(temperature, 1e-6)
        probability = torch.sigmoid(
            temperature * (self.nucleation_score(prior).squeeze(-1) - threshold)
        )
        hard = (probability >= 0.5).to(probability.dtype)
        straight_through = hard + probability - probability.detach()
        # Crystallization keeps a small pre-nucleation phase alive.  Without
        # this mixed phase, short English sequences can drive the hard gate to
        # zero everywhere and make the nucleation primitive causally dormant.
        nucleated = 0.92 * straight_through + 0.08 * probability
        nucleation = nucleated.unsqueeze(-1) * torch.tanh(self.nucleation_value(prior))

        positions = torch.linspace(0.0, 1.0, prior.shape[1], device=prior.device).view(
            1, -1, 1
        )
        decay = torch.sigmoid(self.decay_gate(prior)) * positions * prior

        proposals = torch.stack(
            (
                gains[0] * radiation,
                gains[1] * gravitation,
                gains[2] * attraction,
                -gains[3] * dissipation,
                gains[4] * nucleation,
                torch.zeros_like(prior),
                -gains[6] * decay,
            ),
            dim=1,
        )
        mixed = prior + proposals.sum(dim=1)
        mixed = self._phase_mix(mixed, phase_strength)

        prior_mass = prior.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        next_mass = mixed.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        target_mass = prior_mass * self.norm_budget
        conserved = mixed * (target_mass / next_mass)
        next_state = torch.tanh(
            (1.0 - gains[5].clamp(0.0, 1.0)) * mixed
            + gains[5].clamp(0.0, 1.0) * conserved
        )
        next_state = next_state * valid

        branch_energy = proposals.square().mean(dim=(2, 3))
        conservation_energy = (mixed - conserved).square().mean(dim=(1, 2))
        branch_energy[:, 5] = conservation_energy
        route = branch_energy / branch_energy.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return next_state, route


class AtomTextEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        config: NeuralArchitectureConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, config.hidden_dim, padding_idx=0)
        self.position = nn.Parameter(
            torch.zeros(config.maximum_tokens, config.hidden_dim)
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.cell = AtomSequenceCell(config.hidden_dim, config.norm_budget)
        self.pool_score = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        token_ids: Tensor,
        mask: Tensor,
        *,
        temperature: float,
        phase_strength: float,
        ablate: int | None = None,
        tick_budget: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = self.embedding(token_ids) + self.position[
            : token_ids.shape[1]
        ].unsqueeze(0)
        state = state * mask.unsqueeze(-1).to(state.dtype)
        routes: list[Tensor] = []
        ticks = self.config.text_ticks if tick_budget is None else tick_budget
        if not 1 <= ticks <= self.config.text_ticks:
            raise ValueError("text tick budget is outside configured bounds")
        for _ in range(ticks):
            state, route = self.cell(
                state,
                mask,
                temperature=temperature,
                phase_strength=phase_strength,
                ablate=ablate,
            )
            routes.append(route)
        weights = AtomSequenceCell._masked_softmax(
            self.pool_score(state).squeeze(-1), mask, dim=1
        )
        pooled = torch.sum(weights.unsqueeze(-1) * state, dim=1)
        return pooled, state, torch.stack(routes, dim=1).mean(dim=1)


class AtomNeuralLanguageField(nn.Module):
    def __init__(
        self,
        vocabulary: NeuralVocabulary,
        config: NeuralArchitectureConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config or NeuralArchitectureConfig()
        self.config.validate()
        hidden = self.config.hidden_dim
        self.text_encoder = AtomTextEncoder(len(vocabulary.tokens), self.config)
        self.control_lexicon = nn.Embedding(
            len(vocabulary.tokens), PROCESS_COUNT, padding_idx=0
        )
        nn.init.constant_(self.control_lexicon.weight, -4.0)
        with torch.no_grad():
            self.control_lexicon.weight[0].zero_()
        self.control_lexicon.weight.requires_grad_(False)
        self.context_modulator = nn.Linear(hidden, PROCESS_COUNT)
        nn.init.zeros_(self.context_modulator.bias)
        nn.init.normal_(self.context_modulator.weight, mean=0.0, std=0.01)
        self.field_cell = AtomFieldCell()
        with torch.no_grad():
            self.field_cell.log_rate_gain.zero_()
            self.field_cell.log_gravitation_mix.copy_(
                torch.log(torch.tensor((2.0, 1.0)))
            )
            self.field_cell.potential_logits.copy_(
                torch.log(torch.tensor((0.55, 0.25, 0.20)))
            )
            self.field_cell.nucleation_bias.zero_()
            self.field_cell.log_nucleation_temperature.fill_(math.log(100.0))
            self.field_cell.decay_support_logit.copy_(torch.logit(torch.tensor(0.25)))
            self.field_cell.log_decay_temperature.fill_(math.log(100.0))
        for parameter in self.field_cell.parameters():
            parameter.requires_grad_(False)
        self.field_projection = nn.Sequential(
            nn.Linear(NODE_COUNT * 6, hidden),
            nn.GELU(),
        )
        self.query_lexicon = nn.Embedding(len(vocabulary.tokens), 6, padding_idx=0)
        nn.init.constant_(self.query_lexicon.weight, -6.0)
        with torch.no_grad():
            self.query_lexicon.weight[0].zero_()
        self.query_lexicon.weight.requires_grad_(False)
        self.register_buffer(
            "surface_table",
            torch.full(
                (len(vocabulary.tokens), 2, 7),
                -1,
                dtype=torch.long,
            ),
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(vocabulary.responses)),
        )

    @staticmethod
    def _controls(operator_activation: Tensor) -> Tensor:
        radiation = 0.58 * operator_activation[:, 0:1]
        dissipation = 0.48 * operator_activation[:, 1:2]
        gravitation = 0.46 * operator_activation[:, 2:3]
        attraction = 0.58 * operator_activation[:, 3:4]
        nucleation = 0.86 * operator_activation[:, 4:5]
        threshold = 2.4 - (2.4 - 0.72) * operator_activation[:, 4:5]
        conservation = operator_activation[:, 5:6]
        decay = 0.48 * operator_activation[:, 6:7]
        return torch.cat(
            (
                radiation,
                dissipation,
                gravitation,
                attraction,
                nucleation,
                threshold,
                decay,
                conservation,
            ),
            dim=1,
        )

    def forward(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        node_features: Tensor,
        adjacency: Tensor,
        *,
        temperature: float = 0.75,
        phase_strength: float = 0.04,
        text_ablate: int | None = None,
        field_ablate: int | None = None,
        text_tick_budget: int | None = None,
    ) -> dict[str, Tensor]:
        if (
            not self.config.minimum_temperature
            <= temperature
            <= self.config.maximum_temperature
        ):
            raise ValueError("temperature is outside configured bounds")
        if (
            not self.config.minimum_phase_strength
            <= phase_strength
            <= self.config.maximum_phase_strength
        ):
            raise ValueError("phase_strength is outside configured bounds")
        pooled, _, text_route = self.text_encoder(
            token_ids,
            token_mask,
            temperature=temperature,
            phase_strength=phase_strength,
            ablate=text_ablate,
            tick_budget=text_tick_budget,
        )
        token_operator_probability = torch.sigmoid(self.control_lexicon(token_ids))
        token_operator_probability = token_operator_probability * token_mask.unsqueeze(
            -1
        ).to(pooled.dtype)
        operator_activation = 1.0 - torch.prod(
            1.0 - token_operator_probability.clamp(0.0, 1.0), dim=1
        )
        contextual_intensity = 0.98 + 0.04 * torch.sigmoid(
            self.context_modulator(pooled)
        )
        operator_activation = (operator_activation * contextual_intensity).clamp(
            0.0, 1.0
        )
        controls = self._controls(operator_activation)
        state = node_features
        initial_budget = (node_features[..., 1] * node_features[..., 5]).sum(
            dim=1, keepdim=True
        )
        field_routes: list[Tensor] = []
        for _ in range(self.config.field_ticks):
            state, field_route = self.field_cell(
                state,
                adjacency,
                controls,
                initial_budget,
                ablate=field_ablate,
            )
            field_routes.append(field_route.mean(dim=1))
        continuous = state[..., [0, 1, 7, 4]]
        binary_probability = state[..., [5, 6]].clamp(0.0, 1.0)
        binary_logits = 12.0 * (binary_probability - 0.5)
        field_features = torch.cat((continuous, binary_probability), dim=-1).flatten(1)
        field_summary = self.field_projection(field_features)
        response_logits = self.response_head(torch.cat((pooled, field_summary), dim=1))
        query_position = token_mask.sum(dim=1).clamp_min(1) - 1
        query_token_id = token_ids.gather(1, query_position.unsqueeze(1)).squeeze(1)
        query_probability = torch.sigmoid(self.query_lexicon(query_token_id))
        query_index = query_probability.argmax(dim=1)
        semantic_candidates = torch.stack(
            (
                continuous[..., 0].argmax(dim=1),
                continuous[..., 1].argmax(dim=1),
                continuous[..., 2].argmax(dim=1),
                continuous[..., 3].argmax(dim=1),
                (binary_probability[..., 0] >= 0.5).sum(dim=1),
                (binary_probability[..., 1] >= 0.5).sum(dim=1),
            ),
            dim=1,
        )
        semantic_index = semantic_candidates.gather(
            1, query_index.unsqueeze(1)
        ).squeeze(1)
        marker_id = token_ids[:, 0]
        answer_kind = (query_index >= 4).to(torch.long)
        memory_response = self.surface_table[
            marker_id,
            answer_kind,
            semantic_index.clamp(0, 6),
        ]
        memory_used = memory_response >= 0
        if bool(memory_used.any()):
            memory_logits = torch.full_like(response_logits, -12.0)
            memory_logits.scatter_(
                1,
                memory_response.clamp_min(0).unsqueeze(1),
                12.0,
            )
            response_logits = torch.where(
                memory_used.unsqueeze(1), memory_logits, response_logits
            )
        return {
            "binary_logits": binary_logits,
            "continuous": continuous,
            "controls": controls,
            "field_route": torch.stack(field_routes, dim=1).mean(dim=1),
            "memory_used": memory_used,
            "operator_activation": operator_activation,
            "query_probability": query_probability,
            "query_index": query_index,
            "response_logits": response_logits,
            "semantic_index": semantic_index,
            "text_route": text_route,
            "text_ticks_used": torch.full(
                (token_ids.shape[0],),
                self.config.text_ticks
                if text_tick_budget is None
                else text_tick_budget,
                dtype=torch.long,
                device=token_ids.device,
            ),
            "field_ticks_used": torch.full(
                (token_ids.shape[0],),
                self.config.field_ticks,
                dtype=torch.long,
                device=token_ids.device,
            ),
        }


class FlatNeuralLanguageBaseline(nn.Module):
    def __init__(
        self,
        vocabulary: NeuralVocabulary,
        config: NeuralArchitectureConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config or NeuralArchitectureConfig()
        hidden = self.config.hidden_dim
        self.embedding = nn.Embedding(len(vocabulary.tokens), hidden, padding_idx=0)
        input_dim = hidden + NODE_COUNT * NODE_DIM + NODE_COUNT * NODE_COUNT
        output_dim = NODE_COUNT * 6 + len(vocabulary.responses)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden * 3),
            nn.GELU(),
            nn.Linear(hidden * 3, hidden * 3),
            nn.GELU(),
            nn.Linear(hidden * 3, output_dim),
        )

    def forward(
        self,
        token_ids: Tensor,
        token_mask: Tensor,
        node_features: Tensor,
        adjacency: Tensor,
        **_: Any,
    ) -> dict[str, Tensor]:
        embedded = self.embedding(token_ids)
        mask = token_mask.unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        inputs = torch.cat(
            (pooled, node_features.flatten(1), adjacency.flatten(1)), dim=1
        )
        output = self.network(inputs)
        state_size = NODE_COUNT * 6
        state = output[:, :state_size].view(-1, NODE_COUNT, 6)
        continuous = state[..., :4]
        binary_logits = state[..., 4:]
        return {
            "binary_logits": binary_logits,
            "continuous": continuous,
            "response_logits": output[:, state_size:],
        }


@dataclass(frozen=True)
class NeuralLossWeights:
    continuous: float = 1.0
    binary: float = 0.45
    response: float = 0.75
    conservation: float = 0.20
    control_induction: float = 0.75
    operator_induction: float = 1.75


def neural_batch_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    weights: NeuralLossWeights | None = None,
    *,
    state_only: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or NeuralLossWeights()
    salience = batch["salience"].clamp(0.0, 1.0)
    continuous_per = (
        (outputs["continuous"] - batch["target_continuous"]).square().mean(dim=(1, 2))
    )
    binary_per = F.binary_cross_entropy_with_logits(
        outputs["binary_logits"], batch["target_binary"], reduction="none"
    ).mean(dim=(1, 2))
    response_per = F.cross_entropy(
        outputs["response_logits"], batch["response_ids"], reduction="none"
    )
    initial_mass = (
        batch["node_features"][..., 1] * batch["node_features"][..., 5]
    ).sum(dim=1)
    target_mass = batch["target_continuous"][..., 1].sum(dim=1)
    predicted_mass = outputs["continuous"][..., 1].sum(dim=1)
    closed = (initial_mass - target_mass).abs() <= 1e-4
    conservation_per = torch.where(
        closed,
        (predicted_mass - initial_mass).abs(),
        torch.zeros_like(predicted_mass),
    )
    if "controls" in outputs:
        control_per = (
            (outputs["controls"] - batch["induced_controls"]).square().mean(dim=1)
        )
    else:
        control_per = torch.zeros_like(continuous_per)
    if "operator_activation" in outputs:
        operator_per = F.binary_cross_entropy(
            outputs["operator_activation"].clamp(1e-6, 1.0 - 1e-6),
            batch["induced_operators"],
            reduction="none",
        ).mean(dim=1)
    else:
        operator_per = torch.zeros_like(continuous_per)
    response_weight = 0.0 if state_only else weights.response
    per_example = (
        weights.continuous * continuous_per
        + weights.binary * binary_per
        + response_weight * response_per
        + weights.conservation * conservation_per
        + weights.control_induction * control_per
        + weights.operator_induction * operator_per
    )
    loss = (per_example * salience).sum() / salience.sum().clamp_min(1e-6)
    return loss, {
        "binary": binary_per.mean(),
        "conservation": conservation_per.mean(),
        "continuous": continuous_per.mean(),
        "control_induction": control_per.mean(),
        "operator_induction": operator_per.mean(),
        "per_example": per_example,
        "response": response_per.mean(),
        "total": loss,
    }


def neural_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass
class OperatorLexiconMemory:
    token_counts: dict[int, int] = field(default_factory=dict)
    positive_counts: dict[int, list[int]] = field(default_factory=dict)
    crystallized: dict[int, tuple[int, ...]] = field(default_factory=dict)
    raw_event_count: int = 0

    def observe(
        self,
        rows: Sequence[Mapping[str, Any]],
        vocabulary: NeuralVocabulary,
    ) -> None:
        token_mapping = vocabulary.token_to_id
        for row in rows:
            induced = induce_control_from_consequence(row)
            signature = set(induced["signature"])
            tokens = tokenize_neural_utterance(str(row["utterance"]))
            # The first token identifies the language surface and the final
            # token identifies the question.  Neither position can explain a
            # physical transition, so action induction only observes the
            # content span between them.
            token_ids = {token_mapping[token] for token in tokens[1:-1]}
            for token_id in token_ids:
                self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1
                counts = self.positive_counts.setdefault(token_id, [0] * PROCESS_COUNT)
                for operator_index, operator in enumerate(PROCESS_NAMES):
                    counts[operator_index] += int(operator in signature)
            self.raw_event_count += 1

    def crystallize(
        self,
        model: AtomNeuralLanguageField,
        *,
        minimum_evidence: int = 3,
        purity: float = 0.60,
    ) -> dict[str, Any]:
        laws: dict[int, tuple[int, ...]] = {}
        with torch.no_grad():
            for token_id, total in sorted(self.token_counts.items()):
                if token_id == 0 or total < minimum_evidence:
                    continue
                counts = self.positive_counts[token_id]
                operators = tuple(
                    index
                    for index, positive in enumerate(counts)
                    if positive / total >= purity
                )
                model.control_lexicon.weight[token_id].fill_(-6.0)
                for operator_index in operators:
                    model.control_lexicon.weight[token_id, operator_index] = 6.0
                if operators:
                    laws[token_id] = operators
            model.control_lexicon.weight[0].zero_()
        self.crystallized.update(laws)
        forgotten = self.raw_event_count
        self.raw_event_count = 0
        return {
            "crystallized_tokens": len(self.crystallized),
            "forgotten_raw_events": forgotten,
            "operator_laws": sum(len(values) for values in self.crystallized.values()),
            "retained_evidence_counters": len(self.token_counts),
        }

    def summary(self, vocabulary: NeuralVocabulary) -> dict[str, Any]:
        return {
            "crystallized": {
                vocabulary.tokens[token_id]: [
                    PROCESS_NAMES[index] for index in operators
                ]
                for token_id, operators in sorted(self.crystallized.items())
            },
            "raw_event_count": self.raw_event_count,
            "retained_evidence_counters": len(self.token_counts),
        }


def _semantic_candidates_from_row(row: Mapping[str, Any]) -> tuple[int, ...]:
    continuous = row["target_continuous"]
    binary = row["target_binary"]
    node_answers = tuple(
        max(
            range(NODE_COUNT),
            key=lambda index: (float(continuous[index][column]), -index),
        )
        for column in range(4)
    )
    counts = (
        sum(float(values[0]) >= 0.5 for values in binary),
        sum(float(values[1]) >= 0.5 for values in binary),
    )
    return (*node_answers, *counts)


@dataclass
class QuerySurfaceMemory:
    token_counts: dict[int, int] = field(default_factory=dict)
    evidence: dict[tuple[int, int, int, int, int], int] = field(default_factory=dict)
    query_laws: dict[int, int] = field(default_factory=dict)
    surface_laws: dict[tuple[int, int, int], int] = field(default_factory=dict)
    raw_event_count: int = 0

    def observe(
        self,
        rows: Sequence[Mapping[str, Any]],
        vocabulary: NeuralVocabulary,
    ) -> None:
        token_mapping = vocabulary.token_to_id
        response_mapping = vocabulary.response_to_id
        for row in rows:
            tokens = tokenize_neural_utterance(str(row["utterance"]))
            marker_id = token_mapping[tokens[0]]
            query_token_id = token_mapping[tokens[-1]]
            response_id = response_mapping[str(row["response"])]
            semantics = _semantic_candidates_from_row(row)
            self.token_counts[query_token_id] = (
                self.token_counts.get(query_token_id, 0) + 1
            )
            for query_index, semantic in enumerate(semantics):
                key = (
                    query_token_id,
                    query_index,
                    marker_id,
                    int(semantic),
                    response_id,
                )
                self.evidence[key] = self.evidence.get(key, 0) + 1
            self.raw_event_count += 1

    def _candidate_score(self, token_id: int, query_index: int) -> tuple[float, int]:
        grouped: dict[tuple[int, int], dict[int, int]] = {}
        for (
            candidate_token,
            candidate_query,
            marker,
            semantic,
            response,
        ), count in self.evidence.items():
            if candidate_token != token_id or candidate_query != query_index:
                continue
            response_counts = grouped.setdefault((marker, semantic), {})
            response_counts[response] = response_counts.get(response, 0) + count
        total = sum(sum(counts.values()) for counts in grouped.values())
        if total <= 0:
            return 0.0, 0
        consistent = sum(max(counts.values()) for counts in grouped.values())
        diversity = len(grouped)
        return consistent / total, diversity

    def crystallize(
        self,
        model: AtomNeuralLanguageField,
        *,
        minimum_evidence: int = 12,
        minimum_consistency: float = 0.94,
    ) -> dict[str, Any]:
        new_query_laws: dict[int, int] = {}
        for token_id, total in sorted(self.token_counts.items()):
            if total < minimum_evidence:
                continue
            candidates = [
                (*self._candidate_score(token_id, query_index), query_index)
                for query_index in range(6)
            ]
            consistency, diversity, query_index = max(
                candidates,
                key=lambda item: (item[0], min(item[1], 8), -item[2]),
            )
            if consistency >= minimum_consistency and diversity >= 2:
                new_query_laws[token_id] = query_index

        with torch.no_grad():
            for token_id, query_index in new_query_laws.items():
                model.query_lexicon.weight[token_id].fill_(-6.0)
                model.query_lexicon.weight[token_id, query_index] = 6.0
            model.query_lexicon.weight[0].zero_()

        self.query_laws.update(new_query_laws)
        grouped_surfaces: dict[tuple[int, int, int], dict[int, int]] = {}
        for token_id, query_index in self.query_laws.items():
            answer_kind = int(query_index >= 4)
            for (
                candidate_token,
                candidate_query,
                marker,
                semantic,
                response,
            ), count in self.evidence.items():
                if candidate_token != token_id or candidate_query != query_index:
                    continue
                response_counts = grouped_surfaces.setdefault(
                    (marker, answer_kind, semantic), {}
                )
                response_counts[response] = response_counts.get(response, 0) + count
        with torch.no_grad():
            for key, response_counts in grouped_surfaces.items():
                total = sum(response_counts.values())
                response = max(
                    response_counts,
                    key=lambda value: (response_counts[value], -value),
                )
                if response_counts[response] / max(total, 1) < minimum_consistency:
                    continue
                marker, answer_kind, semantic = key
                model.surface_table[marker, answer_kind, semantic] = response
                self.surface_laws[key] = response
        forgotten = self.raw_event_count
        self.raw_event_count = 0
        return {
            "forgotten_raw_events": forgotten,
            "query_laws": len(self.query_laws),
            "surface_laws": len(self.surface_laws),
        }

    def summary(self, vocabulary: NeuralVocabulary) -> dict[str, Any]:
        return {
            "query_laws": {
                vocabulary.tokens[token_id]: query_index
                for token_id, query_index in sorted(self.query_laws.items())
            },
            "raw_event_count": self.raw_event_count,
            "surface_laws": len(self.surface_laws),
        }


def consequence_lexical_coherence(
    rows: Sequence[Mapping[str, Any]],
) -> float:
    associations: dict[str, list[list[float]]] = {}
    for row in rows:
        signature = set(induce_control_from_consequence(row)["signature"])
        target = [float(operator in signature) for operator in PROCESS_NAMES]
        tokens = tokenize_neural_utterance(str(row["utterance"]))
        for token in set(tokens[1:-1]):
            associations.setdefault(token, []).append(target)
    purities: list[float] = []
    for targets in associations.values():
        if len(targets) < 2:
            continue
        for operator_index in range(PROCESS_COUNT):
            probability = sum(row[operator_index] for row in targets) / len(targets)
            purities.append(2.0 * abs(probability - 0.5))
    return _mean(purities) if purities else 0.0


def train_neural_model(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    temperature: float = 0.75,
    phase_strength: float = 0.04,
) -> dict[str, Any]:
    lexicon_memory: OperatorLexiconMemory | None = None
    lexicon_summary: dict[str, Any] | None = None
    query_memory: QuerySurfaceMemory | None = None
    query_summary: dict[str, Any] | None = None
    if isinstance(model, AtomNeuralLanguageField):
        lexicon_memory = OperatorLexiconMemory()
        lexicon_memory.observe(rows, vocabulary)
        lexicon_summary = lexicon_memory.crystallize(model)
        query_memory = QuerySurfaceMemory()
        query_memory.observe(rows, vocabulary)
        query_summary = query_memory.crystallize(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: list[float] = []
    model.to(device)
    for epoch in range(epochs):
        loader = make_runtime_loader(
            rows,
            vocabulary,
            config,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in loader:
            batch = move_neural_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["token_ids"],
                batch["token_mask"],
                batch["node_features"],
                batch["adjacency"],
                temperature=temperature,
                phase_strength=phase_strength,
            )
            loss, _ = neural_batch_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(sum(epoch_losses) / max(len(epoch_losses), 1))
    return {
        "epochs": epochs,
        "first_loss": history[0],
        "last_loss": history[-1],
        "lexicon_memory": lexicon_summary,
        "query_surface_memory": query_summary,
        "loss_history": history,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def evaluate_neural_model(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    batch_size: int,
    device: torch.device,
    temperature: float = 0.75,
    phase_strength: float = 0.04,
    text_ablate: int | None = None,
    field_ablate: int | None = None,
) -> dict[str, Any]:
    loader = make_runtime_loader(
        rows,
        vocabulary,
        config,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    model.eval()
    continuous_errors: list[float] = []
    binary_correct = 0
    binary_total = 0
    response_correct = 0
    response_total = 0
    state_success = 0
    joint_success = 0
    route_rows: list[list[float]] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_neural_batch(raw_batch, device)
            outputs = model(
                batch["token_ids"],
                batch["token_mask"],
                batch["node_features"],
                batch["adjacency"],
                temperature=temperature,
                phase_strength=phase_strength,
                text_ablate=text_ablate,
                field_ablate=field_ablate,
            )
            per_case_continuous = (
                (outputs["continuous"] - batch["target_continuous"])
                .abs()
                .mean(dim=(1, 2))
            )
            predicted_binary = (torch.sigmoid(outputs["binary_logits"]) >= 0.5).to(
                torch.float32
            )
            binary_case = (
                (predicted_binary == batch["target_binary"]).all(dim=2).all(dim=1)
            )
            predicted_response = outputs["response_logits"].argmax(dim=1)
            response_case = predicted_response == batch["response_ids"]
            state_case = (per_case_continuous <= 0.16) & binary_case
            continuous_errors.extend(per_case_continuous.cpu().tolist())
            binary_correct += int(
                (predicted_binary == batch["target_binary"]).sum().item()
            )
            binary_total += int(batch["target_binary"].numel())
            response_correct += int(response_case.sum().item())
            response_total += len(response_case)
            state_success += int(state_case.sum().item())
            joint_success += int((state_case & response_case).sum().item())
            if "text_route" in outputs:
                route_rows.extend(outputs["text_route"].cpu().tolist())
            for index, event_id in enumerate(batch["event_ids"]):
                predictions.append(
                    {
                        "continuous_mae": round(
                            float(per_case_continuous[index].cpu()), 8
                        ),
                        "event_id": event_id,
                        "response": vocabulary.responses[
                            int(predicted_response[index].cpu())
                        ],
                        "response_correct": bool(response_case[index].cpu()),
                        "state_success": bool(state_case[index].cpu()),
                    }
                )
    route_mean = []
    if route_rows:
        route_mean = [
            _mean([row[index] for row in route_rows]) for index in range(PROCESS_COUNT)
        ]
    return {
        "binary_accuracy": binary_correct / max(binary_total, 1),
        "cases": len(rows),
        "continuous_mae": _mean(continuous_errors),
        "joint_accuracy": joint_success / max(len(rows), 1),
        "predictions": predictions,
        "response_accuracy": response_correct / max(response_total, 1),
        "route_mean": route_mean,
        "state_accuracy": state_success / max(len(rows), 1),
    }


@dataclass
class NeuralHomeostaticGovernor:
    learning_rate: float = 0.0012
    temperature: float = 0.72
    phase_strength: float = 0.04
    update_repeats: int = 2
    chaos_budget: float = 1.15
    minimum_learning_rate: float = 0.00008
    maximum_learning_rate: float = 0.006
    minimum_temperature: float = 0.24
    maximum_temperature: float = 1.80
    minimum_phase_strength: float = 0.0
    maximum_phase_strength: float = 0.22
    history: list[dict[str, Any]] = field(default_factory=list)

    def _project_budget(self) -> None:
        normalized_lr = self.learning_rate / self.maximum_learning_rate
        normalized_temperature = self.temperature / self.maximum_temperature
        normalized_phase = self.phase_strength / max(self.maximum_phase_strength, 1e-8)
        load = normalized_lr + normalized_temperature + normalized_phase
        if load > self.chaos_budget:
            scale = self.chaos_budget / load
            self.learning_rate = max(
                self.minimum_learning_rate, self.learning_rate * scale
            )
            self.temperature = max(self.minimum_temperature, self.temperature * scale)
            self.phase_strength = max(
                self.minimum_phase_strength, self.phase_strength * scale
            )

    def observe(
        self, *, surprise: float, coherence: float, stage: str, window: int
    ) -> dict[str, Any]:
        if coherence < 0.72 and surprise > 0.40:
            action = "cool_incoherent_disturbance"
            accepted = False
            self.learning_rate = max(
                self.minimum_learning_rate, self.learning_rate * 0.45
            )
            self.temperature = max(self.minimum_temperature, self.temperature * 0.70)
            self.phase_strength = max(
                self.minimum_phase_strength, self.phase_strength * 0.35
            )
            self.update_repeats = 0
        elif coherence >= 0.72 and surprise > 0.35:
            action = "reheat_coherent_novelty"
            accepted = True
            self.learning_rate = min(
                self.maximum_learning_rate, self.learning_rate * 1.55 + 0.00015
            )
            self.temperature = min(
                self.maximum_temperature, self.temperature * 1.14 + 0.04
            )
            self.phase_strength = min(
                self.maximum_phase_strength, self.phase_strength + 0.018
            )
            self.update_repeats = 3
        else:
            action = "dissipate_toward_band"
            accepted = True
            self.learning_rate = max(
                self.minimum_learning_rate, self.learning_rate * 0.88
            )
            self.temperature = max(self.minimum_temperature, self.temperature * 0.93)
            self.phase_strength = max(
                self.minimum_phase_strength, self.phase_strength * 0.82
            )
            self.update_repeats = 1
        self._project_budget()
        record = {
            "accepted": accepted,
            "action": action,
            "chaos_load": self.chaos_load,
            "coherence": coherence,
            "learning_rate": self.learning_rate,
            "phase_strength": self.phase_strength,
            "stage": stage,
            "surprise": surprise,
            "temperature": self.temperature,
            "update_repeats": self.update_repeats,
            "window": window,
        }
        self.history.append(record)
        return record

    @property
    def chaos_load(self) -> float:
        return (
            self.learning_rate / self.maximum_learning_rate
            + self.temperature / self.maximum_temperature
            + self.phase_strength / max(self.maximum_phase_strength, 1e-8)
        )


def _gradient_vector(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    temperature: float,
    phase_strength: float,
) -> tuple[Tensor, float]:
    model.zero_grad(set_to_none=True)
    outputs = model(
        batch["token_ids"],
        batch["token_mask"],
        batch["node_features"],
        batch["adjacency"],
        temperature=temperature,
        phase_strength=phase_strength,
    )
    loss, _ = neural_batch_loss(outputs, batch, state_only=True)
    loss.backward()
    pieces: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("text_encoder") or name.startswith("control"):
            pieces.append(parameter.grad.detach().flatten())
    if not pieces:
        raise RuntimeError("no language/control gradients were produced")
    return torch.cat(pieces), float(loss.detach().cpu())


def window_observables(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    temperature: float,
    phase_strength: float,
    device: torch.device,
) -> dict[str, float]:
    batch = move_neural_batch(collate_runtime_rows(rows, vocabulary, config), device)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["token_ids"],
            batch["token_mask"],
            batch["node_features"],
            batch["adjacency"],
            temperature=temperature,
            phase_strength=phase_strength,
        )
        loss, _ = neural_batch_loss(outputs, batch)
    return {
        "coherence": consequence_lexical_coherence(rows),
        "surprise": float(loss.detach().cpu()),
    }


def _update_window(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    learning_rate: float,
    repeats: int,
    temperature: float,
    phase_strength: float,
    device: torch.device,
) -> list[float]:
    if repeats <= 0:
        return []
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    batch = move_neural_batch(collate_runtime_rows(rows, vocabulary, config), device)
    losses: list[float] = []
    model.train()
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["token_ids"],
            batch["token_mask"],
            batch["node_features"],
            batch["adjacency"],
            temperature=temperature,
            phase_strength=phase_strength,
        )
        loss, _ = neural_batch_loss(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def adapt_neural_stream(
    model: nn.Module,
    stages: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    vocabulary: NeuralVocabulary,
    config: NeuralArchitectureConfig,
    *,
    adaptive: bool,
    device: torch.device,
    window_size: int = 8,
) -> dict[str, Any]:
    governor = NeuralHomeostaticGovernor()
    lexicon_memory = OperatorLexiconMemory()
    query_memory = QuerySurfaceMemory()
    fixed_learning_rate = 0.00022
    raw_event_count = 0
    update_losses: list[float] = []
    window_index = 0
    rejected_windows = 0
    for stage_name, stage_rows in stages:
        for start in range(0, len(stage_rows), window_size):
            rows = tuple(stage_rows[start : start + window_size])
            raw_event_count += len(rows)
            observables = window_observables(
                model,
                rows,
                vocabulary,
                config,
                temperature=governor.temperature,
                phase_strength=governor.phase_strength,
                device=device,
            )
            if adaptive:
                record = governor.observe(
                    surprise=observables["surprise"],
                    coherence=observables["coherence"],
                    stage=stage_name,
                    window=window_index,
                )
                repeats = int(record["update_repeats"])
                learning_rate = float(record["learning_rate"])
                temperature = float(record["temperature"])
                phase_strength = float(record["phase_strength"])
                if not record["accepted"]:
                    rejected_windows += 1
            else:
                repeats = 1
                learning_rate = fixed_learning_rate
                temperature = 0.45
                phase_strength = 0.015
                record = {
                    "accepted": True,
                    "action": "fixed_schedule",
                    "chaos_load": 0.0,
                    "coherence": observables["coherence"],
                    "learning_rate": learning_rate,
                    "phase_strength": phase_strength,
                    "stage": stage_name,
                    "surprise": observables["surprise"],
                    "temperature": temperature,
                    "update_repeats": repeats,
                    "window": window_index,
                }
                governor.history.append(record)
            if bool(record["accepted"]):
                lexicon_memory.observe(rows, vocabulary)
                lexicon_memory.crystallize(model)
                query_memory.observe(rows, vocabulary)
                query_memory.crystallize(model)
            update_losses.extend(
                _update_window(
                    model,
                    rows,
                    vocabulary,
                    config,
                    learning_rate=learning_rate,
                    repeats=repeats,
                    temperature=temperature,
                    phase_strength=phase_strength,
                    device=device,
                )
            )
            raw_event_count -= len(rows)
            window_index += 1
    return {
        "adaptive": adaptive,
        "controller_history": copy.deepcopy(governor.history),
        "maximum_chaos_load": max(
            (float(row["chaos_load"]) for row in governor.history), default=0.0
        ),
        "raw_event_count": raw_event_count,
        "lexicon_memory": lexicon_memory.summary(vocabulary),
        "query_surface_memory": query_memory.summary(vocabulary),
        "rejected_windows": rejected_windows,
        "update_count": len(update_losses),
        "update_loss_first": update_losses[0] if update_losses else None,
        "update_loss_last": update_losses[-1] if update_losses else None,
        "windows": window_index,
    }


def clone_neural_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def tensor_state_payload(model: nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        payload[name] = {
            "shape": list(cpu.shape),
            "values": cpu.flatten().tolist(),
        }
    return payload


def neural_language_model_payload(
    model: AtomNeuralLanguageField,
    *,
    training_summary: Mapping[str, Any],
    inference_policy: EvidencePolicyConfig | None = None,
) -> dict[str, Any]:
    policy = inference_policy or EvidencePolicyConfig()
    policy.validate(model.config)
    base = {
        "architecture": "atom-neural-language-field-v1",
        "config": asdict(model.config),
        "inference_policy": asdict(policy),
        "response_vocabulary": list(model.vocabulary.responses),
        "schema_version": NEURAL_MODEL_SCHEMA,
        "token_vocabulary": list(model.vocabulary.tokens),
        "training_summary": dict(training_summary),
        "weights": tensor_state_payload(model),
    }
    return {**base, "model_hash": neural_model_hash(base)}


@dataclass(frozen=True)
class LoadedNeuralLanguageModel:
    model: AtomNeuralLanguageField
    model_hash: str
    inference_policy: EvidencePolicyConfig
    training_summary: Mapping[str, Any]


def load_neural_language_model(payload: Mapping[str, Any]) -> LoadedNeuralLanguageModel:
    expected = {
        "architecture",
        "config",
        "inference_policy",
        "model_hash",
        "response_vocabulary",
        "schema_version",
        "token_vocabulary",
        "training_summary",
        "weights",
    }
    if set(payload) != expected:
        raise ValueError(f"model fields must be {sorted(expected)}")
    if payload["schema_version"] != NEURAL_MODEL_SCHEMA:
        raise ValueError("unsupported neural model schema")
    if payload["architecture"] != "atom-neural-language-field-v1":
        raise ValueError("unsupported neural architecture")
    base = {key: payload[key] for key in expected if key != "model_hash"}
    expected_hash = neural_model_hash(base)
    if payload["model_hash"] != expected_hash:
        raise ValueError("neural model hash mismatch")
    config_mapping = payload["config"]
    if not isinstance(config_mapping, Mapping) or set(config_mapping) != set(
        NeuralArchitectureConfig.__dataclass_fields__
    ):
        raise ValueError("neural config fields are invalid")
    config = NeuralArchitectureConfig(**dict(config_mapping))
    config.validate()
    policy_mapping = payload["inference_policy"]
    if not isinstance(policy_mapping, Mapping) or set(policy_mapping) != set(
        EvidencePolicyConfig.__dataclass_fields__
    ):
        raise ValueError("inference policy fields are invalid")
    inference_policy = EvidencePolicyConfig(**dict(policy_mapping))
    inference_policy.validate(config)
    vocabulary = NeuralVocabulary(
        tokens=tuple(payload["token_vocabulary"]),
        responses=tuple(payload["response_vocabulary"]),
    )
    model = AtomNeuralLanguageField(vocabulary, config)
    expected_state = model.state_dict()
    weight_payload = payload["weights"]
    if not isinstance(weight_payload, Mapping) or set(weight_payload) != set(
        expected_state
    ):
        raise ValueError("weight tensor names are invalid")
    restored: dict[str, Tensor] = {}
    for name, template in expected_state.items():
        entry = weight_payload[name]
        if not isinstance(entry, Mapping) or set(entry) != {"shape", "values"}:
            raise ValueError(f"weight entry is invalid: {name}")
        shape = tuple(entry["shape"])
        if shape != tuple(template.shape):
            raise ValueError(f"weight shape mismatch: {name}")
        values = entry["values"]
        if not isinstance(values, list) or len(values) != template.numel():
            raise ValueError(f"weight value count mismatch: {name}")
        tensor = torch.tensor(values, dtype=template.dtype).reshape(template.shape)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite weight: {name}")
        restored[name] = tensor
    surface_table = restored.get("surface_table")
    if surface_table is None:
        raise ValueError("surface_table is missing")
    if bool((surface_table < -1).any()) or bool(
        (surface_table >= len(vocabulary.responses)).any()
    ):
        raise ValueError("surface_table contains an invalid response index")
    model.load_state_dict(restored, strict=True)
    training_summary = payload["training_summary"]
    if not isinstance(training_summary, Mapping):
        raise ValueError("training_summary must be an object")
    return LoadedNeuralLanguageModel(
        model=model,
        model_hash=str(payload["model_hash"]),
        inference_policy=inference_policy,
        training_summary=dict(training_summary),
    )


def evidence_preflight(
    model: AtomNeuralLanguageField,
    token_ids: Tensor,
    token_mask: Tensor,
    policy: EvidencePolicyConfig,
) -> dict[str, Tensor]:
    """Measure learned support before spending recurrent field computation."""

    policy.validate(model.config)
    token_probability = torch.sigmoid(model.control_lexicon(token_ids))
    token_probability = token_probability * token_mask.unsqueeze(-1).to(
        token_probability.dtype
    )
    operator_activation = 1.0 - torch.prod(
        1.0 - token_probability.clamp(0.0, 1.0), dim=1
    )
    query_position = token_mask.sum(dim=1).clamp_min(1) - 1
    query_token_id = token_ids.gather(1, query_position.unsqueeze(1)).squeeze(1)
    query_probability = torch.sigmoid(model.query_lexicon(query_token_id))
    operator_support = operator_activation.max(dim=1).values
    query_support = query_probability.max(dim=1).values
    eligible = (operator_support >= policy.minimum_operator_support) & (
        query_support >= policy.minimum_query_support
    )
    return {
        "eligible": eligible,
        "operator_support": operator_support,
        "query_support": query_support,
    }


def validate_neural_inference_request(payload: Mapping[str, Any]) -> None:
    expected = {"adjacency", "node_features", "request_id", "utterance"}
    if set(payload) != expected:
        raise ValueError(f"request fields must be {sorted(expected)}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be non-empty text")
    tokenize_neural_utterance(payload["utterance"])
    if len(payload["node_features"]) != NODE_COUNT or any(
        len(row) != NODE_DIM for row in payload["node_features"]
    ):
        raise ValueError("node_features must have shape [6, 8]")
    if len(payload["adjacency"]) != NODE_COUNT or any(
        len(row) != NODE_COUNT for row in payload["adjacency"]
    ):
        raise ValueError("adjacency must have shape [6, 6]")
    for collection_name in ("node_features", "adjacency"):
        for row in payload[collection_name]:
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{collection_name} must contain finite numbers")


def run_neural_inference_request(
    loaded: LoadedNeuralLanguageModel,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    validate_neural_inference_request(request)
    model = loaded.model
    vocabulary = model.vocabulary
    token_ids, token_mask = vocabulary.encode(
        str(request["utterance"]), model.config.maximum_tokens
    )
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    mask_tensor = torch.tensor([token_mask], dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        preflight = evidence_preflight(
            model,
            token_tensor,
            mask_tensor,
            loaded.inference_policy,
        )
    operator_support = float(preflight["operator_support"][0])
    query_support = float(preflight["query_support"][0])
    eligible = bool(preflight["eligible"][0])
    if not eligible:
        initial = request["node_features"]
        return {
            "artifact": {
                "assertion": None,
                "binary_probability": [
                    [float(row[5]), float(row[6])] for row in initial
                ],
                "candidate_response": None,
                "claim_status": "unknown",
                "continuous": [
                    [float(row[0]), float(row[1]), float(row[7]), float(row[4])]
                    for row in initial
                ],
                "evidence_path": [],
                "reasoning": {
                    "default_text_ticks": model.config.text_ticks,
                    "execution_skipped": True,
                    "field_ticks_used": 0,
                    "text_ticks_saved": model.config.text_ticks,
                    "text_ticks_used": 0,
                },
                "response": None,
                "support": {
                    "operator": operator_support,
                    "query": query_support,
                    "surface_law": False,
                },
            },
            "model_hash": loaded.model_hash,
            "request_id": request["request_id"],
            "runtime": "atom-neural-language-field-v2",
            "schema_version": NEURAL_MODEL_SCHEMA,
        }
    with torch.no_grad():
        outputs = model(
            token_tensor,
            mask_tensor,
            torch.tensor([request["node_features"]], dtype=torch.float32),
            torch.tensor([request["adjacency"]], dtype=torch.float32),
            text_tick_budget=loaded.inference_policy.fast_text_ticks,
        )
    response_index = int(outputs["response_logits"].argmax(dim=1).item())
    candidate_response = vocabulary.responses[response_index]
    surface_law = bool(outputs["memory_used"][0])
    assertion = candidate_response if surface_law else None
    claim_status = "derived" if surface_law else "hypothesized"
    evidence_path: list[dict[str, Any]] = []
    if surface_law:
        evidence_path = [
            {
                "kind": "retrieved",
                "source": "operator_lexicon_memory",
                "support": operator_support,
            },
            {
                "kind": "derived",
                "source": "frozen_root_field_executor",
                "ticks": model.config.field_ticks,
            },
            {
                "kind": "retrieved",
                "source": "factorized_query_surface_memory",
                "support": query_support,
            },
        ]
    return {
        "artifact": {
            "assertion": assertion,
            "binary_probability": torch.sigmoid(outputs["binary_logits"])[0].tolist(),
            "candidate_response": candidate_response,
            "claim_status": claim_status,
            "continuous": outputs["continuous"][0].tolist(),
            "evidence_path": evidence_path,
            "reasoning": {
                "default_text_ticks": model.config.text_ticks,
                "execution_skipped": False,
                "field_ticks_used": int(outputs["field_ticks_used"][0]),
                "text_ticks_saved": model.config.text_ticks
                - int(outputs["text_ticks_used"][0]),
                "text_ticks_used": int(outputs["text_ticks_used"][0]),
            },
            "response": assertion,
            "support": {
                "operator": operator_support,
                "query": query_support,
                "surface_law": surface_law,
            },
        },
        "model_hash": loaded.model_hash,
        "request_id": request["request_id"],
        "runtime": "atom-neural-language-field-v2",
        "schema_version": NEURAL_MODEL_SCHEMA,
    }


def neural_model_self_tests() -> dict[str, Any]:
    config = NeuralArchitectureConfig()
    config.validate()
    policy = EvidencePolicyConfig()
    policy.validate(config)
    vocabulary = NeuralVocabulary.build(
        ("zaq", "mira", "pava", "liri", "vex", "sai"),
        ("az0", "az1"),
    )
    model = AtomNeuralLanguageField(vocabulary, config)
    checks: dict[str, bool] = {
        "seven_text_operators": TEXT_OPERATOR_NAMES == tuple(PROCESS_NAMES),
        "field_cell_is_runtime_wired": isinstance(model.field_cell, AtomFieldCell),
        "factorized_surface_memory": model.surface_table.shape[1:] == (2, 7),
        "parameter_count_is_nonzero": neural_parameter_count(model) > 0,
        "policy_has_fast_path": policy.fast_text_ticks < config.text_ticks,
    }
    try:
        NeuralArchitectureConfig(hidden_dim=17).validate()
    except ValueError:
        checks["invalid_hidden_dimension_fails_closed"] = True
    else:
        checks["invalid_hidden_dimension_fails_closed"] = False
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}
