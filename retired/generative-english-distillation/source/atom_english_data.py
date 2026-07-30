"""Broad English curriculum, tokenizer contract, and streaming token packing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

ATOM_ENGLISH_DATA_RUNTIME = "atom-broad-english-curriculum-v1"
DEFAULT_FOUNDATION_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_FOUNDATION_TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DEFAULT_TOKENIZER_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DEFAULT_BASE_TEACHER_ID = "HuggingFaceTB/SmolLM2-1.7B"
DEFAULT_DIALOGUE_TEACHER_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEFAULT_BASE_TEACHER_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
DEFAULT_DIALOGUE_TEACHER_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
SMOLLM_CORPUS_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"
SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    stage: str
    dataset_id: str
    dataset_config: str | None
    split: str
    content_field: str
    content_kind: str
    revision: str
    license_name: str
    weight: float = 1.0
    minimum_characters: int = 160

    def __post_init__(self) -> None:
        if not self.source_id or not self.dataset_id:
            raise ValueError("corpus source identities cannot be empty")
        if self.stage not in {"foundation", "dialogue", "evaluation"}:
            raise ValueError("unsupported English curriculum stage")
        if self.content_kind not in {"text", "messages"}:
            raise ValueError("unsupported corpus content kind")
        if not self.split or not self.content_field:
            raise ValueError("corpus split and content field are required")
        if not self.revision or not self.license_name:
            raise ValueError("corpus revision and license are required")
        if self.weight <= 0.0:
            raise ValueError("corpus weight must be positive")
        if self.minimum_characters < 0:
            raise ValueError("minimum_characters cannot be negative")


@dataclass(frozen=True)
class EnglishCurriculum:
    tokenizer_id: str
    tokenizer_revision: str
    foundation_tokenizer_id: str
    foundation_tokenizer_revision: str
    base_teacher_id: str
    base_teacher_revision: str
    dialogue_teacher_id: str
    dialogue_teacher_revision: str
    sources: tuple[CorpusSource, ...]
    schema_version: int = 1
    runtime: str = ATOM_ENGLISH_DATA_RUNTIME

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported English curriculum schema")
        if self.runtime != ATOM_ENGLISH_DATA_RUNTIME:
            raise ValueError("unsupported English curriculum runtime")
        tokenizer_fields = (
            self.tokenizer_id,
            self.tokenizer_revision,
            self.foundation_tokenizer_id,
            self.foundation_tokenizer_revision,
        )
        if not all(tokenizer_fields):
            raise ValueError("tokenizer identities and revisions are required")
        teacher_fields = (
            self.base_teacher_id,
            self.base_teacher_revision,
            self.dialogue_teacher_id,
            self.dialogue_teacher_revision,
        )
        if not all(teacher_fields):
            raise ValueError("teacher identities and revisions are required")
        identities = [source.source_id for source in self.sources]
        if len(identities) != len(set(identities)):
            raise ValueError("corpus source identities must be unique")
        stages = {source.stage for source in self.sources}
        if stages != {"foundation", "dialogue", "evaluation"}:
            raise ValueError("curriculum must cover all three language stages")

    def stage(self, name: str) -> tuple[CorpusSource, ...]:
        selected = tuple(source for source in self.sources if source.stage == name)
        if not selected:
            raise ValueError(f"curriculum stage is empty: {name}")
        return selected

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(source) for source in self.sources]
        return payload


def broad_english_curriculum() -> EnglishCurriculum:
    """Return the full scalable corpus contract, not a prompt fixture."""

    return EnglishCurriculum(
        tokenizer_id=DEFAULT_TOKENIZER_ID,
        tokenizer_revision=DEFAULT_TOKENIZER_REVISION,
        foundation_tokenizer_id=DEFAULT_FOUNDATION_TOKENIZER_ID,
        foundation_tokenizer_revision=DEFAULT_FOUNDATION_TOKENIZER_REVISION,
        base_teacher_id=DEFAULT_BASE_TEACHER_ID,
        base_teacher_revision=DEFAULT_BASE_TEACHER_REVISION,
        dialogue_teacher_id=DEFAULT_DIALOGUE_TEACHER_ID,
        dialogue_teacher_revision=DEFAULT_DIALOGUE_TEACHER_REVISION,
        sources=(
            CorpusSource(
                source_id="fineweb-edu-full",
                stage="foundation",
                dataset_id="HuggingFaceFW/fineweb-edu",
                dataset_config="default",
                split="train",
                content_field="text",
                content_kind="text",
                revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                license_name="ODC-By-1.0",
                weight=0.72,
                minimum_characters=256,
            ),
            CorpusSource(
                source_id="cosmopedia-v2-full",
                stage="foundation",
                dataset_id="HuggingFaceTB/smollm-corpus",
                dataset_config="cosmopedia-v2",
                split="train",
                content_field="text",
                content_kind="text",
                revision=SMOLLM_CORPUS_REVISION,
                license_name="ODC-By-1.0",
                weight=0.28,
                minimum_characters=256,
            ),
            CorpusSource(
                source_id="smoltalk-all",
                stage="dialogue",
                dataset_id="HuggingFaceTB/smoltalk",
                dataset_config="all",
                split="train",
                content_field="messages",
                content_kind="messages",
                revision=SMOLTALK_REVISION,
                license_name="mixed-see-source-dataset-cards",
                weight=1.0,
                minimum_characters=0,
            ),
            CorpusSource(
                source_id="wikitext-103-heldout",
                stage="evaluation",
                dataset_id="Salesforce/wikitext",
                dataset_config="wikitext-103-raw-v1",
                split="test",
                content_field="text",
                content_kind="text",
                revision="b08601e04326c79dfdd32d625aee71d232d685c3",
                license_name="CC-BY-SA-3.0",
                weight=1.0,
                minimum_characters=40,
            ),
            CorpusSource(
                source_id="lambada-openai-heldout",
                stage="evaluation",
                dataset_id="EleutherAI/lambada_openai",
                dataset_config="en",
                split="test",
                content_field="text",
                content_kind="text",
                revision="900124bf3b8235c6daf21033af9948b3f07346c4",
                license_name="research-use",
                weight=1.0,
                minimum_characters=40,
            ),
        ),
    )


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_aligned_tokenizer(
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    *,
    revision: str = "main",
    local_files_only: bool = False,
) -> Any:
    """Load the teacher-aligned tokenizer without adding vocabulary items."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Atom English requires a deterministic fast tokenizer")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an end-of-text token")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != int(tokenizer.vocab_size):
        raise ValueError(
            "tokenizer has added vocabulary and cannot align teacher logits"
        )
    return tokenizer


def tokenizer_manifest(
    tokenizer: Any,
    *,
    tokenizer_id: str,
    revision: str,
) -> dict[str, Any]:
    serialized = tokenizer.backend_tokenizer.to_str()
    return {
        "schema_version": 1,
        "tokenizer_id": tokenizer_id,
        "requested_revision": revision,
        "vocab_size": len(tokenizer),
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "fast_backend_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "normalizer": type(tokenizer.backend_tokenizer.normalizer).__name__,
        "pre_tokenizer": type(tokenizer.backend_tokenizer.pre_tokenizer).__name__,
    }


def validate_tokenizer_alignment(
    foundation_tokenizer: Any,
    dialogue_tokenizer: Any,
) -> dict[str, Any]:
    """Prove that stage-specific special tokens share one output vocabulary."""

    foundation_vocab = foundation_tokenizer.get_vocab()
    dialogue_vocab = dialogue_tokenizer.get_vocab()
    if foundation_vocab != dialogue_vocab:
        raise ValueError("foundation and dialogue token vocabularies differ")
    vocabulary_hash = canonical_json_hash(foundation_vocab)
    return {
        "schema_version": 1,
        "vocab_size": len(foundation_vocab),
        "vocabulary_hash": vocabulary_hash,
        "foundation_special_tokens": {
            "bos_token_id": int(foundation_tokenizer.bos_token_id),
            "eos_token_id": int(foundation_tokenizer.eos_token_id),
            "pad_token_id": int(foundation_tokenizer.pad_token_id),
        },
        "dialogue_special_tokens": {
            "bos_token_id": int(dialogue_tokenizer.bos_token_id),
            "eos_token_id": int(dialogue_tokenizer.eos_token_id),
            "pad_token_id": int(dialogue_tokenizer.pad_token_id),
        },
    }


def stream_source(
    source: CorpusSource,
    *,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[Mapping[str, Any]]:
    """Load a source lazily so runs consume tokens instead of whole datasets."""

    from datasets import load_dataset

    dataset = load_dataset(
        source.dataset_id,
        source.dataset_config,
        split=source.split,
        revision=source.revision,
        streaming=True,
    )
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def weighted_interleave_streams(
    streams: Sequence[Iterable[Mapping[str, Any]]],
    weights: Sequence[float],
    *,
    seed: int,
) -> Iterator[Mapping[str, Any]]:
    """Interleave complete streams without materializing or truncating them."""

    if not streams or len(streams) != len(weights):
        raise ValueError("streams and weights must be non-empty and aligned")
    if any(float(weight) <= 0.0 for weight in weights):
        raise ValueError("stream weights must be positive")
    randomizer = random.Random(int(seed))
    iterators = [iter(stream) for stream in streams]
    active = list(range(len(iterators)))
    while active:
        total = sum(float(weights[index]) for index in active)
        selection = randomizer.random() * total
        selected = active[-1]
        cumulative = 0.0
        for index in active:
            cumulative += float(weights[index])
            if selection < cumulative:
                selected = index
                break
        try:
            yield next(iterators[selected])
        except StopIteration:
            active.remove(selected)


def stream_stage_sources(
    sources: Sequence[CorpusSource],
    *,
    seed: int,
    shuffle_buffer: int,
    epochs: int = 1,
) -> Iterator[Mapping[str, Any]]:
    """Stream every declared source with deterministic epoch-wise reshuffling."""

    if not sources:
        raise ValueError("training stage has no corpus sources")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    content_kinds = {source.content_kind for source in sources}
    content_fields = {source.content_field for source in sources}
    if len(content_kinds) != 1 or len(content_fields) != 1:
        raise ValueError("interleaved sources must share one content schema")
    for epoch in range(epochs):
        epoch_seed = int(seed) + epoch * 10_007
        streams = [
            stream_source(
                source,
                seed=epoch_seed + index * 1_009,
                shuffle_buffer=shuffle_buffer,
            )
            for index, source in enumerate(sources)
        ]
        yield from weighted_interleave_streams(
            streams,
            [source.weight for source in sources],
            seed=epoch_seed,
        )


class PackedEnglishStream(IterableDataset):
    """Pack unbounded documents into dense autoregressive token blocks."""

    def __init__(
        self,
        documents: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        *,
        text_field: str,
        sequence_length: int,
        maximum_tokens: int | None,
        minimum_characters: int,
        seed: int,
        deduplication_window: int = 100_000,
    ) -> None:
        super().__init__()
        if sequence_length < 16:
            raise ValueError("sequence_length must be at least 16")
        if maximum_tokens is not None and maximum_tokens < sequence_length:
            raise ValueError("maximum_tokens is smaller than one sequence")
        if deduplication_window < 1:
            raise ValueError("deduplication_window must be positive")
        self.documents = documents
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.sequence_length = int(sequence_length)
        self.maximum_tokens = maximum_tokens
        self.minimum_characters = int(minimum_characters)
        self.seed = int(seed)
        self.deduplication_window = int(deduplication_window)

    def _worker_shard(self) -> tuple[int, int]:
        worker = get_worker_info()
        if worker is None:
            return 0, 1
        return int(worker.id), int(worker.num_workers)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        shard_index, shard_count = self._worker_shard()
        buffer: list[int] = []
        seen: dict[str, None] = {}
        emitted = 0
        eos = int(self.tokenizer.eos_token_id)
        bos = int(self.tokenizer.bos_token_id)
        for document_index, row in enumerate(self.documents):
            if document_index % shard_count != shard_index:
                continue
            text = row.get(self.text_field)
            if not isinstance(text, str):
                continue
            normalized = " ".join(text.split())
            if len(normalized) < self.minimum_characters:
                continue
            identity = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if identity in seen:
                continue
            seen[identity] = None
            if len(seen) > self.deduplication_window:
                remove_count = max(1, self.deduplication_window // 10)
                for key in tuple(seen)[:remove_count]:
                    del seen[key]
            token_ids = self.tokenizer.encode(normalized, add_special_tokens=False)
            if not token_ids:
                continue
            if not buffer:
                buffer.append(bos)
            buffer.extend(int(token) for token in token_ids)
            buffer.append(eos)
            block_width = self.sequence_length + 1
            while len(buffer) >= block_width:
                if self.maximum_tokens is not None:
                    if emitted + self.sequence_length > self.maximum_tokens:
                        return
                block = buffer[:block_width]
                del buffer[: self.sequence_length]
                emitted += self.sequence_length
                yield {
                    "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                    "labels": torch.tensor(block[1:], dtype=torch.long),
                }


def _validated_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("dialogue row messages must be a sequence")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("dialogue message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("dialogue message role is unsupported")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("dialogue message content must be non-empty")
        messages.append({"role": role, "content": content.strip()})
    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("dialogue requires at least one assistant response")
    return messages


def chat_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, Tensor):
        encoded = encoded.flatten().tolist()
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise ValueError("chat template returned unsupported token data")
    return [int(token) for token in encoded]


def encode_dialogue(
    tokenizer: Any,
    messages: Any,
    *,
    sequence_length: int,
) -> dict[str, Tensor]:
    """Encode chat while applying loss only to assistant-authored tokens."""

    validated = _validated_messages(messages)
    token_ids: list[int] = []
    assistant_mask: list[bool] = []
    prefix: list[dict[str, str]] = []
    for message in validated:
        before = (
            chat_token_ids(
                tokenizer,
                prefix,
                add_generation_prompt=False,
            )
            if prefix
            else []
        )
        prefix.append(message)
        after = chat_token_ids(
            tokenizer,
            prefix,
            add_generation_prompt=False,
        )
        if after[: len(before)] != before:
            raise ValueError("chat template is not prefix stable")
        addition = [int(token) for token in after[len(before) :]]
        token_ids.extend(addition)
        assistant_mask.extend([message["role"] == "assistant"] * len(addition))
    if len(token_ids) < 2:
        raise ValueError("dialogue tokenization is empty")
    token_ids = token_ids[: sequence_length + 1]
    assistant_mask = assistant_mask[: sequence_length + 1]
    input_ids = token_ids[:-1]
    labels = [
        token_ids[index + 1] if assistant_mask[index + 1] else -100
        for index in range(len(input_ids))
    ]
    pad = int(tokenizer.pad_token_id)
    missing = sequence_length - len(input_ids)
    if missing > 0:
        input_ids.extend([pad] * missing)
        labels.extend([-100] * missing)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class DialogueEnglishStream(IterableDataset):
    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        *,
        messages_field: str,
        sequence_length: int,
        maximum_examples: int | None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.tokenizer = tokenizer
        self.messages_field = messages_field
        self.sequence_length = int(sequence_length)
        self.maximum_examples = maximum_examples

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        worker = get_worker_info()
        shard_index = 0 if worker is None else int(worker.id)
        shard_count = 1 if worker is None else int(worker.num_workers)
        emitted = 0
        for index, row in enumerate(self.rows):
            if index % shard_count != shard_index:
                continue
            try:
                encoded = encode_dialogue(
                    self.tokenizer,
                    row.get(self.messages_field),
                    sequence_length=self.sequence_length,
                )
            except ValueError:
                continue
            if not bool((encoded["labels"] != -100).any()):
                continue
            yield encoded
            emitted += 1
            if self.maximum_examples is not None and emitted >= self.maximum_examples:
                return


def atom_english_data_self_test() -> dict[str, bool]:
    curriculum = broad_english_curriculum()
    mixture_a = list(
        weighted_interleave_streams(
            (
                [{"source": "a", "index": index} for index in range(5)],
                [{"source": "b", "index": index} for index in range(3)],
            ),
            (0.7, 0.3),
            seed=31,
        )
    )
    mixture_b = list(
        weighted_interleave_streams(
            (
                [{"source": "a", "index": index} for index in range(5)],
                [{"source": "b", "index": index} for index in range(3)],
            ),
            (0.7, 0.3),
            seed=31,
        )
    )
    return {
        "all_stages_present": {source.stage for source in curriculum.sources}
        == {"foundation", "dialogue", "evaluation"},
        "foundation_is_broad_stream": {
            source.source_id for source in curriculum.stage("foundation")
        }
        == {"fineweb-edu-full", "cosmopedia-v2-full"},
        "dialogue_is_full_free_form": (
            curriculum.stage("dialogue")[0].dataset_id == "HuggingFaceTB/smoltalk"
            and curriculum.stage("dialogue")[0].dataset_config == "all"
            and curriculum.stage("dialogue")[0].content_kind == "messages"
        ),
        "weighted_stream_is_complete": len(mixture_a) == 8
        and {row["source"] for row in mixture_a} == {"a", "b"},
        "weighted_stream_is_deterministic": mixture_a == mixture_b,
        "evaluation_is_external": all(
            source.dataset_id
            not in {item.dataset_id for item in curriculum.stage("foundation")}
            for source in curriculum.stage("evaluation")
        ),
        "manifest_hash_stable": canonical_json_hash(curriculum.to_dict())
        == canonical_json_hash(curriculum.to_dict()),
    }
