"""Strict natural-English codec around the evidence-bound Atom neural field."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from atom_english_language_dataset import (
    QUERY_NOUNS,
    normalize_english_request,
    render_english_answer,
)
from atom_neural_language_model import (
    LoadedNeuralLanguageModel,
    load_neural_language_model,
    run_neural_inference_request,
)


ENGLISH_MODEL_SCHEMA = 1
ENGLISH_LANGUAGE_RUNTIME = "atom-english-cognitive-shell-v1"


def english_model_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnglishShellConfig:
    unknown_answer: str = "I do not have enough grounded evidence to answer that."
    internal_marker: str = "eng"

    def validate(self) -> None:
        if self.internal_marker != "eng":
            raise ValueError("English shell marker must be eng")
        if not self.unknown_answer or len(self.unknown_answer) > 160:
            raise ValueError("unknown answer must contain at most 160 characters")


@dataclass(frozen=True)
class LoadedEnglishLanguageModel:
    core: LoadedNeuralLanguageModel
    model_hash: str
    shell: EnglishShellConfig


def english_language_model_payload(
    core_payload: Mapping[str, Any],
    *,
    dataset_hash: str,
    shell: EnglishShellConfig | None = None,
) -> dict[str, Any]:
    shell = shell or EnglishShellConfig()
    shell.validate()
    load_neural_language_model(core_payload)
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError("dataset_hash must be a SHA-256 string")
    base = {
        "architecture": ENGLISH_LANGUAGE_RUNTIME,
        "core_model": dict(core_payload),
        "dataset_hash": dataset_hash,
        "schema_version": ENGLISH_MODEL_SCHEMA,
        "shell": asdict(shell),
    }
    return {**base, "model_hash": english_model_hash(base)}


def load_english_language_model(
    payload: Mapping[str, Any],
) -> LoadedEnglishLanguageModel:
    expected = {
        "architecture",
        "core_model",
        "dataset_hash",
        "model_hash",
        "schema_version",
        "shell",
    }
    if set(payload) != expected:
        raise ValueError(f"English model fields must be {sorted(expected)}")
    if payload["schema_version"] != ENGLISH_MODEL_SCHEMA:
        raise ValueError("unsupported English model schema")
    if payload["architecture"] != ENGLISH_LANGUAGE_RUNTIME:
        raise ValueError("unsupported English model architecture")
    base = {key: payload[key] for key in expected if key != "model_hash"}
    if payload["model_hash"] != english_model_hash(base):
        raise ValueError("English model hash mismatch")
    dataset_hash = payload["dataset_hash"]
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise ValueError("English model dataset hash is invalid")
    shell_payload = payload["shell"]
    if not isinstance(shell_payload, Mapping) or set(shell_payload) != set(
        EnglishShellConfig.__dataclass_fields__
    ):
        raise ValueError("English shell fields are invalid")
    shell = EnglishShellConfig(**dict(shell_payload))
    shell.validate()
    core_payload = payload["core_model"]
    if not isinstance(core_payload, Mapping):
        raise ValueError("English core model must be an object")
    core = load_neural_language_model(core_payload)
    return LoadedEnglishLanguageModel(
        core=core,
        model_hash=str(payload["model_hash"]),
        shell=shell,
    )


def _query_index(internal_utterance: str) -> int | None:
    final_word = internal_utterance.split()[-1]
    query_names = tuple(QUERY_NOUNS)
    for index, query_type in enumerate(query_names):
        if QUERY_NOUNS[query_type] == final_word:
            return index
    return None


def validate_english_inference_request(payload: Mapping[str, Any]) -> str:
    expected = {"adjacency", "node_features", "request_id", "utterance"}
    if set(payload) != expected:
        raise ValueError(f"English request fields must be {sorted(expected)}")
    if not isinstance(payload["request_id"], str) or not payload["request_id"]:
        raise ValueError("request_id must be non-empty text")
    internal = normalize_english_request(payload["utterance"])
    if len(payload["node_features"]) != 6 or any(
        len(row) != 8 for row in payload["node_features"]
    ):
        raise ValueError("node_features must have shape [6, 8]")
    if len(payload["adjacency"]) != 6 or any(
        len(row) != 6 for row in payload["adjacency"]
    ):
        raise ValueError("adjacency must have shape [6, 6]")
    for name in ("node_features", "adjacency"):
        for row in payload[name]:
            for value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{name} must contain finite numbers")
    return internal


def _unknown_english_response(
    loaded: LoadedEnglishLanguageModel,
    request: Mapping[str, Any],
    internal: str,
    unknown_tokens: list[str],
) -> dict[str, Any]:
    initial = request["node_features"]
    core = loaded.core.model
    return {
        "artifact": {
            "answer": loaded.shell.unknown_answer,
            "assertion": None,
            "binary_probability": [[float(row[5]), float(row[6])] for row in initial],
            "candidate_answer": None,
            "candidate_response": None,
            "claim_status": "unknown",
            "continuous": [
                [float(row[0]), float(row[1]), float(row[7]), float(row[4])]
                for row in initial
            ],
            "evidence_path": [],
            "internal_utterance": internal,
            "reasoning": {
                "default_text_ticks": core.config.text_ticks,
                "execution_skipped": True,
                "field_ticks_used": 0,
                "text_ticks_saved": core.config.text_ticks,
                "text_ticks_used": 0,
            },
            "response": None,
            "support": {"operator": 0.0, "query": 0.0, "surface_law": False},
            "unknown_tokens": unknown_tokens,
            "user_utterance": request["utterance"],
        },
        "core_model_hash": loaded.core.model_hash,
        "model_hash": loaded.model_hash,
        "request_id": request["request_id"],
        "runtime": ENGLISH_LANGUAGE_RUNTIME,
        "schema_version": ENGLISH_MODEL_SCHEMA,
    }


def run_english_inference_request(
    loaded: LoadedEnglishLanguageModel,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    internal = validate_english_inference_request(request)
    vocabulary = set(loaded.core.model.vocabulary.tokens)
    unknown_tokens = sorted(set(internal.split()) - vocabulary)
    if unknown_tokens:
        return _unknown_english_response(loaded, request, internal, unknown_tokens)
    core_response = run_neural_inference_request(
        loaded.core,
        {
            "adjacency": request["adjacency"],
            "node_features": request["node_features"],
            "request_id": request["request_id"],
            "utterance": internal,
        },
    )
    artifact = dict(core_response["artifact"])
    query_index = _query_index(internal)
    candidate = artifact["candidate_response"]
    assertion = artifact["assertion"]
    candidate_answer = None
    answer = loaded.shell.unknown_answer
    if candidate is not None and query_index is not None:
        candidate_answer = render_english_answer(str(candidate), query_index)
    if assertion is not None and query_index is not None:
        answer = render_english_answer(str(assertion), query_index)
    return {
        "artifact": {
            **artifact,
            "answer": answer,
            "candidate_answer": candidate_answer,
            "internal_utterance": internal,
            "unknown_tokens": [],
            "user_utterance": request["utterance"],
        },
        "core_model_hash": loaded.core.model_hash,
        "model_hash": loaded.model_hash,
        "request_id": request["request_id"],
        "runtime": ENGLISH_LANGUAGE_RUNTIME,
        "schema_version": ENGLISH_MODEL_SCHEMA,
    }


def english_model_self_tests() -> dict[str, Any]:
    shell = EnglishShellConfig()
    shell.validate()
    checks = {
        "normalizer_strips_punctuation": normalize_english_request(
            "Please spread, then report SIGNAL!"
        )
        == "eng spread signal",
        "query_mapping": _query_index("eng please report structures") == 5,
        "renderer_is_natural_english": render_english_answer("count3", 4)
        == "3 nodes remain active.",
        "unknown_answer_is_explicit": "enough grounded evidence"
        in shell.unknown_answer,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}
