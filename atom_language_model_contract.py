"""Machine-readable admission contract for the Atom language membrane."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ATOM_LANGUAGE_MODEL_CONTRACT_RUNTIME = "atom-language-model-contract-v1"
MODEL_CONTRACT_PATH = Path(__file__).with_name("atom-language-model.json")
QWEN_CHATML_MANUAL_TEMPLATE = "qwen-chatml-manual-v1"
RAW_PROMPT_TEMPLATE = "raw-prompt-v1"
SUPPORTED_LLAMA_CPP_CHAT_TEMPLATES = frozenset(
    {
        QWEN_CHATML_MANUAL_TEMPLATE,
        RAW_PROMPT_TEMPLATE,
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_language_model_contract(
    path: Path = MODEL_CONTRACT_PATH,
) -> dict[str, Any]:
    """Load and fail closed on a malformed official-model declaration."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Atom language-model contract is unavailable") from error
    if not isinstance(payload, dict):
        raise ValueError("Atom language-model contract must be an object")
    if type(payload.get("schema")) is not int or payload["schema"] != 1:
        raise ValueError("Atom language-model contract schema is invalid")
    if payload.get("runtime") != ATOM_LANGUAGE_MODEL_CONTRACT_RUNTIME:
        raise ValueError("Atom language-model contract runtime is invalid")
    if payload.get("role") != "language-only-membrane":
        raise ValueError("Atom language-model role is invalid")
    if payload.get("default_provider") != "llama-cpp":
        raise ValueError("Atom language-model provider is invalid")

    base_model = payload.get("base_model")
    artifact = payload.get("artifact")
    runtime_policy = payload.get("runtime_policy")
    certification = payload.get("certification")
    if not all(
        isinstance(item, dict)
        for item in (base_model, artifact, runtime_policy, certification)
    ):
        raise ValueError("Atom language-model contract sections are invalid")
    assert isinstance(base_model, dict)
    assert isinstance(artifact, dict)
    assert isinstance(runtime_policy, dict)
    assert isinstance(certification, dict)

    if base_model.get("model_id") != "Qwen/Qwen3-4B-Instruct-2507":
        raise ValueError("Atom base language-model identity is invalid")
    if base_model.get("reasoning_mode") != "non-thinking-only":
        raise ValueError("Atom language-model reasoning mode is invalid")
    if (
        type(base_model.get("parameters")) is not int
        or base_model["parameters"] != 4_000_000_000
    ):
        raise ValueError("Atom language-model parameter count is invalid")

    filename = artifact.get("filename")
    sha256 = artifact.get("sha256")
    byte_count = artifact.get("bytes")
    relative_path = artifact.get("default_relative_path")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename != "qwen3-4b-instruct-2507-q8_0.gguf"
    ):
        raise ValueError("Atom GGUF filename is invalid")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError("Atom GGUF SHA-256 is invalid")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError("Atom GGUF byte count is invalid")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\x00" in relative_path
        or Path(relative_path).is_absolute()
    ):
        raise ValueError("Atom GGUF default path is invalid")
    if runtime_policy.get("model_integrity_required") is not True:
        raise ValueError("Atom GGUF integrity admission is not required")
    if runtime_policy.get("local_is_default") is not True:
        raise ValueError("Atom local-model default is not declared")
    if runtime_policy.get("harness_context_tokens") != 32_768:
        raise ValueError("Atom language-model context window is invalid")
    if runtime_policy.get("chat_template") != QWEN_CHATML_MANUAL_TEMPLATE:
        raise ValueError("Atom language-model chat template is invalid")
    if runtime_policy.get("executable") != "llama-server":
        raise ValueError("Atom language-model executable is invalid")
    if (
        runtime_policy.get("structured_output") != "llama.cpp-resident-json-schema"
        or runtime_policy.get("server_api") != "authenticated-loopback-/completion"
    ):
        raise ValueError("Atom language-model server boundary is invalid")
    resident_lane = runtime_policy.get("resident_lane")
    if not isinstance(resident_lane, dict):
        raise ValueError("Atom resident language lane is absent")
    if (
        resident_lane.get("runtime") != "atom-resident-language-lane-v1"
        or resident_lane.get("topology") != "spiderweb-permanent-elevated-language-lane"
        or resident_lane.get("host") != "127.0.0.1"
        or resident_lane.get("api_key_in_memory_only") is not True
        or resident_lane.get("external_proxy_disabled") is not True
        or resident_lane.get("preload_inference_path") is not True
        or resident_lane.get("automatic_restart_on_next_request") is not True
        or resident_lane.get("web_ui_enabled") is not False
        or resident_lane.get("parallel_slots") != 1
        or resident_lane.get("max_queue_depth") != 8
        or resident_lane.get("startup_timeout_seconds") != 180
        or resident_lane.get("acquire_timeout_seconds") != 30
    ):
        raise ValueError("Atom resident language lane policy is invalid")
    if certification.get("script") != "scripts/certify_resident_language_lane.py":
        raise ValueError("Atom resident certification script is invalid")
    required_surfaces = certification.get("required_surfaces")
    if not isinstance(required_surfaces, list) or not {
        "wiki-graph",
        "artifact-side-view",
        "model-load-latency",
        "generation-throughput",
        "single-model-load",
        "warm-request-reuse",
        "resident-backpressure",
        "crash-recovery",
        "expanded-domain-matrix",
        "exact-vocabulary-intent-assistance",
    } <= set(required_surfaces):
        raise ValueError("Atom language-model certification surfaces are incomplete")
    if payload.get("adoption_status") != "certified-resident-local-default":
        raise ValueError("Atom language-model adoption status is invalid")
    latest_evidence = certification.get("latest_evidence")
    if not isinstance(latest_evidence, dict):
        raise ValueError("Atom language-model certification evidence is absent")
    if latest_evidence.get("runtime") != "atom-resident-language-certification-v1":
        raise ValueError("Atom language-model certification runtime is invalid")
    if (
        not isinstance(latest_evidence.get("report_sha256"), str)
        or _SHA256.fullmatch(latest_evidence["report_sha256"]) is None
    ):
        raise ValueError("Atom language-model certification hash is invalid")
    if latest_evidence.get("context_tokens") != 32_768:
        raise ValueError("Atom language-model certification context is invalid")
    if latest_evidence.get("case_count") != 20:
        raise ValueError("Atom language-model certification case count is invalid")
    if latest_evidence.get("completion_count") != 36:
        raise ValueError("Atom language-model completion count is invalid")
    if latest_evidence.get("domain_count") != 8:
        raise ValueError("Atom language-model domain count is invalid")
    for evidence_field in (
        "all_cases_passed",
        "machine_grounding_passed",
        "wiki_graph_and_rag_passed",
        "artifact_side_view_passed",
        "single_model_load_before_fault_passed",
        "warm_request_reuse_passed",
        "resident_backpressure_passed",
        "crash_recovery_passed",
        "expanded_domain_matrix_passed",
        "exact_vocabulary_intent_assistance_passed",
    ):
        if latest_evidence.get(evidence_field) is not True:
            raise ValueError(
                f"Atom language-model certification failed: {evidence_field}"
            )
    expected_counts = {
        "pre_fault_process_generation": 1,
        "pre_fault_model_load_count": 1,
        "warm_completion_count": 35,
        "recovery_process_generation": 2,
        "recovery_model_load_count": 2,
        "recovery_restart_count": 1,
    }
    for field, expected in expected_counts.items():
        if latest_evidence.get(field) != expected:
            raise ValueError(
                f"Atom language-model certification count is invalid: {field}"
            )
    if (
        type(latest_evidence.get("backpressure_wait_ms")) is not int
        or latest_evidence["backpressure_wait_ms"] <= 0
    ):
        raise ValueError("Atom language-model backpressure evidence is invalid")
    for field in (
        "cold_start_latency_ms",
        "warm_request_latency_ms",
        "generation_throughput_tokens_per_second",
    ):
        metrics = latest_evidence.get(field)
        if (
            not isinstance(metrics, dict)
            or any(
                isinstance(metrics.get(item), bool)
                or not isinstance(metrics.get(item), (int, float))
                or metrics[item] <= 0
                for item in ("minimum", "median", "maximum")
            )
            or not metrics["minimum"] <= metrics["median"] <= metrics["maximum"]
        ):
            raise ValueError(
                f"Atom language-model certification metrics are invalid: {field}"
            )
    return payload


def default_official_model_path(project_root: Path | None = None) -> Path:
    """Return the declared sibling model-store path for this checkout."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    relative = load_language_model_contract()["artifact"]["default_relative_path"]
    return (root / str(relative)).resolve()


def resolve_model_integrity(
    model_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> tuple[str, int | None]:
    """Resolve explicit integrity data, with the official artifact as a safe default."""

    contract = load_language_model_contract()
    artifact = contract["artifact"]
    if Path(model_path).name.casefold() == str(artifact["filename"]).casefold():
        expected_sha256 = expected_sha256 or str(artifact["sha256"])
        expected_bytes = (
            int(artifact["bytes"]) if expected_bytes is None else expected_bytes
        )
    if not isinstance(expected_sha256, str):
        raise ValueError("local GGUF requires an expected SHA-256")
    normalized = expected_sha256.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError("local GGUF expected SHA-256 is invalid")
    if expected_bytes is not None and (
        type(expected_bytes) is not int or expected_bytes <= 0
    ):
        raise ValueError("local GGUF expected byte count is invalid")
    return normalized, expected_bytes


def resolve_chat_template(
    model_path: Path,
    *,
    chat_template: str | None = None,
) -> str:
    """Resolve an explicit prompt transport, with a bound official default."""

    contract = load_language_model_contract()
    artifact = contract["artifact"]
    is_official = (
        Path(model_path).name.casefold() == str(artifact["filename"]).casefold()
    )
    if chat_template is None and is_official:
        chat_template = str(contract["runtime_policy"]["chat_template"])
    if chat_template is None:
        raise ValueError("custom local GGUF requires an explicit chat template")
    normalized = str(chat_template).strip()
    if normalized not in SUPPORTED_LLAMA_CPP_CHAT_TEMPLATES:
        raise ValueError("local GGUF chat template is unsupported")
    if is_official and normalized != QWEN_CHATML_MANUAL_TEMPLATE:
        raise ValueError("official Atom GGUF requires its declared chat template")
    return normalized
