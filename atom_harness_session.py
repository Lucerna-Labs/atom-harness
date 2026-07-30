"""Reusable multi-request session for the resident Atom language harness."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_harness_experiment import run_atom_language_harness
from atom_language_model_contract import (
    default_official_model_path,
    load_language_model_contract,
    resolve_chat_template,
    resolve_model_integrity,
)
from atom_llm_provider import LlamaCppResidentJsonLanguageModel
from atom_llm_protocol import CancellationToken, ProviderLocation
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy


ATOM_HARNESS_SESSION_RUNTIME = "atom-resident-language-harness-session-v1"


class AtomHarnessSession:
    """Own one provider fabric and reuse its resident lane across requests."""

    def __init__(
        self,
        *,
        provider_fabric: ProviderFabric,
        output_root: Path,
        forge_path: Path = DEFAULT_FORGE,
        evidence_path: Path = DEFAULT_EVIDENCE,
        model_path: Path = DEFAULT_MODEL,
    ) -> None:
        self.provider_fabric = provider_fabric
        self.output_root = Path(output_root).resolve()
        self.forge_path = Path(forge_path).resolve()
        self.evidence_path = Path(evidence_path).resolve()
        self.model_path = Path(model_path).resolve()
        self._lock = threading.RLock()
        self._request_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._closed = False

    @classmethod
    def official_local(
        cls,
        *,
        output_root: Path,
        model_path: Path | None = None,
        llama_server: str | None = None,
        gpu_layers: str = "auto",
        provider_timeout_seconds: int = 240,
        startup_timeout_seconds: int | None = None,
        lane_acquire_timeout_seconds: float | None = None,
        parallel_slots: int | None = None,
        max_queue_depth: int | None = None,
        max_concurrency: int = 2,
    ) -> AtomHarnessSession:
        contract = load_language_model_contract()
        runtime_policy = contract["runtime_policy"]
        lane_policy = runtime_policy["resident_lane"]
        selected_model = (
            Path(model_path)
            if model_path is not None
            else default_official_model_path()
        )
        expected_sha256, expected_bytes = resolve_model_integrity(selected_model)
        chat_template = resolve_chat_template(selected_model)
        provider = LlamaCppResidentJsonLanguageModel(
            selected_model,
            executable=llama_server or str(runtime_policy["executable"]),
            expected_model_sha256=expected_sha256,
            expected_model_bytes=expected_bytes,
            chat_template=chat_template,
            context_length=int(runtime_policy["harness_context_tokens"]),
            gpu_layers=gpu_layers,
            timeout_seconds=provider_timeout_seconds,
            startup_timeout_seconds=(
                int(startup_timeout_seconds)
                if startup_timeout_seconds is not None
                else int(lane_policy["startup_timeout_seconds"])
            ),
            lane_acquire_timeout_seconds=(
                float(lane_acquire_timeout_seconds)
                if lane_acquire_timeout_seconds is not None
                else float(lane_policy["acquire_timeout_seconds"])
            ),
            parallel_slots=(
                int(parallel_slots)
                if parallel_slots is not None
                else int(lane_policy["parallel_slots"])
            ),
            max_queue_depth=(
                int(max_queue_depth)
                if max_queue_depth is not None
                else int(lane_policy["max_queue_depth"])
            ),
        )
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                max_retries_per_provider=1,
                retry_backoff_seconds=0.25,
                circuit_failure_threshold=1,
                circuit_cooldown_seconds=0.1,
                max_concurrency=max_concurrency,
                acquire_timeout_seconds=30,
            ),
        )
        return cls(provider_fabric=fabric, output_root=output_root)

    def _claim_output_dir(
        self,
        question: str,
        output_dir: Path | None,
    ) -> Path:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
        with self._lock:
            if self._closed:
                raise RuntimeError("Atom harness session is closed")
            self._request_count += 1
            ordinal = self._request_count
        if output_dir is not None:
            return Path(output_dir).resolve()
        return self.output_root / f"request-{ordinal:04d}-{digest}"

    def answer(
        self,
        question: str,
        *,
        output_dir: Path | None = None,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        target = self._claim_output_dir(question, output_dir)
        try:
            artifact = run_atom_language_harness(
                target,
                question=question,
                language_model=self.provider_fabric,
                forge_path=self.forge_path,
                evidence_path=self.evidence_path,
                model_path=self.model_path,
                cancellation=cancellation,
            )
        except BaseException:
            with self._lock:
                self._failed_count += 1
            raise
        with self._lock:
            self._completed_count += 1
        return artifact

    def manifest(self) -> Mapping[str, Any]:
        with self._lock:
            counts = {
                "request_count": self._request_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "closed": self._closed,
            }
        providers = []
        for provider in self.provider_fabric.providers:
            snapshot = getattr(provider, "lane_snapshot", None)
            providers.append(
                {
                    "provider": provider.capabilities().provider_id,
                    "lane": dict(snapshot()) if callable(snapshot) else None,
                }
            )
        return {
            "schema": 1,
            "runtime": ATOM_HARNESS_SESSION_RUNTIME,
            "created_output_root": str(self.output_root),
            **counts,
            "providers": providers,
            "secrets_persisted": False,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.provider_fabric.close()

    def __enter__(self) -> AtomHarnessSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def default_session_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return Path("atom_harness_outputs") / f"resident-session-{stamp}"
