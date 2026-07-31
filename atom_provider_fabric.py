"""Resilient Spiderweb provider fabric for the Atom language membrane."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import canonical_hash
from atom_llm_protocol import (
    ATOM_LANGUAGE_MODEL_PROTOCOL,
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    JsonLanguageModel,
    LanguageBoundaryError,
    ProviderCancelledError,
    ProviderCapabilities,
    ProviderError,
    ProviderExhaustedError,
    ProviderInternalError,
    ProviderLocation,
)
from atom_resident_language_lane import ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME


ATOM_PROVIDER_FABRIC_RUNTIME = "atom-resilient-provider-fabric-v3"
ATOM_PROVIDER_ROUTE_RUNTIME = "atom-provider-route-v3"

_PROVIDER_MANIFEST_FIELDS = frozenset(
    {
        "api_key_env",
        "api_key_present",
        "available",
        "capabilities",
        "chat_template",
        "context_length",
        "endpoint",
        "gpu_layers",
        "model",
        "model_bytes",
        "model_sha256",
        "protocol",
        "provider_name",
        "provider_runtime",
        "reason_code",
        "reason_sha256",
        "resident_lane",
        "require_parameters",
        "schema",
        "secrets_persisted",
        "structured_output",
        "test_only",
    }
)


@dataclass(frozen=True)
class ProviderFabricPolicy:
    """Fail-closed admission, privacy, retry, and concurrency policy."""

    allowed_locations: frozenset[ProviderLocation] = frozenset(
        {ProviderLocation.LOCAL, ProviderLocation.PRIVATE}
    )
    allow_cloud_data: bool = False
    allow_test_providers: bool = False
    max_retries_per_provider: int = 1
    retry_backoff_seconds: float = 0.25
    circuit_failure_threshold: int = 1
    circuit_cooldown_seconds: float = 60.0
    max_concurrency: int = 2
    acquire_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        locations = frozenset(ProviderLocation(item) for item in self.allowed_locations)
        object.__setattr__(self, "allowed_locations", locations)
        if not 0 <= self.max_retries_per_provider <= 3:
            raise ValueError("provider retry count must be between zero and three")
        if not 0 <= self.retry_backoff_seconds <= 30:
            raise ValueError("provider retry backoff is outside the supported range")
        if not 1 <= self.circuit_failure_threshold <= 10:
            raise ValueError("circuit failure threshold must be between one and ten")
        if not 0.1 <= self.circuit_cooldown_seconds <= 86_400:
            raise ValueError("circuit cooldown is outside the supported range")
        if not 1 <= self.max_concurrency <= 64:
            raise ValueError("provider concurrency must be between one and 64")
        if not 0.1 <= self.acquire_timeout_seconds <= 3_600:
            raise ValueError("provider acquire timeout is outside the supported range")
        if ProviderLocation.CLOUD in locations and not self.allow_cloud_data:
            raise ValueError(
                "cloud location cannot be admitted without explicit cloud-data consent"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "allowed_locations": sorted(item.value for item in self.allowed_locations),
            "allow_cloud_data": self.allow_cloud_data,
            "allow_test_providers": self.allow_test_providers,
            "max_retries_per_provider": self.max_retries_per_provider,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "circuit_failure_threshold": self.circuit_failure_threshold,
            "circuit_cooldown_seconds": self.circuit_cooldown_seconds,
            "max_concurrency": self.max_concurrency,
            "acquire_timeout_seconds": self.acquire_timeout_seconds,
        }


@dataclass
class _Circuit:
    failures: int = 0
    state: str = "closed"
    opened_monotonic: float | None = None
    opened_epoch_ms: int | None = None
    half_open_trial_active: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failures": self.failures,
            "opened_epoch_ms": self.opened_epoch_ms,
            "half_open_trial_active": self.half_open_trial_active,
        }


def _error_kind(error: BaseException) -> tuple[str, bool]:
    if isinstance(error, LanguageBoundaryError):
        return "boundary", False
    if isinstance(error, ProviderError):
        return error.failure_kind, error.retryable
    return "internal", False


def _error_hash(error: BaseException) -> str:
    identity = f"{type(error).__name__}:{error}"
    return hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()


def _validated_provider_manifest(
    provider: JsonLanguageModel,
    capabilities: ProviderCapabilities,
) -> dict[str, Any]:
    manifest = dict(provider.manifest())
    extra = sorted(set(manifest) - _PROVIDER_MANIFEST_FIELDS)
    if extra:
        raise ValueError(
            "provider manifest contains undeclared fields: " + ", ".join(extra)
        )
    if type(manifest.get("schema")) is not int or manifest["schema"] != 1:
        raise ValueError("provider manifest schema is invalid")
    if manifest.get("protocol") != ATOM_LANGUAGE_MODEL_PROTOCOL:
        raise ValueError("provider manifest protocol is invalid")
    if manifest.get("secrets_persisted") is not False:
        raise ValueError("provider manifest does not guarantee secret redaction")
    if manifest.get("capabilities") != capabilities.manifest():
        raise ValueError("provider manifest capabilities do not match admission facts")
    if manifest.get("model") != capabilities.model:
        raise ValueError("provider manifest model does not match admission facts")
    if type(manifest.get("available")) is not bool:
        raise ValueError("provider manifest availability is invalid")
    if (
        not isinstance(manifest.get("provider_runtime"), str)
        or not manifest["provider_runtime"]
    ):
        raise ValueError("provider manifest runtime is invalid")
    if "api_key_present" in manifest and type(manifest["api_key_present"]) is not bool:
        raise ValueError("provider manifest credential-presence flag is invalid")
    if manifest.get("test_only", False) is not capabilities.test_only:
        raise ValueError("provider manifest test-only flag is invalid")
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ValueError("provider manifest exceeds the safe byte limit")
    canonical_hash(manifest)
    return manifest


def _validated_lane_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    lane = dict(payload)
    if not lane:
        return {}
    if type(lane.get("schema")) is not int or lane["schema"] != 1:
        raise LanguageBoundaryError("provider lane evidence schema is invalid")
    if (
        not isinstance(lane.get("runtime"), str)
        or not lane["runtime"].strip()
        or len(lane["runtime"]) > 256
    ):
        raise LanguageBoundaryError("provider lane evidence runtime is invalid")
    if lane["runtime"] != ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME:
        raise LanguageBoundaryError("provider lane evidence runtime is unsupported")
    for field in ("stage",):
        value = lane.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or "\x00" in value
            or len(value) > 128
        ):
            raise LanguageBoundaryError(f"provider lane {field} is invalid")
    for field, minimum in (
        ("process_generation", 1),
        ("model_load_count", 1),
        ("restart_count", 0),
        ("request_ordinal", 1),
        ("queue_wait_ms", 0),
    ):
        value = lane.get(field)
        if type(value) is not int or value < minimum:
            raise LanguageBoundaryError(f"provider lane {field} is invalid")
    if lane["model_load_count"] > lane["process_generation"]:
        raise LanguageBoundaryError("provider lane model-load count is impossible")
    if lane["restart_count"] >= lane["model_load_count"]:
        raise LanguageBoundaryError("provider lane restart count is impossible")
    if type(lane.get("resident_reused")) is not bool:
        raise LanguageBoundaryError("provider lane reuse flag is invalid")
    if lane["resident_reused"] and lane["request_ordinal"] <= 1:
        raise LanguageBoundaryError("provider lane reuse evidence is impossible")
    expected_ramps = {
        "on_ramp": {
            "from": "L1:typed-language-message",
            "to": "resident-language-highway",
            "message": "JsonGenerationRequest",
        },
        "off_ramp": {
            "from": "resident-language-highway",
            "to": "L1:typed-language-result",
            "message": "JsonGenerationResult",
        },
    }
    for field, expected in expected_ramps.items():
        if lane.get(field) != expected:
            raise LanguageBoundaryError(f"provider lane {field} is invalid")
    envelope_sha256 = lane.get("transport_envelope_sha256")
    if envelope_sha256 is not None and (
        not isinstance(envelope_sha256, str)
        or len(envelope_sha256) != 64
        or any(character not in "0123456789abcdef" for character in envelope_sha256)
    ):
        raise LanguageBoundaryError("provider lane transport hash is invalid")
    vibrations = lane.get("vibrations")
    if (
        not isinstance(vibrations, list)
        or len(vibrations) > 64
        or any(not isinstance(item, Mapping) for item in vibrations)
    ):
        raise LanguageBoundaryError("provider lane vibrations are invalid")
    for vibration in vibrations:
        if vibration.get("kind") not in {"horizontal", "vertical"}:
            raise LanguageBoundaryError("provider lane vibration kind is invalid")
        for field in ("signal", "origin"):
            value = vibration.get(field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or len(value) > 256
            ):
                raise LanguageBoundaryError(
                    f"provider lane vibration {field} is invalid"
                )
        propagates_to = vibration.get("propagates_to")
        if (
            not isinstance(propagates_to, list)
            or len(propagates_to) > 16
            or any(
                not isinstance(item, str)
                or not item.strip()
                or "\x00" in item
                or len(item) > 256
                for item in propagates_to
            )
        ):
            raise LanguageBoundaryError(
                "provider lane vibration propagation is invalid"
            )
    encoded = json.dumps(
        lane,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise LanguageBoundaryError("provider lane evidence exceeds the safe limit")
    canonical_hash(lane)
    return lane


class ProviderFabric:
    """Route strict JSON requests through an ordered, policy-bound fabric."""

    def __init__(
        self,
        providers: Sequence[JsonLanguageModel],
        *,
        policy: ProviderFabricPolicy | None = None,
    ) -> None:
        if not 1 <= len(providers) <= 32:
            raise ValueError("provider fabric needs between one and 32 providers")
        self.providers = tuple(providers)
        self.policy = policy or ProviderFabricPolicy()
        self._semaphore = threading.BoundedSemaphore(self.policy.max_concurrency)
        self._lock = threading.RLock()
        self._closed = False
        self._circuits: dict[str, _Circuit] = {}
        self._provider_keys: list[str] = []
        capabilities_rows: list[ProviderCapabilities] = []
        provider_manifests: list[dict[str, Any]] = []
        for index, provider in enumerate(self.providers):
            capabilities = provider.capabilities()
            manifest = _validated_provider_manifest(provider, capabilities)
            key = (
                f"{index}:{capabilities.provider_id}:"
                f"{capabilities.model}:{capabilities.location.value}"
            )
            self._provider_keys.append(key)
            self._circuits[key] = _Circuit()
            capabilities_rows.append(capabilities)
            provider_manifests.append(manifest)
        self._capabilities = tuple(capabilities_rows)
        self._provider_manifests = tuple(provider_manifests)

    def capabilities(self) -> ProviderCapabilities:
        eligible = [
            item
            for item in self._capabilities
            if item.location in self.policy.allowed_locations
            and (not item.test_only or self.policy.allow_test_providers)
            and item.strict_json_schema
        ]
        return ProviderCapabilities(
            provider_id=ATOM_PROVIDER_FABRIC_RUNTIME,
            model="ordered-resilient-fabric",
            location=ProviderLocation.PRIVATE,
            strict_json_schema=True,
            max_context_tokens=max(
                (item.max_context_tokens for item in eligible),
                default=0,
            ),
            max_output_tokens=max(
                (item.max_output_tokens for item in eligible),
                default=0,
            ),
            supports_cancellation=bool(eligible)
            and all(item.supports_cancellation for item in eligible),
            cost_tier="policy-routed",
            test_only=bool(eligible) and all(item.test_only for item in eligible),
        )

    def preload_manifest(self) -> dict[str, Any]:
        """Preload provider capabilities before any question data is routed."""

        providers: list[dict[str, Any]] = []
        with self._lock:
            for key, capabilities, manifest in zip(
                self._provider_keys,
                self._capabilities,
                self._provider_manifests,
                strict=True,
            ):
                providers.append(
                    {
                        "provider_key": key,
                        "capabilities": capabilities.manifest(),
                        "manifest": dict(manifest),
                        "circuit": self._circuits[key].manifest(),
                    }
                )
        identity_core = {
            "schema": 1,
            "runtime": ATOM_PROVIDER_FABRIC_RUNTIME,
            "protocol": ATOM_LANGUAGE_MODEL_PROTOCOL,
            "ordered": True,
            "policy": self.policy.manifest(),
            "providers": [
                {key: value for key, value in provider.items() if key != "circuit"}
                for provider in providers
            ],
        }
        state_core = {
            **identity_core,
            "providers": providers,
        }
        return {
            **state_core,
            "preload_hash": canonical_hash(identity_core),
            "state_hash": canonical_hash(state_core),
        }

    def preload_runtime(self) -> dict[str, Any]:
        """Warm admitted permanent lanes without routing question data."""

        with self._lock:
            if self._closed:
                raise RuntimeError("provider fabric is closed")
        providers: list[dict[str, Any]] = []
        for key, capabilities, provider in zip(
            self._provider_keys,
            self._capabilities,
            self.providers,
            strict=True,
        ):
            admitted = (
                capabilities.location in self.policy.allowed_locations
                and (not capabilities.test_only or self.policy.allow_test_providers)
                and capabilities.strict_json_schema
            )
            preload = getattr(provider, "preload", None)
            if admitted and callable(preload):
                evidence = dict(preload())
                mode = "runtime-warmed"
            elif admitted:
                evidence = {"manifest": dict(provider.manifest())}
                mode = "manifest-only"
            else:
                evidence = {}
                mode = "policy-excluded"
            providers.append(
                {
                    "provider_key": key,
                    "provider_id": capabilities.provider_id,
                    "model": capabilities.model,
                    "location": capabilities.location.value,
                    "admitted": admitted,
                    "preload_mode": mode,
                    "evidence": evidence,
                }
            )
        core = {
            "schema": 1,
            "runtime": ATOM_PROVIDER_FABRIC_RUNTIME,
            "operation": "provider-fabric-runtime-preload",
            "providers": providers,
            "secrets_persisted": False,
        }
        return {**core, "preload_hash": canonical_hash(core)}

    def manifest(self) -> Mapping[str, Any]:
        preload = self.preload_manifest()
        return {
            "schema": 1,
            "protocol": ATOM_LANGUAGE_MODEL_PROTOCOL,
            "provider_runtime": ATOM_PROVIDER_FABRIC_RUNTIME,
            "model": "ordered-resilient-fabric",
            "ordered": True,
            "provider_count": len(self.providers),
            "policy": preload["policy"],
            "providers": preload["providers"],
            "capabilities": self.capabilities().manifest(),
            "preload_hash": preload["preload_hash"],
            "state_hash": preload["state_hash"],
            "secrets_persisted": False,
        }

    def close(self) -> None:
        """Release resources owned by all providers in reverse route order."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        for provider in reversed(self.providers):
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> ProviderFabric:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _circuit_allows(self, key: str) -> tuple[bool, str]:
        with self._lock:
            circuit = self._circuits[key]
            if circuit.state == "closed":
                return True, "closed"
            if circuit.state == "half-open":
                if circuit.half_open_trial_active:
                    return False, "half-open-busy"
                circuit.half_open_trial_active = True
                return True, "half-open"
            assert circuit.opened_monotonic is not None
            elapsed = time.monotonic() - circuit.opened_monotonic
            if elapsed >= self.policy.circuit_cooldown_seconds:
                circuit.state = "half-open"
                circuit.half_open_trial_active = True
                return True, "half-open"
            return False, "open"

    def _circuit_manifest(self, key: str) -> dict[str, Any]:
        with self._lock:
            return self._circuits[key].manifest()

    def _record_success(self, key: str) -> dict[str, Any]:
        with self._lock:
            circuit = self._circuits[key]
            circuit.failures = 0
            circuit.state = "closed"
            circuit.opened_monotonic = None
            circuit.opened_epoch_ms = None
            circuit.half_open_trial_active = False
            return circuit.manifest()

    def _record_failure(self, key: str) -> dict[str, Any]:
        with self._lock:
            circuit = self._circuits[key]
            circuit.failures += 1
            if circuit.failures >= self.policy.circuit_failure_threshold:
                circuit.state = "open"
                circuit.opened_monotonic = time.monotonic()
                circuit.opened_epoch_ms = round(time.time() * 1000)
            circuit.half_open_trial_active = False
            return circuit.manifest()

    def _record_cancelled(self, key: str) -> dict[str, Any]:
        with self._lock:
            circuit = self._circuits[key]
            if circuit.state == "half-open":
                circuit.state = "open"
                circuit.opened_monotonic = time.monotonic()
                circuit.opened_epoch_ms = round(time.time() * 1000)
            circuit.half_open_trial_active = False
            return circuit.manifest()

    def _admission_reason(
        self,
        capabilities: ProviderCapabilities,
        manifest: Mapping[str, Any],
        request: JsonGenerationRequest,
    ) -> tuple[str | None, str | None]:
        if capabilities.location not in self.policy.allowed_locations:
            return (
                "privacy",
                f"{capabilities.location.value} provider location is not allowed",
            )
        if (
            capabilities.location is ProviderLocation.CLOUD
            and not self.policy.allow_cloud_data
        ):
            return "privacy", "cloud data egress lacks explicit consent"
        if capabilities.test_only and not self.policy.allow_test_providers:
            return "admission", "test-only provider is forbidden in production"
        if not capabilities.strict_json_schema:
            return "admission", "provider lacks strict JSON-schema generation"
        if request.max_tokens > capabilities.max_output_tokens:
            return "admission", "request exceeds provider output-token capability"
        input_bytes = len(request.system_prompt.encode("utf-8")) + len(
            json.dumps(
                {
                    "payload": request.payload,
                    "schema": request.schema,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        estimated_input_tokens = max(1, (input_bytes + 2) // 3)
        if (
            estimated_input_tokens + request.max_tokens
            > capabilities.max_context_tokens
        ):
            return "admission", "request exceeds provider context capability"
        if manifest.get("available") is False:
            return "admission", "provider reports unavailable"
        return None, None

    @staticmethod
    def _finalize_route(core: Mapping[str, Any]) -> dict[str, Any]:
        route_core = dict(core)
        return {**route_core, "route_hash": canonical_hash(route_core)}

    def _acquire(
        self,
        *,
        cancellation: CancellationToken,
        vibrations: list[dict[str, Any]],
    ) -> None:
        started = time.monotonic()
        deadline = started + self.policy.acquire_timeout_seconds
        while True:
            cancellation.raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderExhaustedError(
                    "provider fabric concurrency admission timed out"
                )
            if self._semaphore.acquire(timeout=min(0.05, remaining)):
                waited_ms = round((time.monotonic() - started) * 1000)
                if waited_ms > 0:
                    vibrations.append(
                        {
                            "kind": "vertical",
                            "signal": "provider-backpressure",
                            "origin": "L0:provider-semaphore",
                            "propagates_to": [
                                "L2:provider-route",
                                "L3:orchestration",
                            ],
                            "waited_ms": waited_ms,
                        }
                    )
                return

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        token = cancellation or CancellationToken()
        token.raise_if_cancelled()
        with self._lock:
            if self._closed:
                raise ProviderExhaustedError("provider fabric is closed")
        started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        vibrations: list[dict[str, Any]] = []
        acquired = False
        try:
            try:
                self._acquire(cancellation=token, vibrations=vibrations)
            except ProviderExhaustedError as error:
                vibrations.append(
                    {
                        "kind": "vertical",
                        "signal": "provider-backpressure-timeout",
                        "origin": "L0:provider-semaphore",
                        "propagates_to": ["L3:orchestration"],
                        "waited_ms": round((time.perf_counter() - started) * 1000),
                    }
                )
                route = self._finalize_route(
                    {
                        "schema": 1,
                        "runtime": ATOM_PROVIDER_ROUTE_RUNTIME,
                        "stage": request.stage,
                        "data_sensitivity": request.data_sensitivity,
                        "completed": False,
                        "disposition": "exhausted",
                        "selected_provider": None,
                        "attempts": attempts,
                        "vibrations": vibrations,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    }
                )
                raise ProviderExhaustedError(
                    "provider fabric concurrency admission timed out",
                    route=route,
                ) from error
            acquired = True
            previous_failed = False
            for index, (key, provider, capabilities, manifest) in enumerate(
                zip(
                    self._provider_keys,
                    self.providers,
                    self._capabilities,
                    self._provider_manifests,
                    strict=True,
                )
            ):
                if previous_failed:
                    vibrations.append(
                        {
                            "kind": "vertical",
                            "signal": "provider-fallback",
                            "origin": "L2:provider-route",
                            "propagates_to": ["L3:orchestration"],
                            "provider_key": key,
                        }
                    )
                circuit_admitted, circuit_state = self._circuit_allows(key)
                if not circuit_admitted:
                    attempts.append(
                        {
                            "ordinal": len(attempts),
                            "provider_index": index,
                            "provider_key": key,
                            **capabilities.manifest(),
                            "attempt": 0,
                            "outcome": "skipped",
                            "failure_kind": (
                                "circuit-half-open-busy"
                                if circuit_state == "half-open-busy"
                                else "circuit-open"
                            ),
                            "retryable": False,
                            "error_sha256": None,
                            "elapsed_ms": 0,
                            "circuit_before": circuit_state,
                            "circuit_after": self._circuit_manifest(key),
                        }
                    )
                    vibrations.append(
                        {
                            "kind": "vertical",
                            "signal": (
                                "provider-circuit-half-open-busy"
                                if circuit_state == "half-open-busy"
                                else "provider-circuit-open"
                            ),
                            "origin": "L2:provider-route",
                            "propagates_to": ["L3:orchestration"],
                            "provider_key": key,
                        }
                    )
                    previous_failed = True
                    continue
                failure_kind, admission_reason = self._admission_reason(
                    capabilities,
                    manifest,
                    request,
                )
                if failure_kind is not None:
                    attempts.append(
                        {
                            "ordinal": len(attempts),
                            "provider_index": index,
                            "provider_key": key,
                            **capabilities.manifest(),
                            "attempt": 0,
                            "outcome": "blocked",
                            "failure_kind": failure_kind,
                            "retryable": False,
                            "error_sha256": hashlib.sha256(
                                str(admission_reason).encode("utf-8")
                            ).hexdigest(),
                            "elapsed_ms": 0,
                            "circuit_before": circuit_state,
                            "circuit_after": self._circuit_manifest(key),
                        }
                    )
                    vibrations.append(
                        {
                            "kind": "vertical",
                            "signal": (
                                "provider-privacy-block"
                                if failure_kind == "privacy"
                                else "provider-admission-block"
                            ),
                            "origin": "L1:provider-capability",
                            "propagates_to": [
                                "L2:provider-route",
                                "L3:orchestration",
                            ],
                            "provider_key": key,
                        }
                    )
                    previous_failed = True
                    continue

                for provider_attempt in range(
                    1,
                    self.policy.max_retries_per_provider + 2,
                ):
                    token.raise_if_cancelled()
                    attempt_circuit_before = self._circuit_manifest(key)["state"]
                    attempt_started = time.perf_counter()
                    try:
                        result = provider.generate_json(
                            request,
                            cancellation=token,
                        )
                        if (
                            result.provider != capabilities.provider_id
                            or result.model != capabilities.model
                        ):
                            raise LanguageBoundaryError(
                                "provider result identity differs from "
                                "its admitted capabilities"
                            )
                        normalized = (
                            request.validator(result.payload)
                            if request.validator is not None
                            else result.payload
                        )
                        lane_evidence = _validated_lane_evidence(result.lane)
                        for vibration in lane_evidence.get("vibrations", []):
                            vibrations.append(
                                {
                                    **dict(vibration),
                                    "provider_key": key,
                                }
                            )
                    except ProviderCancelledError as error:
                        circuit = self._record_cancelled(key)
                        attempts.append(
                            {
                                "ordinal": len(attempts),
                                "provider_index": index,
                                "provider_key": key,
                                **capabilities.manifest(),
                                "attempt": provider_attempt,
                                "outcome": "cancelled",
                                "failure_kind": "cancelled",
                                "retryable": False,
                                "error_sha256": _error_hash(error),
                                "elapsed_ms": round(
                                    (time.perf_counter() - attempt_started) * 1000
                                ),
                                "circuit_before": attempt_circuit_before,
                                "circuit_after": circuit,
                            }
                        )
                        route = self._finalize_route(
                            {
                                "schema": 1,
                                "runtime": ATOM_PROVIDER_ROUTE_RUNTIME,
                                "stage": request.stage,
                                "data_sensitivity": request.data_sensitivity,
                                "completed": False,
                                "disposition": "cancelled",
                                "selected_provider": None,
                                "attempts": attempts,
                                "vibrations": vibrations,
                                "elapsed_ms": round(
                                    (time.perf_counter() - started) * 1000
                                ),
                            }
                        )
                        raise ProviderCancelledError(
                            "provider route cancelled",
                            route=route,
                        ) from error
                    except Exception as error:
                        kind, retryable = _error_kind(error)
                        if not isinstance(
                            error, (ProviderError, LanguageBoundaryError)
                        ):
                            error = ProviderInternalError(
                                f"provider raised {type(error).__name__}"
                            )
                            kind, retryable = _error_kind(error)
                        circuit = self._record_failure(key)
                        elapsed_ms = round(
                            (time.perf_counter() - attempt_started) * 1000
                        )
                        attempts.append(
                            {
                                "ordinal": len(attempts),
                                "provider_index": index,
                                "provider_key": key,
                                **capabilities.manifest(),
                                "attempt": provider_attempt,
                                "outcome": "failed",
                                "failure_kind": kind,
                                "retryable": retryable,
                                "error_sha256": _error_hash(error),
                                "elapsed_ms": elapsed_ms,
                                "circuit_before": attempt_circuit_before,
                                "circuit_after": circuit,
                            }
                        )
                        vibrations.append(
                            {
                                "kind": "vertical",
                                "signal": f"provider-{kind}-failure",
                                "origin": "L2:provider-route",
                                "propagates_to": ["L3:orchestration"],
                                "provider_key": key,
                            }
                        )
                        should_retry = (
                            retryable
                            and provider_attempt <= self.policy.max_retries_per_provider
                            and attempt_circuit_before != "half-open"
                        )
                        if should_retry:
                            delay_seconds = min(
                                self.policy.retry_backoff_seconds
                                * (2 ** (provider_attempt - 1)),
                                30.0,
                            )
                            vibrations.append(
                                {
                                    "kind": "horizontal",
                                    "signal": "provider-retry",
                                    "origin": "L2:provider-route",
                                    "propagates_to": ["L2:provider-route"],
                                    "provider_key": key,
                                    "next_attempt": provider_attempt + 1,
                                    "delay_ms": round(delay_seconds * 1000),
                                }
                            )
                            token.wait(delay_seconds)
                            continue
                        previous_failed = True
                        break
                    else:
                        circuit = self._record_success(key)
                        attempts.append(
                            {
                                "ordinal": len(attempts),
                                "provider_index": index,
                                "provider_key": key,
                                **capabilities.manifest(),
                                "attempt": provider_attempt,
                                "outcome": "completed",
                                "failure_kind": None,
                                "retryable": False,
                                "error_sha256": None,
                                "elapsed_ms": round(
                                    (time.perf_counter() - attempt_started) * 1000
                                ),
                                "circuit_before": attempt_circuit_before,
                                "circuit_after": circuit,
                            }
                        )
                        route = self._finalize_route(
                            {
                                "schema": 1,
                                "runtime": ATOM_PROVIDER_ROUTE_RUNTIME,
                                "stage": request.stage,
                                "data_sensitivity": request.data_sensitivity,
                                "completed": True,
                                "disposition": "completed",
                                "selected_provider": {
                                    "provider_key": key,
                                    **capabilities.manifest(),
                                },
                                "language_lane": lane_evidence or None,
                                "attempts": attempts,
                                "vibrations": vibrations,
                                "elapsed_ms": round(
                                    (time.perf_counter() - started) * 1000
                                ),
                            }
                        )
                        return replace(
                            result,
                            payload=dict(normalized),
                            route=route,
                        )

            route = self._finalize_route(
                {
                    "schema": 1,
                    "runtime": ATOM_PROVIDER_ROUTE_RUNTIME,
                    "stage": request.stage,
                    "data_sensitivity": request.data_sensitivity,
                    "completed": False,
                    "disposition": "exhausted",
                    "selected_provider": None,
                    "attempts": attempts,
                    "vibrations": vibrations,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            raise ProviderExhaustedError(
                f"all admitted providers failed during {request.stage}",
                route=route,
            )
        finally:
            if acquired:
                self._semaphore.release()
