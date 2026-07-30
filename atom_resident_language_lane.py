"""Supervised resident llama.cpp lane for the Atom language membrane."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from atom_llm_protocol import (
    CancellationToken,
    ProviderAdmissionError,
    ProviderCancelledError,
    ProviderCapacityError,
    ProviderError,
    ProviderTimeoutError,
    ProviderTransportError,
)


ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME = "atom-resident-language-lane-v1"
ATOM_RESIDENT_LANGUAGE_PERFORMANCE_RUNTIME = "atom-resident-language-performance-v1"
LOOPBACK_HOST = "127.0.0.1"
MAX_SERVER_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_SERVER_ERROR_BYTES = 256 * 1024
MAX_SERVER_STDOUT_BYTES = 2 * 1024 * 1024
MAX_SERVER_STDERR_BYTES = 2 * 1024 * 1024


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_limited(stream, *, limit: int, label: str) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ProviderTransportError(f"{label} exceeds the safe byte limit")
    return data


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        return int(listener.getsockname()[1])


def _finite_number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    rendered = float(value)
    return rendered if math.isfinite(rendered) and rendered >= 0 else default


def _open_loopback(request: urllib.request.Request, *, timeout: float):
    """Open a loopback request without consulting process proxy settings."""

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _safe_timing_metrics(
    timings: Mapping[str, Any],
    *,
    cold_start_ms: int,
    request_elapsed_ms: int,
) -> dict[str, Any]:
    prompt_tokens = int(_finite_number(timings.get("prompt_n")))
    cached_tokens = int(_finite_number(timings.get("cache_n")))
    generated_tokens = int(_finite_number(timings.get("predicted_n")))
    prompt_ms = _finite_number(timings.get("prompt_ms"))
    generation_ms = _finite_number(timings.get("predicted_ms"))
    prompt_rate = _finite_number(timings.get("prompt_per_second"))
    generation_rate = _finite_number(timings.get("predicted_per_second"))
    return {
        "runtime": ATOM_RESIDENT_LANGUAGE_PERFORMANCE_RUNTIME,
        "cold_start_ms": cold_start_ms,
        "model_load_ms": cold_start_ms,
        "warm_request": cold_start_ms == 0,
        "request_elapsed_ms": request_elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "generated_tokens": generated_tokens,
        "prompt_ms": round(prompt_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "prompt_tokens_per_second": round(prompt_rate, 3),
        "generation_tokens_per_second": round(generation_rate, 3),
    }


@dataclass(frozen=True)
class ResidentLaneCompletion:
    """One response demoted from the resident highway to the provider fabric."""

    content: str
    envelope_sha256: str
    elapsed_ms: int
    performance: Mapping[str, Any]
    lane: Mapping[str, Any]


class ResidentLanguageLane:
    """Keep one authenticated loopback llama-server warm across requests."""

    def __init__(
        self,
        model_path: Path,
        *,
        executable: str = "llama-server",
        context_length: int = 32_768,
        gpu_layers: str = "auto",
        startup_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = 240.0,
        acquire_timeout_seconds: float = 30.0,
        parallel_slots: int = 1,
        max_queue_depth: int = 8,
        warmup_prompt: str | None = None,
        warmup_schema: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_executable = shutil.which(executable)
        if resolved_executable is None:
            candidate = Path(executable)
            if not candidate.is_file():
                raise ValueError(f"llama-server executable is absent: {executable}")
            resolved_executable = str(candidate.resolve())
        if Path(resolved_executable).stem.casefold() != "llama-server":
            raise ValueError("resident language lane requires llama-server")
        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file() or resolved_model.suffix.lower() != ".gguf":
            raise ValueError("resident language lane requires a local GGUF")
        if not 1024 <= context_length <= 131_072:
            raise ValueError("resident language lane context is invalid")
        for label, value in (
            ("startup timeout", startup_timeout_seconds),
            ("request timeout", request_timeout_seconds),
            ("acquire timeout", acquire_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"resident language lane {label} is invalid")
            if not 0.1 <= float(value) <= 3_600:
                raise ValueError(f"resident language lane {label} is invalid")
        if not 1 <= parallel_slots <= 16:
            raise ValueError("resident language lane parallel slots are invalid")
        if not 0 <= max_queue_depth <= 256:
            raise ValueError("resident language lane queue depth is invalid")
        if warmup_prompt is not None and (
            not isinstance(warmup_prompt, str)
            or not warmup_prompt.strip()
            or "\x00" in warmup_prompt
        ):
            raise ValueError("resident language lane warmup prompt is invalid")
        if warmup_schema is not None and not isinstance(warmup_schema, Mapping):
            raise ValueError("resident language lane warmup schema is invalid")
        if (warmup_prompt is None) is not (warmup_schema is None):
            raise ValueError("resident language lane warmup must be fully declared")

        self.executable = str(Path(resolved_executable).resolve())
        self.model_path = resolved_model
        self.context_length = context_length
        self.gpu_layers = str(gpu_layers)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.acquire_timeout_seconds = float(acquire_timeout_seconds)
        self.parallel_slots = parallel_slots
        self.max_queue_depth = max_queue_depth
        self.warmup_prompt = warmup_prompt
        self.warmup_schema = dict(warmup_schema or {})

        self._api_key = secrets.token_urlsafe(32)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="atom-resident-language-lane-"
        )
        self._temporary_path = Path(self._temporary.name)
        self._lifecycle_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._admission = threading.BoundedSemaphore(parallel_slots)
        self._process: subprocess.Popen[bytes] | None = None
        self._port: int | None = None
        self._stdout_path: Path | None = None
        self._stderr_path: Path | None = None
        self._closed = False
        self._state = "cold"
        self._process_generation = 0
        self._model_load_count = 0
        self._restart_count = 0
        self._forced_termination_count = 0
        self._request_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self._active_requests = 0
        self._queued_requests = 0
        self._last_cold_start_ms = 0
        self._last_warmup_ms = 0
        self._last_exit_code: int | None = None
        atexit.register(self.close)

    def static_manifest(self) -> dict[str, Any]:
        """Return deterministic lane facts safe for provider admission hashes."""

        return {
            "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
            "topology": "spiderweb-permanent-elevated-language-lane",
            "transport": "authenticated-loopback-http",
            "host": LOOPBACK_HOST,
            "external_proxy_disabled": True,
            "api_key_in_memory_only": True,
            "model_preloaded_once_per_process_generation": True,
            "inference_path_preloaded_before_user_traffic": (
                self.warmup_prompt is not None
            ),
            "automatic_restart_on_next_request": True,
            "typed_on_ramp": "JsonGenerationRequest",
            "typed_off_ramp": "JsonGenerationResult",
            "parallel_slots": self.parallel_slots,
            "max_queue_depth": self.max_queue_depth,
            "context_length": self.context_length,
            "gpu_layers": self.gpu_layers,
            "metrics_endpoint_enabled": True,
            "slot_endpoint_enabled": True,
            "web_ui_enabled": False,
            "secrets_persisted": False,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return dynamic, non-secret supervision evidence."""

        with self._lifecycle_lock, self._state_lock:
            process = self._process
            alive = process is not None and process.poll() is None
            return {
                "schema": 1,
                "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
                "state": self._state,
                "alive": alive,
                "process_generation": self._process_generation,
                "model_load_count": self._model_load_count,
                "restart_count": self._restart_count,
                "forced_termination_count": self._forced_termination_count,
                "request_count": self._request_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "cancelled_count": self._cancelled_count,
                "active_requests": self._active_requests,
                "queued_requests": self._queued_requests,
                "last_cold_start_ms": self._last_cold_start_ms,
                "last_warmup_ms": self._last_warmup_ms,
                "last_exit_code": self._last_exit_code,
                "api_key_persisted": False,
            }

    def _command(self, port: int) -> list[str]:
        return [
            self.executable,
            "--model",
            str(self.model_path),
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(port),
            "--ctx-size",
            str(self.context_length),
            "--n-gpu-layers",
            self.gpu_layers,
            "--parallel",
            str(self.parallel_slots),
            "--cont-batching",
            "--cache-prompt",
            "--metrics",
            "--slots",
            "--no-webui",
            "--log-disable",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
        ]

    def _check_log_limits(self) -> None:
        for path, limit, label in (
            (self._stdout_path, MAX_SERVER_STDOUT_BYTES, "llama-server stdout"),
            (self._stderr_path, MAX_SERVER_STDERR_BYTES, "llama-server stderr"),
        ):
            if path is not None and path.is_file() and path.stat().st_size > limit:
                raise ProviderTransportError(f"{label} exceeds the safe byte limit")

    def _log_detail(self) -> str:
        chunks: list[str] = []
        for path, limit in (
            (self._stderr_path, MAX_SERVER_ERROR_BYTES),
            (self._stdout_path, MAX_SERVER_ERROR_BYTES),
        ):
            if path is None or not path.is_file():
                continue
            try:
                with path.open("rb") as stream:
                    raw = _read_limited(
                        stream,
                        limit=limit,
                        label="llama-server diagnostic",
                    )
                rendered = raw.decode("utf-8", errors="replace").strip()
            except (OSError, ProviderError):
                rendered = ""
            if rendered:
                chunks.append(rendered[-2048:])
        return "\n".join(chunks)[-4096:]

    def _health_ready(self, port: int) -> bool:
        request = urllib.request.Request(
            f"http://{LOOPBACK_HOST}:{port}/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with _open_loopback(request, timeout=1.0) as response:
                raw = _read_limited(
                    response,
                    limit=64 * 1024,
                    label="llama-server health response",
                )
        except urllib.error.HTTPError as error:
            if error.code == 503:
                return False
            raise ProviderTransportError(
                f"llama-server health returned HTTP {error.code}"
            ) from error
        except (TimeoutError, urllib.error.URLError, OSError):
            return False
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProviderTransportError(
                "llama-server health returned invalid JSON"
            ) from error
        return isinstance(payload, dict) and payload.get("status") == "ok"

    def _stop_process_locked(self, *, reason: str) -> None:
        del reason
        process = self._process
        self._process = None
        self._port = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        self._last_exit_code = process.returncode

    def terminate_for_recovery(self, reason: str = "operator recovery probe") -> None:
        """Terminate the current process without closing the reusable lane."""

        normalized = str(reason).strip() or "operator recovery probe"
        with self._lifecycle_lock:
            if self._closed:
                return
            self._forced_termination_count += 1
            self._stop_process_locked(reason=normalized)
            self._state = "stopped"

    def close(self) -> None:
        """Stop the child server and erase its temporary authentication state."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_process_locked(reason="resident lane closed")
            self._state = "closed"
        try:
            self._temporary.cleanup()
        except OSError:
            pass

    def _start_locked(self) -> tuple[int, list[dict[str, Any]]]:
        if self._closed:
            raise ProviderAdmissionError("resident language lane is closed")
        if self._process is not None and self._process.poll() is None:
            if self._port is not None and self._health_ready(self._port):
                return 0, []
            self._stop_process_locked(reason="resident health check failed")

        was_loaded = self._model_load_count > 0
        self._process_generation += 1
        generation = self._process_generation
        port = _loopback_port()
        self._stdout_path = self._temporary_path / f"server-{generation}.stdout.log"
        self._stderr_path = self._temporary_path / f"server-{generation}.stderr.log"
        environment = os.environ.copy()
        environment["LLAMA_API_KEY"] = self._api_key
        started = time.perf_counter()
        self._state = "starting"
        try:
            with (
                self._stdout_path.open("wb") as stdout_stream,
                self._stderr_path.open("wb") as stderr_stream,
            ):
                self._process = subprocess.Popen(
                    self._command(port),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as error:
            self._process = None
            self._state = "failed"
            raise ProviderTransportError("llama-server could not start") from error
        self._port = port
        deadline = time.monotonic() + self.startup_timeout_seconds
        while True:
            self._check_log_limits()
            process = self._process
            if process is None:
                self._state = "failed"
                raise ProviderTransportError("llama-server start state was lost")
            returncode = process.poll()
            if returncode is not None:
                self._last_exit_code = returncode
                detail = self._log_detail()
                self._process = None
                self._port = None
                self._state = "failed"
                lowered = detail.lower()
                if "out of memory" in lowered or "failed to allocate" in lowered:
                    raise ProviderCapacityError(
                        "llama-server could not load the model within capacity"
                    )
                raise ProviderTransportError(
                    "llama-server exited before becoming healthy"
                )
            if self._health_ready(port):
                break
            if time.monotonic() >= deadline:
                self._stop_process_locked(reason="resident startup timeout")
                self._state = "failed"
                raise ProviderTimeoutError(
                    "llama-server did not become healthy before timeout"
                )
            time.sleep(0.05)

        warmup_ms = 0
        if self.warmup_prompt is not None:
            warmup_started = time.perf_counter()
            try:
                warmup_content, _, _ = self._post_completion(
                    port=port,
                    prompt=self.warmup_prompt,
                    schema=self.warmup_schema,
                    max_tokens=32,
                )
                warmup_payload = json.loads(warmup_content)
            except (json.JSONDecodeError, ProviderError) as error:
                self._stop_process_locked(reason="resident inference warmup failed")
                self._state = "failed"
                raise ProviderAdmissionError(
                    "llama-server failed its schema warmup"
                ) from error
            if warmup_payload != {"ready": True}:
                self._stop_process_locked(reason="resident inference warmup invalid")
                self._state = "failed"
                raise ProviderAdmissionError(
                    "llama-server warmup crossed its JSON boundary"
                )
            warmup_ms = round((time.perf_counter() - warmup_started) * 1000)

        cold_start_ms = round((time.perf_counter() - started) * 1000)
        self._model_load_count += 1
        if was_loaded:
            self._restart_count += 1
        self._last_cold_start_ms = cold_start_ms
        self._last_warmup_ms = warmup_ms
        self._state = "ready"
        signal = (
            "resident-language-lane-restarted"
            if was_loaded
            else ("resident-language-lane-cold-start")
        )
        return (
            cold_start_ms,
            [
                {
                    "kind": "vertical",
                    "signal": signal,
                    "origin": "L0:resident-language-transport",
                    "propagates_to": [
                        "L1:language-message",
                        "L2:language-flow",
                        "L3:orchestration",
                    ],
                    "process_generation": generation,
                    "cold_start_ms": cold_start_ms,
                    "warmup_ms": warmup_ms,
                }
            ],
        )

    def _ensure_ready(self) -> tuple[int, list[dict[str, Any]]]:
        with self._lifecycle_lock:
            return self._start_locked()

    def _post_completion(
        self,
        *,
        port: int,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> tuple[str, Mapping[str, Any], str]:
        body = _canonical_json(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0,
                "seed": 1,
                "stream": False,
                "cache_prompt": True,
                "json_schema": schema,
            }
        )
        request = urllib.request.Request(
            f"http://{LOOPBACK_HOST}:{port}/completion",
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _open_loopback(
                request,
                timeout=self.request_timeout_seconds + 5,
            ) as response:
                raw = _read_limited(
                    response,
                    limit=MAX_SERVER_RESPONSE_BYTES,
                    label="llama-server completion response",
                )
        except urllib.error.HTTPError as error:
            try:
                detail = _read_limited(
                    error,
                    limit=MAX_SERVER_ERROR_BYTES,
                    label="llama-server error response",
                ).decode("utf-8", errors="replace")
            except ProviderError:
                detail = ""
            message = f"llama-server rejected completion with HTTP {error.code}"
            if error.code in {408, 504}:
                raise ProviderTimeoutError(message) from error
            if error.code in {429, 503}:
                raise ProviderCapacityError(message) from error
            if error.code in {400, 401, 403, 404, 422}:
                raise ProviderAdmissionError(message) from error
            raise ProviderTransportError(
                message + (f": {detail[:512]}" if detail else "")
            ) from error
        except TimeoutError as error:
            raise ProviderTimeoutError("llama-server completion timed out") from error
        except (urllib.error.URLError, OSError) as error:
            raise ProviderTransportError(
                "llama-server completion transport failed"
            ) from error

        digest = hashlib.sha256(raw).hexdigest()
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProviderTransportError(
                "llama-server completion returned invalid JSON"
            ) from error
        if not isinstance(envelope, dict):
            raise ProviderTransportError(
                "llama-server completion envelope is not an object"
            )
        content = envelope.get("content")
        timings = envelope.get("timings", {})
        if not isinstance(content, str):
            raise ProviderTransportError(
                "llama-server completion envelope has no text content"
            )
        if not isinstance(timings, Mapping):
            timings = {}
        return content, timings, digest

    def _invoke_cancellable(
        self,
        *,
        port: int,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        cancellation: CancellationToken,
        stage: str,
    ) -> tuple[str, Mapping[str, Any], str]:
        done = threading.Event()
        result: list[tuple[str, Mapping[str, Any], str]] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(
                    self._post_completion(
                        port=port,
                        prompt=prompt,
                        schema=schema,
                        max_tokens=max_tokens,
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        worker = threading.Thread(
            target=invoke,
            name=f"atom-resident-lane-{stage}",
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + self.request_timeout_seconds
        while not done.wait(0.025):
            if cancellation.cancelled:
                self.terminate_for_recovery(cancellation.reason)
                done.wait(3)
                raise ProviderCancelledError(cancellation.reason)
            if time.monotonic() >= deadline:
                self.terminate_for_recovery(f"timeout during {stage}")
                done.wait(3)
                raise ProviderTimeoutError(
                    f"resident language lane timed out during {stage}"
                )
            with self._lifecycle_lock:
                process = self._process
                if process is None or process.poll() is not None:
                    if process is not None:
                        self._last_exit_code = process.returncode
                    self._process = None
                    self._port = None
                    self._state = "failed"
                    done.wait(3)
                    raise ProviderTransportError(
                        f"resident language lane stopped during {stage}"
                    )
            self._check_log_limits()

        if errors:
            error = errors[0]
            if isinstance(error, ProviderError):
                raise error
            raise ProviderTransportError(
                f"resident language lane failed during {stage}"
            ) from error
        if len(result) != 1:
            raise ProviderTransportError(
                f"resident language lane returned no result during {stage}"
            )
        return result[0]

    def complete(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        stage: str,
        cancellation: CancellationToken | None = None,
    ) -> ResidentLaneCompletion:
        """Promote one request, run it on the warm lane, and demote its result."""

        if not isinstance(prompt, str) or not prompt or "\x00" in prompt:
            raise ValueError("resident language lane prompt is invalid")
        if not isinstance(schema, Mapping):
            raise ValueError("resident language lane schema is invalid")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("resident language lane token limit is invalid")
        if not isinstance(stage, str) or not stage.strip() or "\x00" in stage:
            raise ValueError("resident language lane stage is invalid")
        token = cancellation or CancellationToken()
        token.raise_if_cancelled()

        started = time.perf_counter()
        with self._state_lock:
            if (
                self._queued_requests + self._active_requests
                >= self.parallel_slots + self.max_queue_depth
            ):
                raise ProviderCapacityError(
                    "resident language lane queue is at capacity"
                )
            self._queued_requests += 1
        acquired = False
        queue_started = time.monotonic()
        try:
            deadline = queue_started + self.acquire_timeout_seconds
            while True:
                token.raise_if_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderCapacityError(
                        "resident language lane admission timed out"
                    )
                if self._admission.acquire(timeout=min(0.025, remaining)):
                    acquired = True
                    break
            queue_wait_ms = round((time.monotonic() - queue_started) * 1000)
            with self._state_lock:
                self._queued_requests -= 1
                self._active_requests += 1
                self._request_count += 1
                request_ordinal = self._request_count

            cold_start_ms, vibrations = self._ensure_ready()
            if queue_wait_ms > 0:
                vibrations.append(
                    {
                        "kind": "vertical",
                        "signal": "resident-language-lane-backpressure",
                        "origin": "L0:resident-language-admission",
                        "propagates_to": [
                            "L2:language-flow",
                            "L3:orchestration",
                        ],
                        "waited_ms": queue_wait_ms,
                    }
                )
            with self._lifecycle_lock:
                port = self._port
                generation = self._process_generation
                model_load_count = self._model_load_count
                restart_count = self._restart_count
            if port is None:
                raise ProviderTransportError("resident language lane has no endpoint")

            content, timings, envelope_sha256 = self._invoke_cancellable(
                port=port,
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
                cancellation=token,
                stage=stage,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            performance = _safe_timing_metrics(
                timings,
                cold_start_ms=cold_start_ms,
                request_elapsed_ms=elapsed_ms,
            )
            lane = {
                "schema": 1,
                "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
                "stage": stage,
                "process_generation": generation,
                "model_load_count": model_load_count,
                "restart_count": restart_count,
                "request_ordinal": request_ordinal,
                "resident_reused": request_ordinal > 1 and cold_start_ms == 0,
                "queue_wait_ms": queue_wait_ms,
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
                "vibrations": vibrations,
            }
            with self._state_lock:
                self._completed_count += 1
            return ResidentLaneCompletion(
                content=content,
                envelope_sha256=envelope_sha256,
                elapsed_ms=elapsed_ms,
                performance=performance,
                lane=lane,
            )
        except ProviderCancelledError:
            with self._state_lock:
                self._cancelled_count += 1
            raise
        except Exception:
            with self._state_lock:
                self._failed_count += 1
            raise
        finally:
            with self._state_lock:
                if self._queued_requests > 0 and not acquired:
                    self._queued_requests -= 1
                if acquired:
                    self._active_requests -= 1
            if acquired:
                self._admission.release()
