from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience import load_experience_corpus
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_causal_world_schema import canonical_hash
from atom_harness_session import AtomHarnessSession
from atom_llm_protocol import (
    ATOM_GROUNDED_RESPONSE_RUNTIME,
    ATOM_LANGUAGE_INTENT_RUNTIME,
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    ProviderCapabilities,
    ProviderCapacityError,
    ProviderExhaustedError,
    ProviderLocation,
    ProviderTransportError,
)
from atom_llm_provider import ScriptedJsonLanguageModel
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy
from atom_resident_language_lane import (
    ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
    ResidentLanguageLane,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _one(record, role: str) -> str:
    values = record.feature_values(role)
    if len(values) != 1:
        raise ValueError(f"test record has invalid {role}")
    return values[0]


def _intent_for(record, question: str) -> dict[str, Any]:
    return {
        "schema": 1,
        "runtime": ATOM_LANGUAGE_INTENT_RUNTIME,
        "action": "retrieve",
        "question": question,
        "features": [
            {
                "role": role,
                "value": _one(record, role),
                "required": True,
            }
            for role in ("kind", "domain", "cause", "effect", "direction")
        ],
    }


def _response_for(record) -> dict[str, Any]:
    grounding = {
        "source_experience_id": record.experience_id,
        "kind": _one(record, "kind"),
        "status": _one(record, "status"),
        "domain": _one(record, "domain"),
        "cause": _one(record, "cause"),
        "effect": _one(record, "effect"),
        "direction": _one(record, "direction"),
    }
    return {
        "schema": 1,
        "runtime": ATOM_GROUNDED_RESPONSE_RUNTIME,
        "answerable": True,
        "answer": (
            f"Atom records {_one(record, 'cause')} leading to "
            f"{_one(record, 'effect')} with direction "
            f"{_one(record, 'direction')} in the "
            f"{_one(record, 'domain')} domain."
        ),
        "citations": [record.experience_id],
        "limitations": "This describes the retrieved structural experience only.",
        "grounding": grounding,
    }


class _FakeResidentLanguageLane(ResidentLanguageLane):
    """Exercise lane mechanics without starting an external model process."""

    def __init__(
        self,
        model_path: Path,
        *,
        executable: Path,
        parallel_slots: int = 1,
        max_queue_depth: int = 8,
    ) -> None:
        super().__init__(
            model_path,
            executable=str(executable),
            context_length=1024,
            startup_timeout_seconds=1,
            request_timeout_seconds=2,
            acquire_timeout_seconds=1,
            parallel_slots=parallel_slots,
            max_queue_depth=max_queue_depth,
        )
        self.blocked_calls: set[int] = set()
        self.call_entered = threading.Event()
        self.release_call = threading.Event()
        self._fake_call_lock = threading.Lock()
        self._fake_call_count = 0

    def _ensure_ready(self) -> tuple[int, list[dict[str, Any]]]:
        with self._lifecycle_lock:
            if self._closed:
                raise ProviderTransportError("fake resident lane is closed")
            if self._port is not None:
                return 0, []
            was_loaded = self._model_load_count > 0
            self._process_generation += 1
            self._model_load_count += 1
            if was_loaded:
                self._restart_count += 1
            self._port = 31337
            self._state = "ready"
            self._last_cold_start_ms = 7
            signal = (
                "resident-language-lane-restarted"
                if was_loaded
                else "resident-language-lane-cold-start"
            )
            return (
                7,
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
                        "process_generation": self._process_generation,
                        "cold_start_ms": 7,
                        "warmup_ms": 1,
                    }
                ],
            )

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
        del port, prompt, schema, max_tokens, stage
        with self._fake_call_lock:
            self._fake_call_count += 1
            call = self._fake_call_count
        self.call_entered.set()
        if call in self.blocked_calls:
            while not self.release_call.wait(0.005):
                cancellation.raise_if_cancelled()
                with self._lifecycle_lock:
                    if self._port is None:
                        raise ProviderTransportError(
                            "fake resident process stopped during completion"
                        )
        content = json.dumps({"ready": True}, sort_keys=True)
        return (
            content,
            {
                "prompt_n": 8,
                "cache_n": 4,
                "predicted_n": 3,
                "prompt_ms": 2.0,
                "predicted_ms": 4.0,
                "prompt_per_second": 4000.0,
                "predicted_per_second": 750.0,
            },
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class _LaneEvidenceProvider:
    def __init__(self) -> None:
        self.closed = False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="atom-test-resident-provider-v1",
            model="test-resident-model",
            location=ProviderLocation.LOCAL,
            strict_json_schema=True,
            max_context_tokens=4096,
            max_output_tokens=4096,
            supports_cancellation=True,
            cost_tier="test-fixture",
            test_only=True,
        )

    def manifest(self) -> Mapping[str, Any]:
        return {
            "schema": 1,
            "protocol": "atom-json-language-model-v2",
            "provider_runtime": "atom-test-resident-provider-v1",
            "model": "test-resident-model",
            "resident_lane": {
                "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
                "topology": "spiderweb-permanent-elevated-language-lane",
            },
            "test_only": True,
            "capabilities": self.capabilities().manifest(),
            "available": True,
            "secrets_persisted": False,
        }

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        del cancellation
        lane = {
            "schema": 1,
            "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
            "stage": request.stage,
            "process_generation": 1,
            "model_load_count": 1,
            "restart_count": 0,
            "request_ordinal": 2,
            "resident_reused": True,
            "queue_wait_ms": 3,
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
            "vibrations": [
                {
                    "kind": "vertical",
                    "signal": "resident-language-lane-backpressure",
                    "origin": "L0:resident-language-admission",
                    "propagates_to": ["L2:language-flow", "L3:orchestration"],
                    "waited_ms": 3,
                }
            ],
        }
        return JsonGenerationResult(
            payload={"ready": True},
            provider=self.capabilities().provider_id,
            model=self.capabilities().model,
            elapsed_ms=3,
            raw_sha256=hashlib.sha256(request.stage.encode("utf-8")).hexdigest(),
            lane=lane,
        )

    def close(self) -> None:
        self.closed = True


class _MalformedLaneEvidenceProvider(_LaneEvidenceProvider):
    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        result = super().generate_json(request, cancellation=cancellation)
        return replace(
            result,
            lane={
                **dict(result.lane),
                "process_generation": 1,
                "model_load_count": 2,
            },
        )


class AtomResidentLanguageLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="atom-resident-lane-test-")
        self.root = Path(self.temporary.name)
        self.executable = self.root / "llama-server.exe"
        self.executable.write_bytes(b"test executable")
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"test model")
        self.lanes: list[_FakeResidentLanguageLane] = []

    def tearDown(self) -> None:
        for lane in self.lanes:
            lane.close()
        self.temporary.cleanup()

    def _lane(
        self,
        *,
        parallel_slots: int = 1,
        max_queue_depth: int = 8,
    ) -> _FakeResidentLanguageLane:
        lane = _FakeResidentLanguageLane(
            self.model,
            executable=self.executable,
            parallel_slots=parallel_slots,
            max_queue_depth=max_queue_depth,
        )
        self.lanes.append(lane)
        return lane

    @staticmethod
    def _complete(lane: ResidentLanguageLane, stage: str):
        return lane.complete(
            prompt="Return the required JSON object.",
            schema={"type": "object"},
            max_tokens=32,
            stage=stage,
        )

    def test_first_request_loads_once_and_later_requests_reuse_lane(self) -> None:
        lane = self._lane()
        first = self._complete(lane, "first")
        second = self._complete(lane, "second")

        self.assertEqual(first.lane["process_generation"], 1)
        self.assertEqual(first.lane["model_load_count"], 1)
        self.assertFalse(first.lane["resident_reused"])
        self.assertEqual(first.performance["cold_start_ms"], 7)
        self.assertEqual(second.lane["process_generation"], 1)
        self.assertEqual(second.lane["model_load_count"], 1)
        self.assertTrue(second.lane["resident_reused"])
        self.assertEqual(second.performance["cold_start_ms"], 0)
        snapshot = lane.snapshot()
        self.assertTrue(lane.static_manifest()["external_proxy_disabled"])
        self.assertEqual(snapshot["model_load_count"], 1)
        self.assertEqual(snapshot["completed_count"], 2)

    def test_waiting_request_emits_vertical_backpressure(self) -> None:
        lane = self._lane(max_queue_depth=1)
        lane.blocked_calls.add(1)
        results: list[Any] = []

        first = threading.Thread(
            target=lambda: results.append(self._complete(lane, "first")),
            daemon=True,
        )
        first.start()
        self.assertTrue(lane.call_entered.wait(1))
        lane.call_entered.clear()
        second = threading.Thread(
            target=lambda: results.append(self._complete(lane, "second")),
            daemon=True,
        )
        second.start()
        deadline = time.monotonic() + 1
        while lane.snapshot()["queued_requests"] != 1:
            if time.monotonic() >= deadline:
                self.fail("second resident request never entered the bounded queue")
            time.sleep(0.005)
        time.sleep(0.02)
        lane.release_call.set()
        first.join(2)
        second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        waited = next(item for item in results if item.lane["stage"] == "second")
        self.assertGreater(waited.lane["queue_wait_ms"], 0)
        self.assertIn(
            "resident-language-lane-backpressure",
            {item["signal"] for item in waited.lane["vibrations"]},
        )

    def test_zero_queue_depth_rejects_excess_work(self) -> None:
        lane = self._lane(max_queue_depth=0)
        lane.blocked_calls.add(1)
        failures: list[BaseException] = []

        def first_request() -> None:
            try:
                self._complete(lane, "first")
            except BaseException as error:
                failures.append(error)

        first = threading.Thread(target=first_request, daemon=True)
        first.start()
        self.assertTrue(lane.call_entered.wait(1))
        with self.assertRaisesRegex(ProviderCapacityError, "queue is at capacity"):
            self._complete(lane, "rejected")
        lane.release_call.set()
        first.join(2)
        self.assertFalse(first.is_alive())
        self.assertEqual(failures, [])

    def test_forced_process_loss_is_typed_and_next_request_recovers(self) -> None:
        lane = self._lane()
        self._complete(lane, "prime")
        lane.blocked_calls.add(2)
        errors: list[BaseException] = []

        def interrupted_request() -> None:
            try:
                self._complete(lane, "interrupted")
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=interrupted_request, daemon=True)
        lane.call_entered.clear()
        worker.start()
        self.assertTrue(lane.call_entered.wait(1))
        lane.terminate_for_recovery("test crash")
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ProviderTransportError)
        recovered = self._complete(lane, "recovered")
        self.assertEqual(recovered.lane["process_generation"], 2)
        self.assertEqual(recovered.lane["model_load_count"], 2)
        self.assertEqual(recovered.lane["restart_count"], 1)
        self.assertIn(
            "resident-language-lane-restarted",
            {item["signal"] for item in recovered.lane["vibrations"]},
        )

    def test_fabric_hash_binds_lane_evidence_and_closes_provider(self) -> None:
        provider = _LaneEvidenceProvider()
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        request = JsonGenerationRequest(
            stage="lane_route_test",
            system_prompt="Return the schema-bound object.",
            payload={"ready": True},
            schema={"type": "object"},
            max_tokens=16,
        )
        result = fabric.generate_json(request)

        self.assertTrue(result.route["completed"])
        self.assertEqual(result.route["language_lane"], result.lane)
        route_core = {
            key: result.route[key]
            for key in sorted(result.route)
            if key != "route_hash"
        }
        self.assertEqual(result.route["route_hash"], canonical_hash(route_core))
        self.assertIn(
            "resident-language-lane-backpressure",
            {item["signal"] for item in result.route["vibrations"]},
        )
        fabric.close()
        self.assertTrue(provider.closed)
        with self.assertRaises(ProviderExhaustedError):
            fabric.generate_json(request)

    def test_fabric_rejects_impossible_resident_lane_counters(self) -> None:
        provider = _MalformedLaneEvidenceProvider()
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        request = JsonGenerationRequest(
            stage="malformed_lane_route_test",
            system_prompt="Return the schema-bound object.",
            payload={"ready": True},
            schema={"type": "object"},
            max_tokens=16,
        )
        with self.assertRaises(ProviderExhaustedError) as raised:
            fabric.generate_json(request)
        route = raised.exception.route
        self.assertEqual(route["attempts"][0]["failure_kind"], "boundary")
        self.assertEqual(
            route["route_hash"],
            canonical_hash(
                {key: route[key] for key in sorted(route) if key != "route_hash"}
            ),
        )
        fabric.close()

    def test_session_reuses_one_fabric_across_multiple_harness_requests(self) -> None:
        corpus = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        target = sorted(
            (
                record
                for record in corpus.laws
                if record.feature_values("status") == ("crystallized",)
            ),
            key=lambda item: item.experience_id,
        )[0]
        questions = [
            (
                f"In the {_one(target, 'domain')} domain, what is the known "
                f"direction from {_one(target, 'cause')} to "
                f"{_one(target, 'effect')}?"
            ),
            (
                f"Using only Atom evidence, tell me how "
                f"{_one(target, 'cause')} affects {_one(target, 'effect')} "
                f"in {_one(target, 'domain')}."
            ),
        ]
        provider = ScriptedJsonLanguageModel(
            [
                _intent_for(target, questions[0]),
                _response_for(target),
                _intent_for(target, questions[1]),
                _response_for(target),
            ],
            model="session-reuse-fixture",
        )
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        output_root = self.root / "session"
        session = AtomHarnessSession(
            provider_fabric=fabric,
            output_root=output_root,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        artifacts = [session.answer(question) for question in questions]
        manifest = session.manifest()

        self.assertTrue(all(item["passed"] for item in artifacts))
        self.assertEqual(manifest["request_count"], 2)
        self.assertEqual(manifest["completed_count"], 2)
        self.assertEqual(manifest["failed_count"], 0)
        self.assertTrue(output_root.is_absolute())
        expected_dirs = [
            output_root / f"request-{ordinal:04d}-"
            f"{hashlib.sha256(question.encode('utf-8')).hexdigest()[:12]}"
            for ordinal, question in enumerate(questions, start=1)
        ]
        self.assertTrue(
            all(
                (path / "atom_harness_artifact.json").is_file()
                for path in expected_dirs
            )
        )
        session.close()
        self.assertTrue(session.manifest()["closed"])
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            session.answer(questions[0])


if __name__ == "__main__":
    unittest.main()
