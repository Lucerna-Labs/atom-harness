from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience import load_experience_corpus
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_causal_memory import RELEASE_BINARY
from atom_causal_world_schema import canonical_hash
from atom_harness_experiment import run_atom_language_harness
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
    reopen_harness_knowledge,
)
from atom_harness_runtime import ATOM_LANGUAGE_HARNESS_RUNTIME, AtomLanguageHarness
from atom_harness_side_view import (
    ATOM_HARNESS_SIDE_VIEW_RUNTIME,
    render_atom_harness_artifact,
)
from atom_llm_protocol import (
    ATOM_GROUNDED_RESPONSE_RUNTIME,
    ATOM_LANGUAGE_INTENT_RUNTIME,
    ATOM_ABSTENTION,
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    ProviderCapabilities,
    ProviderCancelledError,
    ProviderExhaustedError,
    ProviderLocation,
    ProviderTimeoutError,
    ProviderTransportError,
)
from atom_llm_provider import ScriptedJsonLanguageModel
from atom_provider_fabric import (
    ATOM_PROVIDER_FABRIC_RUNTIME,
    ProviderFabric,
    ProviderFabricPolicy,
)
from atom_run_transaction import (
    ATOM_RUN_TRANSACTION_FILENAME,
    ATOM_RUN_TRANSACTION_RUNTIME,
    RunIntegrityError,
    RunLockedError,
    RunTransaction,
    recover_transactions,
    verify_committed_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one(record, role: str) -> str:
    values = record.feature_values(role)
    if len(values) != 1:
        raise ValueError(f"test record has invalid {role}")
    return values[0]


def _intent_for(record) -> dict[str, Any]:
    return {
        "schema": 1,
        "runtime": ATOM_LANGUAGE_INTENT_RUNTIME,
        "action": "retrieve",
        "question": "Explain the known causal relation.",
        "features": [
            {
                "role": role,
                "value": _one(record, role),
                "required": True,
            }
            for role in ("kind", "domain", "cause", "effect", "direction")
        ],
    }


def _grounding_for(record) -> dict[str, Any]:
    return {
        "source_experience_id": record.experience_id,
        "kind": _one(record, "kind"),
        "status": _one(record, "status"),
        "domain": _one(record, "domain"),
        "cause": _one(record, "cause"),
        "effect": _one(record, "effect"),
        "direction": _one(record, "direction"),
    }


def _response_for(record) -> dict[str, Any]:
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
        "grounding": _grounding_for(record),
    }


class _BlockingProvider:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="atom-blocking-test-provider-v2",
            model="blocking-test-model",
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
            "provider_runtime": "atom-blocking-test-provider-v2",
            "model": "blocking-test-model",
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
        token = cancellation or CancellationToken()
        self.entered.set()
        while not self.release.wait(0.01):
            token.raise_if_cancelled()
        return JsonGenerationResult(
            payload={"stage": request.stage},
            provider="atom-blocking-test-provider-v2",
            model="blocking-test-model",
            elapsed_ms=0,
            raw_sha256=hashlib.sha256(request.stage.encode("utf-8")).hexdigest(),
        )


class _HalfOpenProvider(_BlockingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.call_lock = threading.Lock()

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        with self.call_lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            raise ProviderTransportError("open the synthetic circuit")
        return super().generate_json(request, cancellation=cancellation)


class AtomLanguageHarnessV2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="atom-language-harness-v2-integration-"
        )
        cls.root = Path(cls.temporary.name)
        cls.output_dir = cls.root / "committed-run"
        corpus = load_experience_corpus(
            PROJECT_ROOT / DEFAULT_EVIDENCE,
            PROJECT_ROOT / DEFAULT_MODEL,
        )
        cls.target = sorted(
            (
                record
                for record in corpus.laws
                if record.feature_values("status") == ("crystallized",)
            ),
            key=lambda item: item.experience_id,
        )[0]
        cls.intent = _intent_for(cls.target)
        cls.response = _response_for(cls.target)
        cls.provider = ScriptedJsonLanguageModel(
            [cls.intent, cls.response],
            model="v2-integration-provider",
        )
        cls.fabric = ProviderFabric(
            [cls.provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=1,
                circuit_failure_threshold=1,
                max_concurrency=1,
            ),
        )
        cls.artifact = run_atom_language_harness(
            cls.output_dir,
            question=(
                f"In the {_one(cls.target, 'domain')} domain, what is the "
                f"known direction from {_one(cls.target, 'cause')} to "
                f"{_one(cls.target, 'effect')}?"
            ),
            language_model=cls.fabric,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        cls.store_path = cls.output_dir / "runtime" / "atom_harness_knowledge.atomdb"
        cls.workflow = _read_json(cls.output_dir / "atom_harness_workflow.json")
        cls.graph = _read_json(cls.output_dir / "atom_harness_wiki_graph.json")
        cls.side_view = (cls.output_dir / "atom_harness_side_view.html").read_text(
            encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _knowledge(self):
        return reopen_harness_knowledge(
            self.store_path,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
            binary=RELEASE_BINARY,
        )

    @staticmethod
    def _test_policy(**overrides) -> ProviderFabricPolicy:
        values = {
            "allowed_locations": frozenset({ProviderLocation.LOCAL}),
            "allow_test_providers": True,
            "max_retries_per_provider": 0,
            "retry_backoff_seconds": 0,
            "circuit_failure_threshold": 1,
            "max_concurrency": 1,
        }
        values.update(overrides)
        return ProviderFabricPolicy(**values)

    def test_v2_runtime_wires_wiki_rag_side_view_and_transaction(self) -> None:
        self.assertTrue(self.artifact["passed"])
        self.assertEqual(
            self.artifact["runtime"],
            ATOM_LANGUAGE_HARNESS_RUNTIME,
        )
        self.assertEqual(
            self.artifact["intent"]["question"],
            self.artifact["question"],
        )
        self.assertEqual(
            self.artifact["knowledge"]["wiki_runtime"],
            ATOM_HARNESS_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.artifact["knowledge"]["rag_runtime"],
            ATOM_HARNESS_RAG_RUNTIME,
        )
        self.assertEqual(
            self.artifact["language_model"]["provider_runtime"],
            ATOM_PROVIDER_FABRIC_RUNTIME,
        )
        transaction = verify_committed_run(self.output_dir)
        self.assertEqual(transaction["state"], "committed")
        self.assertEqual(
            transaction["transaction_id"],
            self.artifact["transaction"]["transaction_id"],
        )
        rendered = render_atom_harness_artifact(
            self.artifact,
            self.workflow,
            self.graph,
        )
        self.assertEqual(rendered, self.side_view)
        self.assertIn(ATOM_HARNESS_SIDE_VIEW_RUNTIME, rendered)
        self.assertIn("Provider fabric", rendered)
        self.assertIn("Language performance", rendered)
        self.assertIn("Primary Atom claim", rendered)
        self.assertIn("v2-integration-provider", rendered)
        self.assertIn(self.target.experience_id, rendered)
        self.assertEqual(
            self.artifact["response"]["grounding"],
            self.artifact["evidence_packet"]["primary_claim"],
        )
        self.assertEqual(len(self.artifact["completions"]), 2)
        for completion in self.artifact["completions"]:
            self.assertEqual(completion["performance"], {})

    def test_provider_admission_and_routes_are_hash_bound(self) -> None:
        self.assertEqual(len(self.artifact["provider_routes"]), 2)
        for route in self.artifact["provider_routes"]:
            core = {key: route[key] for key in sorted(route) if key != "route_hash"}
            self.assertEqual(route["route_hash"], canonical_hash(core))
            self.assertTrue(route["completed"])
            self.assertEqual(
                route["selected_provider"]["location"],
                ProviderLocation.LOCAL.value,
            )
            self.assertTrue(route["selected_provider"]["strict_json_schema"])
        self.assertFalse(self.artifact["language_model"]["policy"]["allow_cloud_data"])

    def test_ordered_fallback_opens_circuit_without_weakening_schema(self) -> None:
        invalid_intent = dict(self.intent)
        invalid_intent["features"] = [
            {
                "role": "cause",
                "value": "not-in-the-atom-wiki",
                "required": True,
            }
        ]
        primary = ScriptedJsonLanguageModel(
            [invalid_intent],
            model="invalid-primary",
        )
        secondary = ScriptedJsonLanguageModel(
            [self.intent, self.response],
            model="valid-secondary",
        )
        fabric = ProviderFabric(
            [primary, secondary],
            policy=self._test_policy(),
        )
        artifact = AtomLanguageHarness(
            knowledge=self._knowledge(),
            language_model=fabric,
        ).answer("Use the known relation.")
        self.assertTrue(artifact["response"]["answerable"])
        intent_route = artifact["provider_routes"][0]
        self.assertEqual(intent_route["attempts"][0]["failure_kind"], "boundary")
        self.assertEqual(intent_route["attempts"][0]["outcome"], "failed")
        self.assertEqual(intent_route["attempts"][1]["outcome"], "completed")
        self.assertTrue(
            any(
                item["signal"] == "provider-fallback"
                for item in intent_route["vibrations"]
            )
        )
        response_route = artifact["provider_routes"][1]
        self.assertEqual(
            response_route["attempts"][0]["failure_kind"],
            "circuit-open",
        )
        self.assertEqual(response_route["attempts"][1]["outcome"], "completed")

    def test_fallback_route_is_committed_as_one_verified_bundle(self) -> None:
        invalid_intent = dict(self.intent)
        invalid_intent["features"] = [
            {
                "role": "domain",
                "value": "provider-invented-domain",
                "required": True,
            }
        ]
        primary = ScriptedJsonLanguageModel(
            [invalid_intent],
            model="invalid-primary",
        )
        secondary = ScriptedJsonLanguageModel(
            [self.intent, self.response],
            model="valid-secondary",
        )
        output = self.root / "fallback-committed-run"
        artifact = run_atom_language_harness(
            output,
            question="Commit the known relation after safe fallback.",
            language_model=ProviderFabric(
                [primary, secondary],
                policy=self._test_policy(),
            ),
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        self.assertTrue(artifact["passed"])
        self.assertTrue(artifact["response"]["answerable"])
        self.assertEqual(
            artifact["provider_routes"][0]["attempts"][0]["failure_kind"],
            "boundary",
        )
        self.assertEqual(
            artifact["provider_routes"][1]["attempts"][0]["failure_kind"],
            "circuit-open",
        )
        committed = verify_committed_run(output)
        self.assertEqual(
            committed["transaction_id"],
            artifact["transaction"]["transaction_id"],
        )

    def test_cloud_provider_is_blocked_without_consent_and_never_called(self) -> None:
        cloud = ScriptedJsonLanguageModel(
            [self.intent, self.response],
            model="cloud-test-provider",
            location=ProviderLocation.CLOUD,
        )
        fabric = ProviderFabric(
            [cloud],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset(
                    {ProviderLocation.LOCAL, ProviderLocation.PRIVATE}
                ),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        before = _sha256(self.store_path)
        artifact = AtomLanguageHarness(
            knowledge=self._knowledge(),
            language_model=fabric,
        ).answer("Do not send this Atom question to a cloud provider.")
        self.assertEqual(cloud.requests, [])
        self.assertTrue(artifact["degraded"])
        self.assertEqual(artifact["outcome"], "degraded-abstention")
        self.assertEqual(artifact["response"]["answer"], ATOM_ABSTENTION)
        self.assertEqual(
            artifact["provider_routes"][0]["attempts"][0]["failure_kind"],
            "privacy",
        )
        self.assertTrue(
            any(
                item["signal"] == "provider-privacy-block"
                for item in artifact["spiderweb_trace"]["vibrations"]
            )
        )
        self.assertIn(
            "orchestration.provider-degraded-intent",
            artifact["spiderweb_trace"]["thread"]["path"],
        )
        self.assertNotIn(
            "intent.parse",
            artifact["spiderweb_trace"]["thread"]["path"],
        )
        self.assertNotIn(
            "artifact.side_view",
            artifact["spiderweb_trace"]["thread"]["path"],
        )
        self.assertEqual(_sha256(self.store_path), before)

    def test_retry_is_typed_and_limited_to_retryable_transport_failure(self) -> None:
        provider = ScriptedJsonLanguageModel(
            [
                ProviderTransportError("temporary transport failure"),
                self.intent,
                self.response,
            ],
            model="retry-provider",
        )
        fabric = ProviderFabric(
            [provider],
            policy=self._test_policy(
                max_retries_per_provider=1,
                circuit_failure_threshold=2,
            ),
        )
        artifact = AtomLanguageHarness(
            knowledge=self._knowledge(),
            language_model=fabric,
        ).answer("Retry the known relation safely.")
        route = artifact["provider_routes"][0]
        self.assertEqual(
            [item["outcome"] for item in route["attempts"]],
            ["failed", "completed"],
        )
        self.assertEqual(route["attempts"][0]["failure_kind"], "transport")
        self.assertTrue(route["attempts"][0]["retryable"])
        self.assertTrue(
            any(item["signal"] == "provider-retry" for item in route["vibrations"])
        )

    def test_timeout_falls_back_and_response_boundary_failure_abstains(self) -> None:
        timeout = ScriptedJsonLanguageModel(
            [ProviderTimeoutError("synthetic timeout")],
            model="timeout-primary",
        )
        malformed_response = dict(self.response)
        malformed_response["citations"] = ["experience:invented"]
        secondary = ScriptedJsonLanguageModel(
            [self.intent, malformed_response],
            model="boundary-secondary",
        )
        fabric = ProviderFabric(
            [timeout, secondary],
            policy=self._test_policy(),
        )
        before = _sha256(self.store_path)
        artifact = AtomLanguageHarness(
            knowledge=self._knowledge(),
            language_model=fabric,
        ).answer("Render only packet-local evidence.")
        self.assertTrue(artifact["degraded"])
        self.assertFalse(artifact["response"]["answerable"])
        self.assertEqual(artifact["response"]["citations"], [])
        self.assertEqual(
            artifact["provider_routes"][0]["attempts"][0]["failure_kind"],
            "timeout",
        )
        self.assertEqual(
            artifact["provider_routes"][1]["attempts"][-1]["failure_kind"],
            "boundary",
        )
        self.assertEqual(_sha256(self.store_path), before)

    def test_response_grounding_cannot_override_atoms_primary_claim(self) -> None:
        malformed_response = dict(self.response)
        malformed_response["grounding"] = {
            **self.response["grounding"],
            "direction": (
                "+1" if self.response["grounding"]["direction"] != "+1" else "-1"
            ),
        }
        provider = ScriptedJsonLanguageModel(
            [self.intent, malformed_response],
            model="grounding-mismatch-provider",
        )
        before = _sha256(self.store_path)
        artifact = AtomLanguageHarness(
            knowledge=self._knowledge(),
            language_model=provider,
        ).answer("Do not override Atom's selected claim.")
        self.assertTrue(artifact["degraded"])
        self.assertFalse(artifact["response"]["answerable"])
        self.assertIsNone(artifact["response"]["grounding"])
        self.assertEqual(
            artifact["provider_routes"][-1]["attempts"][-1]["failure_kind"],
            "boundary",
        )
        self.assertEqual(_sha256(self.store_path), before)

    def test_provider_semaphore_emits_real_backpressure_vibration(self) -> None:
        provider = _BlockingProvider()
        fabric = ProviderFabric(
            [provider],
            policy=self._test_policy(acquire_timeout_seconds=2.0),
        )
        results: list[JsonGenerationResult] = []
        errors: list[BaseException] = []

        def invoke(stage: str) -> None:
            try:
                results.append(
                    fabric.generate_json(
                        JsonGenerationRequest(
                            stage=stage,
                            system_prompt="Return one JSON object.",
                            payload={"stage": stage},
                            schema={"type": "object"},
                            max_tokens=32,
                        )
                    )
                )
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=invoke, args=("first",))
        second = threading.Thread(target=invoke, args=("second",))
        first.start()
        self.assertTrue(provider.entered.wait(1))
        second.start()
        time.sleep(0.08)
        provider.release.set()
        first.join(2)
        second.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        waited_route = next(
            item.route for item in results if item.route["stage"] == "second"
        )
        self.assertTrue(
            any(
                item["signal"] == "provider-backpressure" and item["waited_ms"] >= 1
                for item in waited_route["vibrations"]
            )
        )

    def test_provider_semaphore_timeout_is_hash_bound_route_evidence(self) -> None:
        provider = _BlockingProvider()
        fabric = ProviderFabric(
            [provider],
            policy=self._test_policy(acquire_timeout_seconds=0.1),
        )
        first_result: list[JsonGenerationResult] = []

        def occupy_provider() -> None:
            first_result.append(
                fabric.generate_json(
                    JsonGenerationRequest(
                        stage="occupy",
                        system_prompt="Return one JSON object.",
                        payload={},
                        schema={"type": "object"},
                        max_tokens=32,
                    )
                )
            )

        first = threading.Thread(target=occupy_provider)
        first.start()
        self.assertTrue(provider.entered.wait(1))
        with self.assertRaises(ProviderExhaustedError) as captured:
            fabric.generate_json(
                JsonGenerationRequest(
                    stage="backpressure-timeout",
                    system_prompt="Return one JSON object.",
                    payload={},
                    schema={"type": "object"},
                    max_tokens=32,
                )
            )
        route = captured.exception.route
        self.assertEqual(route["disposition"], "exhausted")
        self.assertEqual(
            route["route_hash"],
            canonical_hash(
                {key: route[key] for key in sorted(route) if key != "route_hash"}
            ),
        )
        self.assertTrue(
            any(
                item["signal"] == "provider-backpressure-timeout"
                and item["waited_ms"] >= 90
                for item in route["vibrations"]
            )
        )
        provider.release.set()
        first.join(2)
        self.assertEqual(len(first_result), 1)

    def test_circuit_allows_only_one_half_open_probe(self) -> None:
        provider = _HalfOpenProvider()
        fabric = ProviderFabric(
            [provider],
            policy=self._test_policy(
                circuit_cooldown_seconds=0.1,
                acquire_timeout_seconds=1,
                max_concurrency=2,
            ),
        )
        request = JsonGenerationRequest(
            stage="half-open",
            system_prompt="Return one JSON object.",
            payload={},
            schema={"type": "object"},
            max_tokens=32,
        )
        with self.assertRaises(ProviderExhaustedError):
            fabric.generate_json(request)
        time.sleep(0.11)
        probe_results: list[JsonGenerationResult] = []
        probe = threading.Thread(
            target=lambda: probe_results.append(fabric.generate_json(request))
        )
        probe.start()
        self.assertTrue(provider.entered.wait(1))
        with self.assertRaises(ProviderExhaustedError) as captured:
            fabric.generate_json(request)
        self.assertEqual(
            captured.exception.route["attempts"][0]["failure_kind"],
            "circuit-half-open-busy",
        )
        self.assertEqual(provider.calls, 2)
        provider.release.set()
        probe.join(2)
        self.assertEqual(len(probe_results), 1)
        circuit = fabric.preload_manifest()["providers"][0]["circuit"]
        self.assertEqual(circuit["state"], "closed")

    def test_cancellation_aborts_without_publishing_partial_run(self) -> None:
        output = self.root / "cancelled-run"
        token = CancellationToken()
        token.cancel("operator cancelled integration request")
        provider = ScriptedJsonLanguageModel([self.intent, self.response])
        fabric = ProviderFabric(
            [provider],
            policy=self._test_policy(),
        )
        with self.assertRaises(ProviderCancelledError):
            run_atom_language_harness(
                output,
                question="Cancel before any provider receives data.",
                language_model=fabric,
                forge_path=PROJECT_ROOT / DEFAULT_FORGE,
                evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
                model_path=PROJECT_ROOT / DEFAULT_MODEL,
                cancellation=token,
            )
        self.assertFalse(output.exists())
        recovery = output.parent / ".atom-harness-v2" / "recovery"
        failures = list(recovery.glob("*.aborted/atom_harness_failure.json"))
        self.assertTrue(failures)

    def test_output_lock_prevents_concurrent_publication(self) -> None:
        output = self.root / "locked-run"
        first = RunTransaction(output).begin()
        try:
            with self.assertRaises(RunLockedError):
                RunTransaction(output).begin()
        finally:
            first.abort("lock integration test complete")
        self.assertFalse(output.exists())

    def test_dead_process_committed_stage_is_recovered(self) -> None:
        parent = self.root / "recovery-parent"
        final = parent / "recovered-run"
        transaction_id = "a" * 64
        staging = (
            parent / ".atom-harness-v2" / "staging" / f"{final.name}.{transaction_id}"
        )
        staging.mkdir(parents=True)
        payload = staging / "payload.txt"
        payload.write_text("complete\n", encoding="utf-8", newline="\n")
        files = [
            {
                "path": "payload.txt",
                "bytes": payload.stat().st_size,
                "sha256": _sha256(payload),
            }
        ]
        core = {
            "schema": 1,
            "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
            "transaction_id": transaction_id,
            "target_name": final.name,
            "state": "committed",
            "created_at": "2026-07-30T00:00:00+00:00",
            "sealed_at": "2026-07-30T00:00:01+00:00",
            "pid": 999_999_999,
            "required_files": ["payload.txt"],
            "files": files,
            "total_bytes": payload.stat().st_size,
        }
        manifest = {**core, "manifest_hash": canonical_hash(core)}
        (staging / ATOM_RUN_TRANSACTION_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        events = recover_transactions(parent)
        self.assertTrue(final.is_dir())
        self.assertEqual((final / "payload.txt").read_text(), "complete\n")
        self.assertEqual(verify_committed_run(final)["state"], "committed")
        self.assertIn(
            "recovered-commit",
            {item["action"] for item in events},
        )

    def test_committed_bundle_tampering_is_detected(self) -> None:
        tamper_parent = self.root / "tamper-parent"
        tampered = tamper_parent / self.output_dir.name
        shutil.copytree(self.output_dir, tampered)
        artifact_path = tampered / "atom_harness_artifact.json"
        artifact_path.write_text(
            artifact_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RunIntegrityError, "mismatch"):
            verify_committed_run(tampered)

    def test_active_runtime_contracts_declare_all_current_surfaces(self) -> None:
        registry = _read_json(PROJECT_ROOT / "ai-runtime-registry.json")
        knowledge = _read_json(PROJECT_ROOT / "ai-runtime-knowledge.json")
        side_view = _read_json(PROJECT_ROOT / "ai-artifact-side-view.json")
        provider = _read_json(PROJECT_ROOT / "ai-provider-fabric.json")
        transaction = _read_json(PROJECT_ROOT / "ai-run-transaction.json")
        architecture = _read_json(
            PROJECT_ROOT / "atom-language-harness-architecture.json"
        )
        expected_test = "tests/test_atom_universal_knowledge_integration.py"
        self.assertEqual(registry["active_runtime"], "language-harness-v6")
        active = registry["runtimes"]["language-harness-v6"]
        self.assertEqual(active["integration_test"], expected_test)
        self.assertEqual(knowledge["integration_test"], expected_test)
        self.assertEqual(side_view["integration_test"], expected_test)
        self.assertEqual(provider["integration_test"], expected_test)
        self.assertEqual(transaction["integration_test"], expected_test)
        self.assertEqual(architecture["integration_test"], expected_test)
        self.assertEqual(
            architecture["runtime"],
            "atom-language-harness-operator-v6",
        )
        self.assertEqual(
            architecture["knowledge"]["wiki_runtime"],
            ATOM_HARNESS_WIKI_RUNTIME,
        )
        self.assertEqual(
            architecture["knowledge"]["rag_runtime"],
            ATOM_HARNESS_RAG_RUNTIME,
        )
        self.assertEqual(
            architecture["side_view"]["runtime"],
            "atom-language-harness-operator-ui-v6",
        )


if __name__ == "__main__":
    unittest.main()
