from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience import load_experience_corpus
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_harness_operator import (
    ATOM_HARNESS_OPERATOR_RUNTIME,
    AtomHarnessOperator,
)
from atom_harness_operator_server import build_server
from atom_harness_operator_ui import (
    ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING,
    ATOM_HARNESS_OPERATOR_UI_RUNTIME,
    render_operator_surface,
)
from atom_harness_session import AtomHarnessSession
from atom_llm_protocol import (
    ATOM_GROUNDED_RESPONSE_RUNTIME,
    ATOM_LANGUAGE_INTENT_RUNTIME,
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    ProviderLocation,
)
from atom_llm_provider import ScriptedJsonLanguageModel
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy
from atom_resident_language_lane import ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME
from atom_run_transaction import verify_committed_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4_INTEGRATION_TEST = "tests/test_atom_language_harness_v4_integration.py"


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
            {"role": role, "value": _one(record, role), "required": True}
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


class _OperatorResidentFixture(ScriptedJsonLanguageModel):
    def __init__(self, payloads: list[Mapping[str, Any]]) -> None:
        super().__init__(payloads, model="v4-operator-resident-fixture")
        self._lock = threading.RLock()
        self._alive = False
        self._generation = 0
        self._loads = 0
        self._restarts = 0
        self._requests = 0

    def manifest(self) -> Mapping[str, Any]:
        return {
            **dict(super().manifest()),
            "resident_lane": {
                "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
                "topology": "spiderweb-permanent-elevated-language-lane",
                "typed_on_ramp": "JsonGenerationRequest",
                "typed_off_ramp": "JsonGenerationResult",
            },
        }

    def lane_snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            return {
                "schema": 1,
                "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
                "state": "ready" if self._alive else "cold",
                "alive": self._alive,
                "process_generation": self._generation,
                "model_load_count": self._loads,
                "restart_count": self._restarts,
                "forced_termination_count": self._restarts,
                "request_count": self._requests,
                "completed_count": self._requests,
                "failed_count": 0,
                "cancelled_count": 0,
                "active_requests": 0,
                "queued_requests": 0,
                "last_cold_start_ms": 7 if self._loads else 0,
                "last_warmup_ms": 1 if self._loads else 0,
                "last_exit_code": None,
                "api_key_persisted": False,
            }

    def preload(self) -> Mapping[str, Any]:
        with self._lock:
            if not self._alive:
                self._generation += 1
                self._loads += 1
                self._alive = True
            return {
                "schema": 1,
                "provider_runtime": self.capabilities().provider_id,
                "model": self.capabilities().model,
                "lane": dict(self.lane_snapshot()),
                "secrets_persisted": False,
            }

    def terminate_lane_for_recovery(self, reason: str) -> None:
        del reason
        with self._lock:
            self._alive = False
            self._restarts += 1

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        result = super().generate_json(request, cancellation=cancellation)
        with self._lock:
            self._requests += 1
            ordinal = self._requests
            generation = self._generation
            loads = self._loads
            restarts = self._restarts
        lane = {
            "schema": 1,
            "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
            "stage": request.stage,
            "process_generation": generation,
            "model_load_count": loads,
            "restart_count": restarts,
            "request_ordinal": ordinal,
            "resident_reused": ordinal > 1,
            "queue_wait_ms": 0,
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
            "vibrations": [],
        }
        performance = {
            "runtime": "atom-resident-language-performance-v1",
            "cold_start_ms": 0,
            "model_load_ms": 0,
            "warm_request": True,
            "request_elapsed_ms": result.elapsed_ms,
            "prompt_tokens": 12,
            "cached_prompt_tokens": 4,
            "generated_tokens": 8,
            "prompt_ms": 2.0,
            "generation_ms": 4.0,
            "prompt_tokens_per_second": 6000.0,
            "generation_tokens_per_second": 2000.0,
        }
        return replace(result, performance=performance, lane=lane)


def _request(
    url: str,
    *,
    token: str | None = None,
    origin: str | None = None,
    cookie: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> tuple[int, bytes, Mapping[str, str]]:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if token is not None:
        headers["X-Atom-Operator-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    if cookie is not None:
        headers["Cookie"] = cookie
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


class AtomLanguageHarnessV4IntegrationTests(unittest.TestCase):
    def test_operator_runtime_wires_wiki_rag_api_and_real_side_view(self) -> None:
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
        question = (
            f"In the {_one(target, 'domain')} domain, what is the known "
            f"direction from {_one(target, 'cause')} to "
            f"{_one(target, 'effect')}?"
        )
        provider = _OperatorResidentFixture(
            [_intent_for(target, question), _response_for(target)]
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
        temporary = tempfile.TemporaryDirectory(prefix="atom-operator-v4-")
        root = Path(temporary.name)
        session = AtomHarnessSession(
            provider_fabric=fabric,
            output_root=root,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        operator = AtomHarnessOperator(session, state_root=root, max_queue_depth=4)
        server = build_server(operator)
        server_thread: threading.Thread | None = None
        try:
            startup = operator.start()
            self.assertEqual(startup["runtime"], ATOM_HARNESS_OPERATOR_RUNTIME)
            self.assertEqual(
                startup["preload"]["knowledge"]["wiki_runtime"],
                ATOM_HARNESS_WIKI_RUNTIME,
            )
            self.assertEqual(
                startup["preload"]["knowledge"]["rag_runtime"],
                ATOM_HARNESS_RAG_RUNTIME,
            )
            self.assertEqual(
                provider.lane_snapshot()["model_load_count"],
                1,
            )
            self.assertEqual(provider.lane_snapshot()["request_count"], 0)

            server_thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.05},
                daemon=True,
            )
            server_thread.start()
            status, page, headers = _request(server.origin + "/")
            self.assertEqual(status, 200)
            rendered = page.decode("utf-8")
            self.assertIn(ATOM_HARNESS_OPERATOR_UI_RUNTIME, rendered)
            self.assertIn(ATOM_HARNESS_WIKI_RUNTIME, rendered)
            self.assertIn(ATOM_HARNESS_RAG_RUNTIME, rendered)
            self.assertIn(ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING, rendered)
            self.assertIn("REAL ARTIFACT SIDE VIEW", rendered)
            self.assertIn('sandbox=""', rendered)
            self.assertIn('artifactFrame.removeAttribute("srcdoc")', rendered)
            self.assertIn("AtomArtifactToken=", headers["Set-Cookie"])
            self.assertIn("HttpOnly", headers["Set-Cookie"])
            self.assertIn("SameSite=Strict", headers["Set-Cookie"])
            self.assertIn("Content-Security-Policy", headers)
            self.assertNotIn("unsafe-inline", headers["Content-Security-Policy"])
            self.assertEqual(
                render_operator_surface(
                    access_token="test-token",
                    nonce="test-nonce",
                ).count("render_operator_surface"),
                1,
            )

            status, _, _ = _request(server.origin + "/api/status")
            self.assertEqual(status, 401)
            status, _, _ = _request(
                server.origin + "/api/ask",
                token=server.access_token,
                origin="http://invalid.example",
                payload={"question": question},
            )
            self.assertEqual(status, 403)
            status, body, _ = _request(
                server.origin + "/api/ask",
                token=server.access_token,
                origin=server.origin,
                payload={"question": question},
            )
            self.assertEqual(status, 202)
            submitted = json.loads(body)
            terminal = operator.wait_for_terminal(
                submitted["request_id"],
                timeout_seconds=90,
            )
            self.assertEqual(terminal["status"], "completed")
            self.assertTrue(terminal["artifact"]["citations"], terminal)
            output_dir = Path(terminal["output_dir"])
            transaction = verify_committed_run(output_dir)
            self.assertEqual(
                transaction["transaction_id"],
                terminal["artifact"]["transaction_id"],
            )

            status, side_view, side_headers = _request(
                (
                    server.origin
                    + "/api/artifacts/"
                    + submitted["request_id"]
                    + "/side-view"
                ),
                token=server.access_token,
            )
            self.assertEqual(status, 200)
            self.assertIn(b"atom-language-harness-side-view-v3", side_view)
            self.assertIn(
                terminal["artifact"]["answer"].encode("utf-8"),
                side_view,
            )
            self.assertEqual(
                side_headers["Cache-Control"],
                "no-store, max-age=0",
            )
            status, cookie_side_view, cookie_headers = _request(
                (
                    server.origin
                    + "/api/artifacts/"
                    + submitted["request_id"]
                    + "/side-view"
                ),
                cookie=headers["Set-Cookie"].split(";", 1)[0],
            )
            self.assertEqual(status, 200)
            self.assertEqual(cookie_side_view, side_view)
            self.assertEqual(cookie_headers["X-Frame-Options"], "SAMEORIGIN")

            status, _, _ = _request(
                server.origin + "/api/restart",
                token=server.access_token,
                origin=server.origin,
                payload={},
            )
            self.assertEqual(status, 200)
            self.assertEqual(provider.lane_snapshot()["model_load_count"], 2)
            self.assertEqual(provider.lane_snapshot()["restart_count"], 1)

            snapshot = operator.snapshot()
            self.assertFalse(snapshot["atom_store_mutation_allowed"])
            self.assertFalse(snapshot["secrets_persisted"])
            signals = {event["signal"] for event in snapshot["flow"]["events"]}
            self.assertIn("operator-thread-formed", signals)
            self.assertIn("operator-thread-promoted", signals)
            self.assertIn("operator-artifact-demoted", signals)
            self.assertIn("operator-resident-lane-restarted", signals)
            journal = json.loads(operator.journal_path.read_text(encoding="utf-8"))
            self.assertFalse(journal["secrets_persisted"])
            self.assertNotIn(server.access_token, json.dumps(journal))

            status, _, _ = _request(
                server.origin + "/api/shutdown",
                token=server.access_token,
                origin=server.origin,
                payload={"cancel_pending": False},
            )
            self.assertEqual(status, 202)
            server_thread.join(timeout=30)
            self.assertFalse(server_thread.is_alive())
            self.assertEqual(operator.snapshot()["state"], "closed")
        finally:
            if server_thread is not None and server_thread.is_alive():
                server.shutdown()
                server_thread.join(timeout=10)
            server.server_close()
            operator.shutdown(wait=True, cancel_pending=True)
            temporary.cleanup()

    def test_active_declarations_select_operator_v4(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "ai-runtime-registry.json").read_text(encoding="utf-8")
        )
        knowledge = json.loads(
            (PROJECT_ROOT / "ai-runtime-knowledge.json").read_text(encoding="utf-8")
        )
        side_view = json.loads(
            (PROJECT_ROOT / "ai-artifact-side-view.json").read_text(encoding="utf-8")
        )
        fabric = json.loads(
            (PROJECT_ROOT / "ai-provider-fabric.json").read_text(encoding="utf-8")
        )
        transaction = json.loads(
            (PROJECT_ROOT / "ai-run-transaction.json").read_text(encoding="utf-8")
        )
        architecture = json.loads(
            (PROJECT_ROOT / "atom-language-harness-architecture.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(registry["active_runtime"], "language-harness-v4")
        active = registry["runtimes"]["language-harness-v4"]
        for declaration in (
            active,
            knowledge,
            side_view,
            fabric,
            transaction,
        ):
            self.assertEqual(declaration["integration_test"], V4_INTEGRATION_TEST)
        self.assertEqual(
            active["runtime_entrypoint"],
            "atom_harness_operator_server.py",
        )
        self.assertEqual(
            side_view["side_view"]["artifact_binding_marker"],
            "render_operator_surface",
        )
        self.assertEqual(
            side_view["side_view"]["module_path"],
            "atom_harness_operator_ui.py",
        )
        self.assertTrue(knowledge["wiki_graph"]["enabled"])
        self.assertTrue(knowledge["rag"]["enabled"])
        self.assertTrue(fabric["provider_fabric"]["resident_model_preload"])
        self.assertTrue(transaction["run_transaction"]["durable_operator_journal"])
        certification = architecture["operator"]["certification_evidence"]
        self.assertEqual(certification["status"], "certified-live-local")
        self.assertEqual(certification["request_count"], 100)
        self.assertGreaterEqual(certification["elapsed_seconds"], 3600)
        self.assertTrue(certification["all_checks_passed"])
        self.assertEqual(
            certification["source_hash_normalization"],
            "utf-8-lf-v1",
        )
        self.assertEqual(len(certification["source_files_sha256"]), 15)


if __name__ == "__main__":
    unittest.main()
