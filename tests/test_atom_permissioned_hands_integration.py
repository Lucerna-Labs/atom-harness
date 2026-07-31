from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
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
from atom_harness_operator import ATOM_HARNESS_OPERATOR_RUNTIME, AtomHarnessOperator
from atom_harness_operator_server import (
    ATOM_HARNESS_OPERATOR_SERVER_RUNTIME,
    build_server,
)
from atom_harness_operator_ui import (
    ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING,
    ATOM_HARNESS_OPERATOR_UI_RUNTIME,
    render_operator_surface,
)
from atom_harness_session import AtomHarnessSession
from atom_llm_protocol import (
    ATOM_GROUNDED_RESPONSE_RUNTIME,
    ATOM_LANGUAGE_INTENT_RUNTIME,
    ProviderLocation,
)
from atom_llm_provider import ScriptedJsonLanguageModel
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy
from atom_run_transaction import verify_committed_run
from atom_tool_fabric import (
    ATOM_PERMISSIONED_HANDS_RUNTIME,
    PermissionedToolFabric,
)
from atom_tool_protocol import (
    ATOM_TOOL_PROPOSAL_RUNTIME,
    normalize_untrusted_context,
    tool_task_sha256,
)
from atom_tool_side_view import (
    ATOM_TOOL_ARTIFACT_BINDING,
    ATOM_TOOL_SIDE_VIEW_RUNTIME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE6_INTEGRATION_TEST = "tests/test_atom_permissioned_hands_integration.py"


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


def _tool_proposal(task: str) -> dict[str, Any]:
    return {
        "schema": 1,
        "runtime": ATOM_TOOL_PROPOSAL_RUNTIME,
        "task_sha256": tool_task_sha256(task),
        "context_sha256": normalize_untrusted_context([])[1],
        "summary": "Create a simulation program, run it, and write the result note.",
        "actions": [
            {
                "action_id": "action-1",
                "capability": "workspace.make_directory",
                "arguments": {"path": "phase6-live", "parents": False},
                "rationale": "Create the exact experiment directory.",
            },
            {
                "action_id": "action-2",
                "capability": "workspace.write_text",
                "arguments": {
                    "path": "phase6-live/sim.py",
                    "content": (
                        "import json, sys\n"
                        "value = int(sys.argv[1])\n"
                        "print(json.dumps({'cube': value ** 3}))\n"
                    ),
                    "mode": "create",
                    "expected_sha256": None,
                },
                "rationale": "Write the requested simulation source.",
            },
            {
                "action_id": "action-3",
                "capability": "simulation.run",
                "arguments": {
                    "program": sys.executable,
                    "base_arguments": ["sim.py"],
                    "cases": [
                        {"name": "cube-two", "arguments": ["2"]},
                        {"name": "cube-five", "arguments": ["5"]},
                    ],
                    "cwd": "phase6-live",
                    "timeout_seconds_per_case": 20,
                },
                "rationale": "Run two exact simulation cases.",
            },
            {
                "action_id": "action-4",
                "capability": "document.create",
                "arguments": {
                    "path": "phase6-live/RESULT.md",
                    "format": "markdown",
                    "content": "# Phase 6 result\n\nThe cube simulation completed.\n",
                    "mode": "create",
                    "expected_sha256": None,
                },
                "rationale": "Write the user-visible experiment note.",
            },
        ],
        "completion_condition": "The simulation and document have committed receipts.",
    }


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
        with opener.open(request, timeout=60) as response:
            result_headers = dict(response.headers)
            cookies = response.headers.get_all("Set-Cookie") or []
            if cookies:
                result_headers["Set-Cookie"] = "\n".join(cookies)
            return response.status, response.read(), result_headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


class PermissionedHandsIntegrationTests(unittest.TestCase):
    def test_phase6_wires_wiki_rag_permission_hands_and_both_side_views(self) -> None:
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
            f"direction from {_one(target, 'cause')} to {_one(target, 'effect')}?"
        )
        tool_task = "Build a cube simulation and write its result document."
        provider = ScriptedJsonLanguageModel(
            [
                _intent_for(target, question),
                _response_for(target),
                _tool_proposal(tool_task),
            ],
            model="phase6-end-to-end-fixture",
        )
        provider_fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix="atom-phase6-integration-"
        ) as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            session = AtomHarnessSession(
                provider_fabric=provider_fabric,
                output_root=root / "session",
                forge_path=PROJECT_ROOT / DEFAULT_FORGE,
                evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
                model_path=PROJECT_ROOT / DEFAULT_MODEL,
            )
            hands = PermissionedToolFabric(
                provider_fabric=provider_fabric,
                knowledge_loader=session.preload_knowledge,
                workspace_root=workspace,
                state_root=root / "session/permissioned-hands",
                max_queue_depth=4,
            )
            operator = AtomHarnessOperator(
                session,
                state_root=root / "session",
                max_queue_depth=4,
                tool_fabric=hands,
            )
            server = build_server(operator)
            server_thread: threading.Thread | None = None
            try:
                startup = operator.start()
                self.assertEqual(startup["runtime"], ATOM_HARNESS_OPERATOR_RUNTIME)
                self.assertEqual(
                    startup["hands"]["runtime"], ATOM_PERMISSIONED_HANDS_RUNTIME
                )
                self.assertTrue(
                    startup["hands"]["permission_required_for_every_execution"]
                )
                self.assertEqual(
                    startup["preload"]["knowledge"]["wiki_runtime"],
                    ATOM_HARNESS_WIKI_RUNTIME,
                )
                self.assertEqual(
                    startup["preload"]["knowledge"]["rag_runtime"],
                    ATOM_HARNESS_RAG_RUNTIME,
                )
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
                self.assertIn(ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING, rendered)
                self.assertIn(ATOM_PERMISSIONED_HANDS_RUNTIME, rendered)
                self.assertIn(ATOM_TOOL_SIDE_VIEW_RUNTIME, rendered)
                self.assertIn(ATOM_TOOL_ARTIFACT_BINDING, rendered)
                self.assertEqual(
                    ATOM_TOOL_ARTIFACT_BINDING,
                    "render_atom_tool_artifact",
                )
                self.assertIn("Approve exact actions", rendered)
                self.assertIn("Planning failed closed. No tool action ran.", rendered)
                self.assertIn(
                    "Atom normalized the untrusted model proposal before permission",
                    rendered,
                )
                self.assertIn('sandbox=""', rendered)
                self.assertIn("AtomArtifactToken=", headers["Set-Cookie"])
                self.assertIn("AtomToolArtifactToken=", headers["Set-Cookie"])
                self.assertNotIn("unsafe-inline", headers["Content-Security-Policy"])
                self.assertIn(
                    "render_operator_surface",
                    render_operator_surface(
                        access_token="integration-token",
                        nonce="integration-nonce",
                    ),
                )

                status, health, _ = _request(server.origin + "/api/health")
                self.assertEqual(status, 200)
                health_payload = json.loads(health)
                self.assertEqual(
                    health_payload["runtime"],
                    ATOM_HARNESS_OPERATOR_SERVER_RUNTIME,
                )
                self.assertEqual(
                    health_payload["tool_runtime"],
                    ATOM_PERMISSIONED_HANDS_RUNTIME,
                )

                status, body, _ = _request(
                    server.origin + "/api/ask",
                    token=server.access_token,
                    origin=server.origin,
                    payload={"question": question},
                )
                self.assertEqual(status, 202)
                answer_request = json.loads(body)
                answer = operator.wait_for_terminal(
                    answer_request["request_id"],
                    timeout_seconds=90,
                )
                self.assertEqual(answer["status"], "completed")
                answer_transaction = verify_committed_run(Path(answer["output_dir"]))
                self.assertEqual(
                    answer_transaction["transaction_id"],
                    answer["artifact"]["transaction_id"],
                )

                status, body, _ = _request(
                    server.origin + "/api/tools/propose",
                    token=server.access_token,
                    origin=server.origin,
                    payload={"task": tool_task},
                )
                self.assertEqual(status, 202)
                proposed = json.loads(body)
                pending = hands.wait_for_status(
                    proposed["proposal_id"],
                    ("awaiting-permission",),
                    timeout_seconds=90,
                )
                self.assertFalse((workspace / "phase6-live").exists())

                tampered_manifest_hash = "0" * 64
                status, _, _ = _request(
                    server.origin + "/api/tools/approve",
                    token=server.access_token,
                    origin=server.origin,
                    payload={
                        "proposal_id": proposed["proposal_id"],
                        "manifest_hash": tampered_manifest_hash,
                        "decision_nonce": pending["decision_nonce"],
                    },
                )
                self.assertEqual(status, 409)
                self.assertFalse((workspace / "phase6-live").exists())

                status, _, _ = _request(
                    server.origin + "/api/tools/approve",
                    token=server.access_token,
                    origin=server.origin,
                    payload={
                        "proposal_id": proposed["proposal_id"],
                        "manifest_hash": pending["execution_manifest"]["manifest_hash"],
                        "decision_nonce": pending["decision_nonce"],
                    },
                )
                self.assertEqual(status, 202)
                terminal = hands.wait_for_status(
                    proposed["proposal_id"],
                    ("completed", "failed"),
                    timeout_seconds=90,
                )
                self.assertEqual(terminal["status"], "completed", terminal)
                self.assertTrue((workspace / "phase6-live/sim.py").is_file())
                self.assertTrue((workspace / "phase6-live/RESULT.md").is_file())
                tool_transaction = verify_committed_run(Path(terminal["output_dir"]))
                self.assertEqual(
                    tool_transaction["transaction_id"],
                    terminal["artifact"]["transaction_id"],
                )
                tool_artifact = json.loads(
                    (
                        Path(terminal["output_dir"]) / "atom_tool_artifact.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertTrue(tool_artifact["passed"])
                self.assertTrue(tool_artifact["checks"]["atom_memory_unchanged"])
                self.assertEqual(
                    tool_artifact["knowledge"]["wiki_runtime"],
                    ATOM_HARNESS_WIKI_RUNTIME,
                )
                self.assertEqual(
                    tool_artifact["knowledge"]["rag_runtime"],
                    ATOM_HARNESS_RAG_RUNTIME,
                )

                status, tool_view, tool_headers = _request(
                    server.origin
                    + "/api/tool-artifacts/"
                    + proposed["proposal_id"]
                    + "/side-view",
                    cookie="AtomToolArtifactToken=" + server.access_token,
                )
                self.assertEqual(status, 200)
                self.assertIn(ATOM_TOOL_SIDE_VIEW_RUNTIME.encode(), tool_view)
                self.assertIn(ATOM_TOOL_ARTIFACT_BINDING.encode(), tool_view)
                self.assertIn(b"cube-five", tool_view)
                self.assertEqual(tool_headers["X-Frame-Options"], "SAMEORIGIN")

                self.assertEqual(
                    [request.stage for request in provider.requests],
                    ["atom_intent", "atom_grounded_response", "tool.plan"],
                )
                self.assertEqual(
                    provider.requests[-1].data_sensitivity,
                    "private-operator-intent",
                )
                self.assertFalse(
                    provider.requests[-1].payload["authority"]["model_may_execute"]
                )
            finally:
                server.shutdown()
                server.server_close()
                if server_thread is not None:
                    server_thread.join(timeout=5)
                operator.shutdown(wait=True, cancel_pending=True)


if __name__ == "__main__":
    unittest.main()
