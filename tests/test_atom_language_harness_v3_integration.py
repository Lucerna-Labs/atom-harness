from __future__ import annotations

import json
import tempfile
import threading
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
from atom_harness_experiment import run_atom_language_harness
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_harness_runtime import (
    ATOM_LANGUAGE_HARNESS_RUNTIME,
    _lexical_anchors,
    _lexical_proposal,
    _validated_intent_for_question,
)
from atom_harness_side_view import (
    ATOM_HARNESS_SIDE_VIEW_RUNTIME,
    render_atom_harness_artifact,
)
from atom_llm_protocol import (
    ATOM_GROUNDED_RESPONSE_RUNTIME,
    ATOM_LANGUAGE_INTENT_RUNTIME,
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    ProviderLocation,
)
from atom_llm_provider import ScriptedJsonLanguageModel
from atom_provider_fabric import (
    ATOM_PROVIDER_FABRIC_RUNTIME,
    ATOM_PROVIDER_ROUTE_RUNTIME,
    ProviderFabric,
    ProviderFabricPolicy,
)
from atom_resident_language_lane import ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME
from atom_run_transaction import verify_committed_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_INTEGRATION_TEST = "tests/test_atom_language_harness_v3_integration.py"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


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


class _ResidentScriptedProvider(ScriptedJsonLanguageModel):
    def __init__(self, payloads: list[Mapping[str, Any]]) -> None:
        super().__init__(payloads, model="v3-resident-lane-fixture")
        self._lane_lock = threading.Lock()
        self._lane_ordinal = 0

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

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        result = super().generate_json(request, cancellation=cancellation)
        with self._lane_lock:
            self._lane_ordinal += 1
            ordinal = self._lane_ordinal
        vibrations = (
            [
                {
                    "kind": "vertical",
                    "signal": "resident-language-lane-cold-start",
                    "origin": "L0:resident-language-transport",
                    "propagates_to": [
                        "L1:language-message",
                        "L2:language-flow",
                        "L3:orchestration",
                    ],
                    "process_generation": 1,
                    "cold_start_ms": 7,
                    "warmup_ms": 1,
                }
            ]
            if ordinal == 1
            else []
        )
        lane = {
            "schema": 1,
            "runtime": ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
            "stage": request.stage,
            "process_generation": 1,
            "model_load_count": 1,
            "restart_count": 0,
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
            "vibrations": vibrations,
        }
        performance = {
            "runtime": "atom-resident-language-performance-v1",
            "cold_start_ms": 7 if ordinal == 1 else 0,
            "model_load_ms": 7 if ordinal == 1 else 0,
            "warm_request": ordinal > 1,
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


class AtomLanguageHarnessV3IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="atom-language-harness-v3-integration-"
        )
        cls.root = Path(cls.temporary.name)
        cls.output_dir = cls.root / "resident-run"
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
        cls.question = (
            f"In the {_one(cls.target, 'domain')} domain, what is the known "
            f"direction from {_one(cls.target, 'cause')} to "
            f"{_one(cls.target, 'effect')}?"
        )
        cls.provider = _ResidentScriptedProvider(
            [
                _intent_for(cls.target, cls.question),
                _response_for(cls.target),
            ]
        )
        cls.fabric = ProviderFabric(
            [cls.provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                allow_test_providers=True,
                max_retries_per_provider=0,
                max_concurrency=1,
            ),
        )
        cls.artifact = run_atom_language_harness(
            cls.output_dir,
            question=cls.question,
            language_model=cls.fabric,
            forge_path=PROJECT_ROOT / DEFAULT_FORGE,
            evidence_path=PROJECT_ROOT / DEFAULT_EVIDENCE,
            model_path=PROJECT_ROOT / DEFAULT_MODEL,
        )
        cls.workflow = _read_json(cls.output_dir / "atom_harness_workflow.json")
        cls.side_view = (cls.output_dir / "atom_harness_side_view.html").read_text(
            encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fabric.close()
        cls.temporary.cleanup()

    def test_resident_lane_is_runtime_wired_through_artifact_and_side_view(
        self,
    ) -> None:
        self.assertTrue(self.artifact["passed"])
        self.assertEqual(self.artifact["runtime"], ATOM_LANGUAGE_HARNESS_RUNTIME)
        self.assertEqual(
            self.artifact["language_model"]["provider_runtime"],
            ATOM_PROVIDER_FABRIC_RUNTIME,
        )
        self.assertEqual(len(self.artifact["completions"]), 2)
        for completion, route in zip(
            self.artifact["completions"],
            self.artifact["provider_routes"],
            strict=True,
        ):
            self.assertEqual(route["runtime"], ATOM_PROVIDER_ROUTE_RUNTIME)
            self.assertEqual(route["language_lane"], completion["language_lane"])
            core = {key: route[key] for key in sorted(route) if key != "route_hash"}
            self.assertEqual(route["route_hash"], canonical_hash(core))
            self.assertEqual(completion["language_lane"]["model_load_count"], 1)
        self.assertFalse(
            self.artifact["completions"][0]["language_lane"]["resident_reused"]
        )
        self.assertTrue(
            self.artifact["completions"][1]["language_lane"]["resident_reused"]
        )
        self.assertTrue(self.artifact["checks"]["wiki_graph_and_rag_are_runtime_wired"])
        self.assertTrue(self.artifact["checks"]["resident_language_lane_is_hash_bound"])
        self.assertEqual(
            self.artifact["knowledge"]["wiki_runtime"],
            ATOM_HARNESS_WIKI_RUNTIME,
        )
        self.assertEqual(
            self.artifact["knowledge"]["rag_runtime"],
            ATOM_HARNESS_RAG_RUNTIME,
        )
        verify_committed_run(self.output_dir)
        self.assertIn(ATOM_HARNESS_SIDE_VIEW_RUNTIME, self.side_view)
        self.assertIn("resident model reused", self.side_view)
        self.assertIn("lane generation 1", self.side_view)
        self.assertIn("model loads 1", self.side_view)
        self.assertIn("restarts 0", self.side_view)
        self.assertIn("queue wait 0 ms", self.side_view)

    def test_declared_renderer_binds_the_committed_artifact_to_the_side_view(
        self,
    ) -> None:
        committed_artifact = _read_json(self.output_dir / "atom_harness_artifact.json")
        committed_workflow = _read_json(self.output_dir / "atom_harness_workflow.json")
        committed_graph = _read_json(self.output_dir / "atom_harness_wiki_graph.json")
        rendered = render_atom_harness_artifact(
            committed_artifact,
            committed_workflow,
            committed_graph,
        )
        self.assertEqual(rendered, self.side_view)

    def test_spiderweb_trace_exposes_typed_resident_highway(self) -> None:
        trace = self.artifact["spiderweb_trace"]
        resident_event = next(
            event
            for layer in trace["layers"]
            if layer["layer"] == "L0"
            for event in layer["events"]
            if event["type"] == "resident-language-lane"
        )
        self.assertEqual(resident_event["completion_count"], 2)
        self.assertEqual(resident_event["model_load_count"], 1)
        self.assertEqual(resident_event["reuse_count"], 1)
        resident_intersection = next(
            item
            for item in trace["intersections"]
            if item["identity"] == "resident-language-highway-intersection"
        )
        self.assertTrue(resident_intersection["emergent"])
        self.assertEqual(
            resident_intersection["transfer_policy"],
            "typed-ramp-only",
        )
        self.assertTrue(
            any(
                item.get("to") == "resident-language-highway"
                and item.get("message") == "JsonGenerationRequest"
                for item in trace["on_ramps"]
            )
        )
        self.assertTrue(
            any(
                item.get("from") == "resident-language-highway"
                and item.get("message") == "JsonGenerationResult"
                for item in trace["off_ramps"]
            )
        )

    def test_exact_lexical_anchors_promote_explicit_features_to_required(
        self,
    ) -> None:
        question = (
            "Using Atom evidence, describe how trust affects belief in the "
            "language domain and report the direction."
        )
        vocabulary = {
            "domain": ("agent", "language"),
            "cause": ("trust",),
            "effect": ("belief", "trustworthy"),
            "direction": ("+1", "-1"),
        }
        anchors = _lexical_anchors(question, vocabulary)
        self.assertEqual(
            anchors,
            {
                "domain": ["language"],
                "cause": ["trust"],
                "effect": ["belief"],
            },
        )
        self.assertEqual(
            _lexical_proposal(question, vocabulary),
            {
                "cause": ["trust"],
                "effect": ["belief"],
                "domain": ["language"],
            },
        )
        normalized = _validated_intent_for_question(
            {
                "schema": 1,
                "runtime": ATOM_LANGUAGE_INTENT_RUNTIME,
                "action": "retrieve",
                "question": question,
                "features": [
                    {
                        "role": "domain",
                        "value": "language",
                        "required": False,
                    },
                    {
                        "role": "cause",
                        "value": "trust",
                        "required": False,
                    },
                    {
                        "role": "effect",
                        "value": "belief",
                        "required": False,
                    },
                ],
            },
            vocabulary=vocabulary,
            question=question,
        )
        self.assertTrue(all(item["required"] for item in normalized["features"]))
        self.assertNotIn(
            "direction",
            {item["role"] for item in normalized["features"]},
        )
        self_relation = _validated_intent_for_question(
            {
                "schema": 1,
                "runtime": ATOM_LANGUAGE_INTENT_RUNTIME,
                "action": "abstain",
                "question": (
                    "What is the direction from trust to trust in biological?"
                ),
                "features": [],
            },
            vocabulary={
                "domain": ("biological",),
                "cause": ("trust",),
                "effect": ("trust",),
                "direction": ("+1", "-1"),
            },
            question="What is the direction from trust to trust in biological?",
        )
        self.assertEqual(self_relation["action"], "retrieve")
        self.assertEqual(
            {item["role"]: item["value"] for item in self_relation["features"]},
            {
                "domain": "biological",
                "cause": "trust",
                "effect": "trust",
            },
        )

    def test_registry_preserves_the_certified_v3_resident_runtime(self) -> None:
        registry = _read_json(PROJECT_ROOT / "ai-runtime-registry.json")
        side_view = _read_json(PROJECT_ROOT / "ai-artifact-side-view.json")
        fabric = _read_json(PROJECT_ROOT / "ai-provider-fabric.json")
        architecture = _read_json(
            PROJECT_ROOT / "atom-language-harness-architecture.json"
        )
        model = _read_json(PROJECT_ROOT / "atom-language-model.json")

        self.assertEqual(registry["active_runtime"], "language-harness-v6")
        historical = registry["runtimes"]["language-harness-v3"]
        self.assertEqual(historical["integration_test"], V3_INTEGRATION_TEST)
        self.assertEqual(
            historical["runtime_entrypoint"],
            "atom_harness_experiment.py",
        )
        self.assertEqual(architecture["runtime"], "atom-language-harness-operator-v6")
        self.assertEqual(
            architecture["language_membrane"]["configured_default_provider"],
            "LlamaCppResidentJsonLanguageModel",
        )
        self.assertEqual(
            architecture["language_membrane"]["provider_fabric_runtime"],
            ATOM_PROVIDER_FABRIC_RUNTIME,
        )
        self.assertEqual(
            model["runtime_policy"]["resident_lane"]["runtime"],
            ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME,
        )
        self.assertTrue(
            model["runtime_policy"]["resident_lane"]["external_proxy_disabled"]
        )
        self.assertEqual(
            model["adoption_status"],
            "certified-resident-local-default",
        )
        self.assertEqual(
            model["certification"]["latest_evidence"]["runtime"],
            "atom-resident-language-certification-v1",
        )
        self.assertTrue(side_view["side_view"]["resident_lane_state_visible"])
        self.assertTrue(fabric["provider_fabric"]["resident_model_preload"])
        self.assertTrue(
            fabric["provider_fabric"]["external_proxy_disabled_for_loopback"]
        )
        self.assertTrue(fabric["provider_fabric"]["supervised_crash_recovery"])
        self.assertEqual(
            side_view["side_view"]["artifact_binding_marker"],
            "render_operator_surface",
        )


if __name__ == "__main__":
    unittest.main()
