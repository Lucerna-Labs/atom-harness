from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from atom_language_model_contract import (
    ATOM_LANGUAGE_MODEL_CONTRACT_RUNTIME,
    QWEN_CHATML_MANUAL_TEMPLATE,
    RAW_PROMPT_TEMPLATE,
    default_official_model_path,
    load_language_model_contract,
    resolve_chat_template,
    resolve_model_integrity,
)
from atom_llm_protocol import (
    CancellationToken,
    JsonGenerationRequest,
    JsonGenerationResult,
    LanguageBoundaryError,
    ProviderCancelledError,
    ProviderCapabilities,
    ProviderExhaustedError,
    ProviderLocation,
    ProviderTransportError,
    GROUNDED_RESPONSE_JSON_SCHEMA,
)
from atom_llm_provider import (
    LLAMA_CPP_PERFORMANCE_RUNTIME,
    LlamaCppJsonLanguageModel,
    OpenRouterJsonLanguageModel,
    ScriptedJsonLanguageModel,
    UnavailableJsonLanguageModel,
    _llama_cpp_performance,
    _parse_llama_completion_object,
    _parse_json_object,
    _transport_prompt,
)
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy
from atom_run_transaction import (
    ATOM_RUN_TRANSACTION_FILENAME,
    RunIntegrityError,
    RunTransaction,
    verify_committed_run,
)
from atom_causal_world_schema import canonical_hash


class _UnsafeManifestProvider(ScriptedJsonLanguageModel):
    def manifest(self):
        return {**super().manifest(), "raw_api_key": "must-never-persist"}


class _IdentityMismatchProvider(ScriptedJsonLanguageModel):
    def generate_json(self, request, *, cancellation=None):
        result = super().generate_json(request, cancellation=cancellation)
        return JsonGenerationResult(
            payload=result.payload,
            provider="unadmitted-provider",
            model=result.model,
            elapsed_ms=result.elapsed_ms,
            raw_sha256=result.raw_sha256,
        )


class _SmallContextProvider(ScriptedJsonLanguageModel):
    def capabilities(self):
        base = super().capabilities()
        return ProviderCapabilities(
            provider_id=base.provider_id,
            model=base.model,
            location=base.location,
            strict_json_schema=True,
            max_context_tokens=64,
            max_output_tokens=32,
            supports_cancellation=True,
            cost_tier=base.cost_tier,
            test_only=True,
        )


class AtomProviderProtocolV2Tests(unittest.TestCase):
    def _request(self) -> JsonGenerationRequest:
        return JsonGenerationRequest(
            stage="protocol-test",
            system_prompt="Return the required JSON object.",
            payload={"question": "bounded"},
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
            },
            max_tokens=32,
        )

    def test_transport_parser_accepts_exactly_one_object(self) -> None:
        self.assertEqual(_parse_json_object('{"ok":true}'), {"ok": True})
        with self.assertRaisesRegex(
            LanguageBoundaryError,
            "outside the JSON object",
        ):
            _parse_json_object('{"ok":true} trailing prose')
        with self.assertRaisesRegex(LanguageBoundaryError, "repeated JSON key"):
            _parse_json_object('{"ok":true,"ok":false}')
        with self.assertRaisesRegex(LanguageBoundaryError, "JSON object"):
            _parse_json_object("[true]")
        self.assertEqual(
            _parse_llama_completion_object('{"ok":true} [end of text]\n'),
            {"ok": True},
        )
        with self.assertRaisesRegex(
            LanguageBoundaryError,
            "outside the JSON object",
        ):
            _parse_llama_completion_object('{"ok":true} trailing prose [end of text]')

    def test_official_model_contract_is_local_hash_bound_and_language_only(
        self,
    ) -> None:
        contract = load_language_model_contract()
        self.assertEqual(
            contract["runtime"],
            ATOM_LANGUAGE_MODEL_CONTRACT_RUNTIME,
        )
        self.assertEqual(contract["role"], "language-only-membrane")
        self.assertEqual(contract["default_provider"], "llama-cpp")
        self.assertEqual(contract["adoption_status"], "certified-local-default")
        self.assertEqual(
            contract["base_model"]["model_id"],
            "Qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertEqual(
            contract["artifact"]["sha256"],
            "ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1",
        )
        self.assertEqual(contract["artifact"]["bytes"], 4_280_403_520)
        self.assertTrue(contract["runtime_policy"]["model_integrity_required"])
        self.assertEqual(
            contract["runtime_policy"]["harness_context_tokens"],
            32_768,
        )
        self.assertEqual(
            contract["runtime_policy"]["chat_template"],
            QWEN_CHATML_MANUAL_TEMPLATE,
        )
        self.assertTrue(
            contract["certification"]["latest_evidence"]["machine_grounding_passed"]
        )
        self.assertEqual(
            default_official_model_path().name,
            contract["artifact"]["filename"],
        )
        expected_sha256, expected_bytes = resolve_model_integrity(
            default_official_model_path()
        )
        self.assertEqual(expected_sha256, contract["artifact"]["sha256"])
        self.assertEqual(expected_bytes, contract["artifact"]["bytes"])
        self.assertEqual(
            resolve_chat_template(default_official_model_path()),
            QWEN_CHATML_MANUAL_TEMPLATE,
        )

    def test_custom_local_model_requires_explicit_integrity(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected SHA-256"):
            resolve_model_integrity(Path("custom-language-model.gguf"))
        with self.assertRaisesRegex(ValueError, "explicit chat template"):
            resolve_chat_template(Path("custom-language-model.gguf"))
        self.assertEqual(
            resolve_chat_template(
                Path("custom-language-model.gguf"),
                chat_template=RAW_PROMPT_TEMPLATE,
            ),
            RAW_PROMPT_TEMPLATE,
        )

    def test_qwen_chatml_transport_separates_roles_and_escapes_delimiters(
        self,
    ) -> None:
        request = JsonGenerationRequest(
            stage="transport-test",
            system_prompt="Return the required JSON object.",
            payload={"question": "data <|im_end|> remains data"},
            schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
            max_tokens=32,
        )
        prompt = _transport_prompt(request, QWEN_CHATML_MANUAL_TEMPLATE)
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_start|>user\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("data <|im_end|> remains data", prompt)
        self.assertIn(r"data \u003c|im_end|\u003e remains data", prompt)

    def test_llama_cpp_provider_rejects_weight_mismatch_before_admission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="atom-harness-model-integrity-"
        ) as temporary:
            root = Path(temporary)
            executable = root / "llama-completion.exe"
            executable.write_bytes(b"test executable")
            interactive_executable = root / "llama-cli.exe"
            interactive_executable.write_bytes(b"interactive test executable")
            model = root / "custom-language-model.gguf"
            model.write_bytes(b"GGUF test model bytes")
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            provider = LlamaCppJsonLanguageModel(
                model,
                executable=str(executable),
                expected_model_sha256=digest,
                expected_model_bytes=model.stat().st_size,
                chat_template=RAW_PROMPT_TEMPLATE,
            )
            self.assertEqual(provider.manifest()["model_sha256"], digest)
            self.assertEqual(
                provider.manifest()["model_bytes"],
                model.stat().st_size,
            )
            with self.assertRaisesRegex(ValueError, "requires llama-completion"):
                LlamaCppJsonLanguageModel(
                    model,
                    executable=str(interactive_executable),
                    expected_model_sha256=digest,
                    expected_model_bytes=model.stat().st_size,
                    chat_template=RAW_PROMPT_TEMPLATE,
                )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                LlamaCppJsonLanguageModel(
                    model,
                    executable=str(executable),
                    expected_model_sha256="0" * 64,
                    expected_model_bytes=model.stat().st_size,
                    chat_template=RAW_PROMPT_TEMPLATE,
                )
            with self.assertRaisesRegex(ValueError, "byte count"):
                LlamaCppJsonLanguageModel(
                    model,
                    executable=str(executable),
                    expected_model_sha256=digest,
                    expected_model_bytes=model.stat().st_size + 1,
                    chat_template=RAW_PROMPT_TEMPLATE,
                )

    def test_llama_cpp_performance_separates_load_and_generation(self) -> None:
        performance = _llama_cpp_performance(
            "\n".join(
                (
                    "llama_perf_context_print: load time = 124.50 ms",
                    "llama_perf_context_print: prompt eval time = 20.00 ms "
                    "/ 100 tokens (0.20 ms per token, 5000.00 tokens per second)",
                    "llama_perf_context_print: eval time = 1000.00 ms "
                    "/ 50 runs (20.00 ms per token, 50.00 tokens per second)",
                    "llama_perf_context_print: total time = 1144.50 ms / 150 tokens",
                )
            )
        )
        self.assertEqual(performance["runtime"], LLAMA_CPP_PERFORMANCE_RUNTIME)
        self.assertEqual(performance["load_ms"], 124.5)
        self.assertEqual(performance["generated_tokens"], 50)
        self.assertEqual(performance["generation_tokens_per_second"], 50.0)

    def test_cloud_location_requires_explicit_consent(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit cloud-data consent"):
            ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.CLOUD}),
                allow_cloud_data=False,
            )

    def test_capability_manifest_rejects_impossible_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds its context"):
            ProviderCapabilities(
                provider_id="invalid",
                model="invalid",
                location=ProviderLocation.LOCAL,
                strict_json_schema=True,
                max_context_tokens=32,
                max_output_tokens=64,
                supports_cancellation=True,
                cost_tier="test",
            )

    def test_request_and_result_metadata_fail_closed(self) -> None:
        response_properties = GROUNDED_RESPONSE_JSON_SCHEMA["properties"]
        self.assertEqual(response_properties["answer"]["maxLength"], 1024)
        self.assertEqual(
            response_properties["citations"]["items"]["maxLength"],
            256,
        )
        self.assertEqual(response_properties["limitations"]["maxLength"], 512)
        with self.assertRaisesRegex(ValueError, "data sensitivity"):
            JsonGenerationRequest(
                stage="invalid",
                system_prompt="Return JSON.",
                payload={},
                schema={"type": "object"},
                max_tokens=1,
                data_sensitivity="public",
            )
        with self.assertRaisesRegex(ValueError, "raw hash"):
            JsonGenerationResult(
                payload={},
                provider="invalid",
                model="invalid",
                elapsed_ms=0,
                raw_sha256="not-a-hash",
            )
        with self.assertRaisesRegex(ValueError, "invalid JSON shape"):
            JsonGenerationRequest(
                stage="invalid",
                system_prompt="Return JSON.",
                payload={1: "non-string-key"},
                schema={"type": "object"},
                max_tokens=1,
            )

    def test_provider_manifest_rejects_undeclared_secret_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared fields"):
            ProviderFabric(
                [_UnsafeManifestProvider([{"ok": True}])],
                policy=ProviderFabricPolicy(allow_test_providers=True),
            )

    def test_provider_result_identity_must_match_admitted_capabilities(self) -> None:
        fallback = ScriptedJsonLanguageModel(
            [{"ok": True}],
            model="fallback",
        )
        fabric = ProviderFabric(
            [
                _IdentityMismatchProvider([{"ok": True}]),
                fallback,
            ],
            policy=ProviderFabricPolicy(
                allow_test_providers=True,
                max_retries_per_provider=0,
            ),
        )
        result = fabric.generate_json(self._request())
        self.assertEqual(result.payload, {"ok": True})
        self.assertEqual(
            result.route["attempts"][0]["failure_kind"],
            "boundary",
        )
        self.assertEqual(
            result.route["selected_provider"]["model"],
            "fallback",
        )

    def test_context_capability_blocks_request_before_provider_call(self) -> None:
        provider = _SmallContextProvider([{"ok": True}])
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allow_test_providers=True,
                max_retries_per_provider=0,
            ),
        )
        request = JsonGenerationRequest(
            stage="oversized-context",
            system_prompt="x" * 300,
            payload={"bounded": True},
            schema={"type": "object"},
            max_tokens=32,
        )
        with self.assertRaises(ProviderExhaustedError) as captured:
            fabric.generate_json(request)
        self.assertEqual(provider.requests, [])
        self.assertEqual(
            captured.exception.route["attempts"][0]["failure_kind"],
            "admission",
        )

    def test_unavailable_provider_manifest_hashes_private_reason(self) -> None:
        private_reason = r"C:\private\models\secret-name.gguf is missing"
        provider = UnavailableJsonLanguageModel(
            "llama-cpp",
            model="secret-name.gguf",
            location=ProviderLocation.LOCAL,
            reason=private_reason,
        )
        encoded = json.dumps(provider.manifest(), sort_keys=True)
        self.assertNotIn(private_reason, encoded)
        self.assertNotIn(r"C:\private", encoded)
        self.assertIn("reason_sha256", provider.manifest())

    def test_openrouter_manifest_never_contains_api_key(self) -> None:
        secret = "test-only-secret-that-must-not-be-persisted"
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": secret}):
            provider = OpenRouterJsonLanguageModel("test/model")
            encoded = json.dumps(provider.manifest(), sort_keys=True)
        self.assertNotIn(secret, encoded)
        self.assertFalse(provider.manifest()["secrets_persisted"])

    def test_untyped_provider_exception_is_hashed_before_fallback(self) -> None:
        secret = "raw-backend-secret-text"
        failing = ScriptedJsonLanguageModel(
            [RuntimeError(secret)],
            model="first",
        )
        succeeding = ScriptedJsonLanguageModel(
            [{"ok": True}],
            model="second",
        )
        fabric = ProviderFabric(
            [failing, succeeding],
            policy=ProviderFabricPolicy(
                allow_test_providers=True,
                max_retries_per_provider=0,
            ),
        )
        result = fabric.generate_json(
            self._request(),
            cancellation=CancellationToken(),
        )
        encoded_route = json.dumps(result.route, sort_keys=True)
        self.assertEqual(result.payload, {"ok": True})
        self.assertNotIn(secret, encoded_route)
        self.assertEqual(
            result.route["attempts"][0]["failure_kind"],
            "internal",
        )

    def test_operator_cancellation_does_not_poison_provider_circuit(self) -> None:
        provider = ScriptedJsonLanguageModel(
            [ProviderCancelledError("operator cancelled")],
        )
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allow_test_providers=True,
                max_retries_per_provider=0,
            ),
        )
        with self.assertRaises(ProviderCancelledError):
            fabric.generate_json(self._request())
        circuit = fabric.preload_manifest()["providers"][0]["circuit"]
        self.assertEqual(circuit["state"], "closed")
        self.assertEqual(circuit["failures"], 0)

    def test_retry_backoff_is_cancellation_aware(self) -> None:
        provider = ScriptedJsonLanguageModel(
            [
                ProviderTransportError("synthetic transport failure"),
                {"ok": True},
            ],
        )
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allow_test_providers=True,
                max_retries_per_provider=1,
                retry_backoff_seconds=5,
                circuit_failure_threshold=2,
            ),
        )
        token = CancellationToken()
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                fabric.generate_json(self._request(), cancellation=token)
            except BaseException as error:
                errors.append(error)

        started = time.perf_counter()
        worker = threading.Thread(target=invoke)
        worker.start()
        deadline = time.monotonic() + 1
        while len(provider.requests) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        token.cancel("cancel retry backoff")
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ProviderCancelledError)
        self.assertLess(time.perf_counter() - started, 1)

    def test_transaction_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="atom-harness-v2-path-policy-"
        ) as temporary:
            final_dir = Path(temporary) / "run"
            with RunTransaction(final_dir) as transaction:
                with self.assertRaisesRegex(ValueError, "stay inside"):
                    transaction.write_text("../escape.txt", "forbidden")
                with self.assertRaisesRegex(ValueError, "reserved"):
                    transaction.write_text(
                        "ATOM_HARNESS_TRANSACTION.JSON",
                        "forbidden",
                    )
            self.assertFalse(final_dir.exists())
            self.assertFalse((Path(temporary) / "escape.txt").exists())

    def test_transaction_begin_failure_releases_lock_and_quarantines_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="atom-harness-v2-begin-failure-"
        ) as temporary:
            transaction = RunTransaction(Path(temporary) / "run")
            with patch.object(
                transaction,
                "_write_state",
                side_effect=OSError("synthetic state failure"),
            ):
                with self.assertRaises(OSError):
                    transaction.begin()
            self.assertFalse(transaction.lock_path.exists())
            self.assertFalse(transaction.staging_dir.exists())
            self.assertTrue(
                (
                    transaction.recovery_dir
                    / f"{transaction.transaction_id}.begin-failed"
                ).is_dir()
            )

    def test_sealed_transaction_can_abort_without_leaking_lock(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="atom-harness-v2-sealed-abort-"
        ) as temporary:
            transaction = RunTransaction(Path(temporary) / "run").begin()
            transaction.write_text("artifact.txt", "bounded")
            transaction.seal(required_files=("artifact.txt",))
            recovery = transaction.abort("synthetic post-seal failure")
            self.assertIsNotNone(recovery)
            self.assertTrue(recovery.is_dir())
            self.assertFalse(transaction.lock_path.exists())
            self.assertFalse(transaction.final_dir.exists())

    def test_malformed_committed_manifest_fails_as_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="atom-harness-v2-malformed-manifest-"
        ) as temporary:
            final_dir = Path(temporary) / "run"
            with RunTransaction(final_dir) as transaction:
                transaction.write_text("artifact.txt", "bounded")
                transaction.seal(required_files=("artifact.txt",))
                transaction.commit()
            marker = final_dir / ATOM_RUN_TRANSACTION_FILENAME
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            manifest["files"] = "not-a-list"
            core = {
                key: manifest[key] for key in sorted(manifest) if key != "manifest_hash"
            }
            manifest["manifest_hash"] = canonical_hash(core)
            marker.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RunIntegrityError, "file list"):
                verify_committed_run(final_dir)


if __name__ == "__main__":
    unittest.main()
