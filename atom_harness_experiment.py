"""Run a real language request through the Atom evidence harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_causal_memory import RELEASE_BINARY
from atom_causal_world_schema import canonical_hash
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
    bootstrap_harness_knowledge,
)
from atom_harness_runtime import (
    ATOM_LANGUAGE_HARNESS_RUNTIME,
    AtomLanguageHarness,
)
from atom_harness_side_view import (
    ATOM_HARNESS_SIDE_VIEW_RUNTIME,
    render_atom_harness_artifact,
)
from atom_language_model_contract import (
    default_official_model_path,
    load_language_model_contract,
    resolve_chat_template,
    resolve_model_integrity,
)
from atom_llm_provider import (
    LlamaCppResidentJsonLanguageModel,
    OpenRouterJsonLanguageModel,
    UnavailableJsonLanguageModel,
)
from atom_llm_protocol import (
    ATOM_ABSTENTION,
    ATOM_LANGUAGE_MODEL_PROTOCOL,
    CancellationToken,
    JsonLanguageModel,
    ProviderCancelledError,
    ProviderLocation,
)
from atom_provider_fabric import (
    ATOM_PROVIDER_FABRIC_RUNTIME,
    ATOM_PROVIDER_ROUTE_RUNTIME,
    ProviderFabric,
    ProviderFabricPolicy,
)
from atom_run_transaction import (
    ATOM_RUN_TRANSACTION_RUNTIME,
    RunTransaction,
    RunTransactionError,
    recover_transactions,
    verify_committed_run,
)


ATOM_HARNESS_EXPERIMENT_RUNTIME = "atom-language-harness-experiment-v3"
ATOM_HARNESS_WORKFLOW_RUNTIME = "atom-language-harness-workflow-v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provider_state_hashes_are_valid(payload: Mapping[str, Any]) -> bool:
    identity_core = {
        "schema": payload["schema"],
        "runtime": payload.get("runtime", ATOM_PROVIDER_FABRIC_RUNTIME),
        "protocol": payload["protocol"],
        "ordered": payload["ordered"],
        "policy": payload["policy"],
        "providers": [
            {key: value for key, value in provider.items() if key != "circuit"}
            for provider in payload["providers"]
        ],
    }
    state_core = {
        **identity_core,
        "providers": payload["providers"],
    }
    return payload["preload_hash"] == canonical_hash(identity_core) and payload[
        "state_hash"
    ] == canonical_hash(state_core)


def _checks(answer: Mapping[str, Any]) -> dict[str, bool]:
    packet = answer["evidence_packet"]
    response = answer["response"]
    trace = answer["spiderweb_trace"]
    allowed = {item["experience_id"] for item in packet["passages"]}
    layer_names = [item["layer"] for item in trace["layers"]]
    provider_routes = answer["provider_routes"]
    provider_policy = answer["provider_preload"]["policy"]
    allowed_locations = set(provider_policy["allowed_locations"])
    completion_routes = {
        item["stage"]: item for item in provider_routes if item["completed"]
    }
    completions_by_stage = {item["stage"]: item for item in answer["completions"]}
    intent_assistance = answer["intent_assistance"]
    resident_routes = [
        item
        for item in provider_routes
        if isinstance(item.get("language_lane"), Mapping)
    ]
    return {
        "language_model_is_replaceable_json_membrane": (
            answer["language_model"]["protocol"] == ATOM_LANGUAGE_MODEL_PROTOCOL
            and answer["language_model"]["provider_runtime"]
            == ATOM_PROVIDER_FABRIC_RUNTIME
        ),
        "intent_assistance_is_hash_bound_and_non_authoritative": (
            intent_assistance["runtime"] == "atom-exact-vocabulary-anchor-v1"
            and intent_assistance["semantic_authority"] is False
            and intent_assistance["assistance_hash"]
            == canonical_hash(
                {
                    key: intent_assistance[key]
                    for key in sorted(intent_assistance)
                    if key != "assistance_hash"
                }
            )
        ),
        "wiki_graph_and_rag_are_runtime_wired": (
            answer["knowledge"]["wiki_runtime"] == ATOM_HARNESS_WIKI_RUNTIME
            and answer["knowledge"]["rag_runtime"] == ATOM_HARNESS_RAG_RUNTIME
            and answer["knowledge"]["node_count"] > 0
            and answer["knowledge"]["edge_count"] > 0
        ),
        "evidence_packet_is_hash_bound": (
            packet["packet_hash"]
            == canonical_hash(
                {key: packet[key] for key in sorted(packet) if key != "packet_hash"}
            )
        ),
        "response_citations_are_packet_local": (
            set(response["citations"]) <= allowed
            and (not response["answerable"] or bool(response["citations"]))
        ),
        "response_grounding_matches_primary_claim": (
            (
                response["answerable"]
                and response["grounding"] == packet["primary_claim"]
            )
            or (not response["answerable"] and response["grounding"] is None)
        ),
        "graph_snapshot_is_bound_end_to_end": (
            packet["graph_knowledge_hash"]
            == answer["knowledge"]["graph_knowledge_hash"]
        ),
        "insufficient_evidence_forces_abstention": (
            not packet["insufficient_evidence"]
            or (
                response["answerable"] is False
                and response["citations"] == []
                and response["grounding"] is None
            )
        ),
        "llm_cannot_write_atom_memory": (
            answer["memory"]["unchanged"] is True
            and answer["memory"]["llm_write_access"] is False
            and answer["memory"]["store_sha256_before"]
            == answer["memory"]["store_sha256_after"]
        ),
        "spiderweb_layers_are_preserved": (
            layer_names == ["L0", "L1", "L2", "L3"]
            and trace["thread"]["formed_from_observed_flow"] is True
            and bool(trace["on_ramps"])
            and bool(trace["off_ramps"])
            and bool(trace["vibrations"])
        ),
        "provider_routes_are_hash_bound": (
            bool(provider_routes)
            and all(
                route["runtime"] == ATOM_PROVIDER_ROUTE_RUNTIME
                and route["data_sensitivity"] == "private-atom-evidence"
                and route["route_hash"]
                == canonical_hash(
                    {key: route[key] for key in sorted(route) if key != "route_hash"}
                )
                for route in provider_routes
            )
        ),
        "provider_selection_obeys_location_policy": all(
            route["selected_provider"] is None
            or (
                route["selected_provider"]["location"] in allowed_locations
                and (
                    route["selected_provider"]["location"] != "cloud"
                    or provider_policy["allow_cloud_data"] is True
                )
            )
            for route in provider_routes
        ),
        "provider_manifests_are_declared_secret_free": (
            answer["language_model"]["secrets_persisted"] is False
            and all(
                row["manifest"]["secrets_persisted"] is False
                for row in answer["provider_preload"]["providers"]
            )
        ),
        "completion_identity_matches_selected_provider": (
            len(answer["completions"]) == len(completion_routes)
            and all(
                completion["stage"] in completion_routes
                and completion["route_hash"]
                == completion_routes[completion["stage"]]["route_hash"]
                and completion["provider"]
                == completion_routes[completion["stage"]]["selected_provider"][
                    "provider_id"
                ]
                and completion["model"]
                == completion_routes[completion["stage"]]["selected_provider"]["model"]
                for completion in answer["completions"]
            )
        ),
        "resident_language_lane_is_hash_bound": all(
            isinstance(route.get("language_lane"), Mapping)
            and route["stage"] in completions_by_stage
            and route["language_lane"]
            == completions_by_stage[route["stage"]]["language_lane"]
            and isinstance(route["language_lane"].get("on_ramp"), Mapping)
            and isinstance(route["language_lane"].get("off_ramp"), Mapping)
            and route["language_lane"].get("model_load_count", 0) >= 1
            for route in resident_routes
        ),
        "provider_failure_is_fail_closed": (
            not answer["degraded"]
            or (
                answer["response"]["answerable"] is False
                and answer["response"]["answer"] == ATOM_ABSTENTION
                and answer["response"]["citations"] == []
            )
        ),
        "privacy_policy_is_preloaded": (
            answer["provider_preload"]["policy"]["allow_cloud_data"]
            is answer["language_model"]["policy"]["allow_cloud_data"]
            and answer["provider_preload"]["preload_hash"]
            == answer["language_model"]["preload_hash"]
            and _provider_state_hashes_are_valid(answer["provider_preload"])
            and _provider_state_hashes_are_valid(answer["language_model"])
        ),
        "wiki_vocabulary_is_preloaded": (
            trace["preload"]["performed_before_intent"] is True
            and trace["preload"]["vocabulary_hash"]
            == answer["knowledge"]["vocabulary_hash"]
        ),
        "spiderweb_trace_binds_real_provider_routes": (
            trace["provider_routes"] == provider_routes
            and trace["timings"] == answer["timings"]
        ),
    }


def run_atom_language_harness(
    output_dir: Path,
    *,
    question: str,
    language_model: JsonLanguageModel,
    forge_path: Path = DEFAULT_FORGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
    model_path: Path = DEFAULT_MODEL,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Publish one complete, verified harness run as an atomic bundle."""

    final_dir = Path(output_dir).resolve()
    recovery_events = recover_transactions(final_dir.parent)
    token = cancellation or CancellationToken()
    with RunTransaction(final_dir) as transaction:
        token.raise_if_cancelled()
        knowledge = bootstrap_harness_knowledge(
            transaction.staging_dir / "runtime",
            forge_path=Path(forge_path),
            evidence_path=Path(evidence_path),
            model_path=Path(model_path),
        )
        answer = AtomLanguageHarness(
            knowledge=knowledge,
            language_model=language_model,
        ).answer(question, cancellation=token)
        checks = _checks(answer)
        artifact_core: dict[str, Any] = {
            **answer,
            "experiment_runtime": ATOM_HARNESS_EXPERIMENT_RUNTIME,
            "passed": all(checks.values()),
            "checks": checks,
            "transaction": {
                "runtime": ATOM_RUN_TRANSACTION_RUNTIME,
                "transaction_id": transaction.transaction_id,
                "atomic_publication": True,
                "overwrite_allowed": False,
                "recovery_event_count": len(recovery_events),
                "recovery_actions": sorted(
                    {
                        str(item["action"])
                        for item in recovery_events
                        if "action" in item
                    }
                ),
            },
            "side_view_contract": {
                "runtime": ATOM_HARNESS_SIDE_VIEW_RUNTIME,
                "artifact_binding_marker": "render_atom_harness_artifact",
                "placement": "side",
                "user_visible": True,
                "bound_to_real_output": True,
            },
        }
        artifact = {
            **artifact_core,
            "artifact_hash": canonical_hash(artifact_core),
        }
        workflow_core = {
            "schema": 1,
            "runtime": ATOM_HARNESS_WORKFLOW_RUNTIME,
            "harness_runtime": ATOM_LANGUAGE_HARNESS_RUNTIME,
            "transaction_runtime": ATOM_RUN_TRANSACTION_RUNTIME,
            "transaction_id": transaction.transaction_id,
            "artifact_hash": artifact["artifact_hash"],
            "evidence_packet_hash": artifact["evidence_packet"]["packet_hash"],
            "provider_route_hashes": [
                item["route_hash"] for item in artifact["provider_routes"]
            ],
            "knowledge_hash": artifact["knowledge"]["knowledge_hash"],
            "graph_knowledge_hash": knowledge.graph_manifest["knowledge_hash"],
            "store_sha256": artifact["memory"]["store_sha256_after"],
            "binary_sha256": _sha256(RELEASE_BINARY),
            "model_manifest_hash": canonical_hash(artifact["language_model"]),
            "wiki_runtime": ATOM_HARNESS_WIKI_RUNTIME,
            "rag_runtime": ATOM_HARNESS_RAG_RUNTIME,
            "side_view_runtime": ATOM_HARNESS_SIDE_VIEW_RUNTIME,
        }
        workflow = {
            **workflow_core,
            "workflow_hash": canonical_hash(workflow_core),
        }
        side_view = render_atom_harness_artifact(
            artifact,
            workflow,
            knowledge.graph_manifest,
        )

        transaction.write_json("atom_harness_artifact.json", artifact)
        transaction.write_json("atom_harness_workflow.json", workflow)
        transaction.write_json(
            "atom_harness_knowledge.json",
            knowledge.manifest(),
        )
        transaction.write_json(
            "atom_harness_wiki_graph.json",
            knowledge.graph_manifest,
        )
        transaction.write_json(
            "atom_harness_evidence_packet.json",
            artifact["evidence_packet"],
        )
        transaction.write_text("atom_harness_side_view.html", side_view)
        if not artifact["passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError("Atom harness checks failed: " + ", ".join(failed))
        token.raise_if_cancelled()
        transaction.seal(
            required_files=(
                "atom_harness_artifact.json",
                "atom_harness_workflow.json",
                "atom_harness_knowledge.json",
                "atom_harness_wiki_graph.json",
                "atom_harness_evidence_packet.json",
                "atom_harness_side_view.html",
                "runtime/atom_harness_knowledge.atomdb",
            )
        )
        transaction.commit()
    verify_committed_run(final_dir)
    return artifact


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return Path("atom_harness_outputs") / f"run-{stamp}"


def _provider_chain(raw: str) -> tuple[str, ...]:
    names = tuple(item.strip().lower() for item in str(raw).split(",") if item.strip())
    if not names:
        raise ValueError("provider chain is empty")
    if len(names) != len(set(names)):
        raise ValueError("provider chain contains duplicates")
    unknown = sorted(set(names) - {"llama-cpp", "openrouter"})
    if unknown:
        raise ValueError("unknown providers: " + ", ".join(unknown))
    return names


def _build_provider_fabric(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> ProviderFabric:
    raw_chain = (
        arguments.providers
        or arguments.provider
        or os.environ.get("ATOM_LLM_PROVIDERS")
        or os.environ.get("ATOM_LLM_PROVIDER")
        or "llama-cpp"
    )
    try:
        names = _provider_chain(raw_chain)
    except ValueError as error:
        parser.error(str(error))
    providers: list[JsonLanguageModel] = []
    for name in names:
        if name == "openrouter":
            if arguments.llm_model:
                providers.append(
                    OpenRouterJsonLanguageModel(
                        arguments.llm_model,
                        timeout_seconds=arguments.provider_timeout_seconds,
                    )
                )
            else:
                providers.append(
                    UnavailableJsonLanguageModel(
                        "openrouter",
                        model="unconfigured-cloud-model",
                        location=ProviderLocation.CLOUD,
                        reason="--llm-model or ATOM_LLM_MODEL is absent",
                    )
                )
        elif arguments.model_path is None:
            providers.append(
                UnavailableJsonLanguageModel(
                    "llama-cpp",
                    model="unconfigured-gguf",
                    location=ProviderLocation.LOCAL,
                    reason="--model-path or ATOM_LLM_MODEL_PATH is absent",
                )
            )
        else:
            try:
                expected_sha256, expected_bytes = resolve_model_integrity(
                    arguments.model_path,
                    expected_sha256=arguments.model_sha256,
                    expected_bytes=arguments.model_bytes,
                )
                chat_template = resolve_chat_template(
                    arguments.model_path,
                    chat_template=arguments.chat_template,
                )
                providers.append(
                    LlamaCppResidentJsonLanguageModel(
                        arguments.model_path,
                        executable=arguments.llama_server,
                        expected_model_sha256=expected_sha256,
                        expected_model_bytes=expected_bytes,
                        chat_template=chat_template,
                        context_length=arguments.context_length,
                        gpu_layers=arguments.gpu_layers,
                        timeout_seconds=arguments.provider_timeout_seconds,
                        startup_timeout_seconds=(
                            arguments.lane_startup_timeout_seconds
                        ),
                        lane_acquire_timeout_seconds=(
                            arguments.lane_acquire_timeout_seconds
                        ),
                        parallel_slots=arguments.lane_parallel_slots,
                        max_queue_depth=arguments.lane_max_queue_depth,
                    )
                )
            except ValueError as error:
                providers.append(
                    UnavailableJsonLanguageModel(
                        "llama-cpp",
                        model=Path(arguments.model_path).name,
                        location=ProviderLocation.LOCAL,
                        reason=(
                            "local model configuration failed validation "
                            f"({type(error).__name__})"
                        ),
                    )
                )
    locations = {ProviderLocation.LOCAL, ProviderLocation.PRIVATE}
    if arguments.allow_cloud:
        locations.add(ProviderLocation.CLOUD)
    return ProviderFabric(
        providers,
        policy=ProviderFabricPolicy(
            allowed_locations=frozenset(locations),
            allow_cloud_data=arguments.allow_cloud,
            max_retries_per_provider=arguments.max_provider_retries,
            retry_backoff_seconds=arguments.retry_backoff_seconds,
            circuit_failure_threshold=arguments.circuit_failure_threshold,
            circuit_cooldown_seconds=arguments.circuit_cooldown_seconds,
            max_concurrency=arguments.max_concurrency,
            acquire_timeout_seconds=arguments.acquire_timeout_seconds,
        ),
    )


def main() -> None:
    language_contract = load_language_model_contract()
    runtime_policy = language_contract["runtime_policy"]
    resident_policy = runtime_policy["resident_lane"]
    parser = argparse.ArgumentParser(
        description=(
            "Run Atom-owned causal evidence through a policy-routed "
            "LLM language membrane."
        )
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--providers")
    parser.add_argument("--provider", choices=("llama-cpp", "openrouter"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            Path(os.environ["ATOM_LLM_MODEL_PATH"])
            if os.environ.get("ATOM_LLM_MODEL_PATH")
            else default_official_model_path()
        ),
    )
    parser.add_argument(
        "--model-sha256",
        default=os.environ.get("ATOM_LLM_MODEL_SHA256"),
    )
    parser.add_argument("--model-bytes", type=int)
    parser.add_argument(
        "--chat-template",
        default=os.environ.get("ATOM_LLM_CHAT_TEMPLATE"),
    )
    parser.add_argument(
        "--llama-server",
        "--llama-completion",
        dest="llama_server",
        default=(
            os.environ.get("ATOM_LLAMA_SERVER")
            or os.environ.get("ATOM_LLAMA_COMPLETION")
            or runtime_policy["executable"]
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(runtime_policy["harness_context_tokens"]),
    )
    parser.add_argument(
        "--gpu-layers",
        default=os.environ.get("ATOM_LLM_GPU_LAYERS", "auto"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("ATOM_LLM_MODEL", ""),
    )
    parser.add_argument("--forge", type=Path, default=DEFAULT_FORGE)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--allow-cloud",
        action="store_true",
        default=os.environ.get("ATOM_ALLOW_CLOUD_DATA") == "1",
        help="Explicitly permit private Atom request data to reach cloud providers.",
    )
    parser.add_argument("--max-provider-retries", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.25)
    parser.add_argument("--circuit-failure-threshold", type=int, default=1)
    parser.add_argument("--circuit-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--acquire-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--provider-timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--lane-startup-timeout-seconds",
        type=int,
        default=int(resident_policy["startup_timeout_seconds"]),
    )
    parser.add_argument(
        "--lane-acquire-timeout-seconds",
        type=float,
        default=float(resident_policy["acquire_timeout_seconds"]),
    )
    parser.add_argument(
        "--lane-parallel-slots",
        type=int,
        default=int(
            os.environ.get(
                "ATOM_LLM_LANE_PARALLEL_SLOTS",
                resident_policy["parallel_slots"],
            )
        ),
    )
    parser.add_argument(
        "--lane-max-queue-depth",
        type=int,
        default=int(
            os.environ.get(
                "ATOM_LLM_LANE_MAX_QUEUE_DEPTH",
                resident_policy["max_queue_depth"],
            )
        ),
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or _default_output_dir()
    try:
        fabric = _build_provider_fabric(arguments, parser)
    except ValueError as error:
        parser.error(str(error))
    cancellation = CancellationToken()

    def cancel_request(signum, frame) -> None:
        del frame
        cancellation.cancel(f"received process signal {signum}")

    signal.signal(signal.SIGINT, cancel_request)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cancel_request)
    try:
        try:
            artifact = run_atom_language_harness(
                output_dir,
                question=arguments.question,
                language_model=fabric,
                forge_path=arguments.forge,
                evidence_path=arguments.evidence,
                model_path=arguments.model,
                cancellation=cancellation,
            )
        except ProviderCancelledError as error:
            parser.exit(130, f"Atom harness cancelled: {error}\n")
        except (FileExistsError, RunTransactionError) as error:
            parser.exit(2, f"Atom harness transaction refused: {error}\n")
        print(
            json.dumps(
                {
                    "passed": artifact["passed"],
                    "answerable": artifact["response"]["answerable"],
                    "answer": artifact["response"]["answer"],
                    "citations": artifact["response"]["citations"],
                    "outcome": artifact["outcome"],
                    "degraded": artifact["degraded"],
                    "provider_routes": [
                        {
                            "stage": item["stage"],
                            "disposition": item["disposition"],
                            "selected_provider": item["selected_provider"],
                        }
                        for item in artifact["provider_routes"]
                    ],
                    "transaction_id": artifact["transaction"]["transaction_id"],
                    "output_dir": str(output_dir.resolve()),
                    "side_view": str(
                        (output_dir / "atom_harness_side_view.html").resolve()
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    finally:
        fabric.close()


if __name__ == "__main__":
    main()
