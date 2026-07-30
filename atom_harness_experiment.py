"""Run a real language request through the Atom evidence harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from atom_llm_provider import (
    LlamaCppJsonLanguageModel,
    OpenRouterJsonLanguageModel,
)
from atom_llm_protocol import JsonLanguageModel


ATOM_HARNESS_EXPERIMENT_RUNTIME = "atom-language-harness-experiment-v1"
ATOM_HARNESS_WORKFLOW_RUNTIME = "atom-language-harness-workflow-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checks(answer: Mapping[str, Any]) -> dict[str, bool]:
    packet = answer["evidence_packet"]
    response = answer["response"]
    trace = answer["spiderweb_trace"]
    allowed = {item["experience_id"] for item in packet["passages"]}
    layer_names = [item["layer"] for item in trace["layers"]]
    return {
        "language_model_is_replaceable_json_membrane": (
            answer["language_model"]["protocol"] == "atom-json-language-model-v1"
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
        "insufficient_evidence_forces_abstention": (
            not packet["insufficient_evidence"]
            or (response["answerable"] is False and response["citations"] == [])
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
        "wiki_vocabulary_is_preloaded": (
            trace["preload"]["performed_before_intent"] is True
            and trace["preload"]["vocabulary_hash"]
            == answer["knowledge"]["vocabulary_hash"]
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
) -> dict[str, Any]:
    """Bootstrap Atom knowledge, answer once, and render the bound side view."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "atom_harness_artifact.json"
    if artifact_path.exists():
        raise FileExistsError("Atom harness refuses to overwrite an existing artifact")
    knowledge = bootstrap_harness_knowledge(
        output_dir / "runtime",
        forge_path=Path(forge_path),
        evidence_path=Path(evidence_path),
        model_path=Path(model_path),
    )
    answer = AtomLanguageHarness(
        knowledge=knowledge,
        language_model=language_model,
    ).answer(question)
    checks = _checks(answer)
    artifact_core: dict[str, Any] = {
        **answer,
        "experiment_runtime": ATOM_HARNESS_EXPERIMENT_RUNTIME,
        "passed": all(checks.values()),
        "checks": checks,
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
        "artifact_hash": artifact["artifact_hash"],
        "evidence_packet_hash": artifact["evidence_packet"]["packet_hash"],
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

    _write_json(artifact_path, artifact)
    _write_json(
        output_dir / "atom_harness_workflow.json",
        workflow,
    )
    _write_json(
        output_dir / "atom_harness_knowledge.json",
        knowledge.manifest(),
    )
    _write_json(
        output_dir / "atom_harness_wiki_graph.json",
        knowledge.graph_manifest,
    )
    _write_json(
        output_dir / "atom_harness_evidence_packet.json",
        artifact["evidence_packet"],
    )
    (output_dir / "atom_harness_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    if not artifact["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Atom harness checks failed: " + ", ".join(failed))
    return artifact


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("atom_harness_outputs") / f"run-{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Atom-owned causal evidence through a local LLM language membrane."
        )
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--provider",
        choices=("llama-cpp", "openrouter"),
        default=os.environ.get(
            "ATOM_LLM_PROVIDER",
            ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "llama-cpp"),
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=(
            Path(os.environ["ATOM_LLM_MODEL_PATH"])
            if os.environ.get("ATOM_LLM_MODEL_PATH")
            else None
        ),
    )
    parser.add_argument(
        "--llama-cli",
        default=os.environ.get("ATOM_LLAMA_CLI", "llama-cli"),
    )
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--gpu-layers",
        default=os.environ.get("ATOM_LLM_GPU_LAYERS", "auto"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get(
            "ATOM_LLM_MODEL",
            "mistralai/mistral-small-3.2-24b-instruct",
        ),
    )
    parser.add_argument("--forge", type=Path, default=DEFAULT_FORGE)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    arguments = parser.parse_args()
    if arguments.provider == "llama-cpp" and arguments.model_path is None:
        parser.error("--model-path or ATOM_LLM_MODEL_PATH is required")
    output_dir = arguments.output_dir or _default_output_dir()
    if arguments.provider == "openrouter":
        provider = OpenRouterJsonLanguageModel(arguments.llm_model)
    else:
        provider = LlamaCppJsonLanguageModel(
            arguments.model_path,
            executable=arguments.llama_cli,
            context_length=arguments.context_length,
            gpu_layers=arguments.gpu_layers,
        )
    artifact = run_atom_language_harness(
        output_dir,
        question=arguments.question,
        language_model=provider,
        forge_path=arguments.forge,
        evidence_path=arguments.evidence,
        model_path=arguments.model,
    )
    print(
        json.dumps(
            {
                "passed": artifact["passed"],
                "answerable": artifact["response"]["answerable"],
                "answer": artifact["response"]["answer"],
                "citations": artifact["response"]["citations"],
                "output_dir": str(output_dir.resolve()),
                "side_view": str(
                    (output_dir / "atom_harness_side_view.html").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
