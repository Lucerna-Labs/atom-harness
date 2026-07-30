"""Live certification for the official Atom Harness language model."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom_causal_experience import load_experience_corpus
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
)
from atom_harness_experiment import run_atom_language_harness
from atom_language_model_contract import (
    default_official_model_path,
    load_language_model_contract,
    resolve_chat_template,
    resolve_model_integrity,
)
from atom_llm_provider import (
    LLAMA_CPP_PERFORMANCE_RUNTIME,
    LLAMA_CPP_PROVIDER_RUNTIME,
    LlamaCppJsonLanguageModel,
)
from atom_llm_protocol import ProviderLocation
from atom_provider_fabric import ProviderFabric, ProviderFabricPolicy
from atom_run_transaction import verify_committed_run


CERTIFICATION_RUNTIME = "atom-language-model-certification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(record, role: str) -> str:
    values = record.feature_values(role)
    if len(values) != 1:
        raise ValueError(f"certification record has invalid {role}")
    return values[0]


def _llama_cpp_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("llama.cpp version probe failed") from error
    rendered = "\n".join(
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    )
    if not rendered:
        raise RuntimeError("llama.cpp version probe returned no identity")
    return rendered[:2048]


def _selected_local_model(route: Mapping[str, Any], model_name: str) -> bool:
    selected = route.get("selected_provider")
    return (
        isinstance(selected, Mapping)
        and selected.get("provider_id") == LLAMA_CPP_PROVIDER_RUNTIME
        and selected.get("model") == model_name
        and selected.get("location") == ProviderLocation.LOCAL.value
    )


def _performance_rows(
    artifact: Mapping[str, Any],
    *,
    case_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for completion in artifact["completions"]:
        performance = completion.get("performance")
        if not isinstance(performance, Mapping):
            continue
        rows.append(
            {
                "case": case_name,
                "stage": completion["stage"],
                "elapsed_ms": completion["elapsed_ms"],
                **dict(performance),
            }
        )
    return rows


def _certify_case(
    *,
    output_dir: Path,
    name: str,
    question: str,
    expected_answerable: bool,
    fabric: ProviderFabric,
    model_name: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = run_atom_language_harness(
        output_dir,
        question=question,
        language_model=fabric,
        forge_path=ROOT / DEFAULT_FORGE,
        evidence_path=ROOT / DEFAULT_EVIDENCE,
        model_path=ROOT / DEFAULT_MODEL,
    )
    transaction = verify_committed_run(output_dir)
    artifact_path = output_dir / "atom_harness_artifact.json"
    side_view_path = output_dir / "atom_harness_side_view.html"
    side_view = side_view_path.read_text(encoding="utf-8")
    provider_rows = artifact["provider_preload"]["providers"]
    provider_manifest = provider_rows[0]["manifest"] if provider_rows else {}
    performance = _performance_rows(artifact, case_name=name)
    expected_completions = 2 if expected_answerable else 1
    checks = {
        "artifact_passed": artifact["passed"] is True,
        "answerability_matches": (
            artifact["response"]["answerable"] is expected_answerable
        ),
        "not_degraded": artifact["degraded"] is False,
        "expected_completion_count": (
            len(artifact["completions"]) == expected_completions
        ),
        "all_routes_completed_locally": (
            len(artifact["provider_routes"]) == expected_completions
            and all(
                route["completed"] and _selected_local_model(route, model_name)
                for route in artifact["provider_routes"]
            )
        ),
        "model_hash_admitted": (
            provider_manifest.get("model_sha256") == expected_sha256
        ),
        "wiki_and_rag_executed": (
            artifact["knowledge"]["node_count"] > 0
            and artifact["knowledge"]["edge_count"] > 0
            and artifact["checks"]["wiki_graph_and_rag_are_runtime_wired"]
        ),
        "citations_closed_world": artifact["checks"][
            "response_citations_are_packet_local"
        ],
        "machine_grounding_matches": artifact["checks"][
            "response_grounding_matches_primary_claim"
        ],
        "memory_unchanged": artifact["checks"]["llm_cannot_write_atom_memory"],
        "transaction_committed": transaction["state"] == "committed",
        "side_view_bound_to_model": (
            model_name in side_view
            and "Language performance" in side_view
            and "Bound evidence" in side_view
        ),
        "performance_emitted_for_every_completion": (
            len(performance) == expected_completions
            and all(
                item.get("runtime") == LLAMA_CPP_PERFORMANCE_RUNTIME
                and isinstance(item.get("load_ms"), (int, float))
                and item.get("load_ms", 0) > 0
                and isinstance(
                    item.get("generation_tokens_per_second"),
                    (int, float),
                )
                and item.get("generation_tokens_per_second", 0) > 0
                for item in performance
            )
        ),
    }
    return (
        {
            "name": name,
            "question": question,
            "expected_answerable": expected_answerable,
            "passed": all(checks.values()),
            "checks": checks,
            "outcome": artifact["outcome"],
            "answer": artifact["response"]["answer"],
            "citations": artifact["response"]["citations"],
            "limitations": artifact["response"]["limitations"],
            "artifact_sha256": _sha256(artifact_path),
            "side_view_sha256": _sha256(side_view_path),
            "transaction_id": transaction["transaction_id"],
            "performance": performance,
        },
        performance,
    )


def _performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    load_values = [
        float(item["load_ms"])
        for item in rows
        if isinstance(item.get("load_ms"), (int, float))
    ]
    throughput_values = [
        float(item["generation_tokens_per_second"])
        for item in rows
        if isinstance(item.get("generation_tokens_per_second"), (int, float))
    ]
    return {
        "sample_count": len(rows),
        "model_load_latency_ms": {
            "samples": load_values,
            "median": (
                round(statistics.median(load_values), 3) if load_values else None
            ),
            "minimum": round(min(load_values), 3) if load_values else None,
            "maximum": round(max(load_values), 3) if load_values else None,
        },
        "generation_throughput_tokens_per_second": {
            "samples": throughput_values,
            "median": (
                round(statistics.median(throughput_values), 3)
                if throughput_values
                else None
            ),
            "minimum": (
                round(min(throughput_values), 3) if throughput_values else None
            ),
            "maximum": (
                round(max(throughput_values), 3) if throughput_values else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Certify the official local model through the full Atom Harness."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_official_model_path(ROOT),
    )
    parser.add_argument(
        "--llama-completion",
        "--llama-cli",
        dest="llama_completion",
        default="llama-completion",
    )
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--provider-timeout-seconds", type=int, default=600)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()

    contract = load_language_model_contract()
    expected_sha256, expected_bytes = resolve_model_integrity(arguments.model_path)
    chat_template = resolve_chat_template(arguments.model_path)
    provider = LlamaCppJsonLanguageModel(
        arguments.model_path,
        executable=arguments.llama_completion,
        expected_model_sha256=expected_sha256,
        expected_model_bytes=expected_bytes,
        chat_template=chat_template,
        context_length=arguments.context_length,
        gpu_layers=arguments.gpu_layers,
        timeout_seconds=arguments.provider_timeout_seconds,
    )
    corpus = load_experience_corpus(
        ROOT / DEFAULT_EVIDENCE,
        ROOT / DEFAULT_MODEL,
    )
    target = sorted(
        (
            record
            for record in corpus.laws
            if record.feature_values("status") == ("crystallized",)
        ),
        key=lambda item: item.experience_id,
    )[0]
    domain = _one(target, "domain")
    cause = _one(target, "cause")
    effect = _one(target, "effect")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_root = (
        arguments.output_root
        if arguments.output_root is not None
        else ROOT / "atom_harness_outputs" / f"model-certification-{stamp}"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    cases = (
        (
            "direct-known-relation",
            (
                f"In the {domain} domain, what is the known direction "
                f"from {cause} to {effect}?"
            ),
            True,
        ),
        (
            "paraphrased-known-relation",
            (
                f"Using only Atom evidence, tell me how {cause} affects "
                f"{effect} in {domain}."
            ),
            True,
        ),
        (
            "unsupported-open-world-question",
            "What is tomorrow's weather in Paris?",
            False,
        ),
    )
    results: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for name, question, expected_answerable in cases:
        fabric = ProviderFabric(
            [provider],
            policy=ProviderFabricPolicy(
                allowed_locations=frozenset({ProviderLocation.LOCAL}),
                max_retries_per_provider=0,
                circuit_failure_threshold=1,
                max_concurrency=1,
            ),
        )
        result, performance = _certify_case(
            output_dir=output_root / name,
            name=name,
            question=question,
            expected_answerable=expected_answerable,
            fabric=fabric,
            model_name=Path(arguments.model_path).name,
            expected_sha256=expected_sha256,
        )
        results.append(result)
        performance_rows.extend(performance)

    report = {
        "schema": 1,
        "runtime": CERTIFICATION_RUNTIME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in results),
        "contract_runtime": contract["runtime"],
        "contract_adoption_status": contract["adoption_status"],
        "base_model": contract["base_model"]["model_id"],
        "model": Path(arguments.model_path).name,
        "model_bytes": Path(arguments.model_path).stat().st_size,
        "model_sha256": expected_sha256,
        "llama_cpp_version": _llama_cpp_version(provider.executable),
        "context_length": arguments.context_length,
        "gpu_layers": str(arguments.gpu_layers),
        "chat_template": chat_template,
        "case_isolation": "fresh-provider-fabric-per-case",
        "case_results": results,
        "performance_summary": _performance_summary(performance_rows),
    }
    report_path = output_root / "atom_language_model_certification.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_report.replace(report_path)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "model": report["model"],
                "model_sha256": report["model_sha256"],
                "performance_summary": report["performance_summary"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
