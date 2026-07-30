"""Command-line host for a multi-request resident Atom Harness session."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path
from typing import Any

from atom_causal_world_schema import canonical_hash
from atom_harness_session import AtomHarnessSession, default_session_output_root
from atom_llm_protocol import CancellationToken, ProviderCancelledError


ATOM_HARNESS_SESSION_REPORT_RUNTIME = "atom-resident-language-session-report-v1"


def _questions(arguments: argparse.Namespace) -> list[str]:
    values = list(arguments.question or [])
    if arguments.questions_file is not None:
        try:
            payload = json.loads(arguments.questions_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("session questions file is unavailable") from error
        if not isinstance(payload, list) or any(
            not isinstance(item, str) for item in payload
        ):
            raise ValueError("session questions file must contain a JSON string array")
        values.extend(payload)
    normalized = [item.strip() for item in values if item.strip()]
    if not 1 <= len(normalized) <= 256:
        raise ValueError("session requires between one and 256 questions")
    if any("\x00" in item or len(item) > 1024 for item in normalized):
        raise ValueError("session question is invalid")
    return normalized


def _atomic_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run several Atom questions through one resident Qwen lane."
    )
    parser.add_argument("--question", action="append")
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--llama-server")
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--provider-timeout-seconds", type=int, default=240)
    parser.add_argument("--lane-startup-timeout-seconds", type=int)
    parser.add_argument("--lane-acquire-timeout-seconds", type=float)
    parser.add_argument("--lane-parallel-slots", type=int)
    parser.add_argument("--lane-max-queue-depth", type=int)
    parser.add_argument("--max-concurrency", type=int, default=2)
    arguments = parser.parse_args()
    try:
        questions = _questions(arguments)
    except ValueError as error:
        parser.error(str(error))
    output_root = (arguments.output_root or default_session_output_root()).resolve()
    cancellation = CancellationToken()

    def cancel_session(signum, frame) -> None:
        del frame
        cancellation.cancel(f"received process signal {signum}")

    signal.signal(signal.SIGINT, cancel_session)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cancel_session)

    results: list[dict[str, Any]] = []
    session = AtomHarnessSession.official_local(
        output_root=output_root,
        model_path=arguments.model_path,
        llama_server=arguments.llama_server,
        gpu_layers=arguments.gpu_layers,
        provider_timeout_seconds=arguments.provider_timeout_seconds,
        startup_timeout_seconds=arguments.lane_startup_timeout_seconds,
        lane_acquire_timeout_seconds=arguments.lane_acquire_timeout_seconds,
        parallel_slots=arguments.lane_parallel_slots,
        max_queue_depth=arguments.lane_max_queue_depth,
        max_concurrency=arguments.max_concurrency,
    )
    try:
        for question in questions:
            artifact = session.answer(question, cancellation=cancellation)
            lanes = [
                completion["language_lane"]
                for completion in artifact["completions"]
                if completion.get("language_lane")
            ]
            results.append(
                {
                    "request_id": artifact["request_id"],
                    "question_sha256": canonical_hash({"question": question}),
                    "passed": artifact["passed"],
                    "outcome": artifact["outcome"],
                    "answerable": artifact["response"]["answerable"],
                    "artifact_sha256": artifact["artifact_hash"],
                    "transaction_id": artifact["transaction"]["transaction_id"],
                    "output_dir": str(
                        output_root / f"request-{len(results) + 1:04d}-"
                        f"{hashlib.sha256(question.encode('utf-8')).hexdigest()[:12]}"
                    ),
                    "lane_process_generations": [
                        item["process_generation"] for item in lanes
                    ],
                    "lane_model_load_counts": [
                        item["model_load_count"] for item in lanes
                    ],
                }
            )
    except ProviderCancelledError as error:
        parser.exit(130, f"Atom harness session cancelled: {error}\n")
    finally:
        session.close()

    session_manifest = dict(session.manifest())
    report_core = {
        "schema": 1,
        "runtime": ATOM_HARNESS_SESSION_REPORT_RUNTIME,
        "passed": all(item["passed"] for item in results),
        "request_count": len(results),
        "session": session_manifest,
        "results": results,
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    report_path = output_root / "atom_harness_session.json"
    _atomic_report(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "request_count": report["request_count"],
                "model_load_count": session_manifest["providers"][0]["lane"][
                    "model_load_count"
                ],
                "restart_count": session_manifest["providers"][0]["lane"][
                    "restart_count"
                ],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
