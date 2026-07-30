"""Atom-first learning with Rust execution and thin Svelte projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash
from atom_coding_harness import (
    CODE_CAUSAL_MEMORY_RUNTIME,
    CodeCausalGraph,
    baseline_primitives,
    synthesize_with_atom,
)
from atom_coding_knowledge import (
    CODING_RAG_RUNTIME,
    CODING_WIKI_RUNTIME,
    CodingWikiGraph,
    retrieve_coding_context,
)
from atom_frontend_target import (
    ATOM_FRONTEND_TARGET_RUNTIME,
    compile_atom_to_frontend,
    validate_frontend_artifacts,
    write_frontend_project,
)
from atom_language import (
    ATOM_IR_RUNTIME,
    ATOM_LANGUAGE_RUNTIME,
    ATOM_TRAINING_RUNTIME,
    AtomProgram,
    AtomProgramEvaluator,
    compile_atom_blueprint,
    parse_atom_source,
    train_atom_causal_graph,
)
from atom_native_side_view import (
    ATOM_NATIVE_SIDE_VIEW_RUNTIME,
    render_atom_native_artifact,
)
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    PlatformSpec,
    platform_curriculum,
)
from atom_rust_target import (
    ATOM_RUST_TARGET_RUNTIME,
    RustPlatformEvaluator,
    cargo_validate_project,
    compile_atom_to_rust,
    write_rust_project,
)


ATOM_NATIVE_EXPERIMENT_SCHEMA = 1
ATOM_NATIVE_EXPERIMENT_RUNTIME = "atom-native-language-experiment-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _learned_candidate(
    graph: CodeCausalGraph,
    spec: PlatformSpec,
    *,
    label: str,
) -> tuple[AtomProgram, str]:
    bindings = graph.recognize(spec.capabilities)
    selected = synthesize_with_atom(graph, spec)
    if bindings is None or selected is None:
        raise RuntimeError(f"Atom graph abstained on covered spec {spec.spec_id}")
    _, program, source = compile_atom_blueprint(
        spec,
        selected,
        label=label,
        learned_bindings=bindings,
    )
    return program, source


def _atom_record(
    evaluator: AtomProgramEvaluator,
    program: AtomProgram,
    source: str,
    spec: PlatformSpec,
) -> dict[str, Any]:
    evaluation = evaluator.evaluate(program, spec)
    return {
        "spec_id": spec.spec_id,
        "program_hash": program.manifest()["program_hash"],
        "source_hash": canonical_hash({"source": source}),
        "primitive_count": len(program.primitives),
        "evaluation": evaluation,
    }


def _aggregate_atom(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capabilities = [
        item
        for record in records
        for item in record["evaluation"]["capabilities"]
    ]
    return {
        "spec_count": len(records),
        "full_passes": sum(
            bool(record["evaluation"]["passed"]) for record in records
        ),
        "capability_count": len(capabilities),
        "capability_passes": sum(bool(item["passed"]) for item in capabilities),
        "capability_score": (
            sum(bool(item["passed"]) for item in capabilities) / len(capabilities)
        ),
    }


def _build_request(spec: PlatformSpec) -> dict[str, Any]:
    core = {
        "schema": ATOM_NATIVE_EXPERIMENT_SCHEMA,
        "runtime": ATOM_NATIVE_EXPERIMENT_RUNTIME,
        "request_id": "atom-native-spiderweb-platform",
        "description": spec.description,
        "capabilities": list(spec.capabilities),
    }
    return {**core, "request_hash": canonical_hash(core)}


def run_atom_native_experiment(
    output_dir: Path,
    *,
    validator_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validator_dir = (
        Path(validator_dir)
        if validator_dir is not None
        else Path(__file__).resolve().parent / "tooling" / "svelte-validator"
    )
    curriculum = platform_curriculum()
    training = tuple(spec for spec in curriculum if spec.partition == "training")
    benchmark_specs = tuple(
        spec for spec in curriculum if spec.partition in {"validation", "heldout"}
    )
    graph, evidence, atom_evaluator = train_atom_causal_graph(training)
    model = graph.model_payload()
    restored = CodeCausalGraph.from_model_payload(model)

    atom_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    rust_records: list[dict[str, Any]] = []
    rust_evaluator = RustPlatformEvaluator()
    for spec in benchmark_specs:
        program, source = _learned_candidate(
            restored,
            spec,
            label=f"atom-native:{spec.spec_id}",
        )
        atom_records.append(_atom_record(atom_evaluator, program, source, spec))
        rust_records.append(rust_evaluator.evaluate(program, spec))
        _, baseline_program, baseline_source = compile_atom_blueprint(
            spec,
            baseline_primitives(),
            label=f"atom-baseline:{spec.spec_id}",
        )
        baseline_records.append(
            _atom_record(
                atom_evaluator,
                baseline_program,
                baseline_source,
                spec,
            )
        )

    atom_summary = _aggregate_atom(atom_records)
    baseline_summary = _aggregate_atom(baseline_records)
    full_spec = next(
        spec for spec in benchmark_specs if spec.spec_id == "heldout-spiderweb-platform"
    )
    full_program, atom_source = _learned_candidate(
        restored,
        full_spec,
        label="atom-native:live",
    )
    rust_source = compile_atom_to_rust(full_program)
    frontend_files = compile_atom_to_frontend(full_program)
    frontend_validation = validate_frontend_artifacts(
        frontend_files,
        validator_dir,
    )

    rust_dir = output_dir / "atom_generated_rust_platform"
    rust_project = write_rust_project(full_program, rust_dir)
    cargo_validation = cargo_validate_project(rust_dir)
    frontend_dir = output_dir / "atom_generated_frontend"
    frontend_project = write_frontend_project(full_program, frontend_dir)
    frontend_component = frontend_files["src/AtomPlatform.svelte"]

    knowledge = CodingWikiGraph()
    request = _build_request(full_spec)
    context = retrieve_coding_context(
        knowledge,
        request["description"]
        + " Atom Rust Svelte typed causal platform",
    )
    artifact_hashes = {
        "atom": canonical_hash({"source": atom_source}),
        "rust": canonical_hash({"source": rust_source}),
        "frontend": canonical_hash({"source": frontend_component}),
    }
    workflow_core = {
        "schema": ATOM_NATIVE_EXPERIMENT_SCHEMA,
        "runtime": {
            "experiment": ATOM_NATIVE_EXPERIMENT_RUNTIME,
            "atom_language": ATOM_LANGUAGE_RUNTIME,
            "atom_ir": ATOM_IR_RUNTIME,
            "atom_training": ATOM_TRAINING_RUNTIME,
            "causal_memory": CODE_CAUSAL_MEMORY_RUNTIME,
            "rust_target": ATOM_RUST_TARGET_RUNTIME,
            "frontend_target": ATOM_FRONTEND_TARGET_RUNTIME,
            "wiki": CODING_WIKI_RUNTIME,
            "rag": CODING_RAG_RUNTIME,
        },
        "request_hash": request["request_hash"],
        "model_hash": model["model_hash"],
        "claim_status": "derived",
        "program_hash": full_program.manifest()["program_hash"],
        "artifact_hashes": artifact_hashes,
        "knowledge_hash": knowledge.manifest()["knowledge_hash"],
        "knowledge_context": context,
        "rust_evaluation": rust_evaluator.evaluate(full_program, full_spec),
        "frontend_validation": frontend_validation,
    }

    recognized_pairs = {
        (law["requirement"], law["primitive"])
        for law in model["laws"]
        if law["status"] == "crystallized"
    }
    checks = {
        "training_is_atom_native": all(
            "substrate:atom" in context_name
            for law in model["laws"]
            for context_name in law["contexts"]
        ),
        "all_capability_bindings_are_learned": recognized_pairs.issuperset(
            set(PLATFORM_CAPABILITY_PRIMITIVES.items())
        ),
        "atom_passes_every_unseen_spec": (
            atom_summary["full_passes"] == len(benchmark_specs)
        ),
        "atom_beats_fixed_baseline": (
            atom_summary["capability_score"]
            > baseline_summary["capability_score"]
        ),
        "rust_projection_passes_every_unseen_spec": all(
            item["passed"] for item in rust_records
        ),
        "atom_source_round_trips": (
            parse_atom_source(atom_source) == full_program
        ),
        "cargo_project_builds_without_warnings": cargo_validation["passed"],
        "frontend_compiles_without_warnings": (
            frontend_validation["passed"]
            and frontend_validation["result"]["warningCount"] == 0
        ),
        "frontend_and_rust_share_atom_hash": (
            full_program.manifest()["program_hash"] in rust_source
            and full_program.manifest()["program_hash"]
            in frontend_files["src/atom-platform.ts"]
        ),
        "unknown_capability_abstains": (
            restored.recognize(("never_experienced",)) is None
        ),
        "seven_roots_remain_the_only_substrate": (
            full_program.roots == ROOT_MECHANICS
        ),
        "wiki_rag_are_runtime_wired": bool(context),
    }
    benchmark = {
        "partitions": ["validation", "heldout"],
        "spec_count": len(benchmark_specs),
        "atom": atom_summary,
        "baseline": baseline_summary,
        "atom_capability_score": atom_summary["capability_score"],
        "baseline_capability_score": baseline_summary["capability_score"],
        "improvement": (
            atom_summary["capability_score"]
            - baseline_summary["capability_score"]
        ),
        "rust_full_passes": sum(item["passed"] for item in rust_records),
        "rust_records": rust_records,
        "atom_records": atom_records,
        "baseline_records": baseline_records,
    }
    report_core = {
        "schema": ATOM_NATIVE_EXPERIMENT_SCHEMA,
        "runtime": ATOM_NATIVE_EXPERIMENT_RUNTIME,
        "model_hash": model["model_hash"],
        "program": full_program.manifest(),
        "learning": {
            "runtime": ATOM_TRAINING_RUNTIME,
            "interventions": len(evidence),
            "crystallized_laws": sum(
                law["status"] == "crystallized" for law in model["laws"]
            ),
        },
        "benchmark": benchmark,
        "rust_project": rust_project,
        "cargo_validation": cargo_validation,
        "frontend_project": frontend_project,
        "frontend_validation": frontend_validation,
        "checks": checks,
        "passed": all(checks.values()),
        "side_view_contract": {
            "runtime": ATOM_NATIVE_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": "render_atom_native_artifact",
            "placement": "side",
            "user_visible": True,
        },
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    workflow = {
        **workflow_core,
        "report_hash": report["report_hash"],
    }
    workflow["response_hash"] = canonical_hash(workflow)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Atom native experiment failed gates: {failed}")

    side_view = render_atom_native_artifact(
        model,
        report,
        workflow,
        atom_source,
        rust_source,
        frontend_component,
    )
    _write_json(output_dir / "atom_native_model.json", model)
    _write_json(output_dir / "atom_native_report.json", report)
    _write_json(output_dir / "atom_native_workflow_request.json", request)
    _write_json(output_dir / "atom_native_workflow_response.json", workflow)
    (output_dir / "atom_generated_platform.atom").write_text(
        atom_source,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_native_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("atom_native_outputs"),
    )
    args = parser.parse_args()
    report = run_atom_native_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
