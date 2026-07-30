"""End-to-end causal coding experiment for mathematical platform synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import canonical_hash
from atom_coding_harness import (
    CODE_CAUSAL_MEMORY_RUNTIME,
    CODE_GRAPH_RUNTIME,
    CODE_INTERVENTION_RUNTIME,
    CODE_MOLECULAR_RECOGNITION_RUNTIME,
    CODE_PHASE_LOCKED_LOOP_RUNTIME,
    CODE_PHASE_MIXER_RUNTIME,
    CODE_PROJECTIVE_MEASUREMENT_RUNTIME,
    CODE_THERMAL_ANNEALING_RUNTIME,
    CODE_TOPOLOGICAL_PERSISTENCE_RUNTIME,
    CODING_HARNESS_RUNTIME,
    CodeCausalGraph,
    CodeStructureGraph,
    HiddenPlatformEvaluator,
    IsolatedPythonInterventionRunner,
    baseline_primitives,
    synthesize_with_atom,
    train_code_causal_graph,
)
from atom_coding_knowledge import (
    CODING_RAG_RUNTIME,
    CODING_WIKI_RUNTIME,
    CodingWikiGraph,
    retrieve_coding_context,
)
from atom_coding_side_view import (
    ATOM_CODING_SIDE_VIEW_RUNTIME,
    render_coding_artifact,
)
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    PLATFORM_SYNTHESIS_RUNTIME,
    SPIDERWEB_PLATFORM_RUNTIME,
    PlatformSpec,
    build_platform_blueprint,
    compile_platform_source,
    platform_curriculum,
    platform_primitive_manifest,
)


CODING_EXPERIMENT_SCHEMA = 1
CODING_EXPERIMENT_RUNTIME = "atom-causal-coding-experiment-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _evaluate_candidate(
    evaluator: HiddenPlatformEvaluator,
    spec: PlatformSpec,
    primitives: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    blueprint = build_platform_blueprint(
        spec,
        primitives,
        blueprint_label=label,
    )
    source = compile_platform_source(blueprint)
    evaluation = evaluator.evaluate(source, spec)
    code_graph = CodeStructureGraph.from_source(source)
    return {
        "spec_id": spec.spec_id,
        "primitives": list(blueprint.primitives),
        "blueprint": blueprint.manifest(),
        "source": source,
        "artifact_hash": canonical_hash({"source": source}),
        "code_graph": code_graph.manifest(),
        "evaluation": evaluation.manifest(),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("benchmark needs at least one record")
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


def _build_live_request(spec: PlatformSpec) -> dict[str, Any]:
    core = {
        "schema": CODING_EXPERIMENT_SCHEMA,
        "runtime": CODING_EXPERIMENT_RUNTIME,
        "request_id": "coding-live-spiderweb-platform",
        "description": spec.description,
        "capabilities": list(spec.capabilities),
    }
    return {**core, "request_hash": canonical_hash(core)}


def run_coding_workflow(
    model: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    evaluator: HiddenPlatformEvaluator | None = None,
) -> tuple[dict[str, Any], str]:
    expected = {
        "schema",
        "runtime",
        "request_id",
        "description",
        "capabilities",
        "request_hash",
    }
    if set(request) != expected:
        raise ValueError("coding workflow request fields are invalid")
    request_core = {
        key: request[key] for key in expected if key != "request_hash"
    }
    if request["request_hash"] != canonical_hash(request_core):
        raise ValueError("coding workflow request hash mismatch")
    if (
        request["schema"] != CODING_EXPERIMENT_SCHEMA
        or request["runtime"] != CODING_EXPERIMENT_RUNTIME
    ):
        raise ValueError("coding workflow request contract mismatch")
    capabilities = tuple(str(item) for item in request["capabilities"])
    spec = PlatformSpec(
        spec_id=str(request["request_id"]),
        capabilities=capabilities,
        partition="heldout",
        description=str(request["description"]),
    )
    spec.validate()
    graph = CodeCausalGraph.from_model_payload(model)
    selected = synthesize_with_atom(graph, spec)
    knowledge = CodingWikiGraph()
    query = f'{request["description"]} {" ".join(capabilities)}'
    context = retrieve_coding_context(knowledge, query)
    runtime = {
        "experiment": CODING_EXPERIMENT_RUNTIME,
        "harness": CODING_HARNESS_RUNTIME,
        "causal_graph": CODE_CAUSAL_MEMORY_RUNTIME,
        "code_graph": CODE_GRAPH_RUNTIME,
        "intervention": CODE_INTERVENTION_RUNTIME,
        "molecular_recognition": CODE_MOLECULAR_RECOGNITION_RUNTIME,
        "phase_locked_loop": CODE_PHASE_LOCKED_LOOP_RUNTIME,
        "phase_mixer": CODE_PHASE_MIXER_RUNTIME,
        "topological_persistence": CODE_TOPOLOGICAL_PERSISTENCE_RUNTIME,
        "thermal_annealing": CODE_THERMAL_ANNEALING_RUNTIME,
        "projective_measurement": CODE_PROJECTIVE_MEASUREMENT_RUNTIME,
        "platform_synthesis": PLATFORM_SYNTHESIS_RUNTIME,
        "platform": SPIDERWEB_PLATFORM_RUNTIME,
        "wiki": CODING_WIKI_RUNTIME,
        "rag": CODING_RAG_RUNTIME,
    }
    if selected is None:
        core = {
            "schema": CODING_EXPERIMENT_SCHEMA,
            "runtime": runtime,
            "request_hash": request["request_hash"],
            "model_hash": model["model_hash"],
            "claim_status": "unknown",
            "reason": "no crystallized causal path covers every requested capability",
            "knowledge_context": context,
            "knowledge_hash": knowledge.manifest()["knowledge_hash"],
            "primitives": [],
            "blueprint": None,
            "artifact_hash": None,
            "code_graph": None,
            "evaluation": None,
        }
        return {**core, "response_hash": canonical_hash(core)}, ""
    candidate = _evaluate_candidate(
        evaluator or HiddenPlatformEvaluator(IsolatedPythonInterventionRunner()),
        spec,
        selected,
        label=f"live:{request['request_id']}",
    )
    core = {
        "schema": CODING_EXPERIMENT_SCHEMA,
        "runtime": runtime,
        "request_hash": request["request_hash"],
        "model_hash": model["model_hash"],
        "claim_status": (
            "derived" if candidate["evaluation"]["passed"] else "contradicted"
        ),
        "reason": "selected from crystallized intervention laws",
        "knowledge_context": context,
        "knowledge_hash": knowledge.manifest()["knowledge_hash"],
        "primitives": candidate["primitives"],
        "blueprint": candidate["blueprint"],
        "artifact_hash": candidate["artifact_hash"],
        "code_graph": candidate["code_graph"],
        "evaluation": candidate["evaluation"],
    }
    return {**core, "response_hash": canonical_hash(core)}, candidate["source"]


def run_coding_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum = platform_curriculum()
    training = tuple(spec for spec in curriculum if spec.partition == "training")
    benchmark_specs = tuple(
        spec for spec in curriculum if spec.partition in {"validation", "heldout"}
    )
    runner = IsolatedPythonInterventionRunner()
    evaluator = HiddenPlatformEvaluator(runner)
    graph, evidence = train_code_causal_graph(training, evaluator)
    model = graph.model_payload()
    restored = CodeCausalGraph.from_model_payload(model)
    if restored.model_payload() != model:
        raise RuntimeError("code causal model did not survive persistence")

    atom_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    for spec in benchmark_specs:
        selected = synthesize_with_atom(restored, spec)
        if selected is None:
            raise RuntimeError(f"learned graph abstained on covered spec {spec.spec_id}")
        atom_records.append(
            _evaluate_candidate(
                evaluator,
                spec,
                selected,
                label=f"atom:{spec.spec_id}",
            )
        )
        baseline_records.append(
            _evaluate_candidate(
                evaluator,
                spec,
                baseline_primitives(),
                label=f"baseline:{spec.spec_id}",
            )
        )
    atom_summary = _aggregate(atom_records)
    baseline_summary = _aggregate(baseline_records)

    full_spec = next(
        spec for spec in benchmark_specs if spec.spec_id == "heldout-spiderweb-platform"
    )
    full_selected = synthesize_with_atom(restored, full_spec)
    if full_selected is None:
        raise RuntimeError("learned graph did not recognize the full platform")
    no_phase = _evaluate_candidate(
        evaluator,
        full_spec,
        full_selected[:1],
        label="ablation:no-phase-mixing",
    )
    no_memory = _evaluate_candidate(
        evaluator,
        full_spec,
        baseline_primitives(),
        label="ablation:no-causal-memory",
    )
    uncrystallized = CodeCausalGraph.from_model_payload(model)
    for law in uncrystallized.laws.values():
        law.status = "hypothesis"
    no_persistence = synthesize_with_atom(uncrystallized, full_spec)
    unknown_capability = restored.recognize(("capability_never_experienced",))

    request = _build_live_request(full_spec)
    workflow, artifact_source = run_coding_workflow(
        model,
        request,
        evaluator=evaluator,
    )
    if workflow["claim_status"] != "derived":
        raise RuntimeError("live coding workflow did not derive the platform")

    benchmark = {
        "partitions": ["validation", "heldout"],
        "hidden_truth_hash": evaluator.truth_hash,
        "baseline": baseline_summary,
        "atom": atom_summary,
        "baseline_capability_score": baseline_summary["capability_score"],
        "atom_capability_score": atom_summary["capability_score"],
        "improvement": (
            atom_summary["capability_score"]
            - baseline_summary["capability_score"]
        ),
        "atom_full_passes": atom_summary["full_passes"],
        "baseline_full_passes": baseline_summary["full_passes"],
        "records": {
            "baseline": [
                {key: value for key, value in record.items() if key != "source"}
                for record in baseline_records
            ],
            "atom": [
                {key: value for key, value in record.items() if key != "source"}
                for record in atom_records
            ],
        },
    }
    ablations = {
        "no_causal_memory_score": no_memory["evaluation"]["score"],
        "no_phase_mixing_score": no_phase["evaluation"]["score"],
        "no_topological_persistence_abstained": no_persistence is None,
        "unexperienced_capability_abstained": unknown_capability is None,
    }
    checks = {
        "atom_beats_baseline": (
            atom_summary["capability_score"]
            > baseline_summary["capability_score"]
        ),
        "atom_passes_every_benchmark_spec": (
            atom_summary["full_passes"] == len(benchmark_specs)
        ),
        "all_nine_capability_bindings_crystallized": {
            (law["requirement"], law["primitive"])
            for law in model["laws"]
            if law["status"] == "crystallized"
        }.issuperset(set(PLATFORM_CAPABILITY_PRIMITIVES.items())),
        "model_persistence_exact": restored.model_payload() == model,
        "unknown_is_not_guessed": unknown_capability is None,
        "persistence_is_causally_required": no_persistence is None,
        "phase_mixing_is_causally_required": (
            no_phase["evaluation"]["score"]
            < atom_summary["capability_score"]
        ),
        "live_artifact_passed_hidden_evaluation": workflow["evaluation"]["passed"],
        "wiki_rag_runtime_wired": (
            workflow["runtime"]["wiki"] == CODING_WIKI_RUNTIME
            and workflow["runtime"]["rag"] == CODING_RAG_RUNTIME
            and bool(workflow["knowledge_context"])
        ),
    }
    report_core = {
        "schema": CODING_EXPERIMENT_SCHEMA,
        "runtime": CODING_EXPERIMENT_RUNTIME,
        "model_hash": model["model_hash"],
        "platform_registry": platform_primitive_manifest(),
        "curriculum": {
            "training_specs": len(training),
            "validation_specs": sum(
                spec.partition == "validation" for spec in benchmark_specs
            ),
            "heldout_specs": sum(
                spec.partition == "heldout" for spec in benchmark_specs
            ),
            "training_interventions": len(evidence),
        },
        "benchmark": benchmark,
        "ablations": ablations,
        "checks": checks,
        "passed": all(checks.values()),
        "side_view_contract": {
            "runtime": ATOM_CODING_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": "render_coding_artifact",
            "placement": "side",
            "user_visible": True,
        },
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    workflow = {**workflow, "report_hash": report["report_hash"]}
    workflow_core = {
        key: value for key, value in workflow.items() if key != "response_hash"
    }
    workflow["response_hash"] = canonical_hash(workflow_core)

    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"coding experiment failed gates: {failed}")

    side_view = render_coding_artifact(
        model,
        report,
        workflow,
        artifact_source,
    )
    _write_json(output_dir / "atom_coding_model.json", model)
    _write_json(output_dir / "atom_coding_report.json", report)
    _write_json(output_dir / "atom_coding_workflow_request.json", request)
    _write_json(output_dir / "atom_coding_workflow_response.json", workflow)
    (output_dir / "atom_generated_platform.py").write_text(
        artifact_source,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_coding_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Atom causal coding and platform-synthesis experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coding_harness_outputs"),
    )
    args = parser.parse_args()
    report = run_coding_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
