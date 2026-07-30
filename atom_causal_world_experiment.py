"""Integrated executable causal-world experiment and runtime entrypoint."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from atom_causal_graph import (
    CAUSAL_GRAPH_RUNTIME,
    ActiveExperimentScheduler,
    CausalCognition,
    CausalGraph,
    causal_graph_self_test,
    law_condition_signature,
    stable_condition_signature,
)
from atom_formal_domains import (
    FORMAL_CURRICULUM_RUNTIME,
    FORMAL_DOMAIN_RUNTIME,
    FORMAL_TRUTH_ORACLE_RUNTIME,
    formal_domain_self_test,
    run_formal_domain_benchmark,
)
from atom_causal_world_accelerator import (
    CAUSAL_ACCELERATOR_RUNTIME,
    accelerator_self_test,
    build_accelerator_plan,
    probe_jax_accelerator,
    run_jax_massive_shard,
)
from atom_causal_world_knowledge import (
    CAUSAL_WORLD_RAG_RUNTIME,
    CAUSAL_WORLD_WIKI_RUNTIME,
    CausalWorldWikiGraph,
    causal_world_knowledge_self_test,
    retrieve_causal_context,
)
from atom_causal_world_curriculum import (
    TPU_PROGRAMS_PER_SHARD,
    causal_world_curriculum_self_test,
    curriculum_manifest,
)
from atom_causal_world_language import (
    CAUSAL_WORLD_LANGUAGE_RUNTIME,
    causal_world_language_self_test,
    language_space_manifest,
    parse_causal_question,
    render_causal_answer,
    render_causal_question,
)
from atom_causal_world_schema import (
    ARCHITECTURE_COMPONENTS,
    CAUSAL_WORLD_RUNTIME,
    DOMAIN_NAMES,
    ROOT_MECHANICS,
    CausalEvidence,
    CausalWorldConfig,
    canonical_hash,
    causal_world_schema_self_test,
    config_manifest,
    get_profile,
)
from atom_causal_world_side_view import (
    ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME,
    render_causal_world_artifact,
)
from atom_causal_world_simulator import (
    CAUSAL_WORLD_SIMULATOR_RUNTIME,
    ProceduralWorldCompiler,
    causal_world_simulator_self_test,
    generate_interventions,
    rollout_counterfactual_pair,
    summarize_world,
)
from atom_causal_world_transfer import (
    CAUSAL_TRANSFER_RUNTIME,
    causal_transfer_self_test,
    run_causal_transfer_benchmark,
)


CAUSAL_WORLD_EXPERIMENT_RUNTIME = "atom-causal-world-experiment-v2"
CAUSAL_WORLD_EXPERIMENT_SCHEMA = 1
CAUSAL_WORLD_RESUME_RUNTIME = "atom-causal-world-resume-v1"
CAUSAL_WORLD_RESUME_SCHEMA = 1


def _write_causal_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_causal_json(path: Path, value: Any) -> None:
    _write_causal_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_causal_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_causal_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read causal state: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"causal state must be an object: {path.name}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_causal_resume_cursor(
    config: CausalWorldConfig,
    graph: CausalGraph,
    *,
    completed_shards: Sequence[int],
    shard_evidence_hashes: Sequence[str],
    model_lineage: Sequence[str],
) -> dict[str, Any]:
    """Bind a cumulative graph to the exact ordered accelerator shard lineage."""

    config.validate()
    ordered_shards = [int(value) for value in completed_shards]
    evidence_hashes = [str(value) for value in shard_evidence_hashes]
    model_hashes = [str(value) for value in model_lineage]
    expected_shards = list(range(len(ordered_shards)))
    if ordered_shards != expected_shards:
        raise ValueError("completed causal shards must be contiguous from zero")
    if len(ordered_shards) > config.shard_count:
        raise ValueError("completed causal shards exceed the configured world")
    if not ordered_shards:
        raise ValueError("resume state requires at least one completed shard")
    if len(evidence_hashes) != len(ordered_shards):
        raise ValueError("each completed shard requires an evidence hash")
    if len(model_hashes) != len(ordered_shards):
        raise ValueError("each completed shard requires a model hash")
    if not all(_is_sha256(value) for value in evidence_hashes + model_hashes):
        raise ValueError("resume state contains an invalid content hash")

    model = graph.model_payload()
    if model_hashes[-1] != model["model_hash"]:
        raise ValueError("resume lineage does not end at the current graph")
    lineage_core = {
        "profile": config.profile,
        "shard_count": config.shard_count,
        "completed_shards": ordered_shards,
        "shard_evidence_hashes": evidence_hashes,
        "model_lineage": model_hashes,
    }
    core = {
        "schema": CAUSAL_WORLD_RESUME_SCHEMA,
        "runtime": CAUSAL_WORLD_RESUME_RUNTIME,
        "backend": "jax-xla",
        "resumable": True,
        "profile": config.profile,
        "completed_shards": ordered_shards,
        "next_shard": len(ordered_shards),
        "shard_count": config.shard_count,
        "model_hash": model["model_hash"],
        "model_lineage": model_hashes,
        "shard_evidence_hashes": evidence_hashes,
        "cumulative_evidence_count": graph.observation_count,
        "lineage_hash": canonical_hash(lineage_core),
    }
    return {**core, "cursor_hash": canonical_hash(core)}


def validate_causal_resume_cursor(
    cursor: Mapping[str, Any],
    model: Mapping[str, Any],
    config: CausalWorldConfig,
    *,
    expected_next_shard: int,
) -> tuple[CausalGraph, dict[str, Any]]:
    """Validate every model and cursor binding before cumulative learning resumes."""

    fields = {
        "backend",
        "completed_shards",
        "cumulative_evidence_count",
        "cursor_hash",
        "lineage_hash",
        "model_hash",
        "model_lineage",
        "next_shard",
        "profile",
        "resumable",
        "runtime",
        "schema",
        "shard_count",
        "shard_evidence_hashes",
    }
    if set(cursor) != fields:
        raise ValueError("causal resume cursor fields are invalid")
    core = {key: cursor[key] for key in sorted(fields - {"cursor_hash"})}
    if cursor["cursor_hash"] != canonical_hash(core):
        raise ValueError("causal resume cursor hash mismatch")
    if cursor["schema"] != CAUSAL_WORLD_RESUME_SCHEMA:
        raise ValueError("unsupported causal resume schema")
    if cursor["runtime"] != CAUSAL_WORLD_RESUME_RUNTIME:
        raise ValueError("unsupported causal resume runtime")
    if cursor["backend"] != "jax-xla" or cursor["resumable"] is not True:
        raise ValueError("causal resume cursor is not accelerator-resumable")
    if cursor["profile"] != config.profile:
        raise ValueError("causal resume profile mismatch")
    if cursor["shard_count"] != config.shard_count:
        raise ValueError("causal resume shard-count mismatch")
    if (
        isinstance(cursor["next_shard"], bool)
        or not isinstance(cursor["next_shard"], int)
        or cursor["next_shard"] != expected_next_shard
    ):
        raise ValueError("causal resume shard position mismatch")
    completed = cursor["completed_shards"]
    evidence_hashes = cursor["shard_evidence_hashes"]
    model_lineage = cursor["model_lineage"]
    if not isinstance(completed, list) or completed != list(range(expected_next_shard)):
        raise ValueError("causal resume shard lineage is not contiguous")
    if not isinstance(evidence_hashes, list) or len(evidence_hashes) != len(completed):
        raise ValueError("causal resume evidence lineage is invalid")
    if not isinstance(model_lineage, list) or len(model_lineage) != len(completed):
        raise ValueError("causal resume model lineage is invalid")
    if not all(_is_sha256(value) for value in evidence_hashes + model_lineage):
        raise ValueError("causal resume lineage contains an invalid hash")
    if model.get("model_hash") != cursor["model_hash"]:
        raise ValueError("causal resume model binding mismatch")
    if not model_lineage or model_lineage[-1] != cursor["model_hash"]:
        raise ValueError("causal resume lineage does not end at the bound model")
    lineage_core = {
        "profile": cursor["profile"],
        "shard_count": cursor["shard_count"],
        "completed_shards": completed,
        "shard_evidence_hashes": evidence_hashes,
        "model_lineage": model_lineage,
    }
    if cursor["lineage_hash"] != canonical_hash(lineage_core):
        raise ValueError("causal resume lineage hash mismatch")
    graph = CausalGraph.from_model_payload(model)
    if graph.maximum_laws != config.maximum_laws:
        raise ValueError("causal resume graph capacity mismatch")
    if graph.observation_count != cursor["cumulative_evidence_count"]:
        raise ValueError("causal resume evidence count mismatch")
    if graph.model_payload() != dict(model):
        raise ValueError("causal resume graph round trip mismatch")
    return graph, dict(cursor)


def load_causal_resume_state(
    resume_from: Path,
    config: CausalWorldConfig,
    *,
    expected_next_shard: int,
) -> tuple[CausalGraph, dict[str, Any]]:
    state_dir = resume_from.resolve()
    if not state_dir.is_dir():
        raise ValueError("causal resume path must be an existing output directory")
    model = _load_causal_json(state_dir / "atom_causal_world_model.json")
    cursor = _load_causal_json(state_dir / "atom_causal_world_resume_cursor.json")
    return validate_causal_resume_cursor(
        cursor,
        model,
        config,
        expected_next_shard=expected_next_shard,
    )


def _module_avoids_mamba() -> bool:
    source_files = (
        Path(__file__).with_name("atom_causal_world_schema.py"),
        Path(__file__).with_name("atom_causal_world_simulator.py"),
        Path(__file__).with_name("atom_causal_graph.py"),
        Path(__file__).with_name("atom_causal_world_language.py"),
        Path(__file__).with_name("atom_causal_world_knowledge.py"),
        Path(__file__).with_name("atom_causal_world_accelerator.py"),
        Path(__file__),
    )
    audited_paths: set[Path] = set()
    for path in source_files:
        path = path.resolve()
        if path in audited_paths or not path.is_file():
            continue
        audited_paths.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0].lower() for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {str(node.module).split(".", 1)[0].lower()}
            else:
                continue
            if "mamba" in names or "mamba_ssm" in names:
                return False
    return bool(audited_paths)


def _run_numpy_learning(
    config: CausalWorldConfig,
) -> tuple[CausalGraph, list[CausalEvidence], dict[str, Any]]:
    compiler = ProceduralWorldCompiler(config)
    interventions = generate_interventions(config.intervention_candidates)
    graph = CausalGraph(maximum_laws=config.maximum_laws)
    cognition = CausalCognition(graph)
    scheduler = ActiveExperimentScheduler()
    evidence_rows: list[CausalEvidence] = []
    shard_reports: list[dict[str, Any]] = []
    shards_to_run = min(3, config.shard_count)
    anchor_count = min(max(4, config.active_experiments // 3), len(interventions))
    anchors = list(interventions[:anchor_count])
    for shard_index in range(shards_to_run):
        world = compiler.compile_shard(shard_index)
        ranked = scheduler.rank(graph, interventions)
        selected = list(anchors)
        selected_ids = {item.intervention_id for item in selected}
        for intervention, _ in ranked:
            if intervention.intervention_id in selected_ids:
                continue
            selected.append(intervention)
            selected_ids.add(intervention.intervention_id)
            if len(selected) >= config.active_experiments:
                break
        shard_evidence: list[CausalEvidence] = []
        rollout_reports: list[dict[str, Any]] = []
        for intervention in selected:
            evidence, diagnostics = rollout_counterfactual_pair(
                world,
                intervention,
                config.time_steps,
            )
            shard_evidence.extend(evidence)
            rollout_reports.append(diagnostics)
        learning = cognition.learn(shard_evidence)
        evidence_rows.extend(shard_evidence)
        shard_reports.append(
            {
                "shard_index": shard_index,
                "world": summarize_world(world),
                "selected_interventions": [
                    asdict(intervention) for intervention in selected
                ],
                "selection_scores": {
                    intervention.intervention_id: score
                    for intervention, score in ranked
                    if intervention.intervention_id in selected_ids
                },
                "anchor_interventions": [item.intervention_id for item in anchors],
                "rollout_count": len(rollout_reports),
                "maximum_invariant_error": max(
                    report["maximum_invariant_error"] for report in rollout_reports
                ),
                "effect_trace_hash": canonical_hash(
                    [report["effect_trace_digest"] for report in rollout_reports]
                ),
                "learning": learning,
            }
        )
    return (
        graph,
        evidence_rows,
        {
            "backend": "numpy",
            "runtime": CAUSAL_WORLD_SIMULATOR_RUNTIME,
            "shards_executed": shards_to_run,
            "shards": shard_reports,
            "maximum_invariant_error": max(
                report["maximum_invariant_error"] for report in shard_reports
            ),
        },
    )


def _run_accelerator_learning(
    config: CausalWorldConfig,
    shard_index: int,
    *,
    require_tpu: bool,
    require_gpu: bool,
    shards_per_run: int,
    initial_graph: CausalGraph | None = None,
    prior_cursor: Mapping[str, Any] | None = None,
    state_writer: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
) -> tuple[CausalGraph, list[CausalEvidence], dict[str, Any], dict[str, Any]]:
    if isinstance(shards_per_run, bool) or not isinstance(shards_per_run, int):
        raise TypeError("shards per run must be an integer")
    if shards_per_run == 0:
        shards_per_run = config.shard_count - shard_index
    if shards_per_run <= 0:
        raise ValueError("shards per run must be positive")
    shard_stop = shard_index + shards_per_run
    if not 0 <= shard_index < shard_stop <= config.shard_count:
        raise ValueError("accelerator shard range is outside the configured world")
    if initial_graph is None:
        if shard_index != 0 or prior_cursor is not None:
            raise ValueError(
                "nonzero accelerator shards require validated resume state"
            )
        graph = CausalGraph(maximum_laws=config.maximum_laws)
        completed_shards: list[int] = []
        shard_evidence_hashes: list[str] = []
        model_lineage: list[str] = []
    else:
        if prior_cursor is None:
            raise ValueError("a resumed graph requires its validated cursor")
        graph = initial_graph
        completed_shards = list(prior_cursor["completed_shards"])
        shard_evidence_hashes = list(prior_cursor["shard_evidence_hashes"])
        model_lineage = list(prior_cursor["model_lineage"])
        if completed_shards != list(range(shard_index)):
            raise ValueError("validated resume state does not precede requested shards")

    cognition = CausalCognition(graph)
    evidence_rows: list[CausalEvidence] = []
    shard_reports: list[dict[str, Any]] = []
    resume_cursor: dict[str, Any] | None = None
    for current_shard in range(shard_index, shard_stop):
        evidence, accelerator = run_jax_massive_shard(
            config,
            current_shard,
            require_tpu=require_tpu,
            require_gpu=require_gpu,
        )
        learning = cognition.learn(evidence)
        evidence_rows.extend(evidence)
        completed_shards.append(current_shard)
        shard_evidence_hashes.append(accelerator["evidence_hash"])
        model = graph.model_payload()
        model_lineage.append(model["model_hash"])
        resume_cursor = build_causal_resume_cursor(
            config,
            graph,
            completed_shards=completed_shards,
            shard_evidence_hashes=shard_evidence_hashes,
            model_lineage=model_lineage,
        )
        shard_reports.append(
            {
                "shard_index": current_shard,
                "accelerator": accelerator,
                "learning": learning,
                "cumulative_evidence_count": graph.observation_count,
                "model_hash": model["model_hash"],
                "cursor_hash": resume_cursor["cursor_hash"],
            }
        )
        if state_writer is not None:
            state_writer(model, resume_cursor)

    if resume_cursor is None or not shard_reports:
        raise RuntimeError("accelerator execution produced no shard state")
    accelerator_reports = [item["accelerator"] for item in shard_reports]
    deterministic_by_shard = {
        str(item["shard_index"]): item["accelerator"]["deterministic_replay"]
        for item in shard_reports
    }
    deterministic_replay = {
        "passed": all(value["passed"] for value in deterministic_by_shard.values()),
        "shards": deterministic_by_shard,
    }
    device_counts = {int(item["devices_used"]) for item in accelerator_reports}
    executor_modes = {str(item["executor_mode"]) for item in accelerator_reports}
    if len(device_counts) != 1 or len(executor_modes) != 1:
        raise RuntimeError("accelerator topology changed across cumulative shards")
    accelerator_summary = {
        "runtime": CAUSAL_ACCELERATOR_RUNTIME,
        "probe": accelerator_reports[0]["probe"],
        "plan": accelerator_reports[0]["plan"],
        "shard_start": shard_index,
        "shard_stop": shard_stop,
        "shards_executed": shards_per_run,
        "evidence_count": len(evidence_rows),
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in accelerator_reports),
        "entity_updates": sum(item["entity_updates"] for item in accelerator_reports),
        "relation_updates": sum(
            item["relation_updates"] for item in accelerator_reports
        ),
        "devices_used": next(iter(device_counts)),
        "executor_mode": next(iter(executor_modes)),
        "maximum_invariant_error": max(
            item["maximum_invariant_error"] for item in accelerator_reports
        ),
        "jit_executor_constructions": max(
            item["jit_executor_constructions"] for item in accelerator_reports
        ),
        "deterministic_replay": deterministic_replay,
        "evidence_hash": canonical_hash(
            [item["evidence_hash"] for item in accelerator_reports]
        ),
        "world_program_ids": sorted(
            {
                int(program["program_id"])
                for item in accelerator_reports
                for program in item["world_programs"]
            }
        ),
    }
    accelerator_summary["world_program_count"] = len(
        accelerator_summary["world_program_ids"]
    )
    return (
        graph,
        evidence_rows,
        {
            "backend": "jax-xla",
            "runtime": CAUSAL_ACCELERATOR_RUNTIME,
            "shards_executed": shards_per_run,
            "shard_start": shard_index,
            "shard_stop": shard_stop,
            "shards": shard_reports,
            "accelerator": accelerator_summary,
            "learning": {
                "current_run_evidence": len(evidence_rows),
                "cumulative_evidence": graph.observation_count,
                "graph_laws": len(graph.laws),
                "graph_nodes": len(graph.nodes),
            },
            "maximum_invariant_error": accelerator_summary["maximum_invariant_error"],
        },
        resume_cursor,
    )


def _top_supported_laws(graph: CausalGraph, limit: int = 6) -> list[Any]:
    laws = [law for law in graph.laws.values() if law.status == "crystallized"]
    laws.sort(
        key=lambda law: (
            -law.persistence,
            -law.confidence,
            -law.support,
            law.law_id,
        )
    )
    return laws[:limit]


def _build_supported_composed_turn(
    graph: CausalGraph, *, variant: int
) -> dict[str, Any] | None:
    cognition = CausalCognition(graph)
    laws = _top_supported_laws(graph, limit=min(128, len(graph.laws)))
    for first in laws:
        continuations = graph.candidate_laws(
            cause_feature=first.effect_feature,
            domain=first.domain,
            condition_signature=law_condition_signature(first),
        )
        for second in continuations:
            if second.status != "crystallized":
                continue
            if second.effect_feature == first.cause_feature:
                continue
            why_variant = variant + ((2 - variant) % 4)
            request = render_causal_question(
                query_id="composed-00",
                domain=first.domain,
                cause_feature=first.cause_feature,
                effect_feature=second.effect_feature,
                variant=why_variant,
                condition_signature=law_condition_signature(first),
            )
            artifact = cognition.answer(parse_causal_question(request))
            if artifact["claim_status"] == "derived" and artifact["path_length"] > 1:
                return request
    return None


def build_causal_workflow_request(
    graph: CausalGraph,
) -> tuple[dict[str, Any], dict[str, str]]:
    supported = _top_supported_laws(graph, limit=6)
    if len(supported) < 2:
        raise RuntimeError(
            "causal graph did not crystallize enough laws for a workflow"
        )
    turns: list[dict[str, Any]] = []
    truth: dict[str, str] = {}
    for index, law in enumerate(supported):
        request = render_causal_question(
            query_id=f"supported-{index:02d}",
            domain=law.domain,
            cause_feature=law.cause_feature,
            effect_feature=law.effect_feature,
            variant=index,
            condition_signature=law_condition_signature(law),
        )
        turns.append(request)
        truth[request["query_id"]] = "derived"
    composed_request = _build_supported_composed_turn(graph, variant=len(turns))
    if composed_request is not None:
        turns.append(composed_request)
        truth[composed_request["query_id"]] = "derived"
    unsupported_pairs = (
        ("language", "ownership", "mass"),
        ("chemical", "existence", "trust"),
    )
    unsupported_conditions = law_condition_signature(supported[0])
    for index, (domain, cause, effect) in enumerate(unsupported_pairs):
        request = render_causal_question(
            query_id=f"unsupported-{index:02d}",
            domain=domain,
            cause_feature=cause,
            effect_feature=effect,
            variant=index + len(turns),
            condition_signature=unsupported_conditions,
        )
        turns.append(request)
        truth[request["query_id"]] = "unknown"
    return (
        {
            "schema": CAUSAL_WORLD_EXPERIMENT_SCHEMA,
            "runtime": CAUSAL_WORLD_EXPERIMENT_RUNTIME,
            "turns": turns,
        },
        truth,
    )


def run_causal_workflow(
    model_payload: Mapping[str, Any],
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if set(request_payload) != {"runtime", "schema", "turns"}:
        raise ValueError("causal workflow request fields are invalid")
    if request_payload["schema"] != CAUSAL_WORLD_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported causal workflow schema")
    if request_payload["runtime"] != CAUSAL_WORLD_EXPERIMENT_RUNTIME:
        raise ValueError("causal workflow runtime marker mismatch")
    turns_payload = request_payload["turns"]
    if not isinstance(turns_payload, Sequence) or isinstance(
        turns_payload, (str, bytes)
    ):
        raise ValueError("causal workflow turns must be a sequence")
    graph = CausalGraph.from_model_payload(model_payload)
    cognition = CausalCognition(graph)
    wiki = CausalWorldWikiGraph(graph)
    turns: list[dict[str, Any]] = []
    for request in turns_payload:
        if not isinstance(request, Mapping):
            raise ValueError("causal workflow turn must be an object")
        query = parse_causal_question(request)
        context = retrieve_causal_context(
            wiki,
            str(request["text"]),
            limit=8,
            domain=query.domain,
            cause_feature=query.cause_feature,
            effect_feature=query.effect_feature,
        )
        artifact = cognition.answer(query)
        turns.append(
            {
                "request": dict(request),
                "artifact": artifact,
                "answer": render_causal_answer(artifact),
                "knowledge_context": context,
            }
        )
    core = {
        "schema": CAUSAL_WORLD_EXPERIMENT_SCHEMA,
        "runtime": {
            "experiment": CAUSAL_WORLD_EXPERIMENT_RUNTIME,
            "world": CAUSAL_WORLD_RUNTIME,
            "causal_graph": CAUSAL_GRAPH_RUNTIME,
            "language": CAUSAL_WORLD_LANGUAGE_RUNTIME,
            "wiki": CAUSAL_WORLD_WIKI_RUNTIME,
            "rag": CAUSAL_WORLD_RAG_RUNTIME,
            "side_view": ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME,
        },
        "model_hash": model_payload["model_hash"],
        "turns": turns,
    }
    return {**core, "workflow_hash": canonical_hash(core)}


def _workflow_evaluation(
    workflow: Mapping[str, Any], truth: Mapping[str, str]
) -> dict[str, Any]:
    decisions = {
        turn["request"]["query_id"]: turn["artifact"]["claim_status"]
        for turn in workflow["turns"]
    }
    correct = sum(
        decisions.get(query_id) == expected for query_id, expected in truth.items()
    )
    return {
        "turns": len(truth),
        "correct": correct,
        "accuracy": correct / max(len(truth), 1),
        "derived_answers": sum(value == "derived" for value in decisions.values()),
        "unknown_answers": sum(value == "unknown" for value in decisions.values()),
        "decisions": decisions,
        "expected": dict(truth),
    }


def _corruption_checks(model: Mapping[str, Any]) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    wrong_hash = copy.deepcopy(model)
    wrong_hash["model_hash"] = "0" * 64
    variants.append(wrong_hash)
    altered_law = copy.deepcopy(model)
    if altered_law["graph"]["laws"]:
        altered_law["graph"]["laws"][0]["confidence"] = 1.0
    variants.append(altered_law)
    wrong_architecture = copy.deepcopy(model)
    wrong_architecture["architecture"] = "sequence-model"
    variants.append(wrong_architecture)
    rejected = 0
    errors: list[str] = []
    for variant in variants:
        try:
            CausalGraph.from_model_payload(variant)
        except (TypeError, ValueError) as error:
            rejected += 1
            errors.append(type(error).__name__)
    return {
        "variants": len(variants),
        "rejected": rejected,
        "errors": errors,
        "passed": rejected == len(variants),
    }


def _deterministic_numpy_check(config: CausalWorldConfig) -> dict[str, Any]:
    compiler = ProceduralWorldCompiler(config)
    first = compiler.compile_shard(0)
    second = compiler.compile_shard(0)
    state_equal = bool((first.state == second.state).all())
    relation_equal = bool((first.relations == second.relations).all())
    intervention = generate_interventions(1)[0]
    first_evidence, first_report = rollout_counterfactual_pair(
        first, intervention, config.time_steps
    )
    second_evidence, second_report = rollout_counterfactual_pair(
        second, intervention, config.time_steps
    )
    evidence_equal = [asdict(item) for item in first_evidence] == [
        asdict(item) for item in second_evidence
    ]
    return {
        "state_equal": state_equal,
        "relations_equal": relation_equal,
        "evidence_equal": evidence_equal,
        "effect_trace_equal": first_report["effect_trace_digest"]
        == second_report["effect_trace_digest"],
        "passed": state_equal
        and relation_equal
        and evidence_equal
        and first_report["effect_trace_digest"] == second_report["effect_trace_digest"],
    }


def run_causal_world_experiment(
    output_dir: Path,
    *,
    profile: str = "local",
    backend: str = "auto",
    shard_index: int = 0,
    shards_per_run: int = 1,
    resume_from: Path | None = None,
    require_tpu: bool = False,
    require_gpu: bool = False,
) -> dict[str, Any]:
    config = get_profile(profile)
    if backend not in {"auto", "numpy", "jax-xla"}:
        raise ValueError("backend must be auto, numpy, or jax-xla")
    selected_backend = (
        "jax-xla" if backend == "auto" and profile == "tpu-massive" else backend
    )
    if selected_backend == "auto":
        selected_backend = "numpy"
    if require_tpu and require_gpu:
        raise ValueError("an experiment cannot require both TPU and GPU")
    if (require_tpu or require_gpu) and selected_backend != "jax-xla":
        raise ValueError("an accelerator requirement needs the jax-xla backend")
    if profile == "tpu-massive" and selected_backend != "jax-xla":
        raise ValueError("the tpu-massive profile requires the jax-xla backend")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_model_path = output_dir / "atom_causal_world_model.json"
    state_cursor_path = output_dir / "atom_causal_world_resume_cursor.json"
    self_tests = {
        "schema": causal_world_schema_self_test(),
        "curriculum": causal_world_curriculum_self_test(),
        "simulator": causal_world_simulator_self_test(),
        "causal_graph": causal_graph_self_test(),
        "language": causal_world_language_self_test(),
        "formal_domains": formal_domain_self_test(),
        "knowledge": causal_world_knowledge_self_test(),
        "accelerator": accelerator_self_test(),
        "transfer": causal_transfer_self_test(),
    }
    resume_cursor: dict[str, Any] | None = None
    if selected_backend == "jax-xla":
        initial_graph: CausalGraph | None = None
        prior_cursor: dict[str, Any] | None = None
        if resume_from is not None:
            initial_graph, prior_cursor = load_causal_resume_state(
                resume_from,
                config,
                expected_next_shard=shard_index,
            )
        elif shard_index != 0:
            raise ValueError("nonzero accelerator shards require --resume-from")
        state_write_count = 0

        def persist_accelerator_state(
            state_model: Mapping[str, Any], state_cursor: Mapping[str, Any]
        ) -> None:
            nonlocal state_write_count
            write_causal_json(state_model_path, state_model)
            write_causal_json(state_cursor_path, state_cursor)
            state_write_count += 1

        graph, evidence, execution, resume_cursor = _run_accelerator_learning(
            config,
            shard_index,
            require_tpu=require_tpu,
            require_gpu=require_gpu,
            shards_per_run=shards_per_run,
            initial_graph=initial_graph,
            prior_cursor=prior_cursor,
            state_writer=persist_accelerator_state,
        )
        execution["atomic_state_writes"] = state_write_count
    else:
        if resume_from is not None:
            raise ValueError(
                "resume state is supported only by the accelerator backend"
            )
        if shard_index != 0 or shards_per_run not in {0, 1}:
            raise ValueError("numpy validation runs use their configured shard sample")
        graph, evidence, execution = _run_numpy_learning(config)

    cognition = CausalCognition(graph)
    consolidation = cognition.persistence.consolidate(graph)
    model = graph.model_payload()
    workflow_request, workflow_truth = build_causal_workflow_request(graph)
    workflow_response = run_causal_workflow(model, workflow_request)
    evaluation = _workflow_evaluation(workflow_response, workflow_truth)
    corruption = _corruption_checks(model)
    deterministic = (
        _deterministic_numpy_check(get_profile("test"))
        if selected_backend == "numpy"
        else dict(execution["accelerator"]["deterministic_replay"])
    )
    wiki = CausalWorldWikiGraph(graph)
    knowledge_manifest = wiki.manifest()
    evidence_payloads = [asdict(item) for item in evidence]
    evidence_hash = canonical_hash(evidence_payloads)
    if resume_cursor is None:
        local_cursor_core = {
            "schema": CAUSAL_WORLD_RESUME_SCHEMA,
            "runtime": CAUSAL_WORLD_RESUME_RUNTIME,
            "backend": "numpy",
            "resumable": False,
            "profile": config.profile,
            "completed_shards": list(range(int(execution["shards_executed"]))),
            "next_shard": int(execution["shards_executed"]),
            "shard_count": config.shard_count,
            "model_hash": model["model_hash"],
            "evidence_hash": evidence_hash,
            "cumulative_evidence_count": graph.observation_count,
        }
        resume_cursor = {
            **local_cursor_core,
            "cursor_hash": canonical_hash(local_cursor_core),
        }
    crystallized = sum(law.status == "crystallized" for law in graph.laws.values())
    all_components_executed = all(
        {entry["component"] for entry in turn["artifact"]["execution_trace"]}
        == set(ARCHITECTURE_COMPONENTS)
        for turn in workflow_response["turns"]
    )
    graph_rag_exercised = all(
        turn["knowledge_context"] for turn in workflow_response["turns"]
    )
    multi_hop_workflow_exercised = any(
        turn["artifact"]["claim_status"] == "derived"
        and turn["artifact"]["path_length"] > 1
        for turn in workflow_response["turns"]
    )
    evidence_has_no_root_labels = all(
        not (set(asdict(item)) & set(ROOT_MECHANICS)) for item in evidence
    )
    maximum_invariant_error = float(execution["maximum_invariant_error"])
    execution_program_ids = sorted(
        {
            int(program["program_id"])
            for shard in execution["shards"]
            for program in (
                shard["accelerator"]["world_programs"]
                if selected_backend == "jax-xla"
                else shard["world"]["programs"]
            )
        }
    )
    massive_curriculum = curriculum_manifest(
        shard_count=get_profile("tpu-massive").shard_count,
        programs_per_shard=TPU_PROGRAMS_PER_SHARD,
    )
    full_accelerator_execution = (
        selected_backend == "jax-xla"
        and execution["shard_start"] == 0
        and execution["shard_stop"] == config.shard_count
    )
    transfer_artifacts = run_causal_transfer_benchmark(
        model,
        execution_program_ids,
        profile_name="massive" if full_accelerator_execution else "quick",
    )
    transfer_report = transfer_artifacts["report"]
    formal_artifacts = run_formal_domain_benchmark(
        cases_per_primitive=512 if full_accelerator_execution else 24
    )
    formal_report = formal_artifacts["report"]
    evidence_has_world_conditions = all(
        len(stable_condition_signature(item.context_signature)) == 11
        for item in evidence
    )

    gate_checks = {
        "architecture_is_pure_causal_graph": model["architecture"]
        == "pure-executable-causal-phase-hypergraph",
        "mamba_not_imported": _module_avoids_mamba(),
        "all_eight_domains_exercised": {item.domain for item in evidence}
        == set(DOMAIN_NAMES),
        "interventional_treated_and_baseline_evidence": all(
            item.treated_worlds > 0 and item.baseline_worlds > 0 for item in evidence
        ),
        "simulator_truth_not_in_runtime_evidence": evidence_has_no_root_labels,
        "causal_laws_crystallized": crystallized >= 2,
        "all_seven_components_executed": all_components_executed,
        "graph_rag_exercised_every_turn": graph_rag_exercised,
        "multi_hop_causal_workflow": multi_hop_workflow_exercised,
        "supported_and_unknown_workflow": evaluation["derived_answers"] >= 2
        and evaluation["unknown_answers"] == 2,
        "workflow_exact": evaluation["accuracy"] == 1.0,
        "conservation_bounded": maximum_invariant_error < 0.25,
        "model_corruption_rejected": corruption["passed"],
        "deterministic_replay": deterministic["passed"],
        "accelerator_runtime_observed": selected_backend != "jax-xla"
        or execution["accelerator"]["probe"]["jax_available"],
        "required_tpu_observed": not (selected_backend == "jax-xla" and require_tpu)
        or execution["accelerator"]["probe"]["tpu_available"],
        "required_gpu_observed": not (selected_backend == "jax-xla" and require_gpu)
        or execution["accelerator"]["probe"]["gpu_available"],
        "single_cached_jit_executor": selected_backend != "jax-xla"
        or execution["accelerator"]["jit_executor_constructions"] == 1,
        "all_assigned_accelerator_devices_used": selected_backend != "jax-xla"
        or execution["accelerator"]["devices_used"]
        == len(execution["accelerator"]["probe"]["devices"]),
        "massive_profile_exceeds_billion_entity_updates": get_profile(
            "tpu-massive"
        ).scale_manifest()["entity_updates"]
        > 1_000_000_000,
        "accelerator_shards_are_lineage_bound": selected_backend != "jax-xla"
        or (
            resume_cursor["resumable"] is True
            and resume_cursor["model_hash"] == model["model_hash"]
            and resume_cursor["next_shard"] == execution["shard_stop"]
            and resume_cursor["cumulative_evidence_count"] == graph.observation_count
            and execution["atomic_state_writes"] == execution["shards_executed"]
        ),
        "language_surface_exceeds_million_questions": language_space_manifest()[
            "combined_question_space"
        ]
        > 1_000_000,
        "procedural_world_space_exceeds_fifty_million": massive_curriculum[
            "procedural_program_space"
        ]
        > 50_000_000,
        "world_conditions_bound_to_every_evidence_row": evidence_has_world_conditions,
        "curriculum_programs_change_runtime_state": len(execution_program_ids) >= 1,
        "full_accelerator_curriculum_exercises_sixty_four_programs": (
            selected_backend != "jax-xla"
            or execution["shard_start"] != 0
            or execution["shard_stop"] != config.shard_count
            or len(execution_program_ids) == 64
        ),
        "heldout_transfer_runtime_exercised": (
            transfer_report["runtime"] == CAUSAL_TRANSFER_RUNTIME
            and transfer_report["gates"][
                "heldout_programs_disjoint_from_training"
            ]
            and transfer_report["gates"]["evaluator_truth_sealed_from_runtime"]
            and transfer_report["gates"]["two_independent_truth_replicas"]
            and transfer_report["gates"]["cross_feature_truth_only"]
            and transfer_report["gates"]["truth_directions_are_balanced"]
            and transfer_report["gates"]["adversarial_language_fails_closed"]
            and transfer_report["gates"][
                "validation_programs_disjoint_from_training"
            ]
            and transfer_report["gates"][
                "evaluation_programs_disjoint_from_validation"
            ]
        ),
        "metaplastic_transfer_policy_validated": (
            transfer_report["gates"]["metaplastic_policy_passed_validation"]
            and transfer_report["gates"]["metaplastic_policy_is_model_bound"]
            and transfer_report["metaplastic_calibration"]["passed"]
        ),
        "context_factor_risk_policy_validated": (
            transfer_report["metaplastic_calibration"]["gates"][
                "single_factor_probe_reused"
            ]
            and transfer_report["metaplastic_calibration"]["gates"][
                "pair_motif_controls_exercised"
            ]
            and transfer_report["metaplastic_calibration"]["gates"][
                "selected_policy_overall_risk_bound"
            ]
            and transfer_report["metaplastic_calibration"]["gates"][
                "selected_policy_directional_risk_bounds"
            ]
        ),
        "formal_domain_curriculum_passed": formal_report["passed"],
        "all_formal_domains_match_independent_oracles": (
            formal_report["runtime"] == FORMAL_CURRICULUM_RUNTIME
            and formal_report["formal_runtime"] == FORMAL_DOMAIN_RUNTIME
            and formal_report["oracle_runtime"] == FORMAL_TRUTH_ORACLE_RUNTIME
            and formal_report["gates"]["runtime_matches_independent_oracle"]
            and all(
                domain["accuracy"] == 1.0
                for domain in formal_report["per_domain"].values()
            )
        ),
        "formal_epistemic_states_distinguish_proof_and_contradiction": (
            formal_report["gates"]["every_exact_result_is_marked_proven"]
            and formal_report["gates"]["false_candidates_are_contradicted"]
            and formal_report["epistemic_contract"][
                "deduction_is_not_induction_or_abduction"
            ]
        ),
        "formal_cross_domain_composition_passed": formal_report["gates"][
            "cross_domain_programs_are_proven"
        ],
        "formal_registry_is_bound_into_graph_rag": (
            knowledge_manifest["formal_domains"]["registry_hash"]
            == formal_report["registry_hash"]
        ),
        "full_accelerator_heldout_transfer_passed": (
            not full_accelerator_execution or transfer_report["passed"]
        ),
    }
    experiment_gates = {
        "checks": gate_checks,
        "passed": all(gate_checks.values()),
    }

    report_core = {
        "schema": CAUSAL_WORLD_EXPERIMENT_SCHEMA,
        "runtime": CAUSAL_WORLD_EXPERIMENT_RUNTIME,
        "architecture": model["architecture"],
        "model_hash": model["model_hash"],
        "world": {
            **config_manifest(config),
            "domain_counts": {
                domain: sum(item.domain == domain for item in evidence)
                for domain in DOMAIN_NAMES
            },
            "language_space": language_space_manifest(),
            "curriculum": {
                **massive_curriculum,
                "execution_program_ids": execution_program_ids,
                "execution_program_count": len(execution_program_ids),
            },
        },
        "learning": {
            "evidence_count": graph.observation_count,
            "current_run_evidence_count": len(evidence),
            "cumulative_evidence_count": graph.observation_count,
            "evidence_hash": evidence_hash,
            "graph_nodes": len(graph.nodes),
            "graph_laws": len(graph.laws),
            "crystallized_laws": crystallized,
            "hypothesis_laws": sum(
                law.status == "hypothesis" for law in graph.laws.values()
            ),
            "retired_laws": sum(law.status == "retired" for law in graph.laws.values()),
            "consolidation": consolidation,
        },
        "execution": execution,
        "accelerator_plan": build_accelerator_plan(get_profile("tpu-massive")),
        "accelerator_probe": probe_jax_accelerator(),
        "evaluation": evaluation,
        "transfer_benchmark": transfer_report,
        "formal_domains": formal_report,
        "corruption_checks": corruption,
        "deterministic_replay": deterministic,
        "self_tests": self_tests,
        "knowledge": {
            "wiki_runtime": CAUSAL_WORLD_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_WORLD_RAG_RUNTIME,
            "manifest_hash": canonical_hash(knowledge_manifest),
            "node_count": len(knowledge_manifest["nodes"]),
        },
        "side_view_contract": {
            "runtime": ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME,
            "placement": "side",
            "user_visible": True,
            "artifact_binding_marker": "render_causal_world_artifact",
        },
        "experiment_gates": experiment_gates,
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    paths = {
        "model": state_model_path,
        "manifest": output_dir / "atom_causal_world_manifest.json",
        "evidence": output_dir / "atom_causal_world_evidence.jsonl",
        "evaluator_truth": output_dir / "atom_causal_world_evaluator_truth.json",
        "workflow_request": output_dir / "atom_causal_world_workflow_request.json",
        "workflow_response": output_dir / "atom_causal_world_workflow_response.json",
        "knowledge_graph": output_dir / "atom_causal_world_knowledge_graph.json",
        "resume_cursor": state_cursor_path,
        "report": output_dir / "atom_causal_world_report.json",
        "side_view": output_dir / "atom_causal_world_side_view.html",
        "transfer_validation_truth": output_dir
        / "atom_causal_world_transfer_validation_truth.json",
        "transfer_policy": output_dir / "atom_causal_world_transfer_policy.json",
        "transfer_truth": output_dir / "atom_causal_world_transfer_truth.json",
        "transfer_request": output_dir / "atom_causal_world_transfer_request.json",
        "transfer_exact_response": output_dir
        / "atom_causal_world_transfer_exact_response.json",
        "transfer_response": output_dir / "atom_causal_world_transfer_response.json",
        "transfer_report": output_dir / "atom_causal_world_transfer_report.json",
        "formal_domains": output_dir / "atom_causal_world_formal_domains.json",
    }
    write_causal_json(paths["model"], model)
    write_causal_json(paths["manifest"], config_manifest(config))
    write_causal_jsonl(paths["evidence"], evidence_payloads)
    write_causal_json(paths["evaluator_truth"], workflow_truth)
    write_causal_json(paths["workflow_request"], workflow_request)
    write_causal_json(paths["workflow_response"], workflow_response)
    write_causal_json(paths["knowledge_graph"], knowledge_manifest)
    write_causal_json(paths["resume_cursor"], resume_cursor)
    write_causal_json(paths["report"], report)
    write_causal_json(
        paths["transfer_validation_truth"],
        transfer_artifacts["validation_truth"],
    )
    write_causal_json(
        paths["transfer_policy"],
        transfer_artifacts["transfer_policy"],
    )
    write_causal_json(paths["transfer_truth"], transfer_artifacts["truth"])
    write_causal_json(paths["transfer_request"], transfer_artifacts["request"])
    write_causal_json(
        paths["transfer_exact_response"], transfer_artifacts["exact_response"]
    )
    write_causal_json(paths["transfer_response"], transfer_artifacts["transfer_response"])
    write_causal_json(paths["transfer_report"], transfer_report)
    write_causal_json(paths["formal_domains"], formal_artifacts)
    _write_causal_text(
        paths["side_view"],
        render_causal_world_artifact(model, report, workflow_response),
    )
    return report


def run_self_tests() -> dict[str, Any]:
    checks = {
        "schema": causal_world_schema_self_test(),
        "curriculum": causal_world_curriculum_self_test(),
        "simulator": causal_world_simulator_self_test(),
        "causal_graph": causal_graph_self_test(),
        "language": causal_world_language_self_test(),
        "formal_domains": formal_domain_self_test(),
        "knowledge": causal_world_knowledge_self_test(),
        "accelerator": accelerator_self_test(),
        "transfer": causal_transfer_self_test(),
        "mamba_not_imported": _module_avoids_mamba(),
    }
    return {"passed": all(bool(value) for value in checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    on_kaggle = Path("/kaggle/working").is_dir()
    bundle_accelerator = str(
        globals().get("KAGGLE_BUNDLE_ACCELERATOR", "tpu")
    ).lower()
    if bundle_accelerator not in {"tpu", "gpu"}:
        raise ValueError("Kaggle bundle accelerator must be TPU or GPU")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("/kaggle/working/causal_world_outputs")
            if on_kaggle
            else Path("causal_world_outputs")
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("test", "local", "tpu-massive"),
        default="tpu-massive" if on_kaggle else "local",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "jax-xla"),
        default="auto",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shards-per-run",
        type=int,
        default=0 if on_kaggle else 1,
        help="zero executes every remaining accelerator shard",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="validated prior output directory for a nonzero accelerator shard",
    )
    accelerator_group = parser.add_mutually_exclusive_group()
    accelerator_group.add_argument(
        "--require-tpu",
        action="store_true",
        default=on_kaggle and bundle_accelerator == "tpu",
    )
    accelerator_group.add_argument(
        "--require-gpu",
        action="store_true",
        default=on_kaggle and bundle_accelerator == "gpu",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_self_tests()
    else:
        result = run_causal_world_experiment(
            args.output_dir,
            profile=args.profile,
            backend=args.backend,
            shard_index=args.shard_index,
            shards_per_run=args.shards_per_run,
            resume_from=args.resume_from,
            require_tpu=args.require_tpu,
            require_gpu=args.require_gpu,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
