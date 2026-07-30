"""Discover and continually refine open mathematical primitives from seven roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash
from atom_primitive_forge import (
    CANDIDATE_STATUS,
    CRYSTALLIZED_STATUS,
    PRIMITIVE_FORGE_RUNTIME,
    QUARANTINED_STATUS,
    RETIRED_STATUS,
    REVISED_STATUS,
    PrimitiveForge,
)
from atom_primitive_knowledge import (
    PRIMITIVE_RAG_RUNTIME,
    PRIMITIVE_WIKI_RUNTIME,
    PrimitiveWikiGraph,
    retrieve_primitive_context,
)
from atom_primitive_side_view import (
    PRIMITIVE_SIDE_VIEW_RUNTIME,
    render_primitive_forge_artifact,
)
from atom_primitive_simulation import (
    PRIMITIVE_SIMULATION_RUNTIME,
    counterfactual_world,
    evaluate_primitive,
    evaluate_root_expansion,
    training_world,
)


PRIMITIVE_EXPERIMENT_SCHEMA = 1
PRIMITIVE_EXPERIMENT_RUNTIME = "atom-open-primitive-discovery-v1"
PRIMITIVE_CONTINUAL_RUNTIME = "atom-primitive-continual-use-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _calibration_state(candidate_index: int, trial: int) -> float:
    raw = ((candidate_index + 3) * (trial + 5) * 17) % 181
    return -0.82 + raw / 181.0 * 1.64


def _calibrate_serial_candidate(
    forge: PrimitiveForge,
    primitive_id: str,
    *,
    candidate_index: int,
    stage: str,
) -> list[str]:
    statuses = [forge.get(primitive_id).status]
    roots = forge.expand_to_roots(primitive_id)
    for trial in range(forge.config.promotion_evidence):
        world = training_world(candidate_index * 5 + trial)
        state = _calibration_state(candidate_index, trial)
        predicted = evaluate_primitive(forge, primitive_id, state, world)
        observed = evaluate_root_expansion(roots, state, world)
        updated = forge.observe(
            primitive_id,
            context_id=f"{stage}:candidate-{candidate_index}:trial-{trial}",
            predicted=predicted,
            observed=observed,
            tolerance=1e-12,
            source="bounded-mathematical-simulation",
            provenance=(
                f"experiment:{PRIMITIVE_EXPERIMENT_RUNTIME}",
                f"stage:{stage}",
            ),
        )
        statuses.append(updated.status)
    return statuses


def _discover_inventory(
    forge: PrimitiveForge,
) -> tuple[list[str], list[str], list[str]]:
    """Search compositions by depth without a handwritten outcome registry."""

    first_generation: list[str] = []
    first_trace: list[str] = []
    candidate_index = 0
    for left in forge.root_ids:
        for right in forge.root_ids:
            if left == right:
                continue
            proposed = forge.compose(
                "serial",
                (left, right),
                aliases=(
                    f"generated depth one candidate {candidate_index}",
                    "algorithmic serial field composition",
                ),
                invariants=("bounded scalar output",),
                symmetries=("representation-preserving execution",),
                boundaries=("finite held-out scalar interval",),
                scales=("synthetic scalar field experiment",),
                provenance=(
                    "search:ordered-root-pairs",
                    f"search-index:{candidate_index}",
                ),
            )
            trace = _calibrate_serial_candidate(
                forge,
                proposed.primitive_id,
                candidate_index=candidate_index,
                stage="depth-one",
            )
            if not first_trace:
                first_trace = trace
            if forge.get(proposed.primitive_id).status != CRYSTALLIZED_STATUS:
                raise RuntimeError("depth-one candidate did not crystallize")
            first_generation.append(proposed.primitive_id)
            candidate_index += 1

    second_generation: list[str] = []
    selected = sorted(first_generation)[::2][:18]
    for offset, parent_id in enumerate(selected):
        root = forge.root_ids[(offset * 3 + 1) % len(forge.root_ids)]
        proposed = forge.compose(
            "serial",
            (parent_id, root),
            aliases=(
                f"generated recursive depth two candidate {offset}",
                "recursive primitive reuse",
            ),
            invariants=("recursive root reducibility",),
            symmetries=("canonical associative execution",),
            boundaries=("finite held-out scalar interval",),
            scales=("synthetic counterfactual field experiment",),
            provenance=(
                "search:derived-plus-root",
                f"parent:{parent_id}",
                f"search-index:{offset}",
            ),
        )
        _calibrate_serial_candidate(
            forge,
            proposed.primitive_id,
            candidate_index=candidate_index,
            stage="depth-two",
        )
        if forge.get(proposed.primitive_id).status != CRYSTALLIZED_STATUS:
            raise RuntimeError("recursive candidate did not crystallize")
        second_generation.append(proposed.primitive_id)
        candidate_index += 1
    return first_generation, second_generation, first_trace


def _evaluate_serial_transfer(
    forge: PrimitiveForge,
    primitive_ids: Sequence[str],
    *,
    counterfactual: bool,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    for primitive_index, primitive_id in enumerate(primitive_ids):
        roots = forge.expand_to_roots(primitive_id)
        for trial in range(2):
            world_index = primitive_index * 3 + trial + 50
            world = (
                counterfactual_world(world_index)
                if counterfactual
                else training_world(world_index)
            )
            state = _calibration_state(primitive_index + 71, trial + 9)
            predicted = evaluate_primitive(forge, primitive_id, state, world)
            observed = evaluate_root_expansion(roots, state, world)
            residual = abs(predicted - observed)
            records.append(
                {
                    "primitive_id": primitive_id,
                    "world_index": world_index,
                    "partition": (
                        "counterfactual"
                        if counterfactual
                        else "heldout"
                    ),
                    "state": state,
                    "prediction": predicted,
                    "observation": observed,
                    "residual": residual,
                    "passed": residual <= 1e-12,
                    "root_expansion": list(roots),
                }
            )
    accuracy = (
        sum(bool(record["passed"]) for record in records) / len(records)
        if records
        else 0.0
    )
    return records, accuracy


def _exercise_equivalence(forge: PrimitiveForge) -> dict[str, Any]:
    first = forge.compose(
        "parallel",
        ("radiation", "dissipation"),
        aliases=("equivalence audit forward",),
        provenance=("equivalence-probe:forward",),
    )
    second = forge.compose(
        "parallel",
        ("dissipation", "radiation"),
        aliases=("equivalence audit reverse",),
        provenance=("equivalence-probe:reverse",),
    )
    merged = forge.get(first.primitive_id)
    return {
        "forward_id": first.primitive_id,
        "reverse_id": second.primitive_id,
        "same_identity": first.primitive_id == second.primitive_id,
        "recipe_count": 1 + len(merged.equivalent_recipes),
        "provenance_preserved": {
            "equivalence-probe:forward",
            "equivalence-probe:reverse",
        }.issubset(set(merged.provenance)),
    }


def _exercise_revision_and_decay(
    forge: PrimitiveForge,
    recursive_ids: Sequence[str],
) -> dict[str, Any]:
    revision_id = recursive_ids[0]
    revision_before = forge.get(revision_id)
    world = counterfactual_world(501)
    state = 0.317
    prediction = evaluate_primitive(forge, revision_id, state, world)
    revision_after = forge.observe(
        revision_id,
        context_id="continual-use:contradiction",
        predicted=prediction,
        observed=prediction + 0.25,
        tolerance=1e-12,
        source="bounded-mathematical-simulation",
        provenance=("continual-use:counterexample",),
    )

    decay_candidate = forge.compose(
        "feedback",
        ("radiation", "conservation"),
        aliases=("unsupported continual-use candidate",),
        provenance=("continual-use:quarantine",),
    )
    decay_world = training_world(707)
    decay_prediction = evaluate_primitive(
        forge,
        decay_candidate.primitive_id,
        -0.213,
        decay_world,
    )
    candidate = forge.observe(
        decay_candidate.primitive_id,
        context_id="continual-use:single-observation",
        predicted=decay_prediction,
        observed=decay_prediction,
        tolerance=1e-12,
        source="bounded-mathematical-simulation",
    )
    retired = forge.apply_decay(
        decay_candidate.primitive_id,
        amount=0.5,
        provenance="continual-use:no-repeated-support",
    )
    return {
        "revision_id": revision_id,
        "revision_before": revision_before.status,
        "revision_after": revision_after.status,
        "counterexample_count": len(revision_after.counterexamples),
        "decay_id": decay_candidate.primitive_id,
        "decay_initial": decay_candidate.status,
        "decay_after_one_observation": candidate.status,
        "decay_final": retired.status,
    }


def _root_immutability_probe(forge: PrimitiveForge) -> bool:
    before = forge.get("radiation")
    try:
        forge.observe(
            "radiation",
            context_id="forbidden-root-update",
            predicted=0.0,
            observed=0.0,
        )
    except ValueError:
        return forge.get("radiation") == before
    return False


def run_primitive_forge_experiment(output_dir: Path) -> dict[str, Any]:
    """Run discovery, transfer, continual learning, RAG, and side-view gates."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forge = PrimitiveForge()
    first_generation, second_generation, promotion_trace = _discover_inventory(
        forge
    )
    heldout_records, heldout_accuracy = _evaluate_serial_transfer(
        forge,
        second_generation,
        counterfactual=False,
    )
    counterfactual_records, counterfactual_accuracy = (
        _evaluate_serial_transfer(
            forge,
            second_generation,
            counterfactual=True,
        )
    )
    equivalence = _exercise_equivalence(forge)
    continual = _exercise_revision_and_decay(forge, second_generation)
    roots_immutable = _root_immutability_probe(forge)

    model = forge.model_payload()
    restored = PrimitiveForge.from_model_payload(model)
    knowledge = PrimitiveWikiGraph(restored)
    knowledge.assert_bound_to(restored)
    knowledge_manifest = knowledge.manifest()
    rag_target = second_generation[-1]
    query = (
        f"{rag_target} recursive crystallized mathematical scalar field "
        "root expansion"
    )
    knowledge_context = retrieve_primitive_context(
        knowledge,
        query,
        limit=8,
    )
    request_core = {
        "schema": PRIMITIVE_EXPERIMENT_SCHEMA,
        "runtime": PRIMITIVE_EXPERIMENT_RUNTIME,
        "request_id": "open-primitive-discovery-and-transfer",
        "query": query,
    }
    request = {
        **request_core,
        "request_hash": canonical_hash(request_core),
    }

    recursive_count = sum(
        record.recipe is not None
        and any(not restored.get(item).root for item in record.recipe.components)
        for record in restored.derived_records
    )
    statuses = {
        status: sum(record.status == status for record in restored.derived_records)
        for status in (
            QUARANTINED_STATUS,
            CANDIDATE_STATUS,
            CRYSTALLIZED_STATUS,
            REVISED_STATUS,
            RETIRED_STATUS,
        )
    }
    all_root_expansions_valid = all(
        bool(restored.expand_to_roots(record.primitive_id))
        and set(restored.expand_to_roots(record.primitive_id))
        <= set(ROOT_MECHANICS)
        for record in restored.records
    )
    checks = {
        "exactly_seven_immutable_roots": (
            restored.root_ids == tuple(ROOT_MECHANICS) and roots_immutable
        ),
        "derived_inventory_is_open_ended_and_above_twenty": (
            len(restored.derived_records) > 20
        ),
        "new_primitives_are_reused_recursively": (
            recursive_count >= len(second_generation)
        ),
        "every_primitive_expands_to_roots": all_root_expansions_valid,
        "promotion_requires_repeated_prediction": (
            promotion_trace
            == [
                QUARANTINED_STATUS,
                CANDIDATE_STATUS,
                CANDIDATE_STATUS,
                CRYSTALLIZED_STATUS,
            ]
        ),
        "equivalent_recipes_merge_with_provenance": (
            equivalence["same_identity"]
            and equivalence["recipe_count"] >= 2
            and equivalence["provenance_preserved"]
        ),
        "contradiction_revises_crystallized_structure": (
            continual["revision_before"] == CRYSTALLIZED_STATUS
            and continual["revision_after"] == REVISED_STATUS
            and continual["counterexample_count"] == 1
        ),
        "unsupported_candidate_decays_to_retired": (
            continual["decay_initial"] == QUARANTINED_STATUS
            and continual["decay_after_one_observation"] == CANDIDATE_STATUS
            and continual["decay_final"] == RETIRED_STATUS
        ),
        "heldout_compositions_transfer": heldout_accuracy == 1.0,
        "counterfactual_controls_transfer": counterfactual_accuracy == 1.0,
        "serialization_round_trip_is_hash_stable": (
            restored.graph_hash == model["graph_hash"]
        ),
        "wiki_and_rag_are_graph_bound": (
            knowledge_manifest["source_graph_hash"] == restored.graph_hash
            and bool(knowledge_context)
            and knowledge_context[0]["primitive_id"] == rag_target
        ),
    }
    report_core = {
        "schema": PRIMITIVE_EXPERIMENT_SCHEMA,
        "runtime": PRIMITIVE_EXPERIMENT_RUNTIME,
        "graph_hash": restored.graph_hash,
        "claim_scope": (
            "Measured only in the declared bounded mathematical scalar-field "
            "simulation; this does not establish complete physics, quantum "
            "understanding, or universal mathematical understanding."
        ),
        "hierarchy": [
            "seven immutable roots",
            "open-ended recursively discovered primitives",
            "structures, laws, and systems",
            "later optional coding and interface projections",
        ],
        "inventory": {
            "root_count": len(restored.root_ids),
            "derived_count": len(restored.derived_records),
            "first_generation_count": len(set(first_generation)),
            "second_generation_count": len(set(second_generation)),
            "recursive_count": recursive_count,
            "crystallized_count": statuses[CRYSTALLIZED_STATUS],
            "status_counts": statuses,
        },
        "learning": {
            "promotion_trace": promotion_trace,
            "equivalence": equivalence,
            "continual_use": continual,
        },
        "evaluation": {
            "heldout_accuracy": heldout_accuracy,
            "heldout_count": len(heldout_records),
            "counterfactual_accuracy": counterfactual_accuracy,
            "counterfactual_count": len(counterfactual_records),
            "heldout_records": heldout_records,
            "counterfactual_records": counterfactual_records,
        },
        "knowledge": {
            "wiki_runtime": PRIMITIVE_WIKI_RUNTIME,
            "rag_runtime": PRIMITIVE_RAG_RUNTIME,
            "knowledge_hash": knowledge_manifest["knowledge_hash"],
            "query": query,
            "hit_count": len(knowledge_context),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "side_view_contract": {
            "runtime": PRIMITIVE_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": "render_primitive_forge_artifact",
            "placement": "side",
            "user_visible": True,
        },
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    workflow_core = {
        "schema": PRIMITIVE_EXPERIMENT_SCHEMA,
        "runtime": {
            "experiment": PRIMITIVE_EXPERIMENT_RUNTIME,
            "forge": PRIMITIVE_FORGE_RUNTIME,
            "simulation": PRIMITIVE_SIMULATION_RUNTIME,
            "wiki": PRIMITIVE_WIKI_RUNTIME,
            "rag": PRIMITIVE_RAG_RUNTIME,
            "continual_use": PRIMITIVE_CONTINUAL_RUNTIME,
        },
        "request_hash": request["request_hash"],
        "graph_hash": restored.graph_hash,
        "artifact_hash": canonical_hash(model),
        "report_hash": report["report_hash"],
        "knowledge_hash": knowledge_manifest["knowledge_hash"],
        "knowledge_context": knowledge_context,
        "claim_status": "bounded-mathematical-simulation",
    }
    workflow = {
        **workflow_core,
        "response_hash": canonical_hash(workflow_core),
    }
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"primitive forge experiment failed gates: {failed}")
    side_view = render_primitive_forge_artifact(
        model,
        report,
        workflow,
        knowledge_manifest,
    )

    _write_json(output_dir / "atom_primitive_graph.json", model)
    _write_json(output_dir / "atom_primitive_report.json", report)
    _write_json(
        output_dir / "atom_primitive_knowledge_graph.json",
        knowledge_manifest,
    )
    _write_json(
        output_dir / "atom_primitive_workflow_request.json",
        request,
    )
    _write_json(
        output_dir / "atom_primitive_workflow_response.json",
        workflow,
    )
    (output_dir / "atom_primitive_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    return report


def build_use_observation_request(
    *,
    primitive_id: str,
    context_id: str,
    predicted: float,
    observed: float,
    tolerance: float = 1e-9,
    source: str = "interactive-use",
    provenance: Sequence[str] = (),
) -> dict[str, Any]:
    core = {
        "schema": PRIMITIVE_EXPERIMENT_SCHEMA,
        "runtime": PRIMITIVE_CONTINUAL_RUNTIME,
        "primitive_id": primitive_id,
        "context_id": context_id,
        "predicted": predicted,
        "observed": observed,
        "tolerance": tolerance,
        "source": source,
        "provenance": list(provenance),
    }
    return {**core, "observation_hash": canonical_hash(core)}


def ingest_use_observation(
    model_path: Path,
    observation_path: Path,
    output_model_path: Path,
) -> dict[str, Any]:
    """Apply one hash-bound live-use observation and persist the new graph."""

    forge = PrimitiveForge.load(model_path)
    request = json.loads(Path(observation_path).read_text(encoding="utf-8"))
    required = {
        "schema",
        "runtime",
        "primitive_id",
        "context_id",
        "predicted",
        "observed",
        "tolerance",
        "source",
        "provenance",
        "observation_hash",
    }
    if not isinstance(request, Mapping) or set(request) != required:
        raise ValueError("continual-use observation keys are invalid")
    if request["schema"] != PRIMITIVE_EXPERIMENT_SCHEMA:
        raise ValueError("continual-use observation schema is invalid")
    if request["runtime"] != PRIMITIVE_CONTINUAL_RUNTIME:
        raise ValueError("continual-use runtime marker is invalid")
    core = {key: request[key] for key in request if key != "observation_hash"}
    if canonical_hash(core) != request["observation_hash"]:
        raise ValueError("continual-use observation hash mismatch")
    if not isinstance(request["provenance"], list):
        raise ValueError("continual-use provenance must be a list")
    before_hash = forge.graph_hash
    before_status = forge.get(request["primitive_id"]).status
    updated = forge.observe(
        request["primitive_id"],
        context_id=request["context_id"],
        predicted=request["predicted"],
        observed=request["observed"],
        tolerance=request["tolerance"],
        source=request["source"],
        provenance=request["provenance"],
    )
    forge.save(output_model_path)
    response_core = {
        "schema": PRIMITIVE_EXPERIMENT_SCHEMA,
        "runtime": PRIMITIVE_CONTINUAL_RUNTIME,
        "observation_hash": request["observation_hash"],
        "before_graph_hash": before_hash,
        "after_graph_hash": forge.graph_hash,
        "primitive_id": updated.primitive_id,
        "before_status": before_status,
        "after_status": updated.status,
        "confidence": updated.confidence,
        "persistence": updated.persistence,
        "evidence_count": len(updated.evidence),
        "counterexample_count": len(updated.counterexamples),
    }
    return {**response_core, "response_hash": canonical_hash(response_core)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("primitive_forge_outputs"),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--updated-model", type=Path)
    args = parser.parse_args()
    continual_args = (args.model, args.observation, args.updated_model)
    if any(item is not None for item in continual_args):
        if not all(item is not None for item in continual_args):
            parser.error(
                "--model, --observation, and --updated-model are required "
                "together"
            )
        response = ingest_use_observation(
            args.model,
            args.observation,
            args.updated_model,
        )
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0
    report = run_primitive_forge_experiment(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
