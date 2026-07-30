"""Persist Primitive Forge glyphs and exercise structural causal retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from atom_causal_memory import (
    CAUSAL_MEMORY_RUNTIME,
    CausalMemoryClient,
    binary_sha256,
    build_query_for_primitive,
    build_release_binary,
    build_structural_query,
    load_forge,
    structural_features_for,
)
from atom_causal_memory_knowledge import (
    CAUSAL_MEMORY_RAG_RUNTIME,
    CAUSAL_MEMORY_WIKI_RUNTIME,
    CausalMemoryWikiGraph,
    retrieve_causal_memory_context,
)
from atom_causal_memory_side_view import (
    CAUSAL_MEMORY_SIDE_VIEW_RUNTIME,
    render_causal_memory_artifact,
)
from atom_causal_world_schema import canonical_hash
from atom_primitive_forge import CRYSTALLIZED_STATUS, PrimitiveForge


CAUSAL_MEMORY_EXPERIMENT_RUNTIME = "atom-causal-memory-experiment-v1"
DEFAULT_FORGE_PATH = (
    Path(__file__).resolve().parent
    / "primitive_forge_outputs"
    / "atom_primitive_graph.json"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(forge: PrimitiveForge) -> str:
    recursive = [
        record
        for record in forge.derived_records
        if record.status == CRYSTALLIZED_STATUS
        and record.recipe is not None
        and any(
            not forge.get(component).root
            for component in record.recipe.components
        )
    ]
    if not recursive:
        raise RuntimeError(
            "Forge graph contains no crystallized recursive primitive"
        )
    return sorted(
        recursive,
        key=lambda record: (
            -len(forge.expand_to_roots(record.primitive_id)),
            record.primitive_id,
        ),
    )[0].primitive_id


def _score(report: Mapping[str, Any], primitive_id: str) -> int:
    for hit in report["hits"]:
        if hit["primitive_id"] == primitive_id:
            return int(hit["score"])
    return 0


def _run_checks(
    *,
    forge: PrimitiveForge,
    imported: Mapping[str, Any],
    inventory: Mapping[str, Any],
    target: str,
    competitor: str,
    structural_features: tuple[tuple[str, str], ...],
    before: Mapping[str, Any],
    after_wrong: Mapping[str, Any],
    after_correct: Mapping[str, Any],
    persisted: Mapping[str, Any],
    unknown: Mapping[str, Any],
    wrong_feedback: Mapping[str, Any],
    correct_feedback: Mapping[str, Any],
    knowledge: Mapping[str, Any],
    context: list[dict[str, Any]],
) -> dict[str, bool]:
    wrong_adjustments = wrong_feedback["adjustments"]
    correct_adjustments = correct_feedback["adjustments"]
    return {
        "all_forge_glyphs_are_durable": (
            imported["glyph_count"] == len(forge.records)
            and len(inventory["glyphs"]) == len(forge.records)
            and {item["primitive_id"] for item in inventory["glyphs"]}
            == {record.primitive_id for record in forge.records}
        ),
        "seven_roots_remain_immutable_substrate": (
            imported["root_count"] == 7
            and sum(item["root"] for item in inventory["glyphs"]) == 7
        ),
        "causal_topology_is_materialized_as_motifs": (
            imported["motif_count"] > len(forge.records) * 10
            and imported["durable_bonds"] > imported["motif_count"] * 3
        ),
        "retrieval_uses_no_alias_or_prose_role": all(
            role not in {"alias", "provenance"}
            and not role.startswith(("evidence/", "counterexample/"))
            for role, _ in structural_features
        ),
        "structural_resonance_recovers_target": (
            before["answerable"]
            and before["hits"]
            and before["hits"][0]["primitive_id"] == target
            and before["hits"][0]["coverage_per_million"] == 1_000_000
        ),
        "retrieval_returns_exact_topological_paths": (
            bool(before["hits"][0]["motifs"])
            and all(
                len(item["motif"]) == 64
                and item["role"]
                and item["value"]
                and item["conductance_per_mille"] == 1000
                for item in before["hits"][0]["motifs"]
            )
        ),
        "wrong_prediction_changes_both_competing_paths": (
            wrong_feedback["prediction_correct"] is False
            and any(
                item["primitive_id"] == target
                and item["polarity"] == "strengthen"
                for item in wrong_adjustments
            )
            and any(
                item["primitive_id"] == competitor
                and item["polarity"] == "weaken"
                for item in wrong_adjustments
            )
            and _score(after_wrong, target) > _score(before, target)
            and _score(after_wrong, competitor)
            < _score(before, competitor)
        ),
        "correct_prediction_strengthens_only_selected_target": (
            correct_feedback["prediction_correct"] is True
            and correct_adjustments
            and all(
                item["primitive_id"] == target
                and item["polarity"] == "strengthen"
                for item in correct_adjustments
            )
            and _score(after_correct, target)
            > _score(after_wrong, target)
        ),
        "learning_survives_process_reopen": persisted == after_correct,
        "unknown_required_structure_fails_closed": (
            unknown["answerable"] is False
            and unknown["insufficient_evidence"] is True
            and unknown["hits"] == []
        ),
        "wiki_and_rag_use_active_durable_catalog": (
            knowledge["source_graph_hash"] == forge.graph_hash
            and knowledge["catalog_identity"]
            == inventory["catalog_identity"]
            and len(knowledge["nodes"]) == len(forge.records)
            and context
            and context[0]["primitive_id"] == target
            and bool(context[0]["evidence_paths"])
        ),
    }


def run_causal_memory_experiment(
    output_dir: Path,
    *,
    forge_path: Path = DEFAULT_FORGE_PATH,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "atom_causal_memory.atomdb"
    if store_path.exists():
        raise FileExistsError(
            "causal-memory experiment refuses to overwrite an existing store"
        )
    binary = build_release_binary()
    forge = load_forge(forge_path)
    client = CausalMemoryClient(store_path, binary)
    imported = client.import_forge(forge)
    target = _target(forge)
    structural_features = structural_features_for(forge, target)
    query_wire = build_query_for_primitive(forge, target)
    before = client.query(query_wire)
    competitors = [
        hit["primitive_id"]
        for hit in before["hits"]
        if hit["primitive_id"] != target
    ]
    if not competitors:
        raise RuntimeError(
            "structural query did not expose a competing causal glyph"
        )
    competitor = competitors[0]

    wrong_feedback = client.observe_prediction(
        query_wire,
        expected_glyph=target,
        selected_glyph=competitor,
    )
    after_wrong = client.query(query_wire)
    correct_feedback = client.observe_prediction(
        query_wire,
        expected_glyph=target,
        selected_glyph=target,
    )
    after_correct = client.query(query_wire)
    persisted_client = CausalMemoryClient(store_path, binary)
    persisted = persisted_client.query(query_wire)
    unknown_wire = build_structural_query(
        query_id=f"unobserved-topology:{forge.graph_hash[:16]}",
        features=(
            ("domain", forge.get(target).signature.domain),
            ("kind", "derived"),
            ("recipe/mode", forge.get(target).recipe.mode),
            ("component/9999", "unobserved-causal-glyph"),
        ),
        required_roles=(
            "domain",
            "kind",
            "recipe/mode",
            "component/9999",
        ),
        minimum_support=3,
        minimum_coverage_per_million=500_000,
        limit=8,
    )
    unknown = persisted_client.query(unknown_wire)
    inventory = persisted_client.inventory()
    wiki = CausalMemoryWikiGraph(
        persisted_client,
        forge,
        inventory,
    )
    knowledge = wiki.manifest()
    context = retrieve_causal_memory_context(wiki, query_wire)
    checks = _run_checks(
        forge=forge,
        imported=imported,
        inventory=inventory,
        target=target,
        competitor=competitor,
        structural_features=structural_features,
        before=before,
        after_wrong=after_wrong,
        after_correct=after_correct,
        persisted=persisted,
        unknown=unknown,
        wrong_feedback=wrong_feedback,
        correct_feedback=correct_feedback,
        knowledge=knowledge,
        context=context,
    )
    report_core = {
        "schema": 1,
        "runtime": CAUSAL_MEMORY_EXPERIMENT_RUNTIME,
        "memory_runtime": CAUSAL_MEMORY_RUNTIME,
        "source_graph_hash": forge.graph_hash,
        "claim_scope": (
            "Measurements cover durable structural retrieval and feedback over "
            "the bounded Primitive Forge graph. They do not establish general "
            "language understanding, complete physics, or universal reasoning."
        ),
        "storage": imported,
        "retrieval": {
            "target_glyph": target,
            "competing_glyph": competitor,
            "structural_feature_count": len(structural_features),
            "before_feedback": before,
            "after_wrong_prediction": after_wrong,
            "after_correct_prediction": after_correct,
            "persisted": persisted,
            "unknown": unknown,
            "target_score_increase": (
                _score(persisted, target) - _score(before, target)
            ),
        },
        "learning": {
            "wrong_prediction": wrong_feedback,
            "correct_prediction": correct_feedback,
        },
        "knowledge": {
            "wiki_runtime": CAUSAL_MEMORY_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_MEMORY_RAG_RUNTIME,
            "knowledge_hash": knowledge["knowledge_hash"],
            "context": context,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "side_view_contract": {
            "runtime": CAUSAL_MEMORY_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": "render_causal_memory_artifact",
            "placement": "side",
            "user_visible": True,
        },
    }
    report = {
        **report_core,
        "report_hash": canonical_hash(report_core),
    }
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"causal-memory experiment failed gates: {failed}")
    workflow_core = {
        "schema": 1,
        "runtime": CAUSAL_MEMORY_EXPERIMENT_RUNTIME,
        "source_graph_hash": forge.graph_hash,
        "catalog_identity": inventory["catalog_identity"],
        "store_sha256": _sha256(store_path),
        "binary_sha256": binary_sha256(binary),
        "report_hash": report["report_hash"],
        "knowledge_hash": knowledge["knowledge_hash"],
        "query_id": before["query_id"],
        "snapshot_sequence": persisted["snapshot_sequence"],
    }
    workflow = {
        **workflow_core,
        "workflow_hash": canonical_hash(workflow_core),
    }
    side_view = render_causal_memory_artifact(
        report,
        inventory,
        workflow,
        knowledge,
    )
    _write_json(output_dir / "atom_causal_memory_report.json", report)
    _write_json(
        output_dir / "atom_causal_memory_inventory.json",
        inventory,
    )
    _write_json(
        output_dir / "atom_causal_memory_knowledge.json",
        knowledge,
    )
    _write_json(
        output_dir / "atom_causal_memory_workflow.json",
        workflow,
    )
    (output_dir / "atom_causal_memory_query.txt").write_text(
        query_wire,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_memory_unknown_query.txt").write_text(
        unknown_wire,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_memory_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist and exercise Primitive Forge causal glyphs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_memory_outputs"),
    )
    parser.add_argument(
        "--forge",
        type=Path,
        default=DEFAULT_FORGE_PATH,
    )
    arguments = parser.parse_args()
    report = run_causal_memory_experiment(
        arguments.output_dir,
        forge_path=arguments.forge,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "target": report["retrieval"]["target_glyph"],
                "glyphs": report["storage"]["glyph_count"],
                "motifs": report["storage"]["motif_count"],
                "score_increase": report["retrieval"][
                    "target_score_increase"
                ],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
