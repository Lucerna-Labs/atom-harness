"""Exercise full causal-world experience memory through the Rust store."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience import (
    EXPERIENCE_MEMORY_RUNTIME,
    ExperienceCorpus,
    ExperienceMemoryClient,
    ExperienceRecord,
    build_experience_query,
    build_query_for_record,
    load_experience_corpus,
)
from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
    CausalExperienceWikiGraph,
    retrieve_causal_experience_context,
    validate_causal_experience_knowledge,
)
from atom_causal_experience_side_view import (
    CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME,
    render_causal_experience_artifact,
)
from atom_causal_memory import (
    CausalMemoryClient,
    binary_sha256,
    build_release_binary,
    load_forge,
)
from atom_causal_world_schema import canonical_hash

DEFAULT_FORGE = Path("primitive_forge_outputs/atom_primitive_graph.json")
DEFAULT_EVIDENCE = Path(
    "causal_world_outputs/atom_causal_world_evidence.jsonl"
)
DEFAULT_MODEL = Path("causal_world_outputs/atom_causal_world_model.json")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _value(record: ExperienceRecord, role: str) -> str:
    values = record.feature_values(role)
    if len(values) != 1:
        raise ValueError(
            f"{record.experience_id} has invalid {role} cardinality"
        )
    return values[0]


def _choose_target(
    corpus: ExperienceCorpus,
) -> tuple[ExperienceRecord, ExperienceRecord]:
    groups: dict[
        tuple[str, str, str],
        list[ExperienceRecord],
    ] = defaultdict(list)
    for record in corpus.laws:
        if _value(record, "status") != "crystallized":
            continue
        groups[
            (
                _value(record, "domain"),
                _value(record, "cause"),
                _value(record, "direction"),
            )
        ].append(record)
    candidates = [
        (key, sorted(records, key=lambda item: item.experience_id))
        for key, records in groups.items()
        if len(records) >= 2
        and len({_value(item, "effect") for item in records}) >= 2
    ]
    if not candidates:
        raise ValueError("corpus has no crystallized competitor group")
    _, records = sorted(
        candidates,
        key=lambda item: (-len(item[1]), item[0]),
    )[0]
    target = records[0]
    competitor = next(
        record
        for record in records[1:]
        if _value(record, "effect") != _value(target, "effect")
    )
    return target, competitor


def _broad_query(target: ExperienceRecord) -> str:
    features: list[tuple[str, str, bool]] = [
        ("kind", "law", True),
        ("domain", _value(target, "domain"), True),
        ("cause", _value(target, "cause"), True),
        ("direction", _value(target, "direction"), True),
        ("effect", _value(target, "effect"), False),
    ]
    features.extend(
        (role, value, False)
        for role, value in target.features
        if role.startswith("root/") or role == "context"
    )
    return build_experience_query(
        query_id=f"experience-outcome:{target.experience_id}",
        features=features,
        minimum_support=4,
        minimum_coverage_per_million=400_000,
        limit=64,
    )


def _unknown_query() -> str:
    return build_experience_query(
        query_id="experience-unobserved-topology",
        features=[
            ("kind", "law", True),
            ("domain", "unobserved-domain", True),
            ("cause", "unobserved-cause", True),
            ("effect", "unobserved-effect", True),
            ("direction", "+1", True),
        ],
        minimum_support=5,
        minimum_coverage_per_million=1_000_000,
        limit=16,
    )


def _score(report: Mapping[str, Any], identity: str) -> int:
    for hit in report["hits"]:
        if hit["experience_id"] == identity:
            return int(hit["score"])
    return 0


def _audit_role_rejected() -> bool:
    try:
        build_experience_query(
            query_id="forbidden-audit-query",
            features=[("provenance/hash", "0" * 64, True)],
            minimum_support=1,
        )
    except ValueError:
        return True
    return False


def _run_checks(
    *,
    corpus: ExperienceCorpus,
    forge_import: Mapping[str, Any],
    observation_ingest: Mapping[str, Any],
    law_ingest: Mapping[str, Any],
    duplicate_ingest: Mapping[str, Any],
    inventory_after_observations: Mapping[str, Any],
    inventory: Mapping[str, Any],
    target: ExperienceRecord,
    competitor: ExperienceRecord,
    before: Mapping[str, Any],
    broad: Mapping[str, Any],
    after_wrong: Mapping[str, Any],
    persisted: Mapping[str, Any],
    wrong_feedback: Mapping[str, Any],
    correct_feedback: Mapping[str, Any],
    unknown: Mapping[str, Any],
    read_only_recall: bool,
    knowledge: Mapping[str, Any],
    context: list[dict[str, Any]],
) -> dict[str, bool]:
    observation_labels = {
        _value(record, "source/id") for record in corpus.observations
    }
    inventory_ids = {
        str(item["experience_id"]) for item in inventory["experiences"]
    }
    domain_counts = Counter(
        str(item["domain"]) for item in inventory["experiences"]
    )
    supported_edges = [
        edge
        for edge in knowledge["edges"]
        if edge["relation"] == "supported_by"
    ]
    wrong_adjustments = wrong_feedback["adjustments"]
    return {
        "forge_and_experience_catalogs_coexist": (
            forge_import["glyph_count"] == 69
            and forge_import["root_count"] == 7
            and observation_ingest["total_experiences"]
            == len(corpus.observations)
        ),
        "all_world_experiences_are_durable": (
            len(inventory_ids) == len(corpus.all_records) == 2699
            and inventory_ids
            == {record.experience_id for record in corpus.all_records}
        ),
        "repeated_evidence_labels_are_immutable_revisions": (
            len(corpus.observations) == 2304
            and len(observation_labels) == 845
        ),
        "observations_and_laws_append_in_separate_cells": (
            observation_ingest["committed"] is True
            and law_ingest["committed"] is True
            and len(inventory_after_observations["batches"]) == 1
            and len(inventory["batches"]) == 2
            and law_ingest["snapshot_sequence"]
            > observation_ingest["snapshot_sequence"]
        ),
        "duplicate_batch_is_idempotent": (
            duplicate_ingest["committed"] is False
            and duplicate_ingest["snapshot_sequence"]
            == law_ingest["snapshot_sequence"]
        ),
        "recall_is_read_only": read_only_recall,
        "exact_structural_recall_recovers_law": (
            before["answerable"] is True
            and before["hits"][0]["experience_id"]
            == target.experience_id
            and before["hits"][0]["coverage_per_million"] == 1_000_000
        ),
        "broad_recall_exposes_competing_experience": (
            broad["answerable"] is True
            and any(
                item["experience_id"] == competitor.experience_id
                for item in broad["hits"]
            )
        ),
        "law_evidence_links_are_graph_native": (
            bool(supported_edges)
            and all(
                edge["target"].removeprefix("experience:")
                in inventory_ids
                for edge in supported_edges
            )
        ),
        "wrong_outcome_adjusts_both_paths": (
            wrong_feedback["prediction_correct"] is False
            and any(
                item["experience_id"] == target.experience_id
                and item["polarity"] == "strengthen"
                for item in wrong_adjustments
            )
            and any(
                item["experience_id"] == competitor.experience_id
                and item["polarity"] == "weaken"
                for item in wrong_adjustments
            )
        ),
        "correct_outcome_strengthens_target": (
            correct_feedback["prediction_correct"] is True
            and all(
                item["experience_id"] == target.experience_id
                and item["polarity"] == "strengthen"
                for item in correct_feedback["adjustments"]
            )
        ),
        "learning_survives_process_reopen": (
            _score(persisted, target.experience_id)
            > _score(before, target.experience_id)
            and persisted["snapshot_sequence"]
            > after_wrong["snapshot_sequence"]
        ),
        "unknown_required_structure_abstains": (
            unknown["insufficient_evidence"] is True
            and unknown["hits"] == []
        ),
        "audit_metadata_cannot_enter_recall": _audit_role_rejected(),
        "all_eight_world_domains_are_present": (
            set(domain_counts)
            == {
                "agent",
                "biological",
                "chemical",
                "ecological",
                "language",
                "physical",
                "social",
                "symbolic",
            }
        ),
        "wiki_and_rag_use_the_durable_catalog": (
            knowledge["catalog_identity"] == inventory["catalog_identity"]
            and knowledge["inventory_hash"] == canonical_hash(inventory)
            and bool(context)
            and context[0]["experience_id"] == target.experience_id
        ),
    }


def run_causal_experience_experiment(
    output_dir: Path,
    *,
    forge_path: Path = DEFAULT_FORGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
    model_path: Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "atom_causal_experience.atomdb"
    if store_path.exists():
        raise FileExistsError(
            "causal-experience experiment refuses to overwrite its store"
        )
    binary = build_release_binary()
    forge = load_forge(Path(forge_path))
    corpus = load_experience_corpus(
        Path(evidence_path),
        Path(model_path),
    )
    target, competitor = _choose_target(corpus)
    glyph_client = CausalMemoryClient(store_path, binary)
    experience_client = ExperienceMemoryClient(store_path, binary)
    forge_import = glyph_client.import_forge(forge)

    observation_ingest = experience_client.ingest(
        corpus.observations,
        source_artifact_hash=corpus.evidence_hash,
        batch_id="causal-world-observation-revisions",
    )
    inventory_after_observations = experience_client.inventory()
    law_ingest = experience_client.ingest(
        corpus.laws,
        source_artifact_hash=corpus.model_hash,
        batch_id="causal-world-consolidated-laws",
    )
    duplicate_ingest = experience_client.ingest(
        corpus.laws,
        source_artifact_hash=corpus.model_hash,
        batch_id="causal-world-consolidated-laws",
    )

    exact_query = build_query_for_record(target)
    broad_query = _broad_query(target)
    before_hash = _sha256(store_path)
    before = experience_client.recall(exact_query)
    broad = experience_client.recall(broad_query)
    read_only_recall = before_hash == _sha256(store_path)
    wrong_feedback = experience_client.observe_outcome(
        broad_query,
        expected_experience=target.experience_id,
        selected_experience=competitor.experience_id,
    )
    after_wrong = experience_client.recall(exact_query)
    correct_feedback = experience_client.observe_outcome(
        broad_query,
        expected_experience=target.experience_id,
        selected_experience=target.experience_id,
    )
    persisted_client = ExperienceMemoryClient(store_path, binary)
    persisted = persisted_client.recall(exact_query)
    unknown_query = _unknown_query()
    unknown = persisted_client.recall(unknown_query)
    inventory = persisted_client.inventory()

    wiki = CausalExperienceWikiGraph(
        persisted_client,
        corpus,
        inventory,
    )
    knowledge = wiki.manifest()
    validate_causal_experience_knowledge(
        knowledge,
        inventory=inventory,
        corpus=corpus,
    )
    context = retrieve_causal_experience_context(wiki, exact_query)
    checks = _run_checks(
        corpus=corpus,
        forge_import=forge_import,
        observation_ingest=observation_ingest,
        law_ingest=law_ingest,
        duplicate_ingest=duplicate_ingest,
        inventory_after_observations=inventory_after_observations,
        inventory=inventory,
        target=target,
        competitor=competitor,
        before=before,
        broad=broad,
        after_wrong=after_wrong,
        persisted=persisted,
        wrong_feedback=wrong_feedback,
        correct_feedback=correct_feedback,
        unknown=unknown,
        read_only_recall=read_only_recall,
        knowledge=knowledge,
        context=context,
    )
    report_core: dict[str, Any] = {
        "schema": 1,
        "runtime": "atom-causal-experience-experiment-v1",
        "memory_runtime": EXPERIENCE_MEMORY_RUNTIME,
        "passed": all(checks.values()),
        "claim_scope": (
            "Full saved causal-world experience ingestion, append-only "
            "structural recall, and durable outcome feedback."
        ),
        "source": {
            "forge_graph_hash": forge.graph_hash,
            "evidence_hash": corpus.evidence_hash,
            "model_hash": corpus.model_hash,
            "observation_revisions": len(corpus.observations),
            "laws": len(corpus.laws),
        },
        "forge": forge_import,
        "experience": {
            "catalog_identity": inventory["catalog_identity"],
            "observation_ingest": observation_ingest,
            "law_ingest": law_ingest,
            "duplicate_ingest": duplicate_ingest,
            "batch_count": len(inventory["batches"]),
            "experience_count": len(inventory["experiences"]),
        },
        "recall": {
            "target_experience": target.experience_id,
            "competing_experience": competitor.experience_id,
            "before_feedback": before,
            "broad_context": broad,
            "after_wrong_feedback": after_wrong,
            "persisted": persisted,
            "unknown": unknown,
            "target_score_increase": (
                _score(persisted, target.experience_id)
                - _score(before, target.experience_id)
            ),
        },
        "learning": {
            "wrong_outcome": wrong_feedback,
            "correct_outcome": correct_feedback,
        },
        "knowledge": {
            "wiki_runtime": CAUSAL_EXPERIENCE_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_EXPERIENCE_RAG_RUNTIME,
            "knowledge_hash": knowledge["knowledge_hash"],
            "node_count": knowledge["node_count"],
            "edge_count": knowledge["edge_count"],
            "context_count": len(context),
        },
        "side_view_contract": {
            "runtime": CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": (
                "render_causal_experience_artifact"
            ),
            "placement": "side",
            "user_visible": True,
        },
        "checks": checks,
    }
    report = {
        **report_core,
        "report_hash": canonical_hash(report_core),
    }
    workflow_core = {
        "schema": 1,
        "runtime": "atom-causal-experience-workflow-v1",
        "report_hash": report["report_hash"],
        "knowledge_hash": knowledge["knowledge_hash"],
        "inventory_hash": canonical_hash(inventory),
        "catalog_identity": inventory["catalog_identity"],
        "store_sha256": _sha256(store_path),
        "binary_sha256": binary_sha256(binary),
        "query_id": before["query_id"],
        "target_experience": target.experience_id,
        "wiki_runtime": CAUSAL_EXPERIENCE_WIKI_RUNTIME,
        "rag_runtime": CAUSAL_EXPERIENCE_RAG_RUNTIME,
        "side_view_runtime": CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME,
    }
    workflow = {
        **workflow_core,
        "workflow_hash": canonical_hash(workflow_core),
    }
    side_view = render_causal_experience_artifact(
        report,
        inventory,
        workflow,
        knowledge,
    )
    _write_json(
        output_dir / "atom_causal_experience_report.json",
        report,
    )
    _write_json(
        output_dir / "atom_causal_experience_inventory.json",
        inventory,
    )
    _write_json(
        output_dir / "atom_causal_experience_knowledge.json",
        knowledge,
    )
    _write_json(
        output_dir / "atom_causal_experience_workflow.json",
        workflow,
    )
    (output_dir / "atom_causal_experience_query.txt").write_text(
        exact_query,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_experience_broad_query.txt").write_text(
        broad_query,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_experience_unknown_query.txt").write_text(
        unknown_query,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_experience_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist the full causal-world experience stream."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_experience_outputs"),
    )
    parser.add_argument("--forge", type=Path, default=DEFAULT_FORGE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    arguments = parser.parse_args()
    report = run_causal_experience_experiment(
        arguments.output_dir,
        forge_path=arguments.forge,
        evidence_path=arguments.evidence,
        model_path=arguments.model,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "observations": report["source"][
                    "observation_revisions"
                ],
                "laws": report["source"]["laws"],
                "experiences": report["experience"][
                    "experience_count"
                ],
                "score_increase": report["recall"][
                    "target_score_increase"
                ],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
