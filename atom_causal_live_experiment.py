"""Run two trusted live interactions through the causal Atom memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from atom_causal_experience import (
    ExperienceMemoryClient,
    ExperienceRecord,
    load_experience_corpus,
)
from atom_causal_experience_experiment import (
    DEFAULT_EVIDENCE,
    DEFAULT_FORGE,
    DEFAULT_MODEL,
    run_causal_experience_experiment,
)
from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
    CausalExperienceWikiGraph,
    retrieve_causal_experience_context,
    validate_causal_experience_knowledge,
)
from atom_causal_live import (
    CAUSAL_LIVE_RUNTIME,
    LiveCausalEvent,
    LiveCausalRuntime,
    build_live_outcome_query,
    build_live_prediction_query,
    extend_corpus_from_inventory,
)
from atom_causal_live_side_view import (
    CAUSAL_LIVE_SIDE_VIEW_RUNTIME,
    render_causal_live_artifact,
)
from atom_causal_memory import RELEASE_BINARY, binary_sha256
from atom_causal_world_schema import canonical_hash


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _values(
    record: ExperienceRecord,
    role: str,
) -> tuple[str, ...]:
    return tuple(
        value for name, value in record.features if name == role
    )


def _one(record: ExperienceRecord, role: str) -> str:
    values = _values(record, role)
    if len(values) != 1:
        raise ValueError(
            f"{record.experience_id} has invalid {role} cardinality"
        )
    return values[0]


def _choose_live_situation(
    records: tuple[ExperienceRecord, ...],
    novel_effect: str,
) -> ExperienceRecord:
    candidates = [
        record
        for record in records
        if _one(record, "status") == "crystallized"
        and _one(record, "effect") != novel_effect
        and _values(record, "context")
        and any(role.startswith("root/") for role, _ in record.features)
    ]
    if not candidates:
        raise ValueError("causal corpus has no live-learning situation")
    return sorted(candidates, key=lambda item: item.experience_id)[0]


def _event_manifest(
    *,
    session_id: str,
    interaction_id: str,
    situation: ExperienceRecord,
    effect: str,
    evidence_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roots = [
        value
        for role, value in sorted(situation.features)
        if role.startswith("root/")
    ]
    direction = int(_one(situation, "direction"))
    evidence = {
        "schema": 1,
        "authority_kind": "simulator",
        "authority_id": "deterministic-live-world-v1",
        "session_id": session_id,
        "interaction_id": interaction_id,
        "domain": _one(situation, "domain"),
        "cause": _one(situation, "cause"),
        "effect": effect,
        "direction": direction,
        "trace": {
            "source": "deterministic-live-world",
            "label": evidence_label,
            "result": "observed",
        },
    }
    event = {
        "schema": 1,
        "session_id": session_id,
        "interaction_id": interaction_id,
        "domain": _one(situation, "domain"),
        "cause": _one(situation, "cause"),
        "context": list(_values(situation, "context")),
        "direction": direction,
        "atom_program": roots,
        "outcome": {
            "effect": effect,
            "delay_ticks": int(
                _one(situation, "delay").removeprefix("ticks:")
            ),
            "magnitude": 1.0,
            "invariant_error": 0.0,
            "authority_kind": "simulator",
            "authority_id": "deterministic-live-world-v1",
            "evidence_hash": canonical_hash(evidence),
        },
    }
    return event, evidence


def _adjustment_present(
    cycle: Mapping[str, Any],
    *,
    identity: str,
    polarity: str,
) -> bool:
    feedback = cycle.get("feedback")
    return bool(
        isinstance(feedback, Mapping)
        and any(
            item["experience_id"] == identity
            and item["polarity"] == polarity
            for item in feedback["adjustments"]
        )
    )


def run_causal_live_experiment(
    output_dir: Path,
    *,
    forge_path: Path = DEFAULT_FORGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
    model_path: Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = output_dir / "base_experience"
    if base_dir.exists():
        raise FileExistsError(
            "live causal experiment refuses to overwrite its base store"
        )
    base_report = run_causal_experience_experiment(
        base_dir,
        forge_path=Path(forge_path),
        evidence_path=Path(evidence_path),
        model_path=Path(model_path),
    )
    store_path = base_dir / "atom_causal_experience.atomdb"
    base_corpus = load_experience_corpus(
        Path(evidence_path),
        Path(model_path),
    )
    novel_effect = "live_phase_coherence"
    if any(
        _one(record, "effect") == novel_effect
        for record in base_corpus.all_records
    ):
        raise ValueError("live experiment effect is not novel")
    situation = _choose_live_situation(
        base_corpus.laws,
        novel_effect,
    )
    first_manifest, first_evidence = _event_manifest(
        session_id="live-session-alpha",
        interaction_id="novel-outcome-1",
        situation=situation,
        effect=novel_effect,
        evidence_label="alpha-trace",
    )
    second_manifest, second_evidence = _event_manifest(
        session_id="live-session-beta",
        interaction_id="novel-outcome-2",
        situation=situation,
        effect=novel_effect,
        evidence_label="beta-trace",
    )
    first_event = LiveCausalEvent.from_manifest(first_manifest)
    second_event = LiveCausalEvent.from_manifest(second_manifest)
    runtime = LiveCausalRuntime(
        ExperienceMemoryClient(store_path, RELEASE_BINARY),
        authorities={"deterministic-live-world-v1": "simulator"},
    )
    first = runtime.observe(first_event, first_evidence)
    replay_hash_before = _sha256(store_path)
    replay = runtime.observe(first_event, first_evidence)
    replay_hash_after = _sha256(store_path)
    second = runtime.observe(second_event, second_evidence)

    reopened = ExperienceMemoryClient(store_path, RELEASE_BINARY)
    inventory = reopened.inventory()
    extended_corpus = extend_corpus_from_inventory(
        base_corpus,
        inventory,
    )
    wiki = CausalExperienceWikiGraph(
        reopened,
        extended_corpus,
        inventory,
    )
    knowledge = wiki.manifest()
    validate_causal_experience_knowledge(
        knowledge,
        inventory=inventory,
        corpus=extended_corpus,
    )
    outcome_query = build_live_outcome_query(second_event)
    read_only_hash_before = _sha256(store_path)
    persisted_recall = reopened.recall(outcome_query)
    rag_context = retrieve_causal_experience_context(
        wiki,
        outcome_query,
    )
    read_only_hash_after = _sha256(store_path)
    live_session_nodes = [
        node
        for node in knowledge["nodes"]
        if node["kind"] == "live_session"
    ]
    observed_in_edges = [
        edge
        for edge in knowledge["edges"]
        if edge["relation"] == "observed_in"
    ]
    authority_edges = [
        edge
        for edge in knowledge["edges"]
        if edge["relation"] == "certified_by"
    ]
    checks = {
        "base_world_is_present": (
            base_report["passed"] is True
            and len(base_corpus.all_records) == 2699
        ),
        "first_novel_outcome_is_appended": (
            first["ingest"]["committed"] is True
            and first["experience_count"] == 2700
        ),
        "first_prediction_is_corrected_bidirectionally": (
            first["prediction_correct"] is False
            and first["feedback"]["committed"] is True
            and _adjustment_present(
                first,
                identity=first_event.experience_id,
                polarity="strengthen",
            )
            and _adjustment_present(
                first,
                identity=str(first["selected_experience"]),
                polarity="weaken",
            )
        ),
        "exact_replay_is_mutation_free": (
            replay["replayed"] is True
            and replay["ingest"]["committed"] is False
            and replay["feedback"]["committed"] is False
            and replay_hash_before == replay_hash_after
            and replay["store_hash_before"] == replay["store_hash_after"]
        ),
        "second_session_uses_the_first_experience": (
            second["selected_experience"]
            == first_event.experience_id
            and second["selected_effect"] == novel_effect
            and second["prediction_correct"] is True
        ),
        "correct_prediction_is_strengthened_once": (
            second["feedback"]["committed"] is True
            and second["feedback"]["prediction_correct"] is True
            and _adjustment_present(
                second,
                identity=first_event.experience_id,
                polarity="strengthen",
            )
        ),
        "live_learning_survives_process_reopen": (
            persisted_recall["answerable"] is True
            and persisted_recall["hits"][0]["experience_id"]
            == first_event.experience_id
        ),
        "two_sessions_are_durable": (
            len(inventory["experiences"]) == 2701
            and len(inventory["batches"]) == 4
            and len(live_session_nodes) == 2
            and len(observed_in_edges) == 2
            and len(authority_edges) == 2
        ),
        "wiki_and_rag_include_live_experience": (
            any(
                item["experience_id"]
                == first_event.experience_id
                for item in rag_context
            )
            and any(
                node["node_id"]
                == f"experience:{first_event.experience_id}"
                for node in knowledge["nodes"]
            )
        ),
        "recall_remains_read_only": (
            read_only_hash_before == read_only_hash_after
        ),
        "trusted_outcome_provenance_is_durable": any(
            feature["role"] == "authority/kind"
            and feature["value"] == "simulator"
            for item in inventory["experiences"]
            if item["experience_id"] == first_event.experience_id
            for feature in item["features"]
        ),
    }
    report_core: dict[str, Any] = {
        "schema": 1,
        "runtime": "atom-causal-live-experiment-v1",
        "live_runtime": CAUSAL_LIVE_RUNTIME,
        "passed": all(checks.values()),
        "claim_scope": (
            "Trusted live outcomes append immutable observations, "
            "change later structural prediction, and replay idempotently."
        ),
        "base_report_hash": base_report["report_hash"],
        "source": {
            "forge": str(Path(forge_path)),
            "evidence": str(Path(evidence_path)),
            "model": str(Path(model_path)),
            "base_experiences": len(base_corpus.all_records),
        },
        "situation": {
            "source_law": situation.experience_id,
            "domain": _one(situation, "domain"),
            "cause": _one(situation, "cause"),
            "previous_effect": _one(situation, "effect"),
            "novel_effect": novel_effect,
        },
        "events": {
            "first": first_event.manifest(),
            "first_evidence": first_evidence,
            "first_hash": first_event.event_hash,
            "second": second_event.manifest(),
            "second_evidence": second_evidence,
            "second_hash": second_event.event_hash,
        },
        "cycles": {
            "first_novel_outcome": first,
            "replay": replay,
            "second_outcome": second,
        },
        "persisted_recall": persisted_recall,
        "rag_context": rag_context,
        "rag_context_count": len(rag_context),
        "live_session_count": len(live_session_nodes),
        "experience_count": len(inventory["experiences"]),
        "batch_count": len(inventory["batches"]),
        "catalog_identity": inventory["catalog_identity"],
        "snapshot_sequence": inventory["snapshot_sequence"],
        "store_sha256": _sha256(store_path),
        "knowledge": {
            "wiki_runtime": CAUSAL_EXPERIENCE_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_EXPERIENCE_RAG_RUNTIME,
            "knowledge_hash": knowledge["knowledge_hash"],
            "node_count": knowledge["node_count"],
            "edge_count": knowledge["edge_count"],
        },
        "side_view_contract": {
            "runtime": CAUSAL_LIVE_SIDE_VIEW_RUNTIME,
            "artifact_binding_marker": "render_causal_live_artifact",
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
        "runtime": "atom-causal-live-workflow-v1",
        "report_hash": report["report_hash"],
        "inventory_hash": canonical_hash(inventory),
        "knowledge_hash": knowledge["knowledge_hash"],
        "store_sha256": report["store_sha256"],
        "binary_sha256": binary_sha256(RELEASE_BINARY),
        "catalog_identity": inventory["catalog_identity"],
        "first_event_hash": first_event.event_hash,
        "second_event_hash": second_event.event_hash,
        "wiki_runtime": CAUSAL_EXPERIENCE_WIKI_RUNTIME,
        "rag_runtime": CAUSAL_EXPERIENCE_RAG_RUNTIME,
        "side_view_runtime": CAUSAL_LIVE_SIDE_VIEW_RUNTIME,
    }
    workflow = {
        **workflow_core,
        "workflow_hash": canonical_hash(workflow_core),
    }
    side_view = render_causal_live_artifact(
        report,
        inventory,
        workflow,
        knowledge,
    )
    _write_json(output_dir / "atom_causal_live_report.json", report)
    _write_json(
        output_dir / "atom_causal_live_inventory.json",
        inventory,
    )
    _write_json(
        output_dir / "atom_causal_live_knowledge.json",
        knowledge,
    )
    _write_json(
        output_dir / "atom_causal_live_workflow.json",
        workflow,
    )
    _write_json(
        output_dir / "atom_causal_live_first_event.json",
        first_event.manifest(),
    )
    _write_json(
        output_dir / "atom_causal_live_second_event.json",
        second_event.manifest(),
    )
    _write_json(
        output_dir / "atom_causal_live_first_evidence.json",
        first_evidence,
    )
    _write_json(
        output_dir / "atom_causal_live_second_evidence.json",
        second_evidence,
    )
    _write_json(
        output_dir / "atom_causal_live_first_cycle.json",
        first,
    )
    _write_json(
        output_dir / "atom_causal_live_replay_cycle.json",
        replay,
    )
    _write_json(
        output_dir / "atom_causal_live_second_cycle.json",
        second,
    )
    (
        output_dir / "atom_causal_live_prediction_query.txt"
    ).write_text(
        build_live_prediction_query(second_event),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_live_outcome_query.txt").write_text(
        outcome_query,
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "atom_causal_live_side_view.html").write_text(
        side_view,
        encoding="utf-8",
        newline="\n",
    )
    if not report["passed"]:
        failed = [
            name for name, value in checks.items() if not value
        ]
        raise RuntimeError(
            "live causal experiment failed: " + ", ".join(failed)
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run trusted live causal learning"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("causal_live_outputs"),
    )
    parser.add_argument(
        "--forge",
        type=Path,
        default=DEFAULT_FORGE,
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    arguments = parser.parse_args()
    report = run_causal_live_experiment(
        arguments.output_dir,
        forge_path=arguments.forge,
        evidence_path=arguments.evidence,
        model_path=arguments.model,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "experiences": report["experience_count"],
                "batches": report["batch_count"],
                "first_prediction_correct": report["cycles"][
                    "first_novel_outcome"
                ]["prediction_correct"],
                "second_prediction_correct": report["cycles"][
                    "second_outcome"
                ]["prediction_correct"],
                "replayed": report["cycles"]["replay"]["replayed"],
                "report_hash": report["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
