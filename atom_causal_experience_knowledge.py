"""Runtime wiki graph and structural RAG for causal experience memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from atom_causal_experience import (
    ExperienceCorpus,
    ExperienceMemoryClient,
    ExperienceRecord,
)
from atom_causal_world_schema import canonical_hash

CAUSAL_EXPERIENCE_WIKI_RUNTIME = "atom-causal-experience-wiki-v1"
CAUSAL_EXPERIENCE_RAG_RUNTIME = "atom-causal-experience-rag-v1"


@dataclass(frozen=True)
class ExperienceKnowledgeNode:
    node_id: str
    kind: str
    label: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class ExperienceKnowledgeEdge:
    source: str
    relation: str
    target: str


class CausalExperienceWikiGraph:
    """Graph view built from the records actually present in Atom DB."""

    def __init__(
        self,
        client: ExperienceMemoryClient,
        corpus: ExperienceCorpus,
        inventory: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.corpus = corpus
        self.inventory = (
            client.inventory() if inventory is None else dict(inventory)
        )
        records = {
            record.experience_id: record for record in corpus.all_records
        }
        stored = self.inventory.get("experiences")
        if not isinstance(stored, list):
            raise ValueError("experience inventory records are invalid")
        stored_ids = {
            str(item["experience_id"])
            for item in stored
            if isinstance(item, Mapping)
        }
        if stored_ids != set(records):
            raise ValueError(
                "durable experience inventory differs from the source corpus"
            )
        self.records = records
        self.nodes, self.edges = self._build_graph()

    def _build_graph(
        self,
    ) -> tuple[
        tuple[ExperienceKnowledgeNode, ...],
        tuple[ExperienceKnowledgeEdge, ...],
    ]:
        nodes: dict[str, ExperienceKnowledgeNode] = {}
        edges: set[ExperienceKnowledgeEdge] = set()
        catalog = str(self.inventory["catalog_identity"])
        nodes["catalog"] = ExperienceKnowledgeNode(
            node_id="catalog",
            kind="experience_catalog",
            label="Persistent causal experience",
            attributes={
                "catalog_identity": catalog,
                "snapshot_sequence": int(
                    self.inventory["snapshot_sequence"]
                ),
            },
        )
        for batch in self.inventory["batches"]:
            batch_id = str(batch["batch_id"])
            node_id = f"batch:{batch_id}"
            nodes[node_id] = ExperienceKnowledgeNode(
                node_id=node_id,
                kind="experience_batch",
                label=batch_id,
                attributes=dict(batch),
            )
            edges.add(
                ExperienceKnowledgeEdge(
                    source="catalog",
                    relation="contains_batch",
                    target=node_id,
                )
            )

        observation_ids = {
            record.experience_id
            for record in self.corpus.observations
        }
        for identity in sorted(self.records):
            record = self.records[identity]
            values = _feature_map(record)
            kind = values["kind"][0]
            status = values["status"][0]
            domain = values["domain"][0]
            node_id = f"experience:{identity}"
            attributes: dict[str, Any] = {
                "experience_id": identity,
                "kind": kind,
                "status": status,
                "domain": domain,
                "feature_count": len(record.features),
            }
            if values.get("session/id"):
                attributes["session_id"] = values["session/id"][0]
            if values.get("interaction/id"):
                attributes["interaction_id"] = values[
                    "interaction/id"
                ][0]
            nodes[node_id] = ExperienceKnowledgeNode(
                node_id=node_id,
                kind=f"causal_{kind}",
                label=identity,
                attributes=attributes,
            )
            edges.add(
                ExperienceKnowledgeEdge(
                    source="catalog",
                    relation="contains_experience",
                    target=node_id,
                )
            )
            domain_node = f"domain:{domain}"
            nodes.setdefault(
                domain_node,
                ExperienceKnowledgeNode(
                    node_id=domain_node,
                    kind="world_domain",
                    label=domain,
                    attributes={},
                ),
            )
            edges.add(
                ExperienceKnowledgeEdge(
                    source=node_id,
                    relation="occurs_in",
                    target=domain_node,
                )
            )
            for role in ("cause", "effect"):
                feature = values[role][0]
                feature_node = f"feature:{feature}"
                nodes.setdefault(
                    feature_node,
                    ExperienceKnowledgeNode(
                        node_id=feature_node,
                        kind="world_feature",
                        label=feature,
                        attributes={},
                    ),
                )
                edges.add(
                    ExperienceKnowledgeEdge(
                        source=node_id,
                        relation=role,
                        target=feature_node,
                    )
                )
            for role, value in record.features:
                if role.startswith("root/"):
                    root_node = f"root:{value}"
                    nodes.setdefault(
                        root_node,
                        ExperienceKnowledgeNode(
                            node_id=root_node,
                            kind="universe_root",
                            label=value,
                            attributes={},
                        ),
                    )
                    edges.add(
                        ExperienceKnowledgeEdge(
                            source=node_id,
                            relation="composes_with",
                            target=root_node,
                        )
                    )
                elif role.startswith("evidence/") and value in observation_ids:
                    edges.add(
                        ExperienceKnowledgeEdge(
                            source=node_id,
                            relation="supported_by",
                            target=f"experience:{value}",
                        )
                    )
            sessions = values.get("session/id", ())
            interactions = values.get("interaction/id", ())
            if len(sessions) > 1 or len(interactions) > 1:
                raise ValueError("live experience provenance is ambiguous")
            if sessions:
                session_node = f"session:{sessions[0]}"
                nodes.setdefault(
                    session_node,
                    ExperienceKnowledgeNode(
                        node_id=session_node,
                        kind="live_session",
                        label=sessions[0],
                        attributes={},
                    ),
                )
                edges.add(
                    ExperienceKnowledgeEdge(
                        source=node_id,
                        relation="observed_in",
                        target=session_node,
                    )
                )
            authority_kinds = values.get("authority/kind", ())
            authority_ids = values.get("authority/id", ())
            if len(authority_kinds) > 1 or len(authority_ids) > 1:
                raise ValueError("live outcome authority is ambiguous")
            if authority_kinds and authority_ids:
                authority_node = (
                    f"authority:{authority_kinds[0]}:{authority_ids[0]}"
                )
                nodes.setdefault(
                    authority_node,
                    ExperienceKnowledgeNode(
                        node_id=authority_node,
                        kind="outcome_authority",
                        label=authority_ids[0],
                        attributes={"kind": authority_kinds[0]},
                    ),
                )
                edges.add(
                    ExperienceKnowledgeEdge(
                        source=node_id,
                        relation="certified_by",
                        target=authority_node,
                    )
                )
        return (
            tuple(nodes[key] for key in sorted(nodes)),
            tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        edge.source,
                        edge.relation,
                        edge.target,
                    ),
                )
            ),
        )

    def retrieve(self, query_wire: str) -> tuple[dict[str, Any], ...]:
        report = self.client.recall(query_wire)
        contexts: list[dict[str, Any]] = []
        for hit in report["hits"]:
            record = self.records[str(hit["experience_id"])]
            values = _feature_map(record)
            contexts.append(
                {
                    "experience_id": hit["experience_id"],
                    "kind": values["kind"][0],
                    "status": values["status"][0],
                    "domain": values["domain"][0],
                    "cause": values["cause"][0],
                    "effect": values["effect"][0],
                    "direction": values["direction"][0],
                    "score": hit["score"],
                    "coverage_per_million": hit[
                        "coverage_per_million"
                    ],
                    "motifs": list(hit["motifs"]),
                }
            )
        return tuple(contexts)

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema": 1,
            "wiki_runtime": CAUSAL_EXPERIENCE_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_EXPERIENCE_RAG_RUNTIME,
            "catalog_identity": self.inventory["catalog_identity"],
            "snapshot_sequence": self.inventory["snapshot_sequence"],
            "inventory_hash": canonical_hash(self.inventory),
            "evidence_hash": self.corpus.evidence_hash,
            "model_hash": self.corpus.model_hash,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "label": node.label,
                    "attributes": dict(node.attributes),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                }
                for edge in self.edges
            ],
        }
        return {**core, "knowledge_hash": canonical_hash(core)}


def _feature_map(
    record: ExperienceRecord,
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for role, value in record.features:
        values.setdefault(role, []).append(value)
    return {
        role: tuple(sorted(items))
        for role, items in values.items()
    }


def retrieve_causal_experience_context(
    graph: CausalExperienceWikiGraph,
    query_wire: str,
) -> list[dict[str, Any]]:
    return list(graph.retrieve(query_wire))


def validate_causal_experience_knowledge(
    manifest: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    corpus: ExperienceCorpus,
) -> None:
    expected = {
        "schema",
        "wiki_runtime",
        "rag_runtime",
        "catalog_identity",
        "snapshot_sequence",
        "inventory_hash",
        "evidence_hash",
        "model_hash",
        "node_count",
        "edge_count",
        "nodes",
        "edges",
        "knowledge_hash",
    }
    if set(manifest) != expected:
        raise ValueError("experience knowledge manifest fields are invalid")
    core = {
        key: manifest[key] for key in sorted(expected - {"knowledge_hash"})
    }
    if manifest["knowledge_hash"] != canonical_hash(core):
        raise ValueError("experience knowledge hash mismatch")
    if manifest["wiki_runtime"] != CAUSAL_EXPERIENCE_WIKI_RUNTIME:
        raise ValueError("experience wiki runtime is invalid")
    if manifest["rag_runtime"] != CAUSAL_EXPERIENCE_RAG_RUNTIME:
        raise ValueError("experience RAG runtime is invalid")
    if manifest["catalog_identity"] != inventory["catalog_identity"]:
        raise ValueError("experience knowledge catalog is detached")
    if manifest["inventory_hash"] != canonical_hash(inventory):
        raise ValueError("experience knowledge inventory hash mismatch")
    if manifest["evidence_hash"] != corpus.evidence_hash:
        raise ValueError("experience knowledge evidence hash mismatch")
    if manifest["model_hash"] != corpus.model_hash:
        raise ValueError("experience knowledge model hash mismatch")
