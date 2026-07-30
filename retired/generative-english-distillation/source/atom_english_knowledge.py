"""Runtime wiki graph and graph-native RAG for Atom English artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from atom_english_core import (
    ATOM_ROOT_PRIMITIVES,
    AtomEnglishConfig,
    atom_english_architecture_manifest,
)
from atom_english_data import EnglishCurriculum

ATOM_ENGLISH_WIKI_RUNTIME = "atom-english-wiki-graph-v1"
ATOM_ENGLISH_RAG_RUNTIME = "atom-english-graph-rag-v1"


@dataclass(frozen=True)
class EnglishKnowledgeNode:
    node_id: str
    kind: str
    label: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class EnglishKnowledgeEdge:
    source: str
    relation: str
    target: str


def _words(value: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z0-9_]+", value)
        if len(match.group(0)) > 1
    }


def build_english_knowledge_graph(
    config: AtomEnglishConfig,
    curriculum: EnglishCurriculum,
    *,
    run_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: dict[str, EnglishKnowledgeNode] = {}
    edges: set[EnglishKnowledgeEdge] = set()

    nodes["architecture"] = EnglishKnowledgeNode(
        node_id="architecture",
        kind="language_architecture",
        label="Generative Atom causal graph",
        attributes=atom_english_architecture_manifest(config),
    )
    for primitive in ATOM_ROOT_PRIMITIVES:
        node_id = f"primitive:{primitive}"
        nodes[node_id] = EnglishKnowledgeNode(
            node_id=node_id,
            kind="root_primitive",
            label=primitive.replace("_", " "),
            attributes={},
        )
        edges.add(
            EnglishKnowledgeEdge(
                source="architecture",
                relation="composes_from",
                target=node_id,
            )
        )

    nodes["tokenizer:foundation"] = EnglishKnowledgeNode(
        node_id="tokenizer:foundation",
        kind="tokenizer",
        label=curriculum.foundation_tokenizer_id,
        attributes={
            "stage": "foundation",
            "revision": curriculum.foundation_tokenizer_revision,
        },
    )
    nodes["tokenizer:dialogue"] = EnglishKnowledgeNode(
        node_id="tokenizer:dialogue",
        kind="tokenizer",
        label=curriculum.tokenizer_id,
        attributes={
            "stage": "dialogue",
            "revision": curriculum.tokenizer_revision,
        },
    )
    for tokenizer_node in ("tokenizer:foundation", "tokenizer:dialogue"):
        edges.add(
            EnglishKnowledgeEdge(
                source="architecture",
                relation="uses_tokenizer",
                target=tokenizer_node,
            )
        )

    for stage in ("foundation", "dialogue", "evaluation"):
        stage_id = f"stage:{stage}"
        nodes[stage_id] = EnglishKnowledgeNode(
            node_id=stage_id,
            kind="curriculum_stage",
            label=stage,
            attributes={},
        )
        edges.add(
            EnglishKnowledgeEdge(
                source="architecture",
                relation="learns_through",
                target=stage_id,
            )
        )
    for source in curriculum.sources:
        node_id = f"source:{source.source_id}"
        nodes[node_id] = EnglishKnowledgeNode(
            node_id=node_id,
            kind="corpus_source",
            label=source.dataset_id,
            attributes={
                "source_id": source.source_id,
                "stage": source.stage,
                "config": source.dataset_config,
                "split": source.split,
                "revision": source.revision,
                "license": source.license_name,
                "content_kind": source.content_kind,
            },
        )
        edges.add(
            EnglishKnowledgeEdge(
                source=f"stage:{source.stage}",
                relation="reads",
                target=node_id,
            )
        )

    teacher_nodes = (
        (
            "teacher:foundation",
            curriculum.base_teacher_id,
            curriculum.base_teacher_revision,
            "foundation",
        ),
        (
            "teacher:dialogue",
            curriculum.dialogue_teacher_id,
            curriculum.dialogue_teacher_revision,
            "dialogue",
        ),
    )
    for node_id, model_id, revision, stage in teacher_nodes:
        nodes[node_id] = EnglishKnowledgeNode(
            node_id=node_id,
            kind="distillation_teacher",
            label=model_id,
            attributes={"stage": stage, "revision": revision},
        )
        edges.add(
            EnglishKnowledgeEdge(
                source=f"stage:{stage}",
                relation="distills_from",
                target=node_id,
            )
        )

    if run_summary is not None:
        mode = str(run_summary.get("mode", "unknown"))
        nodes["run"] = EnglishKnowledgeNode(
            node_id="run",
            kind="runtime_run",
            label=f"Atom English {mode}",
            attributes={
                key: value
                for key, value in run_summary.items()
                if key not in {"samples", "training_report", "evaluation"}
            },
        )
        edges.add(
            EnglishKnowledgeEdge(
                source="architecture",
                relation="executed_as",
                target="run",
            )
        )
        for index, sample in enumerate(run_summary.get("samples", [])):
            node_id = f"sample:{index}"
            nodes[node_id] = EnglishKnowledgeNode(
                node_id=node_id,
                kind="generated_artifact",
                label=str(sample.get("prompt", f"sample {index}")),
                attributes=dict(sample),
            )
            edges.add(
                EnglishKnowledgeEdge(
                    source="run",
                    relation="produced",
                    target=node_id,
                )
            )

    ordered_nodes = [
        {
            "node_id": node.node_id,
            "kind": node.kind,
            "label": node.label,
            "attributes": dict(node.attributes),
        }
        for node in sorted(nodes.values(), key=lambda item: item.node_id)
    ]
    ordered_edges = [
        {
            "source": edge.source,
            "relation": edge.relation,
            "target": edge.target,
        }
        for edge in sorted(
            edges,
            key=lambda item: (item.source, item.relation, item.target),
        )
    ]
    return {
        "schema_version": 1,
        "wiki_runtime": ATOM_ENGLISH_WIKI_RUNTIME,
        "rag_runtime": ATOM_ENGLISH_RAG_RUNTIME,
        "nodes": ordered_nodes,
        "edges": ordered_edges,
    }


def retrieve_english_knowledge(
    graph: Mapping[str, Any],
    query: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve matching nodes, then expose their direct graph relations."""

    validate_english_knowledge_graph(graph)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("RAG query must be non-empty text")
    if not 1 <= limit <= 64:
        raise ValueError("RAG limit must be inside [1, 64]")
    query_words = _words(query)
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    adjacency: dict[str, list[dict[str, str]]] = {node_id: [] for node_id in nodes}
    for edge in graph["edges"]:
        adjacency[edge["source"]].append(
            {
                "direction": "out",
                "relation": edge["relation"],
                "node_id": edge["target"],
            }
        )
        adjacency[edge["target"]].append(
            {
                "direction": "in",
                "relation": edge["relation"],
                "node_id": edge["source"],
            }
        )
    ranked: list[tuple[int, str]] = []
    for node_id, node in nodes.items():
        searchable = " ".join(
            (
                node["kind"],
                node["label"],
                json.dumps(node["attributes"], sort_keys=True),
            )
        )
        overlap = len(query_words & _words(searchable))
        if overlap:
            ranked.append((overlap, node_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    contexts: list[dict[str, Any]] = []
    for score, node_id in ranked[:limit]:
        contexts.append(
            {
                "score": score,
                "node": nodes[node_id],
                "relations": sorted(
                    adjacency[node_id],
                    key=lambda item: (
                        item["relation"],
                        item["direction"],
                        item["node_id"],
                    ),
                ),
            }
        )
    return contexts


def validate_english_knowledge_graph(graph: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "wiki_runtime",
        "rag_runtime",
        "nodes",
        "edges",
    }
    if set(graph) != expected:
        raise ValueError("English knowledge graph fields are invalid")
    if graph["schema_version"] != 1:
        raise ValueError("unsupported English knowledge graph schema")
    if graph["wiki_runtime"] != ATOM_ENGLISH_WIKI_RUNTIME:
        raise ValueError("English wiki runtime is invalid")
    if graph["rag_runtime"] != ATOM_ENGLISH_RAG_RUNTIME:
        raise ValueError("English RAG runtime is invalid")
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not isinstance(nodes, Sequence) or not isinstance(edges, Sequence):
        raise ValueError("English knowledge collections are invalid")
    node_ids = [node["node_id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("English knowledge node IDs are not unique")
    node_id_set = set(node_ids)
    for edge in edges:
        if edge["source"] not in node_id_set:
            raise ValueError("English knowledge edge source is missing")
        if edge["target"] not in node_id_set:
            raise ValueError("English knowledge edge target is missing")
