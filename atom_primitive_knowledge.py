"""Dynamic wiki graph and graph-first RAG for the Primitive Forge artifact."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash
from atom_primitive_forge import PrimitiveForge, PrimitiveRecord


PRIMITIVE_WIKI_RUNTIME = "atom-primitive-wiki-graph-v1"
PRIMITIVE_RAG_RUNTIME = "atom-primitive-graph-rag-v1"


def _terms(value: str) -> frozenset[str]:
    return frozenset(
        re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
    )


@dataclass(frozen=True)
class PrimitiveKnowledgeNode:
    primitive_id: str
    kind: str
    status: str
    description: str
    aliases: tuple[str, ...]
    components: tuple[str, ...]
    root_expansion: tuple[str, ...]
    evidence_count: int
    counterexample_count: int
    confidence: float


@dataclass(frozen=True)
class PrimitiveRagHit:
    primitive_id: str
    kind: str
    status: str
    score: float
    description: str
    components: tuple[str, ...]
    neighbors: tuple[str, ...]
    root_expansion: tuple[str, ...]
    evidence_count: int
    counterexample_count: int
    confidence: float


def _describe(record: PrimitiveRecord, roots: tuple[str, ...]) -> str:
    if record.root:
        return (
            "Immutable generative root operator in the seven-root Atom "
            "substrate."
        )
    assert record.recipe is not None
    return (
        f"{record.status} {record.recipe.mode} mathematical primitive with "
        f"{len(record.recipe.components)} direct components and "
        f"{len(roots)} root operations in its complete expansion."
    )


class PrimitiveWikiGraph:
    """A read-only graph snapshot generated from the live primitive inventory."""

    def __init__(self, forge: PrimitiveForge) -> None:
        self.source_graph_hash = forge.graph_hash
        self._nodes: dict[str, PrimitiveKnowledgeNode] = {}
        self._metadata_terms: dict[str, frozenset[str]] = {}
        self._incoming: dict[str, set[str]] = {
            record.primitive_id: set() for record in forge.records
        }
        for record in forge.records:
            components = (
                () if record.recipe is None else record.recipe.components
            )
            for component in components:
                self._incoming[component].add(record.primitive_id)
            roots = forge.expand_to_roots(record.primitive_id)
            node = PrimitiveKnowledgeNode(
                primitive_id=record.primitive_id,
                kind="root" if record.root else "discovered_primitive",
                status=record.status,
                description=_describe(record, roots),
                aliases=record.aliases,
                components=components,
                root_expansion=roots,
                evidence_count=len(record.evidence),
                counterexample_count=len(record.counterexamples),
                confidence=record.confidence,
            )
            self._nodes[node.primitive_id] = node
            metadata = " ".join(
                (
                    *record.aliases,
                    *record.invariants,
                    *record.symmetries,
                    *record.boundaries,
                    *record.scales,
                    *record.provenance,
                    record.signature.domain,
                    record.signature.output.kind,
                    node.description,
                )
            )
            self._metadata_terms[node.primitive_id] = _terms(metadata)
        if not self._nodes:
            raise ValueError("primitive wiki graph cannot be empty")

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    def node(self, primitive_id: str) -> PrimitiveKnowledgeNode:
        try:
            return self._nodes[primitive_id]
        except KeyError as error:
            raise ValueError(
                f"unknown primitive knowledge node: {primitive_id}"
            ) from error

    def neighbors(self, primitive_id: str) -> tuple[str, ...]:
        node = self.node(primitive_id)
        return tuple(
            sorted(set(node.components) | self._incoming[primitive_id])
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
    ) -> tuple[PrimitiveRagHit, ...]:
        """Retrieve graph nodes first, then propagate through graph edges."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("RAG query must be non-empty text")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 30
        ):
            raise ValueError("RAG limit must be an integer within [1, 30]")
        query_terms = _terms(query)
        direct: dict[str, float] = {}
        for primitive_id, node in self._nodes.items():
            id_terms = _terms(primitive_id)
            alias_terms = _terms(" ".join(node.aliases))
            root_terms = _terms(" ".join(node.root_expansion))
            score = (
                6.0 * len(query_terms & id_terms)
                + 4.0 * len(query_terms & alias_terms)
                + 2.0 * len(query_terms & root_terms)
                + 1.0
                * len(query_terms & self._metadata_terms[primitive_id])
            )
            if primitive_id.lower() in query.lower():
                score += 1_000.0
            if score > 0.0:
                direct[primitive_id] = score
        if not direct:
            return ()
        propagated = dict(direct)
        for primitive_id, score in direct.items():
            for neighbor in self.neighbors(primitive_id):
                propagated[neighbor] = (
                    propagated.get(neighbor, 0.0) + score * 0.35
                )
                for second_hop in self.neighbors(neighbor):
                    if second_hop != primitive_id:
                        propagated[second_hop] = (
                            propagated.get(second_hop, 0.0) + score * 0.10
                        )
        ranked = sorted(
            propagated,
            key=lambda item: (-propagated[item], item),
        )[:limit]
        return tuple(
            PrimitiveRagHit(
                primitive_id=primitive_id,
                kind=self._nodes[primitive_id].kind,
                status=self._nodes[primitive_id].status,
                score=round(propagated[primitive_id], 8),
                description=self._nodes[primitive_id].description,
                components=self._nodes[primitive_id].components,
                neighbors=self.neighbors(primitive_id),
                root_expansion=self._nodes[primitive_id].root_expansion,
                evidence_count=self._nodes[primitive_id].evidence_count,
                counterexample_count=(
                    self._nodes[primitive_id].counterexample_count
                ),
                confidence=self._nodes[primitive_id].confidence,
            )
            for primitive_id in ranked
        )

    def manifest(self) -> dict[str, Any]:
        nodes = [
            {
                **asdict(self._nodes[primitive_id]),
                "aliases": list(self._nodes[primitive_id].aliases),
                "components": list(self._nodes[primitive_id].components),
                "root_expansion": list(
                    self._nodes[primitive_id].root_expansion
                ),
                "neighbors": list(self.neighbors(primitive_id)),
            }
            for primitive_id in sorted(self._nodes)
        ]
        core = {
            "wiki_runtime": PRIMITIVE_WIKI_RUNTIME,
            "rag_runtime": PRIMITIVE_RAG_RUNTIME,
            "source_graph_hash": self.source_graph_hash,
            "nodes": nodes,
        }
        return {**core, "knowledge_hash": canonical_hash(core)}

    def assert_bound_to(self, forge: PrimitiveForge) -> None:
        if self.source_graph_hash != forge.graph_hash:
            raise ValueError("wiki graph is not bound to the primitive artifact")
        if set(self._nodes) != {
            record.primitive_id for record in forge.records
        }:
            raise ValueError("wiki graph inventory differs from the forge")


def retrieve_primitive_context(
    graph: PrimitiveWikiGraph,
    query: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Runtime RAG adapter returning graph lineage with every hit."""

    return [
        {
            **asdict(hit),
            "components": list(hit.components),
            "neighbors": list(hit.neighbors),
            "root_expansion": list(hit.root_expansion),
        }
        for hit in graph.retrieve(query, limit=limit)
    ]


def validate_knowledge_manifest(
    manifest: Mapping[str, Any],
    forge: PrimitiveForge,
) -> None:
    """Reject detached or modified serialized knowledge artifacts."""

    if not isinstance(manifest, Mapping):
        raise ValueError("knowledge manifest must be an object")
    if manifest.get("wiki_runtime") != PRIMITIVE_WIKI_RUNTIME:
        raise ValueError("primitive wiki runtime marker is invalid")
    if manifest.get("rag_runtime") != PRIMITIVE_RAG_RUNTIME:
        raise ValueError("primitive RAG runtime marker is invalid")
    if manifest.get("source_graph_hash") != forge.graph_hash:
        raise ValueError("knowledge manifest is detached from the forge")
    knowledge_hash = manifest.get("knowledge_hash")
    if not isinstance(knowledge_hash, str) or len(knowledge_hash) != 64:
        raise ValueError("knowledge manifest hash is invalid")
    core = {key: manifest[key] for key in manifest if key != "knowledge_hash"}
    if canonical_hash(core) != knowledge_hash:
        raise ValueError("knowledge manifest hash mismatch")
