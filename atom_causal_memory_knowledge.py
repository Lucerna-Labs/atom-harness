"""Runtime wiki graph and structural RAG over the persisted causal memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from atom_causal_memory import CausalMemoryClient
from atom_causal_world_schema import canonical_hash
from atom_primitive_forge import PrimitiveForge


CAUSAL_MEMORY_WIKI_RUNTIME = "atom-causal-memory-wiki-v1"
CAUSAL_MEMORY_RAG_RUNTIME = "atom-causal-structural-rag-v1"


@dataclass(frozen=True)
class CausalMemoryKnowledgeNode:
    primitive_id: str
    kind: str
    status: str
    components: tuple[str, ...]
    dependents: tuple[str, ...]
    root_expansion: tuple[str, ...]
    feature_count: int
    strengthened_motifs: int
    weakened_motifs: int


class CausalMemoryWikiGraph:
    """Graph documentation generated from the active durable catalog."""

    def __init__(
        self,
        client: CausalMemoryClient,
        forge: PrimitiveForge,
        inventory: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.forge = forge
        self.inventory = (
            client.inventory() if inventory is None else dict(inventory)
        )
        if self.inventory.get("source_graph_hash") != forge.graph_hash:
            raise ValueError(
                "causal-memory inventory is detached from the Forge graph"
            )
        glyphs = self.inventory.get("glyphs")
        if not isinstance(glyphs, list):
            raise ValueError("causal-memory inventory glyphs are invalid")
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in glyphs:
            if not isinstance(item, Mapping):
                raise ValueError("causal-memory inventory node is invalid")
            primitive_id = item.get("primitive_id")
            if not isinstance(primitive_id, str) or primitive_id in by_id:
                raise ValueError(
                    "causal-memory inventory identities are invalid"
                )
            by_id[primitive_id] = item
        forge_ids = {record.primitive_id for record in forge.records}
        if set(by_id) != forge_ids:
            raise ValueError(
                "durable causal-memory inventory differs from the Forge graph"
            )
        incoming: dict[str, set[str]] = {
            primitive_id: set() for primitive_id in forge_ids
        }
        for record in forge.records:
            if record.recipe is not None:
                for component in record.recipe.components:
                    incoming[component].add(record.primitive_id)
        self._nodes: dict[str, CausalMemoryKnowledgeNode] = {}
        for record in forge.records:
            stored = by_id[record.primitive_id]
            components = (
                () if record.recipe is None else record.recipe.components
            )
            self._nodes[record.primitive_id] = CausalMemoryKnowledgeNode(
                primitive_id=record.primitive_id,
                kind="root" if record.root else "derived",
                status=str(stored["status"]),
                components=components,
                dependents=tuple(sorted(incoming[record.primitive_id])),
                root_expansion=forge.expand_to_roots(record.primitive_id),
                feature_count=int(stored["feature_count"]),
                strengthened_motifs=int(stored["strengthened_motifs"]),
                weakened_motifs=int(stored["weakened_motifs"]),
            )

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    def node(self, primitive_id: str) -> CausalMemoryKnowledgeNode:
        try:
            return self._nodes[primitive_id]
        except KeyError as error:
            raise ValueError(
                f"unknown causal-memory wiki node: {primitive_id}"
            ) from error

    def retrieve(
        self,
        query_wire: str,
    ) -> tuple[dict[str, Any], ...]:
        """Run the real Rust structural field and attach graph lineage."""

        report = self.client.query(query_wire)
        context: list[dict[str, Any]] = []
        for hit in report["hits"]:
            node = self.node(hit["primitive_id"])
            context.append(
                {
                    "primitive_id": node.primitive_id,
                    "kind": node.kind,
                    "status": node.status,
                    "score": hit["score"],
                    "coverage_per_million": hit[
                        "coverage_per_million"
                    ],
                    "matched_support": hit["matched_support"],
                    "components": list(node.components),
                    "dependents": list(node.dependents),
                    "root_expansion": list(node.root_expansion),
                    "evidence_paths": hit["motifs"],
                }
            )
        return tuple(context)

    def manifest(self) -> dict[str, Any]:
        nodes = [
            {
                **asdict(self._nodes[primitive_id]),
                "components": list(
                    self._nodes[primitive_id].components
                ),
                "dependents": list(
                    self._nodes[primitive_id].dependents
                ),
                "root_expansion": list(
                    self._nodes[primitive_id].root_expansion
                ),
            }
            for primitive_id in sorted(self._nodes)
        ]
        edges = [
            {
                "source": record.primitive_id,
                "relation": f"component/{index:04}",
                "target": component,
            }
            for record in self.forge.records
            if record.recipe is not None
            for index, component in enumerate(record.recipe.components)
        ]
        core = {
            "schema": 1,
            "wiki_runtime": CAUSAL_MEMORY_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_MEMORY_RAG_RUNTIME,
            "source_graph_hash": self.forge.graph_hash,
            "catalog_identity": self.inventory["catalog_identity"],
            "snapshot_sequence": self.inventory["snapshot_sequence"],
            "nodes": nodes,
            "edges": edges,
        }
        return {**core, "knowledge_hash": canonical_hash(core)}


def retrieve_causal_memory_context(
    graph: CausalMemoryWikiGraph,
    query_wire: str,
) -> list[dict[str, Any]]:
    """Runtime marker for the graph-first structural RAG path."""

    return list(graph.retrieve(query_wire))


def validate_causal_memory_knowledge(
    manifest: Mapping[str, Any],
    *,
    graph_hash: str,
    catalog_identity: str,
) -> None:
    if manifest.get("wiki_runtime") != CAUSAL_MEMORY_WIKI_RUNTIME:
        raise ValueError("causal-memory wiki runtime marker is invalid")
    if manifest.get("rag_runtime") != CAUSAL_MEMORY_RAG_RUNTIME:
        raise ValueError("causal-memory RAG runtime marker is invalid")
    if manifest.get("source_graph_hash") != graph_hash:
        raise ValueError("causal-memory knowledge has the wrong graph")
    if manifest.get("catalog_identity") != catalog_identity:
        raise ValueError("causal-memory knowledge has the wrong catalog")
    knowledge_hash = manifest.get("knowledge_hash")
    if not isinstance(knowledge_hash, str) or len(knowledge_hash) != 64:
        raise ValueError("causal-memory knowledge hash is invalid")
    core = {
        key: manifest[key]
        for key in manifest
        if key != "knowledge_hash"
    }
    if canonical_hash(core) != knowledge_hash:
        raise ValueError("causal-memory knowledge hash mismatch")
