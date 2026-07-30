"""Executable wiki graph and graph-first retrieval for the Atom coding harness."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from atom_causal_world_schema import canonical_hash
from atom_runtime_knowledge import (
    DEFAULT_COMPOSITION,
    DEFAULT_NODES,
    AtomWikiGraph,
    KnowledgeNode,
)


CODING_WIKI_RUNTIME = "atom-coding-wiki-graph-v1"
CODING_RAG_RUNTIME = "atom-coding-graph-rag-v1"


CODING_NODES = (
    KnowledgeNode(
        "identity",
        "mathematical_platform_primitive",
        "Preserve a typed message across a platform boundary.",
        ("type", "typed message", "same value"),
    ),
    KnowledgeNode(
        "directed_relation",
        "mathematical_platform_primitive",
        "Connect a source port to a target port as a directed relation.",
        ("route", "edge", "directed graph"),
    ),
    KnowledgeNode(
        "composition",
        "mathematical_platform_primitive",
        "Compose transformations and promote work onto parallel lanes.",
        ("compose", "parallel", "highway"),
    ),
    KnowledgeNode(
        "ordering",
        "mathematical_platform_primitive",
        "Use a partial order to schedule supported work.",
        ("priority", "schedule", "before"),
    ),
    KnowledgeNode(
        "feedback",
        "mathematical_platform_primitive",
        "Return load information as backpressure through the fabric.",
        ("backpressure", "vibration", "control"),
    ),
    KnowledgeNode(
        "fixed_point",
        "mathematical_platform_primitive",
        "Repeat a bounded transition until it settles or reaches its limit.",
        ("retry", "converge", "bounded iteration"),
    ),
    KnowledgeNode(
        "topology",
        "mathematical_platform_primitive",
        "Let stable threads and intersections emerge from routed work.",
        ("thread", "intersection", "network shape"),
    ),
    KnowledgeNode(
        "projection",
        "mathematical_platform_primitive",
        "Collapse supported runtime state into a discrete result.",
        ("output", "decision", "measurement"),
    ),
    KnowledgeNode(
        "layer_0_transport",
        "spiderweb_layer",
        "Move bytes or events through the ground transport.",
        ("transport", "ground bus"),
    ),
    KnowledgeNode(
        "layer_1_message",
        "spiderweb_layer",
        "Preserve typed message identity and routing intent.",
        ("message", "envelope"),
    ),
    KnowledgeNode(
        "layer_2_flow",
        "spiderweb_layer",
        "Promote parallel work, preload destinations, and form flow threads.",
        ("flow", "ramp", "preload"),
    ),
    KnowledgeNode(
        "layer_3_orchestration",
        "spiderweb_layer",
        "Coordinate priorities, capacity, recovery, and output projection.",
        ("orchestrator", "platform control"),
    ),
    KnowledgeNode(
        "code_causal_graph",
        "coding_cognition",
        "Learn which mathematical primitive causes each executable behavior.",
        ("causal coding", "intervention", "learned law"),
    ),
    KnowledgeNode(
        "platform_synthesis",
        "coding_cognition",
        "Compose learned mathematical primitives into an executable platform.",
        ("generate platform", "build system", "artifact"),
    ),
    KnowledgeNode(
        "hidden_behavioral_measurement",
        "coding_cognition",
        "Measure generated behavior with sealed requests outside candidate code.",
        ("hidden test", "evaluation", "projective measurement"),
    ),
    KnowledgeNode(
        "atom_language",
        "primary_construction_language",
        "Express causal intent, invariants, capabilities, and composition directly.",
        ("atom source", "mathematical language", "causal language"),
    ),
    KnowledgeNode(
        "typed_causal_ir",
        "primary_construction_language",
        "Preserve Atom meaning as a typed graph before any target projection.",
        ("intermediate representation", "atom graph", "typed graph"),
    ),
    KnowledgeNode(
        "rust_projection",
        "execution_projection",
        "Lower validated Atom structure into compiled Rust execution.",
        ("rust target", "compiler", "native execution"),
    ),
    KnowledgeNode(
        "thin_frontend_projection",
        "interface_projection",
        "Render Atom structure through a thin Svelte and TypeScript surface.",
        ("svelte", "typescript", "visual interface"),
    ),
)


CODING_COMPOSITION = {
    "identity": ("conservation",),
    "directed_relation": ("radiation", "attraction_repulsion"),
    "composition": ("radiation", "nucleation"),
    "ordering": ("gravitation", "attraction_repulsion"),
    "feedback": ("attraction_repulsion", "dissipation"),
    "fixed_point": ("nucleation", "dissipation", "decay"),
    "topology": ("gravitation", "nucleation", "radiation"),
    "projection": ("radiation", "conservation"),
    "layer_0_transport": ("directed_relation", "conservation"),
    "layer_1_message": ("identity", "directed_relation"),
    "layer_2_flow": ("composition", "feedback", "topology"),
    "layer_3_orchestration": (
        "ordering",
        "fixed_point",
        "projection",
        "conservation",
    ),
    "code_causal_graph": (
        "layer_1_message",
        "feedback",
        "fixed_point",
        "conservation",
    ),
    "platform_synthesis": (
        "layer_0_transport",
        "layer_1_message",
        "layer_2_flow",
        "layer_3_orchestration",
    ),
    "hidden_behavioral_measurement": (
        "projection",
        "feedback",
        "conservation",
    ),
    "atom_language": (
        "platform_synthesis",
        "identity",
        "composition",
        "conservation",
        "projection",
    ),
    "typed_causal_ir": (
        "atom_language",
        "topology",
        "ordering",
        "conservation",
    ),
    "rust_projection": (
        "typed_causal_ir",
        "fixed_point",
        "hidden_behavioral_measurement",
    ),
    "thin_frontend_projection": (
        "typed_causal_ir",
        "projection",
        "radiation",
    ),
}


class CodingWikiGraph:
    """Hash-bound coding knowledge graph used directly by the runtime."""

    def __init__(self) -> None:
        self.graph = AtomWikiGraph(
            nodes=(*DEFAULT_NODES, *CODING_NODES),
            composition={**DEFAULT_COMPOSITION, **CODING_COMPOSITION},
        )
        self.graph.assert_all_leaves_are_universe_primitives()

    def retrieve(self, query: str, limit: int = 7) -> list[dict[str, Any]]:
        return [
            {**asdict(hit), "neighbors": list(hit.neighbors)}
            for hit in self.graph.retrieve(query, limit=limit)
        ]

    def manifest(self) -> dict[str, Any]:
        atom_graph = self.graph.manifest()
        core = {
            "schema": 1,
            "wiki_runtime": CODING_WIKI_RUNTIME,
            "rag_runtime": CODING_RAG_RUNTIME,
            "atom_graph": atom_graph,
        }
        return {**core, "knowledge_hash": canonical_hash(core)}


def retrieve_coding_context(
    graph: CodingWikiGraph,
    query: str,
    limit: int = 7,
) -> list[dict[str, Any]]:
    """Runtime RAG adapter; retrieval starts from graph relationships."""

    return graph.retrieve(query, limit=limit)


def coding_knowledge_self_test() -> dict[str, bool]:
    graph = CodingWikiGraph()
    manifest = graph.manifest()
    hits = retrieve_coding_context(
        graph,
        "build a parallel platform with backpressure and emergent threads",
    )
    return {
        "runtime_wiki_graph": manifest["wiki_runtime"] == CODING_WIKI_RUNTIME,
        "runtime_graph_rag": manifest["rag_runtime"] == CODING_RAG_RUNTIME,
        "hash_bound": manifest["knowledge_hash"]
        == canonical_hash(
            {
                key: value
                for key, value in manifest.items()
                if key != "knowledge_hash"
            }
        ),
        "platform_recipe_reaches_roots": bool(
            graph.graph.expand("platform_synthesis")
        ),
        "retrieval_returns_platform_context": bool(hits),
    }
