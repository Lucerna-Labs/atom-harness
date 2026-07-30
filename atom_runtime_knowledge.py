"""Runtime wiki graph and graph-native retrieval for Atom experiments.

The graph is executable architecture, not documentation beside the runtime.
Cognitive atoms resolve through graph edges until the seven universe primitives
are reached. Retrieval starts from matching graph nodes and expands through
their composition neighborhood.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


ATOM_WIKI_GRAPH_RUNTIME = "atom-wiki-graph-v1"
ATOM_RAG_RUNTIME = "atom-graph-rag-v1"

UNIVERSE_PRIMITIVE_NAMES = (
    "radiation",
    "dissipation",
    "gravitation",
    "attraction_repulsion",
    "nucleation",
    "conservation",
    "decay",
)


@dataclass(frozen=True)
class KnowledgeNode:
    name: str
    kind: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeHit:
    name: str
    kind: str
    score: float
    description: str
    neighbors: tuple[str, ...]


DEFAULT_NODES = (
    KnowledgeNode(
        "radiation",
        "universe_primitive",
        "Propagate energy, state, or information outward through a field.",
        ("transmit", "signal", "spread"),
    ),
    KnowledgeNode(
        "dissipation",
        "universe_primitive",
        "Diffuse concentrated energy, cool transients, and erase unsupported detail.",
        ("cool", "erase", "thermal"),
    ),
    KnowledgeNode(
        "gravitation",
        "universe_primitive",
        "Aggregate distributed influence into a shared attractor.",
        ("group", "attractor", "aggregate"),
    ),
    KnowledgeNode(
        "attraction_repulsion",
        "universe_primitive",
        "Bind compatible structures and separate incompatible structures.",
        ("connect", "disconnect", "bind", "repel"),
    ),
    KnowledgeNode(
        "nucleation",
        "universe_primitive",
        "Crystallize persistent structure when repeated evidence crosses a threshold.",
        ("structure", "threshold", "phase transition", "crystallize"),
    ),
    KnowledgeNode(
        "conservation",
        "universe_primitive",
        "Preserve invariants while energy and information change form.",
        ("persist", "remember", "invariant"),
    ),
    KnowledgeNode(
        "decay",
        "universe_primitive",
        "Retire unstable structures that are not actively maintained.",
        ("expire", "timeout", "forget"),
    ),
    KnowledgeNode(
        "phase_mix",
        "composite_atom",
        "Combine relation waves so coherent evidence reinforces and conflict cancels.",
        ("interference", "chaos", "phase"),
    ),
    KnowledgeNode(
        "thermal_anneal",
        "composite_atom",
        "Explore structural alternatives while hot, then settle as the field cools.",
        ("annealing", "temperature", "explore", "settle"),
    ),
    KnowledgeNode(
        "attention",
        "cognitive_atom",
        "Propagate a cue, form an attractor, and select compatible structure.",
        ("focus", "select"),
    ),
    KnowledgeNode(
        "learn",
        "cognitive_atom",
        "Nucleate repeated evidence and reshape the field around its relation.",
        ("plasticity", "adapt"),
    ),
    KnowledgeNode(
        "remember",
        "cognitive_atom",
        "Persist a learned relation as a conserved phase law.",
        ("memory", "retain"),
    ),
    KnowledgeNode(
        "forget",
        "cognitive_atom",
        "Cool, dissipate, and remove unsupported evidence.",
        ("prune", "erase"),
    ),
    KnowledgeNode(
        "retrieve",
        "cognitive_atom",
        "Let a cue propagate through conserved laws and settle on a state.",
        ("recall", "predict"),
    ),
    KnowledgeNode(
        "revise",
        "cognitive_atom",
        "Dissolve contradicted structure before learning replacement evidence.",
        ("correct", "update"),
    ),
    KnowledgeNode(
        "abstract",
        "cognitive_atom",
        "Coarse-grain many transition traces into a small reusable law.",
        ("compress", "generalize", "law"),
    ),
)


DEFAULT_COMPOSITION = {
    "phase_mix": ("radiation", "gravitation"),
    "thermal_anneal": (
        "phase_mix",
        "attraction_repulsion",
        "dissipation",
        "conservation",
    ),
    "attention": ("phase_mix", "attraction_repulsion"),
    "learn": ("nucleation", "attention", "conservation"),
    "remember": ("learn",),
    "forget": ("dissipation", "decay", "conservation"),
    "retrieve": ("attention",),
    "revise": ("forget", "remember"),
    "abstract": (
        "phase_mix",
        "nucleation",
        "conservation",
        "decay",
    ),
}


LANGUAGE_NODES = (
    KnowledgeNode(
        "ground",
        "language_atom",
        "Align an utterance pulse with an observable world consequence.",
        ("meaning", "world", "consequence"),
    ),
    KnowledgeNode(
        "lexical_nucleation",
        "language_atom",
        "Crystallize recurring surface spans into grounded lexical atoms.",
        ("word", "character", "segment", "lexeme"),
    ),
    KnowledgeNode(
        "role_bind",
        "language_atom",
        "Bind grounded concepts to agent, patient, destination, and recipient roles.",
        ("grammar", "syntax", "semantic role"),
    ),
    KnowledgeNode(
        "understand",
        "language_atom",
        "Let a surface utterance settle into an executable meaning frame.",
        ("parse", "interpret", "comprehend"),
    ),
    KnowledgeNode(
        "resolve_reference",
        "language_atom",
        "Attract a pronoun toward the most recent compatible context atom.",
        ("pronoun", "context", "anaphora"),
    ),
    KnowledgeNode(
        "speak",
        "language_atom",
        "Radiate a meaning frame into learned surface order while conserving roles.",
        ("generate", "utterance", "answer"),
    ),
    KnowledgeNode(
        "language_learn",
        "language_atom",
        "Turn grounded episodes into compact lexical, frame, and reference laws.",
        ("language learning", "grounded learning"),
    ),
    KnowledgeNode(
        "language_abstract",
        "language_atom",
        "Coarse-grain grounded episode traces into persistent language laws.",
        ("compress language", "grammar law"),
    ),
)


LANGUAGE_COMPOSITION = {
    "ground": ("radiation", "gravitation", "attraction_repulsion"),
    "lexical_nucleation": ("ground", "nucleation", "conservation"),
    "role_bind": ("ground", "conservation"),
    "understand": ("ground", "conservation"),
    "resolve_reference": (
        "gravitation",
        "attraction_repulsion",
        "conservation",
        "decay",
    ),
    "speak": (
        "gravitation",
        "attraction_repulsion",
        "radiation",
        "conservation",
        "decay",
    ),
    "language_learn": (
        "radiation",
        "gravitation",
        "attraction_repulsion",
        "nucleation",
        "conservation",
        "dissipation",
        "decay",
    ),
    "language_abstract": (
        "phase_mix",
        "nucleation",
        "conservation",
        "decay",
    ),
}


HOMEOSTATIC_NODES = (
    KnowledgeNode(
        "sense_order",
        "homeostatic_atom",
        "Measure crystallization, free mass, surprise, churn, and accepted disturbances.",
        ("order parameter", "observable", "criticality"),
    ),
    KnowledgeNode(
        "regulate_chaos",
        "homeostatic_atom",
        "Keep exploration and persistence inside bounded target bands through feedback.",
        ("homeostasis", "governor", "feedback", "control parameter"),
    ),
    KnowledgeNode(
        "metaplasticity",
        "homeostatic_atom",
        "Change temperature, disturbance strength, and commitment pressure from field state.",
        ("plasticity of plasticity", "adaptive learning"),
    ),
    KnowledgeNode(
        "homeostatic_observe",
        "homeostatic_recipe",
        "Assimilate one opaque consequence through every universe primitive.",
        ("observe consequence", "online learn"),
    ),
    KnowledgeNode(
        "homeostatic_govern",
        "homeostatic_recipe",
        "Aggregate a window, derive signed control errors, and commit bounded controls.",
        ("acceptance feedback", "reheat", "cool", "critical governor"),
    ),
    KnowledgeNode(
        "homeostatic_forget",
        "homeostatic_recipe",
        "Remove raw evidence while conserving learned laws and controller summaries.",
        ("forget evidence", "retain law"),
    ),
)


HOMEOSTATIC_COMPOSITION = {
    "sense_order": ("radiation", "gravitation"),
    "regulate_chaos": (
        "sense_order",
        "attraction_repulsion",
        "radiation",
        "dissipation",
        "conservation",
    ),
    "metaplasticity": ("regulate_chaos", "nucleation", "decay"),
    "homeostatic_observe": (
        "radiation",
        "gravitation",
        "attraction_repulsion",
        "nucleation",
        "conservation",
        "dissipation",
        "decay",
    ),
    "homeostatic_govern": (
        "gravitation",
        "attraction_repulsion",
        "radiation",
        "dissipation",
        "conservation",
        "nucleation",
        "decay",
    ),
    "homeostatic_forget": ("dissipation", "decay", "conservation"),
}


NEURAL_LANGUAGE_NODES = (
    KnowledgeNode(
        "consequence_induction",
        "neural_language_atom",
        "Infer latent root-operator controls by replaying candidate mechanics against an observed consequence.",
        ("inverse dynamics", "weak supervision", "latent program"),
    ),
    KnowledgeNode(
        "operator_lexicon_memory",
        "neural_language_atom",
        "Crystallize opaque surface tokens into reusable operator activations without retaining raw episodes.",
        ("lexicon memory", "operator word", "remember"),
    ),
    KnowledgeNode(
        "query_surface_memory",
        "neural_language_atom",
        "Discover opaque query meanings and bind grounded semantic answers back to surface tokens.",
        ("question answering", "surface law", "response"),
    ),
    KnowledgeNode(
        "neural_field_execute",
        "neural_language_atom",
        "Drive the differentiable seven-operator field from induced language controls.",
        ("neural field", "execute language", "world model"),
    ),
    KnowledgeNode(
        "evidence_bound_claim",
        "neural_language_atom",
        "Permit a factual surface assertion only when learned operator, derivation, query, and surface evidence form a complete path.",
        ("abstain", "unknown", "provenance", "hallucination control"),
    ),
    KnowledgeNode(
        "adaptive_latent_compute",
        "neural_language_atom",
        "Skip unsupported requests before field execution and shorten recurrent text processing when crystallized laws supply sufficient support.",
        ("early exit", "fast path", "dynamic compute", "reasoning budget"),
    ),
    KnowledgeNode(
        "lifelong_language_adapt",
        "neural_language_recipe",
        "Use homeostatic feedback to accept coherent lexical novelty and cool incoherent disturbance.",
        ("lifelong learning", "transfer", "noise rejection"),
    ),
    KnowledgeNode(
        "neural_language_forget",
        "neural_language_recipe",
        "Erase raw language events while retaining induced operator, query, surface, and neural laws.",
        ("forget raw events", "retain neural law"),
    ),
)


NEURAL_LANGUAGE_COMPOSITION = {
    "consequence_induction": (
        "radiation",
        "gravitation",
        "attraction_repulsion",
        "dissipation",
        "nucleation",
        "conservation",
        "decay",
    ),
    "operator_lexicon_memory": (
        "consequence_induction",
        "nucleation",
        "conservation",
        "decay",
    ),
    "query_surface_memory": (
        "ground",
        "gravitation",
        "attraction_repulsion",
        "nucleation",
        "conservation",
    ),
    "neural_field_execute": (
        "radiation",
        "dissipation",
        "gravitation",
        "attraction_repulsion",
        "nucleation",
        "conservation",
        "decay",
    ),
    "evidence_bound_claim": (
        "operator_lexicon_memory",
        "neural_field_execute",
        "query_surface_memory",
        "conservation",
        "dissipation",
    ),
    "adaptive_latent_compute": (
        "gravitation",
        "conservation",
        "dissipation",
        "decay",
    ),
    "lifelong_language_adapt": (
        "homeostatic_govern",
        "operator_lexicon_memory",
        "query_surface_memory",
        "neural_field_execute",
        "evidence_bound_claim",
        "adaptive_latent_compute",
    ),
    "neural_language_forget": (
        "dissipation",
        "decay",
        "conservation",
        "operator_lexicon_memory",
        "query_surface_memory",
    ),
}


ENGLISH_LANGUAGE_NODES = (
    KnowledgeNode(
        "english_language_codec",
        "english_language_atom",
        "Normalize natural English into the compact Atom token field and render grounded semantic answers back into ordinary English.",
        ("English shell", "language codec", "natural answer"),
    ),
    KnowledgeNode(
        "english_synonym_adapt",
        "english_language_recipe",
        "Ground previously unseen English action words through observed consequences while retaining earlier vocabulary.",
        ("synonym transfer", "few shot English", "lexical adaptation"),
    ),
    KnowledgeNode(
        "evidence_bound_english_answer",
        "english_language_recipe",
        "Render an English factual answer only after the internal operator, field, query, and surface evidence path is complete.",
        ("English abstention", "supported answer", "claim lineage"),
    ),
)


ENGLISH_LANGUAGE_COMPOSITION = {
    "english_language_codec": (
        "radiation",
        "gravitation",
        "attraction_repulsion",
        "conservation",
        "decay",
    ),
    "english_synonym_adapt": (
        "lifelong_language_adapt",
        "english_language_codec",
    ),
    "evidence_bound_english_answer": (
        "english_language_codec",
        "evidence_bound_claim",
        "adaptive_latent_compute",
    ),
}


def _terms(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


class AtomWikiGraph:
    """Executable graph for cognitive composition and graph-first retrieval."""

    def __init__(
        self,
        nodes: Sequence[KnowledgeNode] = DEFAULT_NODES,
        composition: Mapping[str, Sequence[str]] = DEFAULT_COMPOSITION,
    ) -> None:
        self._nodes = {node.name: node for node in nodes}
        self._composition = {
            str(name): tuple(str(step) for step in steps)
            for name, steps in composition.items()
        }
        if len(self._nodes) != len(nodes):
            raise ValueError("Knowledge node names must be unique")
        if set(UNIVERSE_PRIMITIVE_NAMES) - set(self._nodes):
            raise ValueError("All seven universe primitives require graph nodes")
        for name, steps in self._composition.items():
            if name not in self._nodes:
                raise ValueError(f"Composition node is missing: {name}")
            for step in steps:
                if step not in self._nodes:
                    raise ValueError(f"Unknown composition edge: {name} -> {step}")
        for name in self._composition:
            self.expand(name)

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    def neighbors(self, name: str) -> tuple[str, ...]:
        if name not in self._nodes:
            raise ValueError(f"Unknown knowledge node: {name}")
        outgoing = set(self._composition.get(name, ()))
        incoming = {
            parent for parent, steps in self._composition.items() if name in steps
        }
        return tuple(sorted(outgoing | incoming))

    def expand(self, name: str) -> tuple[str, ...]:
        if name not in self._nodes:
            raise ValueError(f"Unknown Atom recipe: {name}")

        def visit(item: str, ancestry: tuple[str, ...]) -> tuple[str, ...]:
            if item in UNIVERSE_PRIMITIVE_NAMES:
                return (item,)
            if item not in self._composition:
                raise ValueError(f"Node is not executable: {item}")
            if item in ancestry:
                chain = " -> ".join((*ancestry, item))
                raise ValueError(f"Cyclic Atom composition: {chain}")
            leaves: list[str] = []
            for child in self._composition[item]:
                leaves.extend(visit(child, (*ancestry, item)))
            return tuple(leaves)

        return visit(name, ())

    def retrieve(self, query: str, limit: int = 5) -> tuple[KnowledgeHit, ...]:
        """Retrieve graph nodes, then expand relevance through direct neighbors."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty text")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 20
        ):
            raise ValueError("limit must be an integer within [1, 20]")
        query_terms = _terms(query)
        direct: dict[str, float] = {}
        for name, node in self._nodes.items():
            name_terms = _terms(name)
            alias_terms = _terms(" ".join(node.aliases))
            description_terms = _terms(node.description)
            score = (
                4.0 * len(query_terms & name_terms)
                + 2.0 * len(query_terms & alias_terms)
                + 1.0 * len(query_terms & description_terms)
            )
            if score > 0.0:
                direct[name] = score
        if not direct:
            return ()

        propagated = dict(direct)
        for name, score in direct.items():
            for neighbor in self.neighbors(name):
                propagated[neighbor] = propagated.get(neighbor, 0.0) + 0.35 * score
        ranked = sorted(propagated, key=lambda item: (-propagated[item], item))[:limit]
        return tuple(
            KnowledgeHit(
                name=name,
                kind=self._nodes[name].kind,
                score=round(propagated[name], 6),
                description=self._nodes[name].description,
                neighbors=self.neighbors(name),
            )
            for name in ranked
        )

    def manifest(self) -> dict[str, object]:
        return {
            "wiki_runtime": ATOM_WIKI_GRAPH_RUNTIME,
            "rag_runtime": ATOM_RAG_RUNTIME,
            "nodes": [asdict(self._nodes[name]) for name in sorted(self._nodes)],
            "composition": {
                name: {
                    "steps": list(self._composition[name]),
                    "primitive_leaves": list(self.expand(name)),
                }
                for name in sorted(self._composition)
            },
        }

    def assert_all_leaves_are_universe_primitives(self) -> None:
        allowed = set(UNIVERSE_PRIMITIVE_NAMES)
        for name in self._composition:
            leaves = self.expand(name)
            if not leaves or set(leaves) - allowed:
                raise AssertionError(f"Recipe {name} has invalid leaves: {leaves}")


def retrieve_atom_context(
    graph: AtomWikiGraph, query: str, limit: int = 5
) -> list[dict[str, object]]:
    """RAG adapter used by the runtime and its serialized report path."""

    return [
        {**asdict(hit), "neighbors": list(hit.neighbors)}
        for hit in graph.retrieve(query, limit=limit)
    ]


def composition_edges(
    graph: AtomWikiGraph, names: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    return {name: graph.expand(name) for name in names}


def build_language_graph() -> AtomWikiGraph:
    """Return the executable Atom graph extended with language compositions."""

    return AtomWikiGraph(
        nodes=(*DEFAULT_NODES, *LANGUAGE_NODES),
        composition={**DEFAULT_COMPOSITION, **LANGUAGE_COMPOSITION},
    )


def build_homeostatic_graph() -> AtomWikiGraph:
    """Return the executable graph extended with feedback-control recipes."""

    return AtomWikiGraph(
        nodes=(*DEFAULT_NODES, *LANGUAGE_NODES, *HOMEOSTATIC_NODES),
        composition={
            **DEFAULT_COMPOSITION,
            **LANGUAGE_COMPOSITION,
            **HOMEOSTATIC_COMPOSITION,
        },
    )


def build_neural_language_graph() -> AtomWikiGraph:
    """Return the graph for the lifelong neural language-field runtime."""

    return AtomWikiGraph(
        nodes=(
            *DEFAULT_NODES,
            *LANGUAGE_NODES,
            *HOMEOSTATIC_NODES,
            *NEURAL_LANGUAGE_NODES,
        ),
        composition={
            **DEFAULT_COMPOSITION,
            **LANGUAGE_COMPOSITION,
            **HOMEOSTATIC_COMPOSITION,
            **NEURAL_LANGUAGE_COMPOSITION,
        },
    )


def build_english_language_graph() -> AtomWikiGraph:
    """Return the graph for the evidence-bound natural-English runtime."""

    return AtomWikiGraph(
        nodes=(
            *DEFAULT_NODES,
            *LANGUAGE_NODES,
            *HOMEOSTATIC_NODES,
            *NEURAL_LANGUAGE_NODES,
            *ENGLISH_LANGUAGE_NODES,
        ),
        composition={
            **DEFAULT_COMPOSITION,
            **LANGUAGE_COMPOSITION,
            **HOMEOSTATIC_COMPOSITION,
            **NEURAL_LANGUAGE_COMPOSITION,
            **ENGLISH_LANGUAGE_COMPOSITION,
        },
    )
