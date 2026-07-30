"""Runtime-wired wiki graph and graph-first RAG for the causal world."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from atom_causal_graph import CausalGraph, law_condition_signature
from atom_formal_domains import (
    FORMAL_DOMAIN_NAMES,
    FORMAL_PRIMITIVES,
    formal_domain_manifest,
)
from atom_causal_world_curriculum import (
    WORLD_PROGRAM_AXES,
    world_program_space_size,
)
from atom_causal_world_schema import (
    ARCHITECTURE_COMPONENTS,
    DOMAIN_MECHANISMS,
    DOMAIN_NAMES,
    FEATURE_NAMES,
    ROOT_MECHANICS,
    canonical_hash,
)


CAUSAL_WORLD_WIKI_RUNTIME = "atom-causal-world-wiki-graph-v9"
CAUSAL_WORLD_RAG_RUNTIME = "atom-causal-world-graph-rag-v3"


@dataclass(frozen=True)
class CausalKnowledgeNode:
    name: str
    kind: str
    description: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CausalKnowledgeHit:
    name: str
    kind: str
    score: float
    description: str
    neighbors: tuple[str, ...]
    evidence: Mapping[str, Any]


COMPONENT_COMPOSITION: Mapping[str, tuple[str, ...]] = {
    "causal_graph": (
        "nucleation",
        "conservation",
        "attraction_repulsion",
        "decay",
    ),
    "phase_locked_loop": (
        "radiation",
        "attraction_repulsion",
        "conservation",
    ),
    "phase_mixer": ("radiation", "gravitation", "attraction_repulsion"),
    "molecular_recognition": ("gravitation", "attraction_repulsion"),
    "topological_persistence": ("nucleation", "conservation", "decay"),
    "thermal_annealing": (
        "radiation",
        "dissipation",
        "attraction_repulsion",
        "conservation",
    ),
    "projective_measurement": ("gravitation", "nucleation", "conservation"),
}


def _static_nodes() -> tuple[CausalKnowledgeNode, ...]:
    nodes: list[CausalKnowledgeNode] = []
    root_descriptions = {
        "radiation": "Propagate state, energy, signal, or evidence through relations.",
        "dissipation": "Reduce unsupported gradients and release unstable structure.",
        "gravitation": "Aggregate distributed influence around a context attractor.",
        "attraction_repulsion": "Bind compatible states and separate incompatible states.",
        "nucleation": "Commit recurring evidence into a persistent causal structure.",
        "conservation": "Preserve invariants and support lineage through transformations.",
        "decay": "Retire unsupported hypotheses and stale transient structure.",
    }
    for root in ROOT_MECHANICS:
        nodes.append(
            CausalKnowledgeNode(
                root,
                "universe_primitive",
                root_descriptions[root],
            )
        )
    component_descriptions = {
        "causal_graph": "Store conditional executable laws with delay, confidence, contradiction, and provenance.",
        "phase_locked_loop": "Synchronize the currently active causal subgraph through time.",
        "phase_mixer": "Combine compatible causal hypotheses while conflicting phases cancel.",
        "molecular_recognition": "Retrieve laws whose causal binding sites match the current context.",
        "topological_persistence": "Keep laws that survive evidence and scale thresholds.",
        "thermal_annealing": "Explore competing explanations while hot and settle coherently while cooling.",
        "projective_measurement": "Expose a discrete answer only when a supported causal state survives settlement.",
    }
    for component in ARCHITECTURE_COMPONENTS:
        nodes.append(
            CausalKnowledgeNode(
                component,
                "causal_architecture",
                component_descriptions[component],
                tuple(component.split("_")),
            )
        )
    for domain in DOMAIN_NAMES:
        nodes.append(
            CausalKnowledgeNode(
                domain,
                "world_domain",
                f"Procedural {domain} worlds with interventions, hidden causes, and delayed consequences.",
                DOMAIN_MECHANISMS[domain],
            )
        )
        nodes.extend(
            CausalKnowledgeNode(
                mechanism,
                "domain_mechanism",
                f"Executable {domain} mechanism generated from root causal dynamics.",
                (domain,),
            )
            for mechanism in DOMAIN_MECHANISMS[domain]
        )
    for domain in FORMAL_DOMAIN_NAMES:
        domain_primitives = tuple(
            primitive.name
            for primitive in FORMAL_PRIMITIVES
            if primitive.domain == domain
        )
        nodes.append(
            CausalKnowledgeNode(
                f"formal_domain_{domain}",
                "formal_domain",
                (
                    f"Typed executable {domain.replace('_', ' ')} curriculum "
                    "with an independent truth oracle and explicit epistemic states."
                ),
                (domain.replace("_", " "), *domain_primitives),
            )
        )
    nodes.extend(
        CausalKnowledgeNode(
            primitive.name,
            "formal_primitive",
            primitive.description,
            (
                primitive.domain.replace("_", " "),
                *primitive.invariants,
                *primitive.input_fields,
            ),
        )
        for primitive in FORMAL_PRIMITIVES
    )
    nodes.extend(
        CausalKnowledgeNode(
            feature,
            "world_feature",
            f"Observable or intervenable causal variable: {feature}.",
        )
        for feature in FEATURE_NAMES
    )
    for axis, values in WORLD_PROGRAM_AXES:
        nodes.append(
            CausalKnowledgeNode(
                f"world_regime_{axis}",
                "world_program_axis",
                (
                    f"A compositional world axis for {axis.replace('_', ' ')}: "
                    + ", ".join(value.replace("_", " ") for value in values)
                    + "."
                ),
                tuple(value.replace("_", " ") for value in values),
            )
        )
    nodes.append(
        CausalKnowledgeNode(
            "active_experimentation",
            "learning_protocol",
            "Choose interventions that distinguish competing causal explanations.",
            ("counterfactual", "experiment", "intervention", "causal discovery"),
        )
    )
    nodes.append(
        CausalKnowledgeNode(
            "contextual_causal_transfer",
            "learning_protocol",
            (
                "Infer a causal direction in an unseen world regime only when "
                "multiple persistent laws from related regimes agree; otherwise "
                "return an explicit unknown."
            ),
            (
                "held out causal transfer",
                "unseen regime",
                "context similarity",
                "conservative generalization",
                "abstention",
            ),
        )
    )
    nodes.append(
        CausalKnowledgeNode(
            "metaplastic_transfer_governor",
            "learning_protocol",
            (
                "Select direction priors and acceptance bands on validation "
                "worlds disjoint from training and final evaluation, then freeze "
                "the selected projection policy before evaluator questions."
            ),
            (
                "metaplastic calibration",
                "homeostatic transfer control",
                "validation world",
                "direction acceptance band",
                "frozen evaluator policy",
            ),
        )
    )
    nodes.append(
        CausalKnowledgeNode(
            "context_factor_risk_governor",
            "learning_protocol",
            (
                "Compose singleton world conditions with pairwise context "
                "motifs, reuse one direction-neutral factor trace across policy "
                "search, and accept only policies whose validation errors remain "
                "inside declared Wilson-score risk limits. Compute Wilson "
                "statistics, condition log-likelihoods, and consensus with "
                "fixed-precision decimal functions and a twelve-decimal "
                "half-even projection before thresholding and hashing. Bind "
                "replay to the complete searchable policy-projection lattice "
                "and its diagnostic trace rather than platform-specific math "
                "library tails. Bind evaluator evidence to rounded semantic "
                "measurements rather than raw arrays."
            ),
            (
                "context factor graph",
                "pairwise causal motif",
                "risk limiting calibration",
                "selective error upper bound",
                "epistemic immune gate",
                "portable projection lattice digest",
                "semantic evaluator provenance",
            ),
        )
    )
    return tuple(nodes)


def _terms(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))


class CausalWorldWikiGraph:
    """Executable architecture graph bound to the learned causal-law graph."""

    def __init__(self, graph: CausalGraph) -> None:
        self.graph = graph
        self.formal_manifest = formal_domain_manifest()
        nodes = _static_nodes()
        self.nodes = {node.name: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("causal knowledge node names must be unique")
        self.composition = {
            name: tuple(values) for name, values in COMPONENT_COMPOSITION.items()
        }
        if set(self.composition) != set(ARCHITECTURE_COMPONENTS):
            raise ValueError("every causal architecture component needs a composition")
        for component in self.composition:
            self.expand(component)
        self.formal_domains = {
            f"formal_domain_{domain}": tuple(
                primitive.name
                for primitive in FORMAL_PRIMITIVES
                if primitive.domain == domain
            )
            for domain in FORMAL_DOMAIN_NAMES
        }
        self.formal_composition = {
            primitive.name: primitive.root_mechanics
            for primitive in FORMAL_PRIMITIVES
        }
        self._law_terms: dict[str, frozenset[str]] = {}
        self._law_term_index: dict[str, set[str]] = {}
        self._law_domain_index: dict[str, set[str]] = {}
        self._law_cause_index: dict[str, set[str]] = {}
        self._law_effect_index: dict[str, set[str]] = {}
        for law in self.graph.laws.values():
            terms = _terms(
                " ".join(
                    (
                        law.domain,
                        law.cause_feature,
                        law.effect_feature,
                        *(
                            context.split(":", 1)[-1]
                            for context in law.contexts
                        ),
                        *law.atom_program,
                    )
                )
            )
            self._law_terms[law.law_id] = terms
            self._law_domain_index.setdefault(law.domain, set()).add(law.law_id)
            self._law_cause_index.setdefault(law.cause_feature, set()).add(law.law_id)
            self._law_effect_index.setdefault(law.effect_feature, set()).add(
                law.law_id
            )
            for term in terms:
                self._law_term_index.setdefault(term, set()).add(law.law_id)

    def expand(self, name: str) -> tuple[str, ...]:
        if name in ROOT_MECHANICS:
            return (name,)
        if name not in self.composition:
            raise ValueError(f"knowledge node is not executable: {name}")
        leaves: list[str] = []
        for child in self.composition[name]:
            if child not in ROOT_MECHANICS:
                raise ValueError(f"unknown root mechanic in {name}: {child}")
            leaves.append(child)
        return tuple(leaves)

    def neighbors(self, name: str) -> tuple[str, ...]:
        if name not in self.nodes:
            raise ValueError(f"unknown causal knowledge node: {name}")
        outgoing = set(self.composition.get(name, ()))
        outgoing.update(self.formal_domains.get(name, ()))
        outgoing.update(self.formal_composition.get(name, ()))
        incoming = {
            component for component, roots in self.composition.items() if name in roots
        }
        incoming.update(
            primitive
            for primitive, roots in self.formal_composition.items()
            if name in roots
        )
        incoming.update(
            domain
            for domain, primitives in self.formal_domains.items()
            if name in primitives
        )
        if name in DOMAIN_NAMES:
            outgoing.update(DOMAIN_MECHANISMS[name])
        return tuple(sorted(outgoing | incoming))

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        *,
        domain: str | None = None,
        cause_feature: str | None = None,
        effect_feature: str | None = None,
    ) -> tuple[CausalKnowledgeHit, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("causal RAG query must be non-empty text")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 32
        ):
            raise ValueError("causal RAG limit must be within [1, 32]")
        query_terms = _terms(query)
        hits: list[CausalKnowledgeHit] = []
        for name, node in self.nodes.items():
            score = (
                4.0 * len(query_terms & _terms(name))
                + 2.0 * len(query_terms & _terms(" ".join(node.aliases)))
                + len(query_terms & _terms(node.description))
            )
            if score > 0.0:
                hits.append(
                    CausalKnowledgeHit(
                        name=name,
                        kind=node.kind,
                        score=float(score),
                        description=node.description,
                        neighbors=self.neighbors(name),
                        evidence={"source": "architecture_wiki"},
                    )
                )
        candidate_law_ids: set[str] = set()
        for term in query_terms:
            candidate_law_ids.update(self._law_term_index.get(term, ()))
        for value, index in (
            (domain, self._law_domain_index),
            (cause_feature, self._law_cause_index),
            (effect_feature, self._law_effect_index),
        ):
            if value is not None:
                candidate_law_ids.intersection_update(index.get(value, ()))
        for law_id in sorted(candidate_law_ids):
            law = self.graph.laws[law_id]
            law_terms = self._law_terms[law_id]
            overlap = len(query_terms & law_terms)
            if overlap <= 0:
                continue
            score = 3.0 * overlap + 2.0 * law.confidence + law.persistence
            hits.append(
                CausalKnowledgeHit(
                    name=law.law_id,
                    kind="learned_causal_law",
                    score=float(score),
                    description=(
                        f"{law.cause_feature} causes {law.effect_feature} to "
                        f"{'increase' if law.direction > 0 else 'decrease'} in "
                        f"the {law.domain} domain under "
                        + ", ".join(law_condition_signature(law))
                        + "."
                    ),
                    neighbors=(
                        f"feature:{law.cause_feature}",
                        f"feature:{law.effect_feature}",
                        f"domain:{law.domain}",
                    ),
                    evidence={
                        "confidence": law.confidence,
                        "persistence": law.persistence,
                        "support": law.support,
                        "contradictions": law.contradictions,
                        "condition_signature": list(law_condition_signature(law)),
                        "provenance_hashes": list(law.provenance_hashes),
                    },
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.name))
        return tuple(hits[:limit])

    def manifest(self) -> dict[str, Any]:
        graph_model = self.graph.model_payload()
        return {
            "wiki_runtime": CAUSAL_WORLD_WIKI_RUNTIME,
            "rag_runtime": CAUSAL_WORLD_RAG_RUNTIME,
            "nodes": [asdict(self.nodes[name]) for name in sorted(self.nodes)],
            "composition": {
                name: {
                    "steps": list(self.composition[name]),
                    "root_leaves": list(self.expand(name)),
                }
                for name in sorted(self.composition)
            },
            "live_causal_graph": {
                "model_hash": graph_model["model_hash"],
                "law_count": len(self.graph.laws),
                "observation_count": self.graph.observation_count,
            },
            "formal_domains": self.formal_manifest,
            "world_program_space": world_program_space_size(),
        }


def retrieve_causal_context(
    wiki: CausalWorldWikiGraph,
    query: str,
    limit: int = 8,
    *,
    domain: str | None = None,
    cause_feature: str | None = None,
    effect_feature: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            **asdict(hit),
            "neighbors": list(hit.neighbors),
            "evidence": dict(hit.evidence),
        }
        for hit in wiki.retrieve(
            query,
            limit=limit,
            domain=domain,
            cause_feature=cause_feature,
            effect_feature=effect_feature,
        )
    ]


def causal_world_knowledge_self_test() -> dict[str, bool]:
    graph = CausalGraph(maximum_laws=64)
    wiki = CausalWorldWikiGraph(graph)
    hits = retrieve_causal_context(
        wiki, "phase synchronization and causal intervention", limit=8
    )
    transfer_hits = retrieve_causal_context(
        wiki, "held out causal transfer in an unseen regime", limit=8
    )
    calibration_hits = retrieve_causal_context(
        wiki, "metaplastic calibration on validation worlds", limit=8
    )
    risk_hits = retrieve_causal_context(
        wiki, "pairwise context factor risk limit", limit=8
    )
    formal_hits = retrieve_causal_context(
        wiki,
        "algebra derivative geometry stoichiometric entropy Mendelian",
        limit=16,
    )
    manifest = wiki.manifest()
    checks = {
        "all_components_expand_to_roots": all(
            set(wiki.expand(component)) <= set(ROOT_MECHANICS)
            for component in ARCHITECTURE_COMPONENTS
        ),
        "rag_returns_context": bool(hits),
        "rag_exposes_contextual_transfer": any(
            hit["name"] == "contextual_causal_transfer" for hit in transfer_hits
        ),
        "rag_exposes_metaplastic_calibration": any(
            hit["name"] == "metaplastic_transfer_governor"
            for hit in calibration_hits
        ),
        "rag_exposes_context_factor_risk": any(
            hit["name"] == "context_factor_risk_governor"
            for hit in risk_hits
        ),
        "wiki_bound_to_live_graph": manifest["live_causal_graph"]["law_count"] == 0,
        "manifest_hashable": len(canonical_hash(manifest)) == 64,
        "runtime_markers_present": (
            manifest["wiki_runtime"] == CAUSAL_WORLD_WIKI_RUNTIME
            and manifest["rag_runtime"] == CAUSAL_WORLD_RAG_RUNTIME
        ),
        "curriculum_axes_are_in_the_wiki": all(
            f"world_regime_{axis}" in wiki.nodes for axis, _ in WORLD_PROGRAM_AXES
        ),
        "formal_domains_are_in_the_wiki": all(
            f"formal_domain_{domain}" in wiki.nodes for domain in FORMAL_DOMAIN_NAMES
        ),
        "formal_primitives_are_executable_neighbors": all(
            primitive.name
            in wiki.neighbors(f"formal_domain_{primitive.domain}")
            for primitive in FORMAL_PRIMITIVES
        ),
        "rag_exposes_formal_primitives": len(
            {
                hit["name"]
                for hit in formal_hits
                if hit["kind"] == "formal_primitive"
            }
        )
        >= 4,
        "formal_registry_is_hash_bound": len(
            manifest["formal_domains"]["registry_hash"]
        )
        == 64,
        "world_program_space_is_massive": manifest["world_program_space"] > 50_000_000,
    }
    if not all(checks.values()):
        raise AssertionError(f"causal-world knowledge self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(causal_world_knowledge_self_test())
