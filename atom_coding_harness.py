"""Causal coding harness with isolated code interventions and hidden evaluation."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import canonical_hash
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    PLATFORM_PRIMITIVES,
    PlatformSpec,
    build_platform_blueprint,
    compile_platform_source,
)


CODING_HARNESS_SCHEMA = 1
CODING_HARNESS_RUNTIME = "atom-causal-coding-harness-v1"
CODE_GRAPH_RUNTIME = "atom-code-structure-graph-v1"
CODE_INTERVENTION_RUNTIME = "atom-isolated-code-intervention-v1"
CODE_CAUSAL_MEMORY_RUNTIME = "atom-code-causal-memory-v1"
CODE_MOLECULAR_RECOGNITION_RUNTIME = "atom-code-molecular-recognition-v1"
CODE_PHASE_LOCKED_LOOP_RUNTIME = "atom-code-phase-locked-loop-v1"
CODE_PHASE_MIXER_RUNTIME = "atom-code-phase-mixer-v1"
CODE_TOPOLOGICAL_PERSISTENCE_RUNTIME = "atom-code-topological-persistence-v1"
CODE_THERMAL_ANNEALING_RUNTIME = "atom-code-thermal-annealing-v1"
CODE_PROJECTIVE_MEASUREMENT_RUNTIME = "atom-code-projective-measurement-v1"


@dataclass(frozen=True)
class CodeNode:
    node_id: str
    kind: str
    label: str
    line: int


@dataclass(frozen=True)
class CodeEdge:
    source: str
    target: str
    relation: str


class CodeStructureGraph:
    """AST-derived structure graph for a concrete produced code artifact."""

    def __init__(
        self,
        nodes: Sequence[CodeNode],
        edges: Sequence[CodeEdge],
        *,
        source_hash: str,
    ) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.source_hash = source_hash
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("code graph node IDs must be unique")

    @classmethod
    def from_source(
        cls, source: str, *, filename: str = "generated_platform.py"
    ) -> "CodeStructureGraph":
        if not isinstance(source, str) or not source.strip():
            raise ValueError("code source must be non-empty text")
        tree = ast.parse(source, filename=filename)
        nodes: list[CodeNode] = [
            CodeNode(f"file:{filename}", "file", filename, 1)
        ]
        edges: list[CodeEdge] = []
        definitions: dict[str, str] = {}
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(item, ast.ClassDef) else "function"
                node_id = f"{kind}:{item.name}"
                definitions[item.name] = node_id
                nodes.append(CodeNode(node_id, kind, item.name, item.lineno))
                edges.append(CodeEdge(f"file:{filename}", node_id, "defines"))
        for item in ast.walk(tree):
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            source_id = f"function:{item.name}"
            for child in ast.walk(item):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    target = definitions.get(child.func.id)
                    if target is not None:
                        edges.append(CodeEdge(source_id, target, "calls"))
        return cls(
            sorted(nodes, key=lambda node: node.node_id),
            sorted(
                set(edges),
                key=lambda edge: (edge.source, edge.target, edge.relation),
            ),
            source_hash=canonical_hash({"source": source}),
        )

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema": CODING_HARNESS_SCHEMA,
            "runtime": CODE_GRAPH_RUNTIME,
            "source_hash": self.source_hash,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }
        return {**core, "graph_hash": canonical_hash(core)}


@dataclass(frozen=True)
class InterventionRun:
    request_hash: str
    response: Mapping[str, Any] | None
    return_code: int
    stderr: str
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and not self.timed_out and self.response is not None


class IsolatedPythonInterventionRunner:
    """Execute a candidate artifact in a disposable process and directory."""

    runtime = CODE_INTERVENTION_RUNTIME

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or not 0.5 <= timeout_seconds <= 30.0:
            raise ValueError("intervention timeout must be within [0.5, 30] seconds")
        self.timeout_seconds = float(timeout_seconds)

    def run(self, source: str, request: Mapping[str, Any]) -> InterventionRun:
        request_text = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with tempfile.TemporaryDirectory(prefix="atom-code-intervention-") as temporary:
            candidate = Path(temporary) / "candidate_platform.py"
            candidate.write_text(source, encoding="utf-8", newline="\n")
            try:
                process = subprocess.run(
                    [sys.executable, "-I", str(candidate)],
                    input=request_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    cwd=temporary,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return InterventionRun(
                    request_hash=canonical_hash(request),
                    response=None,
                    return_code=-1,
                    stderr=str(error),
                    timed_out=True,
                )
        response: Mapping[str, Any] | None = None
        stderr = process.stderr.strip()
        if process.returncode == 0:
            try:
                parsed = json.loads(process.stdout)
                if isinstance(parsed, Mapping):
                    response = dict(parsed)
                else:
                    stderr = "candidate response was not a JSON object"
            except json.JSONDecodeError as error:
                stderr = f"candidate response was invalid JSON: {error}"
        return InterventionRun(
            request_hash=canonical_hash(request),
            response=response,
            return_code=process.returncode,
            stderr=stderr,
            timed_out=False,
        )


@dataclass(frozen=True)
class CapabilityEvaluation:
    capability: str
    passed: bool
    request_hash: str
    response_hash: str | None
    error: str | None


@dataclass(frozen=True)
class PlatformEvaluation:
    spec_id: str
    passed: bool
    score: float
    capabilities: tuple[CapabilityEvaluation, ...]
    evaluator_hash: str

    def manifest(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "passed": self.passed,
            "score": self.score,
            "capabilities": [asdict(item) for item in self.capabilities],
            "evaluator_hash": self.evaluator_hash,
        }


class HiddenPlatformEvaluator:
    """Sealed behavioral evaluator; expected answers never enter candidate code."""

    _CASES: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {
        "typed_messages": (
            {"action": "typed_message", "payload": {"value": {"kind": "event", "id": 7}}},
            {"preserved": True, "value": {"kind": "event", "id": 7}},
        ),
        "directed_routing": (
            {
                "action": "route",
                "payload": {"path": ["source", "worker", "sink"], "parallel": False},
            },
            {"delivered": True},
        ),
        "parallel_promotion": (
            {
                "action": "route",
                "payload": {"path": ["source", "worker", "sink"], "parallel": True},
            },
            {"promoted": True, "off_ramp": "sink", "preloaded": "worker"},
        ),
        "bounded_capacity": (
            {"action": "capacity", "payload": {"load": 9, "capacity": 4}},
            {"accepted": 4, "bounded": True},
        ),
        "priority_scheduling": (
            {
                "action": "priority",
                "payload": {
                    "tasks": [
                        {"id": "low", "priority": 1},
                        {"id": "high", "priority": 9},
                        {"id": "mid", "priority": 4},
                    ]
                },
            },
            {"task_ids": ["high", "mid", "low"]},
        ),
        "backpressure": (
            {"action": "backpressure", "payload": {"load": 9, "capacity": 4}},
            {"signal": "slow_down", "vertical_vibration": True},
        ),
        "bounded_retries": (
            {"action": "retry", "payload": {"success_after": 3, "maximum": 4}},
            {"attempts": 3, "success": True},
        ),
        "emergent_topology": (
            {
                "action": "route",
                "payload": {
                    "path": ["source", "junction", "sink"],
                    "parallel": False,
                    "prior_threads": [["other", "junction", "archive"]],
                },
            },
            {
                "thread": ["source", "junction", "sink"],
                "intersections": [["junction"]],
            },
        ),
        "discrete_output": (
            {
                "action": "project",
                "payload": {
                    "candidates": [
                        {"value": "reject", "support": 0.2},
                        {"value": "accept", "support": 0.9},
                    ]
                },
            },
            {"status": "derived", "value": "accept"},
        ),
    }

    def __init__(self, runner: IsolatedPythonInterventionRunner) -> None:
        self.runner = runner
        self._truth_hash = canonical_hash(
            {
                capability: {"request": request, "expected": expected}
                for capability, (request, expected) in self._CASES.items()
            }
        )

    @property
    def truth_hash(self) -> str:
        return self._truth_hash

    def evaluate_capability(
        self, source: str, capability: str
    ) -> CapabilityEvaluation:
        if capability not in self._CASES:
            raise ValueError(f"unknown evaluator capability: {capability}")
        request, expected = self._CASES[capability]
        run = self.runner.run(source, request)
        response = dict(run.response) if run.response is not None else None
        passed = bool(
            run.succeeded
            and response is not None
            and all(response.get(key) == value for key, value in expected.items())
        )
        return CapabilityEvaluation(
            capability=capability,
            passed=passed,
            request_hash=run.request_hash,
            response_hash=canonical_hash(response) if response is not None else None,
            error=None if run.succeeded else run.stderr or "candidate execution failed",
        )

    def evaluate(self, source: str, spec: PlatformSpec) -> PlatformEvaluation:
        spec.validate()
        outcomes = tuple(
            self.evaluate_capability(source, capability)
            for capability in spec.capabilities
        )
        score = sum(outcome.passed for outcome in outcomes) / len(outcomes)
        core = {
            "truth_hash": self.truth_hash,
            "spec_id": spec.spec_id,
            "capabilities": [
                {
                    "capability": outcome.capability,
                    "passed": outcome.passed,
                    "request_hash": outcome.request_hash,
                    "response_hash": outcome.response_hash,
                }
                for outcome in outcomes
            ],
        }
        return PlatformEvaluation(
            spec_id=spec.spec_id,
            passed=all(outcome.passed for outcome in outcomes),
            score=score,
            capabilities=outcomes,
            evaluator_hash=canonical_hash(core),
        )


@dataclass(frozen=True)
class CodeInterventionEvidence:
    evidence_id: str
    requirement: str
    primitive: str
    control_score: float
    treated_score: float
    context: str
    control_source_hash: str
    treated_source_hash: str
    provenance_hash: str

    @property
    def effect(self) -> float:
        return self.treated_score - self.control_score

    def validate(self) -> None:
        if self.requirement not in PLATFORM_CAPABILITY_PRIMITIVES:
            raise ValueError("code evidence requirement is unknown")
        if self.primitive not in {
            primitive.name for primitive in PLATFORM_PRIMITIVES
        }:
            raise ValueError("code evidence primitive is unknown")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.control_score, self.treated_score)
        ):
            raise ValueError("code evidence scores are invalid")
        if not self.context:
            raise ValueError("code evidence context cannot be empty")
        for digest in (
            self.control_source_hash,
            self.treated_source_hash,
            self.provenance_hash,
        ):
            if len(digest) != 64:
                raise ValueError("code evidence hash is invalid")


@dataclass
class CodeCausalLaw:
    requirement: str
    primitive: str
    support: int = 0
    contradictions: int = 0
    effect_mean: float = 0.0
    confidence: float = 0.0
    persistence: float = 0.0
    status: str = "hypothesis"
    contexts: set[str] = field(default_factory=set)
    evidence_ids: list[str] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "primitive": self.primitive,
            "support": self.support,
            "contradictions": self.contradictions,
            "effect_mean": round(self.effect_mean, 12),
            "confidence": round(self.confidence, 12),
            "persistence": round(self.persistence, 12),
            "status": self.status,
            "contexts": sorted(self.contexts),
            "evidence_ids": list(self.evidence_ids),
        }


class CodeCausalGraph:
    """Persistent requirement-to-primitive laws learned from executed patches."""

    runtime = CODE_CAUSAL_MEMORY_RUNTIME

    def __init__(self) -> None:
        self.laws: dict[tuple[str, str], CodeCausalLaw] = {}
        self.observation_count = 0

    def observe(self, evidence: CodeInterventionEvidence) -> CodeCausalLaw:
        evidence.validate()
        key = (evidence.requirement, evidence.primitive)
        law = self.laws.setdefault(
            key,
            CodeCausalLaw(
                requirement=evidence.requirement,
                primitive=evidence.primitive,
            ),
        )
        positive = evidence.effect > 0.5
        if positive:
            previous = law.support
            law.support += 1
            law.effect_mean = (
                law.effect_mean * previous + evidence.effect
            ) / law.support
        else:
            law.contradictions += 1
        law.contexts.add(evidence.context)
        if evidence.evidence_id not in law.evidence_ids:
            law.evidence_ids.append(evidence.evidence_id)
        total = law.support + law.contradictions
        law.confidence = (law.support + 1.0) / (total + 2.0)
        law.persistence = min(
            1.0,
            0.18 * law.support
            + 0.08 * len(law.contexts)
            - 0.10 * law.contradictions,
        )
        if law.support >= 3 and law.confidence >= 0.72 and law.persistence >= 0.70:
            law.status = "crystallized"
        elif law.contradictions >= 3 and law.support == 0:
            law.status = "retired"
        else:
            law.status = "hypothesis"
        self.observation_count += 1
        return law

    def retrieve(self, requirement: str) -> tuple[CodeCausalLaw, ...]:
        if requirement not in PLATFORM_CAPABILITY_PRIMITIVES:
            return ()
        candidates = [
            law
            for law in self.laws.values()
            if law.requirement == requirement and law.status != "retired"
        ]
        candidates.sort(
            key=lambda law: (
                law.status != "crystallized",
                -law.persistence,
                -law.confidence,
                -law.effect_mean,
                law.primitive,
            )
        )
        return tuple(candidates)

    def recognize(
        self,
        requirements: Sequence[str],
    ) -> dict[str, tuple[str, ...]] | None:
        """Return every crystallized primitive required by each capability.

        Removal interventions can identify conjunctive causes. A capability such
        as emergent topology needs both a directed route and topology formation,
        so recognition must preserve all supported necessary causes.
        """

        recognized: dict[str, tuple[str, ...]] = {}
        for requirement in requirements:
            candidates = self.retrieve(requirement)
            supported = [
                law
                for law in candidates
                if law.status == "crystallized"
                and law.confidence >= 0.72
                and law.persistence >= 0.70
            ]
            if not supported:
                return None
            recognized[requirement] = tuple(
                sorted(law.primitive for law in supported)
            )
        return recognized

    def model_payload(self) -> dict[str, Any]:
        core = {
            "schema": CODING_HARNESS_SCHEMA,
            "runtime": CODE_CAUSAL_MEMORY_RUNTIME,
            "observation_count": self.observation_count,
            "laws": [
                self.laws[key].manifest()
                for key in sorted(self.laws)
            ],
        }
        return {**core, "model_hash": canonical_hash(core)}

    @classmethod
    def from_model_payload(cls, payload: Mapping[str, Any]) -> "CodeCausalGraph":
        expected = {
            "schema",
            "runtime",
            "observation_count",
            "laws",
            "model_hash",
        }
        if set(payload) != expected:
            raise ValueError("code causal model fields are invalid")
        core = {key: payload[key] for key in expected - {"model_hash"}}
        if payload["model_hash"] != canonical_hash(core):
            raise ValueError("code causal model hash mismatch")
        if (
            payload["schema"] != CODING_HARNESS_SCHEMA
            or payload["runtime"] != CODE_CAUSAL_MEMORY_RUNTIME
        ):
            raise ValueError("code causal model contract mismatch")
        graph = cls()
        graph.observation_count = int(payload["observation_count"])
        for saved in payload["laws"]:
            law = CodeCausalLaw(
                requirement=str(saved["requirement"]),
                primitive=str(saved["primitive"]),
                support=int(saved["support"]),
                contradictions=int(saved["contradictions"]),
                effect_mean=float(saved["effect_mean"]),
                confidence=float(saved["confidence"]),
                persistence=float(saved["persistence"]),
                status=str(saved["status"]),
                contexts=set(str(value) for value in saved["contexts"]),
                evidence_ids=[str(value) for value in saved["evidence_ids"]],
            )
            graph.laws[(law.requirement, law.primitive)] = law
        return graph


def train_code_causal_graph(
    training_specs: Sequence[PlatformSpec],
    evaluator: HiddenPlatformEvaluator,
) -> tuple[CodeCausalGraph, tuple[CodeInterventionEvidence, ...]]:
    graph = CodeCausalGraph()
    evidence_rows: list[CodeInterventionEvidence] = []
    all_names = tuple(primitive.name for primitive in PLATFORM_PRIMITIVES)
    for spec in training_specs:
        spec.validate()
        if spec.partition != "training" or len(spec.capabilities) != 1:
            raise ValueError("code causal training requires isolated training specs")
        capability = spec.capabilities[0]
        for primitive in all_names:
            control_names = tuple(name for name in all_names if name != primitive)
            treated_names = all_names
            control_blueprint = build_platform_blueprint(
                spec,
                control_names,
                blueprint_label=f"{spec.spec_id}:without:{primitive}",
            )
            treated_blueprint = build_platform_blueprint(
                spec,
                treated_names,
                blueprint_label=f"{spec.spec_id}:with:{primitive}",
            )
            control_source = compile_platform_source(control_blueprint)
            treated_source = compile_platform_source(treated_blueprint)
            control = evaluator.evaluate_capability(control_source, capability)
            treated = evaluator.evaluate_capability(treated_source, capability)
            provenance = {
                "spec_id": spec.spec_id,
                "capability": capability,
                "primitive": primitive,
                "control_request_hash": control.request_hash,
                "treated_request_hash": treated.request_hash,
                "truth_hash": evaluator.truth_hash,
            }
            evidence = CodeInterventionEvidence(
                evidence_id=f"code-evidence:{canonical_hash(provenance)[:24]}",
                requirement=capability,
                primitive=primitive,
                control_score=float(control.passed),
                treated_score=float(treated.passed),
                context=f"partition:{spec.partition}|spec:{spec.spec_id}",
                control_source_hash=CodeStructureGraph.from_source(
                    control_source
                ).source_hash,
                treated_source_hash=CodeStructureGraph.from_source(
                    treated_source
                ).source_hash,
                provenance_hash=canonical_hash(provenance),
            )
            graph.observe(evidence)
            evidence_rows.append(evidence)
    return graph, tuple(evidence_rows)


def baseline_primitives() -> tuple[str, ...]:
    """Conventional minimal platform assumption without learned causal memory."""

    return ("identity", "directed_relation", "projection")


def synthesize_with_atom(
    graph: CodeCausalGraph,
    spec: PlatformSpec,
) -> tuple[str, ...] | None:
    spec.validate()
    recognized = graph.recognize(spec.capabilities)
    if recognized is None:
        return None
    selected = {
        primitive
        for primitives in recognized.values()
        for primitive in primitives
    }
    return tuple(
        primitive.name
        for primitive in PLATFORM_PRIMITIVES
        if primitive.name in selected
    )


def coding_harness_self_test() -> dict[str, bool]:
    runner = IsolatedPythonInterventionRunner()
    evaluator = HiddenPlatformEvaluator(runner)
    spec = PlatformSpec(
        "self-test",
        ("typed_messages", "directed_routing", "discrete_output"),
        "heldout",
        "Self-test platform",
    )
    blueprint = build_platform_blueprint(
        spec,
        ("identity", "directed_relation", "projection"),
        blueprint_label="self-test",
    )
    source = compile_platform_source(blueprint)
    graph = CodeStructureGraph.from_source(source)
    evaluation = evaluator.evaluate(source, spec)
    return {
        "isolated_execution": evaluation.passed,
        "code_graph_has_definitions": any(
            edge.relation == "defines" for edge in graph.edges
        ),
        "artifact_is_hash_bound": graph.manifest()["graph_hash"]
        == canonical_hash(
            {
                key: value
                for key, value in graph.manifest().items()
                if key != "graph_hash"
            }
        ),
        "hidden_truth_is_hash_bound": len(evaluator.truth_hash) == 64,
        "runtime_components_declared": len(
            {
                CODE_MOLECULAR_RECOGNITION_RUNTIME,
                CODE_PHASE_LOCKED_LOOP_RUNTIME,
                CODE_PHASE_MIXER_RUNTIME,
                CODE_TOPOLOGICAL_PERSISTENCE_RUNTIME,
                CODE_THERMAL_ANNEALING_RUNTIME,
                CODE_PROJECTIVE_MEASUREMENT_RUNTIME,
            }
        )
        == 6,
    }
