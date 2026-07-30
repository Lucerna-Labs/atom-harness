"""Atom-first causal construction language, parser, IR, and interpreter."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash
from atom_coding_harness import (
    CodeCausalGraph,
    CodeInterventionEvidence,
)
from atom_platform_synthesis import (
    PLATFORM_CAPABILITY_PRIMITIVES,
    PLATFORM_PRIMITIVES,
    PLATFORM_PRIMITIVE_INDEX,
    SPIDERWEB_LAYERS,
    PlatformBlueprint,
    PlatformSpec,
    build_platform_blueprint,
)


ATOM_LANGUAGE_SCHEMA = 1
ATOM_LANGUAGE_RUNTIME = "atom-causal-construction-language-v1"
ATOM_IR_RUNTIME = "atom-typed-causal-ir-v1"
ATOM_INTERPRETER_RUNTIME = "atom-reference-interpreter-v1"
ATOM_TRAINING_RUNTIME = "atom-native-causal-training-v1"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

LAYER_PRIMITIVES: Mapping[str, tuple[str, ...]] = {
    "layer_0_transport": (
        "identity",
        "directed_relation",
        "conservation",
    ),
    "layer_1_message": (
        "identity",
        "directed_relation",
        "projection",
    ),
    "layer_2_flow": (
        "composition",
        "feedback",
        "topology",
    ),
    "layer_3_orchestration": (
        "ordering",
        "fixed_point",
        "projection",
        "conservation",
    ),
}


@dataclass(frozen=True)
class AtomPrimitiveDeclaration:
    name: str
    roots: tuple[str, ...]


@dataclass(frozen=True)
class AtomCapabilityBinding:
    capability: str
    primitives: tuple[str, ...]


@dataclass(frozen=True)
class AtomLayerDeclaration:
    name: str
    primitives: tuple[str, ...]


@dataclass(frozen=True)
class AtomProgram:
    """Typed causal IR produced by parsing Atom source."""

    name: str
    roots: tuple[str, ...]
    primitives: tuple[AtomPrimitiveDeclaration, ...]
    capabilities: tuple[AtomCapabilityBinding, ...]
    layers: tuple[AtomLayerDeclaration, ...]

    def validate(self) -> None:
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("Atom platform name must be a lowercase identifier")
        if self.roots != ROOT_MECHANICS:
            raise ValueError("Atom programs must declare the seven roots canonically")
        primitive_names = tuple(item.name for item in self.primitives)
        if len(set(primitive_names)) != len(primitive_names):
            raise ValueError("Atom primitive declarations must be unique")
        unknown_primitives = set(primitive_names) - set(PLATFORM_PRIMITIVE_INDEX)
        if unknown_primitives:
            raise ValueError(
                f"Atom program uses unsupported primitives: {sorted(unknown_primitives)}"
            )
        for declaration in self.primitives:
            expected = PLATFORM_PRIMITIVE_INDEX[declaration.name].root_mechanics
            if declaration.roots != expected:
                raise ValueError(
                    f"Atom primitive {declaration.name} has invalid root expansion"
                )
        capability_names = tuple(item.capability for item in self.capabilities)
        if len(set(capability_names)) != len(capability_names):
            raise ValueError("Atom capability declarations must be unique")
        unknown_capabilities = set(capability_names) - set(
            PLATFORM_CAPABILITY_PRIMITIVES
        )
        if unknown_capabilities:
            raise ValueError(
                f"Atom program uses unsupported capabilities: "
                f"{sorted(unknown_capabilities)}"
            )
        declared = set(primitive_names)
        for binding in self.capabilities:
            if not binding.primitives or set(binding.primitives) - declared:
                raise ValueError(
                    f"Atom capability {binding.capability} has invalid bindings"
                )
        if tuple(layer.name for layer in self.layers) != SPIDERWEB_LAYERS:
            raise ValueError("Atom program must declare the four Spiderweb layers")
        for layer in self.layers:
            if set(layer.primitives) - declared:
                raise ValueError(f"Atom layer {layer.name} has invalid primitives")

    @property
    def primitive_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.primitives)

    def binding_for(self, capability: str) -> tuple[str, ...]:
        for binding in self.capabilities:
            if binding.capability == capability:
                return binding.primitives
        return ()

    def manifest(self) -> dict[str, Any]:
        self.validate()
        core = {
            "schema": ATOM_LANGUAGE_SCHEMA,
            "language_runtime": ATOM_LANGUAGE_RUNTIME,
            "ir_runtime": ATOM_IR_RUNTIME,
            "name": self.name,
            "roots": list(self.roots),
            "primitives": [
                {**asdict(item), "roots": list(item.roots)}
                for item in self.primitives
            ],
            "capabilities": [
                {
                    **asdict(item),
                    "primitives": list(item.primitives),
                }
                for item in self.capabilities
            ],
            "layers": [
                {
                    **asdict(item),
                    "primitives": list(item.primitives),
                }
                for item in self.layers
            ],
        }
        return {**core, "program_hash": canonical_hash(core)}


def _ordered_primitives(names: Sequence[str]) -> tuple[str, ...]:
    selected = set(names)
    unknown = selected - set(PLATFORM_PRIMITIVE_INDEX)
    if unknown:
        raise ValueError(f"unknown Atom primitives: {sorted(unknown)}")
    return tuple(
        primitive.name
        for primitive in PLATFORM_PRIMITIVES
        if primitive.name in selected
    )


def build_atom_program(
    spec: PlatformSpec,
    primitive_names: Sequence[str],
    *,
    learned_bindings: Mapping[str, Sequence[str]] | None = None,
) -> AtomProgram:
    """Build typed Atom IR without giving the learner evaluator truth."""

    spec.validate()
    ordered = _ordered_primitives(primitive_names)
    if not ordered:
        raise ValueError("Atom program needs at least one mathematical primitive")
    declarations = tuple(
        AtomPrimitiveDeclaration(
            name=name,
            roots=PLATFORM_PRIMITIVE_INDEX[name].root_mechanics,
        )
        for name in ordered
    )
    bindings: list[AtomCapabilityBinding] = []
    for capability in spec.capabilities:
        candidates = (
            tuple(learned_bindings[capability])
            if learned_bindings is not None
            else ordered
        )
        bound = tuple(name for name in ordered if name in set(candidates))
        if not bound:
            raise ValueError(f"Atom capability {capability} has no learned binding")
        bindings.append(AtomCapabilityBinding(capability, bound))
    layers = tuple(
        AtomLayerDeclaration(
            layer,
            tuple(
                primitive
                for primitive in LAYER_PRIMITIVES[layer]
                if primitive in set(ordered)
            ),
        )
        for layer in SPIDERWEB_LAYERS
    )
    program = AtomProgram(
        name=re.sub(r"[^a-z0-9_]", "_", spec.spec_id.lower()).strip("_"),
        roots=ROOT_MECHANICS,
        primitives=declarations,
        capabilities=tuple(bindings),
        layers=layers,
    )
    program.validate()
    return program


def render_atom_source(program: AtomProgram) -> str:
    """Render canonical, human-readable Atom source from the typed IR."""

    program.validate()
    lines = [
        f"atom_language {ATOM_LANGUAGE_SCHEMA}",
        f"platform {program.name}",
        "roots " + " ".join(program.roots),
    ]
    lines.extend(
        f"primitive {item.name} <- {' '.join(item.roots)}"
        for item in program.primitives
    )
    lines.extend(
        f"capability {item.capability} <- {' '.join(item.primitives)}"
        for item in program.capabilities
    )
    lines.extend(
        f"layer {item.name} <- "
        + (" ".join(item.primitives) if item.primitives else "none")
        for item in program.layers
    )
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_atom_source(source: str) -> AtomProgram:
    """Parse strict Atom text and fail closed on ambiguity or extra syntax."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Atom source must be non-empty text")
    lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 5 or lines[0] != f"atom_language {ATOM_LANGUAGE_SCHEMA}":
        raise ValueError("Atom language header is invalid")
    if not lines[1].startswith("platform "):
        raise ValueError("Atom platform declaration is missing")
    name = lines[1].removeprefix("platform ").strip()
    if not lines[2].startswith("roots "):
        raise ValueError("Atom root declaration is missing")
    roots = tuple(lines[2].removeprefix("roots ").split())
    if lines[-1] != "end":
        raise ValueError("Atom program must end explicitly")
    primitives: list[AtomPrimitiveDeclaration] = []
    capabilities: list[AtomCapabilityBinding] = []
    layers: list[AtomLayerDeclaration] = []
    phase = "primitive"
    for line in lines[3:-1]:
        if " <- " not in line:
            raise ValueError(f"Atom declaration has no composition arrow: {line}")
        head, raw_children = line.split(" <- ", maxsplit=1)
        children = tuple(raw_children.split())
        if head.startswith("primitive "):
            if phase != "primitive":
                raise ValueError("Atom primitive declarations are out of order")
            primitives.append(
                AtomPrimitiveDeclaration(
                    head.removeprefix("primitive ").strip(),
                    children,
                )
            )
        elif head.startswith("capability "):
            if phase == "layer":
                raise ValueError("Atom capability declarations are out of order")
            phase = "capability"
            capabilities.append(
                AtomCapabilityBinding(
                    head.removeprefix("capability ").strip(),
                    children,
                )
            )
        elif head.startswith("layer "):
            phase = "layer"
            layers.append(
                AtomLayerDeclaration(
                    head.removeprefix("layer ").strip(),
                    () if children == ("none",) else children,
                )
            )
        else:
            raise ValueError(f"unknown Atom declaration: {head}")
    program = AtomProgram(
        name=name,
        roots=roots,
        primitives=tuple(primitives),
        capabilities=tuple(capabilities),
        layers=tuple(layers),
    )
    program.validate()
    if render_atom_source(program) != source:
        raise ValueError("Atom source is not canonical")
    return program


class AtomInterpreter:
    """Reference semantics for the Atom language."""

    runtime = ATOM_INTERPRETER_RUNTIME

    def __init__(self, program: AtomProgram) -> None:
        program.validate()
        self.program = program

    def _has(self, capability: str, *primitives: str) -> bool:
        bound = set(self.program.binding_for(capability))
        return all(primitive in bound for primitive in primitives)

    def execute(
        self,
        capability: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        if capability not in {
            binding.capability for binding in self.program.capabilities
        }:
            raise ValueError("Atom program does not declare the capability")
        action = request.get("action")
        payload = request.get("payload")
        if set(request) != {"action", "payload"} or not isinstance(payload, Mapping):
            raise ValueError("Atom request fields are invalid")
        if capability == "typed_messages" and action == "typed_message":
            return {
                "preserved": self._has(capability, "identity"),
                "value": payload.get("value"),
            }
        if capability in {
            "directed_routing",
            "parallel_promotion",
            "emergent_topology",
        } and action == "route":
            path = payload.get("path")
            if (
                not isinstance(path, list)
                or len(path) < 2
                or not all(isinstance(item, str) for item in path)
            ):
                raise ValueError("Atom route path is invalid")
            delivered = self._has(capability, "directed_relation")
            promoted = bool(payload.get("parallel")) and self._has(
                capability,
                "directed_relation",
                "composition",
            )
            thread = (
                path
                if delivered
                and self._has(capability, "directed_relation", "topology")
                else []
            )
            intersections: list[list[str]] = []
            for prior in payload.get("prior_threads", []):
                if thread and isinstance(prior, list):
                    shared = sorted(set(thread) & set(prior))
                    if shared:
                        intersections.append(shared)
            return {
                "delivered": delivered,
                "promoted": promoted,
                "off_ramp": path[-1] if promoted else None,
                "preloaded": (
                    path[1]
                    if delivered and self._has(capability, "composition")
                    else None
                ),
                "thread": thread,
                "intersections": intersections,
            }
        if capability == "bounded_capacity" and action == "capacity":
            load = payload.get("load")
            capacity = payload.get("capacity")
            if (
                isinstance(load, bool)
                or not isinstance(load, int)
                or load < 0
                or isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity <= 0
            ):
                raise ValueError("Atom capacity values are invalid")
            accepted = (
                min(load, capacity)
                if self._has(capability, "conservation")
                else load
            )
            return {"accepted": accepted, "bounded": accepted <= capacity}
        if capability == "priority_scheduling" and action == "priority":
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                raise ValueError("Atom priority tasks are invalid")
            ordered = (
                sorted(
                    tasks,
                    key=lambda task: (-task["priority"], task["id"]),
                )
                if self._has(capability, "ordering")
                else tasks
            )
            return {"task_ids": [task["id"] for task in ordered]}
        if capability == "backpressure" and action == "backpressure":
            active = (
                payload.get("load", 0) > payload.get("capacity", 0)
                and self._has(capability, "feedback")
            )
            return {
                "signal": "slow_down" if active else "none",
                "vertical_vibration": active,
            }
        if capability == "bounded_retries" and action == "retry":
            success_after = payload.get("success_after")
            maximum = payload.get("maximum")
            if not isinstance(success_after, int) or not isinstance(maximum, int):
                raise ValueError("Atom retry values are invalid")
            attempts = (
                min(success_after, maximum)
                if self._has(capability, "fixed_point")
                else 1
            )
            return {
                "attempts": attempts,
                "success": attempts >= success_after,
            }
        if capability == "discrete_output" and action == "project":
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("Atom projection candidates are invalid")
            if not self._has(capability, "projection"):
                return {"status": "unknown", "value": None}
            selected = sorted(
                candidates,
                key=lambda item: (
                    -float(item.get("support", 0.0)),
                    str(item.get("value")),
                ),
            )[0]
            return {"status": "derived", "value": selected.get("value")}
        raise ValueError("Atom request does not match the declared capability")


ATOM_HIDDEN_CASES: Mapping[
    str,
    tuple[Mapping[str, Any], Mapping[str, Any]],
] = {
    "typed_messages": (
        {
            "action": "typed_message",
            "payload": {"value": {"kind": "event", "id": 7}},
        },
        {"preserved": True, "value": {"kind": "event", "id": 7}},
    ),
    "directed_routing": (
        {
            "action": "route",
            "payload": {
                "path": ["source", "worker", "sink"],
                "parallel": False,
            },
        },
        {"delivered": True},
    ),
    "parallel_promotion": (
        {
            "action": "route",
            "payload": {
                "path": ["source", "worker", "sink"],
                "parallel": True,
            },
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
        {
            "action": "backpressure",
            "payload": {"load": 9, "capacity": 4},
        },
        {"signal": "slow_down", "vertical_vibration": True},
    ),
    "bounded_retries": (
        {
            "action": "retry",
            "payload": {"success_after": 3, "maximum": 4},
        },
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


class AtomProgramEvaluator:
    """Sealed semantic evaluator for Atom programs."""

    runtime = "atom-hidden-semantic-evaluator-v1"

    def __init__(self) -> None:
        self.truth_hash = canonical_hash(
            {
                capability: {"request": request, "expected": expected}
                for capability, (request, expected) in ATOM_HIDDEN_CASES.items()
            }
        )

    def evaluate_capability(
        self,
        program: AtomProgram,
        capability: str,
    ) -> dict[str, Any]:
        request, expected = ATOM_HIDDEN_CASES[capability]
        try:
            response = AtomInterpreter(program).execute(capability, request)
            passed = all(
                response.get(key) == value for key, value in expected.items()
            )
            error = None
        except (TypeError, ValueError, KeyError) as exc:
            response = {}
            passed = False
            error = str(exc)
        return {
            "capability": capability,
            "passed": passed,
            "request_hash": canonical_hash(request),
            "response_hash": canonical_hash(response),
            "error": error,
        }

    def evaluate(self, program: AtomProgram, spec: PlatformSpec) -> dict[str, Any]:
        outcomes = [
            self.evaluate_capability(program, capability)
            for capability in spec.capabilities
        ]
        core = {
            "runtime": self.runtime,
            "spec_id": spec.spec_id,
            "truth_hash": self.truth_hash,
            "capabilities": outcomes,
            "passed": all(item["passed"] for item in outcomes),
            "score": sum(item["passed"] for item in outcomes) / len(outcomes),
        }
        return {**core, "evaluation_hash": canonical_hash(core)}


def compile_atom_blueprint(
    spec: PlatformSpec,
    primitive_names: Sequence[str],
    *,
    label: str,
    learned_bindings: Mapping[str, Sequence[str]] | None = None,
) -> tuple[PlatformBlueprint, AtomProgram, str]:
    blueprint = build_platform_blueprint(
        spec,
        primitive_names,
        blueprint_label=label,
    )
    program = build_atom_program(
        spec,
        blueprint.primitives,
        learned_bindings=learned_bindings,
    )
    source = render_atom_source(program)
    if parse_atom_source(source) != program:
        raise RuntimeError("Atom source did not survive canonical parsing")
    return blueprint, program, source


def train_atom_causal_graph(
    training_specs: Sequence[PlatformSpec],
) -> tuple[
    CodeCausalGraph,
    tuple[CodeInterventionEvidence, ...],
    AtomProgramEvaluator,
]:
    """Learn requirement laws through Atom-native removal interventions."""

    evaluator = AtomProgramEvaluator()
    graph = CodeCausalGraph()
    evidence_rows: list[CodeInterventionEvidence] = []
    all_names = tuple(primitive.name for primitive in PLATFORM_PRIMITIVES)
    for spec in training_specs:
        spec.validate()
        if spec.partition != "training" or len(spec.capabilities) != 1:
            raise ValueError("Atom training requires isolated training specs")
        capability = spec.capabilities[0]
        for primitive in all_names:
            control_names = tuple(
                name for name in all_names if name != primitive
            )
            _, control_program, control_source = compile_atom_blueprint(
                spec,
                control_names,
                label=f"atom:{spec.spec_id}:without:{primitive}",
            )
            _, treated_program, treated_source = compile_atom_blueprint(
                spec,
                all_names,
                label=f"atom:{spec.spec_id}:with:{primitive}",
            )
            control = evaluator.evaluate_capability(
                control_program,
                capability,
            )
            treated = evaluator.evaluate_capability(
                treated_program,
                capability,
            )
            provenance = {
                "runtime": ATOM_TRAINING_RUNTIME,
                "spec_id": spec.spec_id,
                "capability": capability,
                "primitive": primitive,
                "truth_hash": evaluator.truth_hash,
                "control_program_hash": control_program.manifest()["program_hash"],
                "treated_program_hash": treated_program.manifest()["program_hash"],
            }
            evidence = CodeInterventionEvidence(
                evidence_id=f"atom-evidence:{canonical_hash(provenance)[:24]}",
                requirement=capability,
                primitive=primitive,
                control_score=float(control["passed"]),
                treated_score=float(treated["passed"]),
                context=(
                    f"substrate:atom|partition:{spec.partition}|spec:{spec.spec_id}"
                ),
                control_source_hash=canonical_hash({"source": control_source}),
                treated_source_hash=canonical_hash({"source": treated_source}),
                provenance_hash=canonical_hash(provenance),
            )
            graph.observe(evidence)
            evidence_rows.append(evidence)
    return graph, tuple(evidence_rows), evaluator


def atom_language_self_test() -> dict[str, bool]:
    spec = PlatformSpec(
        "atom-self-test",
        tuple(PLATFORM_CAPABILITY_PRIMITIVES),
        "heldout",
        "Atom language self test",
    )
    _, program, source = compile_atom_blueprint(
        spec,
        tuple(PLATFORM_PRIMITIVE_INDEX),
        label="atom-language-self-test",
    )
    evaluation = AtomProgramEvaluator().evaluate(program, spec)
    return {
        "atom_is_primary_ir": program.manifest()["ir_runtime"] == ATOM_IR_RUNTIME,
        "canonical_round_trip": parse_atom_source(source) == program,
        "seven_roots_are_required": program.roots == ROOT_MECHANICS,
        "four_layers_are_required": len(program.layers) == 4,
        "all_hidden_semantics_pass": evaluation["passed"],
        "program_is_hash_bound": len(program.manifest()["program_hash"]) == 64,
    }
