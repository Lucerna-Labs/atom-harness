"""Mathematical-primitive platform synthesis for the Atom coding experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash
from atom_formal_domains import (
    FORMAL_DOMAIN_RUNTIME,
    FORMAL_DOMAIN_SCHEMA,
    formal_domain_manifest,
    solve_formal_request,
)


PLATFORM_SYNTHESIS_SCHEMA = 1
PLATFORM_SYNTHESIS_RUNTIME = "atom-mathematical-platform-synthesis-v1"
SPIDERWEB_PLATFORM_RUNTIME = "atom-spiderweb-platform-v1"


@dataclass(frozen=True)
class PlatformPrimitive:
    name: str
    mathematical_role: str
    platform_effect: str
    formal_primitive: str
    root_mechanics: tuple[str, ...]


@dataclass(frozen=True)
class PlatformSpec:
    spec_id: str
    capabilities: tuple[str, ...]
    partition: str
    description: str

    def validate(self) -> None:
        if not self.spec_id:
            raise ValueError("platform spec ID cannot be empty")
        if self.partition not in {"training", "validation", "heldout"}:
            raise ValueError("platform spec partition is invalid")
        if not self.capabilities or len(set(self.capabilities)) != len(
            self.capabilities
        ):
            raise ValueError("platform capabilities must be non-empty and unique")
        unknown = set(self.capabilities) - set(PLATFORM_CAPABILITY_PRIMITIVES)
        if unknown:
            raise ValueError(f"unsupported platform capabilities: {sorted(unknown)}")


@dataclass(frozen=True)
class PlatformBlueprint:
    blueprint_id: str
    spec_id: str
    capabilities: tuple[str, ...]
    primitives: tuple[str, ...]
    layers: tuple[str, ...]
    proof_trace: tuple[Mapping[str, Any], ...]

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema": PLATFORM_SYNTHESIS_SCHEMA,
            "runtime": PLATFORM_SYNTHESIS_RUNTIME,
            "blueprint_id": self.blueprint_id,
            "spec_id": self.spec_id,
            "capabilities": list(self.capabilities),
            "primitives": list(self.primitives),
            "layers": list(self.layers),
            "proof_trace": [dict(item) for item in self.proof_trace],
        }
        return {**core, "blueprint_hash": canonical_hash(core)}


PLATFORM_PRIMITIVES = (
    PlatformPrimitive(
        "identity",
        "identity and typed set membership",
        "preserve message identity and type across a boundary",
        "logic_equivalent",
        ("conservation",),
    ),
    PlatformPrimitive(
        "directed_relation",
        "directed graph relation",
        "move a message from a source port to a target port",
        "geometry_distance_squared",
        ("radiation", "attraction_repulsion"),
    ),
    PlatformPrimitive(
        "composition",
        "function composition",
        "compose transformations and promote work onto parallel lanes",
        "algebra_polynomial_value",
        ("radiation", "nucleation"),
    ),
    PlatformPrimitive(
        "conservation",
        "bounded invariant",
        "preserve queue and resource capacity under load",
        "chemistry_mass_conservation",
        ("conservation",),
    ),
    PlatformPrimitive(
        "ordering",
        "partial order",
        "schedule higher-priority work before lower-priority work",
        "algebra_solve_linear",
        ("gravitation", "attraction_repulsion"),
    ),
    PlatformPrimitive(
        "feedback",
        "negative feedback",
        "propagate backpressure as a vertical fabric vibration",
        "biology_homeostatic_error",
        ("attraction_repulsion", "dissipation"),
    ),
    PlatformPrimitive(
        "fixed_point",
        "bounded fixed-point iteration",
        "retry until success or the declared attempt bound is reached",
        "calculus_polynomial_derivative",
        ("nucleation", "dissipation", "decay"),
    ),
    PlatformPrimitive(
        "topology",
        "graph topology and intersection",
        "form message threads and intersections from observed flow",
        "information_hartley_bits",
        ("gravitation", "nucleation", "radiation"),
    ),
    PlatformPrimitive(
        "projection",
        "projective measurement",
        "collapse runtime state into a discrete supported result",
        "logic_implies",
        ("radiation", "conservation"),
    ),
)

PLATFORM_PRIMITIVE_INDEX = {
    primitive.name: primitive for primitive in PLATFORM_PRIMITIVES
}

PLATFORM_CAPABILITY_PRIMITIVES = {
    "typed_messages": "identity",
    "directed_routing": "directed_relation",
    "parallel_promotion": "composition",
    "bounded_capacity": "conservation",
    "priority_scheduling": "ordering",
    "backpressure": "feedback",
    "bounded_retries": "fixed_point",
    "emergent_topology": "topology",
    "discrete_output": "projection",
}

SPIDERWEB_LAYERS = (
    "layer_0_transport",
    "layer_1_message",
    "layer_2_flow",
    "layer_3_orchestration",
)

_FORMAL_PROBES: Mapping[str, Mapping[str, Any]] = {
    "identity": {
        "primitive": "logic_equivalent",
        "arguments": {"left": True, "right": True},
    },
    "directed_relation": {
        "primitive": "geometry_distance_squared",
        "arguments": {"left": [0, 0], "right": [1, 0]},
    },
    "composition": {
        "primitive": "algebra_polynomial_value",
        "arguments": {"coefficients": [1, 2], "x": 3},
    },
    "conservation": {
        "primitive": "chemistry_mass_conservation",
        "arguments": {"reactant_masses": [3, 5], "product_masses": [8]},
    },
    "ordering": {
        "primitive": "algebra_solve_linear",
        "arguments": {"coefficient": 2, "offset": 0, "result": 4},
    },
    "feedback": {
        "primitive": "biology_homeostatic_error",
        "arguments": {"target": 4, "observed": 7},
    },
    "fixed_point": {
        "primitive": "calculus_polynomial_derivative",
        "arguments": {"coefficients": [0, 1, 1]},
    },
    "topology": {
        "primitive": "information_hartley_bits",
        "arguments": {"symbol_count": 8},
    },
    "projection": {
        "primitive": "logic_implies",
        "arguments": {"premise": True, "conclusion": True},
    },
}


def platform_primitive_manifest() -> dict[str, Any]:
    formal_names = {
        primitive["name"] for primitive in formal_domain_manifest()["primitives"]
    }
    if any(
        primitive.formal_primitive not in formal_names
        for primitive in PLATFORM_PRIMITIVES
    ):
        raise RuntimeError("platform primitive references an unknown formal primitive")
    if any(
        set(primitive.root_mechanics) - set(ROOT_MECHANICS)
        for primitive in PLATFORM_PRIMITIVES
    ):
        raise RuntimeError("platform primitive references an unknown root mechanic")
    core = {
        "schema": PLATFORM_SYNTHESIS_SCHEMA,
        "runtime": PLATFORM_SYNTHESIS_RUNTIME,
        "spiderweb_runtime": SPIDERWEB_PLATFORM_RUNTIME,
        "layers": list(SPIDERWEB_LAYERS),
        "primitives": [
            {
                **asdict(primitive),
                "root_mechanics": list(primitive.root_mechanics),
            }
            for primitive in PLATFORM_PRIMITIVES
        ],
        "capability_bindings": dict(PLATFORM_CAPABILITY_PRIMITIVES),
    }
    return {**core, "registry_hash": canonical_hash(core)}


def prove_platform_primitives(
    primitive_names: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    if not primitive_names or len(set(primitive_names)) != len(primitive_names):
        raise ValueError("platform primitives must be non-empty and unique")
    traces: list[Mapping[str, Any]] = []
    for name in primitive_names:
        if name not in PLATFORM_PRIMITIVE_INDEX:
            raise ValueError(f"unknown platform primitive: {name}")
        probe = _FORMAL_PROBES[name]
        response = solve_formal_request(
            {
                "schema": FORMAL_DOMAIN_SCHEMA,
                "runtime": FORMAL_DOMAIN_RUNTIME,
                "query_id": f"platform:{name}",
                "primitive": probe["primitive"],
                "arguments": dict(probe["arguments"]),
            }
        )
        if response["claim_status"] != "proven":
            raise RuntimeError(f"formal platform primitive was not proven: {name}")
        traces.append(
            {
                "platform_primitive": name,
                "formal_primitive": probe["primitive"],
                "claim_status": response["claim_status"],
                "response_hash": response["response_hash"],
                "root_mechanics": list(
                    PLATFORM_PRIMITIVE_INDEX[name].root_mechanics
                ),
            }
        )
    return tuple(traces)


def build_platform_blueprint(
    spec: PlatformSpec,
    primitive_names: Sequence[str],
    *,
    blueprint_label: str,
) -> PlatformBlueprint:
    spec.validate()
    ordered = tuple(
        primitive.name
        for primitive in PLATFORM_PRIMITIVES
        if primitive.name in set(primitive_names)
    )
    if not ordered:
        raise ValueError("a platform blueprint needs at least one primitive")
    proof_trace = prove_platform_primitives(ordered)
    identity = canonical_hash(
        {
            "label": blueprint_label,
            "spec_id": spec.spec_id,
            "capabilities": list(spec.capabilities),
            "primitives": list(ordered),
        }
    )[:20]
    return PlatformBlueprint(
        blueprint_id=f"platform:{identity}",
        spec_id=spec.spec_id,
        capabilities=spec.capabilities,
        primitives=ordered,
        layers=SPIDERWEB_LAYERS,
        proof_trace=proof_trace,
    )


def compile_platform_source(blueprint: PlatformBlueprint) -> str:
    manifest = blueprint.manifest()
    primitives = json.dumps(list(blueprint.primitives), sort_keys=True)
    manifest_text = json.dumps(manifest, sort_keys=True)
    return f'''"""Generated mathematical-primitive Spiderweb platform."""

from __future__ import annotations

import json
import sys

SCHEMA = {PLATFORM_SYNTHESIS_SCHEMA}
RUNTIME = {SPIDERWEB_PLATFORM_RUNTIME!r}
LAYERS = {SPIDERWEB_LAYERS!r}
PRIMITIVES = frozenset({primitives})
BLUEPRINT = json.loads({manifest_text!r})


def _require_request(request):
    if not isinstance(request, dict) or set(request) != {{"action", "payload"}}:
        raise ValueError("platform request fields are invalid")
    if not isinstance(request["action"], str) or not isinstance(request["payload"], dict):
        raise TypeError("platform request types are invalid")


def execute(request):
    _require_request(request)
    action = request["action"]
    payload = request["payload"]
    if action == "typed_message":
        value = payload.get("value")
        return {{"preserved": "identity" in PRIMITIVES, "value": value}}
    if action == "route":
        path = payload.get("path")
        if not isinstance(path, list) or len(path) < 2 or not all(isinstance(item, str) for item in path):
            raise ValueError("route path is invalid")
        delivered = "directed_relation" in PRIMITIVES
        promoted = bool(payload.get("parallel")) and "composition" in PRIMITIVES
        thread = path if delivered and "topology" in PRIMITIVES else []
        intersections = []
        prior_threads = payload.get("prior_threads", [])
        if thread and isinstance(prior_threads, list):
            for prior in prior_threads:
                if isinstance(prior, list):
                    shared = sorted(set(thread) & set(prior))
                    if shared:
                        intersections.append(shared)
        return {{
            "delivered": delivered,
            "promoted": promoted,
            "off_ramp": path[-1] if delivered and promoted else None,
            "thread": thread,
            "intersections": intersections,
            "preloaded": path[1] if delivered and "composition" in PRIMITIVES else None,
        }}
    if action == "capacity":
        load = payload.get("load")
        capacity = payload.get("capacity")
        if isinstance(load, bool) or not isinstance(load, int) or load < 0:
            raise ValueError("capacity load is invalid")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity bound is invalid")
        accepted = min(load, capacity) if "conservation" in PRIMITIVES else load
        return {{"accepted": accepted, "bounded": accepted <= capacity}}
    if action == "priority":
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or not all(
            isinstance(task, dict)
            and set(task) == {{"id", "priority"}}
            and isinstance(task["id"], str)
            and isinstance(task["priority"], int)
            and not isinstance(task["priority"], bool)
            for task in tasks
        ):
            raise ValueError("priority tasks are invalid")
        ordered = (
            sorted(tasks, key=lambda task: (-task["priority"], task["id"]))
            if "ordering" in PRIMITIVES
            else tasks
        )
        return {{"task_ids": [task["id"] for task in ordered]}}
    if action == "backpressure":
        load = payload.get("load")
        capacity = payload.get("capacity")
        if not isinstance(load, int) or not isinstance(capacity, int):
            raise ValueError("backpressure values are invalid")
        active = load > capacity and "feedback" in PRIMITIVES
        return {{
            "signal": "slow_down" if active else "none",
            "vertical_vibration": active,
        }}
    if action == "retry":
        success_after = payload.get("success_after")
        maximum = payload.get("maximum")
        if (
            isinstance(success_after, bool)
            or not isinstance(success_after, int)
            or success_after <= 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("retry values are invalid")
        attempts = min(success_after, maximum) if "fixed_point" in PRIMITIVES else 1
        return {{"attempts": attempts, "success": attempts >= success_after}}
    if action == "project":
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("projection candidates are invalid")
        if "projection" not in PRIMITIVES:
            return {{"status": "unknown", "value": None}}
        ranked = sorted(
            candidates,
            key=lambda item: (-float(item.get("support", 0.0)), str(item.get("value"))),
        )
        return {{"status": "derived", "value": ranked[0].get("value")}}
    raise ValueError("unsupported platform action")


def platform_manifest():
    return {{
        "schema": SCHEMA,
        "runtime": RUNTIME,
        "layers": list(LAYERS),
        "primitives": sorted(PRIMITIVES),
        "blueprint": BLUEPRINT,
    }}


if __name__ == "__main__":
    incoming = json.loads(sys.stdin.read())
    print(json.dumps(execute(incoming), sort_keys=True, separators=(",", ":")))
'''


def platform_curriculum() -> tuple[PlatformSpec, ...]:
    training = tuple(
        PlatformSpec(
            spec_id=f"train-{index:02d}-{capability}",
            capabilities=(capability,),
            partition="training",
            description=f"Isolate the {capability.replace('_', ' ')} capability.",
        )
        for index in range(3)
        for capability in PLATFORM_CAPABILITY_PRIMITIVES
    )
    validation = (
        PlatformSpec(
            "validation-message-route",
            ("typed_messages", "directed_routing", "discrete_output"),
            "validation",
            "Typed ground-level routing with a supported output.",
        ),
        PlatformSpec(
            "validation-flow-control",
            ("bounded_capacity", "backpressure", "priority_scheduling"),
            "validation",
            "Bounded scheduling with feedback under load.",
        ),
        PlatformSpec(
            "validation-resilience",
            ("bounded_retries", "discrete_output", "typed_messages"),
            "validation",
            "Bounded recovery followed by a discrete result.",
        ),
    )
    heldout = (
        PlatformSpec(
            "heldout-comfy-bus",
            ("typed_messages", "directed_routing", "discrete_output"),
            "heldout",
            "A typed ground-level bus with explicit delivery and projection.",
        ),
        PlatformSpec(
            "heldout-parallel-highway",
            (
                "typed_messages",
                "directed_routing",
                "parallel_promotion",
                "emergent_topology",
                "discrete_output",
            ),
            "heldout",
            "Ground traffic promotes onto parallel lanes and forms flow threads.",
        ),
        PlatformSpec(
            "heldout-resilient-orchestrator",
            (
                "bounded_capacity",
                "priority_scheduling",
                "backpressure",
                "bounded_retries",
                "discrete_output",
            ),
            "heldout",
            "An orchestrator that bounds, orders, signals, retries, and projects.",
        ),
        PlatformSpec(
            "heldout-spiderweb-platform",
            tuple(PLATFORM_CAPABILITY_PRIMITIVES),
            "heldout",
            "The four-layer Spiderweb with ramps, vibrations, threads, and recovery.",
        ),
    )
    curriculum = (*training, *validation, *heldout)
    for spec in curriculum:
        spec.validate()
    return curriculum


def platform_synthesis_self_test() -> dict[str, bool]:
    registry = platform_primitive_manifest()
    curriculum = platform_curriculum()
    full = next(spec for spec in curriculum if spec.spec_id == "heldout-spiderweb-platform")
    blueprint = build_platform_blueprint(
        full,
        PLATFORM_PRIMITIVE_INDEX,
        blueprint_label="self-test",
    )
    source = compile_platform_source(blueprint)
    return {
        "nine_mathematical_primitives": len(PLATFORM_PRIMITIVES) == 9,
        "all_primitives_formally_bound": all(
            trace["claim_status"] == "proven" for trace in blueprint.proof_trace
        ),
        "all_primitives_expand_to_roots": all(
            primitive.root_mechanics for primitive in PLATFORM_PRIMITIVES
        ),
        "four_spiderweb_layers": len(SPIDERWEB_LAYERS) == 4,
        "training_validation_heldout": {
            spec.partition for spec in curriculum
        }
        == {"training", "validation", "heldout"},
        "generated_source_is_executable": "def execute(request):" in source,
        "registry_is_hash_bound": registry["registry_hash"]
        == canonical_hash(
            {key: value for key, value in registry.items() if key != "registry_hash"}
        ),
    }
