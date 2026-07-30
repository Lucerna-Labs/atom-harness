"""Open-ended mathematical primitive composition above Atom's seven roots.

The seven root operators are immutable generative substrate.  Every other
primitive is a typed, recursively executable composition that remains
expandable to those roots.  The forge is deliberately domain-neutral: it
records mathematical simulation evidence and does not treat software concepts
as its learning ontology.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_causal_world_schema import ROOT_MECHANICS, canonical_hash


PRIMITIVE_FORGE_SCHEMA = 1
PRIMITIVE_FORGE_RUNTIME = "atom-open-primitive-forge-v1"
ROOT_STATUS = "immutable_root"
QUARANTINED_STATUS = "quarantined"
CANDIDATE_STATUS = "candidate"
CRYSTALLIZED_STATUS = "crystallized"
REVISED_STATUS = "revised"
RETIRED_STATUS = "retired"

VALID_STATUSES = frozenset(
    {
        ROOT_STATUS,
        QUARANTINED_STATUS,
        CANDIDATE_STATUS,
        CRYSTALLIZED_STATUS,
        REVISED_STATUS,
        RETIRED_STATUS,
    }
)
VALID_RECIPE_MODES = frozenset({"serial", "parallel", "feedback"})
ASSOCIATIVE_MODES = frozenset({"serial", "parallel"})
COMMUTATIVE_MODES = frozenset({"parallel"})


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class Dimension:
    """Canonical dimensional exponents, such as state^1 or length^1*time^-1."""

    powers: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        labels: set[str] = set()
        previous = ""
        for label, exponent in self.powers:
            _require_text(label, "dimension label")
            if label in labels or label < previous:
                raise ValueError("dimension powers must be unique and sorted")
            if isinstance(exponent, bool) or not isinstance(exponent, int):
                raise ValueError("dimension exponents must be integers")
            if exponent == 0:
                raise ValueError("zero dimension exponents must be omitted")
            labels.add(label)
            previous = label

    def to_payload(self) -> list[list[Any]]:
        return [[label, exponent] for label, exponent in self.powers]

    @classmethod
    def from_payload(cls, payload: Any) -> Dimension:
        if not isinstance(payload, list):
            raise ValueError("dimension payload must be a list")
        powers: list[tuple[str, int]] = []
        for item in payload:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("dimension entries must be [label, exponent]")
            label = _require_text(item[0], "dimension label")
            exponent = item[1]
            if isinstance(exponent, bool) or not isinstance(exponent, int):
                raise ValueError("dimension exponents must be integers")
            powers.append((label, exponent))
        return cls(tuple(powers))


@dataclass(frozen=True)
class PortSignature:
    """A typed mathematical value crossing a primitive boundary."""

    kind: str
    dimension: Dimension

    def __post_init__(self) -> None:
        _require_text(self.kind, "port kind")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dimension": self.dimension.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> PortSignature:
        if not isinstance(payload, Mapping):
            raise ValueError("port signature must be an object")
        if set(payload) != {"kind", "dimension"}:
            raise ValueError("port signature keys are invalid")
        return cls(
            kind=_require_text(payload["kind"], "port kind"),
            dimension=Dimension.from_payload(payload["dimension"]),
        )


@dataclass(frozen=True)
class TypeSignature:
    """Typed/domain signature retained by every root and composition."""

    domain: str
    inputs: tuple[PortSignature, ...]
    output: PortSignature

    def __post_init__(self) -> None:
        _require_text(self.domain, "signature domain")
        if not self.inputs:
            raise ValueError("a primitive requires at least one typed input")

    def to_payload(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "inputs": [item.to_payload() for item in self.inputs],
            "output": self.output.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> TypeSignature:
        if not isinstance(payload, Mapping):
            raise ValueError("type signature must be an object")
        if set(payload) != {"domain", "inputs", "output"}:
            raise ValueError("type signature keys are invalid")
        inputs = payload["inputs"]
        if not isinstance(inputs, list):
            raise ValueError("signature inputs must be a list")
        return cls(
            domain=_require_text(payload["domain"], "signature domain"),
            inputs=tuple(PortSignature.from_payload(item) for item in inputs),
            output=PortSignature.from_payload(payload["output"]),
        )


@dataclass(frozen=True)
class CompositionRecipe:
    """A direct recipe; canonical identity is computed from its full meaning."""

    mode: str
    components: tuple[str, ...]
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in VALID_RECIPE_MODES:
            raise ValueError(f"unsupported composition mode: {self.mode}")
        if len(self.components) < 2:
            raise ValueError("a composition requires at least two components")
        if self.mode == "feedback" and len(self.components) != 2:
            raise ValueError("feedback composition requires exactly two components")
        for component in self.components:
            _require_text(component, "component reference")
        previous = ""
        seen: set[str] = set()
        for name, value in self.parameters:
            _require_text(name, "parameter name")
            _require_finite(value, f"parameter {name}")
            if name in seen or name < previous:
                raise ValueError("recipe parameters must be unique and sorted")
            seen.add(name)
            previous = name

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "components": list(self.components),
            "parameters": [[name, value] for name, value in self.parameters],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> CompositionRecipe:
        if not isinstance(payload, Mapping):
            raise ValueError("composition recipe must be an object")
        if set(payload) != {"mode", "components", "parameters"}:
            raise ValueError("composition recipe keys are invalid")
        components = payload["components"]
        parameters = payload["parameters"]
        if not isinstance(components, list) or not isinstance(parameters, list):
            raise ValueError("recipe components and parameters must be lists")
        parsed_parameters: list[tuple[str, float]] = []
        for item in parameters:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("recipe parameters must be [name, value]")
            parsed_parameters.append(
                (
                    _require_text(item[0], "parameter name"),
                    _require_finite(item[1], "parameter value"),
                )
            )
        return cls(
            mode=_require_text(payload["mode"], "recipe mode"),
            components=tuple(
                _require_text(item, "component reference") for item in components
            ),
            parameters=tuple(parsed_parameters),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """A provenance-bound comparison between prediction and observation."""

    evidence_id: str
    context_id: str
    predicted: float
    observed: float
    tolerance: float
    residual: float
    success: bool
    source: str
    provenance: tuple[str, ...]
    sequence: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "context_id": self.context_id,
            "predicted": self.predicted,
            "observed": self.observed,
            "tolerance": self.tolerance,
            "residual": self.residual,
            "success": self.success,
            "source": self.source,
            "provenance": list(self.provenance),
            "sequence": self.sequence,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> EvidenceRecord:
        required = {
            "evidence_id",
            "context_id",
            "predicted",
            "observed",
            "tolerance",
            "residual",
            "success",
            "source",
            "provenance",
            "sequence",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("evidence payload keys are invalid")
        if not isinstance(payload["success"], bool):
            raise ValueError("evidence success must be boolean")
        provenance = payload["provenance"]
        if not isinstance(provenance, list):
            raise ValueError("evidence provenance must be a list")
        sequence = payload["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("evidence sequence must be a positive integer")
        tolerance = _require_finite(payload["tolerance"], "evidence tolerance")
        residual = _require_finite(payload["residual"], "evidence residual")
        if tolerance < 0.0 or residual < 0.0:
            raise ValueError("evidence tolerance and residual cannot be negative")
        return cls(
            evidence_id=_require_text(payload["evidence_id"], "evidence id"),
            context_id=_require_text(payload["context_id"], "context id"),
            predicted=_require_finite(payload["predicted"], "prediction"),
            observed=_require_finite(payload["observed"], "observation"),
            tolerance=tolerance,
            residual=residual,
            success=payload["success"],
            source=_require_text(payload["source"], "evidence source"),
            provenance=_ordered_unique(
                _require_text(item, "evidence provenance") for item in provenance
            ),
            sequence=sequence,
        )


@dataclass(frozen=True)
class PrimitiveRecord:
    """A root or learned primitive with its complete derivation and history."""

    primitive_id: str
    aliases: tuple[str, ...]
    root: bool
    signature: TypeSignature
    recipe: CompositionRecipe | None
    equivalent_recipes: tuple[CompositionRecipe, ...]
    invariants: tuple[str, ...]
    symmetries: tuple[str, ...]
    boundaries: tuple[str, ...]
    scales: tuple[str, ...]
    evidence: tuple[EvidenceRecord, ...]
    counterexamples: tuple[EvidenceRecord, ...]
    confidence: float
    persistence: float
    status: str
    provenance: tuple[str, ...]
    created_sequence: int
    updated_sequence: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "aliases": list(self.aliases),
            "root": self.root,
            "signature": self.signature.to_payload(),
            "recipe": None if self.recipe is None else self.recipe.to_payload(),
            "equivalent_recipes": [
                recipe.to_payload() for recipe in self.equivalent_recipes
            ],
            "invariants": list(self.invariants),
            "symmetries": list(self.symmetries),
            "boundaries": list(self.boundaries),
            "scales": list(self.scales),
            "evidence": [item.to_payload() for item in self.evidence],
            "counterexamples": [
                item.to_payload() for item in self.counterexamples
            ],
            "confidence": self.confidence,
            "persistence": self.persistence,
            "status": self.status,
            "provenance": list(self.provenance),
            "created_sequence": self.created_sequence,
            "updated_sequence": self.updated_sequence,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> PrimitiveRecord:
        required = {
            "primitive_id",
            "aliases",
            "root",
            "signature",
            "recipe",
            "equivalent_recipes",
            "invariants",
            "symmetries",
            "boundaries",
            "scales",
            "evidence",
            "counterexamples",
            "confidence",
            "persistence",
            "status",
            "provenance",
            "created_sequence",
            "updated_sequence",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("primitive record keys are invalid")
        if not isinstance(payload["root"], bool):
            raise ValueError("primitive root flag must be boolean")
        status = _require_text(payload["status"], "primitive status")
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown primitive status: {status}")
        list_fields = (
            "aliases",
            "equivalent_recipes",
            "invariants",
            "symmetries",
            "boundaries",
            "scales",
            "evidence",
            "counterexamples",
            "provenance",
        )
        if any(not isinstance(payload[field], list) for field in list_fields):
            raise ValueError("primitive collection fields must be lists")
        created = payload["created_sequence"]
        updated = payload["updated_sequence"]
        if (
            isinstance(created, bool)
            or not isinstance(created, int)
            or isinstance(updated, bool)
            or not isinstance(updated, int)
            or created < 0
            or updated < created
        ):
            raise ValueError("primitive sequence bounds are invalid")
        recipe_payload = payload["recipe"]
        recipe = (
            None
            if recipe_payload is None
            else CompositionRecipe.from_payload(recipe_payload)
        )
        return cls(
            primitive_id=_require_text(payload["primitive_id"], "primitive id"),
            aliases=_ordered_unique(
                _require_text(item, "primitive alias")
                for item in payload["aliases"]
            ),
            root=payload["root"],
            signature=TypeSignature.from_payload(payload["signature"]),
            recipe=recipe,
            equivalent_recipes=tuple(
                CompositionRecipe.from_payload(item)
                for item in payload["equivalent_recipes"]
            ),
            invariants=_ordered_unique(
                _require_text(item, "invariant")
                for item in payload["invariants"]
            ),
            symmetries=_ordered_unique(
                _require_text(item, "symmetry")
                for item in payload["symmetries"]
            ),
            boundaries=_ordered_unique(
                _require_text(item, "boundary")
                for item in payload["boundaries"]
            ),
            scales=_ordered_unique(
                _require_text(item, "scale") for item in payload["scales"]
            ),
            evidence=tuple(
                EvidenceRecord.from_payload(item) for item in payload["evidence"]
            ),
            counterexamples=tuple(
                EvidenceRecord.from_payload(item)
                for item in payload["counterexamples"]
            ),
            confidence=_require_finite(
                payload["confidence"], "primitive confidence"
            ),
            persistence=_require_finite(
                payload["persistence"], "primitive persistence"
            ),
            status=status,
            provenance=_ordered_unique(
                _require_text(item, "primitive provenance")
                for item in payload["provenance"]
            ),
            created_sequence=created,
            updated_sequence=updated,
        )


@dataclass(frozen=True)
class ForgeConfig:
    promotion_evidence: int = 3
    promotion_confidence: float = 0.75
    retirement_persistence: float = 0.15

    def __post_init__(self) -> None:
        if (
            isinstance(self.promotion_evidence, bool)
            or not isinstance(self.promotion_evidence, int)
            or self.promotion_evidence < 2
        ):
            raise ValueError("promotion evidence must be an integer of at least 2")
        if not 0.5 <= self.promotion_confidence <= 1.0:
            raise ValueError("promotion confidence must be within [0.5, 1]")
        if not 0.0 <= self.retirement_persistence < 1.0:
            raise ValueError("retirement persistence must be within [0, 1)")

    def to_payload(self) -> dict[str, Any]:
        return {
            "promotion_evidence": self.promotion_evidence,
            "promotion_confidence": self.promotion_confidence,
            "retirement_persistence": self.retirement_persistence,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ForgeConfig:
        required = {
            "promotion_evidence",
            "promotion_confidence",
            "retirement_persistence",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("forge configuration keys are invalid")
        return cls(
            promotion_evidence=payload["promotion_evidence"],
            promotion_confidence=_require_finite(
                payload["promotion_confidence"], "promotion confidence"
            ),
            retirement_persistence=_require_finite(
                payload["retirement_persistence"], "retirement persistence"
            ),
        )


_FIELD_SIGNATURE = TypeSignature(
    domain="mathematical_scalar_field",
    inputs=(
        PortSignature(
            kind="bounded_scalar_field",
            dimension=Dimension((("state", 1),)),
        ),
    ),
    output=PortSignature(
        kind="bounded_scalar_field",
        dimension=Dimension((("state", 1),)),
    ),
)

_ROOT_INVARIANTS = {
    "radiation": ("finite propagation", "typed field continuity"),
    "dissipation": ("non-amplifying attenuation", "typed field continuity"),
    "gravitation": ("bounded attraction", "typed field continuity"),
    "attraction_repulsion": (
        "signed relational response",
        "typed field continuity",
    ),
    "nucleation": ("threshold-bounded formation", "typed field continuity"),
    "conservation": ("bounded state budget", "typed field continuity"),
    "decay": ("monotonic unsupported weakening", "typed field continuity"),
}


class PrimitiveForge:
    """Dynamic, evidence-gated graph above an immutable seven-root substrate."""

    def __init__(self, config: ForgeConfig | None = None) -> None:
        self.config = config or ForgeConfig()
        self._sequence = 0
        self._records: dict[str, PrimitiveRecord] = {}
        for root_name in ROOT_MECHANICS:
            self._records[root_name] = PrimitiveRecord(
                primitive_id=root_name,
                aliases=(root_name.replace("_", " "),),
                root=True,
                signature=_FIELD_SIGNATURE,
                recipe=None,
                equivalent_recipes=(),
                invariants=_ROOT_INVARIANTS[root_name],
                symmetries=("representation invariance",),
                boundaries=("finite scalar simulation state",),
                scales=("dimensionless mathematical simulation",),
                evidence=(),
                counterexamples=(),
                confidence=1.0,
                persistence=1.0,
                status=ROOT_STATUS,
                provenance=("operator:immutable-seven-root-substrate",),
                created_sequence=0,
                updated_sequence=0,
            )

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def root_ids(self) -> tuple[str, ...]:
        return tuple(ROOT_MECHANICS)

    @property
    def records(self) -> tuple[PrimitiveRecord, ...]:
        return tuple(self._records[name] for name in sorted(self._records))

    @property
    def derived_records(self) -> tuple[PrimitiveRecord, ...]:
        return tuple(record for record in self.records if not record.root)

    def get(self, primitive_id: str) -> PrimitiveRecord:
        try:
            return self._records[primitive_id]
        except KeyError as error:
            raise ValueError(f"unknown primitive reference: {primitive_id}") from error

    def dependents(self, primitive_id: str) -> tuple[str, ...]:
        self.get(primitive_id)
        return tuple(
            sorted(
                record.primitive_id
                for record in self.derived_records
                if record.recipe is not None
                and primitive_id in record.recipe.components
            )
        )

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _derive_signature(
        self, mode: str, components: Sequence[str]
    ) -> TypeSignature:
        signatures = [self.get(item).signature for item in components]
        domains = {signature.domain for signature in signatures}
        if len(domains) != 1:
            raise ValueError("composition domains are incompatible")
        if mode == "serial":
            for left, right in zip(signatures, signatures[1:], strict=False):
                if len(right.inputs) != 1 or left.output != right.inputs[0]:
                    raise ValueError(
                        "serial composition has incompatible type or dimension"
                    )
            return TypeSignature(
                domain=signatures[0].domain,
                inputs=signatures[0].inputs,
                output=signatures[-1].output,
            )
        if mode == "parallel":
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError(
                    "parallel composition requires identical type signatures"
                )
            return signatures[0]
        if mode == "feedback":
            plant, controller = signatures
            if (
                len(plant.inputs) != 1
                or len(controller.inputs) != 1
                or plant.inputs[0] != plant.output
                or controller.inputs[0] != controller.output
                or plant.output != controller.inputs[0]
            ):
                raise ValueError(
                    "feedback composition requires a closed typed dimension"
                )
            return plant
        raise ValueError(f"unsupported composition mode: {mode}")

    def _expression(
        self,
        primitive_id: str,
        ancestry: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if primitive_id in ancestry:
            path = " -> ".join((*ancestry, primitive_id))
            raise ValueError(f"cyclic primitive composition: {path}")
        record = self.get(primitive_id)
        if record.root:
            return {"root": primitive_id}
        if record.recipe is None:
            raise ValueError(f"derived primitive has no recipe: {primitive_id}")
        return self._recipe_expression(
            record.recipe,
            ancestry=(*ancestry, primitive_id),
        )

    def _recipe_expression(
        self,
        recipe: CompositionRecipe,
        *,
        ancestry: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        children = [
            self._expression(component, ancestry) for component in recipe.components
        ]
        normalized: list[dict[str, Any]] = []
        for child in children:
            if (
                recipe.mode in ASSOCIATIVE_MODES
                and child.get("mode") == recipe.mode
                and child.get("parameters")
                == [list(item) for item in recipe.parameters]
            ):
                normalized.extend(child["children"])
            else:
                normalized.append(child)
        if recipe.mode in COMMUTATIVE_MODES:
            normalized.sort(key=_canonical_json)
        return {
            "mode": recipe.mode,
            "parameters": [list(item) for item in recipe.parameters],
            "children": normalized,
        }

    @staticmethod
    def _identity(
        expression: Mapping[str, Any],
        signature: TypeSignature,
    ) -> str:
        digest = canonical_hash(
            {
                "expression": expression,
                "signature": signature.to_payload(),
            }
        )
        return f"primitive-{digest[:24]}"

    def compose(
        self,
        mode: str,
        components: Sequence[str],
        *,
        parameters: Mapping[str, float] | None = None,
        expected_signature: TypeSignature | None = None,
        aliases: Sequence[str] = (),
        invariants: Sequence[str] = (),
        symmetries: Sequence[str] = (),
        boundaries: Sequence[str] = (),
        scales: Sequence[str] = (),
        provenance: Sequence[str] = (),
    ) -> PrimitiveRecord:
        """Propose or merge a recursively compositional primitive.

        New proposals always enter quarantine.  Reordered parallel recipes and
        associative rewrites share a canonical identity and merge metadata
        without losing either recipe or provenance.
        """

        parameter_items = tuple(
            sorted(
                (
                    _require_text(name, "parameter name"),
                    _require_finite(value, f"parameter {name}"),
                )
                for name, value in (parameters or {}).items()
            )
        )
        recipe = CompositionRecipe(
            mode=_require_text(mode, "composition mode"),
            components=tuple(
                _require_text(item, "component reference") for item in components
            ),
            parameters=parameter_items,
        )
        for component in recipe.components:
            self.get(component)
        signature = self._derive_signature(recipe.mode, recipe.components)
        if expected_signature is not None and signature != expected_signature:
            raise ValueError(
                "declared type/dimension signature does not match composition"
            )
        expression = self._recipe_expression(recipe)
        primitive_id = self._identity(expression, signature)
        component_records = [self.get(item) for item in recipe.components]
        inherited_invariants = (
            item
            for record in component_records
            for item in record.invariants
        )
        inherited_symmetries = (
            item
            for record in component_records
            for item in record.symmetries
        )
        inherited_boundaries = (
            item
            for record in component_records
            for item in record.boundaries
        )
        inherited_scales = (
            item for record in component_records for item in record.scales
        )
        merged_aliases = _ordered_unique(
            (*aliases, f"{recipe.mode} composition {primitive_id[-8:]}")
        )
        merged_invariants = _ordered_unique((*inherited_invariants, *invariants))
        merged_symmetries = _ordered_unique((*inherited_symmetries, *symmetries))
        merged_boundaries = _ordered_unique((*inherited_boundaries, *boundaries))
        merged_scales = _ordered_unique((*inherited_scales, *scales))
        merged_provenance = _ordered_unique(
            (*provenance, "runtime:primitive-forge-composition")
        )
        existing = self._records.get(primitive_id)
        if existing is not None:
            if existing.root:
                raise ValueError("a derived identity cannot replace a root")
            all_recipes = {
                _canonical_json(item.to_payload()): item
                for item in (
                    existing.recipe,
                    *existing.equivalent_recipes,
                    recipe,
                )
                if item is not None
            }
            primary_key = _canonical_json(existing.recipe.to_payload())
            alternatives = tuple(
                all_recipes[key]
                for key in sorted(all_recipes)
                if key != primary_key
            )
            sequence = self._next_sequence()
            merged = replace(
                existing,
                aliases=_ordered_unique((*existing.aliases, *merged_aliases)),
                equivalent_recipes=alternatives,
                invariants=_ordered_unique(
                    (*existing.invariants, *merged_invariants)
                ),
                symmetries=_ordered_unique(
                    (*existing.symmetries, *merged_symmetries)
                ),
                boundaries=_ordered_unique(
                    (*existing.boundaries, *merged_boundaries)
                ),
                scales=_ordered_unique((*existing.scales, *merged_scales)),
                provenance=_ordered_unique(
                    (*existing.provenance, *merged_provenance)
                ),
                updated_sequence=sequence,
            )
            self._records[primitive_id] = merged
            return merged
        sequence = self._next_sequence()
        created = PrimitiveRecord(
            primitive_id=primitive_id,
            aliases=merged_aliases,
            root=False,
            signature=signature,
            recipe=recipe,
            equivalent_recipes=(),
            invariants=merged_invariants,
            symmetries=merged_symmetries,
            boundaries=merged_boundaries,
            scales=merged_scales,
            evidence=(),
            counterexamples=(),
            confidence=0.5,
            persistence=0.0,
            status=QUARANTINED_STATUS,
            provenance=merged_provenance,
            created_sequence=sequence,
            updated_sequence=sequence,
        )
        self._records[primitive_id] = created
        return created

    def expand_to_roots(self, primitive_id: str) -> tuple[str, ...]:
        """Return the complete ordered/multiplicity-preserving root expansion."""

        def visit(item: str, ancestry: tuple[str, ...]) -> tuple[str, ...]:
            if item in ancestry:
                path = " -> ".join((*ancestry, item))
                raise ValueError(f"cyclic primitive composition: {path}")
            record = self.get(item)
            if record.root:
                return (item,)
            if record.recipe is None:
                raise ValueError(f"derived primitive has no recipe: {item}")
            leaves: list[str] = []
            components = record.recipe.components
            if record.recipe.mode in COMMUTATIVE_MODES:
                components = tuple(sorted(components))
            for component in components:
                leaves.extend(visit(component, (*ancestry, item)))
            return tuple(leaves)

        roots = visit(primitive_id, ())
        if not roots or set(roots) - set(ROOT_MECHANICS):
            raise ValueError(f"invalid root expansion for {primitive_id}")
        return roots

    def observe(
        self,
        primitive_id: str,
        *,
        context_id: str,
        predicted: float,
        observed: float,
        tolerance: float = 1e-9,
        source: str = "runtime-use",
        provenance: Sequence[str] = (),
    ) -> PrimitiveRecord:
        """Admit evidence through quarantine and revise contradictory laws."""

        record = self.get(primitive_id)
        if record.root:
            raise ValueError("immutable root operators cannot learn observations")
        context = _require_text(context_id, "observation context")
        prediction = _require_finite(predicted, "prediction")
        observation = _require_finite(observed, "observation")
        accepted_tolerance = _require_finite(tolerance, "tolerance")
        if accepted_tolerance < 0.0:
            raise ValueError("tolerance cannot be negative")
        evidence_source = _require_text(source, "evidence source")
        evidence_provenance = _ordered_unique(
            (*provenance, f"source:{evidence_source}")
        )
        residual = abs(prediction - observation)
        success = residual <= accepted_tolerance
        evidence_core = {
            "primitive_id": primitive_id,
            "context_id": context,
            "predicted": prediction,
            "observed": observation,
            "tolerance": accepted_tolerance,
            "source": evidence_source,
            "provenance": list(evidence_provenance),
        }
        evidence_id = canonical_hash(evidence_core)
        existing_evidence = (*record.evidence, *record.counterexamples)
        if any(item.evidence_id == evidence_id for item in existing_evidence):
            return record
        sequence = self._next_sequence()
        evidence = EvidenceRecord(
            evidence_id=evidence_id,
            context_id=context,
            predicted=prediction,
            observed=observation,
            tolerance=accepted_tolerance,
            residual=residual,
            success=success,
            source=evidence_source,
            provenance=evidence_provenance,
            sequence=sequence,
        )
        supports = (
            (*record.evidence, evidence) if success else record.evidence
        )
        counterexamples = (
            record.counterexamples
            if success
            else (*record.counterexamples, evidence)
        )
        support_count = len(supports)
        contradiction_count = len(counterexamples)
        confidence = (support_count + 1.0) / (
            support_count + contradiction_count + 2.0
        )
        net_support = support_count - contradiction_count
        persistence = min(
            1.0,
            max(0.0, net_support / self.config.promotion_evidence),
        )
        distinct_contexts = {item.context_id for item in supports}
        can_crystallize = (
            net_support >= self.config.promotion_evidence
            and len(distinct_contexts) >= self.config.promotion_evidence
            and confidence >= self.config.promotion_confidence
        )
        if record.status == RETIRED_STATUS:
            status = RETIRED_STATUS
        elif can_crystallize:
            status = CRYSTALLIZED_STATUS
        elif record.status == CRYSTALLIZED_STATUS and not success:
            status = REVISED_STATUS
        elif contradiction_count > support_count:
            status = RETIRED_STATUS
        elif support_count:
            status = (
                REVISED_STATUS
                if record.status == REVISED_STATUS
                else CANDIDATE_STATUS
            )
        else:
            status = QUARANTINED_STATUS
        revised = replace(
            record,
            evidence=supports,
            counterexamples=counterexamples,
            confidence=confidence,
            persistence=persistence,
            status=status,
            provenance=_ordered_unique(
                (*record.provenance, *evidence_provenance)
            ),
            updated_sequence=sequence,
        )
        self._records[primitive_id] = revised
        return revised

    def apply_decay(
        self,
        primitive_id: str,
        *,
        amount: float,
        provenance: str = "runtime:unsupported-decay",
    ) -> PrimitiveRecord:
        """Weaken an unsupported derived primitive and retire it at the floor."""

        record = self.get(primitive_id)
        if record.root:
            raise ValueError("immutable root operators cannot decay or retire")
        decay_amount = _require_finite(amount, "decay amount")
        if not 0.0 < decay_amount <= 1.0:
            raise ValueError("decay amount must be within (0, 1]")
        sequence = self._next_sequence()
        persistence = max(0.0, record.persistence - decay_amount)
        confidence = max(0.0, record.confidence - decay_amount * 0.25)
        status = (
            RETIRED_STATUS
            if persistence < self.config.retirement_persistence
            else record.status
        )
        decayed = replace(
            record,
            persistence=persistence,
            confidence=confidence,
            status=status,
            provenance=_ordered_unique(
                (*record.provenance, _require_text(provenance, "decay provenance"))
            ),
            updated_sequence=sequence,
        )
        self._records[primitive_id] = decayed
        return decayed

    def _validate_graph(self) -> None:
        if tuple(ROOT_MECHANICS) != self.root_ids:
            raise ValueError("root substrate does not match the immutable seven")
        root_template = PrimitiveForge(self.config)
        for root_name in ROOT_MECHANICS:
            if (
                self.get(root_name).to_payload()
                != root_template.get(root_name).to_payload()
            ):
                raise ValueError(f"immutable root record was modified: {root_name}")
        for record in self.records:
            if not 0.0 <= record.confidence <= 1.0:
                raise ValueError("primitive confidence must be within [0, 1]")
            if not 0.0 <= record.persistence <= 1.0:
                raise ValueError("primitive persistence must be within [0, 1]")
            if record.root:
                if record.recipe is not None or record.status != ROOT_STATUS:
                    raise ValueError("root recipe or status is mutable")
                continue
            if record.recipe is None or record.status == ROOT_STATUS:
                raise ValueError("derived primitive requires a recipe and status")
            for component in record.recipe.components:
                if component not in self._records:
                    raise ValueError(
                        f"unknown primitive reference: {record.primitive_id} -> "
                        f"{component}"
                    )
            evidence_ids = [
                item.evidence_id
                for item in (*record.evidence, *record.counterexamples)
            ]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError("duplicate evidence ids are not permitted")
            for item in (*record.evidence, *record.counterexamples):
                evidence_core = {
                    "primitive_id": record.primitive_id,
                    "context_id": item.context_id,
                    "predicted": item.predicted,
                    "observed": item.observed,
                    "tolerance": item.tolerance,
                    "source": item.source,
                    "provenance": list(item.provenance),
                }
                if canonical_hash(evidence_core) != item.evidence_id:
                    raise ValueError("primitive evidence identity is invalid")
                expected_residual = abs(item.predicted - item.observed)
                if not math.isclose(
                    item.residual,
                    expected_residual,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError("primitive evidence residual is invalid")
                if item.success != (item.residual <= item.tolerance):
                    raise ValueError("primitive evidence outcome is invalid")
        states: dict[str, int] = {}

        def visit(item: str) -> None:
            state = states.get(item, 0)
            if state == 1:
                raise ValueError(f"cyclic primitive composition includes {item}")
            if state == 2:
                return
            states[item] = 1
            record = self.get(item)
            if record.recipe is not None:
                for component in record.recipe.components:
                    visit(component)
            states[item] = 2

        for primitive_id in sorted(self._records):
            visit(primitive_id)
        for record in self.derived_records:
            signature = self._derive_signature(
                record.recipe.mode,
                record.recipe.components,
            )
            if signature != record.signature:
                raise ValueError(
                    f"stored signature is incompatible: {record.primitive_id}"
                )
            expression = self._expression(record.primitive_id)
            expected_id = self._identity(expression, record.signature)
            if expected_id != record.primitive_id:
                raise ValueError(
                    f"primitive canonical identity is invalid: "
                    f"{record.primitive_id}"
                )
            for equivalent in record.equivalent_recipes:
                for component in equivalent.components:
                    if component not in self._records:
                        raise ValueError(
                            "equivalent recipe has an unknown primitive "
                            f"reference: {record.primitive_id} -> {component}"
                        )
                equivalent_signature = self._derive_signature(
                    equivalent.mode,
                    equivalent.components,
                )
                equivalent_expression = self._recipe_expression(
                    equivalent,
                    ancestry=(record.primitive_id,),
                )
                if (
                    equivalent_signature != record.signature
                    or self._identity(
                        equivalent_expression,
                        equivalent_signature,
                    )
                    != record.primitive_id
                ):
                    raise ValueError(
                        "stored equivalent recipe does not share canonical "
                        f"identity: {record.primitive_id}"
                    )
            self.expand_to_roots(record.primitive_id)

    def model_payload(self) -> dict[str, Any]:
        core = {
            "schema": PRIMITIVE_FORGE_SCHEMA,
            "runtime": PRIMITIVE_FORGE_RUNTIME,
            "root_substrate": list(ROOT_MECHANICS),
            "config": self.config.to_payload(),
            "sequence": self._sequence,
            "primitives": [record.to_payload() for record in self.records],
        }
        return {**core, "graph_hash": canonical_hash(core)}

    @property
    def graph_hash(self) -> str:
        return self.model_payload()["graph_hash"]

    @classmethod
    def from_model_payload(cls, payload: Mapping[str, Any]) -> PrimitiveForge:
        required = {
            "schema",
            "runtime",
            "root_substrate",
            "config",
            "sequence",
            "primitives",
            "graph_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("primitive graph payload keys are invalid")
        if payload["schema"] != PRIMITIVE_FORGE_SCHEMA:
            raise ValueError("unsupported primitive graph schema")
        if payload["runtime"] != PRIMITIVE_FORGE_RUNTIME:
            raise ValueError("primitive graph runtime marker is invalid")
        if payload["root_substrate"] != list(ROOT_MECHANICS):
            raise ValueError("primitive graph root substrate is invalid")
        graph_hash = payload["graph_hash"]
        if not isinstance(graph_hash, str) or len(graph_hash) != 64:
            raise ValueError("primitive graph hash is invalid")
        core = {key: payload[key] for key in payload if key != "graph_hash"}
        if canonical_hash(core) != graph_hash:
            raise ValueError("primitive graph hash mismatch; payload is corrupt")
        sequence = payload["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("primitive graph sequence is invalid")
        primitives = payload["primitives"]
        if not isinstance(primitives, list):
            raise ValueError("primitive graph records must be a list")
        forge = cls(ForgeConfig.from_payload(payload["config"]))
        parsed = [PrimitiveRecord.from_payload(item) for item in primitives]
        records = {record.primitive_id: record for record in parsed}
        if len(records) != len(parsed):
            raise ValueError("primitive graph ids must be unique")
        forge._records = records
        forge._sequence = sequence
        if any(record.updated_sequence > sequence for record in parsed):
            raise ValueError("primitive graph sequence trails a record update")
        forge._validate_graph()
        if forge.model_payload()["graph_hash"] != graph_hash:
            raise ValueError("primitive graph cannot reproduce its bound hash")
        return forge

    def save(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.model_payload(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return destination

    @classmethod
    def load(cls, path: Path) -> PrimitiveForge:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key in primitive graph: {key}")
                result[key] = value
            return result

        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("primitive graph file must contain an object")
        return cls.from_model_payload(payload)
