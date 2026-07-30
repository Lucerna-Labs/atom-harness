"""Typed executable primitives for formal and scientific domain curricula."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from atom_causal_world_schema import canonical_hash


FORMAL_DOMAIN_RUNTIME = "atom-formal-domain-runtime-v1"
FORMAL_PRIMITIVE_REGISTRY_RUNTIME = "atom-formal-primitive-registry-v1"
FORMAL_CURRICULUM_RUNTIME = "atom-formal-curriculum-v1"
FORMAL_TRUTH_ORACLE_RUNTIME = "atom-formal-truth-oracle-v1"
FORMAL_PROGRAM_RUNTIME = "atom-formal-cross-domain-program-v1"
FORMAL_DOMAIN_SCHEMA = 1

FORMAL_DOMAIN_NAMES = (
    "logic",
    "algebra",
    "geometry",
    "calculus",
    "chemistry",
    "biology",
    "information_theory",
)

FORMAL_EPISTEMIC_STATES = (
    "proven",
    "observed",
    "causally_supported",
    "probable",
    "hypothesis",
    "unknown",
    "contradicted",
)


@dataclass(frozen=True)
class FormalPrimitive:
    name: str
    domain: str
    input_fields: tuple[str, ...]
    output_type: str
    precision: str
    root_mechanics: tuple[str, ...]
    invariants: tuple[str, ...]
    description: str


FORMAL_PRIMITIVES = (
    FormalPrimitive(
        "logic_implies",
        "logic",
        ("premise", "conclusion"),
        "boolean",
        "exact",
        ("radiation", "conservation"),
        ("truth_table",),
        "Evaluate material implication without converting uncertainty into truth.",
    ),
    FormalPrimitive(
        "logic_equivalent",
        "logic",
        ("left", "right"),
        "boolean",
        "exact",
        ("attraction_repulsion", "conservation"),
        ("truth_equivalence", "symmetry"),
        "Test whether two Boolean claims have the same truth value.",
    ),
    FormalPrimitive(
        "algebra_solve_linear",
        "algebra",
        ("coefficient", "offset", "result"),
        "rational",
        "exact",
        ("conservation", "attraction_repulsion"),
        ("equality", "inverse_operation"),
        "Solve coefficient*x + offset = result over exact rational numbers.",
    ),
    FormalPrimitive(
        "algebra_polynomial_value",
        "algebra",
        ("coefficients", "x"),
        "integer",
        "exact",
        ("nucleation", "conservation"),
        ("substitution", "operation_order"),
        "Evaluate a low-to-high integer polynomial with Horner composition.",
    ),
    FormalPrimitive(
        "geometry_distance_squared",
        "geometry",
        ("left", "right"),
        "integer",
        "exact",
        ("gravitation", "conservation"),
        ("translation_invariance", "nonnegative_distance"),
        "Measure squared Euclidean separation between two integer points.",
    ),
    FormalPrimitive(
        "geometry_triangle_twice_area",
        "geometry",
        ("a", "b", "c"),
        "integer",
        "exact",
        ("attraction_repulsion", "conservation"),
        ("translation_invariance", "orientation_independence"),
        "Measure twice a triangle's area using an exact determinant.",
    ),
    FormalPrimitive(
        "calculus_polynomial_derivative",
        "calculus",
        ("coefficients",),
        "integer_polynomial",
        "exact",
        ("radiation", "decay"),
        ("linearity", "local_change"),
        "Differentiate a low-to-high integer polynomial symbolically.",
    ),
    FormalPrimitive(
        "calculus_definite_integral",
        "calculus",
        ("coefficients", "lower", "upper"),
        "rational",
        "exact",
        ("radiation", "conservation"),
        ("linearity", "accumulation", "fundamental_theorem"),
        "Integrate an integer polynomial over rationally exact integer bounds.",
    ),
    FormalPrimitive(
        "chemistry_mass_conservation",
        "chemistry",
        ("reactant_masses", "product_masses"),
        "boolean",
        "exact",
        ("conservation",),
        ("mass_balance",),
        "Test whether a proposed reaction preserves total mass.",
    ),
    FormalPrimitive(
        "chemistry_stoichiometric_extent",
        "chemistry",
        ("available_moles", "coefficients"),
        "rational",
        "exact",
        ("conservation", "decay"),
        ("limiting_reagent", "nonnegative_extent"),
        "Compute the exact reaction extent allowed by the limiting reagent.",
    ),
    FormalPrimitive(
        "biology_homeostatic_error",
        "biology",
        ("target", "observed"),
        "integer",
        "exact",
        ("gravitation", "attraction_repulsion"),
        ("signed_feedback_error",),
        "Compute the signed correction needed to restore a target state.",
    ),
    FormalPrimitive(
        "biology_mendelian_distribution",
        "biology",
        ("parent_a", "parent_b"),
        "rational_distribution",
        "exact",
        ("nucleation", "conservation"),
        ("probability_mass", "allele_conservation"),
        "Derive exact one-locus offspring genotype probabilities.",
    ),
    FormalPrimitive(
        "information_binary_entropy",
        "information_theory",
        ("successes", "trials"),
        "decimal12_bits",
        "decimal12",
        ("dissipation", "conservation"),
        ("entropy_bounds", "binary_symmetry"),
        "Measure binary uncertainty with a deterministic decimal projection.",
    ),
    FormalPrimitive(
        "information_mutual_information",
        "information_theory",
        ("joint_counts",),
        "decimal12_bits",
        "decimal12",
        ("attraction_repulsion", "conservation"),
        ("nonnegative_information", "joint_normalization"),
        "Measure dependence in a two-by-two count table.",
    ),
    FormalPrimitive(
        "information_hartley_bits",
        "information_theory",
        ("symbol_count",),
        "integer",
        "exact",
        ("nucleation", "conservation"),
        ("injective_code_space",),
        "Return the minimum fixed-width bits needed to name every symbol.",
    ),
)

FORMAL_PRIMITIVE_INDEX = {primitive.name: primitive for primitive in FORMAL_PRIMITIVES}
if len(FORMAL_PRIMITIVE_INDEX) != len(FORMAL_PRIMITIVES):
    raise ValueError("formal primitive names must be unique")
if {primitive.domain for primitive in FORMAL_PRIMITIVES} != set(FORMAL_DOMAIN_NAMES):
    raise ValueError("every formal domain must own at least one primitive")


def _require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _require_integer_sequence(
    value: Any,
    field: str,
    *,
    minimum_length: int = 1,
) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) < minimum_length
    ):
        raise TypeError(f"{field} must be an integer sequence")
    return tuple(_require_integer(item, field) for item in value)


def _require_point(value: Any, field: str) -> tuple[int, int]:
    point = _require_integer_sequence(value, field, minimum_length=2)
    if len(point) != 2:
        raise ValueError(f"{field} must contain exactly two coordinates")
    return point


def _require_genotype(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 2
        or any(allele not in {"A", "a"} for allele in value)
    ):
        raise ValueError(f"{field} must be a two-allele A/a genotype")
    return value


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _decimal12(value: Decimal) -> str:
    quantum = Decimal("0.000000000001")
    projected = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    if projected == Decimal("-0.000000000000"):
        projected = Decimal("0.000000000000")
    return format(projected, "f")


def _decimal_log2(value: Decimal) -> Decimal:
    if value <= 0:
        raise ValueError("logarithm input must be positive")
    return value.ln() / Decimal(2).ln()


def _binary_entropy(successes: int, trials: int) -> str:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("binary counts must satisfy 0 <= successes <= trials")
    if successes in {0, trials}:
        return "0.000000000000"
    with localcontext() as context:
        context.prec = 80
        probability = Decimal(successes) / Decimal(trials)
        complement = Decimal(1) - probability
        entropy = -(
            probability * _decimal_log2(probability)
            + complement * _decimal_log2(complement)
        )
        return _decimal12(entropy)


def _runtime_logic_implies(arguments: Mapping[str, Any]) -> bool:
    premise = _require_boolean(arguments["premise"], "premise")
    conclusion = _require_boolean(arguments["conclusion"], "conclusion")
    return (not premise) or conclusion


def _oracle_logic_implies(arguments: Mapping[str, Any]) -> bool:
    values = (
        _require_boolean(arguments["premise"], "premise"),
        _require_boolean(arguments["conclusion"], "conclusion"),
    )
    return values != (True, False)


def _runtime_logic_equivalent(arguments: Mapping[str, Any]) -> bool:
    return _require_boolean(arguments["left"], "left") == _require_boolean(
        arguments["right"], "right"
    )


def _oracle_logic_equivalent(arguments: Mapping[str, Any]) -> bool:
    left = _require_boolean(arguments["left"], "left")
    right = _require_boolean(arguments["right"], "right")
    return (left and right) or (not left and not right)


def _runtime_algebra_solve_linear(arguments: Mapping[str, Any]) -> str:
    coefficient = _require_integer(arguments["coefficient"], "coefficient")
    if coefficient == 0:
        raise ValueError("linear coefficient must be nonzero")
    offset = _require_integer(arguments["offset"], "offset")
    result = _require_integer(arguments["result"], "result")
    return _fraction_text(Fraction(result - offset, coefficient))


def _oracle_algebra_solve_linear(arguments: Mapping[str, Any]) -> str:
    coefficient = _require_integer(arguments["coefficient"], "coefficient")
    offset = _require_integer(arguments["offset"], "offset")
    result = _require_integer(arguments["result"], "result")
    if coefficient == 0:
        raise ValueError("linear coefficient must be nonzero")
    solution = Fraction(result, coefficient) - Fraction(offset, coefficient)
    if coefficient * solution + offset != result:
        raise AssertionError("linear oracle failed its equality check")
    return _fraction_text(solution)


def _runtime_algebra_polynomial_value(arguments: Mapping[str, Any]) -> int:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    x_value = _require_integer(arguments["x"], "x")
    result = 0
    for coefficient in reversed(coefficients):
        result = result * x_value + coefficient
    return result


def _oracle_algebra_polynomial_value(arguments: Mapping[str, Any]) -> int:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    x_value = _require_integer(arguments["x"], "x")
    return sum(
        coefficient * x_value**power for power, coefficient in enumerate(coefficients)
    )


def _runtime_geometry_distance_squared(arguments: Mapping[str, Any]) -> int:
    left = _require_point(arguments["left"], "left")
    right = _require_point(arguments["right"], "right")
    delta_x = right[0] - left[0]
    delta_y = right[1] - left[1]
    return delta_x * delta_x + delta_y * delta_y


def _oracle_geometry_distance_squared(arguments: Mapping[str, Any]) -> int:
    left = _require_point(arguments["left"], "left")
    right = _require_point(arguments["right"], "right")
    return sum((right[index] - left[index]) ** 2 for index in range(2))


def _runtime_geometry_triangle_twice_area(arguments: Mapping[str, Any]) -> int:
    a = _require_point(arguments["a"], "a")
    b = _require_point(arguments["b"], "b")
    c = _require_point(arguments["c"], "c")
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _oracle_geometry_triangle_twice_area(arguments: Mapping[str, Any]) -> int:
    a = _require_point(arguments["a"], "a")
    b = _require_point(arguments["b"], "b")
    c = _require_point(arguments["c"], "c")
    shoelace = a[0] * b[1] + b[0] * c[1] + c[0] * a[1]
    reverse = a[1] * b[0] + b[1] * c[0] + c[1] * a[0]
    return abs(shoelace - reverse)


def _runtime_calculus_polynomial_derivative(
    arguments: Mapping[str, Any],
) -> list[int]:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    if len(coefficients) == 1:
        return [0]
    return [
        power * coefficient
        for power, coefficient in enumerate(coefficients)
        if power > 0
    ]


def _oracle_calculus_polynomial_derivative(
    arguments: Mapping[str, Any],
) -> list[int]:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    derived = [coefficients[index] * index for index in range(1, len(coefficients))]
    return derived or [0]


def _runtime_calculus_definite_integral(arguments: Mapping[str, Any]) -> str:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    lower = _require_integer(arguments["lower"], "lower")
    upper = _require_integer(arguments["upper"], "upper")
    result = sum(
        Fraction(coefficient, power + 1) * (upper ** (power + 1) - lower ** (power + 1))
        for power, coefficient in enumerate(coefficients)
    )
    return _fraction_text(result)


def _oracle_calculus_definite_integral(arguments: Mapping[str, Any]) -> str:
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    lower = _require_integer(arguments["lower"], "lower")
    upper = _require_integer(arguments["upper"], "upper")
    antiderivative = [
        Fraction(0),
        *(
            Fraction(coefficient, power + 1)
            for power, coefficient in enumerate(coefficients)
        ),
    ]
    upper_value = sum(
        coefficient * upper**power for power, coefficient in enumerate(antiderivative)
    )
    lower_value = sum(
        coefficient * lower**power for power, coefficient in enumerate(antiderivative)
    )
    return _fraction_text(upper_value - lower_value)


def _runtime_chemistry_mass_conservation(arguments: Mapping[str, Any]) -> bool:
    reactants = _require_integer_sequence(
        arguments["reactant_masses"], "reactant_masses"
    )
    products = _require_integer_sequence(arguments["product_masses"], "product_masses")
    if any(value < 0 for value in (*reactants, *products)):
        raise ValueError("masses must be nonnegative")
    return sum(reactants) == sum(products)


def _oracle_chemistry_mass_conservation(arguments: Mapping[str, Any]) -> bool:
    reactants = _require_integer_sequence(
        arguments["reactant_masses"], "reactant_masses"
    )
    products = _require_integer_sequence(arguments["product_masses"], "product_masses")
    if min(*reactants, *products) < 0:
        raise ValueError("masses must be nonnegative")
    balance = sum((*reactants, *(-value for value in products)))
    return balance == 0


def _runtime_chemistry_stoichiometric_extent(arguments: Mapping[str, Any]) -> str:
    available = _require_integer_sequence(
        arguments["available_moles"], "available_moles"
    )
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    if len(available) != len(coefficients):
        raise ValueError("mole and coefficient vectors must have equal length")
    if any(value < 0 for value in available) or any(
        coefficient <= 0 for coefficient in coefficients
    ):
        raise ValueError("stoichiometric values must be positive")
    return _fraction_text(
        min(
            Fraction(value, coefficient)
            for value, coefficient in zip(available, coefficients, strict=True)
        )
    )


def _oracle_chemistry_stoichiometric_extent(arguments: Mapping[str, Any]) -> str:
    available = _require_integer_sequence(
        arguments["available_moles"], "available_moles"
    )
    coefficients = _require_integer_sequence(arguments["coefficients"], "coefficients")
    if len(available) != len(coefficients):
        raise ValueError("mole and coefficient vectors must have equal length")
    candidates = sorted(
        Fraction(available[index], coefficients[index])
        for index in range(len(available))
        if coefficients[index] > 0 and available[index] >= 0
    )
    if len(candidates) != len(available):
        raise ValueError("stoichiometric values must be positive")
    return _fraction_text(candidates[0])


def _runtime_biology_homeostatic_error(arguments: Mapping[str, Any]) -> int:
    target = _require_integer(arguments["target"], "target")
    observed = _require_integer(arguments["observed"], "observed")
    return target - observed


def _oracle_biology_homeostatic_error(arguments: Mapping[str, Any]) -> int:
    target = _require_integer(arguments["target"], "target")
    observed = _require_integer(arguments["observed"], "observed")
    return -(observed - target)


def _canonical_genotype(left: str, right: str) -> str:
    return "".join(sorted((left, right), key=lambda allele: allele.islower()))


def _runtime_biology_mendelian_distribution(
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    parent_a = _require_genotype(arguments["parent_a"], "parent_a")
    parent_b = _require_genotype(arguments["parent_b"], "parent_b")
    counts = {"AA": 0, "Aa": 0, "aa": 0}
    for left in parent_a:
        for right in parent_b:
            counts[_canonical_genotype(left, right)] += 1
    return {name: _fraction_text(Fraction(count, 4)) for name, count in counts.items()}


def _oracle_biology_mendelian_distribution(
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    parent_a = _require_genotype(arguments["parent_a"], "parent_a")
    parent_b = _require_genotype(arguments["parent_b"], "parent_b")
    left_dominant = parent_a.count("A")
    right_dominant = parent_b.count("A")
    aa_count = left_dominant * right_dominant
    recessive_count = (2 - left_dominant) * (2 - right_dominant)
    mixed_count = 4 - aa_count - recessive_count
    return {
        "AA": _fraction_text(Fraction(aa_count, 4)),
        "Aa": _fraction_text(Fraction(mixed_count, 4)),
        "aa": _fraction_text(Fraction(recessive_count, 4)),
    }


def _runtime_information_binary_entropy(arguments: Mapping[str, Any]) -> str:
    return _binary_entropy(
        _require_integer(arguments["successes"], "successes"),
        _require_integer(arguments["trials"], "trials"),
    )


def _oracle_information_binary_entropy(arguments: Mapping[str, Any]) -> str:
    successes = _require_integer(arguments["successes"], "successes")
    trials = _require_integer(arguments["trials"], "trials")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("binary counts must satisfy 0 <= successes <= trials")
    if successes in {0, trials}:
        return "0.000000000000"
    with localcontext() as context:
        context.prec = 80
        probabilities = (
            Decimal(successes) / Decimal(trials),
            Decimal(trials - successes) / Decimal(trials),
        )
        return _decimal12(
            sum(
                -probability * _decimal_log2(probability)
                for probability in probabilities
            )
        )


def _runtime_information_mutual_information(
    arguments: Mapping[str, Any],
) -> str:
    counts = _require_integer_sequence(
        arguments["joint_counts"], "joint_counts", minimum_length=4
    )
    if len(counts) != 4 or any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("joint_counts must be four nonnegative counts")
    with localcontext() as context:
        context.prec = 80
        total = Decimal(sum(counts))
        row_totals = (counts[0] + counts[1], counts[2] + counts[3])
        column_totals = (counts[0] + counts[2], counts[1] + counts[3])
        information = Decimal(0)
        for index, count in enumerate(counts):
            if count == 0:
                continue
            row = index // 2
            column = index % 2
            joint = Decimal(count) / total
            independent = (
                Decimal(row_totals[row])
                / total
                * Decimal(column_totals[column])
                / total
            )
            information += joint * _decimal_log2(joint / independent)
        return _decimal12(information)


def _oracle_information_mutual_information(
    arguments: Mapping[str, Any],
) -> str:
    counts = _require_integer_sequence(
        arguments["joint_counts"], "joint_counts", minimum_length=4
    )
    if len(counts) != 4 or any(value < 0 for value in counts) or sum(counts) <= 0:
        raise ValueError("joint_counts must be four nonnegative counts")
    with localcontext() as context:
        context.prec = 80
        total = Decimal(sum(counts))

        def entropy(values: Sequence[int]) -> Decimal:
            return sum(
                -(Decimal(value) / total * _decimal_log2(Decimal(value) / total))
                for value in values
                if value > 0
            )

        rows = (counts[0] + counts[1], counts[2] + counts[3])
        columns = (counts[0] + counts[2], counts[1] + counts[3])
        return _decimal12(entropy(rows) + entropy(columns) - entropy(counts))


def _runtime_information_hartley_bits(arguments: Mapping[str, Any]) -> int:
    symbol_count = _require_integer(arguments["symbol_count"], "symbol_count")
    if symbol_count <= 0:
        raise ValueError("symbol_count must be positive")
    bits = 0
    capacity = 1
    while capacity < symbol_count:
        capacity *= 2
        bits += 1
    return bits


def _oracle_information_hartley_bits(arguments: Mapping[str, Any]) -> int:
    symbol_count = _require_integer(arguments["symbol_count"], "symbol_count")
    if symbol_count <= 0:
        raise ValueError("symbol_count must be positive")
    bits = max(0, (symbol_count - 1).bit_length())
    if 2**bits < symbol_count:
        raise AssertionError("Hartley oracle under-allocated its code space")
    return bits


FormalEvaluator = Callable[[Mapping[str, Any]], Any]

_RUNTIME_EVALUATORS: Mapping[str, FormalEvaluator] = {
    "logic_implies": _runtime_logic_implies,
    "logic_equivalent": _runtime_logic_equivalent,
    "algebra_solve_linear": _runtime_algebra_solve_linear,
    "algebra_polynomial_value": _runtime_algebra_polynomial_value,
    "geometry_distance_squared": _runtime_geometry_distance_squared,
    "geometry_triangle_twice_area": _runtime_geometry_triangle_twice_area,
    "calculus_polynomial_derivative": _runtime_calculus_polynomial_derivative,
    "calculus_definite_integral": _runtime_calculus_definite_integral,
    "chemistry_mass_conservation": _runtime_chemistry_mass_conservation,
    "chemistry_stoichiometric_extent": _runtime_chemistry_stoichiometric_extent,
    "biology_homeostatic_error": _runtime_biology_homeostatic_error,
    "biology_mendelian_distribution": _runtime_biology_mendelian_distribution,
    "information_binary_entropy": _runtime_information_binary_entropy,
    "information_mutual_information": _runtime_information_mutual_information,
    "information_hartley_bits": _runtime_information_hartley_bits,
}

_ORACLE_EVALUATORS: Mapping[str, FormalEvaluator] = {
    "logic_implies": _oracle_logic_implies,
    "logic_equivalent": _oracle_logic_equivalent,
    "algebra_solve_linear": _oracle_algebra_solve_linear,
    "algebra_polynomial_value": _oracle_algebra_polynomial_value,
    "geometry_distance_squared": _oracle_geometry_distance_squared,
    "geometry_triangle_twice_area": _oracle_geometry_triangle_twice_area,
    "calculus_polynomial_derivative": _oracle_calculus_polynomial_derivative,
    "calculus_definite_integral": _oracle_calculus_definite_integral,
    "chemistry_mass_conservation": _oracle_chemistry_mass_conservation,
    "chemistry_stoichiometric_extent": _oracle_chemistry_stoichiometric_extent,
    "biology_homeostatic_error": _oracle_biology_homeostatic_error,
    "biology_mendelian_distribution": _oracle_biology_mendelian_distribution,
    "information_binary_entropy": _oracle_information_binary_entropy,
    "information_mutual_information": _oracle_information_mutual_information,
    "information_hartley_bits": _oracle_information_hartley_bits,
}


def formal_domain_manifest() -> dict[str, Any]:
    primitive_payloads = [
        {
            **asdict(primitive),
            "input_fields": list(primitive.input_fields),
            "root_mechanics": list(primitive.root_mechanics),
            "invariants": list(primitive.invariants),
        }
        for primitive in FORMAL_PRIMITIVES
    ]
    core = {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "registry_runtime": FORMAL_PRIMITIVE_REGISTRY_RUNTIME,
        "domains": list(FORMAL_DOMAIN_NAMES),
        "epistemic_states": list(FORMAL_EPISTEMIC_STATES),
        "primitives": primitive_payloads,
        "domain_counts": {
            domain: sum(primitive.domain == domain for primitive in FORMAL_PRIMITIVES)
            for domain in FORMAL_DOMAIN_NAMES
        },
        "composition_contract": (
            "typed primitive outputs may feed compatible primitive inputs; "
            "every stage retains its proof trace and invariant obligations"
        ),
    }
    return {**core, "registry_hash": canonical_hash(core)}


def _validate_formal_request(request: Mapping[str, Any]) -> FormalPrimitive:
    if not isinstance(request, Mapping):
        raise TypeError("formal request must be an object")
    allowed = {
        "schema",
        "runtime",
        "query_id",
        "primitive",
        "arguments",
        "candidate",
    }
    if not set(request).issubset(allowed):
        raise ValueError("formal request fields are invalid")
    if request.get("schema") != FORMAL_DOMAIN_SCHEMA:
        raise ValueError("unsupported formal request schema")
    if request.get("runtime") != FORMAL_DOMAIN_RUNTIME:
        raise ValueError("formal request runtime mismatch")
    query_id = request.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("formal query ID must be non-empty")
    primitive_name = request.get("primitive")
    if (
        not isinstance(primitive_name, str)
        or primitive_name not in FORMAL_PRIMITIVE_INDEX
    ):
        raise ValueError("unknown formal primitive")
    arguments = request.get("arguments")
    if not isinstance(arguments, Mapping):
        raise TypeError("formal arguments must be an object")
    primitive = FORMAL_PRIMITIVE_INDEX[primitive_name]
    if set(arguments) != set(primitive.input_fields):
        raise ValueError("formal primitive arguments do not match its typed signature")
    return primitive


def solve_formal_request(request: Mapping[str, Any]) -> dict[str, Any]:
    primitive = _validate_formal_request(request)
    arguments = request["arguments"]
    value = _RUNTIME_EVALUATORS[primitive.name](arguments)
    has_candidate = "candidate" in request
    candidate_matches = not has_candidate or request["candidate"] == value
    claim_status = "proven" if candidate_matches else "contradicted"
    core = {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "runtime": FORMAL_DOMAIN_RUNTIME,
        "registry_runtime": FORMAL_PRIMITIVE_REGISTRY_RUNTIME,
        "query_id": request["query_id"],
        "domain": primitive.domain,
        "primitive": primitive.name,
        "claim_status": claim_status,
        "value": value,
        "candidate_matches": candidate_matches if has_candidate else None,
        "precision": primitive.precision,
        "root_mechanics": list(primitive.root_mechanics),
        "invariants": list(primitive.invariants),
        "proof_trace": [
            {
                "stage": "typed_binding",
                "input_fields": list(primitive.input_fields),
            },
            {
                "stage": "primitive_execution",
                "primitive": primitive.name,
                "runtime": FORMAL_DOMAIN_RUNTIME,
            },
            {
                "stage": "invariant_check",
                "invariants": list(primitive.invariants),
            },
            {
                "stage": "projective_measurement",
                "epistemic_state": claim_status,
            },
        ],
        "primitive_hash": canonical_hash(asdict(primitive)),
    }
    return {**core, "response_hash": canonical_hash(core)}


def _resolve_program_value(value: Any, outputs: Sequence[Any]) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            if (
                isinstance(reference, bool)
                or not isinstance(reference, int)
                or not 0 <= reference < len(outputs)
            ):
                raise ValueError("formal program reference is invalid")
            return outputs[reference]
        return {
            key: _resolve_program_value(item, outputs) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_program_value(item, outputs) for item in value]
    return value


def execute_formal_program(
    program_id: str,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(program_id, str) or not program_id:
        raise ValueError("formal program ID must be non-empty")
    if (
        isinstance(stages, (str, bytes))
        or not isinstance(stages, Sequence)
        or not stages
    ):
        raise ValueError("formal program must contain stages")
    outputs: list[Any] = []
    responses: list[dict[str, Any]] = []
    domains: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or set(stage) != {"primitive", "arguments"}:
            raise ValueError("formal program stages have invalid fields")
        arguments = _resolve_program_value(stage["arguments"], outputs)
        request = {
            "schema": FORMAL_DOMAIN_SCHEMA,
            "runtime": FORMAL_DOMAIN_RUNTIME,
            "query_id": f"{program_id}:{index}",
            "primitive": stage["primitive"],
            "arguments": arguments,
        }
        response = solve_formal_request(request)
        outputs.append(response["value"])
        responses.append(response)
        domains.append(response["domain"])
    core = {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "runtime": FORMAL_PROGRAM_RUNTIME,
        "program_id": program_id,
        "domains": list(dict.fromkeys(domains)),
        "stage_count": len(responses),
        "stages": responses,
        "value": outputs[-1],
        "claim_status": (
            "proven"
            if all(response["claim_status"] == "proven" for response in responses)
            else "contradicted"
        ),
    }
    return {**core, "program_hash": canonical_hash(core)}


def _case_arguments(primitive_name: str, index: int) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("formal curriculum index must be nonnegative")
    signed = (index % 11) - 5
    if primitive_name == "logic_implies":
        return {"premise": bool(index & 1), "conclusion": bool(index & 2)}
    if primitive_name == "logic_equivalent":
        return {"left": bool(index & 1), "right": bool(index & 4)}
    if primitive_name == "algebra_solve_linear":
        coefficient = (index % 7) + 1
        if index % 2:
            coefficient *= -1
        return {
            "coefficient": coefficient,
            "offset": signed,
            "result": signed * 2 + (index % 5),
        }
    if primitive_name == "algebra_polynomial_value":
        return {
            "coefficients": [signed, (index % 5) - 2, 2 - (index % 3), index % 2],
            "x": (index % 7) - 3,
        }
    if primitive_name == "geometry_distance_squared":
        return {
            "left": [signed, (index * 3) % 13 - 6],
            "right": [(index * 5) % 17 - 8, (index * 7) % 19 - 9],
        }
    if primitive_name == "geometry_triangle_twice_area":
        return {
            "a": [signed, (index % 5) - 2],
            "b": [signed + 2, (index * 2) % 7 - 3],
            "c": [signed - 1, (index * 3) % 11 - 5],
        }
    if primitive_name == "calculus_polynomial_derivative":
        return {
            "coefficients": [
                signed,
                (index % 7) - 3,
                (index * 2) % 9 - 4,
                (index % 3) - 1,
            ]
        }
    if primitive_name == "calculus_definite_integral":
        lower = (index % 5) - 3
        return {
            "coefficients": [signed, (index % 5) - 2, (index % 3) - 1],
            "lower": lower,
            "upper": lower + 1 + (index % 4),
        }
    if primitive_name == "chemistry_mass_conservation":
        reactants = [1 + index % 7, 2 + (index * 3) % 11]
        product_total = sum(reactants) + (1 if index % 3 == 0 else 0)
        return {
            "reactant_masses": reactants,
            "product_masses": [product_total // 2, product_total - product_total // 2],
        }
    if primitive_name == "chemistry_stoichiometric_extent":
        return {
            "available_moles": [2 + index % 9, 3 + (index * 2) % 11],
            "coefficients": [1 + index % 3, 1 + (index * 3) % 4],
        }
    if primitive_name == "biology_homeostatic_error":
        return {"target": 50 + signed, "observed": 45 + (index * 3) % 17}
    if primitive_name == "biology_mendelian_distribution":
        genotypes = ("AA", "Aa", "aa")
        return {
            "parent_a": genotypes[index % len(genotypes)],
            "parent_b": genotypes[(index // len(genotypes)) % len(genotypes)],
        }
    if primitive_name == "information_binary_entropy":
        trials = 2 + index % 31
        return {"successes": (index * 7) % (trials + 1), "trials": trials}
    if primitive_name == "information_mutual_information":
        return {
            "joint_counts": [
                1 + index % 13,
                1 + (index * 3) % 11,
                1 + (index * 5) % 17,
                1 + (index * 7) % 19,
            ]
        }
    if primitive_name == "information_hartley_bits":
        return {"symbol_count": 1 + (index * 37) % 4096}
    raise ValueError("unknown formal curriculum primitive")


def _cross_domain_programs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "program_id": "calculus-to-algebra",
            "stages": [
                {
                    "primitive": "calculus_polynomial_derivative",
                    "arguments": {"coefficients": [1, 2, 3]},
                },
                {
                    "primitive": "algebra_polynomial_value",
                    "arguments": {"coefficients": {"$ref": 0}, "x": 2},
                },
            ],
            "expected": 14,
        },
        {
            "program_id": "geometry-to-algebra",
            "stages": [
                {
                    "primitive": "geometry_distance_squared",
                    "arguments": {"left": [0, 0], "right": [3, 4]},
                },
                {
                    "primitive": "algebra_solve_linear",
                    "arguments": {
                        "coefficient": 1,
                        "offset": 0,
                        "result": {"$ref": 0},
                    },
                },
            ],
            "expected": "25/1",
        },
        {
            "program_id": "chemistry-to-logic",
            "stages": [
                {
                    "primitive": "chemistry_mass_conservation",
                    "arguments": {
                        "reactant_masses": [2, 14],
                        "product_masses": [16],
                    },
                },
                {
                    "primitive": "logic_implies",
                    "arguments": {"premise": True, "conclusion": {"$ref": 0}},
                },
            ],
            "expected": True,
        },
        {
            "program_id": "biology-to-algebra",
            "stages": [
                {
                    "primitive": "biology_homeostatic_error",
                    "arguments": {"target": 100, "observed": 92},
                },
                {
                    "primitive": "algebra_solve_linear",
                    "arguments": {
                        "coefficient": 2,
                        "offset": 0,
                        "result": {"$ref": 0},
                    },
                },
            ],
            "expected": "4/1",
        },
        {
            "program_id": "information-to-algebra",
            "stages": [
                {
                    "primitive": "information_hartley_bits",
                    "arguments": {"symbol_count": 257},
                },
                {
                    "primitive": "algebra_solve_linear",
                    "arguments": {
                        "coefficient": 3,
                        "offset": 0,
                        "result": {"$ref": 0},
                    },
                },
            ],
            "expected": "3/1",
        },
    )


def run_formal_domain_benchmark(
    *,
    cases_per_primitive: int = 24,
) -> dict[str, Any]:
    if (
        isinstance(cases_per_primitive, bool)
        or not isinstance(cases_per_primitive, int)
        or cases_per_primitive < 12
    ):
        raise ValueError("formal benchmark needs at least twelve cases per primitive")
    records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    domain_totals = {domain: 0 for domain in FORMAL_DOMAIN_NAMES}
    domain_correct = {domain: 0 for domain in FORMAL_DOMAIN_NAMES}
    partition_counts = {"demonstration": 0, "validation": 0, "heldout": 0}
    samples: list[dict[str, Any]] = []
    for primitive in FORMAL_PRIMITIVES:
        for index in range(cases_per_primitive):
            remainder = index % 4
            partition = (
                "demonstration"
                if remainder in {0, 1}
                else "validation"
                if remainder == 2
                else "heldout"
            )
            arguments = _case_arguments(primitive.name, index)
            request = {
                "schema": FORMAL_DOMAIN_SCHEMA,
                "runtime": FORMAL_DOMAIN_RUNTIME,
                "query_id": f"{primitive.name}:{index}",
                "primitive": primitive.name,
                "arguments": arguments,
            }
            response = solve_formal_request(request)
            oracle_value = _ORACLE_EVALUATORS[primitive.name](arguments)
            correct = response["value"] == oracle_value
            record = {
                "query_id": request["query_id"],
                "domain": primitive.domain,
                "primitive": primitive.name,
                "partition": partition,
                "claim_status": response["claim_status"],
                "correct": correct,
                "response_hash": response["response_hash"],
            }
            truth = {
                "query_id": request["query_id"],
                "oracle_runtime": FORMAL_TRUTH_ORACLE_RUNTIME,
                "value": oracle_value,
            }
            records.append(record)
            truth_records.append(truth)
            domain_totals[primitive.domain] += 1
            domain_correct[primitive.domain] += int(correct)
            partition_counts[partition] += 1
            if partition == "heldout" and not any(
                sample["domain"] == primitive.domain for sample in samples
            ):
                samples.append(
                    {
                        "domain": primitive.domain,
                        "primitive": primitive.name,
                        "arguments": arguments,
                        "value": response["value"],
                        "claim_status": response["claim_status"],
                    }
                )
    contradiction_checks: list[dict[str, Any]] = []
    for domain in FORMAL_DOMAIN_NAMES:
        primitive = next(item for item in FORMAL_PRIMITIVES if item.domain == domain)
        arguments = _case_arguments(primitive.name, cases_per_primitive + 1)
        response = solve_formal_request(
            {
                "schema": FORMAL_DOMAIN_SCHEMA,
                "runtime": FORMAL_DOMAIN_RUNTIME,
                "query_id": f"contradiction:{domain}",
                "primitive": primitive.name,
                "arguments": arguments,
                "candidate": {"intentionally_incorrect": True},
            }
        )
        contradiction_checks.append(
            {
                "domain": domain,
                "primitive": primitive.name,
                "claim_status": response["claim_status"],
                "passed": response["claim_status"] == "contradicted",
            }
        )
    program_results = [
        {
            **execute_formal_program(program["program_id"], program["stages"]),
            "expected": program["expected"],
        }
        for program in _cross_domain_programs()
    ]
    for result in program_results:
        result["passed"] = (
            result["claim_status"] == "proven"
            and result["value"] == result["expected"]
            and len(result["domains"]) == 2
        )
    manifest = formal_domain_manifest()
    gate_checks = {
        "all_formal_domains_exercised": all(domain_totals.values()),
        "runtime_and_oracle_implementations_are_disjoint": all(
            _RUNTIME_EVALUATORS[name] is not _ORACLE_EVALUATORS[name]
            for name in FORMAL_PRIMITIVE_INDEX
        ),
        "runtime_matches_independent_oracle": all(
            record["correct"] for record in records
        ),
        "heldout_partition_exercised": partition_counts["heldout"] > 0,
        "every_exact_result_is_marked_proven": all(
            record["claim_status"] == "proven" for record in records
        ),
        "false_candidates_are_contradicted": all(
            check["passed"] for check in contradiction_checks
        ),
        "cross_domain_programs_are_proven": all(
            result["passed"] for result in program_results
        ),
        "registry_is_hash_bound": manifest["registry_hash"]
        == canonical_hash(
            {key: value for key, value in manifest.items() if key != "registry_hash"}
        ),
    }
    report_core = {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "runtime": FORMAL_CURRICULUM_RUNTIME,
        "formal_runtime": FORMAL_DOMAIN_RUNTIME,
        "oracle_runtime": FORMAL_TRUTH_ORACLE_RUNTIME,
        "program_runtime": FORMAL_PROGRAM_RUNTIME,
        "registry_hash": manifest["registry_hash"],
        "primitive_count": len(FORMAL_PRIMITIVES),
        "domain_count": len(FORMAL_DOMAIN_NAMES),
        "cases_per_primitive": cases_per_primitive,
        "case_count": len(records),
        "partition_counts": partition_counts,
        "truth_hash": canonical_hash(truth_records),
        "response_hash": canonical_hash(records),
        "per_domain": {
            domain: {
                "cases": domain_totals[domain],
                "correct": domain_correct[domain],
                "accuracy": domain_correct[domain] / domain_totals[domain],
            }
            for domain in FORMAL_DOMAIN_NAMES
        },
        "epistemic_contract": {
            "states": list(FORMAL_EPISTEMIC_STATES),
            "exact_runtime_state": "proven",
            "false_candidate_state": "contradicted",
            "unsupported_state": "unknown",
            "deduction_is_not_induction_or_abduction": True,
        },
        "contradiction_checks": contradiction_checks,
        "cross_domain_programs": program_results,
        "samples": samples,
        "gates": gate_checks,
        "passed": all(gate_checks.values()),
    }
    report = {**report_core, "report_hash": canonical_hash(report_core)}
    return {
        "schema": FORMAL_DOMAIN_SCHEMA,
        "manifest": manifest,
        "report": report,
    }


def formal_domain_self_test() -> dict[str, bool]:
    benchmark = run_formal_domain_benchmark(cases_per_primitive=12)
    report = benchmark["report"]
    checks = {
        "seven_domains": report["domain_count"] == 7,
        "typed_registry": report["primitive_count"] == len(FORMAL_PRIMITIVES),
        "oracle_agreement": report["gates"]["runtime_matches_independent_oracle"],
        "contradiction_detected": report["gates"]["false_candidates_are_contradicted"],
        "cross_domain_composition": report["gates"]["cross_domain_programs_are_proven"],
        "benchmark_passed": report["passed"],
    }
    if not all(checks.values()):
        raise AssertionError(f"formal-domain self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(formal_domain_self_test())
