"""Exhaustively audit the causal-world Wilson risk boundary through 576 turns."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atom_causal_world_transfer import selective_error_upper_bound  # noqa: E402


Z_SCORE = 1.959963984540054


def legacy_float_wilson(false_assertions: int, asserted: int) -> float:
    if asserted == 0:
        return 1.0
    rate = false_assertions / asserted
    squared = Z_SCORE**2
    denominator = 1.0 + squared / asserted
    center = rate + squared / (2.0 * asserted)
    radius = Z_SCORE * (
        rate * (1.0 - rate) / asserted
        + squared / (4.0 * asserted**2)
    ) ** 0.5
    return round(float((center + radius) / denominator), 12)


def audit_wilson_portability(maximum_assertions: int) -> dict[str, object]:
    if (
        isinstance(maximum_assertions, bool)
        or not isinstance(maximum_assertions, int)
        or maximum_assertions <= 0
    ):
        raise ValueError("maximum assertions must be a positive integer")
    boundary_differences: list[dict[str, object]] = []
    values: list[dict[str, int | float]] = []
    for asserted in range(maximum_assertions + 1):
        for false_assertions in range(asserted + 1):
            deterministic = selective_error_upper_bound(
                false_assertions,
                asserted,
            )
            if (
                not 0.0 <= deterministic <= 1.0
                or deterministic != round(deterministic, 12)
            ):
                raise AssertionError("deterministic Wilson contract failed")
            legacy = legacy_float_wilson(false_assertions, asserted)
            values.append(
                {
                    "false_assertions": false_assertions,
                    "asserted": asserted,
                    "deterministic": deterministic,
                }
            )
            if legacy != deterministic:
                boundary_differences.append(
                    {
                        "false_assertions": false_assertions,
                        "asserted": asserted,
                        "legacy_float": legacy,
                        "deterministic_decimal": deterministic,
                    }
                )
    values_json = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": 1,
        "maximum_assertions": maximum_assertions,
        "count_pairs": len(values),
        "deterministic_value_sha256": hashlib.sha256(values_json).hexdigest(),
        "legacy_boundary_difference_count": len(boundary_differences),
        "legacy_boundary_differences": boundary_differences,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-assertions", type=int, default=576)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_wilson_portability(args.maximum_assertions)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
