"""Deterministic opaque stream for homeostatic learning-control experiments.

Runtime rows expose only an opaque cue, an opaque observed effect, and salience.
Regime boundaries, whether an observation is noise, and evaluator truth live in
a separate map that is never passed to the learning controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


HOMEOSTATIC_DATASET_SCHEMA = 1
HOMEOSTATIC_DATASET_SEED = 20_260_721_03
OPAQUE_CUES = ("k0m", "k1r", "k2v", "k3x")
OPAQUE_EFFECTS = ("s0q", "s1b", "s2n", "s3t")
BASE_LAWS = dict(zip(OPAQUE_CUES, OPAQUE_EFFECTS, strict=True))
SHIFTED_LAWS = {
    "k0m": "s1b",
    "k1r": "s0q",
    "k2v": "s3t",
    "k3x": "s2n",
}

_OPAQUE_TOKEN = re.compile(r"[a-z][a-z0-9]{2,15}")


def homeostatic_dataset_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_homeostatic_observation(row: Mapping[str, Any]) -> None:
    expected = {"event_id", "cue", "effect", "salience"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"observation fields must be {sorted(expected)}")
    if not isinstance(row["event_id"], str) or not row["event_id"]:
        raise ValueError("event_id must be non-empty text")
    for field in ("cue", "effect"):
        value = row[field]
        if not isinstance(value, str) or _OPAQUE_TOKEN.fullmatch(value) is None:
            raise ValueError(f"{field} must be an opaque token")
    salience = row["salience"]
    if (
        isinstance(salience, bool)
        or not isinstance(salience, (int, float))
        or not math.isfinite(float(salience))
        or not 0.1 <= float(salience) <= 2.0
    ):
        raise ValueError("salience must be finite and within [0.1, 2.0]")


def _event_id(index: int, cue: str, effect: str) -> str:
    digest = homeostatic_dataset_hash(
        [HOMEOSTATIC_DATASET_SEED, index, cue, effect]
    )[:18]
    return f"field-event-{digest}"


def _append_observation(
    rows: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    cue: str,
    effect: str,
    expected_effect: str,
    regime: str,
    *,
    is_noise: bool,
) -> None:
    index = len(rows)
    event_id = _event_id(index, cue, effect)
    row = {
        "event_id": event_id,
        "cue": cue,
        "effect": effect,
        "salience": 1.0,
    }
    validate_homeostatic_observation(row)
    rows.append(row)
    truth[event_id] = {
        "regime": regime,
        "expected_effect": expected_effect,
        "is_noise": is_noise,
    }


def build_homeostatic_program() -> dict[str, Any]:
    """Build stable, noisy, and coherent-shift phases without runtime labels."""

    rows: list[dict[str, Any]] = []
    evaluator_truth: dict[str, dict[str, Any]] = {}

    for index in range(48):
        cue = OPAQUE_CUES[index % len(OPAQUE_CUES)]
        _append_observation(
            rows,
            evaluator_truth,
            cue,
            BASE_LAWS[cue],
            BASE_LAWS[cue],
            "initial_crystallization",
            is_noise=False,
        )

    for index in range(40):
        cue = OPAQUE_CUES[index % len(OPAQUE_CUES)]
        alternatives = tuple(
            effect for effect in OPAQUE_EFFECTS if effect != BASE_LAWS[cue]
        )
        observed = alternatives[(index // len(OPAQUE_CUES)) % len(alternatives)]
        _append_observation(
            rows,
            evaluator_truth,
            cue,
            observed,
            BASE_LAWS[cue],
            "noise_burst",
            is_noise=True,
        )

    for index in range(16):
        cue = OPAQUE_CUES[index % len(OPAQUE_CUES)]
        _append_observation(
            rows,
            evaluator_truth,
            cue,
            BASE_LAWS[cue],
            BASE_LAWS[cue],
            "recovery",
            is_noise=False,
        )

    for index in range(96):
        cue = OPAQUE_CUES[index % len(OPAQUE_CUES)]
        _append_observation(
            rows,
            evaluator_truth,
            cue,
            SHIFTED_LAWS[cue],
            SHIFTED_LAWS[cue],
            "law_shift",
            is_noise=False,
        )

    for index in range(32):
        cue = OPAQUE_CUES[index % len(OPAQUE_CUES)]
        _append_observation(
            rows,
            evaluator_truth,
            cue,
            SHIFTED_LAWS[cue],
            SHIFTED_LAWS[cue],
            "consolidation",
            is_noise=False,
        )

    regime_counts: dict[str, int] = {}
    for item in evaluator_truth.values():
        regime = str(item["regime"])
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    manifest = {
        "schema_version": HOMEOSTATIC_DATASET_SCHEMA,
        "seed": HOMEOSTATIC_DATASET_SEED,
        "observations": len(rows),
        "cue_count": len(OPAQUE_CUES),
        "effect_count": len(OPAQUE_EFFECTS),
        "regime_counts": dict(sorted(regime_counts.items())),
        "runtime_fields": sorted(rows[0]),
        "evaluator_fields": sorted(next(iter(evaluator_truth.values()))),
        "evaluator_truth_separate": all(
            not ({"regime", "expected_effect", "is_noise"} & set(row))
            for row in rows
        ),
        "stream_sha256": homeostatic_dataset_hash(rows),
        "evaluator_sha256": homeostatic_dataset_hash(evaluator_truth),
    }
    return {
        "observations": rows,
        "evaluator_truth": evaluator_truth,
        "final_truth": dict(sorted(SHIFTED_LAWS.items())),
        "manifest": manifest,
    }


def write_homeostatic_program(output_dir: Path) -> dict[str, Path]:
    program = build_homeostatic_program()
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "atom_homeostatic_observations.jsonl"
    observations_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in program["observations"]
        ),
        encoding="utf-8",
        newline="\n",
    )
    paths = {
        "observations": observations_path,
        "evaluator_truth": output_dir / "atom_homeostatic_evaluator_truth.json",
        "manifest": output_dir / "atom_homeostatic_dataset_manifest.json",
    }
    for name in ("evaluator_truth", "manifest"):
        paths[name].write_text(
            json.dumps(program[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return paths


if __name__ == "__main__":
    target = Path("homeostatic_outputs")
    for artifact, path in write_homeostatic_program(target).items():
        print(f"{artifact}: {path}")
