"""Diagnose exact transfer-policy replay differences against downloaded source."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_runtime(source_file: Path) -> ModuleType:
    resolved = source_file.resolve()
    module_name = f"atom_causal_policy_diagnostic_{resolved.stat().st_size}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load downloaded causal-world source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def collect_differences(
    saved: Any,
    regenerated: Any,
    *,
    path: str = "$",
    limit: int = 100,
) -> list[dict[str, Any]]:
    if saved == regenerated:
        return []
    if isinstance(saved, dict) and isinstance(regenerated, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(saved) | set(regenerated)):
            if len(differences) >= limit:
                break
            child = f"{path}.{key}"
            if key not in saved:
                differences.append(
                    {
                        "path": child,
                        "saved": "<missing>",
                        "regenerated": regenerated[key],
                    }
                )
            elif key not in regenerated:
                differences.append(
                    {"path": child, "saved": saved[key], "regenerated": "<missing>"}
                )
            else:
                differences.extend(
                    collect_differences(
                        saved[key],
                        regenerated[key],
                        path=child,
                        limit=limit - len(differences),
                    )
                )
        return differences
    if isinstance(saved, list) and isinstance(regenerated, list):
        differences = []
        maximum = max(len(saved), len(regenerated))
        for index in range(maximum):
            if len(differences) >= limit:
                break
            child = f"{path}[{index}]"
            if index >= len(saved):
                differences.append(
                    {
                        "path": child,
                        "saved": "<missing>",
                        "regenerated": regenerated[index],
                    }
                )
            elif index >= len(regenerated):
                differences.append(
                    {
                        "path": child,
                        "saved": saved[index],
                        "regenerated": "<missing>",
                    }
                )
            else:
                differences.extend(
                    collect_differences(
                        saved[index],
                        regenerated[index],
                        path=child,
                        limit=limit - len(differences),
                    )
                )
        return differences
    return [{"path": path, "saved": saved, "regenerated": regenerated}]


def diagnose_policy_replay(
    source_file: Path,
    artifact_dir: Path,
    *,
    wilson_decimals: int | None = None,
) -> dict[str, Any]:
    runtime = load_runtime(source_file)
    if wilson_decimals is not None:
        if not 0 <= wilson_decimals <= 15:
            raise ValueError("Wilson decimal normalization must be within [0, 15]")
        original_wilson = runtime.selective_error_upper_bound

        def normalized_wilson(false_assertions: int, asserted: int) -> float:
            return round(
                float(original_wilson(false_assertions, asserted)),
                wilson_decimals,
            )

        runtime.selective_error_upper_bound = normalized_wilson
    model = json.loads(
        (artifact_dir / "atom_causal_world_model.json").read_text(encoding="utf-8")
    )
    truth = json.loads(
        (artifact_dir / "atom_causal_world_transfer_validation_truth.json").read_text(
            encoding="utf-8"
        )
    )
    saved = json.loads(
        (artifact_dir / "atom_causal_world_transfer_policy.json").read_text(
            encoding="utf-8"
        )
    )
    regenerated = runtime.fit_transfer_policy(model, truth)
    differences = collect_differences(saved, regenerated)
    return {
        "schema": 1,
        "matched": not differences,
        "wilson_decimals": wilson_decimals,
        "saved_policy_hash": saved["policy_hash"],
        "regenerated_policy_hash": regenerated["policy_hash"],
        "differing_top_level_fields": sorted(
            key
            for key in set(saved) | set(regenerated)
            if saved.get(key) != regenerated.get(key)
        ),
        "difference_count": len(differences),
        "differences": differences,
    }


def diagnose_response_replay(
    source_file: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    runtime = load_runtime(source_file)
    model = json.loads(
        (artifact_dir / "atom_causal_world_model.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (artifact_dir / "atom_causal_world_transfer_request.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (artifact_dir / "atom_causal_world_transfer_policy.json").read_text(
            encoding="utf-8"
        )
    )
    saved = json.loads(
        (artifact_dir / "atom_causal_world_transfer_response.json").read_text(
            encoding="utf-8"
        )
    )
    regenerated = runtime.run_transfer_workflow(
        model,
        request,
        allow_contextual_transfer=True,
        transfer_policy=policy,
    )
    differences = collect_differences(saved, regenerated)
    return {
        "schema": 1,
        "target": "response",
        "matched": not differences,
        "saved_response_hash": saved["response_hash"],
        "regenerated_response_hash": regenerated["response_hash"],
        "differing_top_level_fields": sorted(
            key
            for key in set(saved) | set(regenerated)
            if saved.get(key) != regenerated.get(key)
        ),
        "difference_count": len(differences),
        "differences": differences,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_file", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--round-wilson-decimals", type=int)
    parser.add_argument(
        "--target",
        choices=("policy", "response"),
        default="policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target == "response":
        if args.round_wilson_decimals is not None:
            raise ValueError("Wilson override applies only to policy diagnosis")
        result = diagnose_response_replay(
            args.source_file.resolve(),
            args.artifact_dir.resolve(),
        )
    else:
        result = diagnose_policy_replay(
            args.source_file.resolve(),
            args.artifact_dir.resolve(),
            wilson_decimals=args.round_wilson_decimals,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
