"""Build a self-contained private Kaggle accelerator causal-world bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    PROJECT_ROOT / "atom_causal_world_schema.py",
    PROJECT_ROOT / "atom_causal_world_curriculum.py",
    PROJECT_ROOT / "atom_causal_world_simulator.py",
    PROJECT_ROOT / "atom_causal_graph.py",
    PROJECT_ROOT / "atom_causal_world_language.py",
    PROJECT_ROOT / "atom_formal_domains.py",
    PROJECT_ROOT / "atom_causal_world_knowledge.py",
    PROJECT_ROOT / "atom_causal_world_transfer.py",
    PROJECT_ROOT / "atom_causal_world_side_view.py",
    PROJECT_ROOT / "atom_causal_world_accelerator.py",
    PROJECT_ROOT / "atom_causal_world_experiment.py",
)
BUNDLE_IMPORTS = """import argparse
import ast
import copy
import hashlib
import html
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
"""


def causal_bundle_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def strip_causal_module_declarations(path: Path, *, keep_main: bool) -> str:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    removed: set[int] = set()
    for node in tree.body:
        remove = (
            isinstance(node, (ast.Import, ast.ImportFrom))
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
            or (
                not keep_main
                and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "main"
            )
            or (not keep_main and _is_main_guard(node))
        )
        if remove:
            removed.update(range(node.lineno, node.end_lineno + 1))
    return "".join(
        line
        for line_number, line in enumerate(lines, start=1)
        if line_number not in removed
    ).strip()


def _format_bundle(bundle_path: Path) -> None:
    formatter = shutil.which("ruff")
    if formatter is None:
        raise RuntimeError("ruff executable is required to format the Kaggle bundle")
    subprocess.run(
        [formatter, "format", str(bundle_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def build_causal_world_bundle(
    output_dir: Path,
    *,
    accelerator: str,
) -> dict[str, object]:
    accelerator = accelerator.lower()
    if accelerator not in {"tpu", "gpu"}:
        raise ValueError("Kaggle accelerator must be TPU or GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "atom_causal_world_kaggle.py"
    sections = [
        strip_causal_module_declarations(
            path,
            keep_main=path == SOURCE_FILES[-1],
        )
        for path in SOURCE_FILES
    ]
    bundle = (
        '"""Generated Kaggle accelerator bundle for the Atom causal world."""\n\n'
        "from __future__ import annotations\n\n"
        + BUNDLE_IMPORTS
        + f'\nKAGGLE_BUNDLE_ACCELERATOR = "{accelerator}"\n'
        + "\n"
        + "\n\n".join(sections)
        + "\n"
    )
    ast.parse(bundle)
    bundle_path.write_text(bundle, encoding="utf-8", newline="\n")
    _format_bundle(bundle_path)
    kernel_slug = (
        "atom-massive-causal-world-v1"
        if accelerator == "tpu"
        else "atom-massive-causal-world-gpu-v1"
    )
    kernel_title = (
        "Atom Massive Causal World v1"
        if accelerator == "tpu"
        else "Atom Massive Causal World GPU v1"
    )
    metadata = {
        "id": f"jessealicea/{kernel_slug}",
        "title": kernel_title,
        "code_file": bundle_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(accelerator == "gpu").lower(),
        "enable_tpu": str(accelerator == "tpu").lower(),
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema": 1,
        "bundle": bundle_path.name,
        "bundle_sha256": causal_bundle_file_hash(bundle_path),
        "metadata_sha256": causal_bundle_file_hash(metadata_path),
        "private": True,
        "accelerator": accelerator.upper(),
        "entrypoint_profile": "tpu-massive",
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): causal_bundle_file_hash(path)
            for path in SOURCE_FILES
        },
    }
    manifest_path = output_dir / "bundle-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_causal_bundle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination directory; defaults to the selected accelerator lane",
    )
    parser.add_argument(
        "--accelerator",
        choices=("tpu", "gpu"),
        default="tpu",
    )
    return parser.parse_args()


def causal_bundle_main() -> None:
    args = parse_causal_bundle_args()
    output_dir = args.output_dir
    if output_dir is None:
        lane = (
            "causal-world-gpu-v1"
            if args.accelerator == "gpu"
            else "causal-world-v1"
        )
        output_dir = PROJECT_ROOT / "kaggle" / lane
    manifest = build_causal_world_bundle(
        output_dir.resolve(),
        accelerator=args.accelerator,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    causal_bundle_main()
