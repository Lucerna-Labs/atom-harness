"""Build the private Kaggle GPU bundle for causal platform synthesis."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    PROJECT_ROOT / "atom_causal_world_schema.py",
    PROJECT_ROOT / "atom_formal_domains.py",
    PROJECT_ROOT / "atom_runtime_knowledge.py",
    PROJECT_ROOT / "atom_platform_synthesis.py",
    PROJECT_ROOT / "atom_coding_harness.py",
    PROJECT_ROOT / "atom_coding_knowledge.py",
    PROJECT_ROOT / "atom_coding_side_view.py",
    PROJECT_ROOT / "atom_coding_experiment.py",
)
BUNDLE_IMPORTS = """import argparse
import ast
import hashlib
import html
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
"""


def coding_bundle_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def strip_coding_module_declarations(path: Path, *, keep_main: bool) -> str:
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
    command = (
        [formatter, "format", str(bundle_path)]
        if formatter is not None
        else [sys.executable, "-m", "ruff", "format", str(bundle_path)]
    )
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def build_coding_bundle(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "atom_coding_platform_kaggle.py"
    sections = [
        strip_coding_module_declarations(
            path,
            keep_main=path == SOURCE_FILES[-1],
        )
        for path in SOURCE_FILES
    ]
    bundle = (
        '"""Generated Kaggle GPU bundle for Atom causal platform synthesis."""\n\n'
        "from __future__ import annotations\n\n"
        + BUNDLE_IMPORTS
        + '\nKAGGLE_BUNDLE_ACCELERATOR = "GPU"\n\n'
        + "\n\n".join(sections)
        + "\n"
    )
    ast.parse(bundle)
    bundle_path.write_text(bundle, encoding="utf-8", newline="\n")
    _format_bundle(bundle_path)
    metadata = {
        "id": "jessealicea/atom-causal-coding-platform-v1",
        "title": "Atom Causal Coding Platform v1",
        "code_file": bundle_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
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
        "bundle_sha256": coding_bundle_file_hash(bundle_path),
        "metadata_sha256": coding_bundle_file_hash(metadata_path),
        "private": True,
        "accelerator": "GPU",
        "workload": "isolated-causal-code-interventions",
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): coding_bundle_file_hash(path)
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


def parse_coding_bundle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "coding-platform-v1",
    )
    return parser.parse_args()


def coding_bundle_main() -> None:
    args = parse_coding_bundle_args()
    manifest = build_coding_bundle(args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    coding_bundle_main()
