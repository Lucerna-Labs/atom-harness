"""Build the private Kaggle bundle for the Atom neural language field."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    PROJECT_ROOT / "atom_field_proof.py",
    PROJECT_ROOT / "atom_neural_language_dataset.py",
    PROJECT_ROOT / "atom_runtime_knowledge.py",
    PROJECT_ROOT / "atom_neural_language_model.py",
    PROJECT_ROOT / "atom_neural_language_side_view.py",
    PROJECT_ROOT / "atom_neural_language_experiment.py",
)
BUNDLE_IMPORTS = """import argparse
import ast
import copy
import hashlib
import html
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
"""


def neural_bundle_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def strip_neural_module_declarations(path: Path, *, keep_main: bool) -> str:
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


def build_neural_language_bundle(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "atom_neural_language_kaggle.py"
    sections = [
        strip_neural_module_declarations(
            path,
            keep_main=path == SOURCE_FILES[-1],
        )
        for path in SOURCE_FILES
    ]
    bundle = (
        '"""Generated Kaggle bundle for the Atom neural language field."""\n\n'
        "from __future__ import annotations\n\n"
        + BUNDLE_IMPORTS
        + "\n"
        + "\n\n".join(sections)
        + "\n"
    )
    ast.parse(bundle)
    bundle_path.write_text(bundle, encoding="utf-8", newline="\n")
    subprocess.run(
        ["ruff", "format", str(bundle_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = {
        "id": "jessealicea/atom-lifelong-neural-language-field-v1",
        "title": "Atom Lifelong Neural Language Field v1",
        "code_file": bundle_path.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
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
        "schema_version": 1,
        "bundle": bundle_path.name,
        "bundle_sha256": neural_bundle_file_hash(bundle_path),
        "metadata_sha256": neural_bundle_file_hash(metadata_path),
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): neural_bundle_file_hash(path)
            for path in SOURCE_FILES
        },
    }
    (output_dir / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_neural_bundle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "neural-language-v1",
    )
    return parser.parse_args()


def neural_bundle_main() -> None:
    args = parse_neural_bundle_args()
    manifest = build_neural_language_bundle(args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    neural_bundle_main()
