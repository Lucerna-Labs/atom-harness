"""Build the single-file Kaggle script for the Atom phase-law experiment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    PROJECT_ROOT / "atom_runtime_knowledge.py",
    PROJECT_ROOT / "atom_phase_side_view.py",
    PROJECT_ROOT / "atom_phase_law_experiment.py",
)
BUNDLE_IMPORTS = """import argparse
import ast
import hashlib
import html
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
"""


def stable_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_module_declarations(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    removed: set[int] = set()
    for node in tree.body:
        remove = isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if not remove:
            continue
        for line_number in range(node.lineno, node.end_lineno + 1):
            removed.add(line_number)
    return "".join(
        line
        for line_number, line in enumerate(lines, start=1)
        if line_number not in removed
    ).strip()


def build_bundle(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "atom_phase_law_kaggle.py"
    sections = [strip_module_declarations(path) for path in SOURCE_FILES]
    bundle = (
        '"""Generated single-file Kaggle bundle for Atom phase-law v3."""\n\n'
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
        "id": "jessealicea/atom-emergent-phase-law-v3",
        "title": "Atom Emergent Phase Law v3",
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
        "bundle_sha256": stable_file_hash(bundle_path),
        "metadata_sha256": stable_file_hash(metadata_path),
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): stable_file_hash(path)
            for path in SOURCE_FILES
        },
    }
    (output_dir / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "phase-law-v3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_bundle(args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
