"""Build the private Kaggle bundle for emergent ontology discovery."""

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
    PROJECT_ROOT / "atom_ontology_dataset.py",
    PROJECT_ROOT / "atom_ontology_discovery.py",
    PROJECT_ROOT / "atom_ontology_side_view.py",
    PROJECT_ROOT / "atom_ontology_experiment.py",
)
BUNDLE_IMPORTS = """import argparse
import ast
import hashlib
import html
import json
import math
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
"""


def ontology_bundle_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_ontology_module_declarations(path: Path) -> str:
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
        if remove:
            removed.update(range(node.lineno, node.end_lineno + 1))
    return "".join(
        line
        for line_number, line in enumerate(lines, start=1)
        if line_number not in removed
    ).strip()


def build_ontology_bundle(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "atom_ontology_discovery_kaggle.py"
    sections = [strip_ontology_module_declarations(path) for path in SOURCE_FILES]
    bundle = (
        '"""Generated Kaggle bundle for emergent ontology discovery."""\n\n'
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
        "id": "jessealicea/atom-emergent-ontology-discovery-v1",
        "title": "Atom Emergent Ontology Discovery v1",
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
        "bundle_sha256": ontology_bundle_file_hash(bundle_path),
        "metadata_sha256": ontology_bundle_file_hash(metadata_path),
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): ontology_bundle_file_hash(path)
            for path in SOURCE_FILES
        },
    }
    (output_dir / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def parse_ontology_bundle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "ontology-v1",
    )
    return parser.parse_args()


def ontology_bundle_main() -> None:
    args = parse_ontology_bundle_args()
    manifest = build_ontology_bundle(args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    ontology_bundle_main()
