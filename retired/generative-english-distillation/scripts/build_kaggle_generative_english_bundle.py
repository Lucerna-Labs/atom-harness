"""Build the single-source Kaggle runner for generative Atom English."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "kaggle" / "generative-english-v1"
SOURCE_FILES = (
    "atom_english_core.py",
    "atom_english_data.py",
    "atom_english_training.py",
    "atom_english_evaluation.py",
    "atom_english_context.py",
    "atom_english_knowledge.py",
    "atom_english_side_view.py",
    "atom_english_kaggle.py",
)
LOCAL_MODULES = {Path(name).stem for name in SOURCE_FILES}
KAGGLE_TORCH_BOOTSTRAP = r"""
import os as _bootstrap_os
import subprocess as _bootstrap_subprocess
import sys as _bootstrap_sys


def _ensure_kaggle_pascal_torch() -> None:
    if _bootstrap_os.environ.get("ATOM_TORCH_COMPAT_BOOTSTRAPPED") == "1":
        return
    try:
        capabilities = _bootstrap_subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            stderr=_bootstrap_subprocess.DEVNULL,
            text=True,
            timeout=15,
        ).splitlines()
    except (
        FileNotFoundError,
        _bootstrap_subprocess.SubprocessError,
    ):
        return
    requires_compatible_wheel = any(
        value.strip().startswith(("5.", "6."))
        for value in capabilities
    )
    if not requires_compatible_wheel:
        return
    _bootstrap_subprocess.check_call(
        [
            _bootstrap_sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--upgrade",
            "--force-reinstall",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )
    environment = dict(_bootstrap_os.environ)
    environment["ATOM_TORCH_COMPAT_BOOTSTRAPPED"] = "1"
    _bootstrap_os.execve(
        _bootstrap_sys.executable,
        [_bootstrap_sys.executable, *_bootstrap_sys.argv],
        environment,
    )


_ensure_kaggle_pascal_torch()
""".strip()
KAGGLE_EVALUATION_BOOTSTRAP = r"""
import importlib.metadata as _evaluation_metadata
import subprocess as _evaluation_subprocess
import sys as _evaluation_sys


def _ensure_kaggle_language_evaluation() -> None:
    if _bundle_os.environ.get("ATOM_ENGLISH_DEFAULT_MODE") != "evaluate":
        return
    target = "0.4.12"
    try:
        installed = _evaluation_metadata.version("lm_eval")
    except _evaluation_metadata.PackageNotFoundError:
        installed = None
    if installed == target:
        return
    _evaluation_subprocess.check_call(
        [
            _evaluation_sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "lm_eval[ifeval]==0.4.12",
        ]
    )


_ensure_kaggle_language_evaluation()
""".strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _without_local_imports(source: str, filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    removed: set[int] = set()
    for node in ast.walk(tree):
        remove = False
        if isinstance(node, ast.ImportFrom):
            remove = node.module == "__future__" or node.module in LOCAL_MODULES
        elif isinstance(node, ast.Import):
            remove = any(alias.name in LOCAL_MODULES for alias in node.names)
        if remove:
            end = int(getattr(node, "end_lineno", node.lineno))
            removed.update(range(int(node.lineno), end + 1))
    lines = source.splitlines()
    retained = [
        line for number, line in enumerate(lines, start=1) if number not in removed
    ]
    return "\n".join(retained).strip() + "\n"


def build_bundle(
    output_directory: Path,
    *,
    kernel_id: str = "jessealicea/atom-generative-english-v1",
    title: str = "Atom Generative English v1",
    default_mode: str = "train",
    default_stage: str = "foundation",
    default_sequence_tokens: int | None = None,
    default_gradient_accumulation: int = 16,
    kernel_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    if default_mode not in {"train", "evaluate", "self-test"}:
        raise ValueError("unsupported bundle mode")
    if default_stage not in {"foundation", "dialogue", "context"}:
        raise ValueError("unsupported bundle stage")
    if "/" not in kernel_id:
        raise ValueError("kernel_id must have owner/slug form")
    if default_sequence_tokens is None:
        default_sequence_tokens = (
            1_024
            if default_stage == "foundation"
            else (2_048 if default_stage == "dialogue" else 256)
        )
    if default_sequence_tokens < 1:
        raise ValueError("default sequence tokens must be positive")
    if default_gradient_accumulation < 1:
        raise ValueError("default gradient accumulation must be positive")
    if default_mode == "evaluate":
        corpus = (
            "Wikitext-103, LAMBADA, BLiMP, HellaSwag, full IFEval, "
            "and 32K-through-512K RULER-style context probes"
        )
        teacher = None
    elif default_stage == "context":
        corpus = (
            "deterministic multi-family context curriculum from 2K through "
            "264K and 512K over held-out Wikitext distractors"
        )
        teacher = None
    elif default_stage == "dialogue":
        corpus = "complete SmolTalk all-config free-form dialogue corpus"
        teacher = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    else:
        corpus = "full FineWeb-Edu and full Cosmopedia-v2 streaming mixture"
        teacher = "HuggingFaceTB/SmolLM2-1.7B"
    output_directory.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    sections = [
        '"""Generated single-source Kaggle runner for Atom English."""',
        "",
        "from __future__ import annotations",
        "",
        "import os as _bundle_os",
        (
            '_bundle_os.environ.setdefault("ATOM_ENGLISH_DEFAULT_MODE", '
            f"{default_mode!r})"
        ),
        (
            '_bundle_os.environ.setdefault("ATOM_ENGLISH_DEFAULT_STAGE", '
            f"{default_stage!r})"
        ),
        (
            '_bundle_os.environ.setdefault("ATOM_ENGLISH_DEFAULT_SEQUENCE_LENGTH", '
            f"{str(default_sequence_tokens)!r})"
        ),
        (
            "_bundle_os.environ.setdefault("
            '"ATOM_ENGLISH_DEFAULT_GRADIENT_ACCUMULATION", '
            f"{str(default_gradient_accumulation)!r})"
        ),
        "",
        KAGGLE_TORCH_BOOTSTRAP,
        "",
        KAGGLE_EVALUATION_BOOTSTRAP,
        "",
    ]
    for name in SOURCE_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[name] = _sha256(path)
        sections.extend(
            (
                "",
                f"# ---- embedded source: {name} ----",
                _without_local_imports(
                    path.read_text(encoding="utf-8"),
                    name,
                ),
            )
        )
    bundle_path = output_directory / "atom_generative_english_kaggle.py"
    bundle_path.write_text("\n".join(sections), encoding="utf-8")
    metadata = {
        "code_file": bundle_path.name,
        "competition_sources": [],
        "dataset_sources": [],
        "enable_gpu": "true",
        "enable_internet": "true",
        "enable_tpu": "false",
        "id": kernel_id,
        "is_private": "true",
        "kernel_sources": list(kernel_sources),
        "kernel_type": "script",
        "language": "python",
        "model_sources": [],
        "title": title,
    }
    metadata_path = output_directory / "kernel-metadata.json"
    _write_json(metadata_path, metadata)
    manifest_body = {
        "schema_version": 1,
        "bundle": bundle_path.name,
        "bundle_sha256": _sha256(bundle_path),
        "metadata_sha256": _sha256(metadata_path),
        "sources": source_hashes,
        "execution": {
            "default_mode": default_mode,
            "default_stage": default_stage,
            "profile": "scale-227m",
            "corpus": corpus,
            "teacher": teacher,
            "foundation_target_tokens": 4_915_200_000,
            "dialogue_target_optimizer_steps": 130_490,
            "context_target_optimizer_steps": 2_600,
            "model_context_tokens": 524_288,
            "required_context_evaluation_tokens": [
                32_768,
                65_536,
                131_072,
                264_000,
                524_288,
            ],
            "default_sequence_tokens": default_sequence_tokens,
            "default_gradient_accumulation": default_gradient_accumulation,
            "internet_required": True,
            "accelerator_requested": "GPU",
            "evaluation_dependency": (
                "lm_eval[ifeval]==0.4.12" if default_mode == "evaluate" else None
            ),
            "pascal_compatibility": (
                "PyTorch 2.7.1 CUDA 12.6 bootstrap when compute capability is below 7.0"
            ),
        },
    }
    manifest = dict(manifest_body)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(output_directory / "bundle-manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--kernel-id",
        default="jessealicea/atom-generative-english-v1",
    )
    parser.add_argument("--title", default="Atom Generative English v1")
    parser.add_argument(
        "--default-mode",
        choices=("train", "evaluate", "self-test"),
        default="train",
    )
    parser.add_argument(
        "--default-stage",
        choices=("foundation", "dialogue", "context"),
        default="foundation",
    )
    parser.add_argument("--default-sequence-tokens", type=int)
    parser.add_argument(
        "--default-gradient-accumulation",
        type=int,
        default=16,
    )
    parser.add_argument("--kernel-source", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle(
        args.output_directory,
        kernel_id=args.kernel_id,
        title=args.title,
        default_mode=args.default_mode,
        default_stage=args.default_stage,
        default_sequence_tokens=args.default_sequence_tokens,
        default_gradient_accumulation=args.default_gradient_accumulation,
        kernel_sources=tuple(args.kernel_source),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
