"""Fail-closed repository policy checks for the Atom Harness V2 release."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
MAX_RUST_CRATE_LINES = 4_000
RUST_SUFFIX = "." + "r" + "s"

REQUIRED_FILES = (
    ".github/workflows/atom-harness-v2-ci.yml",
    "ai-artifact-side-view.json",
    "ai-provider-fabric.json",
    "ai-run-transaction.json",
    "ai-runtime-knowledge.json",
    "ai-runtime-registry.json",
    "atom-language-harness-architecture.json",
    "atom_harness_experiment.py",
    "atom_harness_knowledge.py",
    "atom_harness_runtime.py",
    "atom_harness_side_view.py",
    "atom_llm_protocol.py",
    "atom_llm_provider.py",
    "atom_provider_fabric.py",
    "atom_run_transaction.py",
    "rust-toolchain.toml",
    "tests/test_atom_language_harness_v2_integration.py",
    "tests/test_atom_provider_protocol_v2.py",
)

FORBIDDEN_GIT_SUFFIXES = {
    ".atomdb",
    ".ckpt",
    ".gguf",
    ".key",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
FORBIDDEN_GIT_PARTS = {
    "__pycache__",
    ".atom-harness-v2",
    ".pytest_cache",
    ".ruff_cache",
    "atom_harness_outputs",
    "target",
}
TEXT_SUFFIXES = {
    ".env",
    ".example",
    ".html",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    RUST_SUFFIX,
    ".svelte",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


class PolicyFailure(RuntimeError):
    """Raised when the repository is not safe to publish as V2."""


def _load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyFailure(f"{relative_path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PolicyFailure(f"{relative_path}: top-level JSON must be an object")
    return payload


def _git_candidates() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    names = result.stdout.decode("utf-8", errors="strict").split("\0")
    return [ROOT / name for name in names if name]


def _check_required_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise PolicyFailure("required V2 files are absent: " + ", ".join(missing))


def _check_git_surface() -> dict[str, int]:
    candidates = _git_candidates()
    secret_files_scanned = 0
    for path in candidates:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.casefold() for part in relative.parts}
        if lowered_parts & FORBIDDEN_GIT_PARTS:
            raise PolicyFailure(f"generated path is a Git candidate: {relative}")
        if path.suffix.casefold() in FORBIDDEN_GIT_SUFFIXES:
            raise PolicyFailure(f"forbidden artifact is a Git candidate: {relative}")
        size = path.stat().st_size
        if size > MAX_GIT_FILE_BYTES:
            raise PolicyFailure(
                f"Git candidate exceeds {MAX_GIT_FILE_BYTES} bytes: {relative}"
            )
        if path.suffix.casefold() not in TEXT_SUFFIXES or size > 2 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise PolicyFailure(f"credential-shaped text found in {relative}")
        secret_files_scanned += 1
    return {
        "git_candidates": len(candidates),
        "secret_files_scanned": secret_files_scanned,
    }


def _require_true(mapping: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if mapping.get(key) is not True:
            raise PolicyFailure(f"required boolean is not true: {key}")


def _check_runtime_declarations() -> dict[str, str]:
    registry = _load_json("ai-runtime-registry.json")
    knowledge = _load_json("ai-runtime-knowledge.json")
    side_view = _load_json("ai-artifact-side-view.json")
    provider = _load_json("ai-provider-fabric.json")
    transaction = _load_json("ai-run-transaction.json")
    architecture = _load_json("atom-language-harness-architecture.json")

    if registry.get("active_runtime") != "language-harness-v2":
        raise PolicyFailure("language-harness-v2 is not the active runtime")
    runtime = registry.get("runtimes", {}).get("language-harness-v2")
    if not isinstance(runtime, dict):
        raise PolicyFailure("language-harness-v2 registry entry is absent")
    expected_test = "tests/test_atom_language_harness_v2_integration.py"
    declarations = (knowledge, side_view, provider, transaction)
    for declaration in declarations:
        if declaration.get("runtime_entrypoint") != "atom_harness_experiment.py":
            raise PolicyFailure("runtime entrypoint declarations disagree")
        if declaration.get("integration_test") != expected_test:
            raise PolicyFailure("V2 integration-test declarations disagree")
    if runtime.get("integration_test") != expected_test:
        raise PolicyFailure("registry V2 integration test disagrees")

    wiki = knowledge.get("wiki_graph")
    rag = knowledge.get("rag")
    side = side_view.get("side_view")
    fabric = provider.get("provider_fabric")
    run_transaction = transaction.get("run_transaction")
    for name, value in (
        ("wiki_graph", wiki),
        ("rag", rag),
        ("side_view", side),
        ("provider_fabric", fabric),
        ("run_transaction", run_transaction),
    ):
        if not isinstance(value, dict):
            raise PolicyFailure(f"{name} declaration is absent")
        _require_true(value, "enabled")

    _require_true(
        side,
        "user_visible",
        "bound_to_real_artifact_output",
    )
    if side.get("placement") != "side":
        raise PolicyFailure("artifact view is not declared at the side")
    _require_true(
        fabric,
        "ordered_fallback",
        "strict_json_required",
        "bounded_retry_backoff",
        "explicit_provider_locations",
        "cloud_requires_explicit_consent",
        "circuit_breakers",
        "bounded_concurrency",
        "cancellation",
        "provider_cancellation_capability_exposed",
        "backpressure_vibrations",
    )
    if fabric.get("all_providers_must_be_preemptible") is not False:
        raise PolicyFailure("provider cancellation capability must remain honest")
    _require_true(
        run_transaction,
        "staged_writes",
        "atomic_directory_publication",
        "target_locking",
        "crash_recovery",
        "committed_file_manifest",
    )
    if run_transaction.get("overwrite_allowed") is not False:
        raise PolicyFailure("run transactions must refuse overwrite")

    architecture_text = json.dumps(architecture, sort_keys=True)
    for marker in (
        "atom-language-harness-v2",
        "atom-resilient-provider-fabric-v2",
        "atom-run-transaction-v2",
        "atom-language-harness-wiki-v2",
        "atom-language-harness-graph-rag-v2",
        "atom-language-harness-side-view-v2",
    ):
        if marker not in architecture_text:
            raise PolicyFailure(f"architecture is missing runtime marker: {marker}")
    return {
        "active_runtime": str(registry["active_runtime"]),
        "integration_test": expected_test,
    }


def _check_ci_contract() -> dict[str, str]:
    workflow = (ROOT / ".github" / "workflows" / "atom-harness-v2-ci.yml").read_text(
        encoding="utf-8"
    )
    for marker in (
        "ruff format --check",
        "ruff check",
        "py_compile",
        "scripts/verify_atom_harness_v2.py",
        "test_atom_language_harness_v2_integration.py",
        "test_atom_provider_protocol_v2.py",
        "test_atom_language_harness_integration.py",
        "test_atom_causal_live_integration.py",
        "npm ci --ignore-scripts",
        "test_*.py",
        "Run privacy-blocked launcher end to end",
        "verify_committed_run",
        "cargo fmt",
        "cargo clippy",
        "-D warnings",
        "cargo test",
        "Language.Parser",
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "persist-credentials: false",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
        "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
        'node-version: "24"',
    ):
        if marker not in workflow:
            raise PolicyFailure(f"root CI is missing required gate: {marker}")
    lowered = workflow.casefold()
    if (
        "allow-cloud" in lowered
        or "openrouter_api_key" in lowered
        or "secrets." in lowered
    ):
        raise PolicyFailure("root CI must not authorize or inject cloud-provider data")

    toolchain_path = ROOT / "rust-toolchain.toml"
    with toolchain_path.open("rb") as stream:
        toolchain = tomllib.load(stream)
    channel = toolchain.get("toolchain", {}).get("channel")
    if channel != "1.96.0":
        raise PolicyFailure("Rust toolchain is not pinned to 1.96.0")
    components = set(toolchain.get("toolchain", {}).get("components", []))
    if not {"clippy", "rustfmt"} <= components:
        raise PolicyFailure("Rust toolchain lacks clippy or rustfmt")

    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    for requirement in ("numpy==2.4.6", "ruff==0.13.0"):
        if requirement not in requirements.splitlines():
            raise PolicyFailure(f"development dependency is not pinned: {requirement}")
    return {
        "python_ci": "3.13",
        "numpy_ci": "2.4.6",
        "node_ci": "24",
        "ruff_ci": "0.13.0",
        "rust_ci": str(channel),
    }


def _cargo_metadata() -> dict[str, Any]:
    manifest = ROOT / "atom_causal_memory_rust" / "Cargo.toml"
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise PolicyFailure("Cargo metadata is not an object")
    return payload


def _check_rust_crate_sizes() -> dict[str, int]:
    metadata = _cargo_metadata()
    packages: dict[str, dict[str, Any]] = {}
    explicit_owners: dict[Path, str] = {}
    for package in metadata.get("packages", []):
        name = str(package["name"])
        packages[name] = package
        for target in package.get("targets", []):
            source = Path(target["src_path"]).resolve()
            previous = explicit_owners.setdefault(source, name)
            if previous != name:
                raise PolicyFailure(
                    f"Rust target {source} is owned by both {previous} and {name}"
                )
    if not packages:
        raise PolicyFailure("Cargo metadata reported no Rust packages")

    rust_root = (ROOT / "atom_causal_memory_rust").resolve()
    all_sources = {
        path.resolve()
        for path in rust_root.rglob("*" + RUST_SUFFIX)
        if "target" not in {part.casefold() for part in path.parts}
    }
    assigned: set[Path] = set()
    line_counts: dict[str, int] = {}
    library_filename = "lib" + RUST_SUFFIX
    for name, package in sorted(packages.items()):
        sources: set[Path] = set()
        for target in package.get("targets", []):
            source = Path(target["src_path"]).resolve()
            sources.add(source)
            if source.name != library_filename:
                continue
            for candidate in source.parent.rglob("*" + RUST_SUFFIX):
                resolved = candidate.resolve()
                owner = explicit_owners.get(resolved)
                if owner is None or owner == name:
                    sources.add(resolved)
        line_count = sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in sorted(sources)
        )
        if line_count > MAX_RUST_CRATE_LINES:
            raise PolicyFailure(
                f"Rust crate {name} has {line_count} source lines; "
                f"limit is {MAX_RUST_CRATE_LINES}"
            )
        assigned.update(sources)
        line_counts[name] = line_count
    unassigned = sorted(all_sources - assigned)
    if unassigned:
        rendered = ", ".join(str(path.relative_to(ROOT)) for path in unassigned)
        raise PolicyFailure("Rust source is not assigned to a crate: " + rendered)
    return line_counts


def main() -> int:
    try:
        _check_required_files()
        git_surface = _check_git_surface()
        declarations = _check_runtime_declarations()
        ci_contract = _check_ci_contract()
        rust_crates = _check_rust_crate_sizes()
    except (
        PolicyFailure,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        print(f"POLICY FAILURE: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "policy": "atom-harness-v2",
                **git_surface,
                **declarations,
                **ci_contract,
                "rust_crate_lines": rust_crates,
                "rust_crate_line_limit": MAX_RUST_CRATE_LINES,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
