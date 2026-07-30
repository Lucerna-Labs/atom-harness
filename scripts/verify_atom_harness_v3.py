"""Fail-closed repository policy checks for the Atom Harness V3 release."""

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
    ".github/workflows/atom-harness-v3-ci.yml",
    "ai-artifact-side-view.json",
    "ai-provider-fabric.json",
    "ai-run-transaction.json",
    "ai-runtime-knowledge.json",
    "ai-runtime-registry.json",
    "atom-language-model.json",
    "atom-language-harness-architecture.json",
    "atom_harness_experiment.py",
    "atom_harness_knowledge.py",
    "atom_harness_runtime.py",
    "atom_harness_session.py",
    "atom_harness_session_cli.py",
    "atom_harness_side_view.py",
    "atom_llm_protocol.py",
    "atom_llm_provider.py",
    "atom_language_model_contract.py",
    "atom_provider_fabric.py",
    "atom_resident_language_lane.py",
    "atom_run_transaction.py",
    "install-atom-language-model.ps1",
    "run-atom-harness-session.ps1",
    "rust-toolchain.toml",
    "scripts/certify_atom_language_model.py",
    "scripts/certify_resident_language_lane.py",
    "scripts/verify_atom_harness_v3.py",
    "tests/test_atom_language_harness_v2_integration.py",
    "tests/test_atom_language_harness_v3_integration.py",
    "tests/test_atom_provider_protocol_v2.py",
    "tests/test_atom_resident_language_lane.py",
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
    ".atom-harness-v3",
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
    """Raised when the repository is not safe to publish as V3."""


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
        raise PolicyFailure("required V3 files are absent: " + ", ".join(missing))


def _check_git_surface() -> dict[str, int]:
    candidates = _git_candidates()
    secret_files_scanned = 0
    for path in candidates:
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
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
    language_model = _load_json("atom-language-model.json")

    if registry.get("active_runtime") != "language-harness-v3":
        raise PolicyFailure("language-harness-v3 is not the active runtime")
    runtime = registry.get("runtimes", {}).get("language-harness-v3")
    if not isinstance(runtime, dict):
        raise PolicyFailure("language-harness-v3 registry entry is absent")
    expected_test = "tests/test_atom_language_harness_v3_integration.py"
    declarations = (knowledge, side_view, provider, transaction)
    for declaration in declarations:
        if declaration.get("project_kind") != "ai-harness":
            raise PolicyFailure("runtime declaration project kind is invalid")
        if declaration.get("runtime_entrypoint") != "atom_harness_experiment.py":
            raise PolicyFailure("runtime entrypoint declarations disagree")
        if declaration.get("integration_test") != expected_test:
            raise PolicyFailure("V3 integration-test declarations disagree")
    if runtime.get("integration_test") != expected_test:
        raise PolicyFailure("registry V3 integration test disagrees")
    if runtime.get("language_model_contract") != "atom-language-model.json":
        raise PolicyFailure("registry language-model contract is absent")
    if (
        runtime.get("language_model_certification")
        != "scripts/certify_resident_language_lane.py"
    ):
        raise PolicyFailure("registry language-model certification is absent")
    if runtime.get("session_entrypoint") != "atom_harness_session_cli.py":
        raise PolicyFailure("registry resident session entrypoint is absent")
    if runtime.get("resident_lane_runtime") != "ATOM_RESIDENT_LANGUAGE_LANE_RUNTIME":
        raise PolicyFailure("registry resident lane runtime is absent")
    if language_model.get("runtime") != "atom-language-model-contract-v1":
        raise PolicyFailure("official language-model runtime is invalid")

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
    _require_true(
        side,
        "selected_model_identity_visible",
        "model_load_latency_visible",
        "generation_throughput_visible",
        "primary_atom_claim_visible",
        "resident_lane_state_visible",
        "cold_start_and_warm_reuse_visible",
        "queue_wait_visible",
        "process_generation_visible",
        "intent_assistance_visible",
    )
    if side.get("placement") != "side":
        raise PolicyFailure("artifact view is not declared at the side")
    if side.get("module_path") != "atom_harness_side_view.py":
        raise PolicyFailure("artifact side-view module declaration is invalid")
    runtime_marker = side.get("runtime_marker")
    binding_marker = side.get("artifact_binding_marker")
    if not isinstance(runtime_marker, str) or len(runtime_marker.strip()) < 3:
        raise PolicyFailure("artifact side-view runtime marker is invalid")
    if not isinstance(binding_marker, str) or len(binding_marker.strip()) < 3:
        raise PolicyFailure("artifact side-view binding marker is invalid")
    entrypoint_text = (ROOT / "atom_harness_experiment.py").read_text(encoding="utf-8")
    side_view_text = (ROOT / "atom_harness_side_view.py").read_text(encoding="utf-8")
    integration_text = (ROOT / expected_test).read_text(encoding="utf-8")
    if runtime_marker not in entrypoint_text:
        raise PolicyFailure("runtime entrypoint does not wire the side-view marker")
    if binding_marker not in side_view_text:
        raise PolicyFailure("side-view module does not bind the artifact")
    for marker in (runtime_marker, binding_marker):
        if marker not in integration_text:
            raise PolicyFailure(
                f"V3 integration test does not exercise side-view marker {marker}"
            )
    _require_true(
        fabric,
        "ordered_fallback",
        "strict_json_required",
        "model_integrity_required",
        "noninteractive_completion_backend_required",
        "prompt_transport_declared",
        "machine_grounding_required",
        "default_local_provider",
        "bounded_retry_backoff",
        "explicit_provider_locations",
        "cloud_requires_explicit_consent",
        "circuit_breakers",
        "bounded_concurrency",
        "cancellation",
        "provider_cancellation_capability_exposed",
        "backpressure_vibrations",
        "resident_model_preload",
        "authenticated_loopback_transport",
        "external_proxy_disabled_for_loopback",
        "in_memory_transport_secret",
        "typed_resident_on_off_ramps",
        "bounded_resident_queue",
        "supervised_crash_recovery",
        "one_model_load_per_process_generation",
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

    artifact = language_model.get("artifact")
    policy = language_model.get("runtime_policy")
    if not isinstance(artifact, dict) or not isinstance(policy, dict):
        raise PolicyFailure("official language-model contract is incomplete")
    if language_model.get("default_provider") != "llama-cpp":
        raise PolicyFailure("official language-model provider is not local")
    if language_model.get("role") != "language-only-membrane":
        raise PolicyFailure("official language-model role is invalid")
    if language_model.get("adoption_status") != "certified-resident-local-default":
        raise PolicyFailure("official language-model adoption is not certified")
    if artifact.get("filename") != "qwen3-4b-instruct-2507-q8_0.gguf":
        raise PolicyFailure("official GGUF filename is invalid")
    if artifact.get("bytes") != 4_280_403_520:
        raise PolicyFailure("official GGUF byte count is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", ""))):
        raise PolicyFailure("official GGUF SHA-256 is invalid")
    if policy.get("harness_context_tokens") != 32_768:
        raise PolicyFailure("official harness context is invalid")
    if policy.get("chat_template") != "qwen-chatml-manual-v1":
        raise PolicyFailure("official chat template is invalid")
    if policy.get("executable") != "llama-server":
        raise PolicyFailure("official llama.cpp executable is invalid")
    resident_lane = policy.get("resident_lane")
    if not isinstance(resident_lane, dict):
        raise PolicyFailure("official resident language lane is absent")
    if (
        resident_lane.get("runtime") != "atom-resident-language-lane-v1"
        or resident_lane.get("topology") != "spiderweb-permanent-elevated-language-lane"
        or resident_lane.get("host") != "127.0.0.1"
        or resident_lane.get("parallel_slots") != 1
        or resident_lane.get("max_queue_depth") != 8
    ):
        raise PolicyFailure("official resident language lane policy is invalid")
    _require_true(
        resident_lane,
        "api_key_in_memory_only",
        "external_proxy_disabled",
        "preload_inference_path",
        "automatic_restart_on_next_request",
    )
    if resident_lane.get("web_ui_enabled") is not False:
        raise PolicyFailure("resident language lane web UI must remain disabled")
    _require_true(
        policy,
        "model_integrity_required",
        "local_is_default",
        "cloud_requires_explicit_provider_and_consent",
    )
    latest_evidence = language_model.get("certification", {}).get("latest_evidence")
    if not isinstance(latest_evidence, dict):
        raise PolicyFailure("official live-model certification is absent")
    _require_true(
        latest_evidence,
        "all_cases_passed",
        "machine_grounding_passed",
        "wiki_graph_and_rag_passed",
        "artifact_side_view_passed",
        "single_model_load_before_fault_passed",
        "warm_request_reuse_passed",
        "resident_backpressure_passed",
        "crash_recovery_passed",
        "expanded_domain_matrix_passed",
        "exact_vocabulary_intent_assistance_passed",
    )
    if latest_evidence.get("runtime") != "atom-resident-language-certification-v1":
        raise PolicyFailure("official resident certification runtime is invalid")
    if latest_evidence.get("case_count") != 20:
        raise PolicyFailure("official live-model case count is invalid")
    if latest_evidence.get("completion_count") != 36:
        raise PolicyFailure("official live-model completion count is invalid")
    if latest_evidence.get("domain_count") != 8:
        raise PolicyFailure("official live-model domain count is invalid")
    if not re.fullmatch(
        r"[0-9a-f]{64}",
        str(latest_evidence.get("report_sha256", "")),
    ):
        raise PolicyFailure("official live-model report hash is invalid")

    architecture_text = json.dumps(architecture, sort_keys=True)
    for marker in (
        "atom-language-harness-v3",
        "atom-resilient-provider-fabric-v3",
        "atom-run-transaction-v2",
        "atom-language-harness-wiki-v2",
        "atom-language-harness-graph-rag-v2",
        "atom-language-harness-side-view-v3",
        "atom-resident-language-lane-v1",
        "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-4b-instruct-2507-q8_0.gguf",
        "llama-server",
        "atom-resident-language-certification-v1",
        str(artifact["sha256"]),
    ):
        if marker not in architecture_text:
            raise PolicyFailure(f"architecture is missing runtime marker: {marker}")
    return {
        "active_runtime": str(registry["active_runtime"]),
        "integration_test": expected_test,
        "official_language_model": str(language_model["base_model"]["model_id"]),
    }


def _check_ci_contract() -> dict[str, str]:
    workflow = (ROOT / ".github" / "workflows" / "atom-harness-v3-ci.yml").read_text(
        encoding="utf-8"
    )
    for marker in (
        "ruff format --check",
        "ruff check",
        "py_compile",
        "scripts/verify_atom_harness_v3.py",
        "test_atom_language_harness_v3_integration.py",
        "test_atom_resident_language_lane.py",
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
        "atom_language_model_contract.py",
        "certify_resident_language_lane.py",
        "install-atom-language-model.ps1",
        "run-atom-harness-session.ps1",
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
    for requirement in ("numpy==2.4.6", "ruff==0.13.0", "torch==2.10.0"):
        if requirement not in requirements.splitlines():
            raise PolicyFailure(f"development dependency is not pinned: {requirement}")
    return {
        "python_ci": "3.13",
        "numpy_ci": "2.4.6",
        "node_ci": "24",
        "ruff_ci": "0.13.0",
        "rust_ci": str(channel),
        "torch_ci": "2.10.0",
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
                "policy": "atom-harness-v3",
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
