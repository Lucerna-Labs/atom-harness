"""Bridge the hash-bound Primitive Forge graph into Rust causal memory."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from atom_primitive_forge import PrimitiveForge, PrimitiveRecord


CAUSAL_MEMORY_RUNTIME = "atom-causal-memory-v1"
CAUSAL_MANIFEST_RUNTIME = "atom-causal-memory-manifest-v1"
CAUSAL_QUERY_RUNTIME = "atom-causal-memory-query-v1"
PROJECT_ROOT = Path(__file__).resolve().parent
RUST_WORKSPACE = PROJECT_ROOT / "atom_causal_memory_rust"
RELEASE_BINARY = (
    RUST_WORKSPACE
    / "target"
    / "release"
    / ("atom-causal-memory.exe" if __import__("os").name == "nt" else "atom-causal-memory")
)


def _strict_json(text: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in Rust response: {key}")
            result[key] = value
        return result

    payload = json.loads(text, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("Rust causal-memory response must be an object")
    return payload


def _hex(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("wire values must be non-empty NUL-free text")
    return value.encode("utf-8").hex()


def _dimension(powers: Sequence[tuple[str, int]]) -> str:
    if not powers:
        return "dimensionless"
    return "*".join(f"{label}^{exponent}" for label, exponent in powers)


def _feature(
    features: set[tuple[str, str]],
    role: str,
    value: str,
) -> None:
    if not role or not value:
        raise ValueError("causal motif role and value must be non-empty")
    features.add((role, value))


def _record_features(
    forge: PrimitiveForge,
    record: PrimitiveRecord,
) -> tuple[tuple[str, str], ...]:
    features: set[tuple[str, str]] = set()
    _feature(features, "kind", "root" if record.root else "derived")
    _feature(features, "status", record.status)
    _feature(features, "domain", record.signature.domain)
    for index, port in enumerate(record.signature.inputs):
        _feature(features, f"input/{index:04}/kind", port.kind)
        _feature(
            features,
            f"input/{index:04}/dimension",
            _dimension(port.dimension.powers),
        )
    _feature(features, "output/kind", record.signature.output.kind)
    _feature(
        features,
        "output/dimension",
        _dimension(record.signature.output.dimension.powers),
    )
    if record.recipe is not None:
        _feature(features, "recipe/mode", record.recipe.mode)
        for index, component in enumerate(record.recipe.components):
            _feature(features, f"component/{index:04}", component)
        for name, value in record.recipe.parameters:
            _feature(features, f"parameter/{name}", format(value, ".17g"))
        for index, root in enumerate(
            forge.expand_to_roots(record.primitive_id)
        ):
            _feature(features, f"root-expansion/{index:04}", root)
    for role, values in (
        ("alias", record.aliases),
        ("invariant", record.invariants),
        ("symmetry", record.symmetries),
        ("boundary", record.boundaries),
        ("scale", record.scales),
        ("provenance", record.provenance),
    ):
        for value in values:
            _feature(features, role, value)
    for recipe_index, recipe in enumerate(record.equivalent_recipes):
        _feature(
            features,
            f"equivalent/{recipe_index:04}/mode",
            recipe.mode,
        )
        for component_index, component in enumerate(recipe.components):
            _feature(
                features,
                (
                    f"equivalent/{recipe_index:04}/"
                    f"component/{component_index:04}"
                ),
                component,
            )
    for role, records in (
        ("evidence", record.evidence),
        ("counterexample", record.counterexamples),
    ):
        for index, item in enumerate(records):
            prefix = f"{role}/{index:04}"
            _feature(features, f"{prefix}/id", item.evidence_id)
            _feature(features, f"{prefix}/context", item.context_id)
            _feature(features, f"{prefix}/source", item.source)
            _feature(features, f"{prefix}/success", str(item.success).lower())
            _feature(features, f"{prefix}/residual", format(item.residual, ".17g"))
            for provenance in item.provenance:
                _feature(features, f"{prefix}/provenance", provenance)
    _feature(features, "metric/confidence", format(record.confidence, ".17g"))
    _feature(features, "metric/persistence", format(record.persistence, ".17g"))
    _feature(features, "sequence/created", str(record.created_sequence))
    _feature(features, "sequence/updated", str(record.updated_sequence))
    return tuple(sorted(features))


def build_causal_manifest(forge: PrimitiveForge) -> str:
    """Create deterministic, dependency-free wire data for the Rust importer."""

    forge_payload = forge.model_payload()
    if forge_payload["graph_hash"] != forge.graph_hash:
        raise ValueError("Primitive Forge graph changed during manifest creation")
    lines = [CAUSAL_MANIFEST_RUNTIME, f"H\t{forge.graph_hash}"]
    for record in forge.records:
        lines.append(f"G\t{_hex(record.primitive_id)}")
        for role, value in _record_features(forge, record):
            lines.append(f"F\t{_hex(role)}\t{_hex(value)}")
    return "\n".join(lines) + "\n"


STRUCTURAL_ROLE_PREFIXES = (
    "component/",
    "root-expansion/",
    "input/",
    "output/",
)
STRUCTURAL_ROLES = frozenset(
    {
        "boundary",
        "domain",
        "invariant",
        "kind",
        "recipe/mode",
        "scale",
        "status",
        "symmetry",
    }
)


def _is_structural_role(role: str) -> bool:
    return role in STRUCTURAL_ROLES or role.startswith(
        STRUCTURAL_ROLE_PREFIXES
    )


def structural_features_for(
    forge: PrimitiveForge,
    primitive_id: str,
) -> tuple[tuple[str, str], ...]:
    record = forge.get(primitive_id)
    return tuple(
        (role, value)
        for role, value in _record_features(forge, record)
        if _is_structural_role(role) and role != "status"
    )


def build_structural_query(
    *,
    query_id: str,
    features: Iterable[tuple[str, str]],
    required_roles: Iterable[str] = (),
    excluded_glyphs: Iterable[str] = (),
    minimum_support: int = 5,
    minimum_coverage_per_million: int = 650_000,
    limit: int = 12,
) -> str:
    feature_list = sorted(set(features))
    if not feature_list:
        raise ValueError("structural query requires at least one feature")
    if any(not _is_structural_role(role) for role, _ in feature_list):
        raise ValueError("structural query contains a lexical or audit-only role")
    required = frozenset(required_roles)
    if not 1 <= minimum_support <= len(feature_list):
        raise ValueError("minimum support is outside the query feature count")
    if not 0 <= minimum_coverage_per_million <= 1_000_000:
        raise ValueError("minimum coverage is outside [0, 1000000]")
    if not 1 <= limit <= 1024:
        raise ValueError("query limit is outside [1, 1024]")
    lines = [
        CAUSAL_QUERY_RUNTIME,
        (
            f"Q\t{_hex(query_id)}\t{minimum_support}\t{limit}\t"
            f"{minimum_coverage_per_million}"
        ),
    ]
    for role, value in feature_list:
        required_flag = int(
            role in required
            or any(
                role.startswith(f"{prefix}/")
                for prefix in required
                if prefix not in STRUCTURAL_ROLES
            )
        )
        lines.append(
            f"F\t{_hex(role)}\t{_hex(value)}\t{required_flag}"
        )
    for primitive_id in sorted(set(excluded_glyphs)):
        lines.append(f"X\t{_hex(primitive_id)}")
    return "\n".join(lines) + "\n"


def build_query_for_primitive(
    forge: PrimitiveForge,
    primitive_id: str,
    *,
    limit: int = 12,
) -> str:
    features = structural_features_for(forge, primitive_id)
    return build_structural_query(
        query_id=f"structural-resonance:{primitive_id}:{forge.graph_hash[:16]}",
        features=features,
        required_roles=(
            "domain",
            "kind",
            "recipe/mode",
            "input",
            "output",
        ),
        minimum_support=min(7, len(features)),
        minimum_coverage_per_million=650_000,
        limit=limit,
    )


def build_release_binary() -> Path:
    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "atom-causal-memory",
        ],
        cwd=RUST_WORKSPACE,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust causal-memory build failed:\n"
            + completed.stdout
            + completed.stderr
        )
    if not RELEASE_BINARY.is_file():
        raise RuntimeError("Rust causal-memory binary was not produced")
    return RELEASE_BINARY


def binary_sha256(binary: Path = RELEASE_BINARY) -> str:
    return hashlib.sha256(Path(binary).read_bytes()).hexdigest()


@dataclass(frozen=True)
class CausalMemoryClient:
    store_path: Path
    binary: Path = RELEASE_BINARY

    def __post_init__(self) -> None:
        if not Path(self.binary).is_file():
            raise ValueError(f"causal-memory binary is absent: {self.binary}")
        if Path(self.store_path).is_dir():
            raise ValueError("causal-memory store path cannot be a directory")

    def _run(
        self,
        command: str,
        *arguments: str,
        wire: str | None = None,
    ) -> dict[str, Any]:
        completed = subprocess.run(
            [
                str(self.binary),
                command,
                str(self.store_path),
                *arguments,
            ],
            input=wire,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Rust causal-memory {command} failed: {detail}"
            )
        payload = _strict_json(completed.stdout)
        if payload.get("runtime") != CAUSAL_MEMORY_RUNTIME:
            raise ValueError("Rust causal-memory runtime marker is invalid")
        return payload

    def import_forge(self, forge: PrimitiveForge) -> dict[str, Any]:
        return self._run("import", wire=build_causal_manifest(forge))

    def inventory(self) -> dict[str, Any]:
        return self._run("inventory")

    def query(self, query_wire: str) -> dict[str, Any]:
        return self._run("query", wire=query_wire)

    def observe_prediction(
        self,
        query_wire: str,
        *,
        expected_glyph: str,
        selected_glyph: str,
    ) -> dict[str, Any]:
        return self._run(
            "observe",
            expected_glyph,
            selected_glyph,
            wire=query_wire,
        )


def load_forge(path: Path) -> PrimitiveForge:
    """Strictly load and fully validate the hash-bound Forge artifact."""

    return PrimitiveForge.load(Path(path))
