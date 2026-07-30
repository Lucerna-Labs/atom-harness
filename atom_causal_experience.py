"""Full causal-world experience ingestion for Atom Causal Memory."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from atom_causal_graph import (
    CAUSAL_GRAPH_MODEL_SCHEMA,
    CAUSAL_GRAPH_RUNTIME,
    CausalLaw,
)
from atom_causal_memory import RELEASE_BINARY
from atom_causal_world_schema import (
    DOMAIN_NAMES,
    FEATURE_INDEX,
    CausalEvidence,
    canonical_hash,
)

EXPERIENCE_BATCH_RUNTIME = "atom-causal-experience-batch-v1"
EXPERIENCE_QUERY_RUNTIME = "atom-causal-experience-query-v1"
EXPERIENCE_MEMORY_RUNTIME = "atom-causal-experience-v1"
ROOT_PRIMITIVES = frozenset(
    {
        "attraction_repulsion",
        "conservation",
        "decay",
        "dissipation",
        "gravitation",
        "nucleation",
        "radiation",
    }
)
QUERYABLE_ROLES = frozenset(
    {
        "cause",
        "effect",
        "context",
        "domain",
        "direction",
        "kind",
        "status",
        "delay",
        "magnitude",
        "invariant",
        "support",
        "confidence",
        "contradiction",
    }
)


def _strict_json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    payload = json.loads(text, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("JSON response must be an object")
    return payload


def _hex(value: str) -> str:
    if not value:
        raise ValueError("wire values cannot be empty")
    return value.encode("utf-8").hex()


def _direction(value: int) -> str:
    if value not in {-1, 0, 1}:
        raise ValueError("causal direction must be -1, 0, or 1")
    return f"{value:+d}" if value else "0"


def _log_bucket(value: float) -> str:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("metric bucket input must be finite and nonnegative")
    if value == 0.0:
        return "zero"
    exponent = math.floor(math.log10(value))
    return f"10^{exponent}"


def _count_bucket(value: int) -> str:
    if value < 0:
        raise ValueError("count bucket input cannot be negative")
    if value == 0:
        return "0"
    lower = 1 << (value.bit_length() - 1)
    upper = (lower << 1) - 1
    return f"{lower}-{upper}"


def _confidence_bucket(value: float) -> str:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and inside [0, 1]")
    return f"decile:{min(9, int(value * 10.0))}"


def _experience_id(kind: str, source_id: str) -> str:
    if kind not in {"observation", "law"} or not source_id:
        raise ValueError("experience identity inputs are invalid")
    return f"experience:{kind}:{source_id}"


def _features(
    values: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: set[tuple[str, str]] = set()
    for role, value in values:
        role = str(role)
        value = str(value)
        if not role or not value or "\0" in role or "\0" in value:
            raise ValueError("experience features must be non-empty and NUL-free")
        normalized.add((role, value))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    features: tuple[tuple[str, str], ...]

    def feature_values(self, role: str) -> tuple[str, ...]:
        return tuple(value for name, value in self.features if name == role)


@dataclass(frozen=True)
class ExperienceCorpus:
    evidence_hash: str
    model_hash: str
    observations: tuple[ExperienceRecord, ...]
    laws: tuple[ExperienceRecord, ...]
    law_payloads: tuple[dict[str, Any], ...]

    @property
    def all_records(self) -> tuple[ExperienceRecord, ...]:
        return self.observations + self.laws


def _load_evidence(path: Path) -> tuple[tuple[CausalEvidence, ...], str]:
    expected_fields = {item.name for item in fields(CausalEvidence)}
    payloads: list[dict[str, Any]] = []
    evidence: list[CausalEvidence] = []
    revisions: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"blank evidence line at {line_number}")
        payload = _strict_json(line)
        if set(payload) != expected_fields:
            raise ValueError(f"evidence fields are invalid at line {line_number}")
        values = dict(payload)
        context = values.get("context_signature")
        if not isinstance(context, list) or not all(
            isinstance(item, str) and item for item in context
        ):
            raise ValueError(f"evidence context is invalid at line {line_number}")
        values["context_signature"] = tuple(context)
        item = CausalEvidence(**values)
        item.validate()
        canonical = asdict(item)
        revision = canonical_hash(canonical)
        if revision in revisions:
            raise ValueError(f"duplicate evidence revision: {revision}")
        revisions.add(revision)
        payloads.append(canonical)
        evidence.append(item)
    if not evidence:
        raise ValueError("causal-world evidence artifact is empty")
    return tuple(evidence), canonical_hash(payloads)


def _load_graph(
    path: Path,
) -> tuple[tuple[CausalLaw, ...], dict[str, Any]]:
    payload = _strict_json(Path(path).read_text(encoding="utf-8"))
    expected = {"architecture", "graph", "model_hash", "runtimes", "schema"}
    if set(payload) != expected:
        raise ValueError("causal graph model fields are invalid")
    core = {key: payload[key] for key in sorted(expected - {"model_hash"})}
    if payload["model_hash"] != canonical_hash(core):
        raise ValueError("causal graph model hash mismatch")
    if payload["schema"] != CAUSAL_GRAPH_MODEL_SCHEMA:
        raise ValueError("unsupported causal graph model schema")
    if payload["architecture"] != "pure-executable-causal-phase-hypergraph":
        raise ValueError("unsupported causal graph architecture")
    graph_payload = payload["graph"]
    if not isinstance(graph_payload, Mapping):
        raise ValueError("causal graph payload must be an object")
    if graph_payload.get("runtime") not in {
        "atom-executable-causal-graph-v2",
        CAUSAL_GRAPH_RUNTIME,
    }:
        raise ValueError("causal graph runtime marker is invalid")
    law_payloads = graph_payload.get("laws")
    if not isinstance(law_payloads, list):
        raise ValueError("causal graph laws must be a list")
    laws: list[CausalLaw] = []
    identities: set[str] = set()
    for law_payload in law_payloads:
        if not isinstance(law_payload, Mapping):
            raise ValueError("causal law payload must be an object")
        values = dict(law_payload)
        values.pop("sample_variance", None)
        values["atom_program"] = tuple(values["atom_program"])
        values["source_law_ids"] = tuple(values["source_law_ids"])
        values["contexts"] = {
            str(key): int(value)
            for key, value in values["contexts"].items()
        }
        values["provenance_hashes"] = list(values["provenance_hashes"])
        values["evidence_ids"] = list(values["evidence_ids"])
        law = CausalLaw(**values)
        _validate_law(law)
        if law.law_id in identities:
            raise ValueError(f"duplicate causal law: {law.law_id}")
        identities.add(law.law_id)
        laws.append(law)
    if len(laws) != int(graph_payload.get("law_count", -1)):
        raise ValueError("causal graph law count mismatch")
    return tuple(sorted(laws, key=lambda item: item.law_id)), payload


def _validate_law(law: CausalLaw) -> None:
    if not law.law_id:
        raise ValueError("causal law identity is empty")
    if law.domain not in DOMAIN_NAMES:
        raise ValueError(f"causal law domain is invalid: {law.domain}")
    if law.cause_feature not in FEATURE_INDEX:
        raise ValueError("causal law cause feature is invalid")
    if law.effect_feature not in FEATURE_INDEX:
        raise ValueError("causal law effect feature is invalid")
    if law.direction not in {-1, 1}:
        raise ValueError("causal law direction is invalid")
    if law.status not in {"hypothesis", "crystallized", "retired"}:
        raise ValueError("causal law lifecycle status is invalid")
    if (
        law.support < 0
        or law.contradictions < 0
        or law.treated_worlds <= 0
        or law.baseline_worlds <= 0
    ):
        raise ValueError("causal law counts are invalid")
    nonnegative = (
        law.magnitude_mean,
        law.magnitude_m2,
        law.effect_variance_mean,
        law.delay_mean,
        law.invariant_error_mean,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
        raise ValueError("causal law nonnegative metrics are invalid")
    if not math.isfinite(law.phase):
        raise ValueError("causal law phase is invalid")
    if not 0.0 <= law.confidence <= 1.0:
        raise ValueError("causal law confidence is invalid")
    if not 0.0 <= law.persistence <= 1.0:
        raise ValueError("causal law persistence is invalid")
    if not law.atom_program or not set(law.atom_program) <= ROOT_PRIMITIVES:
        raise ValueError("causal law root program is invalid")
    if not all(
        isinstance(key, str) and key and isinstance(value, int) and value > 0
        for key, value in law.contexts.items()
    ):
        raise ValueError("causal law contexts are invalid")
    if not all(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in law.provenance_hashes
    ):
        raise ValueError("causal law provenance hashes are invalid")
    if not all(isinstance(value, str) and value for value in law.evidence_ids):
        raise ValueError("causal law evidence identities are invalid")


def _observation_record(
    evidence: CausalEvidence,
    laws: Sequence[CausalLaw],
) -> ExperienceRecord:
    revision = canonical_hash(asdict(evidence))
    values: list[tuple[str, str]] = [
        ("kind", "observation"),
        ("status", "observed"),
        ("domain", evidence.domain),
        ("cause", evidence.cause_feature),
        ("effect", evidence.effect_feature),
        ("direction", _direction(evidence.direction)),
        ("delay", f"ticks:{evidence.delay}"),
        ("magnitude", _log_bucket(evidence.magnitude)),
        ("invariant", _log_bucket(evidence.invariant_error)),
        (
            "support",
            _count_bucket(evidence.treated_worlds + evidence.baseline_worlds),
        ),
        ("source/id", evidence.evidence_id),
        ("provenance/hash", evidence.provenance_hash),
    ]
    values.extend(("context", item) for item in evidence.context_signature)
    roots = sorted(
        {
            root
            for law in laws
            for root in law.atom_program
            if root in ROOT_PRIMITIVES
        }
    )
    values.extend(
        (f"root/{index:04d}", root)
        for index, root in enumerate(roots)
    )
    values.extend(
        ("law/ref", _experience_id("law", law.law_id))
        for law in sorted(laws, key=lambda item: item.law_id)
    )
    return ExperienceRecord(
        experience_id=_experience_id(
            "observation",
            f"{evidence.evidence_id}:{revision[:24]}",
        ),
        features=_features(values),
    )


def _law_record(
    law: CausalLaw,
    evidence_experiences: Sequence[str],
) -> ExperienceRecord:
    _validate_law(law)
    values: list[tuple[str, str]] = [
        ("kind", "law"),
        ("status", law.status),
        ("domain", law.domain),
        ("cause", law.cause_feature),
        ("effect", law.effect_feature),
        ("direction", _direction(law.direction)),
        ("delay", f"ticks:{round(law.delay_mean)}"),
        ("magnitude", _log_bucket(abs(law.magnitude_mean))),
        ("invariant", _log_bucket(law.invariant_error_mean)),
        ("support", _count_bucket(law.support)),
        ("confidence", _confidence_bucket(law.confidence)),
        ("contradiction", _count_bucket(law.contradictions)),
        ("source/id", law.law_id),
    ]
    values.extend(("context", item) for item in sorted(law.contexts))
    values.extend(
        (f"root/{index:04d}", root)
        for index, root in enumerate(law.atom_program)
    )
    values.extend(
        (f"evidence/{index:04d}", experience_id)
        for index, experience_id in enumerate(sorted(evidence_experiences))
    )
    values.extend(
        ("provenance/hash", provenance)
        for provenance in sorted(law.provenance_hashes)
    )
    values.extend(
        ("source/law", source)
        for source in sorted(law.source_law_ids)
    )
    return ExperienceRecord(
        experience_id=_experience_id("law", law.law_id),
        features=_features(values),
    )


def load_experience_corpus(
    evidence_path: Path,
    model_path: Path,
) -> ExperienceCorpus:
    evidence, evidence_hash = _load_evidence(Path(evidence_path))
    graph_laws, model_payload = _load_graph(Path(model_path))
    by_evidence: dict[str, list[CausalLaw]] = {}
    for law in graph_laws:
        for evidence_id in law.evidence_ids:
            by_evidence.setdefault(evidence_id, []).append(law)
    observations = tuple(
        _observation_record(item, by_evidence.get(item.evidence_id, ()))
        for item in evidence
    )
    observation_ids: dict[str, list[str]] = {}
    for item, record in zip(evidence, observations, strict=True):
        observation_ids.setdefault(item.evidence_id, []).append(
            record.experience_id
        )
    ordered_laws = graph_laws
    laws = tuple(
        _law_record(
            law,
            tuple(
                identity
                for evidence_id in law.evidence_ids
                for identity in observation_ids.get(evidence_id, ())
            ),
        )
        for law in ordered_laws
    )
    identities = [record.experience_id for record in observations + laws]
    if len(identities) != len(set(identities)):
        raise ValueError("experience corpus identities are not unique")
    if model_payload["model_hash"] != canonical_hash(
        {
            key: model_payload[key]
            for key in sorted(model_payload)
            if key != "model_hash"
        }
    ):
        raise ValueError("causal model hash binding was lost")
    return ExperienceCorpus(
        evidence_hash=evidence_hash,
        model_hash=str(model_payload["model_hash"]),
        observations=observations,
        laws=laws,
        law_payloads=tuple(law.manifest() for law in ordered_laws),
    )


def build_experience_batch(
    records: Sequence[ExperienceRecord],
    *,
    source_artifact_hash: str,
    batch_id: str,
) -> str:
    if (
        len(source_artifact_hash) != 64
        or not all(character in "0123456789abcdef" for character in source_artifact_hash)
    ):
        raise ValueError("source artifact hash must be lowercase SHA-256")
    if not records or not batch_id:
        raise ValueError("experience batch inputs are empty")
    lines = [
        EXPERIENCE_BATCH_RUNTIME,
        f"B\t{source_artifact_hash}\t{_hex(batch_id)}",
    ]
    identities: set[str] = set()
    for record in sorted(records, key=lambda item: item.experience_id):
        if record.experience_id in identities:
            raise ValueError(f"duplicate experience: {record.experience_id}")
        identities.add(record.experience_id)
        lines.append(f"E\t{_hex(record.experience_id)}")
        for role, value in record.features:
            lines.append(f"F\t{_hex(role)}\t{_hex(value)}")
    return "\n".join(lines) + "\n"


def _queryable_role(role: str) -> bool:
    return (
        role in QUERYABLE_ROLES
        or role.startswith("root/")
        or role.startswith("evidence/")
    )


def build_experience_query(
    *,
    query_id: str,
    features: Sequence[tuple[str, str, bool]],
    minimum_support: int,
    minimum_coverage_per_million: int = 650_000,
    limit: int = 32,
    excluded_experiences: Sequence[str] = (),
) -> str:
    if not features:
        raise ValueError("experience query needs structural features")
    normalized = sorted(
        {(str(role), str(value), bool(required)) for role, value, required in features}
    )
    if len({(role, value) for role, value, _ in normalized}) != len(normalized):
        raise ValueError("experience query repeats a feature with conflicting flags")
    if not all(_queryable_role(role) for role, _, _ in normalized):
        raise ValueError("experience query contains audit-only roles")
    if not 1 <= minimum_support <= len(normalized):
        raise ValueError("experience query minimum support is invalid")
    if not 0 <= minimum_coverage_per_million <= 1_000_000:
        raise ValueError("experience query coverage is invalid")
    if not 1 <= limit <= 1024:
        raise ValueError("experience query limit is invalid")
    lines = [
        EXPERIENCE_QUERY_RUNTIME,
        (
            f"Q\t{_hex(query_id)}\t{minimum_support}\t{limit}\t"
            f"{minimum_coverage_per_million}"
        ),
    ]
    lines.extend(
        f"F\t{_hex(role)}\t{_hex(value)}\t{int(required)}"
        for role, value, required in normalized
    )
    lines.extend(f"X\t{_hex(identity)}" for identity in sorted(set(excluded_experiences)))
    return "\n".join(lines) + "\n"


def build_query_for_record(
    record: ExperienceRecord,
    *,
    required_roles: Sequence[str] = (
        "kind",
        "domain",
        "cause",
        "effect",
        "direction",
    ),
    include_roles: Sequence[str] = (
        "kind",
        "status",
        "domain",
        "cause",
        "effect",
        "direction",
        "root",
        "context",
        "delay",
    ),
    limit: int = 32,
) -> str:
    required = set(required_roles)
    features = [
        (
            role,
            value,
            role in required or role.split("/", 1)[0] in required,
        )
        for role, value in record.features
        if role in include_roles or role.split("/", 1)[0] in include_roles
    ]
    return build_experience_query(
        query_id=f"experience-recall:{record.experience_id}",
        features=features,
        minimum_support=min(6, len(features)),
        minimum_coverage_per_million=650_000,
        limit=limit,
    )


@dataclass(frozen=True)
class ExperienceMemoryClient:
    store_path: Path
    binary: Path = RELEASE_BINARY

    def __post_init__(self) -> None:
        if not Path(self.binary).is_file():
            raise ValueError(f"causal-memory binary is absent: {self.binary}")
        if Path(self.store_path).is_dir():
            raise ValueError("experience store path cannot be a directory")

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
                f"Rust experience-memory {command} failed: {detail}"
            )
        payload = _strict_json(completed.stdout)
        if payload.get("runtime") != EXPERIENCE_MEMORY_RUNTIME:
            raise ValueError("Rust experience-memory runtime marker is invalid")
        return payload

    def ingest(
        self,
        records: Sequence[ExperienceRecord],
        *,
        source_artifact_hash: str,
        batch_id: str,
    ) -> dict[str, Any]:
        return self._run(
            "ingest-experiences",
            wire=build_experience_batch(
                records,
                source_artifact_hash=source_artifact_hash,
                batch_id=batch_id,
            ),
        )

    def inventory(self) -> dict[str, Any]:
        return self._run("experience-inventory")

    def recall(self, query_wire: str) -> dict[str, Any]:
        return self._run("recall-experiences", wire=query_wire)

    def observe_outcome(
        self,
        query_wire: str,
        *,
        expected_experience: str,
        selected_experience: str,
    ) -> dict[str, Any]:
        return self._run(
            "observe-experience",
            expected_experience,
            selected_experience,
            wire=query_wire,
        )

    def observe_outcome_once(
        self,
        query_wire: str,
        *,
        outcome_key: str,
        expected_experience: str,
        selected_experience: str,
    ) -> dict[str, Any]:
        if (
            len(outcome_key) != 64
            or any(character not in "0123456789abcdef" for character in outcome_key)
        ):
            raise ValueError("outcome key must be lowercase SHA-256")
        return self._run(
            "observe-experience-once",
            outcome_key,
            expected_experience,
            selected_experience,
            wire=query_wire,
        )


def binary_sha256(binary: Path = RELEASE_BINARY) -> str:
    return hashlib.sha256(Path(binary).read_bytes()).hexdigest()
