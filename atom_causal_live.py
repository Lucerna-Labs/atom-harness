"""Trusted live observation and outcome cycles for causal Atom memory."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from atom_causal_experience import (
    ExperienceCorpus,
    ExperienceMemoryClient,
    ExperienceRecord,
    build_experience_query,
)
from atom_causal_world_schema import DOMAIN_NAMES, canonical_hash

CAUSAL_LIVE_RUNTIME = "atom-causal-live-v1"
LIVE_EVENT_SCHEMA = 1
TRUSTED_AUTHORITY_KINDS = frozenset(
    {"instrument", "operator", "simulator", "verified_test"}
)
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


def _text(name: str, value: Any, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be non-empty bounded text")
    return value


def _sha256(name: str, value: Any) -> str:
    text = _text(name, value, 64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _finite_nonnegative(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _log_bucket(value: float) -> str:
    if value == 0.0:
        return "zero"
    return f"10^{math.floor(math.log10(value))}"


def _direction(value: int) -> str:
    if isinstance(value, bool) or value not in {-1, 0, 1}:
        raise ValueError("direction must be -1, 0, or 1")
    return f"{value:+d}" if value else "0"


def _features(
    values: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: set[tuple[str, str]] = set()
    for role, value in values:
        role = _text("feature role", role, 512)
        value = _text("feature value", value, 16 * 1024)
        normalized.add((role, value))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class TrustedOutcome:
    effect: str
    delay_ticks: int
    magnitude: float
    invariant_error: float
    authority_kind: str
    authority_id: str
    evidence_hash: str

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> TrustedOutcome:
        expected = {
            "effect",
            "delay_ticks",
            "magnitude",
            "invariant_error",
            "authority_kind",
            "authority_id",
            "evidence_hash",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("trusted outcome fields are invalid")
        delay = payload["delay_ticks"]
        if (
            isinstance(delay, bool)
            or not isinstance(delay, int)
            or not 0 <= delay <= 1_000_000_000
        ):
            raise ValueError("delay_ticks is outside the supported range")
        authority_kind = _text(
            "authority_kind", payload["authority_kind"], 128
        )
        if authority_kind not in TRUSTED_AUTHORITY_KINDS:
            raise ValueError("outcome authority kind is not trusted")
        return cls(
            effect=_text("effect", payload["effect"]),
            delay_ticks=delay,
            magnitude=_finite_nonnegative(
                "magnitude", payload["magnitude"]
            ),
            invariant_error=_finite_nonnegative(
                "invariant_error", payload["invariant_error"]
            ),
            authority_kind=authority_kind,
            authority_id=_text(
                "authority_id", payload["authority_id"], 1024
            ),
            evidence_hash=_sha256(
                "evidence_hash", payload["evidence_hash"]
            ),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "effect": self.effect,
            "delay_ticks": self.delay_ticks,
            "magnitude": self.magnitude,
            "invariant_error": self.invariant_error,
            "authority_kind": self.authority_kind,
            "authority_id": self.authority_id,
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True)
class LiveCausalEvent:
    session_id: str
    interaction_id: str
    domain: str
    cause: str
    context: tuple[str, ...]
    direction: int
    atom_program: tuple[str, ...]
    outcome: TrustedOutcome

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> LiveCausalEvent:
        expected = {
            "schema",
            "session_id",
            "interaction_id",
            "domain",
            "cause",
            "context",
            "direction",
            "atom_program",
            "outcome",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("live causal event fields are invalid")
        if payload["schema"] != LIVE_EVENT_SCHEMA:
            raise ValueError("live causal event schema is unsupported")
        domain = _text("domain", payload["domain"], 256)
        if domain not in DOMAIN_NAMES:
            raise ValueError("live event domain is outside the world schema")
        raw_context = payload["context"]
        if (
            not isinstance(raw_context, list)
            or not raw_context
            or len(raw_context) > 256
        ):
            raise ValueError("live event context is invalid")
        context = tuple(
            sorted({_text("context", item) for item in raw_context})
        )
        raw_program = payload["atom_program"]
        if (
            not isinstance(raw_program, list)
            or not raw_program
            or len(raw_program) > 256
        ):
            raise ValueError("live event atom program is invalid")
        atom_program = tuple(
            _text("atom_program", item, 128) for item in raw_program
        )
        if any(root not in ROOT_PRIMITIVES for root in atom_program):
            raise ValueError("live event uses an unknown universe primitive")
        return cls(
            session_id=_text(
                "session_id", payload["session_id"], 1024
            ),
            interaction_id=_text(
                "interaction_id", payload["interaction_id"], 1024
            ),
            domain=domain,
            cause=_text("cause", payload["cause"]),
            context=context,
            direction=int(_direction(payload["direction"])),
            atom_program=atom_program,
            outcome=TrustedOutcome.from_manifest(payload["outcome"]),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": LIVE_EVENT_SCHEMA,
            "session_id": self.session_id,
            "interaction_id": self.interaction_id,
            "domain": self.domain,
            "cause": self.cause,
            "context": list(self.context),
            "direction": self.direction,
            "atom_program": list(self.atom_program),
            "outcome": self.outcome.manifest(),
        }

    @property
    def event_hash(self) -> str:
        return canonical_hash(self.manifest())

    @property
    def experience_id(self) -> str:
        return f"experience:observation:live:{self.event_hash[:32]}"


def _inventory_features(
    item: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    raw = item.get("features")
    if not isinstance(raw, list):
        raise ValueError("experience inventory omits structural features")
    features: list[tuple[str, str]] = []
    for feature in raw:
        if not isinstance(feature, Mapping) or set(feature) != {
            "role",
            "value",
        }:
            raise ValueError("experience inventory feature is malformed")
        features.append(
            (
                _text("inventory feature role", feature["role"], 512),
                _text(
                    "inventory feature value",
                    feature["value"],
                    16 * 1024,
                ),
            )
        )
    normalized = _features(features)
    if len(normalized) != item.get("feature_count"):
        raise ValueError("experience inventory feature count is detached")
    return normalized


def _feature_values(
    item: Mapping[str, Any],
    role: str,
) -> tuple[str, ...]:
    return tuple(
        value
        for name, value in _inventory_features(item)
        if name == role
    )


def _single_feature(item: Mapping[str, Any], role: str) -> str:
    values = _feature_values(item, role)
    if len(values) != 1:
        raise ValueError(
            f"{item.get('experience_id')} has invalid {role} cardinality"
        )
    return values[0]


def _inventory_index(
    inventory: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = inventory.get("experiences")
    if not isinstance(records, list):
        raise ValueError("experience inventory records are invalid")
    index: dict[str, Mapping[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("experience inventory item is invalid")
        identity = _text(
            "experience_id", item.get("experience_id"), 1024
        )
        if identity in index:
            raise ValueError("experience inventory identity is duplicated")
        _inventory_features(item)
        index[identity] = item
    return index


def build_live_prediction_query(event: LiveCausalEvent) -> str:
    features: list[tuple[str, str, bool]] = [
        ("domain", event.domain, True),
        ("cause", event.cause, True),
        ("direction", _direction(event.direction), True),
    ]
    features.extend(("context", item, False) for item in event.context)
    features.extend(
        (f"root/{index:04d}", root, False)
        for index, root in enumerate(event.atom_program)
    )
    return build_experience_query(
        query_id=f"live-predict:{event.event_hash[:24]}",
        features=features,
        minimum_support=min(6, len(features)),
        minimum_coverage_per_million=500_000,
        limit=128,
    )


def build_live_outcome_query(event: LiveCausalEvent) -> str:
    features: list[tuple[str, str, bool]] = [
        ("domain", event.domain, True),
        ("cause", event.cause, True),
        ("direction", _direction(event.direction), True),
        ("effect", event.outcome.effect, False),
    ]
    features.extend(("context", item, False) for item in event.context)
    features.extend(
        (f"root/{index:04d}", root, False)
        for index, root in enumerate(event.atom_program)
    )
    return build_experience_query(
        query_id=f"live-outcome:{event.event_hash[:24]}",
        features=features,
        minimum_support=min(6, len(features)),
        minimum_coverage_per_million=400_000,
        limit=128,
    )


def _select_prediction(
    recall: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> str | None:
    if recall.get("answerable") is not True:
        return None
    records = _inventory_index(inventory)
    hits = recall.get("hits")
    if not isinstance(hits, list):
        raise ValueError("experience recall hits are invalid")
    for hit in hits:
        identity = _text(
            "recalled experience", hit.get("experience_id"), 1024
        )
        record = records.get(identity)
        if record is None:
            raise ValueError("recall returned an experience outside inventory")
        if _single_feature(record, "status") != "retired":
            return identity
    return None


def _observation_record(
    event: LiveCausalEvent,
    *,
    selected_experience: str | None,
    selected_effect: str | None,
    prediction_catalog: str,
) -> ExperienceRecord:
    values: list[tuple[str, str]] = [
        ("kind", "observation"),
        ("status", "observed"),
        ("domain", event.domain),
        ("cause", event.cause),
        ("effect", event.outcome.effect),
        ("direction", _direction(event.direction)),
        ("delay", f"ticks:{event.outcome.delay_ticks}"),
        ("magnitude", _log_bucket(event.outcome.magnitude)),
        ("invariant", _log_bucket(event.outcome.invariant_error)),
        ("support", "1-1"),
        (
            "source/id",
            f"live:{event.session_id}:{event.interaction_id}",
        ),
        ("provenance/hash", event.outcome.evidence_hash),
        ("session/id", event.session_id),
        ("interaction/id", event.interaction_id),
        ("live/event-hash", event.event_hash),
        ("authority/kind", event.outcome.authority_kind),
        ("authority/id", event.outcome.authority_id),
        ("prediction/catalog", prediction_catalog),
        (
            "prediction/state",
            "selected" if selected_experience else "abstained",
        ),
    ]
    values.extend(("context", item) for item in event.context)
    values.extend(
        (f"root/{index:04d}", root)
        for index, root in enumerate(event.atom_program)
    )
    if selected_experience is not None:
        values.append(("prediction/selected", selected_experience))
    if selected_effect is not None:
        values.append(("prediction/effect", selected_effect))
    return ExperienceRecord(
        experience_id=event.experience_id,
        features=_features(values),
    )


def extend_corpus_from_inventory(
    base: ExperienceCorpus,
    inventory: Mapping[str, Any],
) -> ExperienceCorpus:
    index = _inventory_index(inventory)
    base_records = {
        record.experience_id: record for record in base.all_records
    }
    for identity, record in base_records.items():
        stored = index.get(identity)
        if stored is None:
            raise ValueError("base causal experience disappeared from memory")
        if _inventory_features(stored) != record.features:
            raise ValueError("base causal experience structure changed")
    records = tuple(
        ExperienceRecord(
            experience_id=identity,
            features=_inventory_features(item),
        )
        for identity, item in sorted(index.items())
    )
    observations = tuple(
        record
        for record in records
        if record.feature_values("kind") == ("observation",)
    )
    laws = tuple(
        record
        for record in records
        if record.feature_values("kind") == ("law",)
    )
    if len(observations) + len(laws) != len(records):
        raise ValueError("inventory contains an unsupported experience kind")
    live_records = [
        {
            "experience_id": record.experience_id,
            "features": [list(feature) for feature in record.features],
        }
        for record in observations
        if record.experience_id not in base_records
    ]
    evidence_hash = canonical_hash(
        {
            "base_evidence_hash": base.evidence_hash,
            "live_records": live_records,
        }
    )
    return ExperienceCorpus(
        evidence_hash=evidence_hash,
        model_hash=base.model_hash,
        observations=observations,
        laws=laws,
        law_payloads=base.law_payloads,
    )


def validate_outcome_evidence(
    event: LiveCausalEvent,
    evidence: Mapping[str, Any],
    authorities: Mapping[str, str],
) -> None:
    expected = {
        "schema",
        "authority_kind",
        "authority_id",
        "session_id",
        "interaction_id",
        "domain",
        "cause",
        "effect",
        "direction",
        "trace",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected:
        raise ValueError("outcome evidence receipt fields are invalid")
    configured_kind = authorities.get(event.outcome.authority_id)
    if configured_kind is None:
        raise ValueError("outcome authority is not configured")
    if configured_kind != event.outcome.authority_kind:
        raise ValueError("outcome authority kind conflicts with policy")
    bindings = {
        "schema": 1,
        "authority_kind": event.outcome.authority_kind,
        "authority_id": event.outcome.authority_id,
        "session_id": event.session_id,
        "interaction_id": event.interaction_id,
        "domain": event.domain,
        "cause": event.cause,
        "effect": event.outcome.effect,
        "direction": event.direction,
    }
    for name, value in bindings.items():
        if evidence.get(name) != value:
            raise ValueError(
                f"outcome evidence {name} binding is invalid"
            )
    if evidence["trace"] is None:
        raise ValueError("outcome evidence trace is absent")
    try:
        receipt_hash = canonical_hash(dict(evidence))
    except (TypeError, ValueError) as error:
        raise ValueError("outcome evidence is not canonical") from error
    if receipt_hash != event.outcome.evidence_hash:
        raise ValueError("outcome evidence hash binding is invalid")


@dataclass(frozen=True)
class LiveCausalRuntime:
    client: ExperienceMemoryClient
    authorities: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.authorities, Mapping) or not self.authorities:
            raise ValueError("live runtime needs an authority policy")
        for authority_id, authority_kind in self.authorities.items():
            _text("configured authority id", authority_id, 1024)
            if authority_kind not in TRUSTED_AUTHORITY_KINDS:
                raise ValueError(
                    "configured authority kind is unsupported"
                )

    def observe(
        self,
        event: LiveCausalEvent,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        validate_outcome_evidence(
            event,
            evidence,
            self.authorities,
        )
        store_path = Path(self.client.store_path)
        before_store_hash = (
            hashlib.sha256(store_path.read_bytes()).hexdigest()
            if store_path.is_file()
            else None
        )
        inventory_before = self.client.inventory()
        prediction_query = build_live_prediction_query(event)
        prediction = self.client.recall(prediction_query)
        selected = _select_prediction(prediction, inventory_before)
        before_index = _inventory_index(inventory_before)
        selected_effect = (
            _single_feature(before_index[selected], "effect")
            if selected is not None
            else None
        )
        record = _observation_record(
            event,
            selected_experience=selected,
            selected_effect=selected_effect,
            prediction_catalog=str(
                inventory_before["catalog_identity"]
            ),
        )
        ingest = self.client.ingest(
            [record],
            source_artifact_hash=event.event_hash,
            batch_id=f"live-event:{event.event_hash}",
        )
        inventory_after_ingest = self.client.inventory()
        stored_index = _inventory_index(inventory_after_ingest)
        stored = stored_index.get(event.experience_id)
        if stored is None:
            raise ValueError("live observation did not enter durable memory")
        stored_selected_values = _feature_values(
            stored, "prediction/selected"
        )
        if len(stored_selected_values) > 1:
            raise ValueError("live prediction provenance is ambiguous")
        durable_selected = (
            stored_selected_values[0]
            if stored_selected_values
            else None
        )
        durable_selected_effect = (
            _single_feature(
                stored_index[durable_selected],
                "effect",
            )
            if durable_selected is not None
            else None
        )
        expected = None
        feedback = None
        outcome_query = build_live_outcome_query(event)
        if durable_selected is not None:
            expected = (
                durable_selected
                if durable_selected_effect == event.outcome.effect
                else event.experience_id
            )
            feedback = self.client.observe_outcome_once(
                outcome_query,
                outcome_key=event.event_hash,
                expected_experience=expected,
                selected_experience=durable_selected,
            )
        inventory_final = self.client.inventory()
        post_prediction = self.client.recall(prediction_query)
        after_store_hash = hashlib.sha256(
            store_path.read_bytes()
        ).hexdigest()
        core = {
            "schema": 1,
            "runtime": CAUSAL_LIVE_RUNTIME,
            "event": event.manifest(),
            "event_hash": event.event_hash,
            "evidence_hash": event.outcome.evidence_hash,
            "experience_id": event.experience_id,
            "store_hash_before": before_store_hash,
            "store_hash_after": after_store_hash,
            "prediction_query": prediction_query,
            "outcome_query": outcome_query,
            "prediction": prediction,
            "selected_experience": durable_selected,
            "selected_effect": durable_selected_effect,
            "observed_effect": event.outcome.effect,
            "expected_experience": expected,
            "prediction_correct": (
                durable_selected_effect == event.outcome.effect
                if durable_selected is not None
                else None
            ),
            "ingest": ingest,
            "feedback": feedback,
            "post_prediction": post_prediction,
            "catalog_identity": inventory_final[
                "catalog_identity"
            ],
            "snapshot_sequence": inventory_final[
                "snapshot_sequence"
            ],
            "experience_count": len(
                inventory_final["experiences"]
            ),
            "batch_count": len(inventory_final["batches"]),
            "replayed": (
                ingest["committed"] is False
                and (
                    feedback is None
                    or feedback["committed"] is False
                )
            ),
        }
        return {**core, "cycle_hash": canonical_hash(core)}
