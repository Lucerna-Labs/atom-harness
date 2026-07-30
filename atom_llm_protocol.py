"""Strict language-boundary contracts for the Atom harness.

The LLM is allowed to translate natural language into a typed query and to
render retrieved evidence. It is not allowed to create evidence, mutate Atom
memory, choose tools, or relax an abstention decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from atom_causal_experience import build_experience_query
from atom_causal_world_schema import canonical_hash


ATOM_LANGUAGE_INTENT_RUNTIME = "atom-language-intent-v1"
ATOM_GROUNDED_RESPONSE_RUNTIME = "atom-grounded-response-v1"
ATOM_LANGUAGE_MODEL_PROTOCOL = "atom-json-language-model-v1"
ATOM_ABSTENTION = "I do not have enough Atom evidence to answer that."

QUERY_ROLES = frozenset(
    {
        "kind",
        "status",
        "domain",
        "cause",
        "effect",
        "direction",
        "context",
    }
)

INTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "runtime",
        "action",
        "question",
        "features",
    ],
    "properties": {
        "schema": {"type": "integer", "enum": [1]},
        "runtime": {
            "type": "string",
            "enum": [ATOM_LANGUAGE_INTENT_RUNTIME],
        },
        "action": {
            "type": "string",
            "enum": ["retrieve", "abstain"],
        },
        "question": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1024,
        },
        "features": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "value", "required"],
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": sorted(QUERY_ROLES),
                    },
                    "value": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                    },
                    "required": {"type": "boolean"},
                },
            },
        },
    },
}

GROUNDED_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "runtime",
        "answerable",
        "answer",
        "citations",
        "limitations",
    ],
    "properties": {
        "schema": {"type": "integer", "enum": [1]},
        "runtime": {
            "type": "string",
            "enum": [ATOM_GROUNDED_RESPONSE_RUNTIME],
        },
        "answerable": {"type": "boolean"},
        "answer": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4096,
        },
        "citations": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 512,
            },
        },
        "limitations": {
            "type": "string",
            "maxLength": 1024,
        },
    },
}


class LanguageBoundaryError(ValueError):
    """Raised when language-model output crosses an Atom trust boundary."""


@dataclass(frozen=True)
class JsonGenerationRequest:
    """One constrained JSON request sent across the language membrane."""

    stage: str
    system_prompt: str
    payload: Mapping[str, Any]
    schema: Mapping[str, Any]
    max_tokens: int


@dataclass(frozen=True)
class JsonGenerationResult:
    """Validated transport metadata plus an untrusted JSON object."""

    payload: Mapping[str, Any]
    provider: str
    model: str
    elapsed_ms: int
    raw_sha256: str


class JsonLanguageModel(Protocol):
    """Replaceable provider contract used by the harness."""

    def generate_json(
        self,
        request: JsonGenerationRequest,
    ) -> JsonGenerationResult:
        """Return one JSON object constrained by ``request.schema``."""

    def manifest(self) -> Mapping[str, Any]:
        """Return non-secret provider identity and capability metadata."""


def _strict_text(
    name: str,
    value: Any,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise LanguageBoundaryError(f"{name} must be NUL-free text")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise LanguageBoundaryError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise LanguageBoundaryError(f"{name} exceeds {maximum} characters")
    return normalized


def validate_intent(
    payload: Mapping[str, Any],
    *,
    vocabulary: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Validate that the LLM selected only graph-native Atom vocabulary."""

    expected = {
        "schema",
        "runtime",
        "action",
        "question",
        "features",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise LanguageBoundaryError("language intent fields are invalid")
    if payload["schema"] != 1:
        raise LanguageBoundaryError("language intent schema is invalid")
    if payload["runtime"] != ATOM_LANGUAGE_INTENT_RUNTIME:
        raise LanguageBoundaryError("language intent runtime is invalid")
    action = payload["action"]
    if action not in {"retrieve", "abstain"}:
        raise LanguageBoundaryError("language intent action is invalid")
    question = _strict_text(
        "language intent question",
        payload["question"],
        maximum=1024,
    )
    raw_features = payload["features"]
    if not isinstance(raw_features, list) or len(raw_features) > 8:
        raise LanguageBoundaryError("language intent features are invalid")

    allowed = {
        str(role): frozenset(str(value) for value in values)
        for role, values in vocabulary.items()
    }
    features: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for item in raw_features:
        if not isinstance(item, Mapping):
            raise LanguageBoundaryError("language intent feature is not an object")
        if set(item) != {"role", "value", "required"}:
            raise LanguageBoundaryError("language intent feature fields are invalid")
        role = _strict_text(
            "language intent role",
            item["role"],
            maximum=64,
        )
        value = _strict_text(
            "language intent value",
            item["value"],
            maximum=256,
        )
        required = item["required"]
        if role not in QUERY_ROLES:
            raise LanguageBoundaryError(
                f"language intent role is not queryable: {role}"
            )
        if type(required) is not bool:
            raise LanguageBoundaryError("language intent required flag must be boolean")
        if value not in allowed.get(role, frozenset()):
            raise LanguageBoundaryError(
                f"language intent value is absent from Atom wiki: {role}={value}"
            )
        identity = (role, value)
        if identity in identities:
            raise LanguageBoundaryError("language intent repeats a feature")
        identities.add(identity)
        features.append(
            {
                "role": role,
                "value": value,
                "required": required,
            }
        )

    if action == "abstain":
        if features:
            raise LanguageBoundaryError("abstaining intent must not carry features")
    else:
        if not features:
            raise LanguageBoundaryError("retrieval intent needs an Atom feature")
        single_roles = {
            "kind",
            "status",
            "domain",
            "cause",
            "effect",
            "direction",
        }
        counts = {
            role: sum(1 for item in features if item["role"] == role)
            for role in single_roles
        }
        repeated = sorted(role for role, count in counts.items() if count > 1)
        if repeated:
            raise LanguageBoundaryError(
                "language intent repeats single-valued roles: " + ", ".join(repeated)
            )

    return {
        "schema": 1,
        "runtime": ATOM_LANGUAGE_INTENT_RUNTIME,
        "action": action,
        "question": question,
        "features": features,
    }


def build_query_from_intent(
    intent: Mapping[str, Any],
    *,
    request_id: str,
) -> str:
    """Demote a validated language intent into the existing Atom wire query."""

    if intent.get("action") != "retrieve":
        raise LanguageBoundaryError("only retrieval intent can form a query")
    features = [
        (
            str(item["role"]),
            str(item["value"]),
            bool(item["required"]),
        )
        for item in intent["features"]
    ]
    query_identity = canonical_hash(
        {
            "request_id": request_id,
            "intent": dict(intent),
        }
    )[:24]
    return build_experience_query(
        query_id=f"language-harness:{query_identity}",
        features=features,
        minimum_support=max(1, min(4, len(features))),
        minimum_coverage_per_million=400_000,
        limit=12,
    )


def validate_grounded_response(
    payload: Mapping[str, Any],
    *,
    evidence_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject claims or citations that are not licensed by Atom evidence."""

    expected = {
        "schema",
        "runtime",
        "answerable",
        "answer",
        "citations",
        "limitations",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise LanguageBoundaryError("grounded response fields are invalid")
    if payload["schema"] != 1:
        raise LanguageBoundaryError("grounded response schema is invalid")
    if payload["runtime"] != ATOM_GROUNDED_RESPONSE_RUNTIME:
        raise LanguageBoundaryError("grounded response runtime is invalid")
    if type(payload["answerable"]) is not bool:
        raise LanguageBoundaryError("grounded answerable flag must be boolean")
    answer = _strict_text(
        "grounded answer",
        payload["answer"],
        maximum=4096,
    )
    limitations = _strict_text(
        "grounded limitations",
        payload["limitations"],
        maximum=1024,
        allow_empty=True,
    )
    raw_citations = payload["citations"]
    if not isinstance(raw_citations, list) or len(raw_citations) > 12:
        raise LanguageBoundaryError("grounded citations are invalid")
    citations = [
        _strict_text("grounded citation", item, maximum=512) for item in raw_citations
    ]
    if len(citations) != len(set(citations)):
        raise LanguageBoundaryError("grounded citations are duplicated")

    passages = evidence_packet.get("passages")
    if not isinstance(passages, list):
        raise LanguageBoundaryError("evidence packet passages are invalid")
    allowed_citations = {
        str(item["experience_id"])
        for item in passages
        if isinstance(item, Mapping) and "experience_id" in item
    }
    packet_answerable = evidence_packet.get("answerable") is True
    packet_insufficient = evidence_packet.get("insufficient_evidence") is True
    claims_answerable = payload["answerable"] is True

    if packet_insufficient or not packet_answerable:
        if claims_answerable or citations:
            raise LanguageBoundaryError(
                "LLM cannot override Atom insufficient-evidence abstention"
            )
    elif claims_answerable:
        if not citations:
            raise LanguageBoundaryError("answerable response needs an Atom citation")
        unknown = set(citations) - allowed_citations
        if unknown:
            raise LanguageBoundaryError(
                "grounded response cites evidence outside the packet: "
                + ", ".join(sorted(unknown))
            )
    elif citations:
        raise LanguageBoundaryError("abstaining response must not carry citations")

    return {
        "schema": 1,
        "runtime": ATOM_GROUNDED_RESPONSE_RUNTIME,
        "answerable": claims_answerable,
        "answer": answer,
        "citations": citations,
        "limitations": limitations,
    }


def deterministic_abstention(reason: str) -> dict[str, Any]:
    """Produce a fail-closed response without asking the LLM to improvise."""

    limitation = _strict_text(
        "abstention reason",
        reason,
        maximum=1024,
    )
    return {
        "schema": 1,
        "runtime": ATOM_GROUNDED_RESPONSE_RUNTIME,
        "answerable": False,
        "answer": ATOM_ABSTENTION,
        "citations": [],
        "limitations": limitation,
    }
