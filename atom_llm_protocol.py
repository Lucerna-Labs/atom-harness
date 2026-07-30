"""Strict language-boundary contracts for the Atom harness.

The LLM is allowed to translate natural language into a typed query and to
render retrieved evidence. It is not allowed to create evidence, mutate Atom
memory, choose tools, or relax an abstention decision.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from atom_causal_experience import build_experience_query
from atom_causal_world_schema import canonical_hash


ATOM_LANGUAGE_INTENT_RUNTIME = "atom-language-intent-v2"
ATOM_GROUNDED_RESPONSE_RUNTIME = "atom-grounded-response-v2"
ATOM_LANGUAGE_MODEL_PROTOCOL = "atom-json-language-model-v2"
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


class ProviderLocation(str, Enum):
    """Explicit data-boundary location used by the privacy gate."""

    LOCAL = "local"
    PRIVATE = "private"
    CLOUD = "cloud"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Admission facts declared by one provider adapter."""

    provider_id: str
    model: str
    location: ProviderLocation
    strict_json_schema: bool
    max_context_tokens: int
    max_output_tokens: int
    supports_cancellation: bool
    cost_tier: str
    test_only: bool = False

    def __post_init__(self) -> None:
        for field_name in ("provider_id", "model", "cost_tier"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or len(value) > 512
            ):
                raise ValueError(f"provider capability {field_name} is invalid")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(self, "location", ProviderLocation(self.location))
        for field_name in ("strict_json_schema", "supports_cancellation", "test_only"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"provider capability {field_name} must be boolean")
        for field_name in ("max_context_tokens", "max_output_tokens"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 10_000_000
            ):
                raise ValueError(f"provider capability {field_name} is invalid")
        if self.max_output_tokens > self.max_context_tokens:
            raise ValueError("provider output capability exceeds its context limit")

    def manifest(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "location": self.location.value,
            "strict_json_schema": self.strict_json_schema,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "supports_cancellation": self.supports_cancellation,
            "cost_tier": self.cost_tier,
            "test_only": self.test_only,
        }


class ProviderError(RuntimeError):
    """Typed provider failure safe to expose in routing evidence."""

    failure_kind = "provider"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        route: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.route = dict(route or {})


class ProviderAdmissionError(ProviderError):
    failure_kind = "admission"


class ProviderPrivacyError(ProviderError):
    failure_kind = "privacy"


class ProviderTransportError(ProviderError):
    failure_kind = "transport"
    retryable = True


class ProviderTimeoutError(ProviderError):
    failure_kind = "timeout"
    retryable = True


class ProviderCapacityError(ProviderError):
    failure_kind = "capacity"
    retryable = True


class ProviderCancelledError(ProviderError):
    failure_kind = "cancelled"


class ProviderInternalError(ProviderError):
    failure_kind = "internal"


class ProviderExhaustedError(ProviderError):
    failure_kind = "exhausted"


class CancellationToken:
    """Thread-safe cooperative cancellation shared across the full request."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "request cancelled"
        self._lock = threading.Lock()

    def cancel(self, reason: str = "request cancelled") -> None:
        normalized = str(reason).strip() or "request cancelled"
        with self._lock:
            self._reason = normalized[:512]
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProviderCancelledError(self.reason)

    def wait(self, timeout_seconds: float) -> None:
        if timeout_seconds < 0:
            raise ValueError("cancellation wait must not be negative")
        if self._event.wait(timeout_seconds):
            raise ProviderCancelledError(self.reason)


def _validate_json_shape(value: Any, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("JSON data exceeds the maximum nesting depth")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _validate_json_shape(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_shape(item, depth=depth + 1)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise ValueError("JSON data contains an unsupported value")


def _json_mapping(
    value: Any,
    label: str,
    *,
    maximum_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        _validate_json_shape(value)
    except (RecursionError, ValueError) as error:
        raise ValueError(f"{label} has an invalid JSON shape") from error
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must contain finite JSON data") from error
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{label} exceeds the safe byte limit")
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


@dataclass(frozen=True)
class JsonGenerationRequest:
    """One constrained JSON request sent across the language membrane."""

    stage: str
    system_prompt: str
    payload: Mapping[str, Any]
    schema: Mapping[str, Any]
    max_tokens: int
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )
    data_sensitivity: str = "private-atom-evidence"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage, str)
            or not self.stage.strip()
            or "\x00" in self.stage
            or len(self.stage) > 128
        ):
            raise ValueError("language request stage is invalid")
        object.__setattr__(self, "stage", self.stage.strip())
        if (
            not isinstance(self.system_prompt, str)
            or not self.system_prompt.strip()
            or "\x00" in self.system_prompt
            or len(self.system_prompt) > 32_768
        ):
            raise ValueError("language request system prompt is invalid")
        object.__setattr__(self, "system_prompt", self.system_prompt.strip())
        object.__setattr__(
            self,
            "payload",
            _json_mapping(self.payload, "language request payload"),
        )
        object.__setattr__(
            self,
            "schema",
            _json_mapping(
                self.schema,
                "language request schema",
                maximum_bytes=1024 * 1024,
            ),
        )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 65_536
        ):
            raise ValueError("language request token limit is invalid")
        if self.validator is not None and not callable(self.validator):
            raise ValueError("language request validator is not callable")
        if self.data_sensitivity != "private-atom-evidence":
            raise ValueError("language request data sensitivity is invalid")


@dataclass(frozen=True)
class JsonGenerationResult:
    """Validated transport metadata plus an untrusted JSON object."""

    payload: Mapping[str, Any]
    provider: str
    model: str
    elapsed_ms: int
    raw_sha256: str
    route: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload",
            _json_mapping(self.payload, "language result payload"),
        )
        for field_name in ("provider", "model"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or len(value) > 512
            ):
                raise ValueError(f"language result {field_name} is invalid")
            object.__setattr__(self, field_name, value.strip())
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise ValueError("language result elapsed time is invalid")
        if (
            not isinstance(self.raw_sha256, str)
            or len(self.raw_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.raw_sha256)
        ):
            raise ValueError("language result raw hash is invalid")
        object.__setattr__(
            self,
            "route",
            _json_mapping(self.route, "language result route"),
        )


class JsonLanguageModel(Protocol):
    """Replaceable provider contract used by the harness."""

    def generate_json(
        self,
        request: JsonGenerationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> JsonGenerationResult:
        """Return one JSON object constrained by ``request.schema``."""

    def capabilities(self) -> ProviderCapabilities:
        """Return explicit admission and privacy capabilities."""

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
    if type(payload["schema"]) is not int or payload["schema"] != 1:
        raise LanguageBoundaryError("language intent schema is invalid")
    if payload["runtime"] != ATOM_LANGUAGE_INTENT_RUNTIME:
        raise LanguageBoundaryError("language intent runtime is invalid")
    action = payload["action"]
    if not isinstance(action, str) or action not in {"retrieve", "abstain"}:
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
    if type(payload["schema"]) is not int or payload["schema"] != 1:
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
