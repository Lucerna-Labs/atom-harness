"""Structured language boundary for multidisciplinary Atom evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from atom_llm_protocol import LanguageBoundaryError


ATOM_MULTIDISCIPLINARY_RESPONSE_RUNTIME = "atom-multidisciplinary-grounded-response-v1"
MULTIDISCIPLINARY_GROUNDING_FIELDS = (
    "source_claim_id",
    "domain_id",
    "claim_type",
    "epistemic_status",
    "statement_sha256",
)

MULTIDISCIPLINARY_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "runtime",
        "answerable",
        "answer",
        "citations",
        "limitations",
        "grounding",
    ],
    "properties": {
        "schema": {"type": "integer", "enum": [1]},
        "runtime": {
            "type": "string",
            "enum": [ATOM_MULTIDISCIPLINARY_RESPONSE_RUNTIME],
        },
        "answerable": {"type": "boolean"},
        "answer": {"type": "string", "minLength": 1, "maxLength": 1024},
        "citations": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "limitations": {"type": "string", "minLength": 1, "maxLength": 1024},
        "grounding": {
            "type": "object",
            "additionalProperties": False,
            "required": list(MULTIDISCIPLINARY_GROUNDING_FIELDS),
            "properties": {
                field: {"type": "string", "minLength": 1, "maxLength": 256}
                for field in MULTIDISCIPLINARY_GROUNDING_FIELDS
            },
        },
    },
}


def _text(label: str, value: Any, *, maximum: int) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise LanguageBoundaryError(f"{label} must be NUL-free text")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise LanguageBoundaryError(f"{label} length is invalid")
    return normalized


def multidisciplinary_response_schema(
    evidence_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the model's grounding object to Atom's exact selected claim."""

    primary = evidence_packet.get("primary_claim")
    if not isinstance(primary, Mapping) or set(primary) != set(
        MULTIDISCIPLINARY_GROUNDING_FIELDS
    ):
        raise LanguageBoundaryError("multidisciplinary primary claim is invalid")
    schema = deepcopy(MULTIDISCIPLINARY_RESPONSE_JSON_SCHEMA)
    allowed_claim_ids = sorted(
        {
            _text("packet claim ID", item["claim_id"], maximum=128)
            for item in evidence_packet.get("passages", [])
            if isinstance(item, Mapping) and "claim_id" in item
        }
    )
    if not allowed_claim_ids:
        raise LanguageBoundaryError("multidisciplinary packet has no claim IDs")
    schema["properties"]["citations"]["minItems"] = 1
    schema["properties"]["citations"]["items"]["enum"] = allowed_claim_ids
    properties = schema["properties"]["grounding"]["properties"]
    for field in MULTIDISCIPLINARY_GROUNDING_FIELDS:
        properties[field]["enum"] = [
            _text(f"primary claim {field}", primary[field], maximum=256)
        ]
    return schema


def validate_multidisciplinary_response(
    payload: Mapping[str, Any],
    *,
    evidence_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject answers that are not bound to packet-local claims."""

    required = {
        "schema",
        "runtime",
        "answerable",
        "answer",
        "citations",
        "limitations",
        "grounding",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise LanguageBoundaryError("multidisciplinary response fields are invalid")
    if payload.get("schema") != 1 or payload.get("runtime") != (
        ATOM_MULTIDISCIPLINARY_RESPONSE_RUNTIME
    ):
        raise LanguageBoundaryError("multidisciplinary response identity is invalid")
    if not isinstance(payload.get("answerable"), bool):
        raise LanguageBoundaryError("multidisciplinary answerable must be Boolean")
    answer = _text("multidisciplinary answer", payload["answer"], maximum=1024)
    limitations = _text(
        "multidisciplinary limitations", payload["limitations"], maximum=1024
    )
    raw_citations = payload["citations"]
    if not isinstance(raw_citations, list) or len(raw_citations) > 8:
        raise LanguageBoundaryError("multidisciplinary citations are invalid")
    citations = [
        _text("multidisciplinary citation", item, maximum=128) for item in raw_citations
    ]
    if len(citations) != len(set(citations)):
        raise LanguageBoundaryError("multidisciplinary citations contain duplicates")
    allowed = {
        str(item["claim_id"])
        for item in evidence_packet.get("passages", [])
        if isinstance(item, Mapping) and "claim_id" in item
    }
    if not set(citations) <= allowed:
        raise LanguageBoundaryError(
            "multidisciplinary response cites outside the evidence packet"
        )
    packet_answerable = evidence_packet.get("answerable") is True and bool(allowed)
    if payload["answerable"] is not packet_answerable:
        raise LanguageBoundaryError(
            "multidisciplinary response contradicts Atom evidence availability"
        )
    grounding: dict[str, str] | None = None
    if packet_answerable:
        primary = evidence_packet.get("primary_claim")
        raw_grounding = payload.get("grounding")
        if not isinstance(primary, Mapping) or set(primary) != set(
            MULTIDISCIPLINARY_GROUNDING_FIELDS
        ):
            raise LanguageBoundaryError("multidisciplinary primary claim is invalid")
        if not isinstance(raw_grounding, Mapping) or set(raw_grounding) != set(
            MULTIDISCIPLINARY_GROUNDING_FIELDS
        ):
            raise LanguageBoundaryError("multidisciplinary grounding is invalid")
        grounding = {
            field: _text(
                f"multidisciplinary grounding {field}",
                raw_grounding[field],
                maximum=256,
            )
            for field in MULTIDISCIPLINARY_GROUNDING_FIELDS
        }
        expected = {
            field: _text(
                f"multidisciplinary primary {field}", primary[field], maximum=256
            )
            for field in MULTIDISCIPLINARY_GROUNDING_FIELDS
        }
        if grounding != expected:
            raise LanguageBoundaryError(
                "multidisciplinary response grounding changed Atom's claim"
            )
        if grounding["source_claim_id"] not in citations:
            raise LanguageBoundaryError(
                "multidisciplinary response omits its primary citation"
            )
    else:
        if citations or payload.get("grounding") is not None:
            raise LanguageBoundaryError(
                "multidisciplinary abstention carries evidence claims"
            )
    return {
        "schema": 1,
        "runtime": ATOM_MULTIDISCIPLINARY_RESPONSE_RUNTIME,
        "answerable": packet_answerable,
        "answer": answer,
        "citations": citations,
        "limitations": limitations,
        "grounding": grounding,
    }


def deterministic_multidisciplinary_abstention(reason: str) -> dict[str, Any]:
    """Return a fail-closed response in the multidisciplinary protocol."""

    limitation = _text("multidisciplinary abstention", reason, maximum=1024)
    return {
        "schema": 1,
        "runtime": ATOM_MULTIDISCIPLINARY_RESPONSE_RUNTIME,
        "answerable": False,
        "answer": "Atom does not have sufficient verified multidisciplinary evidence to answer.",
        "citations": [],
        "limitations": limitation,
        "grounding": None,
    }
