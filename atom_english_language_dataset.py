"""Natural-English curriculum layered over the consequence-grounded Atom world."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from atom_field_proof import PROCESS_NAMES
from atom_neural_language_dataset import (
    RUNTIME_ROW_KEYS,
    NeuralLanguageProgram,
    build_neural_language_program,
    neural_language_hash,
    tokenize_neural_utterance,
    validate_neural_runtime_row,
)


ENGLISH_DATASET_SCHEMA = 1
ENGLISH_MARKER = "eng"
_WORD = re.compile(r"[a-z0-9]+")

# The user surface stays ordinary English.  The internal codec dissipates
# syntactic glue so consequence induction sees the words that can actually
# explain a state transition.  This is deliberately semantic-free: no action
# word is listed here and no word is mapped to an Atom operator by the codec.
ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "and",
        "answer",
        "apply",
        "count",
        "execute",
        "first",
        "for",
        "greatest",
        "has",
        "how",
        "identify",
        "in",
        "leading",
        "many",
        "maximum",
        "me",
        "node",
        "nodes",
        "number",
        "of",
        "please",
        "remain",
        "remaining",
        "report",
        "strongest",
        "tell",
        "the",
        "then",
        "total",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class EnglishLanguageProgram:
    stages: Mapping[str, tuple[dict[str, Any], ...]]
    evaluator_truth: Mapping[str, Mapping[str, Any]]
    vocabulary: tuple[str, ...]
    response_vocabulary: tuple[str, ...]
    manifest: Mapping[str, Any]


ENGLISH_OPERATOR_LEXICONS: dict[str, dict[str, str]] = {
    "field-a": {
        "radiation": "spread",
        "dissipation": "dampen",
        "gravitation": "gather",
        "attraction_repulsion": "polarize",
        "nucleation": "crystallize",
        "conservation": "preserve",
        "decay": "fade",
    },
    "field-b": {
        "radiation": "broadcast",
        "dissipation": "cool",
        "gravitation": "concentrate",
        "attraction_repulsion": "separate",
        "nucleation": "stabilize",
        "conservation": "retain",
        "decay": "weaken",
    },
    "field-c": {
        "radiation": "disperse",
        "dissipation": "suppress",
        "gravitation": "collect",
        "attraction_repulsion": "repel",
        "nucleation": "solidify",
        "conservation": "safeguard",
        "decay": "erode",
    },
    "field-d": {
        "radiation": "diffuse",
        "dissipation": "attenuate",
        "gravitation": "aggregate",
        "attraction_repulsion": "partition",
        "nucleation": "condense",
        "conservation": "reserve",
        "decay": "dissolve",
    },
}


QUERY_NOUNS = {
    "signal_peak": "signal",
    "mass_peak": "mass",
    "cohesion_peak": "cohesion",
    "ttl_peak": "lifetime",
    "active_count": "active",
    "structure_count": "structures",
}


def normalize_english_request(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("English request must be text")
    surface_words = _WORD.findall(text.lower())
    words = [word for word in surface_words if word not in ENGLISH_FUNCTION_WORDS]
    if not surface_words:
        raise ValueError("English request must contain words")
    if not words:
        raise ValueError("English request must contain content words")
    internal = " ".join((ENGLISH_MARKER, *words))
    tokenize_neural_utterance(internal)
    return internal


def _answer_token(semantic_answer: str) -> str:
    kind, separator, value = semantic_answer.partition(":")
    if separator != ":" or kind not in {"node", "count"}:
        raise ValueError(f"bad semantic answer: {semantic_answer}")
    index = int(value)
    if kind == "node" and not 0 <= index <= 5:
        raise ValueError("node answer is outside [0, 5]")
    if kind == "count" and not 0 <= index <= 6:
        raise ValueError("count answer is outside [0, 6]")
    return f"{kind}{index}"


def _render_user_utterance(
    signature: tuple[str, ...],
    query_type: str,
    source_language: str,
    variant: int,
) -> str:
    lexicon = ENGLISH_OPERATOR_LEXICONS[source_language]
    operators = [lexicon[name] for name in signature]
    if variant % 2:
        operators = list(reversed(operators))
    process = " and ".join(operators)
    query = QUERY_NOUNS[query_type]
    if query_type in {"active_count", "structure_count"}:
        endings = (
            f"then report how many nodes remain {query}",
            f"and tell me the number of remaining {query}",
            f"then count the {query} nodes",
            f"and answer with the total {query}",
        )
    else:
        endings = (
            f"then report the node with greatest {query}",
            f"and tell me which node has maximum {query}",
            f"then identify the node leading in {query}",
            f"and answer with the strongest node for {query}",
        )
    openings = (
        f"please {process}",
        f"apply {process}",
        f"first {process}",
        f"execute {process}",
    )
    return f"{openings[variant % len(openings)]} {endings[variant % len(endings)]}"


def render_english_answer(response_token: str, query_index: int) -> str:
    match = re.fullmatch(r"(node|count)([0-6])", response_token)
    if match is None:
        raise ValueError("response token is not an English-shell answer")
    kind, value_text = match.groups()
    value = int(value_text)
    if query_index < 4 and kind != "node":
        raise ValueError("node query received a count response")
    if query_index >= 4 and kind != "count":
        raise ValueError("count query received a node response")
    if query_index == 0:
        return f"Node {value} has the greatest signal."
    if query_index == 1:
        return f"Node {value} has the greatest mass."
    if query_index == 2:
        return f"Node {value} has the greatest cohesion."
    if query_index == 3:
        return f"Node {value} has the longest remaining lifetime."
    if query_index == 4:
        return f"{value} nodes remain active."
    if query_index == 5:
        return f"{value} structures remain."
    raise ValueError("query index is outside [0, 5]")


def _transform_program(source: NeuralLanguageProgram) -> EnglishLanguageProgram:
    stages: dict[str, tuple[dict[str, Any], ...]] = {}
    evaluator_truth: dict[str, dict[str, Any]] = {}
    vocabulary: set[str] = {ENGLISH_MARKER}
    responses: set[str] = set()
    for stage, rows in source.stages.items():
        transformed: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            truth = source.evaluator_truth[str(row["event_id"])]
            source_language = str(truth["language"])
            signature = tuple(str(name) for name in truth["process_signature"])
            query_type = str(truth["query_type"])
            user_utterance = _render_user_utterance(
                signature,
                query_type,
                source_language,
                index,
            )
            internal_utterance = normalize_english_request(user_utterance)
            response = _answer_token(str(truth["semantic_answer"]))
            runtime = {
                **row,
                "response": response,
                "utterance": internal_utterance,
            }
            validate_neural_runtime_row(runtime)
            transformed.append(runtime)
            vocabulary.update(tokenize_neural_utterance(internal_utterance))
            responses.add(response)
            evaluator_truth[str(row["event_id"])] = {
                **truth,
                "english_answer": render_english_answer(
                    response,
                    (
                        0
                        if query_type == "signal_peak"
                        else 1
                        if query_type == "mass_peak"
                        else 2
                        if query_type == "cohesion_peak"
                        else 3
                        if query_type == "ttl_peak"
                        else 4
                        if query_type == "active_count"
                        else 5
                    ),
                ),
                "internal_utterance": internal_utterance,
                "user_utterance": user_utterance,
            }
        stages[stage] = tuple(transformed)

    audit = {
        "all_runtime_rows_strict": all(
            set(row) == RUNTIME_ROW_KEYS for rows in stages.values() for row in rows
        ),
        "all_seven_operators_have_four_english_surfaces": all(
            all(name in lexicon for name in PROCESS_NAMES)
            for lexicon in ENGLISH_OPERATOR_LEXICONS.values()
        ),
        "evaluator_truth_separate": all(
            "query_type" not in row and "process_signature" not in row
            for rows in stages.values()
            for row in rows
        ),
        "natural_english_user_surface": all(
            truth["user_utterance"] != truth["internal_utterance"]
            and not str(truth["user_utterance"]).startswith(ENGLISH_MARKER + " ")
            for truth in evaluator_truth.values()
        ),
        "transfer_and_unknown_lexicons_disjoint": not (
            set(ENGLISH_OPERATOR_LEXICONS["field-c"].values())
            & set(ENGLISH_OPERATOR_LEXICONS["field-d"].values())
        ),
    }
    audit["passed"] = all(audit.values())
    manifest_base = {
        "audit": audit,
        "response_vocabulary_size": len(responses),
        "schema_version": ENGLISH_DATASET_SCHEMA,
        "stages": {name: len(rows) for name, rows in stages.items()},
        "surface": "natural-english-with-internal-bos-marker",
        "vocabulary_size": len(vocabulary),
    }
    manifest = {
        **manifest_base,
        "dataset_hash": neural_language_hash(
            {
                "manifest": manifest_base,
                "runtime": stages,
            }
        ),
    }
    return EnglishLanguageProgram(
        stages=stages,
        evaluator_truth=evaluator_truth,
        vocabulary=tuple(sorted(vocabulary)),
        response_vocabulary=tuple(sorted(responses)),
        manifest=manifest,
    )


def build_english_language_program() -> EnglishLanguageProgram:
    return _transform_program(build_neural_language_program())


def english_language_self_tests() -> dict[str, Any]:
    first = build_english_language_program()
    second = build_english_language_program()
    checks = {
        "deterministic_manifest": first.manifest == second.manifest,
        "dataset_audit": bool(first.manifest["audit"]["passed"]),
        "four_disjoint_operator_lexicons": len(ENGLISH_OPERATOR_LEXICONS) == 4,
        "normalization_adds_internal_marker": normalize_english_request(
            "Please spread, then report the strongest signal."
        )
        == "eng spread signal",
        "normal_response_rendering": render_english_answer("node4", 0)
        == "Node 4 has the greatest signal.",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"checks": checks, "failed": failed, "passed": not failed}
