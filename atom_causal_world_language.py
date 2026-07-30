"""Natural-language observation and query shell for the causal graph.

Language never supplies a causal answer.  It names observable variables and
contexts, while the graph decides whether an evidence-supported law exists.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from atom_causal_graph import CausalQuery, stable_condition_signature
from atom_causal_world_curriculum import (
    WORLD_PROGRAM_AXES,
    curriculum_programs,
    world_program_space_size,
)
from atom_causal_world_schema import (
    DOMAIN_NAMES,
    FEATURE_NAMES,
    ROOT_MECHANICS,
    CausalEvidence,
    canonical_hash,
)


CAUSAL_WORLD_LANGUAGE_RUNTIME = "atom-causal-world-language-v1"
_WORDS = re.compile(r"[a-z0-9_]+")

DOMAIN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "physical": ("physical", "mechanical", "material"),
    "chemical": ("chemical", "reaction", "molecular"),
    "biological": ("biological", "organism", "living"),
    "ecological": ("ecological", "ecosystem", "environmental"),
    "agent": ("agent", "goal", "decision"),
    "social": ("social", "group", "community"),
    "symbolic": ("symbolic", "logical", "mathematical"),
    "language": ("language", "linguistic", "communication"),
}

FEATURE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "existence": ("existence", "presence", "survival"),
    "energy": ("energy", "power"),
    "mass": ("mass", "weight"),
    "charge": ("charge", "electric charge"),
    "temperature": ("temperature", "heat", "warmth"),
    "pressure": ("pressure", "compression"),
    "cohesion": ("cohesion", "binding", "togetherness"),
    "integrity": ("integrity", "stability", "soundness"),
    "support": ("support", "backing"),
    "lifetime": ("lifetime", "duration", "longevity"),
    "resource": ("resource", "supply", "resources"),
    "health": ("health", "fitness"),
    "signal": ("signal", "message strength"),
    "trust": ("trust", "reliability"),
    "belief": ("belief", "expectation"),
    "goal": ("goal", "intention"),
    "ownership": ("ownership", "possession"),
    "value": ("value", "worth"),
    "position_x": ("horizontal position", "x position"),
    "position_y": ("vertical position", "y position"),
    "position_z": ("depth", "z position"),
    "velocity_x": ("horizontal velocity", "x velocity"),
    "velocity_y": ("vertical velocity", "y velocity"),
    "velocity_z": ("depth velocity", "z velocity"),
    "polarity": ("polarity", "orientation"),
    "phase": ("phase", "timing"),
    "activation": ("activation", "activity"),
    "structure": ("structure", "organization"),
    "uncertainty": ("uncertainty", "doubt"),
    "novelty": ("novelty", "newness"),
    "memory_strength": ("memory strength", "retention"),
    "language_alignment": ("language alignment", "semantic alignment"),
}

QUESTION_OPENINGS = (
    "Based on observed interventions",
    "Using the causal evidence",
    "From the world history",
    "According to repeated consequences",
    "After comparing treated and untreated worlds",
    "From the persistent causal record",
    "Using only supported relationships",
    "After resolving competing explanations",
)

EFFECT_QUESTIONS = (
    "what does changing {cause} affect in the {domain} context?",
    "which consequence follows when {cause} is changed in a {domain} world?",
    "what is the strongest supported effect of changing {cause} for {domain} systems?",
    "when {cause} changes, what follows in the {domain} domain?",
)

DIRECT_QUESTIONS = (
    "does changing {cause} affect {effect} in the {domain} context?",
    "what happens to {effect} after {cause} changes in a {domain} world?",
    "why did {effect} change after the intervention on {cause} in the {domain} domain?",
    "if {cause} were changed, how would {effect} respond in the {domain} context?",
)

_CONDITION_NOUNS: Mapping[str, str] = {
    "scale": "scale",
    "resources": "resources",
    "signal": "signals",
    "relations": "relations",
    "time": "timing",
    "topology": "topology",
    "phase_regime": "phase",
    "energy_regime": "energy regime",
    "boundary": "boundary",
}


def _condition_surface(key: str, value: str) -> str:
    surface_value = value.replace("_", " ")
    if key == "primary_root":
        return f"{surface_value} as the dominant root"
    if key == "secondary_root":
        return f"{surface_value} as the secondary root"
    try:
        noun = _CONDITION_NOUNS[key]
    except KeyError as error:
        raise ValueError(f"unknown world-condition key: {key}") from error
    return f"{surface_value} {noun}"


WORLD_CONDITION_ALIASES: Mapping[str, tuple[str, ...]] = {
    **{
        f"{axis}:{value}": (
            _condition_surface(axis, value),
            f"{value.replace('_', ' ')}-{_CONDITION_NOUNS[axis].replace(' ', '-')}",
        )
        for axis, values in WORLD_PROGRAM_AXES
        for value in values
    },
    **{
        f"primary_root:{root}": (
            _condition_surface("primary_root", root),
            f"dominant {root.replace('_', ' ')}",
        )
        for root in ROOT_MECHANICS
    },
    **{
        f"secondary_root:{root}": (
            _condition_surface("secondary_root", root),
            f"secondary {root.replace('_', ' ')}",
        )
        for root in ROOT_MECHANICS
    },
}


def render_world_condition_clause(context_signature: Sequence[str]) -> str:
    conditions = stable_condition_signature(context_signature)
    if conditions == ("condition:general",):
        return ""
    condition_set = set(conditions)
    ordered: list[str] = []
    for axis, values in WORLD_PROGRAM_AXES:
        for value in values:
            condition = f"{axis}:{value}"
            if condition in condition_set:
                ordered.append(_condition_surface(axis, value))
    for key in ("primary_root", "secondary_root"):
        for root in ROOT_MECHANICS:
            condition = f"{key}:{root}"
            if condition in condition_set:
                ordered.append(_condition_surface(key, root))
    if len(ordered) != len(condition_set):
        unknown = sorted(condition_set - set(WORLD_CONDITION_ALIASES))
        raise ValueError(f"unknown world conditions: {unknown}")
    if len(ordered) == 1:
        joined = ordered[0]
    else:
        joined = ", ".join(ordered[:-1]) + ", and " + ordered[-1]
    return f" The world regime has {joined}."


def _extract_world_conditions(text: str) -> tuple[tuple[str, ...], str]:
    lowered = text.lower().replace("_", " ")
    mentions: list[tuple[int, int, str]] = []
    for condition, aliases in WORLD_CONDITION_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                lowered,
            ):
                mentions.append((match.start(), match.end(), condition))
    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    occupied: list[tuple[int, int]] = []
    conditions: dict[str, str] = {}
    cleaned = list(lowered)
    for start, end, condition in mentions:
        if any(start < right and end > left for left, right in occupied):
            continue
        key = condition.split(":", 1)[0]
        previous = conditions.get(key)
        if previous is not None and previous != condition:
            raise ValueError(f"question contains conflicting {key} conditions")
        conditions[key] = condition
        occupied.append((start, end))
        cleaned[start:end] = " " * (end - start)
    return tuple(sorted(conditions.values())), "".join(cleaned)


def _surface_feature(feature: str, variant: int = 0) -> str:
    aliases = FEATURE_ALIASES.get(feature)
    if aliases is None:
        raise ValueError(f"unknown language feature: {feature}")
    return aliases[variant % len(aliases)]


def render_evidence_observation(evidence: CausalEvidence, variant: int = 0) -> str:
    evidence.validate()
    cause = _surface_feature(evidence.cause_feature, variant)
    effect = _surface_feature(evidence.effect_feature, variant + 1)
    direction = "rose" if evidence.direction > 0 else "fell"
    delay = "one step" if evidence.delay == 1 else f"{evidence.delay} steps"
    forms = (
        (
            f"In matched {evidence.domain} worlds, intervening on {cause} was "
            f"followed by {effect} that {direction} after {delay}."
        ),
        (
            f"The treated {evidence.domain} worlds differed from their controls: "
            f"after {cause} changed, {effect} {direction} within {delay}."
        ),
        (
            f"A controlled change to {cause} in the {evidence.domain} domain "
            f"produced a measurable {direction} in {effect} after {delay}."
        ),
        (
            f"Holding the initial {evidence.domain} state fixed while changing "
            f"{cause} caused {effect} to move {direction} after {delay}."
        ),
    )
    return forms[variant % len(forms)] + render_world_condition_clause(
        evidence.context_signature
    )


def render_causal_question(
    *,
    query_id: str,
    domain: str,
    cause_feature: str,
    effect_feature: str | None,
    variant: int = 0,
    condition_signature: Sequence[str] = (),
) -> dict[str, Any]:
    if domain not in DOMAIN_NAMES:
        raise ValueError("question domain is unknown")
    cause = _surface_feature(cause_feature, variant)
    if effect_feature is None:
        body = EFFECT_QUESTIONS[variant % len(EFFECT_QUESTIONS)].format(
            cause=cause, domain=domain
        )
    else:
        effect = _surface_feature(effect_feature, variant + 1)
        body = DIRECT_QUESTIONS[variant % len(DIRECT_QUESTIONS)].format(
            cause=cause,
            effect=effect,
            domain=domain,
        )
    text = (
        f"{QUESTION_OPENINGS[variant % len(QUESTION_OPENINGS)]}, {body}"
        + render_world_condition_clause(condition_signature)
    )
    return {
        "query_id": query_id,
        "text": text,
        "language_runtime": CAUSAL_WORLD_LANGUAGE_RUNTIME,
    }


def _find_domain(tokens: set[str]) -> str | None:
    for domain in DOMAIN_NAMES:
        if domain in tokens:
            return domain
    for domain, aliases in DOMAIN_ALIASES.items():
        for alias in aliases:
            if set(_WORDS.findall(alias)) <= tokens:
                return domain
    return None


def _feature_mentions(text: str) -> list[tuple[int, str]]:
    lowered = text.lower().replace("_", " ")
    mentions: list[tuple[int, int, str]] = []
    for feature, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            matches = re.finditer(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                lowered,
            )
            mentions.extend((match.start(), match.end(), feature) for match in matches)
    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    unique: list[tuple[int, str]] = []
    occupied_spans: list[tuple[int, int]] = []
    for start, end, feature in mentions:
        if any(
            start < occupied_end and end > occupied_start
            for occupied_start, occupied_end in occupied_spans
        ):
            continue
        unique.append((start, feature))
        occupied_spans.append((start, end))
    return unique


def parse_causal_question(request: Mapping[str, Any]) -> CausalQuery:
    if set(request) != {"language_runtime", "query_id", "text"}:
        raise ValueError("causal language request fields are invalid")
    if request["language_runtime"] != CAUSAL_WORLD_LANGUAGE_RUNTIME:
        raise ValueError("causal language runtime marker mismatch")
    query_id = request["query_id"]
    text = request["text"]
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("causal language query ID must be non-empty text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("causal language question must be non-empty text")
    conditions, causal_text = _extract_world_conditions(text)
    tokens = set(_WORDS.findall(causal_text.lower()))
    domain = _find_domain(tokens)
    mentions = _feature_mentions(causal_text)
    if not mentions:
        raise ValueError("question contains no grounded causal feature")
    mode = (
        "why" if "why" in tokens else "counterfactual" if "if" in tokens else "effect"
    )
    lowered = causal_text.lower()
    effect_before_cause = mode == "why" or "what happens to" in lowered
    if effect_before_cause and len(mentions) > 1 and " after " in lowered:
        after_position = lowered.rfind(" after ")
        cause_mentions = [item for item in mentions if item[0] > after_position]
        effect_mentions = [item for item in mentions if item[0] < after_position]
        cause_feature = cause_mentions[0][1] if cause_mentions else mentions[-1][1]
        effect_feature = effect_mentions[0][1] if effect_mentions else mentions[0][1]
    else:
        cause_feature = mentions[0][1]
        effect_feature = mentions[1][1] if len(mentions) > 1 else None
    context = tuple(
        value
        for value in (
            f"domain:{domain}" if domain else None,
            f"cause:{cause_feature}",
            f"mode:{mode}",
            *conditions,
        )
        if value is not None
    )
    query = CausalQuery(
        query_id=query_id,
        domain=domain,
        cause_feature=cause_feature,
        effect_feature=effect_feature,
        context_signature=context,
        mode=mode,
    )
    query.validate()
    return query


def render_causal_answer(artifact: Mapping[str, Any]) -> str:
    status = artifact.get("claim_status")
    if status == "unknown":
        reason = str(artifact.get("reason", "insufficient_causal_support"))
        return (
            "I do not have enough interventional evidence to answer that causal "
            f"question ({reason.replace('_', ' ')})."
        )
    if status != "derived" or not isinstance(artifact.get("assertion"), str):
        raise ValueError("causal answer artifact is invalid")
    delay = float(artifact["expected_delay"])
    delay_text = f"about {delay:.1f} world steps"
    path_length = int(artifact.get("path_length", 1))
    evidence_subject = (
        "This law" if path_length == 1 else f"This {path_length}-link causal path"
    )
    condition_clause = render_world_condition_clause(
        tuple(str(value) for value in artifact.get("condition_signature", ()))
    )
    condition_text = " " + condition_clause.strip() if condition_clause else ""
    return (
        f"{artifact['assertion']} The effect normally appears after {delay_text}. "
        f"{evidence_subject} has {artifact['support']} supporting interventions, "
        f"{artifact['contradictions']} contradictions, confidence "
        f"{float(artifact['confidence']):.3f}, and persistence "
        f"{float(artifact['persistence']):.3f}.{condition_text}"
    )


def language_space_manifest() -> dict[str, Any]:
    feature_surfaces = sum(len(values) for values in FEATURE_ALIASES.values())
    domain_surfaces = sum(len(values) for values in DOMAIN_ALIASES.values())
    direct_combinations = (
        len(QUESTION_OPENINGS)
        * len(DIRECT_QUESTIONS)
        * feature_surfaces
        * feature_surfaces
        * domain_surfaces
    )
    open_combinations = (
        len(QUESTION_OPENINGS)
        * len(EFFECT_QUESTIONS)
        * feature_surfaces
        * domain_surfaces
    )
    return {
        "runtime": CAUSAL_WORLD_LANGUAGE_RUNTIME,
        "feature_concepts": len(FEATURE_NAMES),
        "feature_surfaces": feature_surfaces,
        "domain_surfaces": domain_surfaces,
        "question_openings": len(QUESTION_OPENINGS),
        "direct_question_space": direct_combinations,
        "open_question_space": open_combinations,
        "combined_question_space": direct_combinations + open_combinations,
        "world_program_space": world_program_space_size(),
        "conditioned_question_space": (direct_combinations + open_combinations)
        * world_program_space_size(),
        "knowledge_authority": "causal_graph_only",
    }


def causal_world_language_self_test() -> dict[str, bool]:
    request = render_causal_question(
        query_id="language-self-test",
        domain="physical",
        cause_feature="temperature",
        effect_feature="energy",
        variant=2,
    )
    query = parse_causal_question(request)
    replay = render_causal_question(
        query_id="language-self-test",
        domain="physical",
        cause_feature="temperature",
        effect_feature="energy",
        variant=2,
    )
    manifest = language_space_manifest()
    condition_signature = curriculum_programs(0)[0].condition_signature()
    conditioned_request = render_causal_question(
        query_id="conditioned-language-self-test",
        domain="physical",
        cause_feature="resource",
        effect_feature="energy",
        variant=1,
        condition_signature=condition_signature,
    )
    conditioned_query = parse_causal_question(conditioned_request)
    direct_roundtrips = []
    repeated_feature_roundtrips = []
    for variant in range(len(DIRECT_QUESTIONS)):
        item = parse_causal_question(
            render_causal_question(
                query_id=f"direct-{variant}",
                domain="social",
                cause_feature="charge",
                effect_feature="cohesion",
                variant=variant,
            )
        )
        direct_roundtrips.append(
            item.domain == "social"
            and item.cause_feature == "charge"
            and item.effect_feature == "cohesion"
        )
        repeated = parse_causal_question(
            render_causal_question(
                query_id=f"repeated-{variant}",
                domain="symbolic",
                cause_feature="pressure",
                effect_feature="pressure",
                variant=variant,
            )
        )
        repeated_feature_roundtrips.append(
            repeated.domain == "symbolic"
            and repeated.cause_feature == "pressure"
            and repeated.effect_feature == "pressure"
        )
    boundary_query = parse_causal_question(
        {
            "query_id": "boundary",
            "text": (
                "Using only supported relationships, why did reliability change "
                "after the intervention on language alignment in the chemical domain?"
            ),
            "language_runtime": CAUSAL_WORLD_LANGUAGE_RUNTIME,
        }
    )
    checks = {
        "roundtrip_domain": query.domain == "physical",
        "roundtrip_cause": query.cause_feature == "temperature",
        "roundtrip_effect": query.effect_feature == "energy",
        "deterministic_render": request == replay,
        "large_language_surface": manifest["combined_question_space"] > 1_000_000,
        "request_hash_stable": canonical_hash(request) == canonical_hash(replay),
        "all_features_have_surfaces": set(FEATURE_ALIASES) == set(FEATURE_NAMES),
        "all_direct_forms_roundtrip": all(direct_roundtrips),
        "repeated_feature_roles_roundtrip": all(repeated_feature_roundtrips),
        "feature_aliases_respect_word_boundaries": (
            boundary_query.cause_feature == "language_alignment"
            and boundary_query.effect_feature == "trust"
        ),
        "conditioned_question_roundtrip": stable_condition_signature(
            conditioned_query.context_signature
        )
        == stable_condition_signature(condition_signature),
        "condition_terms_do_not_replace_causal_features": (
            conditioned_query.cause_feature == "resource"
            and conditioned_query.effect_feature == "energy"
        ),
        "conditioned_language_space_exceeds_hundred_trillion": manifest[
            "conditioned_question_space"
        ]
        > 100_000_000_000_000,
    }
    if not all(checks.values()):
        raise AssertionError(f"causal-world language self-test failed: {checks}")
    return checks


if __name__ == "__main__":
    print(causal_world_language_self_test())
