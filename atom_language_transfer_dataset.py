"""Disjoint-lexicon transfer program for the grounded Atom language field."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping


TRANSFER_SEED = 731_204
TRANSFER_AGENTS = ("agent-4", "agent-5", "agent-6")
TRANSFER_OBJECTS = ("object-3", "object-4", "object-5")
TRANSFER_LOCATIONS = ("location-3", "location-4", "location-5")
TRANSFER_LEXICON = {
    "agent-4": "lumi",
    "agent-5": "varo",
    "agent-6": "zeni",
    "object-3": "gem",
    "object-4": "ring",
    "object-5": "vial",
    "location-3": "cove",
    "location-4": "spire",
    "location-5": "grove",
}


def _transfer_stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _transfer_world() -> dict[str, Any]:
    return {
        "locations": {
            agent: TRANSFER_LOCATIONS[index]
            for index, agent in enumerate(TRANSFER_AGENTS)
        },
        "holders": {item: None for item in TRANSFER_OBJECTS},
    }


def _transfer_frame(speech_act: str, predicate: str, **roles: str) -> dict[str, Any]:
    return {
        "speech_act": speech_act,
        "predicate": predicate,
        "polarity": True,
        "roles": dict(sorted(roles.items())),
    }


def _transfer_episode(
    family: str,
    identity: str,
    text: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    truth: Mapping[str, Any] | None = None,
    *,
    answer_text: str | None = None,
    context_text: str | None = None,
    paraphrase_text: str | None = None,
    salience: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_id = f"transfer-{_transfer_stable_hash([family, identity])[:16]}"
    observation = {
        "case_id": case_id,
        "text": text,
        "context_text": context_text,
        "paraphrase_text": paraphrase_text,
        "before": deepcopy(dict(before)),
        "after": deepcopy(dict(after)),
        "answer_text": answer_text,
        "salience": salience,
    }
    evaluation = None
    if truth is not None:
        evaluation = {
            "family": family,
            "frame": deepcopy(dict(truth)),
            "expected_after": deepcopy(dict(after)),
            "expected_answer": answer_text,
        }
    return observation, evaluation


def _transfer_grounding_demonstrations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in (0, 1):
        for index, agent in enumerate(TRANSFER_AGENTS):
            location = TRANSFER_LOCATIONS[(index + offset) % 3]
            before = _transfer_world()
            before["locations"][agent] = TRANSFER_LOCATIONS[(index + offset + 1) % 3]
            after = deepcopy(before)
            after["locations"][agent] = location
            observation, _ = _transfer_episode(
                "ground_move",
                f"{offset}-{agent}-{location}",
                f"move {TRANSFER_LEXICON[agent]} to {TRANSFER_LEXICON[location]}",
                before,
                after,
            )
            rows.append(observation)

            item = TRANSFER_OBJECTS[(index + offset) % 3]
            before = _transfer_world()
            after = deepcopy(before)
            after["holders"][item] = agent
            observation, _ = _transfer_episode(
                "ground_take",
                f"{offset}-{agent}-{item}",
                f"{TRANSFER_LEXICON[agent]} take {TRANSFER_LEXICON[item]}",
                before,
                after,
            )
            rows.append(observation)
    return rows


def _transfer_transient_disturbances() -> list[dict[str, Any]]:
    false_before = _transfer_world()
    false_before["locations"]["agent-5"] = "location-4"
    false_after = deepcopy(false_before)
    false_after["locations"]["agent-5"] = "location-3"
    false_row, _ = _transfer_episode(
        "transient_false_grounding",
        "lumi-low-salience-wrong-agent",
        "move lumi to cove",
        false_before,
        false_after,
        salience=0.2,
    )

    noise_before = _transfer_world()
    noise_before["locations"]["agent-4"] = "location-3"
    noise_after = deepcopy(noise_before)
    noise_after["locations"]["agent-4"] = "location-4"
    noise_row, _ = _transfer_episode(
        "transient_noise",
        "florp-one-off",
        "move florp to spire",
        noise_before,
        noise_after,
        salience=0.3,
    )
    return [false_row, noise_row]


def _transfer_heldout_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}

    def add(family: str, identity: str, *args: Any, **kwargs: Any) -> None:
        observation, evaluation = _transfer_episode(family, identity, *args, **kwargs)
        if evaluation is None:
            raise AssertionError("held-out transfer row requires evaluator truth")
        rows.append(observation)
        truth[observation["case_id"]] = evaluation

    for index, agent in enumerate(TRANSFER_AGENTS):
        agent_word = TRANSFER_LEXICON[agent]
        location = TRANSFER_LOCATIONS[(index + 2) % 3]
        location_word = TRANSFER_LEXICON[location]
        item = TRANSFER_OBJECTS[(index + 2) % 3]
        item_word = TRANSFER_LEXICON[item]

        before = _transfer_world()
        before["locations"][agent] = TRANSFER_LOCATIONS[index]
        after = deepcopy(before)
        after["locations"][agent] = location
        add(
            "move_cross",
            agent,
            f"move {agent_word} to {location_word}",
            before,
            after,
            _transfer_frame("command", "MOVE", agent=agent, destination=location),
        )

        before = _transfer_world()
        after = deepcopy(before)
        after["holders"][item] = agent
        add(
            "take_cross",
            agent,
            f"{agent_word} take {item_word}",
            before,
            after,
            _transfer_frame("command", "TAKE", agent=agent, patient=item),
        )

        for recipient_offset, object_offset in ((1, 2), (2, 1)):
            recipient = TRANSFER_AGENTS[(index + recipient_offset) % 3]
            give_item = TRANSFER_OBJECTS[(index + object_offset) % 3]
            before = _transfer_world()
            before["holders"][give_item] = agent
            after = deepcopy(before)
            after["holders"][give_item] = recipient
            add(
                "give_novel",
                f"{agent}-{recipient}-{give_item}",
                (
                    f"{agent_word} give {TRANSFER_LEXICON[give_item]} "
                    f"to {TRANSFER_LEXICON[recipient]}"
                ),
                before,
                after,
                _transfer_frame(
                    "command",
                    "GIVE",
                    agent=agent,
                    patient=give_item,
                    recipient=recipient,
                ),
            )

        at_state = _transfer_world()
        at_state["locations"][agent] = location
        add(
            "assert_at_transfer",
            agent,
            f"{agent_word} is at {location_word}",
            at_state,
            at_state,
            _transfer_frame("assertion", "AT", agent=agent, destination=location),
        )
        add(
            "where_transfer",
            agent,
            f"where is {agent_word}",
            at_state,
            at_state,
            _transfer_frame("question", "WHERE", agent=agent),
            answer_text=f"{agent_word} is at {location_word}",
        )

        has_state = _transfer_world()
        has_state["holders"][item] = agent
        add(
            "assert_has_transfer",
            agent,
            f"{agent_word} holds {item_word}",
            has_state,
            has_state,
            _transfer_frame("assertion", "HAS", agent=agent, patient=item),
        )
        add(
            "what_has_transfer",
            agent,
            f"what does {agent_word} hold",
            has_state,
            has_state,
            _transfer_frame("question", "WHAT_HAS", agent=agent),
            answer_text=f"{agent_word} holds {item_word}",
        )
        add(
            "who_has_transfer",
            agent,
            f"who holds {item_word}",
            has_state,
            has_state,
            _transfer_frame("question", "WHO_HAS", patient=item),
            answer_text=f"{agent_word} holds {item_word}",
        )
        add(
            "has_query_true_transfer",
            agent,
            f"does {agent_word} hold {item_word}",
            has_state,
            has_state,
            _transfer_frame("question", "HAS_QUERY", agent=agent, patient=item),
            answer_text="yes",
        )

        false_item = TRANSFER_OBJECTS[(index + 1) % 3]
        false_state = _transfer_world()
        false_state["holders"][false_item] = TRANSFER_AGENTS[(index + 1) % 3]
        add(
            "has_query_false_transfer",
            agent,
            f"does {agent_word} hold {TRANSFER_LEXICON[false_item]}",
            false_state,
            false_state,
            _transfer_frame("question", "HAS_QUERY", agent=agent, patient=false_item),
            answer_text="no",
        )
        add(
            "at_query_true_transfer",
            agent,
            f"is {agent_word} at {location_word}",
            at_state,
            at_state,
            _transfer_frame("question", "AT_QUERY", agent=agent, destination=location),
            answer_text="yes",
        )

        false_location = TRANSFER_LOCATIONS[(index + 1) % 3]
        add(
            "at_query_false_transfer",
            agent,
            f"is {agent_word} at {TRANSFER_LEXICON[false_location]}",
            at_state,
            at_state,
            _transfer_frame(
                "question",
                "AT_QUERY",
                agent=agent,
                destination=false_location,
            ),
            answer_text="no",
        )
        add(
            "dialog_where_transfer",
            agent,
            "where are they",
            at_state,
            at_state,
            _transfer_frame("question", "WHERE", agent=agent),
            context_text=f"{agent_word} is at {location_word}",
            answer_text=f"{agent_word} is at {location_word}",
        )
        add(
            "dialog_who_transfer",
            agent,
            "who holds it",
            has_state,
            has_state,
            _transfer_frame("question", "WHO_HAS", patient=item),
            context_text=f"{agent_word} holds {item_word}",
            answer_text=f"{agent_word} holds {item_word}",
        )
        add(
            "dialog_at_query_transfer",
            agent,
            f"are they at {location_word}",
            at_state,
            at_state,
            _transfer_frame("question", "AT_QUERY", agent=agent, destination=location),
            context_text=f"{agent_word} is at {location_word}",
            paraphrase_text=f"is {agent_word} at {location_word}",
            answer_text="yes",
        )

    return rows, truth


def build_language_transfer_program() -> dict[str, Any]:
    grounding = _transfer_grounding_demonstrations()
    transient = _transfer_transient_disturbances()
    heldout, truth = _transfer_heldout_cases()
    if (len(grounding), len(transient), len(heldout)) != (12, 2, 48):
        raise AssertionError("transfer program must contain 12/2/48 rows")
    grounding_ids = {row["case_id"] for row in grounding}
    heldout_ids = {row["case_id"] for row in heldout}
    if grounding_ids & heldout_ids or len(heldout_ids) != len(heldout):
        raise AssertionError("transfer rows must be unique and disjoint")
    if set(truth) != heldout_ids:
        raise AssertionError("transfer evaluator truth must match held-out rows")

    grounding_pairs = {tuple(row["text"].split()) for row in grounding}
    heldout_action_pairs = {
        tuple(row["text"].split())
        for row in heldout
        if row["before"] != row["after"] and "give" not in row["text"]
    }
    if grounding_pairs & heldout_action_pairs:
        raise AssertionError("held-out action surfaces must not repeat grounding rows")

    manifest = {
        "seed": TRANSFER_SEED,
        "counts": {
            "grounding": len(grounding),
            "transient": len(transient),
            "heldout": len(heldout),
        },
        "new_concepts": 9,
        "gold_meanings_in_grounding": False,
        "disjoint_from_base_concepts": True,
        "heldout_action_surface_overlap": 0,
        "hashes": {
            "grounding": _transfer_stable_hash(grounding),
            "transient": _transfer_stable_hash(transient),
            "heldout": _transfer_stable_hash(heldout),
            "truth": _transfer_stable_hash(truth),
        },
    }
    return {
        "grounding": grounding,
        "transient": transient,
        "heldout": heldout,
        "evaluation_truth": truth,
        "manifest": manifest,
        "evaluator_oracle": {
            "concept_to_surface": dict(TRANSFER_LEXICON),
            "surface_to_concept": {
                surface: concept for concept, surface in TRANSFER_LEXICON.items()
            },
        },
    }
