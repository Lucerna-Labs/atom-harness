"""Opaque compositional grammar program for the Atom language field."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


OPAQUE_SEED = 904_271
OPAQUE_AGENTS = ("agent-7", "agent-8", "agent-9", "agent-10")
OPAQUE_OBJECTS = ("object-6", "object-7", "object-8", "object-9")
OPAQUE_LOCATIONS = ("location-6", "location-7", "location-8", "location-9")
OPAQUE_LEXICON = {
    "agent-7": "zal",
    "agent-8": "mer",
    "agent-9": "tuk",
    "agent-10": "fep",
    "object-6": "piv",
    "object-7": "dos",
    "object-8": "kem",
    "object-9": "rag",
    "location-6": "ral",
    "location-7": "siv",
    "location-8": "bon",
    "location-9": "nul",
}
OPAQUE_PATTERNS = {
    "command:MOVE:1": ("{destination}", "avo", "{agent}"),
    "command:TAKE:1": ("{patient}", "{agent}", "dre"),
    "command:GIVE:1": ("{recipient}", "sok", "{patient}", "{agent}"),
    "assertion:AT:1": ("esh", "{destination}", "{agent}"),
    "assertion:HAS:1": ("lum", "{agent}", "{patient}"),
    "question:WHERE:1": ("{agent}", "qir"),
    "question:WHAT_HAS:1": ("nav", "{agent}"),
    "question:WHO_HAS:1": ("{patient}", "tor"),
    "question:HAS_QUERY:1": ("xem", "{patient}", "{agent}"),
    "question:AT_QUERY:1": ("yor", "{agent}", "{destination}"),
    "answer:YES:1": ("aya",),
    "answer:NO:1": ("nox",),
}


def _opaque_stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_world() -> dict[str, Any]:
    return {
        "locations": {
            agent: OPAQUE_LOCATIONS[index] for index, agent in enumerate(OPAQUE_AGENTS)
        },
        "holders": {item: None for item in OPAQUE_OBJECTS},
    }


def _opaque_frame(speech_act: str, predicate: str, **roles: str) -> dict[str, Any]:
    return {
        "speech_act": speech_act,
        "predicate": predicate,
        "polarity": True,
        "roles": dict(sorted(roles.items())),
    }


def render_opaque_frame(frame: Mapping[str, Any]) -> str:
    key = f"{frame['speech_act']}:{frame['predicate']}:{int(bool(frame['polarity']))}"
    pattern = OPAQUE_PATTERNS.get(key)
    roles = frame.get("roles")
    if pattern is None or not isinstance(roles, Mapping):
        raise ValueError(f"No opaque surface pattern for {key}")
    output: list[str] = []
    for piece in pattern:
        if piece.startswith("{") and piece.endswith("}"):
            role = piece[1:-1]
            concept = roles.get(role)
            surface = OPAQUE_LEXICON.get(str(concept))
            if surface is None:
                raise ValueError(f"No opaque surface for role {role}")
            output.append(surface)
        else:
            output.append(piece)
    return " ".join(output)


def _opaque_episode(
    split: str,
    family: str,
    identity: str,
    frame: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    answer_frame: Mapping[str, Any] | None = None,
    salience: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = f"opaque-{_opaque_stable_hash([split, family, identity])[:16]}"
    answer_text = None if answer_frame is None else render_opaque_frame(answer_frame)
    observation = {
        "case_id": case_id,
        "text": render_opaque_frame(frame),
        "context_text": None,
        "paraphrase_text": None,
        "before": deepcopy(dict(before)),
        "after": deepcopy(dict(after)),
        "answer_text": answer_text,
        "salience": salience,
    }
    evaluation = {
        "family": family,
        "frame": deepcopy(dict(frame)),
        "expected_after": deepcopy(dict(after)),
        "expected_answer": answer_text,
    }
    return observation, evaluation


def _opaque_action(
    split: str,
    family: str,
    identity: str,
    frame: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    roles = frame["roles"]
    before = _opaque_world()
    after = deepcopy(before)
    predicate = frame["predicate"]
    if predicate == "MOVE":
        agent = roles["agent"]
        destination = roles["destination"]
        if before["locations"][agent] == destination:
            before["locations"][agent] = OPAQUE_LOCATIONS[
                (OPAQUE_LOCATIONS.index(destination) + 1) % len(OPAQUE_LOCATIONS)
            ]
            after = deepcopy(before)
        after["locations"][agent] = destination
    elif predicate == "TAKE":
        after["holders"][roles["patient"]] = roles["agent"]
    elif predicate == "GIVE":
        before["holders"][roles["patient"]] = roles["agent"]
        after = deepcopy(before)
        after["holders"][roles["patient"]] = roles["recipient"]
    else:
        raise ValueError("opaque action requires MOVE, TAKE, or GIVE")
    return _opaque_episode(split, family, identity, frame, before, after)


def _opaque_state_row(
    split: str,
    family: str,
    identity: str,
    frame: Mapping[str, Any],
    *,
    answer_frame: Mapping[str, Any] | None = None,
    agent_location: tuple[str, str] | None = None,
    item_holder: tuple[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _opaque_world()
    if agent_location is not None:
        state["locations"][agent_location[0]] = agent_location[1]
    if item_holder is not None:
        state["holders"][item_holder[0]] = item_holder[1]
    return _opaque_episode(
        split,
        family,
        identity,
        frame,
        state,
        state,
        answer_frame=answer_frame,
    )


def _opaque_training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        rows.append(result[0])

    for offset in (0, 1):
        for index, agent in enumerate(OPAQUE_AGENTS):
            destination = OPAQUE_LOCATIONS[(index + offset + 1) % 4]
            add(
                _opaque_action(
                    "train",
                    "ground_move",
                    f"{offset}-{agent}-{destination}",
                    _opaque_frame(
                        "command",
                        "MOVE",
                        agent=agent,
                        destination=destination,
                    ),
                )
            )
            item = OPAQUE_OBJECTS[(index + offset) % 4]
            add(
                _opaque_action(
                    "train",
                    "ground_take",
                    f"{offset}-{agent}-{item}",
                    _opaque_frame("command", "TAKE", agent=agent, patient=item),
                )
            )

    for index, agent in enumerate(OPAQUE_AGENTS):
        recipient = OPAQUE_AGENTS[(index + 1) % 4]
        item = OPAQUE_OBJECTS[(index + 2) % 4]
        add(
            _opaque_action(
                "train",
                "ground_give",
                f"{agent}-{recipient}-{item}",
                _opaque_frame(
                    "command",
                    "GIVE",
                    agent=agent,
                    patient=item,
                    recipient=recipient,
                ),
            )
        )

        destination = OPAQUE_LOCATIONS[(index + 1) % 4]
        item = OPAQUE_OBJECTS[(index + 1) % 4]
        at_frame = _opaque_frame(
            "assertion", "AT", agent=agent, destination=destination
        )
        has_frame = _opaque_frame("assertion", "HAS", agent=agent, patient=item)
        add(
            _opaque_state_row(
                "train",
                "ground_assert_at",
                agent,
                at_frame,
                agent_location=(agent, destination),
            )
        )
        add(
            _opaque_state_row(
                "train",
                "ground_assert_has",
                agent,
                has_frame,
                item_holder=(item, agent),
            )
        )
        add(
            _opaque_state_row(
                "train",
                "ground_where",
                agent,
                _opaque_frame("question", "WHERE", agent=agent),
                answer_frame=at_frame,
                agent_location=(agent, destination),
            )
        )
        add(
            _opaque_state_row(
                "train",
                "ground_what_has",
                agent,
                _opaque_frame("question", "WHAT_HAS", agent=agent),
                answer_frame=has_frame,
                item_holder=(item, agent),
            )
        )
        add(
            _opaque_state_row(
                "train",
                "ground_who_has",
                agent,
                _opaque_frame("question", "WHO_HAS", patient=item),
                answer_frame=has_frame,
                item_holder=(item, agent),
            )
        )

        has_true = index % 2 == 0
        query_item = item if has_true else OPAQUE_OBJECTS[(index + 2) % 4]
        add(
            _opaque_state_row(
                "train",
                "ground_has_query",
                agent,
                _opaque_frame("question", "HAS_QUERY", agent=agent, patient=query_item),
                answer_frame=_opaque_frame("answer", "YES" if has_true else "NO"),
                item_holder=(
                    query_item,
                    agent if has_true else OPAQUE_AGENTS[(index + 1) % 4],
                ),
            )
        )

        at_true = index % 2 == 1
        query_destination = (
            destination if at_true else OPAQUE_LOCATIONS[(index + 2) % 4]
        )
        add(
            _opaque_state_row(
                "train",
                "ground_at_query",
                agent,
                _opaque_frame(
                    "question",
                    "AT_QUERY",
                    agent=agent,
                    destination=query_destination,
                ),
                answer_frame=_opaque_frame("answer", "YES" if at_true else "NO"),
                agent_location=(agent, destination),
            )
        )

    if len(rows) != 48:
        raise AssertionError("opaque training split must contain 48 rows")
    return rows


def _opaque_validation_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}

    def add(result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        observation, evaluation = result
        rows.append(observation)
        truth[observation["case_id"]] = evaluation

    for index, agent in enumerate(OPAQUE_AGENTS):
        destination = OPAQUE_LOCATIONS[(index + 3) % 4]
        item = OPAQUE_OBJECTS[(index + 2) % 4]
        add(
            _opaque_action(
                "validation",
                "validate_move",
                agent,
                _opaque_frame("command", "MOVE", agent=agent, destination=destination),
            )
        )
        add(
            _opaque_action(
                "validation",
                "validate_take",
                agent,
                _opaque_frame("command", "TAKE", agent=agent, patient=item),
            )
        )
        recipient = OPAQUE_AGENTS[(index + 2) % 4]
        give_item = OPAQUE_OBJECTS[(index + 3) % 4]
        add(
            _opaque_action(
                "validation",
                "validate_give",
                agent,
                _opaque_frame(
                    "command",
                    "GIVE",
                    agent=agent,
                    patient=give_item,
                    recipient=recipient,
                ),
            )
        )
        at_frame = _opaque_frame(
            "assertion", "AT", agent=agent, destination=destination
        )
        has_frame = _opaque_frame("assertion", "HAS", agent=agent, patient=item)
        add(
            _opaque_state_row(
                "validation",
                "validate_assert_at",
                agent,
                at_frame,
                agent_location=(agent, destination),
            )
        )
        add(
            _opaque_state_row(
                "validation",
                "validate_assert_has",
                agent,
                has_frame,
                item_holder=(item, agent),
            )
        )

    if len(rows) != 20:
        raise AssertionError("opaque validation split must contain 20 rows")
    return rows, truth


def _opaque_heldout_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}

    def add(result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        observation, evaluation = result
        rows.append(observation)
        truth[observation["case_id"]] = evaluation

    for index, agent in enumerate(OPAQUE_AGENTS):
        destination = OPAQUE_LOCATIONS[index]
        item = OPAQUE_OBJECTS[(index + 3) % 4]
        add(
            _opaque_action(
                "heldout",
                "move_systematic",
                agent,
                _opaque_frame("command", "MOVE", agent=agent, destination=destination),
            )
        )
        add(
            _opaque_action(
                "heldout",
                "take_systematic",
                agent,
                _opaque_frame("command", "TAKE", agent=agent, patient=item),
            )
        )
        for recipient_offset, item_offset in ((2, 0), (3, 1)):
            recipient = OPAQUE_AGENTS[(index + recipient_offset) % 4]
            give_item = OPAQUE_OBJECTS[(index + item_offset) % 4]
            add(
                _opaque_action(
                    "heldout",
                    "give_systematic",
                    f"{agent}-{recipient}-{give_item}",
                    _opaque_frame(
                        "command",
                        "GIVE",
                        agent=agent,
                        patient=give_item,
                        recipient=recipient,
                    ),
                )
            )

        at_frame = _opaque_frame(
            "assertion", "AT", agent=agent, destination=destination
        )
        has_frame = _opaque_frame("assertion", "HAS", agent=agent, patient=item)
        add(
            _opaque_state_row(
                "heldout",
                "assert_at_systematic",
                agent,
                at_frame,
                agent_location=(agent, destination),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "assert_has_systematic",
                agent,
                has_frame,
                item_holder=(item, agent),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "where_systematic",
                agent,
                _opaque_frame("question", "WHERE", agent=agent),
                answer_frame=at_frame,
                agent_location=(agent, destination),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "what_has_systematic",
                agent,
                _opaque_frame("question", "WHAT_HAS", agent=agent),
                answer_frame=has_frame,
                item_holder=(item, agent),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "who_has_systematic",
                agent,
                _opaque_frame("question", "WHO_HAS", patient=item),
                answer_frame=has_frame,
                item_holder=(item, agent),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "has_query_true_systematic",
                agent,
                _opaque_frame("question", "HAS_QUERY", agent=agent, patient=item),
                answer_frame=_opaque_frame("answer", "YES"),
                item_holder=(item, agent),
            )
        )
        false_item = OPAQUE_OBJECTS[(index + 2) % 4]
        add(
            _opaque_state_row(
                "heldout",
                "has_query_false_systematic",
                agent,
                _opaque_frame("question", "HAS_QUERY", agent=agent, patient=false_item),
                answer_frame=_opaque_frame("answer", "NO"),
                item_holder=(false_item, OPAQUE_AGENTS[(index + 1) % 4]),
            )
        )
        add(
            _opaque_state_row(
                "heldout",
                "at_query_true_systematic",
                agent,
                _opaque_frame(
                    "question",
                    "AT_QUERY",
                    agent=agent,
                    destination=destination,
                ),
                answer_frame=_opaque_frame("answer", "YES"),
                agent_location=(agent, destination),
            )
        )
        false_destination = OPAQUE_LOCATIONS[(index + 2) % 4]
        add(
            _opaque_state_row(
                "heldout",
                "at_query_false_systematic",
                agent,
                _opaque_frame(
                    "question",
                    "AT_QUERY",
                    agent=agent,
                    destination=false_destination,
                ),
                answer_frame=_opaque_frame("answer", "NO"),
                agent_location=(agent, destination),
            )
        )

    if len(rows) != 52:
        raise AssertionError("opaque held-out split must contain 52 rows")
    return rows, truth


def _surface_vocabulary(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    surfaces: set[str] = set()
    for row in rows:
        for key in ("text", "answer_text"):
            value = row.get(key)
            if isinstance(value, str):
                surfaces.update(value.split())
    return surfaces


def build_opaque_language_program() -> dict[str, Any]:
    train = _opaque_training_rows()
    validation, validation_truth = _opaque_validation_rows()
    heldout, heldout_truth = _opaque_heldout_rows()
    all_rows = train + validation + heldout
    all_ids = [str(row["case_id"]) for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("opaque program case identifiers must be unique")
    if any("frame" in row or "family" in row for row in all_rows):
        raise AssertionError("opaque observations cannot contain evaluator meanings")
    train_actions = {str(row["text"]) for row in train if row["before"] != row["after"]}
    heldout_actions = {
        str(row["text"]) for row in heldout if row["before"] != row["after"]
    }
    if train_actions & heldout_actions:
        raise AssertionError("opaque held-out actions must be surface novel")
    train_surfaces = {str(row["text"]) for row in train}
    heldout_surfaces = {str(row["text"]) for row in heldout}
    manifest = {
        "seed": OPAQUE_SEED,
        "counts": {"train": 48, "validation": 20, "heldout": 52},
        "concepts": len(OPAQUE_LEXICON),
        "grammar_patterns": len(OPAQUE_PATTERNS),
        "gold_meanings_in_observations": False,
        "heldout_action_surface_overlap": 0,
        "heldout_all_surface_overlap": len(train_surfaces & heldout_surfaces),
        "surface_vocabulary": sorted(_surface_vocabulary(all_rows)),
        "hashes": {
            "train": _opaque_stable_hash(train),
            "validation": _opaque_stable_hash(validation),
            "heldout": _opaque_stable_hash(heldout),
            "validation_truth": _opaque_stable_hash(validation_truth),
            "heldout_truth": _opaque_stable_hash(heldout_truth),
        },
    }
    return {
        "train": train,
        "validation": validation,
        "heldout": heldout,
        "validation_truth": validation_truth,
        "evaluation_truth": heldout_truth,
        "manifest": manifest,
        "evaluator_oracle": {
            "concept_to_surface": dict(OPAQUE_LEXICON),
            "surface_to_concept": {
                surface: concept for concept, surface in OPAQUE_LEXICON.items()
            },
            "patterns": {key: list(value) for key, value in OPAQUE_PATTERNS.items()},
        },
    }
