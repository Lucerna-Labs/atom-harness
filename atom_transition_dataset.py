"""Opaque before/after program for emergent transition-law discovery.

The learner receives only text and executable world consequences.  Human
semantic names live in evaluator maps that are never passed to the runtime.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


TRANSITION_DATASET_SEED = 2_026_072_101
TRANSITION_AGENTS = (
    "agent-11",
    "agent-12",
    "agent-13",
    "agent-14",
    "agent-15",
)
TRANSITION_OBJECTS = (
    "object-10",
    "object-11",
    "object-12",
    "object-13",
    "object-14",
)
TRANSITION_LOCATIONS = (
    "location-10",
    "location-11",
    "location-12",
    "location-13",
    "location-14",
)
TRANSITION_LEXICON = {
    "agent-11": "vra",
    "agent-12": "kel",
    "agent-13": "sot",
    "agent-14": "miv",
    "agent-15": "dan",
    "object-10": "pax",
    "object-11": "gul",
    "object-12": "tem",
    "object-13": "zor",
    "object-14": "bik",
    "location-10": "fen",
    "location-11": "ruk",
    "location-12": "lom",
    "location-13": "cid",
    "location-14": "wex",
}
TRANSITION_PATTERNS = {
    "relocate": ("{destination}", "uja", "{actor}"),
    "acquire": ("{object}", "{actor}", "bek"),
    "transfer": ("{recipient}", "toh", "{object}", "{actor}"),
    "exchange_locations": ("{other}", "iry", "{actor}"),
    "release": ("{object}", "nop", "{actor}"),
}


def transition_dataset_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transition_world() -> dict[str, Any]:
    return {
        "locations": {
            agent: TRANSITION_LOCATIONS[index]
            for index, agent in enumerate(TRANSITION_AGENTS)
        },
        "holders": {item: None for item in TRANSITION_OBJECTS},
    }


def render_transition(
    semantic_label: str,
    participants: Mapping[str, str],
) -> str:
    pattern = TRANSITION_PATTERNS.get(semantic_label)
    if pattern is None:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    output: list[str] = []
    for piece in pattern:
        if piece.startswith("{") and piece.endswith("}"):
            role = piece[1:-1]
            concept = participants.get(role)
            surface = TRANSITION_LEXICON.get(str(concept))
            if surface is None:
                raise ValueError(f"Missing opaque surface for evaluator role {role}")
            output.append(surface)
        else:
            output.append(piece)
    return " ".join(output)


def apply_evaluator_transition(
    semantic_label: str,
    participants: Mapping[str, str],
    before: Mapping[str, Any],
) -> dict[str, Any]:
    after = deepcopy(dict(before))
    if semantic_label == "relocate":
        after["locations"][participants["actor"]] = participants["destination"]
    elif semantic_label == "acquire":
        after["holders"][participants["object"]] = participants["actor"]
    elif semantic_label == "transfer":
        item = participants["object"]
        actor = participants["actor"]
        if before["holders"][item] != actor:
            raise ValueError("Evaluator transfer requires the actor to hold the object")
        after["holders"][item] = participants["recipient"]
    elif semantic_label == "exchange_locations":
        actor = participants["actor"]
        other = participants["other"]
        after["locations"][actor] = before["locations"][other]
        after["locations"][other] = before["locations"][actor]
    elif semantic_label == "release":
        item = participants["object"]
        actor = participants["actor"]
        if before["holders"][item] != actor:
            raise ValueError("Evaluator release requires the actor to hold the object")
        after["holders"][item] = None
    else:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    return after


def _transition_episode(
    split: str,
    semantic_label: str,
    identity: str,
    participants: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = transition_world()
    if semantic_label == "relocate":
        actor = participants["actor"]
        destination = participants["destination"]
        if before["locations"][actor] == destination:
            current = TRANSITION_LOCATIONS.index(destination)
            before["locations"][actor] = TRANSITION_LOCATIONS[(current + 1) % 5]
    elif semantic_label in {"transfer", "release"}:
        before["holders"][participants["object"]] = participants["actor"]
    after = apply_evaluator_transition(semantic_label, participants, before)
    case_id = "law-episode-" + transition_dataset_hash(
        [split, semantic_label, identity]
    )[:18]
    observation = {
        "case_id": case_id,
        "text": render_transition(semantic_label, participants),
        "before": deepcopy(before),
        "after": deepcopy(after),
        "salience": 1.0,
    }
    evaluation = {
        "semantic_label": semantic_label,
        "participants": dict(sorted(participants.items())),
        "expected_after": deepcopy(after),
    }
    return observation, evaluation


def _family_case(
    split: str,
    semantic_label: str,
    index: int,
    primary_offset: int,
    secondary_offset: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor = TRANSITION_AGENTS[index % 5]
    if semantic_label == "relocate":
        participants = {
            "actor": actor,
            "destination": TRANSITION_LOCATIONS[(index + primary_offset) % 5],
        }
    elif semantic_label == "acquire":
        participants = {
            "actor": actor,
            "object": TRANSITION_OBJECTS[(index + primary_offset) % 5],
        }
    elif semantic_label == "transfer":
        participants = {
            "actor": actor,
            "recipient": TRANSITION_AGENTS[(index + primary_offset) % 5],
            "object": TRANSITION_OBJECTS[(index + secondary_offset) % 5],
        }
    elif semantic_label == "exchange_locations":
        participants = {
            "actor": actor,
            "other": TRANSITION_AGENTS[(index + primary_offset) % 5],
        }
    elif semantic_label == "release":
        participants = {
            "actor": actor,
            "object": TRANSITION_OBJECTS[(index + primary_offset) % 5],
        }
    else:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    if len(set(participants.values())) != len(participants):
        raise ValueError("Evaluator participants must occupy distinct concepts")
    identity = transition_dataset_hash(
        [semantic_label, *sorted(participants.items())]
    )[:14]
    return _transition_episode(split, semantic_label, identity, participants)


def _build_split(
    split: str,
    specifications: Sequence[tuple[str, int, int, range]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}
    for semantic_label, primary_offset, secondary_offset, indices in specifications:
        for index in indices:
            observation, evaluation = _family_case(
                split,
                semantic_label,
                index,
                primary_offset,
                secondary_offset,
            )
            observations.append(observation)
            truth[str(observation["case_id"])] = evaluation
    return observations, truth


def _training_program() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth: dict[str, Any] = {}
    for round_index in (0, 1):
        specifications = (
            ("relocate", round_index + 1, 0, range(5)),
            ("acquire", round_index, 0, range(5)),
            ("transfer", round_index + 1, round_index + 2, range(5)),
            ("exchange_locations", round_index + 1, 0, range(5)),
            ("release", round_index + 3, 0, range(5)),
        )
        split_rows, split_truth = _build_split("train", specifications)
        rows.extend(split_rows)
        truth.update(split_truth)
    if len(rows) != 50:
        raise AssertionError("transition training split must contain 50 rows")
    return rows, truth


def build_transition_discovery_program() -> dict[str, Any]:
    train, training_truth = _training_program()
    validation, validation_truth = _build_split(
        "validation",
        (
            ("relocate", 4, 0, range(3)),
            ("acquire", 2, 0, range(3)),
            ("transfer", 3, 4, range(3)),
            ("exchange_locations", 4, 0, range(3)),
            ("release", 1, 0, range(3)),
        ),
    )
    heldout, evaluation_truth = _build_split(
        "heldout",
        (
            ("relocate", 3, 0, range(5)),
            ("acquire", 4, 0, range(5)),
            ("transfer", 4, 0, range(5)),
            ("exchange_locations", 3, 0, range(5)),
            ("release", 2, 0, range(5)),
        ),
    )
    if len(validation) != 15 or len(heldout) != 25:
        raise AssertionError("transition evaluation splits have incorrect sizes")
    all_rows = train + validation + heldout
    all_ids = [str(row["case_id"]) for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("transition case identifiers must be unique")
    forbidden = {"semantic_label", "participants", "family", "predicate", "roles"}
    if any(forbidden & set(row) for row in all_rows):
        raise AssertionError("runtime observations contain evaluator semantics")
    train_surfaces = {str(row["text"]) for row in train}
    validation_surfaces = {str(row["text"]) for row in validation}
    heldout_surfaces = {str(row["text"]) for row in heldout}
    if train_surfaces & validation_surfaces or train_surfaces & heldout_surfaces:
        raise AssertionError("evaluation utterances must be unseen during training")
    if validation_surfaces & heldout_surfaces:
        raise AssertionError("validation and held-out surfaces must be distinct")
    surface_vocabulary = sorted(
        {token for row in all_rows for token in str(row["text"]).split()}
    )
    manifest = {
        "seed": TRANSITION_DATASET_SEED,
        "counts": {"train": 50, "validation": 15, "heldout": 25},
        "entity_concepts": len(TRANSITION_LEXICON),
        "latent_law_count": len(TRANSITION_PATTERNS),
        "surface_vocabulary": surface_vocabulary,
        "semantic_labels_in_runtime_observations": False,
        "heldout_surface_overlap": len(train_surfaces & heldout_surfaces),
        "novel_transition_labels": ["exchange_locations", "release"],
        "hashes": {
            "train": transition_dataset_hash(train),
            "validation": transition_dataset_hash(validation),
            "heldout": transition_dataset_hash(heldout),
            "training_truth": transition_dataset_hash(training_truth),
            "validation_truth": transition_dataset_hash(validation_truth),
            "evaluation_truth": transition_dataset_hash(evaluation_truth),
        },
    }
    return {
        "train": train,
        "validation": validation,
        "heldout": heldout,
        "training_truth": training_truth,
        "validation_truth": validation_truth,
        "evaluation_truth": evaluation_truth,
        "manifest": manifest,
        "evaluator_oracle": {
            "concept_to_surface": dict(TRANSITION_LEXICON),
            "patterns": {
                label: list(pattern)
                for label, pattern in sorted(TRANSITION_PATTERNS.items())
            },
        },
    }

