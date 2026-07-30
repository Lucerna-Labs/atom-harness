"""Opaque worlds for alias-invariant ontology and transition-law discovery.

The runtime receives relation tables, entity identifiers, utterances, and
before/after consequences.  Relation aliases change across every split and
entity identifiers contain no evaluator type prefixes.  Human labels remain in
separate evaluator maps and are never passed to the learner.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


ONTOLOGY_DATASET_SEED = 2_026_072_102
ONTOLOGY_ENTITIES = tuple(f"n{index:02d}" for index in range(15))
ONTOLOGY_ACTORS = ONTOLOGY_ENTITIES[:5]
ONTOLOGY_ITEMS = ONTOLOGY_ENTITIES[5:10]
ONTOLOGY_PLACES = ONTOLOGY_ENTITIES[10:]
ONTOLOGY_LEXICON = {
    "n00": "vra",
    "n01": "kel",
    "n02": "sot",
    "n03": "miv",
    "n04": "dan",
    "n05": "pax",
    "n06": "gul",
    "n07": "tem",
    "n08": "zor",
    "n09": "bik",
    "n10": "fen",
    "n11": "ruk",
    "n12": "lom",
    "n13": "cid",
    "n14": "wex",
}
ONTOLOGY_PATTERNS = {
    "relocate": ("{destination}", "uja", "{actor}"),
    "acquire": ("{object}", "{actor}", "bek"),
    "transfer": ("{recipient}", "toh", "{object}", "{actor}"),
    "exchange_locations": ("{other}", "iry", "{actor}"),
    "release": ("{object}", "nop", "{actor}"),
}
ONTOLOGY_RELATION_ALIASES = {
    "train": {"total": "q7x", "nullable": "v2m"},
    "validation": {"total": "z4u", "nullable": "a9k"},
    "heldout": {"total": "b3x", "nullable": "y8o"},
}


def ontology_dataset_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def opaque_world(split: str) -> dict[str, dict[str, str | None]]:
    aliases = ONTOLOGY_RELATION_ALIASES.get(split)
    if aliases is None:
        raise ValueError(f"Unknown dataset split: {split}")
    return {
        aliases["total"]: {
            actor: ONTOLOGY_PLACES[index] for index, actor in enumerate(ONTOLOGY_ACTORS)
        },
        aliases["nullable"]: {item: None for item in ONTOLOGY_ITEMS},
    }


def render_ontology_transition(
    semantic_label: str,
    participants: Mapping[str, str],
) -> str:
    pattern = ONTOLOGY_PATTERNS.get(semantic_label)
    if pattern is None:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    output: list[str] = []
    for piece in pattern:
        if piece.startswith("{") and piece.endswith("}"):
            role = piece[1:-1]
            concept = participants.get(role)
            surface = ONTOLOGY_LEXICON.get(str(concept))
            if surface is None:
                raise ValueError(f"Missing opaque surface for evaluator role {role}")
            output.append(surface)
        else:
            output.append(piece)
    return " ".join(output)


def apply_ontology_evaluator_transition(
    semantic_label: str,
    participants: Mapping[str, str],
    before: Mapping[str, Mapping[str, str | None]],
    split: str,
) -> dict[str, dict[str, str | None]]:
    aliases = ONTOLOGY_RELATION_ALIASES.get(split)
    if aliases is None:
        raise ValueError(f"Unknown dataset split: {split}")
    after = deepcopy(dict(before))
    total = aliases["total"]
    nullable = aliases["nullable"]
    if semantic_label == "relocate":
        after[total][participants["actor"]] = participants["destination"]
    elif semantic_label == "acquire":
        after[nullable][participants["object"]] = participants["actor"]
    elif semantic_label == "transfer":
        item = participants["object"]
        actor = participants["actor"]
        if before[nullable][item] != actor:
            raise ValueError("Evaluator transfer requires the actor to hold the item")
        after[nullable][item] = participants["recipient"]
    elif semantic_label == "exchange_locations":
        actor = participants["actor"]
        other = participants["other"]
        after[total][actor] = before[total][other]
        after[total][other] = before[total][actor]
    elif semantic_label == "release":
        item = participants["object"]
        actor = participants["actor"]
        if before[nullable][item] != actor:
            raise ValueError("Evaluator release requires the actor to hold the item")
        after[nullable][item] = None
    else:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    return after


def _ontology_episode(
    split: str,
    semantic_label: str,
    identity: str,
    participants: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = ONTOLOGY_RELATION_ALIASES[split]
    before = opaque_world(split)
    if semantic_label == "relocate":
        actor = participants["actor"]
        destination = participants["destination"]
        if before[aliases["total"]][actor] == destination:
            current = ONTOLOGY_PLACES.index(destination)
            before[aliases["total"]][actor] = ONTOLOGY_PLACES[(current + 1) % 5]
    elif semantic_label in {"transfer", "release"}:
        before[aliases["nullable"]][participants["object"]] = participants["actor"]
    after = apply_ontology_evaluator_transition(
        semantic_label,
        participants,
        before,
        split,
    )
    case_id = (
        "ontology-episode-"
        + ontology_dataset_hash([split, semantic_label, identity])[:18]
    )
    observation = {
        "case_id": case_id,
        "text": render_ontology_transition(semantic_label, participants),
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
    actor = ONTOLOGY_ACTORS[index % 5]
    if semantic_label == "relocate":
        participants = {
            "actor": actor,
            "destination": ONTOLOGY_PLACES[(index + primary_offset) % 5],
        }
    elif semantic_label == "acquire":
        participants = {
            "actor": actor,
            "object": ONTOLOGY_ITEMS[(index + primary_offset) % 5],
        }
    elif semantic_label == "transfer":
        participants = {
            "actor": actor,
            "recipient": ONTOLOGY_ACTORS[(index + primary_offset) % 5],
            "object": ONTOLOGY_ITEMS[(index + secondary_offset) % 5],
        }
    elif semantic_label == "exchange_locations":
        participants = {
            "actor": actor,
            "other": ONTOLOGY_ACTORS[(index + primary_offset) % 5],
        }
    elif semantic_label == "release":
        participants = {
            "actor": actor,
            "object": ONTOLOGY_ITEMS[(index + primary_offset) % 5],
        }
    else:
        raise ValueError(f"Unknown evaluator transition: {semantic_label}")
    if len(set(participants.values())) != len(participants):
        raise ValueError("Evaluator participants must be distinct")
    identity = ontology_dataset_hash([semantic_label, *sorted(participants.items())])[
        :14
    ]
    return _ontology_episode(split, semantic_label, identity, participants)


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
        split_rows, split_truth = _build_split(
            "train",
            (
                ("relocate", round_index + 1, 0, range(5)),
                ("acquire", round_index, 0, range(5)),
                ("transfer", round_index + 1, round_index + 2, range(5)),
                ("exchange_locations", round_index + 1, 0, range(5)),
                ("release", round_index + 3, 0, range(5)),
            ),
        )
        rows.extend(split_rows)
        truth.update(split_truth)
    if len(rows) != 50:
        raise AssertionError("ontology training split must contain 50 rows")
    return rows, truth


def build_ontology_discovery_program() -> dict[str, Any]:
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
    all_rows = train + validation + heldout
    all_ids = [str(row["case_id"]) for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("ontology case identifiers must be unique")
    forbidden = {"semantic_label", "participants", "family", "predicate", "roles"}
    if any(forbidden & set(row) for row in all_rows):
        raise AssertionError("runtime observations contain evaluator semantics")
    aliases = {
        split: frozenset(values.values())
        for split, values in ONTOLOGY_RELATION_ALIASES.items()
    }
    if any(
        aliases[left] & aliases[right]
        for left, right in (
            ("train", "validation"),
            ("train", "heldout"),
            ("validation", "heldout"),
        )
    ):
        raise AssertionError("relation aliases must be disjoint across splits")
    train_surfaces = {str(row["text"]) for row in train}
    validation_surfaces = {str(row["text"]) for row in validation}
    heldout_surfaces = {str(row["text"]) for row in heldout}
    if train_surfaces & validation_surfaces or train_surfaces & heldout_surfaces:
        raise AssertionError("evaluation utterances must be unseen during training")
    if validation_surfaces & heldout_surfaces:
        raise AssertionError("evaluation surfaces must be mutually distinct")
    typed_prefixes = ("agent-", "object-", "location-")
    if any(entity.startswith(typed_prefixes) for entity in ONTOLOGY_ENTITIES):
        raise AssertionError("entity identifiers must not reveal evaluator types")
    manifest = {
        "seed": ONTOLOGY_DATASET_SEED,
        "counts": {"train": 50, "validation": 15, "heldout": 25},
        "entity_count": len(ONTOLOGY_ENTITIES),
        "latent_law_count": len(ONTOLOGY_PATTERNS),
        "relation_aliases": {
            split: sorted(values) for split, values in sorted(aliases.items())
        },
        "relation_alias_overlap": 0,
        "typed_entity_prefixes_present": False,
        "semantic_labels_in_runtime_observations": False,
        "heldout_surface_overlap": len(train_surfaces & heldout_surfaces),
        "novel_transition_labels": ["exchange_locations", "release"],
        "hashes": {
            "train": ontology_dataset_hash(train),
            "validation": ontology_dataset_hash(validation),
            "heldout": ontology_dataset_hash(heldout),
            "training_truth": ontology_dataset_hash(training_truth),
            "validation_truth": ontology_dataset_hash(validation_truth),
            "evaluation_truth": ontology_dataset_hash(evaluation_truth),
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
            "concept_to_surface": dict(ONTOLOGY_LEXICON),
            "patterns": {
                label: list(pattern)
                for label, pattern in sorted(ONTOLOGY_PATTERNS.items())
            },
        },
    }
