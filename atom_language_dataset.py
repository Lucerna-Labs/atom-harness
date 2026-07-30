"""Deterministic grounded micro-world for the Atom Language Field experiment.

The runtime receives utterances, observable world states, emitted answers, and
salience.  Gold meaning frames and the surface-to-world permutation remain in
the evaluator-owned portion of the returned program.
"""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from typing import Any, Mapping


LANGUAGE_DATASET_SCHEMA = 1
LANGUAGE_SEED = 20260721

AGENTS = tuple(f"agent-{index}" for index in range(4))
OBJECTS = tuple(f"object-{index}" for index in range(3))
LOCATIONS = tuple(f"location-{index}" for index in range(3))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _surface_lexicon(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    agent_words = ["mira", "nora", "selo", "tavi"]
    object_words = ["key", "orb", "seed"]
    location_words = ["gate", "hill", "pond"]
    rng.shuffle(agent_words)
    rng.shuffle(object_words)
    rng.shuffle(location_words)
    return {
        **dict(zip(AGENTS, agent_words, strict=True)),
        **dict(zip(OBJECTS, object_words, strict=True)),
        **dict(zip(LOCATIONS, location_words, strict=True)),
    }


def _world() -> dict[str, Any]:
    return {
        "locations": {
            agent: LOCATIONS[index % len(LOCATIONS)]
            for index, agent in enumerate(AGENTS)
        },
        "holders": {item: None for item in OBJECTS},
    }


def _frame(
    speech_act: str,
    predicate: str,
    **roles: str,
) -> dict[str, Any]:
    return {
        "speech_act": speech_act,
        "predicate": predicate,
        "polarity": True,
        "roles": dict(sorted(roles.items())),
    }


def _episode(
    family: str,
    identity: str,
    text: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    answer_text: str | None = None,
    context_text: str | None = None,
    paraphrase_text: str | None = None,
    salience: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = f"lang-{stable_hash([family, identity])[:16]}"
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
    evaluation = {
        "family": family,
        "frame": deepcopy(dict(truth)),
        "expected_after": deepcopy(dict(after)),
        "expected_answer": answer_text,
    }
    return observation, evaluation


def _build_base_cases(
    lexicon: Mapping[str, str],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def add(family: str, identity: str, *args: Any, **kwargs: Any) -> None:
        observation, truth = _episode(family, identity, *args, **kwargs)
        rows.append((family, observation, truth))

    for agent_index, agent in enumerate(AGENTS):
        aw = lexicon[agent]
        for location_index, location in enumerate(LOCATIONS):
            lw = lexicon[location]
            before = _world()
            before["locations"][agent] = LOCATIONS[(location_index + 1) % 3]
            after = deepcopy(before)
            after["locations"][agent] = location
            add(
                "move",
                f"{agent}-{location}",
                f"move {aw} to {lw}",
                before,
                after,
                _frame("command", "MOVE", agent=agent, destination=location),
            )

            state = _world()
            state["locations"][agent] = location
            add(
                "assert_at",
                f"{agent}-{location}",
                f"{aw} is at {lw}",
                state,
                state,
                _frame("assertion", "AT", agent=agent, destination=location),
            )
            add(
                "where",
                f"{agent}-{location}",
                f"where is {aw}",
                state,
                state,
                _frame("question", "WHERE", agent=agent),
                answer_text=f"{aw} is at {lw}",
            )
            add(
                "dialog_where",
                f"{agent}-{location}",
                "where are they",
                state,
                state,
                _frame("question", "WHERE", agent=agent),
                context_text=f"{aw} is at {lw}",
                answer_text=f"{aw} is at {lw}",
            )
            add(
                "dialog_at_query",
                f"{agent}-{location}",
                f"are they at {lw}",
                state,
                state,
                _frame(
                    "question",
                    "AT_QUERY",
                    agent=agent,
                    destination=location,
                ),
                context_text=f"{aw} is at {lw}",
                paraphrase_text=f"is {aw} at {lw}",
                answer_text="yes",
            )

        for item in OBJECTS:
            ow = lexicon[item]
            before = _world()
            after = deepcopy(before)
            after["holders"][item] = agent
            add(
                "take",
                f"{agent}-{item}",
                f"{aw} take {ow}",
                before,
                after,
                _frame("command", "TAKE", agent=agent, patient=item),
            )

            state = _world()
            state["holders"][item] = agent
            add(
                "assert_has",
                f"{agent}-{item}",
                f"{aw} holds {ow}",
                state,
                state,
                _frame("assertion", "HAS", agent=agent, patient=item),
            )
            add(
                "what_has",
                f"{agent}-{item}",
                f"what does {aw} hold",
                state,
                state,
                _frame("question", "WHAT_HAS", agent=agent),
                answer_text=f"{aw} holds {ow}",
            )
            add(
                "dialog_who",
                f"{agent}-{item}",
                "who holds it",
                state,
                state,
                _frame("question", "WHO_HAS", patient=item),
                context_text=f"{aw} holds {ow}",
                paraphrase_text=f"who holds {ow}",
                answer_text=f"{aw} holds {ow}",
            )

            true_state = deepcopy(state)
            add(
                "has_query",
                f"true-{agent}-{item}",
                f"does {aw} hold {ow}",
                true_state,
                true_state,
                _frame("question", "HAS_QUERY", agent=agent, patient=item),
                answer_text="yes",
            )
            false_state = _world()
            false_state["holders"][item] = AGENTS[(agent_index + 1) % len(AGENTS)]
            add(
                "has_query",
                f"false-{agent}-{item}",
                f"does {aw} hold {ow}",
                false_state,
                false_state,
                _frame("question", "HAS_QUERY", agent=agent, patient=item),
                answer_text="no",
            )

    for agent in AGENTS:
        aw = lexicon[agent]
        for item in OBJECTS:
            ow = lexicon[item]
            for recipient in AGENTS:
                if recipient == agent:
                    continue
                rw = lexicon[recipient]
                before = _world()
                before["holders"][item] = agent
                after = deepcopy(before)
                after["holders"][item] = recipient
                add(
                    "give",
                    f"{agent}-{item}-{recipient}",
                    f"{aw} give {ow} to {rw}",
                    before,
                    after,
                    _frame(
                        "command",
                        "GIVE",
                        agent=agent,
                        patient=item,
                        recipient=recipient,
                    ),
                )

    if len(rows) != 168:
        raise AssertionError(f"Expected 168 base/dialog cases, found {len(rows)}")
    return rows


def _split_base_cases(
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    seed: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    allocations = {
        "move": (7, 2, 3),
        "take": (7, 2, 3),
        "assert_at": (7, 2, 3),
        "assert_has": (7, 2, 3),
        "where": (7, 2, 3),
        "what_has": (7, 2, 3),
        "dialog_where": (7, 2, 3),
        "dialog_who": (7, 2, 3),
        "dialog_at_query": (7, 2, 3),
        "give": (20, 5, 11),
        "has_query": (13, 1, 10),
    }
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for family, observation, truth in rows:
        grouped.setdefault(family, []).append((observation, truth))

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    truth_map: dict[str, Any] = {}
    for family, counts in allocations.items():
        family_rows = grouped[family]
        family_rows.sort(key=lambda row: stable_hash([seed, family, row[0]["case_id"]]))
        train_count, validation_count, heldout_count = counts
        if train_count + validation_count + heldout_count != len(family_rows):
            raise AssertionError(f"Bad allocation for {family}")
        train.extend(row[0] for row in family_rows[:train_count])
        validation.extend(
            row[0] for row in family_rows[train_count : train_count + validation_count]
        )
        heldout.extend(row[0] for row in family_rows[-heldout_count:])
        for observation, truth in family_rows:
            truth_map[observation["case_id"]] = truth
    return train, validation, heldout, truth_map


def _training_disturbances(
    lexicon: Mapping[str, str], seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    false_agent = AGENTS[0]
    actual_agent = AGENTS[1]
    surface = lexicon[false_agent]
    for index in range(4):
        location = LOCATIONS[index % len(LOCATIONS)]
        before = _world()
        before["locations"][actual_agent] = LOCATIONS[(index + 1) % 3]
        after = deepcopy(before)
        after["locations"][actual_agent] = location
        observation, _ = _episode(
            "correction-low",
            str(index),
            f"move {surface} to {lexicon[location]}",
            before,
            after,
            _frame("command", "MOVE", agent=actual_agent, destination=location),
            salience=0.2,
        )
        rows.append(observation)

    for index in range(8):
        location = LOCATIONS[index % len(LOCATIONS)]
        before = _world()
        before["locations"][false_agent] = LOCATIONS[(index + 1) % 3]
        after = deepcopy(before)
        after["locations"][false_agent] = location
        observation, _ = _episode(
            "correction-high",
            str(index),
            f"move {surface} to {lexicon[location]}",
            before,
            after,
            _frame("command", "MOVE", agent=false_agent, destination=location),
            salience=1.4,
        )
        rows.append(observation)

    rng = random.Random(seed + 99)
    for index in range(12):
        agent = AGENTS[index % len(AGENTS)]
        location = LOCATIONS[(index + 1) % len(LOCATIONS)]
        before = _world()
        after = deepcopy(before)
        after["locations"][agent] = location
        noise = "".join(rng.choice("qvxz") for _ in range(5)) + str(index)
        observation, _ = _episode(
            "one-off-noise",
            str(index),
            f"{noise} {lexicon[agent]} {lexicon[location]}",
            before,
            after,
            _frame("command", "MOVE", agent=agent, destination=location),
            salience=0.18,
        )
        rows.append(observation)
    if len(rows) != 24:
        raise AssertionError("Training disturbances must contain 24 rows")
    return rows


def build_grounded_language_program(seed: int = LANGUAGE_SEED) -> dict[str, Any]:
    lexicon = _surface_lexicon(seed)
    base = _build_base_cases(lexicon)
    train, validation, heldout, truth = _split_base_cases(base, seed)
    train.extend(_training_disturbances(lexicon, seed))
    train.sort(key=lambda row: stable_hash([seed, "train", row["case_id"]]))
    validation.sort(key=lambda row: stable_hash([seed, "validation", row["case_id"]]))
    heldout.sort(key=lambda row: stable_hash([seed, "heldout", row["case_id"]]))
    if (len(train), len(validation), len(heldout)) != (120, 24, 48):
        raise AssertionError("Language splits must be exactly 120/24/48")
    split_ids = {
        name: {row["case_id"] for row in split}
        for name, split in (
            ("train", train),
            ("validation", validation),
            ("heldout", heldout),
        )
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["heldout"]
        or split_ids["validation"] & split_ids["heldout"]
    ):
        raise AssertionError("Language split case IDs overlap")
    return {
        "schema_version": LANGUAGE_DATASET_SCHEMA,
        "seed": seed,
        "train": train,
        "validation": validation,
        "heldout": heldout,
        "evaluation_truth": truth,
        "evaluator_oracle": {
            "concept_to_surface": dict(sorted(lexicon.items())),
            "surface_to_concept": {
                surface: concept for concept, surface in sorted(lexicon.items())
            },
        },
        "manifest": {
            "counts": {"train": 120, "validation": 24, "heldout": 48},
            "hashes": {
                "train": stable_hash(train),
                "validation": stable_hash(validation),
                "heldout": stable_hash(heldout),
                "truth": stable_hash(truth),
            },
            "latent_world": {
                "agents": len(AGENTS),
                "objects": len(OBJECTS),
                "locations": len(LOCATIONS),
            },
            "training_has_gold_frames": False,
        },
    }
