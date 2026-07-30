"""External English benchmarks and an explicit competence boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from atom_english_core import (
    ATOM_LONG_CONTEXT_MILESTONES,
    AtomCausalLanguageModel,
)

ATOM_ENGLISH_EVALUATION_RUNTIME = "atom-external-language-evaluation-v1"
ATOM_LONG_CONTEXT_EVALUATION_RUNTIME = "atom-ruler-style-context-evaluation-v1"
LONG_CONTEXT_TASK_FAMILIES = (
    "single_needle",
    "multi_needle",
    "sequence_order",
    "state_update",
)


def _canonical_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LanguageGateThresholds:
    maximum_wikitext_perplexity: float = 45.0
    minimum_lambada_exact_accuracy: float = 0.28
    minimum_blimp_accuracy: float = 0.67
    minimum_hellaswag_accuracy: float = 0.36
    minimum_ifeval_strict_prompt_accuracy: float = 0.45
    minimum_ifeval_loose_prompt_accuracy: float = 0.55
    minimum_benchmark_examples: int = 500
    minimum_long_context_accuracy: float = 0.75
    minimum_long_context_examples_per_length: int = 4
    minimum_long_context_task_families: int = 4

    def __post_init__(self) -> None:
        if self.maximum_wikitext_perplexity <= 1.0:
            raise ValueError("perplexity threshold must exceed one")
        values = (
            self.minimum_lambada_exact_accuracy,
            self.minimum_blimp_accuracy,
            self.minimum_hellaswag_accuracy,
            self.minimum_ifeval_strict_prompt_accuracy,
            self.minimum_ifeval_loose_prompt_accuracy,
            self.minimum_long_context_accuracy,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("accuracy thresholds must be inside [0, 1]")
        if self.minimum_benchmark_examples < 100:
            raise ValueError("benchmark minimum is too small")
        if self.minimum_long_context_examples_per_length < 4:
            raise ValueError("long-context example minimum is too small")
        if self.minimum_long_context_task_families < 4:
            raise ValueError("long-context task-family minimum is too small")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def collect_long_context_distractor_tokens(
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    maximum_tokens: int = 65_536,
) -> list[int]:
    if maximum_tokens < 4_096:
        raise ValueError("long-context distractor reservoir is too small")
    tokens: list[int] = []
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        tokens.extend(_token_ids(tokenizer, "\n" + text.strip()))
        if len(tokens) >= maximum_tokens:
            return tokens[:maximum_tokens]
    if len(tokens) < 1_024:
        raise ValueError("long-context distractor corpus produced too few tokens")
    return tokens


def _cycled_tokens(
    reservoir: Sequence[int],
    count: int,
    *,
    offset: int,
) -> list[int]:
    if count < 0:
        raise ValueError("cycled token count cannot be negative")
    if not reservoir:
        raise ValueError("long-context distractor reservoir is empty")
    size = len(reservoir)
    return [int(reservoir[(offset + index) % size]) for index in range(count)]


def build_long_context_probe(
    tokenizer: Any,
    reservoir: Sequence[int],
    *,
    prompt_tokens: int,
    family: str,
    seed: int,
) -> tuple[list[int], str]:
    suffix = f"{seed:08X}"
    if family == "single_needle":
        records = [
            (
                "Memorize this unique archive record: "
                f"ORBIT-{suffix} is the value for key amber.\n"
            )
        ]
        query = (
            "\nQuestion: What is the value for key amber? "
            "Return only the exact value.\nAnswer:"
        )
        expected = f"ORBIT-{suffix}"
    elif family == "multi_needle":
        records = [
            f"Archive key cedar has value CEDAR-{suffix}.\n",
            f"Archive key birch has value BIRCH-{suffix}.\n",
            f"Archive key maple has value MAPLE-{suffix}.\n",
        ]
        query = (
            "\nQuestion: What is the value for archive key birch? "
            "Return only the exact value.\nAnswer:"
        )
        expected = f"BIRCH-{suffix}"
    elif family == "sequence_order":
        records = [
            f"Event one carries code FIRST-{suffix}.\n",
            f"Event two carries code SECOND-{suffix}.\n",
            f"Event three carries code THIRD-{suffix}.\n",
        ]
        query = (
            "\nQuestion: Which exact code belongs to event two? "
            "Return only that code.\nAnswer:"
        )
        expected = f"SECOND-{suffix}"
    elif family == "state_update":
        records = [
            f"Register quartz is assigned OLD-{suffix}.\n",
            f"Register quartz is changed to MIDDLE-{suffix}.\n",
            f"Register quartz is finally changed to FINAL-{suffix}.\n",
        ]
        query = (
            "\nQuestion: What is the final value of register quartz? "
            "Return only the exact final value.\nAnswer:"
        )
        expected = f"FINAL-{suffix}"
    else:
        raise ValueError(f"unknown long-context task family: {family}")
    prefix = (
        "Read the entire record stream. Preserve exact relationships and obey "
        "the final question. Unrelated prose is only background.\n"
    )
    fixed_segments = [_token_ids(tokenizer, prefix)]
    fixed_segments.extend(_token_ids(tokenizer, record) for record in records)
    query_ids = _token_ids(tokenizer, query)
    fixed_count = sum(len(segment) for segment in fixed_segments) + len(query_ids)
    filler_count = prompt_tokens - fixed_count
    if filler_count < len(records) + 1:
        raise ValueError("long-context probe length is too short")
    gaps = len(records) + 1
    base_gap, remainder = divmod(filler_count, gaps)
    gap_sizes = [base_gap + (1 if index < remainder else 0) for index in range(gaps)]
    result = list(fixed_segments[0])
    reservoir_offset = seed % len(reservoir)
    for index, record_ids in enumerate(fixed_segments[1:]):
        result.extend(
            _cycled_tokens(
                reservoir,
                gap_sizes[index],
                offset=reservoir_offset,
            )
        )
        reservoir_offset = (reservoir_offset + gap_sizes[index]) % len(reservoir)
        result.extend(record_ids)
    result.extend(
        _cycled_tokens(
            reservoir,
            gap_sizes[-1],
            offset=reservoir_offset,
        )
    )
    result.extend(query_ids)
    if len(result) != prompt_tokens:
        raise AssertionError("long-context probe construction changed token length")
    return result, expected


def _normalized_exact_value(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


@torch.inference_mode()
def evaluate_long_context(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    distractor_tokens: Sequence[int],
    *,
    device: torch.device,
    context_lengths: Sequence[int] = ATOM_LONG_CONTEXT_MILESTONES,
    examples_per_length: int = 4,
    maximum_new_tokens: int = 24,
) -> dict[str, Any]:
    """Exercise retrieval, ordering, and state updates at every declared length."""

    if examples_per_length < len(LONG_CONTEXT_TASK_FAMILIES):
        raise ValueError("every long-context task family must be exercised")
    if maximum_new_tokens < 8:
        raise ValueError("long-context answer budget is too small")
    requested = tuple(int(length) for length in context_lengths)
    if requested != tuple(ATOM_LONG_CONTEXT_MILESTONES):
        raise ValueError("long-context evaluation lengths changed")
    if requested[-1] > model.config.max_seq_len:
        raise ValueError("model context is below the long-context target")
    model.eval()
    lengths: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []
    total_correct = 0
    total_examples = 0
    for length_index, context_length in enumerate(requested):
        prompt_length = context_length - maximum_new_tokens
        correct = 0
        samples: list[dict[str, Any]] = []
        families: set[str] = set()
        for example_index in range(examples_per_length):
            family = LONG_CONTEXT_TASK_FAMILIES[
                example_index % len(LONG_CONTEXT_TASK_FAMILIES)
            ]
            seed = 20_260_724 + length_index * 10_000 + example_index * 101
            prompt, expected = build_long_context_probe(
                tokenizer,
                distractor_tokens,
                prompt_tokens=prompt_length,
                family=family,
                seed=seed,
            )
            input_ids = torch.tensor(
                prompt,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            generated = model.generate(
                input_ids,
                max_new_tokens=maximum_new_tokens,
                temperature=0.2,
                top_p=1.0,
                top_k=1,
                repetition_penalty=1.0,
                eos_token_id=int(tokenizer.eos_token_id),
                seed=seed,
            )
            response_ids = generated[0, input_ids.shape[1] :].tolist()
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()
            passed = _normalized_exact_value(expected) in _normalized_exact_value(
                response
            )
            correct += int(passed)
            families.add(family)
            sample = {
                "context_tokens": context_length,
                "prompt_tokens": prompt_length,
                "family": family,
                "expected": expected,
                "response": response,
                "correct": passed,
            }
            samples.append(sample)
            all_samples.append(sample)
        lengths[str(context_length)] = {
            "examples": examples_per_length,
            "correct": correct,
            "accuracy": correct / examples_per_length,
            "task_families": sorted(families),
            "samples": samples,
        }
        total_correct += correct
        total_examples += examples_per_length
    return {
        "runtime": ATOM_LONG_CONTEXT_EVALUATION_RUNTIME,
        "context_lengths": list(requested),
        "task_families": list(LONG_CONTEXT_TASK_FAMILIES),
        "examples": total_examples,
        "correct": total_correct,
        "accuracy": total_correct / total_examples,
        "lengths": lengths,
        "samples": all_samples,
    }


@torch.inference_mode()
def completion_score(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    context: str,
    completion: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    context_ids = _token_ids(tokenizer, context)
    full_ids = _token_ids(tokenizer, context + completion)
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError("completion tokenization is not prefix stable")
    if len(full_ids) < 2 or len(full_ids) > model.config.max_seq_len:
        raise ValueError("completion is outside the model context")
    target_start = max(len(context_ids), 1)
    input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(
        0
    )
    output = model(input_ids)
    log_probabilities = torch.log_softmax(output.logits[0].float(), dim=-1)
    target_ids = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
    positions = torch.arange(target_ids.shape[0], device=device)
    token_scores = log_probabilities[positions, target_ids]
    start_prediction = target_start - 1
    selected = token_scores[start_prediction:]
    if selected.numel() == 0:
        raise ValueError("completion has no target tokens")
    greedy = output.logits[0, start_prediction:].argmax(dim=-1)
    truth = target_ids[start_prediction:]
    return {
        "token_count": int(selected.numel()),
        "total_log_probability": float(selected.sum().item()),
        "mean_log_probability": float(selected.mean().item()),
        "exact_greedy": bool(torch.equal(greedy, truth)),
    }


def evaluate_lambada(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    maximum_examples: int | None = None,
) -> dict[str, Any]:
    attempted = 0
    scored = 0
    exact = 0
    total_log_probability = 0.0
    total_tokens = 0
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or " " not in text.strip():
            continue
        attempted += 1
        prefix, separator, final_word = text.strip().rpartition(" ")
        if not separator:
            continue
        try:
            score = completion_score(
                model,
                tokenizer,
                prefix + " ",
                final_word,
                device=device,
            )
        except ValueError:
            continue
        scored += 1
        exact += int(score["exact_greedy"])
        total_log_probability += float(score["total_log_probability"])
        total_tokens += int(score["token_count"])
        if maximum_examples is not None and scored >= maximum_examples:
            break
    if scored == 0:
        raise ValueError("LAMBADA produced no scorable examples")
    return {
        "attempted": attempted,
        "examples": scored,
        "exact": exact,
        "exact_accuracy": exact / scored,
        "mean_target_log_probability": (total_log_probability / max(total_tokens, 1)),
    }


def evaluate_blimp(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    config_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    device: torch.device,
    maximum_examples_per_config: int | None = None,
) -> dict[str, Any]:
    config_results: dict[str, Any] = {}
    correct_total = 0
    example_total = 0
    for config_name, rows in config_rows.items():
        correct = 0
        examples = 0
        for row in rows:
            good = row.get("sentence_good")
            bad = row.get("sentence_bad")
            if not isinstance(good, str) or not isinstance(bad, str):
                continue
            try:
                good_score = completion_score(model, tokenizer, "", good, device=device)
                bad_score = completion_score(model, tokenizer, "", bad, device=device)
            except ValueError:
                continue
            correct += int(
                float(good_score["mean_log_probability"])
                > float(bad_score["mean_log_probability"])
            )
            examples += 1
            if (
                maximum_examples_per_config is not None
                and examples >= maximum_examples_per_config
            ):
                break
        if examples == 0:
            raise ValueError(f"BLiMP config has no examples: {config_name}")
        config_results[config_name] = {
            "examples": examples,
            "correct": correct,
            "accuracy": correct / examples,
        }
        correct_total += correct
        example_total += examples
    if not config_results:
        raise ValueError("BLiMP config set is empty")
    return {
        "configs": config_results,
        "config_count": len(config_results),
        "examples": example_total,
        "correct": correct_total,
        "accuracy": correct_total / example_total,
    }


def evaluate_hellaswag(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    maximum_examples: int | None = None,
) -> dict[str, Any]:
    examples = 0
    correct = 0
    for row in rows:
        context = row.get("ctx")
        endings = row.get("endings")
        if (
            not isinstance(context, str)
            or not isinstance(endings, Sequence)
            or isinstance(endings, (str, bytes))
        ):
            continue
        try:
            expected = int(row.get("label"))
        except (TypeError, ValueError):
            continue
        scores: list[float] = []
        for ending in endings:
            if not isinstance(ending, str):
                scores = []
                break
            try:
                score = completion_score(
                    model,
                    tokenizer,
                    context.strip() + " ",
                    ending.lstrip(),
                    device=device,
                )
            except ValueError:
                scores = []
                break
            scores.append(float(score["mean_log_probability"]))
        if not scores or not 0 <= expected < len(scores):
            continue
        prediction = max(range(len(scores)), key=scores.__getitem__)
        correct += int(prediction == expected)
        examples += 1
        if maximum_examples is not None and examples >= maximum_examples:
            break
    if examples == 0:
        raise ValueError("HellaSwag produced no scorable examples")
    return {
        "examples": examples,
        "correct": correct,
        "accuracy": correct / examples,
    }


@torch.inference_mode()
def evaluate_ifeval(
    model: AtomCausalLanguageModel,
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    maximum_examples: int | None,
    maximum_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Run the official deterministic IFEval constraint checkers."""

    from lm_eval.tasks.ifeval.utils import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    model.eval()
    examples = 0
    strict_prompts = 0
    loose_prompts = 0
    strict_instructions = 0
    loose_instructions = 0
    instruction_count = 0
    samples: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        instruction_ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")
        key = row.get("key")
        if (
            not isinstance(prompt, str)
            or not isinstance(instruction_ids, Sequence)
            or isinstance(instruction_ids, (str, bytes))
            or not isinstance(kwargs, Sequence)
            or isinstance(kwargs, (str, bytes))
        ):
            continue
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if isinstance(encoded, Tensor):
            encoded = encoded.flatten().tolist()
        if not isinstance(encoded, Sequence) or isinstance(
            encoded,
            (str, bytes),
        ):
            raise ValueError("IFEval prompt tokenization failed")
        generation_budget = min(
            maximum_new_tokens,
            model.config.max_seq_len - 1,
        )
        available = model.config.max_seq_len - generation_budget
        if available < 1:
            raise ValueError("IFEval generation budget exceeds model context")
        prompt_ids = [int(token) for token in encoded][-available:]
        input_ids = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        generated = model.generate(
            input_ids,
            max_new_tokens=generation_budget,
            temperature=0.2,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.05,
            eos_token_id=int(tokenizer.eos_token_id),
            seed=20260724 + examples,
        )
        response_ids = generated[0, input_ids.shape[1] :].tolist()
        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()
        example = InputExample(
            key=int(key),
            instruction_id_list=[str(value) for value in instruction_ids],
            prompt=prompt,
            kwargs=[dict(value) for value in kwargs],
        )
        strict = test_instruction_following_strict(example, response)
        loose = test_instruction_following_loose(example, response)
        strict_prompts += int(strict.follow_all_instructions)
        loose_prompts += int(loose.follow_all_instructions)
        strict_instructions += sum(strict.follow_instruction_list)
        loose_instructions += sum(loose.follow_instruction_list)
        instruction_count += len(strict.follow_instruction_list)
        if len(samples) < 20:
            samples.append(
                {
                    "key": int(key),
                    "prompt": prompt,
                    "response": response,
                    "strict": strict.follow_all_instructions,
                    "loose": loose.follow_all_instructions,
                }
            )
        examples += 1
        if maximum_examples is not None and examples >= maximum_examples:
            break
    if examples == 0 or instruction_count == 0:
        raise ValueError("IFEval produced no scorable examples")
    return {
        "examples": examples,
        "instructions": instruction_count,
        "strict_prompt_accuracy": strict_prompts / examples,
        "loose_prompt_accuracy": loose_prompts / examples,
        "strict_instruction_accuracy": (strict_instructions / instruction_count),
        "loose_instruction_accuracy": loose_instructions / instruction_count,
        "samples": samples,
    }


def language_competence_gate(
    benchmarks: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: LanguageGateThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or LanguageGateThresholds()
    required = {
        "wikitext",
        "lambada",
        "blimp",
        "hellaswag",
        "ifeval",
        "long_context",
    }
    if set(benchmarks) != required:
        raise ValueError("language gate benchmark set is incomplete")
    long_context = benchmarks["long_context"]
    expected_lengths = [str(value) for value in ATOM_LONG_CONTEXT_MILESTONES]
    reported_lengths = long_context.get("lengths")
    if not isinstance(reported_lengths, Mapping):
        raise ValueError("long-context length results are invalid")
    length_set_matches = set(reported_lengths) == set(expected_lengths)
    if not length_set_matches:
        raise ValueError("long-context length set is incomplete")
    task_families = long_context.get("task_families")
    if not isinstance(task_families, Sequence) or isinstance(
        task_families,
        (str, bytes),
    ):
        raise ValueError("long-context task families are invalid")
    checks = {
        "wikitext_perplexity": (
            float(benchmarks["wikitext"]["perplexity"])
            <= limits.maximum_wikitext_perplexity
        ),
        "lambada_exact_accuracy": (
            float(benchmarks["lambada"]["exact_accuracy"])
            >= limits.minimum_lambada_exact_accuracy
        ),
        "blimp_accuracy": (
            float(benchmarks["blimp"]["accuracy"]) >= limits.minimum_blimp_accuracy
        ),
        "hellaswag_accuracy": (
            float(benchmarks["hellaswag"]["accuracy"])
            >= limits.minimum_hellaswag_accuracy
        ),
        "ifeval_strict_prompt_accuracy": (
            float(benchmarks["ifeval"]["strict_prompt_accuracy"])
            >= limits.minimum_ifeval_strict_prompt_accuracy
        ),
        "ifeval_loose_prompt_accuracy": (
            float(benchmarks["ifeval"]["loose_prompt_accuracy"])
            >= limits.minimum_ifeval_loose_prompt_accuracy
        ),
        "lambada_scale": (
            int(benchmarks["lambada"]["examples"]) >= limits.minimum_benchmark_examples
        ),
        "blimp_scale": (
            int(benchmarks["blimp"]["examples"]) >= limits.minimum_benchmark_examples
        ),
        "hellaswag_scale": (
            int(benchmarks["hellaswag"]["examples"])
            >= limits.minimum_benchmark_examples
        ),
        "ifeval_scale": (
            int(benchmarks["ifeval"]["examples"]) >= limits.minimum_benchmark_examples
        ),
        "long_context_runtime": (
            long_context.get("runtime") == ATOM_LONG_CONTEXT_EVALUATION_RUNTIME
        ),
        "long_context_length_set": length_set_matches,
        "long_context_task_families": (
            len(set(str(value) for value in task_families))
            >= limits.minimum_long_context_task_families
            and set(LONG_CONTEXT_TASK_FAMILIES).issubset(
                set(str(value) for value in task_families)
            )
        ),
    }
    for length in expected_lengths:
        row = reported_lengths.get(length)
        checks[f"long_context_{length}_scale"] = (
            isinstance(row, Mapping)
            and int(row.get("examples", 0))
            >= limits.minimum_long_context_examples_per_length
        )
        checks[f"long_context_{length}_accuracy"] = (
            isinstance(row, Mapping)
            and float(row.get("accuracy", -1.0)) >= limits.minimum_long_context_accuracy
        )
    return {
        "thresholds": asdict(limits),
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_language_evaluation(
    output_directory: Path,
    benchmarks: Mapping[str, Mapping[str, Any]],
    *,
    model_parameter_count: int,
    checkpoint_manifest_hash: str,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    if (
        not isinstance(checkpoint_manifest_hash, str)
        or len(checkpoint_manifest_hash) != 64
    ):
        raise ValueError("checkpoint manifest hash is invalid")
    gate = language_competence_gate(benchmarks)
    body = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_EVALUATION_RUNTIME,
        "checkpoint_manifest_hash": checkpoint_manifest_hash,
        "benchmarks": dict(benchmarks),
        "gate": gate,
        "model_parameter_count": model_parameter_count,
        "sources": dict(sources),
    }
    report = dict(body)
    report["report_hash"] = _canonical_hash(body)
    (output_directory / "language_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_language_evaluation(
    path: Path,
    *,
    checkpoint_manifest_hash: str,
    model_parameter_count: int,
    require_gate: bool = True,
) -> dict[str, Any]:
    """Load a hash-bound external evaluation and fail closed on mismatch."""

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "runtime",
        "checkpoint_manifest_hash",
        "benchmarks",
        "gate",
        "model_parameter_count",
        "sources",
        "report_hash",
    }
    if set(report) != expected:
        raise ValueError("language evaluation fields are invalid")
    if report["schema_version"] != 1:
        raise ValueError("unsupported language evaluation schema")
    if report["runtime"] != ATOM_ENGLISH_EVALUATION_RUNTIME:
        raise ValueError("language evaluation runtime is invalid")
    body = dict(report)
    report_hash = body.pop("report_hash")
    if _canonical_hash(body) != report_hash:
        raise ValueError("language evaluation report hash mismatch")
    if report["checkpoint_manifest_hash"] != checkpoint_manifest_hash:
        raise ValueError("language evaluation is bound to another checkpoint")
    if int(report["model_parameter_count"]) != int(model_parameter_count):
        raise ValueError("language evaluation parameter count mismatch")
    recomputed = language_competence_gate(report["benchmarks"])
    if report["gate"] != recomputed:
        raise ValueError("language evaluation gate result is inconsistent")
    if require_gate and not report["gate"]["passed"]:
        raise ValueError("checkpoint has not passed the external language gate")
    return report


def atom_english_evaluation_self_test() -> dict[str, bool]:
    long_context_lengths = {
        str(length): {
            "examples": 4,
            "correct": 4,
            "accuracy": 1.0,
            "task_families": list(LONG_CONTEXT_TASK_FAMILIES),
            "samples": [],
        }
        for length in ATOM_LONG_CONTEXT_MILESTONES
    }
    passing = {
        "wikitext": {"perplexity": 30.0},
        "lambada": {"exact_accuracy": 0.4, "examples": 500},
        "blimp": {"accuracy": 0.72, "examples": 500},
        "hellaswag": {"accuracy": 0.4, "examples": 500},
        "ifeval": {
            "strict_prompt_accuracy": 0.5,
            "loose_prompt_accuracy": 0.6,
            "examples": 541,
        },
        "long_context": {
            "runtime": ATOM_LONG_CONTEXT_EVALUATION_RUNTIME,
            "context_lengths": list(ATOM_LONG_CONTEXT_MILESTONES),
            "task_families": list(LONG_CONTEXT_TASK_FAMILIES),
            "examples": 4 * len(ATOM_LONG_CONTEXT_MILESTONES),
            "correct": 4 * len(ATOM_LONG_CONTEXT_MILESTONES),
            "accuracy": 1.0,
            "lengths": long_context_lengths,
            "samples": [],
        },
    }
    failing = json.loads(json.dumps(passing))
    failing["lambada"]["exact_accuracy"] = 0.0
    return {
        "full_gate_passes": language_competence_gate(passing)["passed"],
        "single_failure_blocks": not language_competence_gate(failing)["passed"],
        "gate_requires_all_benchmarks": set(passing)
        == {
            "wikitext",
            "lambada",
            "blimp",
            "hellaswag",
            "ifeval",
            "long_context",
        },
    }
