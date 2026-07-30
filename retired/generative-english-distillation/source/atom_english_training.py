"""Resumable broad-English training and teacher distillation for Atom graphs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from atom_english_core import (
    ATOM_ENGLISH_CORE_RUNTIME,
    AtomCausalLanguageModel,
    AtomEnglishConfig,
    atom_english_architecture_manifest,
)

ATOM_ENGLISH_TRAINING_RUNTIME = "atom-english-distillation-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit_training_event(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {
                "runtime": ATOM_ENGLISH_TRAINING_RUNTIME,
                "event": event,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@dataclass(frozen=True)
class AtomTrainingConfig:
    stage: str
    teacher_model_id: str
    teacher_revision: str = "main"
    maximum_steps: int = 20_000
    maximum_wall_seconds: int = 38_000
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 3.0e-4
    minimum_learning_rate_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    hard_label_weight: float = 0.35
    distillation_weight: float = 0.65
    distillation_temperature: float = 2.0
    distillation_top_k: int = 32
    precision: str = "bfloat16"
    log_interval: int = 20
    save_interval: int = 500
    seed: int = 20260724
    schema_version: int = 1
    runtime: str = ATOM_ENGLISH_TRAINING_RUNTIME

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported Atom training schema")
        if self.runtime != ATOM_ENGLISH_TRAINING_RUNTIME:
            raise ValueError("unsupported Atom training runtime")
        if self.stage not in {"foundation", "dialogue", "context"}:
            raise ValueError("training stage must be foundation, dialogue, or context")
        if not self.teacher_model_id or not self.teacher_revision:
            raise ValueError("teacher identity and revision are required")
        if self.maximum_steps < 1 or self.maximum_wall_seconds < 60:
            raise ValueError("training limits are invalid")
        if self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("batch and accumulation sizes must be positive")
        if not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("learning rate is invalid")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum learning-rate ratio is invalid")
        if not 0 <= self.warmup_steps < self.maximum_steps:
            raise ValueError("warmup_steps must be smaller than maximum_steps")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay is invalid")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.hard_label_weight < 0.0 or self.distillation_weight < 0.0:
            raise ValueError("loss weights cannot be negative")
        if self.hard_label_weight + self.distillation_weight <= 0.0:
            raise ValueError("at least one language loss must be active")
        if not 0.1 <= self.distillation_temperature <= 10.0:
            raise ValueError("distillation temperature is invalid")
        if not 2 <= self.distillation_top_k <= 512:
            raise ValueError("distillation_top_k is invalid")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported training precision")
        if self.log_interval < 1 or self.save_interval < 1:
            raise ValueError("logging and save intervals must be positive")


@dataclass(frozen=True)
class DistillationLoss:
    loss: Tensor
    teacher_top_mass: Tensor
    student_top_mass: Tensor


def topk_tail_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    top_k: int,
) -> DistillationLoss:
    """KL on teacher top-k classes plus a single probability-tail bucket."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have equal shapes")
    if student_logits.shape[:-1] != labels.shape:
        raise ValueError("label mask does not match logits")
    mask = labels != -100
    if not bool(mask.any()):
        raise ValueError("distillation batch contains no supervised tokens")
    student = student_logits[mask].float() / temperature
    teacher = teacher_logits[mask].float() / temperature
    count = min(int(top_k), teacher.shape[-1] - 1)
    teacher_values, teacher_indices = torch.topk(teacher, count, dim=-1)
    student_values = torch.gather(student, -1, teacher_indices)
    teacher_log_normalizer = torch.logsumexp(teacher, dim=-1, keepdim=True)
    student_log_normalizer = torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_top = torch.exp(teacher_values - teacher_log_normalizer)
    student_top = torch.exp(student_values - student_log_normalizer)
    teacher_tail = (1.0 - teacher_top.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    student_tail = (1.0 - student_top.sum(dim=-1, keepdim=True)).clamp_min(1e-8)
    teacher_distribution = torch.cat((teacher_top, teacher_tail), dim=-1)
    student_distribution = torch.cat((student_top, student_tail), dim=-1)
    divergence = teacher_distribution * (
        teacher_distribution.clamp_min(1e-8).log()
        - student_distribution.clamp_min(1e-8).log()
    )
    loss = divergence.sum(dim=-1).mean() * (temperature**2)
    return DistillationLoss(
        loss=loss,
        teacher_top_mass=teacher_top.sum(dim=-1).mean(),
        student_top_mass=student_top.sum(dim=-1).mean(),
    )


def _learning_rate(config: AtomTrainingConfig, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        config.maximum_steps - config.warmup_steps, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    ratio = config.minimum_learning_rate_ratio
    return config.learning_rate * (ratio + (1.0 - ratio) * cosine)


def _autocast_context(device: torch.device, precision: str) -> Any:
    enabled = precision != "float32"
    if precision == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def load_teacher(
    model_id: str,
    *,
    revision: str,
    device: torch.device,
    precision: str,
    local_files_only: bool = False,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    if precision == "float16":
        dtype = torch.float16
    elif precision == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def save_atom_english_checkpoint(
    output_directory: Path,
    model: AtomCausalLanguageModel,
    *,
    optimizer: torch.optim.Optimizer | None,
    training_state: Mapping[str, Any],
    tokenizer_manifest: Mapping[str, Any],
    curriculum_manifest: Mapping[str, Any],
    training_config: AtomTrainingConfig,
) -> dict[str, Any]:
    """Write a hash-bound, safely loadable model and resume state."""

    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / "model.safetensors"
    config_path = output_directory / "model_config.json"
    state_path = output_directory / "training_state.json"
    optimizer_path = output_directory / "optimizer.pt"
    tokenizer_path = output_directory / "tokenizer_manifest.json"
    curriculum_path = output_directory / "curriculum_manifest.json"

    weights = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(model_path))
    _write_json(config_path, model.config.to_dict())
    _write_json(state_path, dict(training_state))
    _write_json(tokenizer_path, dict(tokenizer_manifest))
    _write_json(curriculum_path, dict(curriculum_manifest))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_path)
    elif optimizer_path.exists():
        optimizer_path.unlink()

    files = {
        "model.safetensors": _file_sha256(model_path),
        "model_config.json": _file_sha256(config_path),
        "training_state.json": _file_sha256(state_path),
        "tokenizer_manifest.json": _file_sha256(tokenizer_path),
        "curriculum_manifest.json": _file_sha256(curriculum_path),
    }
    if optimizer_path.exists():
        files["optimizer.pt"] = _file_sha256(optimizer_path)
    body = {
        "schema_version": 1,
        "architecture": ATOM_ENGLISH_CORE_RUNTIME,
        "training_runtime": ATOM_ENGLISH_TRAINING_RUNTIME,
        "files": files,
        "model_parameter_count": model.parameter_count(),
        "model_config_hash": _canonical_hash(model.config.to_dict()),
        "tokenizer_hash": _canonical_hash(dict(tokenizer_manifest)),
        "curriculum_hash": _canonical_hash(dict(curriculum_manifest)),
        "training_config": asdict(training_config),
        "training_state": dict(training_state),
        "architecture_manifest": atom_english_architecture_manifest(
            model.config, model
        ),
    }
    manifest = dict(body)
    manifest["manifest_hash"] = _canonical_hash(body)
    _write_json(output_directory / "checkpoint_manifest.json", manifest)
    return manifest


def _verified_checkpoint_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "checkpoint_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "architecture",
        "training_runtime",
        "files",
        "model_parameter_count",
        "model_config_hash",
        "tokenizer_hash",
        "curriculum_hash",
        "training_config",
        "training_state",
        "architecture_manifest",
        "manifest_hash",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint manifest fields are invalid")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    body = dict(payload)
    manifest_hash = body.pop("manifest_hash")
    if _canonical_hash(body) != manifest_hash:
        raise ValueError("checkpoint manifest hash mismatch")
    for name, expected_hash in payload["files"].items():
        path = directory / name
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    return payload


def load_atom_english_checkpoint(
    directory: Path,
    *,
    device: torch.device | str,
) -> tuple[AtomCausalLanguageModel, dict[str, Any]]:
    manifest = _verified_checkpoint_manifest(directory)
    config_payload = json.loads(
        (directory / "model_config.json").read_text(encoding="utf-8")
    )
    if _canonical_hash(config_payload) != manifest["model_config_hash"]:
        raise ValueError("checkpoint model config binding failed")
    config = AtomEnglishConfig.from_dict(config_payload)
    model = AtomCausalLanguageModel(config)
    weights = load_file(str(directory / "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if model.parameter_count() != manifest["model_parameter_count"]:
        raise ValueError("checkpoint parameter count mismatch")
    model.to(device)
    return model, manifest


def load_optimizer_checkpoint(
    directory: Path,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    manifest = _verified_checkpoint_manifest(directory)
    if "optimizer.pt" not in manifest["files"]:
        raise ValueError("checkpoint does not contain optimizer state")
    state = torch.load(
        directory / "optimizer.pt",
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(state)
    return manifest


def evaluate_perplexity(
    model: AtomCausalLanguageModel,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    maximum_batches: int,
    precision: str,
) -> dict[str, float | int]:
    if maximum_batches < 1:
        raise ValueError("maximum_batches must be positive")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    with torch.inference_mode():
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with _autocast_context(device, precision):
                output = model(input_ids, labels)
            assert output.cross_entropy is not None
            tokens = int((labels != -100).sum().item())
            total_loss += float(output.cross_entropy.item()) * tokens
            total_tokens += tokens
            batch_count += 1
            if batch_count >= maximum_batches:
                break
    if total_tokens == 0:
        raise ValueError("evaluation stream produced no supervised tokens")
    mean_loss = total_loss / total_tokens
    return {
        "batches": batch_count,
        "tokens": total_tokens,
        "cross_entropy": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


def train_atom_english_stage(
    model: AtomCausalLanguageModel,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    teacher: nn.Module,
    config: AtomTrainingConfig,
    output_directory: Path,
    tokenizer_manifest: Mapping[str, Any],
    curriculum_manifest: Mapping[str, Any],
    device: torch.device,
    resume_directory: Path | None = None,
    starting_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one resumable broad-language stage and retain exact evidence."""

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    if resume_directory is not None and starting_state is not None:
        raise ValueError("resume_directory and starting_state are mutually exclusive")
    optimizer_step = 0
    consumed_tokens = 0
    micro_step = 0
    if resume_directory is not None:
        resume_manifest = load_optimizer_checkpoint(resume_directory, optimizer)
        state = resume_manifest["training_state"]
        optimizer_step = int(state["optimizer_step"])
        consumed_tokens = int(state["consumed_tokens"])
        micro_step = int(state["micro_step"])
    elif starting_state is not None:
        optimizer_step = int(starting_state["optimizer_step"])
        consumed_tokens = int(starting_state["consumed_tokens"])
        micro_step = int(starting_state["micro_step"])
    if optimizer_step < 0 or consumed_tokens < 0 or micro_step < 0:
        raise ValueError("starting training state cannot be negative")
    for group in optimizer.param_groups:
        group["lr"] = _learning_rate(config, optimizer_step)

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(config.precision == "float16"),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    session_micro_step = 0
    history: list[dict[str, float | int]] = []
    latest: dict[str, float] = {}
    stop_reason = "input_exhausted"
    _emit_training_event(
        "stage_started",
        stage=config.stage,
        device=str(device),
        precision=config.precision,
        model_parameters=model.parameter_count(),
        teacher_parameters=sum(parameter.numel() for parameter in teacher.parameters()),
        starting_optimizer_step=optimizer_step,
        starting_consumed_tokens=consumed_tokens,
    )

    for batch in batches:
        elapsed = time.monotonic() - started
        if (
            elapsed >= config.maximum_wall_seconds
            and session_micro_step % config.gradient_accumulation == 0
        ):
            stop_reason = "wall_time_limit"
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.inference_mode(), _autocast_context(device, config.precision):
            teacher_logits = teacher(input_ids=input_ids).logits
        if teacher_logits.shape[-1] != model.config.vocab_size:
            raise ValueError("teacher and student vocabularies are not aligned")
        with _autocast_context(device, config.precision):
            student = model(input_ids, labels)
            assert student.cross_entropy is not None
            distilled = topk_tail_distillation_loss(
                student.logits,
                teacher_logits,
                labels,
                temperature=config.distillation_temperature,
                top_k=config.distillation_top_k,
            )
            weight_total = config.hard_label_weight + config.distillation_weight
            language_loss = (
                config.hard_label_weight * student.cross_entropy
                + config.distillation_weight * distilled.loss
            ) / weight_total
            loss = (
                language_loss
                + model.config.criticality_weight * student.criticality_loss
            )
            scaled_loss = loss / config.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        session_micro_step += 1
        supervised_tokens = int((labels != -100).sum().item())
        consumed_tokens += supervised_tokens
        latest = {
            "loss": float(loss.detach().item()),
            "cross_entropy": float(student.cross_entropy.detach().item()),
            "distillation": float(distilled.loss.detach().item()),
            "criticality": float(student.criticality_loss.detach().item()),
            "edge_entropy": float(student.mean_edge_entropy.detach().item()),
            "active_edges": float(student.mean_active_edges.detach().item()),
            "temperature": float(student.mean_temperature.detach().item()),
            "teacher_top_mass": float(distilled.teacher_top_mass.detach().item()),
            "student_top_mass": float(distilled.student_top_mass.detach().item()),
        }
        if session_micro_step % config.gradient_accumulation:
            continue
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        learning_rate = _learning_rate(config, optimizer_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        latest["gradient_norm"] = float(gradient_norm)
        latest["learning_rate"] = float(learning_rate)
        if optimizer_step % config.log_interval == 0:
            log_record = {
                "optimizer_step": optimizer_step,
                "consumed_tokens": consumed_tokens,
                "elapsed_seconds": time.monotonic() - started,
                **latest,
            }
            history.append(log_record)
            _emit_training_event("optimizer_progress", **log_record)
        if optimizer_step % config.save_interval == 0:
            periodic_manifest = save_atom_english_checkpoint(
                output_directory / "latest",
                model,
                optimizer=optimizer,
                training_state={
                    "stage": config.stage,
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "consumed_tokens": consumed_tokens,
                    "elapsed_seconds": time.monotonic() - started,
                    "stop_reason": "periodic_save",
                },
                tokenizer_manifest=tokenizer_manifest,
                curriculum_manifest=curriculum_manifest,
                training_config=config,
            )
            _emit_training_event(
                "checkpoint_saved",
                optimizer_step=optimizer_step,
                consumed_tokens=consumed_tokens,
                checkpoint_manifest_hash=periodic_manifest["manifest_hash"],
            )
        if optimizer_step >= config.maximum_steps:
            stop_reason = "step_limit"
            break

    remainder = session_micro_step % config.gradient_accumulation
    if remainder and optimizer_step < config.maximum_steps and latest:
        scaler.unscale_(optimizer)
        correction = config.gradient_accumulation / remainder
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        learning_rate = _learning_rate(config, optimizer_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        latest["gradient_norm"] = float(gradient_norm)
        latest["learning_rate"] = float(learning_rate)
        log_record = {
            "optimizer_step": optimizer_step,
            "consumed_tokens": consumed_tokens,
            "elapsed_seconds": time.monotonic() - started,
            "partial_accumulation_micro_steps": remainder,
            **latest,
        }
        history.append(log_record)
        _emit_training_event("optimizer_progress", **log_record)
        if stop_reason == "input_exhausted":
            stop_reason = "input_exhausted_after_final_accumulation"

    elapsed = time.monotonic() - started
    state = {
        "stage": config.stage,
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "consumed_tokens": consumed_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": consumed_tokens / max(elapsed, 1e-9),
        "stop_reason": stop_reason,
        "latest_metrics": latest,
    }
    checkpoint = save_atom_english_checkpoint(
        output_directory / "latest",
        model,
        optimizer=optimizer,
        training_state=state,
        tokenizer_manifest=tokenizer_manifest,
        curriculum_manifest=curriculum_manifest,
        training_config=config,
    )
    report_body = {
        "schema_version": 1,
        "runtime": ATOM_ENGLISH_TRAINING_RUNTIME,
        "training_config": asdict(config),
        "training_state": state,
        "history": history,
        "checkpoint_manifest_hash": checkpoint["manifest_hash"],
        "process_id": os.getpid(),
    }
    report = dict(report_body)
    report["report_hash"] = _canonical_hash(report_body)
    _write_json(output_directory / "training_report.json", report)
    _emit_training_event(
        "stage_stopped",
        stop_reason=stop_reason,
        optimizer_step=optimizer_step,
        consumed_tokens=consumed_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=state["tokens_per_second"],
        checkpoint_manifest_hash=checkpoint["manifest_hash"],
        report_hash=report["report_hash"],
    )
    return report


def atom_english_training_self_test() -> dict[str, bool]:
    torch.manual_seed(19)
    teacher = torch.randn(2, 5, 64)
    labels = torch.randint(0, 64, (2, 5))
    equal = topk_tail_distillation_loss(
        teacher,
        teacher,
        labels,
        temperature=2.0,
        top_k=8,
    )
    altered = topk_tail_distillation_loss(
        -teacher,
        teacher,
        labels,
        temperature=2.0,
        top_k=8,
    )
    masked = labels.clone()
    masked[:, :2] = -100
    masked_loss = topk_tail_distillation_loss(
        -teacher,
        teacher,
        masked,
        temperature=2.0,
        top_k=8,
    )
    return {
        "equal_logits_zero_divergence": abs(float(equal.loss)) < 1e-6,
        "different_logits_positive_divergence": float(altered.loss) > 0.0,
        "masked_distillation_finite": bool(torch.isfinite(masked_loss.loss)),
        "tail_bucket_has_mass": float(equal.teacher_top_mass) < 1.0,
    }
