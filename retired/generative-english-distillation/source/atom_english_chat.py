"""Interactive multi-turn runtime for a trained Atom generative English model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from atom_english_data import (
    broad_english_curriculum,
    chat_token_ids,
    load_aligned_tokenizer,
    tokenizer_manifest,
)
from atom_english_evaluation import load_language_evaluation
from atom_english_knowledge import (
    ATOM_ENGLISH_RAG_RUNTIME,
    ATOM_ENGLISH_WIKI_RUNTIME,
    build_english_knowledge_graph,
    retrieve_english_knowledge,
    validate_english_knowledge_graph,
)
from atom_english_side_view import (
    ATOM_ENGLISH_SIDE_VIEW_RUNTIME,
    write_english_generation_side_view,
)
from atom_english_training import load_atom_english_checkpoint

ATOM_ENGLISH_CHAT_RUNTIME = "atom-generative-english-chat-v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA execution is unavailable")
    return torch.device(requested)


def _format_contexts(contexts: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        (
            f"[{index}] {item['node']['kind']}: {item['node']['label']} | "
            f"{json.dumps(item['node']['attributes'], sort_keys=True)}"
        )
        for index, item in enumerate(contexts, start=1)
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class AtomEnglishChatAssets:
    model: Any
    tokenizer: Any
    checkpoint_manifest: Mapping[str, Any]
    evaluation: Mapping[str, Any] | None
    knowledge_graph: Mapping[str, Any]
    device: torch.device


def load_chat_assets(
    checkpoint_directory: Path,
    *,
    evaluation_path: Path | None,
    knowledge_graph_path: Path | None,
    device: str = "auto",
    allow_unqualified_checkpoint: bool = False,
    local_files_only: bool = False,
) -> AtomEnglishChatAssets:
    """Load mutually bound runtime artifacts and reject mismatches."""

    checkpoint_directory = Path(checkpoint_directory)
    runtime_device = _device(device)
    model, checkpoint = load_atom_english_checkpoint(
        checkpoint_directory,
        device=runtime_device,
    )
    if checkpoint["training_config"]["stage"] != "context":
        raise ValueError(
            "chat requires a dialogue-trained, context-conditioned checkpoint"
        )

    curriculum = broad_english_curriculum()
    tokenizer = load_aligned_tokenizer(
        curriculum.tokenizer_id,
        revision=curriculum.tokenizer_revision,
        local_files_only=local_files_only,
    )
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("chat tokenizer vocabulary does not match the model")
    saved_tokenizers = _load_json_object(
        checkpoint_directory / "tokenizer_manifest.json",
        "tokenizer manifest",
    )
    expected_dialogue = tokenizer_manifest(
        tokenizer,
        tokenizer_id=curriculum.tokenizer_id,
        revision=curriculum.tokenizer_revision,
    )
    if saved_tokenizers.get("dialogue") != expected_dialogue:
        raise ValueError("chat tokenizer does not match the checkpoint")

    evaluation: Mapping[str, Any] | None = None
    if evaluation_path is not None:
        evaluation = load_language_evaluation(
            Path(evaluation_path),
            checkpoint_manifest_hash=checkpoint["manifest_hash"],
            model_parameter_count=model.parameter_count(),
            require_gate=not allow_unqualified_checkpoint,
        )
    elif not allow_unqualified_checkpoint:
        raise ValueError(
            "chat requires an external language evaluation bound to the checkpoint"
        )

    if knowledge_graph_path is None:
        knowledge_graph = build_english_knowledge_graph(
            model.config,
            curriculum,
        )
    else:
        knowledge_graph = _load_json_object(
            Path(knowledge_graph_path),
            "knowledge graph",
        )
        validate_english_knowledge_graph(knowledge_graph)

    model.eval()
    return AtomEnglishChatAssets(
        model=model,
        tokenizer=tokenizer,
        checkpoint_manifest=checkpoint,
        evaluation=evaluation,
        knowledge_graph=knowledge_graph,
        device=runtime_device,
    )


class AtomEnglishChatSession:
    """Stateful free-form conversation over the causal graph language model."""

    def __init__(
        self,
        assets: AtomEnglishChatAssets,
        *,
        evidence_policy: str = "available",
        maximum_new_tokens: int = 256,
        temperature: float = 0.72,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.08,
        seed: int = 20260724,
    ) -> None:
        if evidence_policy not in {"available", "required", "off"}:
            raise ValueError("unknown evidence policy")
        if maximum_new_tokens < 1:
            raise ValueError("maximum_new_tokens must be positive")
        if maximum_new_tokens >= assets.model.config.max_seq_len:
            raise ValueError("generation budget leaves no room for a prompt")
        self.assets = assets
        self.evidence_policy = evidence_policy
        self.maximum_new_tokens = maximum_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.messages: list[dict[str, str]] = []
        self.samples: list[dict[str, Any]] = []

    def _retrieved_contexts(self, user_text: str) -> list[dict[str, Any]]:
        if self.evidence_policy == "off":
            return []
        return retrieve_english_knowledge(
            self.assets.knowledge_graph,
            user_text,
            limit=8,
        )

    def _prompt_messages(
        self,
        user_text: str,
        contexts: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        messages = [*self.messages, {"role": "user", "content": user_text}]
        if contexts:
            evidence = _format_contexts(contexts)
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Use the supplied evidence when making factual claims. "
                        "Do not invent evidence. If the evidence cannot support "
                        "the requested factual answer, say that clearly.\n\n"
                        f"Evidence:\n{evidence}"
                    ),
                },
            )
        return messages

    def _fit_prompt(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[int]]:
        available = self.assets.model.config.max_seq_len - self.maximum_new_tokens
        working = list(messages)
        while True:
            token_ids = chat_token_ids(
                self.assets.tokenizer,
                working,
                add_generation_prompt=True,
            )
            if len(token_ids) <= available:
                return working, token_ids
            removable = 1 if working and working[0]["role"] == "system" else 0
            if len(working) - removable <= 1:
                raise ValueError("the current request exceeds the model context")
            del working[removable : min(removable + 2, len(working) - 1)]

    @torch.inference_mode()
    def reply(self, user_text: str) -> dict[str, Any]:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("chat input must be non-empty text")
        user_text = user_text.strip()
        contexts = self._retrieved_contexts(user_text)
        if self.evidence_policy == "required" and not contexts:
            response = (
                "I do not have evidence in the connected knowledge graph "
                "that supports a factual answer to that request."
            )
            result = {
                "prompt": user_text,
                "response": response,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "evidence_status": "insufficient",
                "evidence": [],
            }
        else:
            prompt_messages = self._prompt_messages(user_text, contexts)
            _, token_ids = self._fit_prompt(prompt_messages)
            input_ids = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=self.assets.device,
            ).unsqueeze(0)
            generated = self.assets.model.generate(
                input_ids,
                max_new_tokens=self.maximum_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=int(self.assets.tokenizer.eos_token_id),
                seed=self.seed + len(self.samples),
            )
            response_ids = generated[0, input_ids.shape[1] :].tolist()
            response = self.assets.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            ).strip()
            if not response:
                raise RuntimeError("the model generated no conversational text")
            result = {
                "prompt": user_text,
                "response": response,
                "prompt_tokens": len(token_ids),
                "response_tokens": len(response_ids),
                "evidence_status": "available" if contexts else "not_found",
                "evidence": list(contexts),
            }
        self.messages.extend(
            (
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": result["response"]},
            )
        )
        self.samples.append(result)
        return result

    def artifact(self) -> dict[str, Any]:
        evaluation = (
            dict(self.assets.evaluation)
            if self.assets.evaluation is not None
            else {
                "gate": {"passed": False},
                "status": "explicit unqualified-checkpoint override",
            }
        )
        return {
            "schema_version": 1,
            "runtime": ATOM_ENGLISH_CHAT_RUNTIME,
            "mode": "chat",
            "parameter_count": self.assets.model.parameter_count(),
            "checkpoint_manifest_hash": self.assets.checkpoint_manifest[
                "manifest_hash"
            ],
            "evaluation": evaluation,
            "evidence_policy": self.evidence_policy,
            "knowledge_runtime": {
                "wiki": ATOM_ENGLISH_WIKI_RUNTIME,
                "rag": ATOM_ENGLISH_RAG_RUNTIME,
            },
            "side_view_runtime": ATOM_ENGLISH_SIDE_VIEW_RUNTIME,
            "samples": list(self.samples),
        }

    def write_artifacts(
        self,
        *,
        transcript_path: Path | None,
        side_view_path: Path | None,
    ) -> None:
        artifact = self.artifact()
        if transcript_path is not None:
            _write_json(Path(transcript_path), artifact)
        if side_view_path is not None:
            side_view_path = Path(side_view_path)
            side_view_path.parent.mkdir(parents=True, exist_ok=True)
            write_english_generation_side_view(
                side_view_path,
                artifact,
                self.assets.knowledge_graph,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--knowledge-graph", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument(
        "--evidence-policy",
        choices=("available", "required", "off"),
        default="available",
    )
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--side-view", type=Path)
    parser.add_argument("--allow-unqualified-checkpoint", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assets = load_chat_assets(
        args.checkpoint,
        evaluation_path=args.evaluation,
        knowledge_graph_path=args.knowledge_graph,
        device=args.device,
        allow_unqualified_checkpoint=args.allow_unqualified_checkpoint,
        local_files_only=args.local_files_only,
    )
    session = AtomEnglishChatSession(
        assets,
        evidence_policy=args.evidence_policy,
        maximum_new_tokens=args.maximum_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    if args.prompt is not None:
        response = session.reply(args.prompt)
        print(response["response"])
        session.write_artifacts(
            transcript_path=args.transcript,
            side_view_path=args.side_view,
        )
        return 0

    print("Atom English chat. Enter /exit to stop.")
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            break
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if not user_text:
            continue
        response = session.reply(user_text)
        print(f"atom> {response['response']}")
        session.write_artifacts(
            transcript_path=args.transcript,
            side_view_path=args.side_view,
        )
    session.write_artifacts(
        transcript_path=args.transcript,
        side_view_path=args.side_view,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
