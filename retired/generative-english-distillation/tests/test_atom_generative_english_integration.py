from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from atom_english_chat import (
    ATOM_ENGLISH_CHAT_RUNTIME,
    AtomEnglishChatAssets,
    AtomEnglishChatSession,
)
from atom_english_core import (
    ATOM_LONG_CONTEXT_MILESTONES,
    ATOM_LONG_CONTEXT_TARGET,
    ATOM_ROOT_PRIMITIVES,
    AtomCausalLanguageModel,
    AtomGraphStepState,
    atom_english_architecture_manifest,
    atom_english_context_expansion_plan,
    atom_english_core_self_test,
    atom_english_profile,
    expand_atom_english_context,
)
from atom_english_context import (
    CONTEXT_TARGET_OPTIMIZER_STEPS,
    build_context_training_tensors,
    context_length_for_step,
    train_atom_english_context_stage,
)
from atom_english_data import (
    broad_english_curriculum,
    chat_token_ids,
    encode_dialogue,
    load_aligned_tokenizer,
    validate_tokenizer_alignment,
)
from atom_english_evaluation import (
    ATOM_LONG_CONTEXT_EVALUATION_RUNTIME,
    LONG_CONTEXT_TASK_FAMILIES,
    build_long_context_probe,
    language_competence_gate,
    load_language_evaluation,
    write_language_evaluation,
)
from atom_english_kaggle import (
    DIALOGUE_TARGET_OPTIMIZER_STEPS,
    FOUNDATION_TARGET_TOKENS,
    SMOLTALK_TRAIN_ROWS,
    discover_input_checkpoint,
    resolve_training_resume_config,
    run_self_test,
    validate_training_stage_admission,
)
from atom_english_knowledge import (
    ATOM_ENGLISH_RAG_RUNTIME,
    ATOM_ENGLISH_WIKI_RUNTIME,
    build_english_knowledge_graph,
    retrieve_english_knowledge,
)
from atom_english_side_view import (
    ATOM_ENGLISH_SIDE_VIEW_RUNTIME,
    render_english_generation_artifact,
)
from atom_english_training import (
    AtomTrainingConfig,
    load_atom_english_checkpoint,
    save_atom_english_checkpoint,
    train_atom_english_stage,
)
from scripts.build_kaggle_generative_english_bundle import build_bundle


class FixedTeacher(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        vocabulary = torch.arange(
            self.vocab_size,
            device=input_ids.device,
            dtype=torch.float32,
        )
        center = input_ids.unsqueeze(-1).float()
        logits = -((vocabulary - center) / 32.0).square()
        return SimpleNamespace(logits=logits)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        tokens: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


def passing_long_context_result() -> dict[str, object]:
    lengths = {
        str(length): {
            "examples": 4,
            "correct": 4,
            "accuracy": 1.0,
            "task_families": list(LONG_CONTEXT_TASK_FAMILIES),
            "samples": [],
        }
        for length in ATOM_LONG_CONTEXT_MILESTONES
    }
    return {
        "runtime": ATOM_LONG_CONTEXT_EVALUATION_RUNTIME,
        "context_lengths": list(ATOM_LONG_CONTEXT_MILESTONES),
        "task_families": list(LONG_CONTEXT_TASK_FAMILIES),
        "examples": 4 * len(ATOM_LONG_CONTEXT_MILESTONES),
        "correct": 4 * len(ATOM_LONG_CONTEXT_MILESTONES),
        "accuracy": 1.0,
        "lengths": lengths,
        "samples": [],
    }


class AtomGenerativeEnglishIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260724)
        self.config = atom_english_profile(
            "verification",
            vocab_size=512,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        self.model = AtomCausalLanguageModel(self.config)

    def test_parallel_and_recurrent_graphs_are_equivalent(self) -> None:
        checks = atom_english_core_self_test()
        self.assertTrue(all(checks.values()), checks)

    def test_context_expansion_preserves_existing_causal_behavior(self) -> None:
        self.model.eval()
        tokens = torch.randint(3, self.config.vocab_size, (1, 24))
        expected = self.model(tokens).logits.detach()
        target = replace(
            self.config,
            dilation_offsets=(*self.config.dilation_offsets, 96),
            max_seq_len=128,
        )
        expanded = expand_atom_english_context(self.model, target).eval()
        actual = expanded(tokens).logits.detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(expanded.config.max_seq_len, 128)
        self.assertGreater(expanded.parameter_count(), self.model.parameter_count())

    def test_foundation_resume_defers_context_expansion(self) -> None:
        source = atom_english_profile(
            "scale-227m",
            vocab_size=49_152,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        source = replace(
            source,
            dilation_offsets=tuple(
                offset for offset in source.dilation_offsets if offset <= 1_024
            ),
            max_seq_len=2_048,
        )
        target = atom_english_profile(
            "scale-227m",
            vocab_size=49_152,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        plan = atom_english_context_expansion_plan(source, target)
        self.assertEqual(plan["source_context_tokens"], 2_048)
        self.assertEqual(plan["target_context_tokens"], 524_288)
        self.assertEqual(
            plan["initialized_causal_distances"],
            [1_536, 2_048, 3_072, 4_096, 6_144],
        )
        selected, deferred = resolve_training_resume_config(
            "foundation",
            "foundation",
            source,
            target,
        )
        self.assertIs(selected, source)
        assert deferred is not None
        self.assertTrue(deferred["optimizer_state_preserved"])
        self.assertEqual(deferred["deferred_until_stage"], "context")

    def test_chunked_prefill_matches_parallel_and_recurrent_execution(self) -> None:
        self.model.eval()
        tokens = torch.randint(3, self.config.vocab_size, (2, 24))
        with torch.no_grad():
            parallel = self.model(tokens).logits[:, -1]
            prefill, states, _ = self.model.prefill(tokens)
            recurrent_states = self.model.initial_generation_state()
            recurrent = None
            for index in range(tokens.shape[1]):
                recurrent, _ = self.model.forward_step(
                    tokens[:, index],
                    recurrent_states,
                )
        assert recurrent is not None
        torch.testing.assert_close(prefill, parallel, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(prefill, recurrent, atol=2e-5, rtol=2e-4)
        self.assertTrue(all(state.position == tokens.shape[1] for state in states))

    def test_streaming_tail_logits_retain_gradients_over_frozen_state(self) -> None:
        self.model.train()
        prefix = torch.randint(3, self.config.vocab_size, (1, 32))
        tail = torch.randint(3, self.config.vocab_size, (1, 16))
        with torch.no_grad():
            _, states, _ = self.model.prefill(prefix)
        logits, _, _ = self.model.forward_stream(tail, states)
        logits.float().square().mean().backward()
        gradient = self.model.blocks[0].graph.qkv_projection.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(all(state.position == 48 for state in states))

    def test_completed_chunk_exposes_exact_ordered_episodic_landmarks(self) -> None:
        persistence = self.model.blocks[0].graph.persistence.eval()
        width = self.config.d_model
        values = torch.zeros((1, self.config.persistence_chunk * 2, width))
        for index in range(values.shape[1]):
            values[0, index, index] = float(index + 1)
        memory, valid, positions = persistence(values)
        summary_slots = self.config.persistence_slots
        episodic = memory[0, self.config.persistence_chunk, summary_slots:]
        episodic_valid = valid[0, self.config.persistence_chunk, summary_slots:]
        episodic_positions = positions[
            0,
            self.config.persistence_chunk,
            summary_slots:,
        ]
        self.assertTrue(bool(episodic_valid.all()))
        self.assertEqual(
            set(episodic_positions.tolist()),
            set(range(self.config.persistence_chunk)),
        )
        original = values[0, : self.config.persistence_chunk]
        distances = torch.cdist(original.float(), episodic.float())
        torch.testing.assert_close(
            distances.amin(dim=1),
            torch.zeros(self.config.persistence_chunk),
            rtol=0.0,
            atol=0.0,
        )

    def test_symbolic_causal_copy_recalls_observed_transition_at_512k(
        self,
    ) -> None:
        config = replace(
            self.config,
            max_seq_len=ATOM_LONG_CONTEXT_TARGET,
        )
        model = AtomCausalLanguageModel(config).eval()
        key = [401, 402, 403, 404]
        expected = 207
        sequence = [11] * ATOM_LONG_CONTEXT_TARGET
        sequence[1_000:1_005] = [*key, expected]
        sequence[-len(key) :] = key
        input_ids = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        base_logits = torch.zeros((1, config.vocab_size))
        base_logits[0, 206] = 2.0
        state = AtomGraphStepState(
            exact_cache_capacity=1,
            persistence_levels=1,
        )
        actual = model._apply_symbolic_copy(
            base_logits,
            input_ids,
            state,
        )
        self.assertEqual(int(actual.argmax(dim=-1).item()), expected)
        self.assertEqual(
            len(state.symbolic_histories[0]),
            ATOM_LONG_CONTEXT_TARGET,
        )
        self.assertLessEqual(
            len(state.symbolic_transitions[0]),
            config.max_seq_len * len(config.symbolic_copy_orders),
        )

    def test_symbolic_copy_respects_neural_compatibility_margin(self) -> None:
        model = self.model.eval()
        key = [401, 402, 403, 404]
        expected = 207
        input_ids = torch.tensor(
            [[*key, expected, *key]],
            dtype=torch.long,
        )
        base_logits = torch.zeros((1, self.config.vocab_size))
        rejected = 206
        base_logits[0, rejected] = self.config.symbolic_copy_neural_margin + 1.0
        state = AtomGraphStepState(
            exact_cache_capacity=1,
            persistence_levels=1,
        )
        actual = model._apply_symbolic_copy(
            base_logits,
            input_ids,
            state,
        )
        self.assertEqual(int(actual.argmax(dim=-1).item()), rejected)

    def test_long_context_state_is_bounded_and_reaches_512k_boundary(self) -> None:
        target = replace(
            self.config,
            max_seq_len=ATOM_LONG_CONTEXT_TARGET,
        )
        model = AtomCausalLanguageModel(target).eval()
        tokens = torch.randint(3, target.vocab_size, (1, 256))
        with torch.no_grad():
            _, states, _ = model.prefill(tokens)
        for state in states:
            self.assertEqual(
                state.resident_exact_tokens,
                target.exact_cache_tokens,
            )
            self.assertEqual(
                state.rotated_keys.maxlen,
                target.exact_cache_tokens,
            )
            self.assertEqual(state.position, 256)
            self.assertEqual(
                len(state.hierarchy_numerators),
                target.persistence_levels,
            )

        boundary_states = model.initial_generation_state()
        for state in boundary_states:
            state.position = ATOM_LONG_CONTEXT_TARGET - 1
        final_token = torch.randint(3, target.vocab_size, (1,))
        with torch.no_grad():
            logits, _ = model.forward_step(final_token, boundary_states)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            all(state.position == ATOM_LONG_CONTEXT_TARGET for state in boundary_states)
        )
        with self.assertRaisesRegex(ValueError, "context"):
            model.forward_step(final_token, boundary_states)

    def test_scale_profile_targets_512k_with_264k_gate(self) -> None:
        config = atom_english_profile(
            "scale-227m",
            vocab_size=49_152,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        self.assertEqual(config.max_seq_len, 524_288)
        self.assertIn(264_000, ATOM_LONG_CONTEXT_MILESTONES)
        self.assertEqual(
            ATOM_LONG_CONTEXT_MILESTONES[-1],
            config.max_seq_len,
        )
        self.assertLess(config.exact_cache_tokens, config.max_seq_len)
        self.assertGreater(config.persistence_levels, 10)

    def test_long_context_probe_is_exact_length_and_exercises_each_family(
        self,
    ) -> None:
        tokenizer = CharacterTokenizer()
        reservoir = tokenizer.encode(
            "Background scientific prose. ",
            add_special_tokens=False,
        )
        for index, family in enumerate(LONG_CONTEXT_TASK_FAMILIES):
            prompt, expected = build_long_context_probe(
                tokenizer,
                reservoir,
                prompt_tokens=8_192,
                family=family,
                seed=20260724 + index,
            )
            self.assertEqual(len(prompt), 8_192)
            decoded = tokenizer.decode(prompt, skip_special_tokens=True)
            self.assertIn(expected, decoded)
            self.assertTrue(decoded.endswith("Answer:"))

    def test_context_schedule_repeatedly_reaches_264k_and_512k(self) -> None:
        final_lengths = {
            context_length_for_step(
                step,
                target_steps=CONTEXT_TARGET_OPTIMIZER_STEPS,
            )
            for step in range(
                int(CONTEXT_TARGET_OPTIMIZER_STEPS * 0.95),
                CONTEXT_TARGET_OPTIMIZER_STEPS,
            )
        }
        self.assertIn(264_000, final_lengths)
        self.assertIn(524_288, final_lengths)

    def test_context_conditioning_trains_answer_tail_and_writes_checkpoint(
        self,
    ) -> None:
        tokenizer = CharacterTokenizer()
        reservoir = tokenizer.encode(
            "Stable scientific background sentence. ",
            add_special_tokens=False,
        )
        config = replace(self.config, max_seq_len=512)
        model = AtomCausalLanguageModel(config)
        example = build_context_training_tensors(
            tokenizer,
            reservoir,
            context_tokens=512,
            family="multi_needle",
            seed=20260724,
            trainable_tail_tokens=64,
        )
        self.assertEqual(example["context_tokens"], 512)
        self.assertGreater(example["supervised_tokens"], 1)
        before = model.blocks[0].graph.qkv_projection.weight.detach().clone()
        training = AtomTrainingConfig(
            stage="context",
            teacher_model_id="deterministic-context-curriculum",
            teacher_revision="test-v1",
            maximum_steps=4,
            maximum_wall_seconds=60,
            batch_size=1,
            gradient_accumulation=1,
            warmup_steps=0,
            precision="float32",
            log_interval=1,
            save_interval=4,
            learning_rate=1.0e-4,
            hard_label_weight=1.0,
            distillation_weight=0.0,
        )
        with tempfile.TemporaryDirectory(prefix="atom-context-training-") as temporary:
            report = train_atom_english_context_stage(
                model,
                tokenizer,
                reservoir,
                config=training,
                output_directory=Path(temporary),
                tokenizer_manifest={"schema_version": 1},
                curriculum_manifest={"schema_version": 1},
                device=torch.device("cpu"),
                training_lengths=(512,),
                trainable_tail_tokens=64,
                differentiable_context_limit=512,
            )
            self.assertEqual(
                report["training_state"]["optimizer_step"],
                4,
            )
            self.assertEqual(
                report["training_state"]["per_length_examples"]["512"],
                4,
            )
            self.assertTrue((Path(temporary) / "checkpoint_manifest.json").is_file())
        after = model.blocks[0].graph.qkv_projection.weight.detach()
        self.assertFalse(torch.equal(before, after))

    def test_every_root_mechanism_receives_gradient(self) -> None:
        tokens = torch.randint(3, self.config.vocab_size, (2, 24))
        labels = torch.roll(tokens, shifts=-1, dims=1)
        output = self.model(tokens, labels)
        self.assertIsNotNone(output.loss)
        assert output.loss is not None
        output.loss.backward()
        parameters = dict(self.model.named_parameters())
        bindings = {
            "radiation": "blocks.0.graph.radiation_gain",
            "dissipation": "blocks.0.graph.dissipation_gate.weight",
            "gravitation": "blocks.0.graph.qkv_projection.weight",
            "attraction_repulsion": ("blocks.0.graph.attraction_gate.weight"),
            "nucleation": "blocks.0.graph.nucleation_gate.weight",
            "conservation": "blocks.0.graph_residual.gain",
            "decay": "blocks.0.graph.decay_rate",
        }
        self.assertEqual(tuple(bindings), ATOM_ROOT_PRIMITIVES)
        for root, parameter_name in bindings.items():
            gradient = parameters[parameter_name].grad
            self.assertIsNotNone(gradient, root)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all(), root)
            self.assertGreater(float(gradient.abs().sum()), 0.0, root)
        persistence = parameters[
            "blocks.0.graph.persistence.persistence_gate.weight"
        ].grad
        phase = parameters["blocks.0.phase_mixer.in_projection.weight"].grad
        self.assertIsNotNone(persistence)
        self.assertIsNotNone(phase)
        assert persistence is not None and phase is not None
        self.assertGreater(float(persistence.abs().sum()), 0.0)
        self.assertGreater(float(phase.abs().sum()), 0.0)

    def test_checkpoint_round_trip_and_corruption_rejection(self) -> None:
        training_config = AtomTrainingConfig(
            stage="foundation",
            teacher_model_id="fixed-teacher",
            maximum_steps=2,
            maximum_wall_seconds=60,
            batch_size=1,
            gradient_accumulation=1,
            warmup_steps=0,
            precision="float32",
        )
        optimizer = torch.optim.AdamW(self.model.parameters())
        tokens = torch.randint(3, self.config.vocab_size, (1, 16))
        expected = self.model.eval()(tokens).logits.detach()
        with tempfile.TemporaryDirectory(
            prefix="atom-english-checkpoint-"
        ) as temporary:
            directory = Path(temporary)
            save_atom_english_checkpoint(
                directory,
                self.model,
                optimizer=optimizer,
                training_state={
                    "stage": "foundation",
                    "optimizer_step": 0,
                    "micro_step": 0,
                    "consumed_tokens": 0,
                },
                tokenizer_manifest={"schema_version": 1, "vocab_size": 512},
                curriculum_manifest={"schema_version": 1, "sources": []},
                training_config=training_config,
            )
            loaded, manifest = load_atom_english_checkpoint(directory, device="cpu")
            actual = loaded.eval()(tokens).logits.detach()
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(
                manifest["model_parameter_count"],
                self.model.parameter_count(),
            )
            model_path = directory / "model.safetensors"
            content = bytearray(model_path.read_bytes())
            content[-1] ^= 1
            model_path.write_bytes(bytes(content))
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                load_atom_english_checkpoint(directory, device="cpu")

    def test_real_optimizer_step_writes_resumable_checkpoint(self) -> None:
        teacher = FixedTeacher(self.config.vocab_size)
        batches = [
            {
                "input_ids": torch.randint(3, self.config.vocab_size, (2, 16)),
                "labels": torch.randint(3, self.config.vocab_size, (2, 16)),
            }
        ]
        config = AtomTrainingConfig(
            stage="foundation",
            teacher_model_id="fixed-teacher",
            maximum_steps=1,
            maximum_wall_seconds=60,
            batch_size=2,
            gradient_accumulation=1,
            warmup_steps=0,
            precision="float32",
            log_interval=1,
            save_interval=1,
        )
        with tempfile.TemporaryDirectory(prefix="atom-english-training-") as temporary:
            directory = Path(temporary)
            report = train_atom_english_stage(
                self.model,
                batches,
                teacher=teacher,
                config=config,
                output_directory=directory,
                tokenizer_manifest={"schema_version": 1, "vocab_size": 512},
                curriculum_manifest={"schema_version": 1, "sources": []},
                device=torch.device("cpu"),
            )
            self.assertEqual(report["training_state"]["optimizer_step"], 1)
            self.assertGreater(report["training_state"]["consumed_tokens"], 0)
            loaded, _ = load_atom_english_checkpoint(directory / "latest", device="cpu")
            self.assertEqual(loaded.parameter_count(), self.model.parameter_count())

    def test_context_migration_preserves_cumulative_training_state(self) -> None:
        teacher = FixedTeacher(self.config.vocab_size)
        batches = [
            {
                "input_ids": torch.randint(3, self.config.vocab_size, (1, 16)),
                "labels": torch.randint(3, self.config.vocab_size, (1, 16)),
            }
        ]
        config = AtomTrainingConfig(
            stage="foundation",
            teacher_model_id="fixed-teacher",
            maximum_steps=2,
            maximum_wall_seconds=60,
            batch_size=1,
            gradient_accumulation=1,
            warmup_steps=0,
            precision="float32",
            log_interval=1,
            save_interval=2,
        )
        with tempfile.TemporaryDirectory(
            prefix="atom-english-migrated-training-"
        ) as temporary:
            report = train_atom_english_stage(
                self.model,
                batches,
                teacher=teacher,
                config=config,
                output_directory=Path(temporary),
                tokenizer_manifest={"schema_version": 1, "vocab_size": 512},
                curriculum_manifest={"schema_version": 1, "sources": []},
                device=torch.device("cpu"),
                starting_state={
                    "optimizer_step": 1,
                    "micro_step": 3,
                    "consumed_tokens": 48,
                },
            )
        state = report["training_state"]
        self.assertEqual(state["optimizer_step"], 2)
        self.assertEqual(state["micro_step"], 4)
        self.assertEqual(state["consumed_tokens"], 64)

    def test_final_partial_gradient_accumulation_is_applied(self) -> None:
        teacher = FixedTeacher(self.config.vocab_size)
        batches = [
            {
                "input_ids": torch.randint(3, self.config.vocab_size, (1, 16)),
                "labels": torch.randint(3, self.config.vocab_size, (1, 16)),
            }
            for _ in range(3)
        ]
        config = AtomTrainingConfig(
            stage="dialogue",
            teacher_model_id="fixed-teacher",
            maximum_steps=2,
            maximum_wall_seconds=60,
            batch_size=1,
            gradient_accumulation=2,
            warmup_steps=0,
            precision="float32",
            log_interval=10,
            save_interval=10,
        )
        with tempfile.TemporaryDirectory(
            prefix="atom-english-final-accumulation-"
        ) as temporary:
            report = train_atom_english_stage(
                self.model,
                batches,
                teacher=teacher,
                config=config,
                output_directory=Path(temporary),
                tokenizer_manifest={"schema_version": 1, "vocab_size": 512},
                curriculum_manifest={"schema_version": 1, "sources": []},
                device=torch.device("cpu"),
            )
        state = report["training_state"]
        self.assertEqual(state["optimizer_step"], 2)
        self.assertEqual(state["micro_step"], 3)
        self.assertEqual(state["consumed_tokens"], 48)
        self.assertEqual(
            state["stop_reason"],
            "input_exhausted_after_final_accumulation",
        )

    def test_foundation_and_dialogue_tokenizers_share_vocabulary(self) -> None:
        curriculum = broad_english_curriculum()
        foundation = load_aligned_tokenizer(
            curriculum.foundation_tokenizer_id,
            revision=curriculum.foundation_tokenizer_revision,
            local_files_only=True,
        )
        dialogue = load_aligned_tokenizer(
            curriculum.tokenizer_id,
            revision=curriculum.tokenizer_revision,
            local_files_only=True,
        )
        alignment = validate_tokenizer_alignment(foundation, dialogue)
        self.assertEqual(alignment["vocab_size"], 49_152)
        self.assertNotEqual(
            alignment["foundation_special_tokens"],
            alignment["dialogue_special_tokens"],
        )
        messages = [
            {"role": "user", "content": "Explain gravity."},
            {
                "role": "assistant",
                "content": "Gravity is an interaction between masses.",
            },
        ]
        encoded = encode_dialogue(
            dialogue,
            messages,
            sequence_length=64,
        )
        self.assertEqual(encoded["input_ids"].shape, (64,))
        self.assertGreater(int((encoded["labels"] != -100).sum()), 0)
        self.assertGreater(int((encoded["labels"] == -100).sum()), 0)
        full_chat = chat_token_ids(
            dialogue,
            messages,
            add_generation_prompt=False,
        )[:65]
        supervised = torch.nonzero(
            encoded["labels"] != -100,
            as_tuple=False,
        ).flatten()
        for position in supervised.tolist():
            self.assertEqual(
                int(encoded["labels"][position]),
                full_chat[position + 1],
            )

    def test_competence_gate_rejects_unscaled_evidence(self) -> None:
        benchmarks = {
            "wikitext": {"perplexity": 1.1},
            "lambada": {"exact_accuracy": 1.0, "examples": 10},
            "blimp": {"accuracy": 1.0, "examples": 10},
            "hellaswag": {"accuracy": 1.0, "examples": 10},
            "ifeval": {
                "strict_prompt_accuracy": 1.0,
                "loose_prompt_accuracy": 1.0,
                "examples": 10,
            },
            "long_context": passing_long_context_result(),
        }
        result = language_competence_gate(benchmarks)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["lambada_scale"])

    def test_competence_gate_requires_264k_and_512k_context_results(self) -> None:
        benchmarks = {
            "wikitext": {"perplexity": 20.0},
            "lambada": {"exact_accuracy": 0.40, "examples": 500},
            "blimp": {"accuracy": 0.75, "examples": 500},
            "hellaswag": {"accuracy": 0.45, "examples": 500},
            "ifeval": {
                "strict_prompt_accuracy": 0.50,
                "loose_prompt_accuracy": 0.60,
                "examples": 541,
            },
            "long_context": passing_long_context_result(),
        }
        benchmarks["long_context"]["lengths"]["264000"]["accuracy"] = 0.50
        result = language_competence_gate(benchmarks)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["long_context_264000_accuracy"])
        benchmarks["long_context"]["lengths"].pop("524288")
        with self.assertRaisesRegex(ValueError, "length"):
            language_competence_gate(benchmarks)

    def test_kaggle_input_discovery_prefers_same_stage_progress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atom-english-inputs-") as temporary:
            root = Path(temporary)
            candidates = (
                ("foundation-old", "foundation", 10_000, 10, "a" * 64),
                ("foundation-new", "foundation", 20_000, 20, "b" * 64),
                ("dialogue", "dialogue", 1_000, 2, "c" * 64),
                ("context", "context", 100, 3, "d" * 64),
            )
            for name, stage, tokens, step, manifest_hash in candidates:
                directory = root / name
                directory.mkdir()
                payload = {
                    "schema_version": 1,
                    "architecture": "atom-generative-english-core-v1",
                    "training_config": {"stage": stage},
                    "training_state": {
                        "consumed_tokens": tokens,
                        "optimizer_step": step,
                    },
                    "manifest_hash": manifest_hash,
                }
                (directory / "checkpoint_manifest.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            self.assertEqual(
                discover_input_checkpoint("foundation", input_root=root),
                root / "foundation-new",
            )
            self.assertEqual(
                discover_input_checkpoint("dialogue", input_root=root),
                root / "dialogue",
            )
            self.assertEqual(
                discover_input_checkpoint("context", input_root=root),
                root / "context",
            )
            self.assertEqual(
                discover_input_checkpoint("evaluation", input_root=root),
                root / "context",
            )

    def test_dialogue_stage_rejects_immature_foundation(self) -> None:
        immature = {
            "training_config": {"stage": "foundation"},
            "training_state": {
                "consumed_tokens": FOUNDATION_TARGET_TOKENS - 1,
            },
        }
        mature = {
            "training_config": {"stage": "foundation"},
            "training_state": {
                "consumed_tokens": FOUNDATION_TARGET_TOKENS,
            },
        }
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_training_stage_admission("dialogue", None)
        with self.assertRaisesRegex(ValueError, "full token target"):
            validate_training_stage_admission("dialogue", immature)
        validate_training_stage_admission("dialogue", mature)

    def test_context_stage_requires_dialogue_maturity(self) -> None:
        immature = {
            "training_config": {"stage": "dialogue"},
            "training_state": {
                "optimizer_step": DIALOGUE_TARGET_OPTIMIZER_STEPS - 1,
            },
        }
        mature = {
            "training_config": {"stage": "dialogue"},
            "training_state": {
                "optimizer_step": DIALOGUE_TARGET_OPTIMIZER_STEPS,
            },
        }
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_training_stage_admission("context", None)
        with self.assertRaisesRegex(ValueError, "optimizer-step target"):
            validate_training_stage_admission("context", immature)
        validate_training_stage_admission("context", mature)

    def test_evaluation_is_bound_to_the_exact_checkpoint(self) -> None:
        benchmarks = {
            "wikitext": {"perplexity": 20.0},
            "lambada": {"exact_accuracy": 0.40, "examples": 500},
            "blimp": {"accuracy": 0.75, "examples": 500},
            "hellaswag": {"accuracy": 0.45, "examples": 500},
            "ifeval": {
                "strict_prompt_accuracy": 0.50,
                "loose_prompt_accuracy": 0.60,
                "examples": 541,
            },
            "long_context": passing_long_context_result(),
        }
        checkpoint_hash = "a" * 64
        with tempfile.TemporaryDirectory(
            prefix="atom-english-evaluation-"
        ) as temporary:
            directory = Path(temporary)
            write_language_evaluation(
                directory,
                benchmarks,
                model_parameter_count=self.model.parameter_count(),
                checkpoint_manifest_hash=checkpoint_hash,
                sources={"external": True},
            )
            loaded = load_language_evaluation(
                directory / "language_evaluation.json",
                checkpoint_manifest_hash=checkpoint_hash,
                model_parameter_count=self.model.parameter_count(),
            )
            self.assertTrue(loaded["gate"]["passed"])
            with self.assertRaisesRegex(ValueError, "another checkpoint"):
                load_language_evaluation(
                    directory / "language_evaluation.json",
                    checkpoint_manifest_hash="b" * 64,
                    model_parameter_count=self.model.parameter_count(),
                )

    def test_generated_kaggle_source_executes_the_same_self_tests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atom-english-bundle-") as temporary:
            output = Path(temporary)
            manifest = build_bundle(output)
            source = (output / "atom_generative_english_kaggle.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("CausalAtomGraph", source)
            self.assertIn("topk_tail_distillation_loss", source)
            self.assertIn("evaluate_blimp", source)
            self.assertIn("evaluate_ifeval", source)
            self.assertIn("evaluate_long_context", source)
            self.assertIn("train_atom_english_context_stage", source)
            self.assertIn("_ensure_kaggle_pascal_torch", source)
            self.assertIn("torch==2.7.1", source)
            self.assertEqual(manifest["execution"]["profile"], "scale-227m")
            self.assertEqual(
                manifest["execution"]["model_context_tokens"],
                524_288,
            )
            self.assertEqual(
                manifest["execution"]["required_context_evaluation_tokens"],
                [32_768, 65_536, 131_072, 264_000, 524_288],
            )
            self.assertEqual(
                manifest["execution"]["foundation_target_tokens"],
                4_915_200_000,
            )
            self.assertEqual(
                manifest["execution"]["default_gradient_accumulation"],
                16,
            )
            self.assertEqual(
                manifest["execution"]["dialogue_target_optimizer_steps"],
                DIALOGUE_TARGET_OPTIMIZER_STEPS,
            )
            dialogue_manifest = build_bundle(
                output / "dialogue",
                kernel_id="jessealicea/atom-generative-english-dialogue-v1",
                title="Atom Generative English Dialogue v1",
                default_stage="dialogue",
                kernel_sources=("jessealicea/atom-generative-english-v1",),
            )
            dialogue_metadata = json.loads(
                (output / "dialogue" / "kernel-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                dialogue_manifest["execution"]["default_stage"],
                "dialogue",
            )
            self.assertEqual(
                dialogue_metadata["kernel_sources"],
                ["jessealicea/atom-generative-english-v1"],
            )
            continuation_manifest = build_bundle(
                output / "continuation",
                kernel_id=(
                    "jessealicea/atom-generative-english-foundation-continue-v1"
                ),
                title="Atom Generative English Foundation Continue v1",
                default_stage="foundation",
                default_sequence_tokens=512,
                default_gradient_accumulation=32,
                kernel_sources=("jessealicea/atom-generative-english-v1",),
            )
            continuation_source = (
                output / "continuation" / "atom_generative_english_kaggle.py"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                continuation_manifest["execution"]["default_sequence_tokens"],
                512,
            )
            self.assertEqual(
                continuation_manifest["execution"]["default_gradient_accumulation"],
                32,
            )
            self.assertIn(
                "ATOM_ENGLISH_DEFAULT_SEQUENCE_LENGTH\", '512'",
                continuation_source,
            )
            self.assertIn(
                "ATOM_ENGLISH_DEFAULT_GRADIENT_ACCUMULATION\", '32'",
                continuation_source,
            )
            context_manifest = build_bundle(
                output / "context",
                kernel_id="jessealicea/atom-generative-english-context-v1",
                title="Atom Generative English Context v1",
                default_stage="context",
                kernel_sources=("jessealicea/atom-generative-english-dialogue-v1",),
            )
            context_metadata = json.loads(
                (output / "context" / "kernel-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                context_manifest["execution"]["default_stage"],
                "context",
            )
            self.assertEqual(
                context_manifest["execution"]["context_target_optimizer_steps"],
                CONTEXT_TARGET_OPTIMIZER_STEPS,
            )
            self.assertEqual(
                context_metadata["kernel_sources"],
                ["jessealicea/atom-generative-english-dialogue-v1"],
            )
            evaluation_manifest = build_bundle(
                output / "evaluation",
                kernel_id="jessealicea/atom-generative-english-evaluation-v1",
                title="Atom Generative English Evaluation v1",
                default_mode="evaluate",
                kernel_sources=("jessealicea/atom-generative-english-context-v1",),
            )
            evaluation_source = (
                output / "evaluation" / "atom_generative_english_kaggle.py"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                evaluation_manifest["execution"]["evaluation_dependency"],
                "lm_eval[ifeval]==0.4.12",
            )
            self.assertIn("_ensure_kaggle_language_evaluation", evaluation_source)
            self.assertIn("lm_eval[ifeval]==0.4.12", evaluation_source)
        self.assertTrue(run_self_test()["passed"])

    def test_wiki_rag_and_side_view_bind_generated_language(self) -> None:
        curriculum = broad_english_curriculum()
        summary = {
            "schema_version": 1,
            "runtime": "atom-english-kaggle-runner-v1",
            "mode": "train",
            "parameter_count": self.model.parameter_count(),
            "training_report": {"training_state": {"consumed_tokens": 128}},
            "samples": [
                {
                    "prompt": "Explain causal memory.",
                    "response": (
                        "Causal memory retains relationships that survived "
                        "observation and correction."
                    ),
                    "prompt_tokens": 4,
                    "response_tokens": 11,
                }
            ],
        }
        graph = build_english_knowledge_graph(
            self.config,
            curriculum,
            run_summary=summary,
        )
        self.assertEqual(graph["wiki_runtime"], ATOM_ENGLISH_WIKI_RUNTIME)
        self.assertEqual(graph["rag_runtime"], ATOM_ENGLISH_RAG_RUNTIME)
        contexts = retrieve_english_knowledge(
            graph, "causal English corpus teacher", limit=8
        )
        self.assertGreater(len(contexts), 0)
        document = render_english_generation_artifact(summary, graph)
        self.assertIn(ATOM_ENGLISH_SIDE_VIEW_RUNTIME, document)
        self.assertIn("render_english_generation_artifact", document)
        self.assertIn("Causal memory retains", document)

        chat_summary = dict(summary)
        chat_summary["runtime"] = ATOM_ENGLISH_CHAT_RUNTIME
        chat_summary["mode"] = "chat"
        chat_summary["evaluation"] = {"gate": {"passed": True}}
        chat_document = render_english_generation_artifact(
            chat_summary,
            graph,
        )
        self.assertIn("external language gate passed", chat_document)
        self.assertIn("Causal memory retains", chat_document)

    def test_required_evidence_policy_abstains_before_generation(self) -> None:
        curriculum = broad_english_curriculum()
        graph = build_english_knowledge_graph(self.config, curriculum)
        assets = AtomEnglishChatAssets(
            model=self.model,
            tokenizer=None,
            checkpoint_manifest={"manifest_hash": "a" * 64},
            evaluation={"gate": {"passed": True}},
            knowledge_graph=graph,
            device=torch.device("cpu"),
        )
        session = AtomEnglishChatSession(
            assets,
            evidence_policy="required",
            maximum_new_tokens=16,
        )
        response = session.reply("zzzxxyy qqqvvv")
        self.assertEqual(response["evidence_status"], "insufficient")
        self.assertIn("do not have evidence", response["response"])
        self.assertEqual(len(session.samples), 1)

    def test_curriculum_uses_the_full_stream_and_pinned_teacher(self) -> None:
        curriculum = broad_english_curriculum()
        foundation = curriculum.stage("foundation")
        self.assertEqual(
            {(source.dataset_id, source.dataset_config) for source in foundation},
            {
                ("HuggingFaceFW/fineweb-edu", "default"),
                ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"),
            },
        )
        dialogue = curriculum.stage("dialogue")[0]
        self.assertEqual(dialogue.dataset_id, "HuggingFaceTB/smoltalk")
        self.assertEqual(dialogue.dataset_config, "all")
        self.assertNotEqual(dialogue.revision, "main")
        self.assertEqual(
            DIALOGUE_TARGET_OPTIMIZER_STEPS,
            (2 * SMOLTALK_TRAIN_ROWS + 15) // 16,
        )
        self.assertEqual(
            curriculum.base_teacher_id,
            "HuggingFaceTB/SmolLM2-1.7B",
        )
        self.assertNotEqual(curriculum.base_teacher_revision, "main")

    def test_architecture_manifest_is_not_a_bounded_parser(self) -> None:
        manifest = atom_english_architecture_manifest(self.config, self.model)
        self.assertEqual(
            manifest["sequence_engine"],
            "sparse-directed-temporal-causal-graph",
        )
        self.assertEqual(
            manifest["generation_execution"],
            "recurrent-cached-graph-state",
        )
        self.assertEqual(
            manifest["context_system"]["maximum_tokens"],
            self.config.max_seq_len,
        )
        self.assertEqual(
            manifest["context_system"]["persistent_memory_growth"],
            "logarithmic",
        )
        self.assertNotIn("query_type", json.dumps(manifest))

    def test_runtime_declarations_bind_chat_knowledge_and_side_view(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        knowledge = json.loads(
            (root / "ai-runtime-knowledge.json").read_text(encoding="utf-8")
        )
        side_view = json.loads(
            (root / "ai-artifact-side-view.json").read_text(encoding="utf-8")
        )
        registry = json.loads(
            (root / "ai-runtime-registry.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registry["active_runtime"], "generative-english")
        self.assertIn("causal-live", registry["runtimes"])
        self.assertIn("generative-english", registry["runtimes"])
        self.assertEqual(
            knowledge["runtime_entrypoint"],
            "atom_english_chat.py",
        )
        self.assertEqual(
            knowledge["wiki_graph"]["runtime_marker"],
            "ATOM_ENGLISH_WIKI_RUNTIME",
        )
        self.assertEqual(
            knowledge["rag"]["runtime_marker"],
            "ATOM_ENGLISH_RAG_RUNTIME",
        )
        self.assertEqual(
            side_view["side_view"]["runtime_marker"],
            "ATOM_ENGLISH_SIDE_VIEW_RUNTIME",
        )
        self.assertEqual(
            side_view["side_view"]["artifact_binding_marker"],
            "render_english_generation_artifact",
        )


if __name__ == "__main__":
    unittest.main()
