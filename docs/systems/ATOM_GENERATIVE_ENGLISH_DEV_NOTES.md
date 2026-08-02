# Atom Generative English — Developer Notes

Last refreshed: 2026-07-25 12:30 EDT

These notes are the technical handoff record for the causal-graph English
runtime. Update this document whenever model semantics, training admission,
context state, evaluation thresholds, Kaggle lineage, or measured results
change. Update `ATOM_GENERATIVE_ENGLISH_USER_NOTES.md` in the same change when
the change affects behavior visible to the operator.

## Retirement decision

On 2026-07-25 the operator retired the complete teacher/student generative
English track. The 227M Atom causal-language student, SmolLM2-1.7B teacher
distillation loop, fixed 4,915,200,000-token admission gate, staged
foundation/dialogue/context/evaluation chain, and Kaggle continuation workflow
are no longer part of the active architecture.

The reason is architectural, not merely operational: continuing the fixed
token schedule would reproduce the conventional large-language-model training
bottleneck the Atom project is intended to avoid. The 4.9152B gate was a
configured schedule (`300,000 * 16,384` tokens), not a measured requirement
established by external language admission.

The retired source, generated Kaggle bundles, test, architecture contract, and
compact evidence now live under
`retired/generative-english-distillation/`. They are preserved for audit but
are not importable from the active root, registered as an active runtime, or
authorized for resubmission. Remote Kaggle resources were left untouched.

`ai-runtime-registry.json` now restores `causal-live` as the active runtime.
The wiki/RAG and side-view declarations point to the causal-live entrypoint and
its real artifact path. The Atom-native application builder remains available
as a separate registered runtime.

The historical sections below document what was attempted and measured before
retirement. They must not be read as a current plan to resume distillation.

## Evidence boundary

The source implements a 524,288-token context ceiling, but configuration is not
evidence of useful recall. Current evidence establishes:

- exact equivalence among parallel, recurrent, and chunked execution on the
  verification profile;
- gradient flow through a trainable tail over a frozen streamed prefix;
- exact ordered landmark retention across completed persistence chunks;
- 4/4 exact held-out answers from the locally trained 512-token diagnostic;
- one correct symbolic transition retrieval across exactly 524,288 tokens;
- bounded state at the declared 524,288-token boundary;
- clean Python and Rust verification;
- successful self-tests from every generated Kaggle runner;
- a completed, hash-verified first foundation checkpoint containing 20,220,928
  consumed tokens and 2,468 optimizer steps.

Current evidence does not establish full-model English comprehension at
264,000 or 524,288 tokens. The completed first foundation checkpoint is valid
continuation input but is 4,894,979,072 tokens short of dialogue admission.
Foundation continuation, dialogue, context conditioning, external evaluation,
and an exercised real conversation still must complete before that claim can
open.

## Defect history and design decisions

The context ceiling originally exceeded the demonstrated semantic reach.
Functional experiments exposed four distinct problems:

| Iteration | Measured behavior | Engineering conclusion |
| --- | --- | --- |
| Averaged persistent summaries | The tiny model learned answer prefixes but generated a fixed hexadecimal suffix, scoring 0/4. | A mean vector cannot preserve arbitrary ordered symbols. |
| Exact landmarks without usable order | Exact vectors survived a completed chunk, but the model still generated a different fixed suffix, scoring 0/4. | Retaining symbols is insufficient if their original spatial relationship is discarded before attention. |
| Position-bearing landmarks | Original positions were phase-bound to landmark keys and the persistence gate received a straight-through gradient path. The tiny neural path still preferred a learned constant suffix. | Neural compression alone remained unreliable for exact unseen identifiers at this scale. |
| Transition orders 2, 4, and 8 | Single-needle recall worked, but a two-token match could hijack ordinary answer prefixes, producing 1/4. | Very short symbolic contexts are too ambiguous. |
| Transition orders 4, 8, and 16 without an acceptance band | All four expected values appeared, but the cache continued into record punctuation and unrelated trailing text. | Observed continuation is useful evidence, but it must not automatically overrule the neural output. |
| Four-logit neural compatibility band | All four held-out families returned only the expected value, scoring 4/4 exact. | Symbolic recall should constrain a plausible neural choice, not replace language judgment. |

The resulting design uses two complementary forms of older-context memory:

1. neural topological persistence for semantic compression and learned
   recognition;
2. deterministic symbolic transition edges for exact continuations already
   observed in the active context.

## Sequence engine

`AtomCausalLanguageModel` remains an autoregressive causal-graph model. It does
not use a Transformer or Mamba backbone.

Each language block contains:

- a sparse causal predecessor graph;
- rotary phase locking for temporal relationships;
- local and explicitly dilated causal edges;
- query-recognized topological persistence;
- a nonlinear phase mixer;
- bounded conservative residual updates;
- homeostatic edge entropy and learned thermal control.

The model binds the seven root primitives as follows:

| Root | Runtime role |
| --- | --- |
| Radiation | Propagate selected edge values. |
| Dissipation | Retain or suppress updates. |
| Gravitation | Pull a query toward recognized predecessors. |
| Attraction/repulsion | Apply signed recognition gating. |
| Nucleation | Form sparse updates and durable landmarks. |
| Conservation | Bound residual energy. |
| Decay | Suppress increasingly distant ordinary edges. |

## Recent exact state

Each layer retains a fixed causal ring whose capacity is the largest configured
graph offset. The scale profile retains 6,144 recent tokens. This ring supplies
exact local and dilated predecessor states without storing a dense
half-million-token attention matrix.

The ring invariants are:

- capacity never grows after state construction;
- recurrent and chunked positions remain synchronized across layers;
- no edge may read a token at or after the current position;
- requesting a token after `max_seq_len` raises an error.

## Topological persistence

Completed 32-token regions enter a binary persistence hierarchy. Each occupied
level stores:

- learned weighted numerators and masses for semantic summaries;
- exact landmark values;
- landmark salience scores;
- original integer positions.

The scale profile uses eight summary slots and four exact landmarks per slot,
or 32 exact episodic candidates per occupied hierarchy level. Initial chunk
selection partitions positions across slots before ranking by learned salience.
This prevents every slot from selecting the same few tokens and guarantees
coverage of the completed chunk before hierarchy merges.

When two occupied regions merge:

- summary numerators and masses add;
- the two landmark banks are combined;
- learned salience chooses four survivors per slot;
- a deterministic position tie-break favors the newer candidate only when
  salience is otherwise equal;
- aggregate and landmark hierarchy fields are cleared or stored together.

At retrieval:

- semantic summaries are recognized across occupied hierarchy levels;
- exact landmarks are recognized against the current query;
- salience and a small recency term refine recognition;
- selected landmark keys receive rotary phase at their original positions;
- a straight-through scale is exactly one in the forward pass while allowing
  recognition and salience to receive gradients.

Neural hierarchy growth is logarithmic in the number of completed chunks.

## Symbolic causal transition memory

The symbolic cache records only relationships observed in the current token
history. It does not contain external facts and cannot manufacture a
continuation that was never observed.

For every new token:

1. suffixes ending at the previous token are hashed at orders 4, 8, and 16;
2. each suffix is bound to the newly observed continuation token;
3. the new token is appended to history;
4. the longest matching current suffix is queried for a possible next token.

The transition key uses a deterministic 64-bit mixing function combined with
the suffix order. A unique observation is stored as one integer. Conflicting
continuations promote the entry to a compact vote map.

The maximum number of transition entries is:

```text
max_seq_len × number_of_orders
524,288 × 3 = 1,572,864
```

This is bounded linear state, not quadratic pairwise attention. The actual
boundary diagnostic used repeated filler and occupied 43 transition entries.
Natural text can occupy more entries, up to the declared bound.

A retrieved continuation receives an 18-logit bonus only when its unmodified
neural logit is within 4.0 of the neural maximum. This compatibility band is a
critical safety and quality invariant:

- unrelated repeated phrases cannot force a highly implausible token;
- exact identifiers can override a close but incorrect digit or symbol;
- once the copied identifier ends, a strongly preferred end token defeats
  trailing punctuation from the original record.

## Context conditioning

The context stage is separate from foundation and dialogue training. Its
deterministic schedule covers:

```text
2,048
4,096
8,192
16,384
32,768
65,536
131,072
264,000
524,288
```

The target is 2,600 optimizer steps. The final five percent repeatedly includes
264,000 and 524,288 tokens.

Examples at or below the differentiable limit run end to end so the neural
persistence path can learn what should survive. Longer examples stream the
prefix without a full autograd tape and train the bounded query-and-answer
tail. The streamed state includes both neural persistence and symbolic
transitions.

Every training example belongs to one of four families:

- single-needle exact retrieval;
- multi-record selection;
- ordered-event retrieval;
- latest-state retrieval after updates.

## Stage and admission lineage

The private Kaggle chain is:

```text
atom-generative-english-v1
  -> atom-generative-english-foundation-continue-v1
  -> atom-generative-english-dialogue-v1
  -> atom-generative-english-context-v1
  -> atom-generative-english-evaluation-v1
```

Admission rules:

- dialogue requires the full foundation token target;
- context conditioning requires all 130,490 dialogue optimizer steps;
- evaluation requires a context-stage training output;
- chat requires a context-stage output and, by default, a hash-bound external
  language report whose gate passes.

Old foundation outputs remain loadable because the new persistence and symbolic
fields have serialized defaults and do not change learned parameter shapes.
Context expansion still rejects semantic configuration changes and only permits
one-way increases in causal reach.

As of the refresh time above,
`jessealicea/atom-generative-english-v1` reports
`KernelWorkerStatus.COMPLETE`. Its downloaded `atom-english/latest` checkpoint
passed `atom_english_training._verified_checkpoint_manifest`, including every
declared file hash and the canonical manifest hash. The training report and run
summary both bind to the same manifest:

- checkpoint manifest SHA-256:
  `ff7b8de322ca9574c149f91b72e4f4d95799e18272aab027aa5bbf31b4011cf2`;
- model SHA-256:
  `5b2bacf7780d64ec98e8df7b5b7534cb73d94c223c6e854a0aa8e641e01832df`;
- optimizer SHA-256:
  `ffc7c9a7d7d1ea7405cef3c774ec4669f4aff3b45563d430e4ba0372ebd89814`;
- consumed tokens: 20,220,928;
- optimizer steps: 2,468;
- stop reason: `wall_time_limit`;
- measured training throughput: 532.0145036097056 tokens/second.

The checkpoint is below the 4,915,200,000-token foundation target, so dialogue
admission remains closed. After the hash and counter checks passed, private
kernel version 1 of
`jessealicea/atom-generative-english-foundation-continue-v1` was submitted and
confirmed `KernelWorkerStatus.RUNNING`. Dialogue, context, and evaluation were
not submitted. The compact verification record is
`kaggle-results/generative-english-foundation-v1/verification.json`.

## Evaluation and measured results

The local diagnostic trained the verification-scale character model for 1,200
context optimizer steps over 1,200 examples at 512 tokens. Its held-out seeds
were separate from the training sequence.

| Family | Expected | Generated | Exact |
| --- | --- | --- | --- |
| Single needle | `ORBIT-01CE5761` | `ORBIT-01CE5761` | yes |
| Multiple records | `BIRCH-01CE7650` | `BIRCH-01CE7650` | yes |
| Sequence order | `SECOND-01CE953F` | `SECOND-01CE953F` | yes |
| State update | `FINAL-01CEB42E` | `FINAL-01CEB42E` | yes |

The separate boundary diagnostic:

- observed key `[401, 402, 403, 404]`;
- observed continuation `207`;
- repeated the key at the end of exactly 524,288 tokens;
- predicted `207`;
- took `3.9047874999814667` seconds;
- retained 524,288 history tokens and 43 transition entries.

This boundary measurement isolates the exact causal memory mechanism. It is not
a substitute for running the trained scale model through all four semantic
families at 264K and 512K.

The machine-readable result is
`local-results/atom-english-long-context-recall-20260724.json`.

## Verification snapshot

Fresh checks after the context-memory changes:

- Ruff: all project Python sources outside generated, backup, and temporary
  directories passed.
- Python: 200 tests passed.
- Rust formatting: passed.
- Rust Clippy with warnings denied: passed.
- Rust: 29 tests passed.
- Generated Kaggle foundation runner self-test: passed.
- Generated Kaggle foundation-continuation runner self-test: passed.
- Generated Kaggle dialogue runner self-test: passed.
- Generated Kaggle context runner self-test: passed.
- Generated Kaggle evaluation runner self-test: passed.

The five runners all exercised the core symbolic-transition check, context
schedule, external evaluation gate, wiki graph, graph RAG, and artifact side
view.

## Important files

| File | Responsibility |
| --- | --- |
| `atom_english_core.py` | Model configuration, causal graph, persistence, symbolic transitions, streaming, and generation. |
| `atom_english_context.py` | Context curriculum and bounded-tail training. |
| `atom_english_evaluation.py` | External language and long-context measurements. |
| `atom_english_kaggle.py` | Stage routing, admission, and Kaggle execution. |
| `atom_english_chat.py` | Context-stage chat admission and stateful conversation. |
| `atom_english_knowledge.py` | Runtime wiki graph and graph RAG. |
| `atom_english_side_view.py` | User-visible artifact rendering. |
| `atom-generative-english-architecture.json` | Machine-readable architecture contract. |
| `ATOM_GENERATIVE_ENGLISH.md` | Main architecture and operations document. |
| `ATOM_GENERATIVE_ENGLISH_USER_NOTES.md` | Plain-English operator record. |

## Current hashes and backups

At this refresh:

- `atom_english_core.py` SHA-256:
  `DD54454006A993889178A07F282AE418295BD2A285585851D66E738AA8C671A4`
- `atom_english_context.py` SHA-256:
  `8756D2309A8F308F787166D5DB498B5F513E4848477EDDCC80EF240FC044A019`
- diagnostic JSON SHA-256:
  `0D50011EF716F12A2BB6B9E6A89C94CEFEE59EEC109201E704C65F53113E80EB`
- pre-notes source archive:
  `backups/atom-generative-english-long-context-20260724-200510.zip`
- pre-notes source archive SHA-256:
  `FC0EDE812ABE93B8572493A62E058A1C3FC961E7C293E8B87D1E80FD943AF2A3`
- downloaded active Kaggle source:
  `kaggle_notebook_backups/atom-generative-english-v1-20260724-200400/`
- post-advancement source archive:
  `backups/atom-generative-english-source-20260725-043552.zip`

Create a new timestamped archive after changing these notes or any referenced
source, and record the newer archive in the next refresh.

## Open measurements

The following questions remain empirical:

- Does the trained scale model meet the 75% minimum at every required
  long-context length?
- How many symbolic transition entries does a natural 512K subword sequence
  occupy?
- What are prefill speed and peak memory at 264K and 512K on the actual Kaggle
  context runner?
- Does the four-logit compatibility band remain optimal after foundation and
  dialogue training?
- Does broad English evaluation remain within threshold after context
  conditioning?
- Does multi-turn chat exercise the expected language, RAG, abstention, and
  side-view paths with the exact evaluated training output?

Do not convert any of these open measurements into positive claims without the
corresponding hash-bound runtime result.

## Foundation-continuation CUDA repair (2026-07-25)

Private continuation version 2 converted the earlier unexplained external
`Killed` event into an actionable traceback. On a Tesla P100-PCIE-16GB,
PyTorch reported 15.75 GiB in use, 143.12 MiB free, and failed a 120 MiB
persistence allocation at process time `318.362437284`. The downloaded log is
109,784 bytes with SHA-256
`D0E7115C2A991A8DA664E5FD3FA4D3BF3D19AC5D836443F24779E9B43F939E3D`.

Two continuation-only corrections were made:

1. foundation and dialogue resumes retain the checkpoint's trained
   configuration and optimizer state; the 2,048-to-524,288 context migration
   and five new causal distances are deferred to the dedicated context stage;
2. the P100 continuation uses 512-token microbatches with gradient
   accumulation 32, preserving 16,384 raw tokens per optimizer update while
   reducing sequence-dependent activation pressure.

The real source checkpoint preflight retained manifest
`ff7b8de322ca9574c149f91b72e4f4d95799e18272aab027aa5bbf31b4011cf2`,
optimizer step 2,468, 20,220,928 consumed tokens, and the trained 2,048-token
graph. Ruff passed, all 31 generative-English integration tests passed, the
generated runner self-test passed, and bundle SHA-256
`3752148e0e40641be02c253235c297134c77f98daeda87574620c28206efd1ac`
matched its manifest.

Private kernel version 3 was submitted and remained
`KernelWorkerStatus.RUNNING` at `2026-07-25T09:13:33-04:00`, beyond the
version-2 OOM process-time boundary. A later current-status check found version
3 in `KernelWorkerStatus.ERROR`, so the earlier running observation was not a
successful repair proof. The repaired settings were active, but the process
was externally `Killed` 106.132174427 seconds after `stage_started`, before a
completed optimizer counter, checkpoint, or hash-bound output.

The machine-readable repair record is
`kaggle-results/generative-english-foundation-v1/continuation-repair-20260725-091333.json`.
The version-3 terminal record is
`kaggle-results/generative-english-foundation-v1/continuation-v3-failure-20260725-091633.json`;
its 99,253-byte source log has SHA-256
`98BCC4CEFDE43A5AB140DE9F1A1CAAD0114661147C78F1E764D9F7387C6B160B`.
Dialogue, context, evaluation, both readiness capabilities, and any positive
continuation claim remain closed.

## Lucerna Labs GitHub safekeeping (2026-07-25)

The canonical source-only recovery snapshot is the private dedicated repository
`Lucerna-Labs/atom-lora`. It contains the English/runtime source, Atom-native
application-builder source, Kaggle runners, focused tests, architecture
contracts, paired notes, and compact hash/counter evidence. Model weights,
optimizer state, build trees, large runtime outputs, logs, credentials, and
machine-local configuration are excluded.

The initially used `safekeeping/atom-lora-20260725` branch in
`Lucerna-Labs/atom-vibe-coder` was a placement error because that repository is
a bulky working application. After dedicated `atom-lora` commit
`abc9dfcbe758c610ac6faaaa9cb4df63a50c78a1` was verified on `main`, the
mistaken branch was removed. `atom-vibe-coder` `main` remained unchanged at
`cd3f5415348c5b5664833862e56a6944e26c5da4`.

The replacement paired source archive for this refresh is
`backups/atom-generative-english-source-20260725-120502.zip`.
