# Atom Generative English — User Notes

Last refreshed: July 25, 2026 at 12:30 PM EDT

This is the plain-English record of what the language project is doing, what
changed, and what has actually been measured. The technical companion is
`ATOM_GENERATIVE_ENGLISH_DEV_NOTES.md`.

## Distillation has been retired

The 227M language model and its 1.7B teacher are no longer the active path.
Their multi-billion-token distillation schedule recreated the same traditional
training bottleneck Atom is meant to escape. Fixing the Kaggle continuation
would not fix that architectural mistake.

The training code, generated Kaggle runners, test, and compact evidence were
moved out of the active project paths into
`retired/generative-english-distillation/`. They remain available for audit,
but the project will not resume or advance them.

The active runtime is again the causal-live Atom path: explicit causal
experience, trusted outcomes, durable graph memory, wiki/RAG retrieval, and
fail-closed abstention. The Atom-native application builder remains preserved.
No remote Kaggle resource and no GitHub backup was changed.

Everything below this section is historical evidence from the retired
experiment, not the current development direction.

## The goal

The goal is not to reproduce a conventional LLM with different names. It is to
build a smaller, faster, more observable language system around:

- an explicit causal graph;
- phase relationships;
- persistent memory;
- controlled disorder and stabilization;
- the seven universe primitives;
- exact experience-based recall;
- ordinary English communication.

It should eventually combine learned language behavior with the evolving Atom
database rather than burying all knowledge inside frozen neural weights.

## What was wrong

The model was configured to accept as many as 524,288 tokens, but accepting a
long prompt is not the same as understanding or remembering it.

The first memory system compressed old text into averages. That worked for a
general impression but destroyed exact ordered information. A small test model
learned that an answer should look like `ORBIT-...`, but it replaced every
unseen identifier with the same invented hexadecimal value.

That was the important failure: the model understood the answer shape but
hallucinated the data.

## What changed

The memory now has three layers:

1. **Recent exact memory.** Nearby tokens remain available exactly.
2. **Persistent conceptual memory.** Older regions become learned summaries
   and selected exact landmarks.
3. **Observed transition memory.** When the active context has already shown
   that a particular token pattern is followed by a particular token, the
   system records that causal relationship.

The third layer is the direct defense against inventing an identifier. If the
context says:

```text
ORBIT-01CE5761
```

and the model later reaches `ORBIT-`, the system can follow the continuation it
actually observed rather than guessing the digits.

It still does not copy blindly. The language model must consider the observed
continuation reasonably plausible. If the remembered token is far outside the
model's current judgment, it is rejected. This stopped ordinary repeated words
from hijacking the answer.

## Why this is still causal

The transition memory never reads the future. A relationship is recorded only
after both the pattern and its next token have already appeared. It can then be
used at a later point in the same context.

It also cannot introduce a fact that was never present. It can repeat or
continue an observed relationship, but it cannot manufacture external
knowledge.

## What the local experiments showed

Before the change, the small trained model scored 0 out of 4 on unseen values.
It returned the right prefixes with an invented fixed suffix.

With the current design it returned all four held-out answers exactly:

| Test | Correct answer | Model answer |
| --- | --- | --- |
| Find one stored value | `ORBIT-01CE5761` | `ORBIT-01CE5761` |
| Choose among several records | `BIRCH-01CE7650` | `BIRCH-01CE7650` |
| Preserve event order | `SECOND-01CE953F` | `SECOND-01CE953F` |
| Use the latest updated state | `FINAL-01CEB42E` | `FINAL-01CEB42E` |

The small model therefore improved from 0/4 to 4/4 on that held-out diagnostic.

A second experiment isolated the long-distance memory. It placed an observed
relationship near the beginning of a sequence, repeated the key at the very
end of exactly 524,288 tokens, and asked for the continuation. The memory
returned the correct token in about 3.90 seconds.

## What that does and does not mean

The 524,288-token experiment proves that the new exact transition mechanism can
carry an observed relationship across the declared distance without quadratic
attention.

It does not yet prove that the full trained English model understands arbitrary
books, conversations, mathematics, or causal arguments across 512K tokens.
That requires the large foundation, dialogue, and context training stages,
followed by the real four-family evaluation at both 264K and 512K.

This distinction matters:

- **Measured now:** exact transition memory reaches 512K.
- **Measured now:** the tiny trained model answers all four 512-token tests.
- **Not measured yet:** full-model semantic recall at 264K and 512K.
- **Not measured yet:** full English competence and natural multi-turn quality.

## Memory size

The new exact transition memory grows with the number of tokens, not with every
possible pair of tokens. For a 524,288-token context, its declared ceiling is
1,572,864 transition entries.

That is more state than the logarithmic conceptual summaries, but far less than
a dense system that compares every token with every other token. The actual
repeated-filler boundary test needed only 43 transition entries; natural text
will use more.

## Current Kaggle state

The first private foundation notebook
`jessealicea/atom-generative-english-v1` has completed on Kaggle. Its saved
checkpoint passed the project's full file-hash and manifest-hash validation.
It contains 20,220,928 trained tokens and 2,468 optimizer steps.

That is a valid first training segment, not a finished English model. The
foundation target is 4,915,200,000 tokens, so dialogue admission remains
closed. The private foundation-continuation notebook was started only after
the completed checkpoint passed those checks, and it is currently running.

The local source contains five linked Kaggle runners:

1. foundation;
2. foundation continuation;
3. dialogue;
4. long-context conditioning;
5. external evaluation.

All five locally generated runners passed their internal checks after the
memory change. Dialogue, long-context conditioning, and external evaluation
were not started early. They remain dependent on the real counters and
hash-bound outputs from the earlier stages.

A timestamped copy of the active Kaggle notebook is stored at:

```text
kaggle_notebook_backups/atom-generative-english-v1-20260724-200400/
```

## Verification record

After the memory changes:

- 200 Python tests passed;
- Python lint checks passed;
- 29 Rust tests passed;
- Rust formatting passed;
- Rust warning checks passed;
- all five generated Kaggle runners passed their self-tests.

The machine-readable experiment record is:

```text
local-results/atom-english-long-context-recall-20260724.json
```

The source archive made immediately before these notes is:

```text
backups/atom-generative-english-long-context-20260724-200510.zip
```

Its SHA-256 is:

```text
FC0EDE812ABE93B8572493A62E058A1C3FC961E7C293E8B87D1E80FD943AF2A3
```

The post-advancement source archive for this refresh is:

```text
backups/atom-generative-english-source-20260725-043552.zip
```

The machine-readable foundation validation is:

```text
kaggle-results/generative-english-foundation-v1/verification.json
```

Neither final readiness threshold has been met. There is no admitted
multi-turn English conversation yet, and there is no same-runtime proof that
such a conversation causally produced, compiled, tested, rendered, and
explained a generated Atom/Rust/Svelte application.

## Notes policy

These user notes should stay focused on behavior, evidence, and honest limits.
The developer notes should retain implementation details, exact values,
lineage, invariants, commands, failures, and hashes.

When a future change affects what the system can do or what has been measured,
both files should be refreshed together. A configured feature should never be
described here as a demonstrated capability until its real evaluation has run.

## Foundation continuation repaired on 2026-07-25

The first retry had a real GPU-memory failure, not a training-data or
checkpoint-hash failure. Kaggle's Tesla P100 had only about 143 MiB free when
the model needed another 120 MiB for its persistence path.

The continuation now keeps the exact trained 2,048-token model and optimizer
state. Expansion to 524,288 tokens waits for the dedicated long-context stage,
where it belongs. The continuation also uses shorter 512-token pieces and
accumulates twice as many pieces before each update, so the amount of raw text
per optimizer update stays the same while peak activation memory is lower.

All 31 focused Python integration tests and the generated Kaggle runner
self-test passed. Private Kaggle version 3 initially ran beyond the exact time
where version 2 ran out of memory, but a later live check found that version 3
also ended in error. Its repaired settings were active, then the process was
externally killed about 106 seconds after training started. It produced no
completed optimizer counter or new checkpoint.

The continuation therefore is not repaired yet. Dialogue admission is still
closed, and neither the real conversation proof nor the
conversation-to-built-application proof has been demonstrated.

## GitHub safekeeping

A source-only recovery copy belongs in the dedicated private Lucerna Labs
repository:

```text
https://github.com/Lucerna-Labs/atom-lora
```

The earlier safekeeping branch inside `atom-vibe-coder` was the wrong location
because that repository is a separate bulky working application. The dedicated
repository keeps the projects separate. The mistaken branch has now been
removed, and the working application's `main` branch was not changed. The copy
includes source, tests, architecture documents, these paired notes, and compact
verification records. It deliberately excludes model weights, optimizer files,
large outputs, build caches, logs, and credentials.

The replacement paired source archive for this refresh is:

```text
backups/atom-generative-english-source-20260725-120502.zip
```
