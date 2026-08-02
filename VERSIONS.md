# Versions

One-line summary per accelerator and architecture version. The full
measured evidence — timings, hashes, audit checks, replay verification —
lives in `docs/versions/kaggle/` and `docs/notes/DEVELOPER_NOTES.md`.

This file is the index, not the evidence. Use it to pick a starting point;
follow the link to read the actual record.

## Causal world

Kaggle runs of `atom_causal_world_experiment.py` against the
`atom_causal_world_kaggle.py` private kernel. The headline metric is
*held-out transfer coverage* against an unseen-regime truth set regenerated
from two independent treated/control simulations, paired with
*selective-accuracy*, *false-assertion rate*, *paraphrase consistency*,
and *cross-platform replay*.

| Version | Status | Headline | Evidence |
|---|---|---|---|
| V2 | superseded | Predates the conservation repair; accelerator dynamics produced no genuine two-hop chain. | — |
| V3 | superseded | First TPU end-to-end: 16 shards, 131,072 evidence rows, all 21 gates passed. Conservation repair pending. | — |
| V4 | rejected | Failed axis coverage and conservation; fail-closed verifier rejects the run. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V4_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V4_EVIDENCE.md) |
| V5 | superseded | Corrected schedule across all nine axes, 524,288 rows. Predates conservation repair. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V5_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V5_EVIDENCE.md) |
| V6 | accepted | Conservation repair landed: max invariant error fell from 1.0 to `3.82e-7`. 25 gates, 16 deterministic shard replays, 9/9 workflow passed. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V6_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V6_EVIDENCE.md) |
| V7 | superseded | 1,048,576 rows; surfaced a 1-bit Linux/Windows float-representation mismatch in a diagnostic similarity value. | — |
| V8 | accepted | Diagnostics projected to a stable 12-decimal representation; exact response replay, all transfer gates, graph-RAG, and side-view binding all pass. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V8_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V8_EVIDENCE.md) |
| V9 | accepted | Metaplastic policy search end-to-end: 500 policies evaluated, selected prior 0.4, decreasing threshold 0.75, increasing 0.95; 64.93% held-out coverage, 92.51% selective accuracy, 100% paraphrase consistency. Independent Windows verifier 59/59. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V9_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V9_EVIDENCE.md) |
| V10 | superseded | Expanded validation to 24 programs but exposed accelerator-local evaluator provenance. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V10_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V10_EVIDENCE.md) |
| V11 | superseded | Corrected provenance and policy identity on fixed-precision decimal arithmetic. 65/66 audit checks. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V11_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_V11_EVIDENCE.md) |

## Causal world (GPU profile)

Single Tesla P100-PCIE-16GB. Same workload as the TPU profile but with
`jit` instead of `pmap`. Closes the cross-platform replay boundary that
the TPU versions left open.

| Version | Status | Headline | Evidence |
|---|---|---|---|
| GPU V1 | superseded | First GPU end-to-end: 2.87× faster than TPU V11, but inherited V11's policy-digest tail. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V1_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V1_EVIDENCE.md) |
| GPU V2 | superseded | Bound the full 5,000-policy projection lattice; isolated residual Wilson-score unrounded tails. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V2_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V2_EVIDENCE.md) |
| GPU V3 | superseded | 67/68 checks; two pair-motif log-likelihoods separated by `1e-12`. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V3_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V3_EVIDENCE.md) |
| GPU V4 | superseded | 68/69; retained the response-trace log tails by design. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V4_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V4_EVIDENCE.md) |
| GPU V5 | accepted | Moved all risk math onto fixed-precision decimal arithmetic. 1,048,576 rows in 117.67s on one P100. Independent Windows reconstruction 69/69 — closes the cross-platform replay boundary. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V5_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V5_EVIDENCE.md) |
| GPU V7 | accepted | Added the seven-domain typed formal substrate; full causal-world workload + 7,680 formal cases (1,920 held out). 1,048,576 rows in 117.60s. Independent reconstruction 72/72. | [`docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V7_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CAUSAL_WORLD_GPU_V7_EVIDENCE.md) |

## Other accelerator runs

| Version | Status | Headline | Evidence |
|---|---|---|---|
| Coding Platform V2 | historical | Atom-coding-harness accelerator run; superseded by the desktop operator path. | [`docs/versions/kaggle/KAGGLE_CODING_PLATFORM_V2_EVIDENCE.md`](docs/versions/kaggle/KAGGLE_CODING_PLATFORM_V2_EVIDENCE.md) |

## Local-only runs

These were not promoted to Kaggle. They are recorded in
`docs/notes/DEVELOPER_NOTES.md` and the corresponding `*_experiment.py`
source files.

- **English-language v1, v2** — synthetic evidence-bound field language.
  25,161-parameter Atom field, 0.916667 joint accuracy on 96 untouched
  transfer compositions, 100% paraphrase consistency. Bounded synthetic
  domain, not open-domain chatbot.
- **Lifelong neural-language field v2** — 30,012-parameter Atom field,
  factorized cognitive memory, Atom-composed homeostatic governor. 0.937500
  joint accuracy on 96 held-out transfer compositions. Zero-shot language
  result is intentionally negative.
- **Homeostatic governor** — chaos becomes a deterministic feedback-
  controlled quantity. The 232-event deterministic stream is the
  experimental reference.

## Phase markers

| Phase | Status | Reference |
|---|---|---|
| Phase 5 (V3 evidence kernel + interactive) | superseded | `docs/notes/DEVELOPER_NOTES.md` |
| Phase 6 (V3 + permissioned hands) | superseded | `docs/notes/DEVELOPER_NOTES.md` |
| Phase 7 (Desktop Phase 7 — current active product) | active | [`docs/roles/ATOM_HARNESS_DESKTOP.md`](docs/roles/ATOM_HARNESS_DESKTOP.md) |

## Retired

- **`retired/generative-english-distillation/`** — the 227M student / SmolLM2-1.7B
  teacher distillation pipeline. Its fixed multi-billion-token schedule
  reproduced the conventional train-a-language-model bottleneck this
  project is intended to escape. Preserved for audit, not active runtime.
