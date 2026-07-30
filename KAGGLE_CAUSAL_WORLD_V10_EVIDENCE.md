# Kaggle causal-world Version 10 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 10  
Canonical submitted-source SHA-256: `3568ec13a2a6130b3db2d1357f5fa4380ff0c5d5be7d6f689f12a32cde22c67e`  
Downloaded Kaggle wrapper SHA-256: `98bea332a00284c92a571245615e6138ea5130dcce3c2e47a68e92d859a4b069`

Version 10 added a direction-neutral context-factor graph, 55 pairwise world
motifs, a 5,000-policy metaplastic search, and Wilson-score risk eligibility.
The Kaggle process ran without a Python traceback and every in-notebook gate
passed. Independent Windows replay then passed 63 of 65 checks. This version is
therefore retained as a negative deterministic-replay result rather than as
accepted cross-platform evidence.

## Measured Kaggle execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`
- Shards: `0..15`, with 64 unique curriculum programs
- Evidence rows: `1048576`, evenly distributed across eight domains
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `337.1696675359984` seconds
- Maximum conservation invariant error: `3.824358714155096e-07`
- Learned laws: `50992` total, `6352` crystallized
- Experiment gates reported by Kaggle: `29/29`
- Policy gates reported by Kaggle: `15/15`
- Transfer gates reported by Kaggle: `18/18`
- Python traceback: absent

Model hash: `f00f7cb22c1da080b3553de1d105e6490df2be081916cfd81bcf45e95926466d`  
Report hash: `6eb3da20ee9fc7c81623f0fef35f2c4a351e48d55c1df3c25b4c4894c37a1121`  
Evidence hash: `ca52983735c1163f397117f84267171780ddea139ec392f5045b488e23ef4dc9`  
Resume cursor hash: `54914de7a727faefab169696c0d1aca5f4e8c8f4f5c446c880d6fa50bb3f1350`

## Validation-policy measurement

Validation used 24 programs excluded from training and final evaluation. It
generated 576 cases and 1,152 English turns. The search evaluated 5,000
policies, of which 557 met the validation risk contract.

The selected controls were:

- direction-prior power: `0.0`
- pair-motif power: `0.05`
- token-likelihood power: `0.75`
- decreasing consensus boundary: `0.75`
- increasing consensus boundary: `0.995`

The selected validation measurement was:

- coverage: `0.6371527777777778`
- correct assertions: `692`
- false assertions: `42`
- selective accuracy: `0.9427792915531336`
- false-assertion rate: `0.036458333333333336`
- safe-direction utility: `0.5642361111111112`
- overall 95% Wilson selective-error upper bound: `0.07644278058788155`

Validation truth hash: `aa686ab41efb868cce2ca9968eb3d2687f2a633896b27b25433f5f1550d43fdd`  
Selected policy hash: `26aecf03eb318911b11b040c519d3bea3c676d1c923f449f1bf71cda58e6f960`

## Sealed evaluation measurement

Evaluation used 12 programs excluded from both training and validation. Its 288
balanced cross-feature cases produced 576 inference turns. Exact condition
matching had zero coverage. The frozen context-factor policy reported:

- coverage: `0.6875`
- correct assertions: `368`
- false assertions: `28`
- selective accuracy: `0.9292929292929293`
- false-assertion rate: `0.04861111111111111`
- safe-direction utility: `0.5902777777777778`
- overall 95% Wilson selective-error upper bound: `0.10029308968159173`
- paraphrase consistency: `1.0`

By expected direction:

- decreasing: coverage `0.5428571428571428`, selective accuracy
  `0.9078947368421053`, upper bound `0.14866091062628103`
- increasing: coverage `0.8243243243243243`, selective accuracy
  `0.9426229508196722`, upper bound `0.09399263167327016`

These evaluation values were downloaded faithfully but are not promoted to
independently reproduced evidence because the strict replay result below
failed.

Evaluation truth hash: `781581d1162545ba9f397c40a6ba64cfe48eaa14d42528e38d808408230d87d4`  
Transfer report hash: `a7f00595c879311e01bb22dbd8f249358de3969fac8e3caad89582cc4ed7d777`

## Independent replay result

The verifier passed 63 of 65 checks. It regenerated the evaluation truth
exactly, replayed both exact and contextual 576-turn responses exactly,
recomputed both transfer evaluations exactly, preserved the side-view and
graph-RAG bindings, and reproduced the selected policy controls and all 5,000
policy scores semantically.

Two checks failed:

1. `validation_truth_regeneration_exact`
2. `transfer_policy_regeneration_exact`

The validation mismatch was confined to one case identity and its truth hash.
Version 10 derived evaluator provenance from a raw floating-array digest; a
platform-tail bit changed the identity even though the rounded causal
measurement, selected cases, program reports, and directions were identical.

The policy mismatch was confined to the policy hash and its probe-response hash.
Version 10 hashed the entire diagnostic response even though the search reuses
only the direction-neutral context-factor traces. Unrelated graph-RAG and
diagnostic fields therefore entered the policy identity.

Version 11 changes only those replay bindings:

- evaluator evidence identity hashes rounded semantic measurements, program
  identity, and replica identity instead of raw platform arrays
- policy identity hashes only the per-turn factor traces actually reused by
  policy search

It does not change the world seeds, training schedule, policy search space,
selection objective, learned graph, or the frozen evaluation procedure in
response to the version-10 evaluation labels.

Verification artifact:
`kaggle-results/version-10/verification.json`  
Verification artifact SHA-256:
`35c048fefcd2ffd1285855ba75b8f7c3d6bd439cb7e7136b385a6b2ca6596305`
