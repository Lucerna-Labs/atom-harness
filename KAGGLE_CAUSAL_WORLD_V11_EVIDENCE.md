# Kaggle causal-world Version 11 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 11  
Canonical submitted-source SHA-256: `d6ae225477705d1ef0db4745e1bfeb298032add2ee5202bb6c9e9ca8e3e5cc58`  
Downloaded Kaggle wrapper SHA-256: `faa66576aef99f3fe2429901547b59a8cf10ffe834c31d5abbe7f6da970dda74`

Version 11 replaced raw floating-point evaluator provenance with rounded
semantic causal measurements. That correction worked: the independent Windows
verifier regenerated both validation and evaluation truth exactly. The run is
still retained as a non-acceptance result because policy identity included a
complete raw factor-trace digest whose numerically insignificant float tails
differed between Kaggle Linux and local Windows.

## Measured execution

- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`
- Accelerator elapsed: `337.7505141209986` seconds
- Shards: `16/16`
- Evidence rows: `1048576`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Learned laws: `50992`
- Crystallized laws: `6352`
- Experiment gates: `29/29`
- Policy gates: `15/15`
- Transfer gates: `18/18`
- Python traceback: absent

Model hash: `f00f7cb22c1da080b3553de1d105e6490df2be081916cfd81bcf45e95926466d`  
Report hash: `4aaa374741f6af09b19acf8203ca5b352bb1c48c71cc5fd962dd13b390d7ce44`

## Held-out behavior

The policy search evaluated `5000` policies, found `557` eligible policies, and
froze direction-prior power `0.0`, pair-motif power `0.05`, decreasing
consensus `0.75`, and increasing consensus `0.995`. On 576 turns generated from
12 evaluator programs excluded from both training and validation, it measured:

- assertions: `396`
- correct assertions: `368`
- false assertions: `28`
- coverage: `0.6875`
- selective accuracy: `0.9292929292929293`
- false-assertion rate: `0.04861111111111111`
- safe-direction utility: `0.5902777777777778`
- paraphrase consistency: `1.0`

These are measurements inside the procedural causal-world distribution, not an
open-domain language or factuality benchmark.

## Independent replay result

The verifier passed `65/66` checks. It exactly reproduced the million-row
evidence stream, model and resume bindings, evaluator truth, validation truth,
exact and contextual answer artifacts, graph-RAG knowledge graph, and rendered
side view. The sole failed check was
`transfer_policy_regeneration_exact`.

The field-level diagnosis found only two differences:

- the policy hash, because it covers the policy body;
- `probe_response_hashes.policy_neutral_context_factor_trace`.

Every selected control, all `5000` policy evaluations, eligibility counts,
validation measurements, gates, and final answers matched. The corrected policy
runtime therefore hashes the complete observable projection lattice—each policy
paired with its evaluation hash—instead of treating non-decision float tails as
policy identity.

Machine-readable verification:
`kaggle-results/version-11/verification.json`  
Verification SHA-256:
`447b52aed6f0462e3c56379ae425b54ab334741e602847d3a4ad99bb8b00c625`  
Field diagnosis:
`kaggle-results/version-11/policy-replay-diagnosis.json`  
Diagnosis SHA-256:
`41e4b03448c8cc52942d39d0e76617de17c4c9916a59d64bec5c2194c9d7e2d2`

