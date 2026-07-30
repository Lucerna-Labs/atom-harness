# Kaggle causal-world GPU Version 3 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 3  
Canonical submitted-source SHA-256: `e004d615631c136788b532e5516364f848eb8a47f89dedfa8acecfce4df3afcb`  
Downloaded Kaggle wrapper SHA-256: `44033c659534260b8cb3328ab0c8446d22be4ebb288e2a3e1c6e4442a7fb68b2`

Version 3 normalized Wilson risk values to twelve decimal places before policy
eligibility decisions and projection-lattice hashing. This made the complete
5,000-policy artifact regenerate exactly across Kaggle Linux and local Windows.
The run remains a non-acceptance result because one response-replay check
exposed two last-decimal condition-log diagnostics.

## Measured execution

- Accelerator: one Tesla P100-PCIE-16GB
- Executor: `jit`, one cached JAX/XLA construction
- Accelerator elapsed: `118.90873969199924` seconds
- Kernel log final timestamp: approximately `1342.208275968` seconds
- Shards: `16/16`
- Evidence rows: `1048576`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Learned laws: `50989`
- Crystallized laws: `6350`
- Experiment gates: `30/30`
- Policy gates: `15/15`
- Transfer gates: `18/18`
- Python traceback: absent

Model hash: `8e2db567e9fcf440cc0aa4c0843f35eee4066af17e9db62f462942f0ce4d7c5e`  
Report hash: `0f17e74a021c2a128ba82d97d80bcfd58c965413af80611a3e7651da01bff282`

The accelerator section was `2.84041791205465` times faster than TPU version
11's measured accelerator section.

## Held-out behavior

The 5,000-policy search again found `560` eligible policies and selected prior
power `0.0`, token-likelihood power `0.75`, pair-motif power `0.05`,
decreasing consensus `0.75`, and increasing consensus `0.995`. The separate
576-turn evaluator measured:

- assertions: `396`
- correct assertions: `368`
- false assertions: `28`
- coverage: `0.6875`
- selective accuracy: `0.9292929292929293`
- safe-direction utility: `0.5902777777777778`
- paraphrase consistency: `1.0`

Policy hash: `399cb98d9641f8490f433afa815b292587d9ae66231d8fae8de753e0146fbe83`  
Projection-lattice digest:
`a54e778d5f4798db8f54fd51619bea109f346dabf45b87493ef5d2910b60027d`

## Independent replay result

The audit passed `67/68` checks. Crucially,
`transfer_policy_regeneration_exact` passed: validation truth, all 5,000
policy projections, eligibility, selection, risk measurements, policy hash,
and projection-lattice digest were identical across operating systems. Model,
evidence, evaluator truth, answer decisions, response evaluation, wiki/RAG,
and side-view bindings also passed.

The sole failed check was `contextual_transfer_replay_exact`. Field diagnosis
found exactly two diagnostic values:

- turns `108` and `109`;
- positive-direction `pair_motif_log_likelihood`;
- Kaggle: `-244.771076928918`;
- Windows: `-244.771076928917`.

The response hash differed because it intentionally binds the complete
diagnostic trace. No direction, assertion, source law, score boundary, answer,
or evaluation metric differed. The next runtime computes condition
log-likelihoods and consensus with fixed-precision decimal functions rather
than weakening response identity.

Machine-readable verification:
`kaggle-results/gpu-version-3/verification.json`  
Verification SHA-256:
`9f36fe97259cbe266f5e87f7dd4e9b9d12db70a3bb79accb936978725192932f`  
Response diagnosis:
`kaggle-results/gpu-version-3/response-replay-diagnosis.json`  
Diagnosis SHA-256:
`c5e1b4080608c7498577eb474ea1cbe3dca497e790a7991210ccf2276b2a16f4`

