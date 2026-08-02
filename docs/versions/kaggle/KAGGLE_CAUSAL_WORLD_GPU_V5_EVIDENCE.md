# Kaggle causal-world GPU Version 5 strict replay evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 5  
Canonical submitted-source SHA-256: `d1d03c02b75da4f2bb3ece9f825ac036c24ddfa8a1681128bd248e155d24ae93`  
Downloaded Kaggle wrapper SHA-256: `7e23540b1c5fcd25e593f1bd5db575bfcb17c4a56457a105e9f6118ac79f48a2`

Version 5 moved the last platform-sensitive causal-factor operations onto a
fixed-precision decimal substrate. Direction priors, singleton likelihoods,
pair-motif likelihoods, consensus transforms, and Wilson risk bounds are
projected with deterministic half-even rounding. The full Kaggle GPU workflow
passed, and an independent Windows reconstruction reproduced every declared
artifact and behavior exactly.

## Measured execution

- Accelerator: one Tesla P100-PCIE-16GB
- Executor: `jit`, one cached JAX/XLA construction
- Accelerator elapsed: `117.67405132099918` seconds
- Kernel log final timestamp: approximately `1371.613868113` seconds
- Evidence rows: `1048576`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Learned laws: `50989`
- Crystallized laws: `6350`
- Experiment gates: `30/30`
- Policy gates: `15/15`
- Transfer gates: `18/18`
- Python traceback: absent

The accelerator section was `2.8702208373846165` times as fast as TPU version
11's measured `337.7505141209986` seconds.

Model hash: `6bf08a4d7f5f6515373439a220a99bdccc01e960a970b7dc833829885802fa38`  
Report hash: `42a91ed7bacf0e09fcc3794fd77d317208d407d91d1748a376632c1dee8304a1`  
Evidence hash: `04d36da4fc8ee32200fbd404a302124701fa70b90c4d011b5081b3ed6a4d5a12`

## Held-out causal transfer

The held-out set used 12 world programs absent from policy selection:

- Turns: `576`
- Assertions: `396`
- Correct assertions: `368`
- False assertions: `28`
- Coverage: `0.6875`
- Selective accuracy: `0.9292929292929293`
- Safe-direction utility: `0.5902777777777778`
- Selective-error upper bound: `0.100293089682`

Policy runtime:
`atom-causal-metaplastic-transfer-policy-v7`  
Risk method:
`wilson_score_upper_bound_decimal12`  
Policy hash:
`d88c540f463ad0279fe26db0250d54a47bac68539d37e14642f331bd0b4d8d7e`

## Independent replay result

The verifier passed `69/69` checks. It independently rebuilt the model from
the downloaded million-row evidence stream and confirmed:

- canonical submitted-source identity;
- accelerator assignment and cached JIT execution;
- evidence, cursor, model, evaluator-truth, and report identities;
- all 5,000 policy evaluations and the selected policy;
- deterministic decimal Wilson and context-factor runtimes;
- exact policy, exact-response, and contextual-response regeneration;
- exact decisions, answers, and causal factor traces;
- the runtime wiki graph and graph-RAG bindings;
- the user-visible side view and its real model, policy, and lattice digests.

Machine-readable verification:
`kaggle-results/gpu-version-5/verification.json`  
Verification SHA-256:
`c8d13fbfb3a1913089d39765d65b836e078b872f54673255aa318b48b223c9cc`

Downloaded source backup:
`kaggle-results/gpu-version-5-source/atom-massive-causal-world-gpu-v1.py`

This result establishes deterministic cross-platform replay for this simulated
causal-world workload and its declared transfer evaluation. It does not by
itself establish broad natural-language competence outside that world.
