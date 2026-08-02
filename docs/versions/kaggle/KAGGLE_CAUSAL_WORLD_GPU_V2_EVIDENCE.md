# Kaggle causal-world GPU Version 2 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 2  
Canonical submitted-source SHA-256: `1d454b20040a1f08ea60dec66748a81845bc4708adb584b46aee8789673954c3`  
Downloaded Kaggle wrapper SHA-256: `e656f846a3fb7a873d4746ee9f9a7f42cbc3d5e3c8aa6c5c4fc3a4d4ffdf32e3`

Version 2 replaced raw factor-trace identity with a digest over the complete
5,000-policy projection lattice. It made every policy-search decision part of
the commitment while excluding incidental RAG scores and response prose. The
executed source, GPU workload, artifact bindings, and self-reported gates all
passed. Independent Windows replay still rejected the policy commitment.

## Measured execution

- Accelerator: one Tesla P100-PCIE-16GB
- Executor: `jit`, one cached JAX/XLA construction
- Accelerator elapsed: `118.75471388900024` seconds
- Kernel log final timestamp: approximately `1322.061540985` seconds
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
Report hash: `538b949e558d37a7cf5f2bc9e385bedb3a0a225a3a06c193b9dff32439fdc895`

The accelerator section was `2.84410195654796` times faster than TPU version
11's measured `337.7505141209986` seconds for the same workload shape.

## Held-out behavior

The search evaluated `5000` policies, found `560` eligible policies, and froze
direction-prior power `0.0`, token-likelihood power `0.75`, pair-motif power
`0.05`, decreasing consensus `0.75`, and increasing consensus `0.995`. The
separate 576-turn evaluator measured:

- assertions: `396`
- correct assertions: `368`
- false assertions: `28`
- coverage: `0.6875`
- selective accuracy: `0.9292929292929293`
- safe-direction utility: `0.5902777777777778`
- paraphrase consistency: `1.0`

Policy hash: `cf308976a7d7526035f8f4364694d251aa58de1772489c0cd7c31449453936d8`  
Projection-lattice digest:
`f4d9fa5e49f933d0293e0c49476b8078096c855ccb9eb9f228ebb2cf4fcb5a5d`

## Independent replay result

The audit passed `64/66` checks. It regenerated the million-row evidence
stream, model, both truth sets, selected controls, validation measurements,
eligibility count, response hashes, wiki/RAG graph, and rendered side view.
It rejected `transfer_policy_regeneration_exact` and the dependent
`contextual_transfer_replay_exact`.

Field diagnosis found only the policy hash and projection-lattice digest
differed. The lattice evaluations included an unrounded Wilson-score
square-root result. Identical integer assertion counts can therefore acquire
different last bits from operating-system math libraries even though every
selected control, persisted measurement, gate, and evaluator decision agrees.
GPU version 3 normalizes the Wilson value to twelve decimal places before both
risk thresholding and artifact hashing.

A counterfactual replay of the downloaded version-2 source applied that
twelve-decimal normalization without changing any other calculation. Every
new non-hash difference was a Wilson value, and each equaled the downloaded
Kaggle value rounded to twelve places; for example,
`0.07644278058788155` became `0.076442780588`. This directly isolates the
normalization boundary while leaving GPU version 3 as the independent
cross-operating-system confirmation.

Machine-readable verification:
`kaggle-results/gpu-version-2/verification.json`  
Verification SHA-256:
`d8ce4c581243aa58a2364062e9ea897ca258f98b4563b2addcdbd2bd4c381820`  
Field diagnosis:
`kaggle-results/gpu-version-2/policy-replay-diagnosis.json`  
Diagnosis SHA-256:
`cdc06197bf9ed8aeb1b09bdad07b775c063b38c84e21ca89947d2dcf7c61ddf2`
Counterfactual Wilson-normalization diagnosis:
`kaggle-results/gpu-version-2/policy-replay-wilson12-diagnosis.json`  
Counterfactual diagnosis SHA-256:
`fa753595cf7a29653e2bbb309f899c1e507ec67aafee0d96ed8c5cb91cf9606d`
