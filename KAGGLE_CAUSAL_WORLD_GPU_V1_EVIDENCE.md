# Kaggle causal-world GPU Version 1 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 1  
Canonical submitted-source SHA-256: `41494b9ba11fd0537392a255ce208b35dcaf45d7875c0375c9f2e5b438edabe8`  
Downloaded Kaggle wrapper SHA-256: `36bca6441fd26c22e1fc975417729af11ea4e6def6919e9ffc9ac773e5382d04`

This was the first fail-closed GPU execution of the full causal-world workload.
It used the same world profile, seeds, validation search, evaluator workload,
RAG graph, and side-view contract as TPU version 11. Kaggle assigned one Tesla
P100-PCIE-16GB and the runtime used one cached JAX/XLA `jit` executor.

## Measured execution

- Accelerator elapsed: `117.83772524400365` seconds
- Shards: `16/16`
- Evidence rows: `1048576`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Maximum conservation error: `2.9098586651343794e-07`
- Learned laws: `50989`
- Crystallized laws: `6350`
- Experiment gates: all passed
- Policy gates: all passed
- Transfer gates: all passed
- Python traceback: absent

Model hash: `8e2db567e9fcf440cc0aa4c0843f35eee4066af17e9db62f462942f0ce4d7c5e`  
Report hash: `1422b8acf7ac123270971f34c55e4b2668cf9103ab733caa193fc5f4215ac6ed`

The P100 accelerator section was `2.86623416585501` times faster than TPU
version 11's measured `337.7505141209986` seconds for this implementation. This
comparison covers only the accelerator section; the notebook's host-side graph
consolidation, policy search, evaluation, serialization, and HTML export still
dominated total wall time.

## Held-out behavior

The frozen policy produced the same aggregate evaluator behavior as TPU version
11:

- assertions: `396/576`
- correct assertions: `368`
- false assertions: `28`
- coverage: `0.6875`
- selective accuracy: `0.9292929292929293`
- false-assertion rate: `0.04861111111111111`
- safe-direction utility: `0.5902777777777778`
- paraphrase consistency: `1.0`

## Independent replay result

The Windows audit passed `64/66` checks. The GPU assignment, `jit` executor,
source identity, evidence stream, model, evaluator truth, response hashes,
wiki/RAG graph, and side view all passed. It rejected
`transfer_policy_regeneration_exact` and the dependent
`contextual_transfer_replay_exact` because this source still used the raw
factor-trace policy digest diagnosed in TPU version 11.

GPU version 2 carries the projection-lattice correction and version 3 adds the
portable Wilson-risk boundary diagnosed from that run. Version 1 is retained as
timing evidence and as confirmation that accelerator selection fails closed; it
is not the replay acceptance result.

Machine-readable verification:
`kaggle-results/gpu-version-1/verification.json`  
Verification SHA-256:
`7527e5520703ab9419c16a6fe18cf7c58562761cabef29f0d1166ac88c146943`
