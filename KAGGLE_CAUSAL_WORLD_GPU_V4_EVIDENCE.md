# Kaggle causal-world GPU Version 4 non-acceptance record

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 4  
Canonical submitted-source SHA-256: `d8e27ebd118e7aead466b234833c5fc572a2625867d3be258749e1d2d1e8bd7f`  
Downloaded Kaggle wrapper SHA-256: `121612778e8a5ce2e0e58e7c8c44996dcac6f19d70f805241fedf64ab83efb14`

Version 4 replaced binary floating-point Wilson square roots with cached
80-digit decimal arithmetic and a twelve-decimal half-even projection. The
full GPU workflow and exact policy replay passed. The run remains a
non-acceptance result only because it intentionally retained version 3's
condition-log implementation so that the two math boundaries could be tested
independently.

## Measured execution

- Accelerator: one Tesla P100-PCIE-16GB
- Executor: `jit`, one cached JAX/XLA construction
- Accelerator elapsed: `117.67138678499941` seconds
- Kernel log final timestamp: approximately `1202.147826595` seconds
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
Report hash: `4c3fc69732322106247026d1e72e39e379b8243fc7752f8e4b28fb3402f279de`

At `2.87028583030224` times TPU version 11's accelerator speed, this was the
fastest of the first four P100 runs. Cached decimal Wilson evaluation did not
create a measurable accelerator penalty; policy search occurs on the host.

## Independent replay result

The audit passed `68/69` checks. Deterministic decimal Wilson declaration,
portable risk statistics, validation truth, all 5,000 policies, policy hash,
projection-lattice digest, evidence, model, evaluator truth, answer decisions,
RAG graph, and side view all passed.

The sole failed check was `contextual_transfer_replay_exact`. Field diagnosis
reproduced version 3's exact two diagnostic tails at turns `108` and `109`:
Kaggle stored `-244.771076928918`, while Windows regenerated
`-244.771076928917`. No causal decision differed. Version 5 moves condition
logarithms and consensus onto the fixed-precision decimal substrate while
continuing to bind the complete trace.

Machine-readable verification:
`kaggle-results/gpu-version-4/verification.json`  
Verification SHA-256:
`dedc1a5ad88b5d08092af282fd84cb6a07832d0579adba4bb702b5c87527c698`  
Response diagnosis:
`kaggle-results/gpu-version-4/response-replay-diagnosis.json`  
Diagnosis SHA-256:
`0a6c1212a56b44c9f4be73eb68e8712589e519b96f64faa804206c14dd19970a`

