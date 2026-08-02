# Kaggle causal-world GPU Version 7 formal-artifact evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>

Source version: 7  
Canonical submitted-source SHA-256: `70705280c0366205166e5c0731beffd3cdf396c0d86bf99624ea4bb8b5704a15`  
Downloaded Kaggle wrapper SHA-256: `3fc697bfa4d3827f09ee69ec7587bf89b386caace8705fda649818141f7d9b44`

Version 7 adds the typed formal-domain substrate to the full GPU causal-world
run and makes its registry artifact JSON-native at the runtime boundary.
Primitive input fields, root mechanics, and invariants are emitted as JSON
arrays instead of retaining Python tuples in memory. This does not change the
canonical registry hash, formal answers, learned causal laws, or transfer
policy. It removes the final in-memory versus serialized representation
difference exposed by GPU version 6.

## Measured execution

- Accelerator: one Tesla P100-PCIE-16GB
- Runtime: JAX `0.7.2`
- Executor: `jit`, one cached JAX/XLA construction
- Accelerator elapsed: `117.6007801620004` seconds
- Kernel log final timestamp: `1398.96274579` seconds
- Evidence rows: `1048576`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Learned laws: `50989`
- Crystallized laws: `6350`
- Maximum invariant error: `2.9098586651343794e-07`
- Experiment gates: `35/35`
- Policy gates: `15/15`
- Transfer gates: `18/18`
- Formal-domain gates: `8/8`
- Python traceback: absent

Model hash: `6bf08a4d7f5f6515373439a220a99bdccc01e960a970b7dc833829885802fa38`  
Report hash: `37d08c307a6eebec87165853671885dc81aa1b1cc4a3452c9f87216747beaeb5`  
Evidence hash: `04d36da4fc8ee32200fbd404a302124701fa70b90c4d011b5081b3ed6a4d5a12`

## Formal-domain evaluation

- Domains: logic, algebra, geometry, calculus, chemistry, biology, and
  information theory
- Executable typed primitives: `15`
- Generated cases: `7680`
- Held-out cases: `1920`
- Independent runtime/oracle agreement: `7680/7680`
- Cross-domain programs: `5/5`
- False-candidate contradiction checks: `7/7`
- Registry hash:
  `dc82181b5b81a94a4819db8d431d389dbf2c7c69bf6f5611cea521ba7135abe4`
- Formal report hash:
  `cc1f139d381df718913a0f73e803c2e3f88e3c605b4c692cd59ccaf6016be617`
- Formal artifact file SHA-256:
  `7d44fc840d4d58b0600fa1e460ff7d0489b17acebbd60cc68c228835ccc3e43a`

Recomputing `run_formal_domain_benchmark(512)` from the downloaded executed
source produced an object exactly equal to the saved formal artifact. The
manifest also survived a JSON serialize/load round trip without type changes.

## Held-out causal transfer

The held-out set used 12 world programs absent from policy selection:

- Turns: `576`
- Cases: `288`
- Coverage: `0.6875`
- Selective accuracy: `0.9292929292929293`
- False-assertion rate: `0.04861111111111111`
- Safe-direction utility: `0.5902777777777778`
- Selective-error upper bound: `0.100293089682`

Policy hash:
`d88c540f463ad0279fe26db0250d54a47bac68539d37e14642f331bd0b4d8d7e`

## Independent replay result

The source-bound verifier passed `72/72` checks with no failures. It
independently reconstructed the million-row run and confirmed:

- canonical submitted-source identity;
- accelerator assignment and cached JIT execution;
- evidence, cursor, model, truth, report, and workflow identities;
- exact formal-domain artifact reconstruction and independent oracle replay;
- all 5,000 transfer-policy evaluations and the selected policy;
- exact policy, validation-truth, transfer-truth, and response regeneration;
- deterministic decimal Wilson and context-factor paths;
- runtime wiki graph and graph-RAG bindings;
- the user-visible side view bound to the real produced artifact;
- absence of a Python traceback.

Machine-readable verification:
`kaggle-results/gpu-version-7/verification.json`  
Verification SHA-256:
`cc0be9c626a2eb8d933cf3e68da536e80383e3d8a4cb82dceb6ea48a6e0215a6`

Downloaded source backup:
`kaggle-results/gpu-version-7-source/atom-massive-causal-world-gpu-v1.py`

Downloaded output backup:
`kaggle-results/gpu-version-7/causal_world_outputs/`

This result is evidence for deterministic replay of the declared simulated
causal-world, formal-domain, and held-out transfer workloads. It does not
establish unrestricted language competence or causal identification from
uncontrolled real-world data.
