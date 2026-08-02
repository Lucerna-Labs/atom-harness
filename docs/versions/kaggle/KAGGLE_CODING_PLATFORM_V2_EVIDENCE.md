# Kaggle causal coding platform version 2 evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-causal-coding-platform-v1>

Kaggle reported version 2 as `COMPLETE`. The private, internet-disabled kernel
was submitted with GPU enabled. This experiment's executable interventions are
isolated Python processes, so the behavioral result does not depend on GPU
arithmetic.

## Measured result

- Training specifications: `27`
- Isolated primitive-removal interventions retained: `243`
- Unseen validation and held-out platform specifications: `7`
- Atom capability results: `31/31`
- Atom whole-specification results: `7/7`
- Fixed baseline capability results: `15/31`
- Fixed baseline whole-specification results: `2/7`
- Capability-score improvement: `0.5161290322580645`
- Experiment gates: `9/9`
- Live workflow status: `derived`
- Live artifact hidden evaluation: passed
- Python traceback: absent

The model learned conjunctive causes rather than forcing a single primitive
onto every behavior. Parallel promotion retained both directed relation and
composition; emergent topology retained both directed relation and topology.

## Ablations

- Without causal memory, full-platform behavior fell to `3/9`.
- Without phase composition, full-platform behavior fell to `1/9`.
- Without topological persistence, synthesis abstained.
- A capability never experienced by the graph also produced an abstention.

## Identity and replay

- Bundle SHA-256:
  `da4b77a279f4b9d10fb2a76eab493f4ab58ae632191661d47e2dda73b8d8c022`
- Downloaded Kaggle wrapper SHA-256:
  `024c6dfd989eb6fd78217e7b6a8b4df7b254bcd8e911b52f3a7bf992d506bd3e`
- Kernel log SHA-256:
  `6f5e0690bb76e1768ac0069f9f9679d83c61986675306a88560f5d2da6252147`
- Model hash:
  `577bafd8d9c8e86e1674d12ce18420c73b6800b05d897be039b14f40727767e2`
- Report hash:
  `63ea8ba3a97fdeb328d8103d8a441db4477e5409342fc3be3e254c31293e13c8`

All six downloaded runtime artifacts were byte-identical to their current
local counterparts:

- persistent causal model;
- experiment report;
- user-visible side view;
- workflow request;
- workflow response;
- generated executable platform.

Downloaded outputs:
`kaggle/runs/coding-platform-v1-run2/coding_harness_outputs/`

Downloaded Kaggle source:
`kaggle/runs/coding-platform-v1-run2-source/`

This run measures compositional platform synthesis inside the declared nine
behavioral capabilities. It does not measure unrestricted software generation
or natural-language programming.
