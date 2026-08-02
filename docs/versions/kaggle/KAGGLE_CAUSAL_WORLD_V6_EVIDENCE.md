# Kaggle causal-world Version 6 measured evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 6  
Canonical submitted-source SHA-256: `36874877c7d9a3c7281512279ab78fa4dd9522f7ba5d50b96297a1b79a43e64b`  
Downloaded CRLF file SHA-256: `e445ef5b184ef82dd6a8b338c531a1ae7cfd88d2bdc06c5ebfc1de3d4dcf8458`

Version 6 is the first large Kaggle run that combines exact nine-axis
curriculum coverage with the carrier-retention and normalized-distribution
conservation repair. The downloaded run passed every check in the independent
fail-closed verifier.

## Measured execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, 16 atomic model/cursor writes
- Evidence rows: `524288`, with `65536` from each of eight domains
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `337.248668387` seconds
- Notebook conversion finished at: `677.668120114` seconds
- Maximum conservation invariant error: `3.824358714155096e-07`
- Learned laws: `26494` total, `6278` crystallized
- Workflow: `9/9`; seven derived answers and two evidence-bound unknowns
- Experiment gates: `25/25`
- Deterministic first-microbatch replay: `16/16` shards
- Python traceback: absent

Model hash: `d7a291e90d3cbad52612f853809be499c1fca0d275abbfc2f272dc22344e024c`  
Report hash: `d33205ce2153d514f7825dcfd0b0aa9fcf968a11b7047f85fe497b773d028fc5`  
Evidence hash: `5c1c5cfe77b24ce93cae71928de10a044870d3fb1f31b74779b2023cd0854c7f`  
Resume cursor hash: `d89d02c81517924b862d054ef599340d88ce05771e81b8044c782cb705849ea6`  
Curriculum schedule hash: `fdaa4dd2ac70d17c4a270ed30f390635cf588f85a7e3059c6b93519e6d68c928`

## Curriculum and causal coverage

The schedule contains 64 unique programs and exercises all seven roots in both
primary and secondary roles. Exact coverage was observed for every value of:

- scale: `macroscopic`, `mesoscopic`, `microscopic`, `networked`, `planetary`
- resources: `abundant`, `balanced`, `competitive`, `pulsed`, `scarce`
- signal: `clear`, `delayed`, `noisy`, `saturated`, `sparse`
- relations: `asymmetric`, `competitive`, `cooperative`, `fluid`, `modular`
- time: `aging`, `oscillatory`, `regenerative`, `stable`, `volatile`
- topology: `clustered`, `hierarchical`, `lattice`, `small_world`, `sparse`
- phase: `drifting`, `locked`, `resonant`, `turbulent`
- energy: `balanced`, `cascading`, `high`, `low`, `pulsed`
- boundary: `closed`, `open`, `porous`, `reflective`

Every evidence row was bound to its world conditions and causal provenance.
The persisted graph answered all seven supported workflow turns by a retained
causal path and abstained on both unsupported turns.

## Version 5 to Version 6 comparison

The workload and coverage stayed fixed. The conservation repair changed the
maximum invariant error from `1.0` in version 5 to
`3.824358714155096e-07` in version 6. The aggregate experiment result moved
from 24/25 to 25/25 without sacrificing axis coverage, evidence count, device
parallelism, deterministic replay, workflow accuracy, or artifact bindings.

## Independent verification

`scripts/verify_kaggle_causal_world_run.py` streamed all 524,288 evidence rows
and independently checked:

- every persisted file and its hash binding;
- model and resume-cursor round trips;
- workflow and answer replay;
- runtime wiki graph and RAG context;
- the user-visible artifact side view rendered from disk;
- TPU device topology, `pmap`, 16 shards, and exact update counts;
- all roots and every curriculum axis value;
- conservation, deterministic replay, experiment gates, and logs;
- canonical submitted-source identity.

It exited `0` with no failed checks. The machine-readable audit is
`kaggle-results/version-6/verification.json`.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `b990af5f0788c0ccf84c340475ed458882f38d2640dd1ffdf579274172149e59`
- `atom_causal_world_evidence.jsonl`: `de0ee8f6afb08391635764f80b046e039992aadf7d8c2b4584b3c808e403a355`
- `atom_causal_world_knowledge_graph.json`: `0ef0e5577d9cc605b54ea53e14e5d3b530f4a13b17135c0ba0fec089df5e78b8`
- `atom_causal_world_manifest.json`: `a2a68606a6615981f11dc0afd139679ec0925c73ea8ce648dfbdb25d6a4457e2`
- `atom_causal_world_model.json`: `92a3f624d08bf4933a0ac8ebe74ec92421654306f26bb7673610aee7ed83a299`
- `atom_causal_world_report.json`: `cef49d5607a75c54e8ee4ee4c5a55cdbef8e162fc55026eac0bb00220f8ad862`
- `atom_causal_world_resume_cursor.json`: `19050cb4927df5bf2e39d4a99274bef6609d4129687277a020479820bf961960`
- `atom_causal_world_side_view.html`: `9a7bc363904f5d069cb16d4aea608c57c58639a47743d3358e81bc0d2748a3e4`
- `atom_causal_world_workflow_request.json`: `bf1dda44ec1ef7b2e89ac8df141cb4d4fa91e267fa87f0db488592b79a858a29`
- `atom_causal_world_workflow_response.json`: `2616bccde4ab72486822aec7c72076ff099f5640f04279b1fd1c6b02b63f56f0`
- `atom-massive-causal-world-v1.log`: `e9835580618797a8925fbcb09fa7e656a9a47b4e37abd2c8b7872a735f57d367`
