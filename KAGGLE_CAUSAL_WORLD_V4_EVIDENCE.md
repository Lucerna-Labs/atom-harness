# Kaggle causal-world Version 4 diagnostic evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 4  
Downloaded source SHA-256: `0e0e69b63b72b873034fcae061626ac5dec3620a223050e35bdc51704a030aa7`  
Curriculum schedule SHA-256: `be06f1457f39a6190c2ec5988a83b94aa1af621b0dcd0cdf7e92c11076fc4aca`

Version 4 is a retained negative result, not an accepted architecture run. The
downloaded artifacts are internally hash-bound and replayable, but the
fail-closed verifier rejects the run for incomplete axis coverage and a failed
conservation invariant.

## Measured execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, 16 atomic model/cursor writes
- Evidence rows: `524288`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `336.942250394` seconds
- Notebook elapsed through conversion: `671.39154126` seconds
- Maximum conservation invariant error: `1.0`
- Learned laws: `26337` total, `6360` crystallized
- Workflow: `9/9`; seven derived answers and two unknowns
- Experiment gates: `24/25`; `conservation_bounded` failed
- Deterministic first-microbatch replay: `16/16` shards

Model hash: `617448269c7f98deb6057020361444dc981d45562657f944553bf05cb846336a`  
Report hash: `4d82f3b6eeaead628cd492c354832603fb74dd2272bee4c1a835cce5c5edfe30`  
Evidence hash: `baf63ab37a6e70a6c5ed3db2831b4c257cdd461473ca4408d7b004fa3f29fc00`  
Resume cursor hash: `b58a829fb3ddda575ae722d11c714b9af9a204271287017f43ec37fb3341ff7a`

## Curriculum failure

The schedule contained 64 unique programs and exercised all seven roots in both
primary and secondary roles. It covered every resources, signal, relations,
time, topology, phase, and energy value. It covered only:

- scales: `microscopic`, `mesoscopic` (2 of 5);
- boundaries: `open`, `closed`, `porous` (3 of 4).

It omitted `macroscopic`, `planetary`, and `networked` scales and the
`reflective` boundary. Counting 64 unique program IDs was therefore not a
sufficient curriculum-coverage test.

## Conservation failure and repair direction

Seven shards reached a relative mass, energy, or resource error of `1.0`; two
more exceeded the `<0.25` gate. The accelerator path allowed all entities in a
world to expire, zeroed their conserved values, and then had no carrier from
which to restore the budget. It also used an unbounded root gain as a blending
coefficient between a normalized and an unnormalized distribution.

The repaired runtime retains at least one carrier and normalizes both the prior
and proposed local distributions to the original world budget. Conservation
strength is clipped to a valid mixing interval and controls local retention
versus redistribution; it no longer controls whether the global budget exists.

## Fail-closed verification

`scripts/verify_kaggle_causal_world_run.py` streamed all 524,288 evidence rows
and independently checked report/evidence hashes, model and cursor round trips,
workflow replay, wiki graph, rendered side view, device topology, paired update
counts, program/root coverage, logs, and artifact file hashes. It exited nonzero
with exactly these failed checks:

- `axis_coverage_exact`
- `conservation_bounded`
- `all_experiment_gates_passed`

The machine-readable audit is
`kaggle-results/version-4/verification.json`.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `b990af5f0788c0ccf84c340475ed458882f38d2640dd1ffdf579274172149e59`
- `atom_causal_world_evidence.jsonl`: `5b55b2ab2703d0f073bc3048c79c585a58b52b485b2cc5bac75c2af39c3a9b38`
- `atom_causal_world_knowledge_graph.json`: `6338078722ba818d1e2a374b29541ef9bdd3168b24c79b0b2b3f1d116b2a9ef3`
- `atom_causal_world_manifest.json`: `a2a68606a6615981f11dc0afd139679ec0925c73ea8ce648dfbdb25d6a4457e2`
- `atom_causal_world_model.json`: `8c36d28189dc125dc2d21855f1fb342ec592019d8d5dbbd99b911bed6626e24b`
- `atom_causal_world_report.json`: `04228e40fd2e45ac413c3478626e580511ebb5277e1708f7a95bbe1c4abe378f`
- `atom_causal_world_resume_cursor.json`: `7eac3ceac41bc0bbde9c8e2d0ce041a0a85acb2f7647b7212a9c866da6858fdf`
- `atom_causal_world_side_view.html`: `d4a0032234a91141529028601573c66cd26d66e3d3c7363cd94a40daeb11842a`
- `atom_causal_world_workflow_request.json`: `4825428f41089422666067c51c7e91da9540aaed3d0679e7d95049a735625573`
- `atom_causal_world_workflow_response.json`: `07fa7efeea1c04cbb39c7041465c754c4924f14c8a26bb557d15ac1992f3a3e8`
- `atom-massive-causal-world-v1.log`: `abee9aa63bb9b7e392f24c0825d8d9f399210aa059ab017904e544135b4c7402`

There was no Python traceback. Kaggle's notebook conversion finished and all
artifacts were downloaded before this diagnostic was written.
