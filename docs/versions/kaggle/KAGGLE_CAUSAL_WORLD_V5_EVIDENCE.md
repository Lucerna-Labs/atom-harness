# Kaggle causal-world Version 5 diagnostic evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 5  
Canonical submitted-source SHA-256: `64d82afa57c1061e54e422c7c405f078edc0a2eba8f89425a6b2c3db3bdb5dda`  
Downloaded CRLF file SHA-256: `48e14d0cc0c8d025110536bd4d285a740a98fe0e20495184bf43222b8aaeada3`

Version 5 is retained as a negative conservation result. It proves that the
expanded 64-program schedule reaches every value of all nine curriculum axes,
but it predates the carrier-retention and normalized-distribution conservation
repair. The fail-closed verifier therefore rejects it on conservation and the
aggregate experiment gate.

## Measured execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, 16 atomic model/cursor writes
- Evidence rows: `524288`, with `65536` from each of eight domains
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `336.925768286` seconds
- Notebook conversion finished at: `674.456408925` seconds
- Maximum conservation invariant error: `1.0`
- Learned laws: `26336` total, `6348` crystallized
- Workflow: `9/9`
- Deterministic first-microbatch replay: passed
- Python traceback: absent

Model hash: `b442cc9a2df03fe5c02ff9adfcd8aa474eb22c9a8c1a58dd826bac28efbf3d3c`  
Report hash: `4f1de6621018a6704299faaacb0edcffc82eaa10dead1b7f0b0a1f6097be6a01`  
Evidence hash: `182e3aeea2c12250b9bbbcbe652c7674ee79858f1399c9e2f1eeb107a206ec00`  
Resume cursor hash: `3ea478ce0a1c3698d46aec57a7cbb288ff3b1eaf10ccffff550c5f37b710eafb`

## Curriculum breadth reached

The schedule contains 64 unique programs and exercises all seven roots in both
primary and secondary roles. Exact coverage was observed for:

- scale: `macroscopic`, `mesoscopic`, `microscopic`, `networked`, `planetary`
- resources: `abundant`, `balanced`, `competitive`, `pulsed`, `scarce`
- signal: `clear`, `delayed`, `noisy`, `saturated`, `sparse`
- relations: `asymmetric`, `competitive`, `cooperative`, `fluid`, `modular`
- time: `aging`, `oscillatory`, `regenerative`, `stable`, `volatile`
- topology: `clustered`, `hierarchical`, `lattice`, `small_world`, `sparse`
- phase: `drifting`, `locked`, `resonant`, `turbulent`
- energy: `balanced`, `cascading`, `high`, `low`, `pulsed`
- boundary: `closed`, `open`, `porous`, `reflective`

Every evidence row was bound to its conditions and provenance, and each domain
contributed exactly one eighth of the evidence corpus.

## Fail-closed verification

`scripts/verify_kaggle_causal_world_run.py` streamed all 524,288 evidence rows
and independently checked artifact hashes, model and cursor round trips,
workflow replay, wiki graph, rendered side view, TPU topology, update counts,
root and axis coverage, source identity, and logs. It exited nonzero with
exactly these failed checks:

- `conservation_bounded`
- `all_experiment_gates_passed`

Kaggle's Windows pull converted source newlines from LF to CRLF. The verifier
records the raw downloaded-file hash but compares the canonical LF source hash;
that canonical hash exactly matches the submitted version-5 bundle.

The machine-readable audit is
`kaggle-results/version-5/verification.json`.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `b990af5f0788c0ccf84c340475ed458882f38d2640dd1ffdf579274172149e59`
- `atom_causal_world_evidence.jsonl`: `a5f800086388fec8d5a22ae24fc07db63db6a406550f81f93bf8c762cdf9023f`
- `atom_causal_world_knowledge_graph.json`: `89ab8c0751b295f39a8316a04f29ef5e4fe3430874179bd8c717138b13a160f8`
- `atom_causal_world_manifest.json`: `a2a68606a6615981f11dc0afd139679ec0925c73ea8ce648dfbdb25d6a4457e2`
- `atom_causal_world_model.json`: `630e53039e02b67066110731846bee65cbb334e29941acde8eb77ecbbc7483d5`
- `atom_causal_world_report.json`: `91e57cfefa1821c7317220275bdaccc3072f0519d3301fae37efe93e022e5a55`
- `atom_causal_world_resume_cursor.json`: `1ef68426cc9708c841526571f8479390a26392c91845aef05a250d1df2a576c6`
- `atom_causal_world_side_view.html`: `0cdd33c87a03cbc5cf99b6949379adf1fdef7324150f5f0b3b84d156feab5bb1`
- `atom_causal_world_workflow_request.json`: `ef1ab9515acfebf6e4f9b7acfadd6784a189a2af83421c2af494629a7713f007`
- `atom_causal_world_workflow_response.json`: `2d05f242259631ccb85d3bbb0843181f0999ebc0c0f1690455ebdc83a0b64c55`
- `atom-massive-causal-world-v1.log`: `dbe56f25c6a22938e7fed75b814f5d0c43ab62f64b6e7e8972ef70f1dbe39e56`
