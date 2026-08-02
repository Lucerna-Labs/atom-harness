# Kaggle causal-world Version 3 evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 3  
Generated bundle SHA-256: `D1ED01AF68165FF80290543A0E20CEFAC62A1B3EF73CC22E8CE3EB479DAEEB50`  
Downloaded source SHA-256: `9F8AB0D8F046A6314E3FE7222A7D751A08D7680DEE5628786537BFE9651B1DB1`  
Normalized downloaded/generated content: exact match (`206033` characters)

## Measured execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, 16 atomic model/cursor writes
- Evidence rows/worlds: `131072`
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `336.883606902` seconds
- Measured entity throughput: `12749113.3673637` updates/second
- Measured relation throughput: `152989360.408365` updates/second
- Maximum conservation invariant error: `0.051592741161584854`
- Learned laws: `412` total, `105` crystallized, `32` hypothesis, `275` retired
- Workflow: `9/9`; seven derived answers including one two-link path, two unknowns
- Experiment gates: `21/21`
- Corruption variants rejected: `3/3`
- Deterministic first-microbatch replay: `16/16` shards

Model hash: `09104bc3a755df807ca3b3732db756565c390879a7ebecd7623e404215f612ec`  
Report hash: `d2fc337fa98bc73ad13aa97efef455eebc867ad66e6a68a3f966f9a24b0cc95a`  
Evidence hash: `d1fab50053cb7a927cd5f0f44e9f1903c8eb0a8aae8aca3eb929215f223d7543`  
Resume cursor hash: `580e4b6987dbd21e7b0d5c9e616512ecbe89a79c64a8786ff9465124fb2c00d8`  
Lineage hash: `d7b60c2cd26fa704f267561db618fce067cbbba76316f8708ad166d1d733551c`

The downloaded report hash recomputed exactly after removing its `report_hash`
field. The downloaded evidence JSONL recomputed to the report's evidence hash.
The model passed strict schema/hash loading and exact serialization round-trip.
Re-executing the nine-turn workflow from that model produced the downloaded
response byte-for-byte. The saved resume cursor validated all 16 contiguous
shards, all 16 evidence hashes, all 16 model-lineage entries, the final model
binding, and the cumulative observation count.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `B990AF5F0788C0CCF84C340475ED458882F38D2640DD1FFDF579274172149E59`
- `atom_causal_world_evidence.jsonl`: `52209964B7AE8EC70A6B59EAE883F48F291533FC55D12ADD99AFFD1A9F6A18F4`
- `atom_causal_world_knowledge_graph.json`: `F4F6458CA62213DB2571CC855FD261A372ABC709F1A018E538C16FFE64C79115`
- `atom_causal_world_manifest.json`: `A2A68606A6615981F11DC0AFD139679EC0925C73EA8CE648DFBDB25D6A4457E2`
- `atom_causal_world_model.json`: `7D0E9CB4E4845B733929933378C62C51D7A3E0CB3CF5C3ED51C8C9544119A0FC`
- `atom_causal_world_report.json`: `AA5A7B787F32BF394F7EB4962B42000DCFA413136EAF9E43181D5F135901615C`
- `atom_causal_world_resume_cursor.json`: `E6FBD28ABDB9BD1520D8DB2CA7D13C47977C918F0D13C2D20FE41A366FD3BE00`
- `atom_causal_world_side_view.html`: `42BD5D15EBE0D2E3783FC749510C6143E29FF62EF2CC6F6A443CC7582B246798`
- `atom_causal_world_workflow_request.json`: `40583ED70A691A524A1CB5F215F5728695D44E05A7B1D42B270470628A24198C`
- `atom_causal_world_workflow_response.json`: `BC4211E6C3A5EC0E42B2598BA620F9176FDC03B11D8D78D9362C398E926F80C0`
- `atom-massive-causal-world-v1.log`: `45084C1FC9DB474D94DB9AB08771D0E0BA8A5B1580A2F822E7DDBE0EA1362CB0`

## Runtime diagnostics

There was no Python traceback. Kaggle's TPU image emitted two environment-level
startup diagnostics: transparent hugepages were not enabled, and the optional
TPU metric-server port was unavailable for the local process address. The full
computation, deterministic replay, artifact writes, and notebook conversion all
continued successfully. These diagnostics are retained in the downloaded log
rather than suppressed.
