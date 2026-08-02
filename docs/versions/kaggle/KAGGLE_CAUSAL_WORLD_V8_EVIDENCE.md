# Kaggle causal-world Version 8 measured evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 8  
Canonical submitted-source SHA-256: `854f77cce11b8714648fb2d6f75d22b8c073841db08ab1694b885ce18721a529`  
Downloaded source-file SHA-256: `65589c2c910f7aa751903fa3ed6c1016d832cc14957a54163022a0bb94170186`

Version 8 is the independently replayable measurement of the massive
axis-conditioned causal-transfer runtime. Version 7 produced the same causal
decisions and scores, but a platform-dependent final bit in one persisted
floating-point diagnostic changed the serialized response hash. Version 8
normalizes persisted transfer diagnostics to 12 decimal places. The downloaded
Kaggle artifacts then passed every strict verifier check on Windows, including
byte-for-byte contextual-response replay and independent regeneration of the
sealed held-out truth.

## Measured world execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, with 64 unique curriculum programs
- Evidence rows: `1048576`, with `131072` from each of eight domains
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `337.058419628` seconds
- Notebook HTML conversion finished at approximately `925.342` seconds
- Maximum conservation invariant error: `3.824358714155096e-07`
- Learned laws: `50992` total, `6352` crystallized
- Persisted workflow: `9/9`; seven derived answers and two evidence-bound unknowns
- Experiment gates: `27/27`
- Deterministic first-microbatch replay: `16/16` shards
- Python traceback: absent
- Wiki graph: `105` nodes, bound to the learned causal graph
- Side view: user-visible, right-side placement, bound through
  `render_causal_world_artifact`

Model hash: `32089f120e8ef7d56f2735cf8074d7a4338ae5041e7ad2e53fb68910464e552f`  
Report hash: `56df52e3e5d21a705f406e7889b2b59bebff89eccf65fa1f27b0534a47c267f4`  
Evidence hash: `ca52983735c1163f397117f84267171780ddea139ec392f5045b488e23ef4dc9`  
Resume cursor hash: `a5c953111e62a866bba2204af9b1cb370073e5931a503f1ce1b7d8c7f0549baa`  
Curriculum schedule hash: `fdaa4dd2ac70d17c4a270ed30f390635cf588f85a7e3059c6b93519e6d68c928`

## Sealed held-out causal transfer

The evaluator used seed `2026072301`, selected 12 program IDs absent from the
64-program training schedule, and regenerated every label from two independent
treated/control simulations. It excluded self-effects and balanced the 288
cross-feature cases between 142 decreasing and 146 increasing relations. Each
case was asked through two independently rendered English paraphrases, producing
576 inference turns. The truth artifact was never provided to the inference
runtime.

Exact condition matching asserted no answer on any unseen regime. The
axis-conditioned composition produced:

- assertions: `328/576`
- abstentions: `248/576`
- coverage: `0.5694444444444444`
- correct assertions: `310`
- false assertions: `18`
- selective accuracy: `0.9451219512195121`
- false-assertion rate: `0.03125`
- safe-direction utility: `0.5069444444444444`
- majority-direction safe utility: `0.013888888888888888`
- decreasing-direction coverage / selective accuracy:
  `0.31690140845070425` / `0.8888888888888888`
- increasing-direction coverage / selective accuracy:
  `0.815068493150685` / `0.9663865546218487`
- paraphrase consistency: `1.0`
- adversarial parser variants rejected: `3/3`
- transfer gates: `14/14`

Every derived transfer retained its source-law identities and provenance.
Questions below the posterior and source-regime thresholds returned `unknown`.
Graph-RAG was present on every transfer turn and used parsed domain, cause, and
effect fields to restrict retrieval before contextual composition.

Transfer truth hash: `c188a57a285e789c6011e849acf339355b2a2a858b4841dc59fa835d5b281b54`  
Transfer request hash: `da35e214592d60ada3e1a4bbd39f7fad97848b069caede14d972eb91acbd9359`  
Exact response hash: `f07d3ade0be2f43104f1bce1d427e27375c77d2fbea9587dfbf5b989e86bde06`  
Contextual response hash: `2723ebe79d39235a460a25f7f42fc5f5158e5f132c695337613d33608f6d29d1`  
Transfer report hash: `abb88ddb3870372f574f4766a4be98e932a3a771f33e52fb2c512da4f38b0d34`

## Independent verification

`scripts/verify_kaggle_causal_world_run.py` streamed all 1,048,576 evidence
rows and independently checked:

- canonical submitted-source identity and every downloaded file hash;
- model, report, resume cursor, and workflow round trips;
- evidence count, evidence provenance, world-condition bindings, domains, roots,
  curriculum axes, update counts, TPU topology, and `pmap` execution;
- conservation, all aggregate experiment gates, deterministic shard replays,
  graph-RAG knowledge, rendered side-view binding, and absence of tracebacks;
- held-out/training program disjointness, cross-feature and balanced truth,
  request/model/truth bindings, all transfer report gates, and all five transfer
  artifacts;
- independent truth regeneration, recomputed exact/contextual evaluation, and
  byte-for-byte replay of both 576-turn response artifacts.

It exited `0`, reported no failed checks, and wrote the machine-readable audit
to `kaggle-results/version-8/verification.json`.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `b990af5f0788c0ccf84c340475ed458882f38d2640dd1ffdf579274172149e59`
- `atom_causal_world_evidence.jsonl`: `c8b6429ef27b241c8420dbcadc9e4830ef001764a86b119dd3b997f1c91cc4df`
- `atom_causal_world_knowledge_graph.json`: `97ba1e2a47d66bef4b24ca28c34fcf82544588de24064adee98f6a2bd02d452d`
- `atom_causal_world_manifest.json`: `032c3c2538f2bb1b3b0fb98b8acfddee3fb15f65e8c4ac8010a5813c5466bc98`
- `atom_causal_world_model.json`: `7026266ca90616f54767d6d3375bfc5abeaedd5126b5469d88f68e5658e3ea6e`
- `atom_causal_world_report.json`: `2c8bbafd63acb43a108d6e1fdcd99d6a81d8be61ae26b19c24a1be26f0f02233`
- `atom_causal_world_resume_cursor.json`: `6f29f8f718153965a9a9292265b62bb1077bcdc05b0500d318e2472ac4cded14`
- `atom_causal_world_side_view.html`: `147e266ea0eea98da080acee2a573762772d78b930ab779bbf55d3966f1b814e`
- `atom_causal_world_transfer_exact_response.json`: `bc40f79ae68384dee52217db91e527366c543ecbca874c834a74d99c73cbbb75`
- `atom_causal_world_transfer_report.json`: `bfc199a03e0e370b8889897f05c171866de30a5c63bdbb36af84cff1777bfb6a`
- `atom_causal_world_transfer_request.json`: `36dd51f66bbfcf0911a30544b3d175cc17d80b6d0974f8a549a77998e13b5c1f`
- `atom_causal_world_transfer_response.json`: `d2d065b51da554d2d9e513d6eae5a217b6b438a00d29ae1d88b084ee762206d5`
- `atom_causal_world_transfer_truth.json`: `bf9dd9ebfbd5e912a3c723b968882bf4ad5be6461b64dad1e7d4cf1c30263487`
- `atom_causal_world_workflow_request.json`: `6c5b7764892e5313b77ebdc63e3ade6904083ec6e4cacee9a8fb69d74238c6bb`
- `atom_causal_world_workflow_response.json`: `58d957e071bbf00d5eddbd2ecea1cb61a42e977cfdbf2646a38b0282f46e789b`
- `atom-massive-causal-world-v1.log`: `707424e3f8c94cfbf08ba8667819774ba0ed7d75c73d90b83a5b4e8dd03ad4c6`

## Interpretation boundary

This run measures conditional causal transfer within procedurally generated
worlds sharing the architecture's feature vocabulary and root mechanics. It
does not measure open-domain knowledge, unrestricted English conversation,
factual grounding against the external world, or performance against modern
language-model benchmarks. Its useful result is narrower: a persistent causal
graph generalized beyond exact stored regimes while preserving abstention,
source provenance, paraphrase invariance, and exact cross-platform replay.
