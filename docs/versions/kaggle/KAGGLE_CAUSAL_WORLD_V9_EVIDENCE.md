# Kaggle causal-world Version 9 measured evidence

Run: <https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>

Source version: 9  
Canonical submitted-source SHA-256: `ac19e3967e15bab8c2930698ba9c7a5db5af1188f9e4c90dd7ccf878fedc3922`  
Downloaded Kaggle wrapper SHA-256: `218e9315c47b75975dad58f409fa34775a66808e4e1f7711de7e2f383c7d5f4c`

Version 9 adds a model-bound metaplastic transfer governor. It fits the
direction-prior strength and separate decreasing/increasing confidence
boundaries on validation programs that are absent from training, freezes the
selected policy, and measures that policy on a second set of world programs
absent from both training and validation. The downloaded artifacts passed an
independent Windows regeneration of the validation truth, all 500 policy
evaluations, the selected policy, the evaluation truth, and both 576-turn
response artifacts.

## Measured world execution

- Backend: `jax-xla`
- Accelerator: eight TPU v5-lite devices
- Executor: `pmap`, one cached XLA construction
- Shards: `0..15`, with 64 unique curriculum programs
- Evidence rows: `1048576`, with `131072` from each of eight domains
- Paired entity updates: `4294967296`
- Paired relation updates: `51539607552`
- Accelerator elapsed: `337.3845425199999` seconds
- Notebook HTML export recorded at approximately `1048.569` seconds
- Maximum conservation invariant error: `3.824358714155096e-07`
- Learned laws: `50992` total, `6352` crystallized
- Persisted workflow: `9/9`; seven derived answers and two evidence-bound unknowns
- Experiment gates: `28/28`
- Deterministic first-microbatch replay: `16/16` shards
- Python traceback: absent
- Wiki graph: `106` nodes, bound to the learned causal graph
- Graph-RAG runtime: `atom-causal-world-graph-rag-v2`
- Side view: user-visible, right-side placement, bound through
  `render_causal_world_artifact`

Model hash: `b8bc75712d15bb0c642e28f8895fee2e8c6fa6c9e46cdec86cdc11c389468586`  
Report hash: `15667aa7d478ae294cc607589be18f0b5f9a860991e1a9dae6548095a00a7dbc`  
Evidence hash: `ca52983735c1163f397117f84267171780ddea139ec392f5045b488e23ef4dc9`  
Resume cursor hash: `6f808dda69e327c33b9a6752d0a4efea6cf58f491f6380b51f579c2824bad1c8`

## Disjoint metaplastic validation

Validation used seed `2026072501` and 12 world programs absent from the
64-program training schedule. Two independent treated/control replicas
regenerated 288 cross-feature cases: 142 decreasing and 146 increasing. Two
English paraphrases per case produced 576 validation turns. The search evaluated
500 policies; 319 met the profile's eligibility bounds.

The default policy measured:

- coverage: `0.5486111111111112`
- correct assertions: `300`
- false assertions: `16`
- selective accuracy: `0.9493670886075949`
- false-assertion rate: `0.027777777777777776`
- safe-direction utility: `0.4930555555555556`

The selected policy used:

- direction-prior power: `0.4`
- token-likelihood power: `0.75`
- decreasing confidence boundary: `0.75`
- increasing confidence boundary: `0.95`

On validation, the selected policy measured:

- coverage: `0.6354166666666666`
- correct assertions: `340`
- false assertions: `26`
- selective accuracy: `0.9289617486338798`
- false-assertion rate: `0.04513888888888889`
- safe-direction utility: `0.5451388888888888`
- paraphrase consistency: `1.0`
- policy gates: `11/11`

The selected policy therefore increased validation coverage by 8.68 percentage
points and safe-direction utility by 5.21 points while remaining inside the
declared false-assertion and direction-preservation bounds. This comparison is
the selection evidence; the separate evaluator below is the out-of-sample
measurement.

Validation truth hash: `2f7ec332e600e348b9e542bda10dda4590f3ad9d28bdee378f342babe05c1389`  
Selected policy hash: `2c54a43ac9f1021c2a650d223b16dec28a9f52f12a269823fcb4c376ebabb614`

## Sealed held-out causal transfer

Evaluation used seed `2026072502` and selected another 12 world programs absent
from the 64 training programs and all 12 validation programs. Its independently
regenerated truth contained exactly 288 balanced cross-feature cases: 144
decreasing and 144 increasing. Two English paraphrases per case produced 576
inference turns. Neither the evaluation labels nor evaluation scores were
available to policy selection or inference.

Exact condition matching asserted no answer on any unseen regime. The frozen
metaplastic policy produced:

- assertions: `374/576`
- abstentions: `202/576`
- coverage: `0.6493055555555556`
- correct assertions: `346`
- false assertions: `28`
- selective accuracy: `0.9251336898395722`
- false-assertion rate: `0.04861111111111111`
- safe-direction utility: `0.5520833333333334`
- majority-direction safe utility: `0.0`
- decreasing-direction coverage / selective accuracy:
  `0.4513888888888889` / `0.9076923076923077`
- increasing-direction coverage / selective accuracy:
  `0.8472222222222222` / `0.9344262295081968`
- paraphrase consistency: `1.0`
- adversarial parser variants rejected: `3/3`
- transfer gates: `18/18`

Every derived transfer retained its source-law identities and provenance.
Questions below the selected confidence or source-regime boundaries returned
`unknown`. Graph-RAG was present on every transfer turn and used parsed domain,
cause, and effect fields to restrict retrieval before contextual composition.

Evaluation truth hash: `f80e16a585f7d5c8f6d250129077e757d1283e72aa57b65e4a447d50c36d6855`  
Transfer request hash: `a781fa8dff228e3397ad0f0046c3deaf17d333859700762584cfb228708ba92b`  
Exact response hash: `9b2a12c2bef0f853c358242e938b1665f4f8579898c7699dc2b1a1ccd57a43f7`  
Contextual response hash: `eea7713b3e0e841a259e5226201ed38235ded097d7eb4e11c60722c6f0d7915e`  
Transfer report hash: `e2907235c7144b0a43dee581baf3b4b4c60bd79e85a1e9100bb27568a242f64a`

## Independent verification

`scripts/verify_kaggle_causal_world_run.py` streamed all 1,048,576 evidence
rows and independently checked:

- canonical submitted-source identity and every downloaded file hash;
- model, report, resume cursor, and workflow round trips;
- evidence count, evidence provenance, world-condition bindings, domains, roots,
  curriculum axes, update counts, accelerator topology, and `pmap` execution;
- conservation, all aggregate experiment gates, deterministic shard replays,
  graph-RAG knowledge, rendered side-view binding, and absence of tracebacks;
- training/validation/evaluation program disjointness and explicit role binding;
- independent regeneration of both truth sets and the balanced cross-feature
  evaluation cases;
- exact regeneration of the 500-policy search and selected policy;
- recomputed exact/contextual evaluations and byte-for-byte replay of both
  576-turn response artifacts.

It exited `0`, reported `59/59` successful checks with no failed checks, and
wrote the machine-readable audit to
`kaggle-results/version-9/verification.json`. The audit file SHA-256 is
`6de6e97a00dc81a62c5953f43d380276974b479854fb53580315d2cf31380d94`.

## Downloaded artifact SHA-256

- `atom_causal_world_evaluator_truth.json`: `b990af5f0788c0ccf84c340475ed458882f38d2640dd1ffdf579274172149e59`
- `atom_causal_world_evidence.jsonl`: `c8b6429ef27b241c8420dbcadc9e4830ef001764a86b119dd3b997f1c91cc4df`
- `atom_causal_world_knowledge_graph.json`: `070a749b7ab8f6acbbae6239f58312c848c20f5ce09f21acd6ddf69a0e92b72f`
- `atom_causal_world_manifest.json`: `032c3c2538f2bb1b3b0fb98b8acfddee3fb15f65e8c4ac8010a5813c5466bc98`
- `atom_causal_world_model.json`: `abde99bc6d446c871b3c2a75eace6a39fb60416ff45d621fd0191a7884a58cd2`
- `atom_causal_world_report.json`: `b720e61f48acfbeb1f0af2ed7ef4b5525bf10eb0245310cb4f1314b4212248ab`
- `atom_causal_world_resume_cursor.json`: `44d63b77e4c52174d9ec572af7bc3f4eebcd799060447e505205ece9be9689c2`
- `atom_causal_world_side_view.html`: `ccb5f2e10ed0a83045f56722b08b86d8d3f3b7ab3069ab8186669cd3253ff6dc`
- `atom_causal_world_transfer_exact_response.json`: `2bf046fe767b96576b93e0ef07b234232128ad1b89159228b91d0a80b5e6be4d`
- `atom_causal_world_transfer_policy.json`: `f1cb6b574047cdf78adc6d22a4cc7ebf2f48ea1282144f244db9f4358f73967f`
- `atom_causal_world_transfer_report.json`: `bf89bcdf23949c178fd01a5b4394c9c7bd7edf1d6bbac53c77b3fcd6f08cdc42`
- `atom_causal_world_transfer_request.json`: `1008e088d13c48bd2283b66a02325b3b989f98144082dcbdf2c9b9e334879045`
- `atom_causal_world_transfer_response.json`: `128c6b6a867de54c58ed5e8e6a3f2d703557432e0218006ca423a0bc999a5b52`
- `atom_causal_world_transfer_truth.json`: `9cad9b4d3a9704e4f19ccb0618b4500695f09561e2083b63729ee8115e0851c0`
- `atom_causal_world_transfer_validation_truth.json`: `b98e3c85198f0c240b6a47d2f2a35e722bc8848d0505f13c1e3e897b2cf1ab8c`
- `atom_causal_world_workflow_request.json`: `6c5b7764892e5313b77ebdc63e3ade6904083ec6e4cacee9a8fb69d74238c6bb`
- `atom_causal_world_workflow_response.json`: `5141db6203a7cd056cfeedd3f1f4bb78a750273b6bbd3b9a9e1951256289ad8c`
- `atom-massive-causal-world-v1.log`: `1013e3bab2977c8941617999475fe015ff62dc512c225a9b9a86fe3192bfd942`

## Interpretation boundary

This run measures policy selection and conditional causal transfer within
procedurally generated worlds sharing the architecture's feature vocabulary and
seven root mechanics. It does not measure unrestricted English, external-world
factual grounding, open-domain conversation, or modern language-model
benchmarks. Its useful result is narrower: the system learned a confidence
policy from disjoint causal experience, froze it, generalized into a second
disjoint world set, retained provenance and abstention, and reproduced the
entire result across Kaggle Linux and local Windows execution.
