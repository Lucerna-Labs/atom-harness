# Atom Language Harness V2

The active product is a local AI harness. Atom remains the semantic authority:
it owns causal evidence, the durable database, the runtime wiki graph, graph
RAG, and the decision to answer or abstain. A replaceable LLM is used only as a
language membrane.

The language path has two passes:

1. The LLM maps a natural-language question onto exact values preloaded from
   the Atom wiki.
2. Atom validates the intent, performs graph-first retrieval, and creates a
   bounded evidence packet.
3. If Atom reports sufficient evidence, the LLM renders that packet into a
   cited answer.
4. Atom validates every citation against the packet. If retrieval is
   insufficient, the harness emits a deterministic abstention without asking
   the LLM to fill the gap.

LLM output never becomes evidence and has no write path to Atom DB. V2 adds a
policy-routed provider fabric and an atomic run transaction around that same
authority boundary.

## Run

Cloud evidence egress is denied by default, even when
`OPENROUTER_API_KEY` is present. To authorize OpenRouter for the current
process, select it and pass `-AllowCloud`. The key remains in the process
environment and is never written into an artifact:

```powershell
.\run-atom-harness.ps1 `
  -Provider openrouter `
  -AllowCloud `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

The second production adapter uses a local GGUF model through `llama-cli`:

```powershell
.\run-atom-harness.ps1 `
  -Provider llama-cpp `
  -ModelPath 'C:\path\to\language-model.gguf' `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

An ordered local-to-cloud route is explicit too:

```powershell
.\run-atom-harness.ps1 `
  -ProviderChain 'llama-cpp,openrouter' `
  -ModelPath 'C:\path\to\language-model.gguf' `
  -AllowCloud `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Retries with bounded exponential backoff apply only to typed transport,
capacity, and timeout failures. Schema failures can fall through to the next
provider, but they are never retried with a weaker schema. Circuits, bounded
concurrency, cancellation, and backpressure are part of the route contract.

Each successful run atomically publishes one ignored directory under
`atom_harness_outputs/` containing:

- `atom_harness_artifact.json`: validated request, answer, routes, and hashes;
- `atom_harness_evidence_packet.json`: bounded evidence shown to the LLM;
- `atom_harness_wiki_graph.json`: runtime graph used for RAG;
- `atom_harness_workflow.json`: cross-file and route hash bindings;
- `atom_harness_transaction.json`: committed file sizes and SHA-256 hashes;
- `atom_harness_side_view.html`: answer beside evidence and route state;
- `runtime/atom_harness_knowledge.atomdb`: the run-local read-only Atom store.

Files are first written into a private staging directory. The run is sealed,
verified, and published with one directory rename. Existing targets are never
overwritten. Dead-process stages are either recovered after full manifest
verification or quarantined.

## Provider boundary

`JsonLanguageModel` in `atom_llm_protocol.py` is the provider interface.
`OpenRouterJsonLanguageModel` is the configured cloud implementation and
`LlamaCppJsonLanguageModel` is the local implementation.
`ScriptedJsonLanguageModel` exists only for deterministic integration tests.
`ProviderFabric` admits providers from preloaded capability manifests before it
routes question or evidence data. A future provider must declare local,
private, or cloud location, return exactly one schema-constrained JSON object,
declare whether it is preemptible, honor cancellation when that capability is
true, and expose a non-secret manifest. It does not receive an Atom DB handle
or a tool runner.

## Spiderweb routing

The harness preserves the four-layer bus:

- L0 carries request bytes.
- L1 carries typed request, intent, evidence, and response messages.
- L2 runs intent validation, Atom query, wiki traversal, graph RAG, and
  response validation.
- L3 applies semantic authority and abstention policy.

On-ramps and off-ramps validate every layer transition. The execution thread is
recorded only after the flow occurs. Evidence intersections emerge from graph
paths. Provider retry, fallback, privacy blocking, circuit opening, and
backpressure propagate as executable Spiderweb vibrations into orchestration.
Insufficient-evidence pressure propagates vertically into the fail-closed
response.

## Verification

Run the policy checker and the declared V2 integration test with Python 3.13
and NumPy:

```powershell
py -3.13 scripts/verify_atom_harness_v2.py
py -3.13 -m unittest discover -s tests `
  -p 'test_atom_language_harness_v2_integration.py' -v
```

The suite builds and exercises the Rust store, wiki graph, graph RAG, language
membrane, citation and abstention guards, provider admission, privacy,
fallback, typed retry, circuits, concurrency, cancellation, crash recovery,
Spiderweb trace, transaction integrity, and the bound side view together.
