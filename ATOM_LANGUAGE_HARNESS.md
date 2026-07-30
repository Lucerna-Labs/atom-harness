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
3. Atom selects a primary claim from the highest-ranked graph evidence.
4. If Atom reports sufficient evidence, the LLM renders that packet into a
   cited answer and copies the primary claim into an enum-constrained grounding
   object.
5. Atom validates the claim, answer fields, and every citation against the
   packet. If retrieval is insufficient, the harness emits a deterministic
   abstention without asking the LLM to fill the gap.

LLM output never becomes evidence and has no write path to Atom DB. V2 adds a
policy-routed provider fabric and an atomic run transaction around that same
authority boundary.

## Run

The default provider is local llama.cpp through its non-interactive
`llama-completion` executable. Its selected model is
`Qwen/Qwen3-4B-Instruct-2507`, using the official ggml-org Q8_0 GGUF. Install
and SHA-256 verify the weight once:

```powershell
.\install-atom-language-model.ps1
```

Then run the harness without a provider or model override:

```powershell
.\run-atom-harness.ps1 `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

The default path is the sibling model store
`C:\Projects\atom-harness-models\Qwen3-4B-Instruct-2507-Q8_0`. The 4.28 GB
weight is not part of Git. The runtime verifies its exact byte count and
SHA-256 before provider admission. The declared Qwen ChatML transport uses
temporary prompt and schema files, a 32,768-token context window, reasoning
off, temperature zero, and seed one.

Run all three live model certification cases, including the real two-pass
wiki/RAG path, unsupported-question abstention, side view, load latency, and
generation throughput:

```powershell
py -3.13 scripts\certify_atom_language_model.py
```

The adoption run on 2026-07-30 passed all three cases and five real local
completions with exact machine grounding, wiki graph and RAG, immutable Atom
memory, committed transactions, and the bound side view. Model-load latency
was 3,817.31 to 4,464.56 ms, with a 3,968.90 ms median. Generation throughput
was 74.24 to 93.55 tokens per second, with a 93.33 median. These measurements
are revision-specific and must be refreshed after model-path changes.

Cloud evidence egress is denied by default, even when
`OPENROUTER_API_KEY` is present. To authorize OpenRouter for the current
process, select it, provide an explicit current OpenRouter model ID, and pass
`-AllowCloud`. The key remains in the process environment and is never written
into an artifact:

```powershell
.\run-atom-harness.ps1 `
  -Provider openrouter `
  -LlmModel '<explicit OpenRouter model ID>' `
  -AllowCloud `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

A custom local GGUF is supported only with an explicit expected SHA-256 and
prompt transport:

```powershell
.\run-atom-harness.ps1 `
  -Provider llama-cpp `
  -ModelPath 'C:\path\to\language-model.gguf' `
  -ModelSha256 '<64 lowercase hex characters>' `
  -ChatTemplate raw-prompt-v1 `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

An ordered local-to-cloud route is explicit too:

```powershell
.\run-atom-harness.ps1 `
  -ProviderChain 'llama-cpp,openrouter' `
  -ModelPath 'C:\path\to\language-model.gguf' `
  -ModelSha256 '<64 lowercase hex characters>' `
  -ChatTemplate raw-prompt-v1 `
  -LlmModel '<explicit OpenRouter model ID>' `
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
`LlamaCppJsonLanguageModel` is the default local implementation and
`OpenRouterJsonLanguageModel` is the optional cloud implementation. The
official local model identity, artifact revision, byte count, SHA-256, runtime
settings, certification surfaces, and escalation policy are declared in
`atom-language-model.json`.
`ScriptedJsonLanguageModel` exists only for deterministic integration tests.
`ProviderFabric` admits providers from preloaded capability manifests before it
routes question or evidence data. A future provider must declare local,
private, or cloud location, return exactly one schema-constrained JSON object,
declare whether it is preemptible, honor cancellation when that capability is
true, and expose a non-secret manifest. It does not receive an Atom DB handle
or a tool runner.

The local adapter refuses `llama-cli`, which is interactive and decorates
stdout. It requires `llama-completion`, closes stdin, passes private prompts
only through temporary files, accepts only its fixed end-of-text sentinel
after one JSON object, and never persists raw prompts or backend logs.

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

When the official weight is installed, run the live certification too:

```powershell
py -3.13 scripts\certify_atom_language_model.py
```

The suite builds and exercises the Rust store, wiki graph, graph RAG, language
membrane, citation and abstention guards, provider admission, privacy,
fallback, typed retry, circuits, concurrency, cancellation, crash recovery,
Spiderweb trace, transaction integrity, and the bound side view together.
