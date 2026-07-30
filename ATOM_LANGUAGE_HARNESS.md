# Atom Language Harness

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

LLM output never becomes evidence and has no write path to Atom DB.

## Run

If `OPENROUTER_API_KEY` is already present, the launcher uses OpenRouter with
strict structured outputs by default. The key remains in the process
environment and is never written into an artifact. The verified default model
is `mistralai/mistral-small-3.2-24b-instruct`:

```powershell
.\run-atom-harness.ps1 `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

The second production adapter uses a local GGUF model through `llama-cli`:

```powershell
.\run-atom-harness.ps1 `
  -Provider llama-cpp `
  -ModelPath 'C:\path\to\language-model.gguf' `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Each run creates an ignored directory under `atom_harness_outputs/` containing:

- `atom_harness_artifact.json` — the validated request, answer, and hashes;
- `atom_harness_evidence_packet.json` — bounded evidence shown to the LLM;
- `atom_harness_wiki_graph.json` — the runtime graph used for RAG;
- `atom_harness_workflow.json` — hash bindings across the run;
- `atom_harness_side_view.html` — the user-facing answer beside its evidence.

The first run builds the dependency-free Rust Atom memory binary and creates a
fresh evidence store from the checked-in causal artifacts.

## Provider boundary

`JsonLanguageModel` in `atom_llm_protocol.py` is the provider interface.
`OpenRouterJsonLanguageModel` is the configured cloud implementation and
`LlamaCppJsonLanguageModel` is the local implementation.
`ScriptedJsonLanguageModel` exists only for deterministic integration tests.
A future provider must return one schema-constrained JSON object and expose a
non-secret manifest; it does not receive an Atom DB handle or a tool runner.

## Spiderweb routing

The harness preserves the four-layer bus:

- L0 carries request bytes.
- L1 carries typed request, intent, evidence, and response messages.
- L2 runs intent validation, Atom query, wiki traversal, graph RAG, and
  response validation.
- L3 applies semantic authority and abstention policy.

On-ramps and off-ramps validate every layer transition. The execution thread is
recorded only after the flow occurs. Evidence intersections emerge from graph
paths, and insufficient-evidence pressure propagates vertically into the
fail-closed response.

## Verification

Run the declared integration test with a Python environment that contains
NumPy:

```powershell
python -m unittest tests.test_atom_language_harness_integration -v
```

The test builds and exercises the Rust store, wiki graph, graph RAG, language
membrane, citation guard, abstention guard, Spiderweb trace, and bound side
view together.
