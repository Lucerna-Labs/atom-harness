# Atom Harness Developer Notes

## 1. Product identity

Atom Harness is a bounded AI orchestration runtime around Atom's causal
evidence system. Atom is the semantic authority. The language model is a
replaceable membrane that performs two constrained jobs:

1. map a natural-language question to an Atom query intent; and
2. render a bounded Atom evidence packet as a cited natural-language answer.

The language model is not a fact store, memory writer, planner with arbitrary
tools, or fallback source of truth. If Atom cannot support an answer, the
harness returns a deterministic abstention. The retired generative-English
distillation work remains available for audit, but it is not an active runtime
or training path.

The active runtime is declared as `language-harness` in
`ai-runtime-registry.json`. The machine-readable architecture contract is
`atom-language-harness-architecture.json`.

## 2. Repository lineage and publication boundary

This repository began as an isolated source fork under
`C:\Projects\atom-harness`. The source and required compact causal artifacts
were copied from `C:\Projects\atom lora`; that source checkout was not modified.
The original Atom LoRA safekeeping repository remains a separate private
project.

The operator authorized this harness as its own private Lucerna Labs
repository:

```text
https://github.com/Lucerna-Labs/atom-harness
```

Do not push harness commits back into `atom-lora` or `atom-vibe-coder` merely
because code lineage overlaps. Synchronization between those projects requires
an explicit decision and a reviewed diff.

Generated outputs, databases, model weights, dependency caches, and large
Kaggle artifacts are intentionally excluded. The committed compact source
artifacts are:

- `causal_world_outputs/atom_causal_world_evidence.jsonl`
- `causal_world_outputs/atom_causal_world_model.json`
- `primitive_forge_outputs/atom_primitive_graph.json`

Those files bootstrap the runtime wiki graph, graph RAG, and Atom DB evidence
store. Do not remove them without replacing the bootstrap contract and its
integration coverage.

## 3. Non-negotiable runtime invariants

Every runtime change must preserve these boundaries:

- **Atom owns meaning.** Only Atom evidence and validated Atom vocabulary may
  determine an answer.
- **The LLM is language-only.** It receives schemas and bounded data, not an
  Atom DB handle or a generic tool runner.
- **Memory is read-only during language requests.** The store SHA-256 is taken
  before and after each request; a mismatch is a runtime failure.
- **Retrieval is graph-first.** The runtime traverses relationships before
  selecting bounded textual evidence.
- **Evidence is untrusted data.** Evidence text cannot issue instructions to
  the model or runtime.
- **Citations are closed-world.** Every cited experience ID must occur in the
  exact evidence packet supplied to the renderer.
- **Abstention is authoritative.** `insufficient_evidence` bypasses response
  generation and forces the fixed Atom abstention.
- **Provider secrets are ephemeral.** Keys come from process environment and
  must never appear in provider manifests, artifacts, logs, prompts, or side
  views.
- **Artifacts are hash-bound.** The artifact, packet, Spiderweb trace, graph,
  workflow, and rendered side view must all refer to the same run.
- **The side view is a real runtime surface.** It renders the produced answer
  beside its evidence; it is not a mock or a separate demo.
- **The Spiderweb remains layered.** Do not flatten the runtime into a generic
  queue, actor system, or ordinary pub/sub pipeline.

These constraints are enforced in code and in
`tests/test_atom_language_harness_integration.py`. Documentation alone is not
evidence that they still hold.

## 4. Source map

| File | Responsibility |
| --- | --- |
| `atom_harness_experiment.py` | CLI entrypoint, provider construction, runtime execution, artifact writing, hash checks |
| `atom_harness_runtime.py` | Two-pass language flow, Atom authority gate, memory immutability check, Spiderweb trace |
| `atom_harness_knowledge.py` | Wiki graph construction, vocabulary preload, graph-first RAG, bounded evidence packets |
| `atom_llm_protocol.py` | Provider-neutral request/result protocol, strict intent and response schemas, boundary validation |
| `atom_llm_provider.py` | OpenRouter and llama.cpp production adapters plus deterministic test adapter |
| `atom_harness_side_view.py` | Hash validation and two-column user-visible artifact/evidence rendering |
| `atom_causal_experience.py` | Loading and interpreting the committed causal evidence corpus |
| `atom_causal_memory.py` | Python bridge to the dependency-free Rust Atom memory binary |
| `atom_causal_memory_rust/` | Atom DB, causal memory, retrieval field, and CLI workspace |
| `run-atom-harness.ps1` | Windows launcher with Python/NumPy discovery and provider selection |
| `ai-runtime-knowledge.json` | Required runtime wiki and RAG declaration |
| `ai-artifact-side-view.json` | Required user-visible side-view declaration |
| `atom-language-harness-architecture.json` | Machine-readable identity, trust, provider, bus, and claim boundary |
| `tests/test_atom_language_harness_integration.py` | Full harness contract and adversarial boundary tests |
| `tests/test_atom_causal_live_integration.py` | Preserved causal learning and evidence-kernel contract |

Historical research modules remain in the repository because the harness is
rooted in the causal Atom work. Their presence does not make each historical
experiment an active entrypoint. Use `ai-runtime-registry.json` to identify the
active product path.

## 5. End-to-end request flow

### 5.1 Bootstrap

On first use, the harness builds the release-mode Rust Atom memory binary if it
is absent, loads the checked-in causal evidence and model artifacts, creates a
fresh local Atom store, and opens the wiki/RAG runtime over that store.
Subsequent runs can reopen the store. Runtime stores are generated data and are
not committed.

### 5.2 Vocabulary preload

Before the language model interprets the question, the harness preloads the
valid query vocabulary from the wiki graph. The vocabulary hash is recorded in
the Spiderweb trace. This prevents an LLM from inventing subjects, relations,
domains, contexts, or query roles that Atom does not know.

### 5.3 Language pass one: intent

The provider receives:

- a fixed intent system prompt;
- the user's question;
- the preloaded Atom vocabulary; and
- `INTENT_JSON_SCHEMA`.

The result is parsed as exactly one JSON object. Duplicate JSON keys, extra
fields, invalid role combinations, unknown vocabulary, and malformed values
fail closed in `validate_intent`. The validated intent is then converted into
an Atom query; the provider cannot construct or execute a query directly.

### 5.4 Atom retrieval

`HarnessKnowledge.retrieve` traverses the wiki graph and selects causal records
that satisfy the validated intent. It emits a bounded packet containing the
query, graph paths, passages, source bindings, an explicit untrusted-data
notice, and a canonical packet hash.

No passage means `insufficient_evidence = true`. That condition is not advice
to the language model; it is an Atom disposition enforced by the runtime.

### 5.5 Language pass two: grounded rendering

Only an answerable packet enters the second language pass. The provider
receives the response schema and the bounded packet. It must return a response
with citations. `validate_grounded_response` checks the response shape,
answerability, abstention state, and citation membership.

If the packet is insufficient, the second pass is never called. The runtime
emits:

```text
I do not have enough Atom evidence to answer that.
```

### 5.6 Artifact and side-view binding

After response validation, the runtime checks that the Atom store hash is
unchanged. The experiment then writes the packet, graph, artifact, workflow,
and HTML side view. Canonical hashes bind those files. The renderer validates
the bindings before producing HTML, so stale or mixed-run inputs fail instead
of creating a plausible-looking view.

## 6. Spiderweb Bus contract

The harness uses Jesse's four-layer Spiderweb architecture:

- **L0 Transport** carries request and result bytes.
- **L1 Message** carries typed request, intent, evidence, and response
  messages.
- **L2 Flow** performs intent validation, Atom querying, graph traversal,
  evidence selection, response generation, and response validation.
- **L3 Orchestration** applies authority, provider, routing, and fail-closed
  disposition policy.

On-ramps promote validated work into the directional execution path. Off-ramps
return validated results to the flexible ground layer. The execution thread is
recorded from observed flow after the request runs; it is not declared as a
static pipeline in advance. Intersections emerge where retrieved graph paths
cross. Vibration records answerability or insufficient-evidence pressure and
propagates that disposition vertically. Vocabulary preload is the current
prefetch behavior.

When extending the flow:

1. preserve all four layers;
2. validate every ramp transition;
3. derive threads and intersections from actual message movement;
4. propagate backpressure or failure as vibration;
5. preload predictable data before the consuming node runs; and
6. add trace assertions to the integration test.

## 7. Provider contract

`JsonLanguageModel` is the only interface the runtime needs. A provider accepts
a `JsonGenerationRequest` and returns a `JsonGenerationResult`. The result
contains parsed JSON plus a non-secret completion manifest.

### OpenRouter

`OpenRouterJsonLanguageModel` reads `OPENROUTER_API_KEY` from the process
environment, sends a schema-constrained chat completion, uses temperature zero,
and requires provider parameter support. The default verified model is:

```text
mistralai/mistral-small-3.2-24b-instruct
```

The key itself must never be copied into a config file. `.env.example` contains
names and non-secret defaults only.

### Local llama.cpp

`LlamaCppJsonLanguageModel` invokes `llama-cli` with a local GGUF, a JSON schema
file, deterministic temperature, configured context length, and configurable
GPU-layer count. Prompt and schema data are passed through temporary files to
avoid Windows command-line length limits.

A GGUF file being present does not prove compatibility. The previously probed
`Ternary-Bonsai-4B-Q2_0_g64.gguf` failed against the installed llama.cpp build
with `invalid ggml type 42`. Use a GGUF supported by the installed llama.cpp
version and run a real two-pass request before claiming that local provider is
verified.

### Scripted provider

`ScriptedJsonLanguageModel` exists only for deterministic integration and
adversarial tests. Do not expose it as a production fallback. A scripted green
test proves the harness boundary; it does not prove external model
compatibility.

### Adding a provider

1. Implement `JsonLanguageModel`.
2. Enforce the supplied JSON schema at generation time when the backend
   supports it.
3. Parse exactly one JSON object and reject duplicate keys.
4. Keep credentials outside request payloads and manifests.
5. Give the provider no Atom DB or generic tool access.
6. Wire explicit provider selection in `atom_harness_experiment.py` and the
   launcher.
7. Add deterministic boundary tests and one real-provider smoke test.
8. Update the architecture contract and these notes.

## 8. Generated run artifacts

Each run writes an ignored directory under `atom_harness_outputs/`:

| Artifact | Meaning |
| --- | --- |
| `atom_harness_artifact.json` | Validated question, intent, answer, citations, memory hashes, provider manifests, checks |
| `atom_harness_evidence_packet.json` | Exact bounded evidence supplied to the rendering pass |
| `atom_harness_wiki_graph.json` | Runtime wiki graph used by retrieval |
| `atom_harness_workflow.json` | Cross-file hashes and source/store bindings |
| `atom_harness_side_view.html` | User-visible answer beside graph and evidence details |

Do not commit these generated directories. A useful report should identify the
output directory and verification results, not add runtime state to Git.

## 9. Setup and operation

Prerequisites:

- Windows PowerShell;
- Python 3.11 or newer with NumPy;
- a current Rust toolchain capable of edition 2024;
- either an OpenRouter key or a compatible `llama-cli` plus GGUF.

Install the Python runtime dependency:

```powershell
python -m pip install -r requirements-harness.txt
```

Run with OpenRouter:

```powershell
$env:OPENROUTER_API_KEY = '<secret from your secret manager>'
.\run-atom-harness.ps1 `
  -Provider openrouter `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Run with llama.cpp:

```powershell
.\run-atom-harness.ps1 `
  -Provider llama-cpp `
  -ModelPath 'C:\models\compatible-model.gguf' `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Use `-OutputDir` when a stable evidence location is needed. Without it, the
Python entrypoint creates a timestamped output directory.

## 10. Verification

Run all relevant checks after the latest change, not before it.

Python formatting, linting, and integration:

```powershell
python -m ruff check `
  atom_llm_protocol.py atom_llm_provider.py atom_harness_knowledge.py `
  atom_harness_runtime.py atom_harness_side_view.py atom_harness_experiment.py `
  tests/test_atom_language_harness_integration.py `
  tests/test_atom_causal_live_integration.py

python -m ruff format --check `
  atom_llm_protocol.py atom_llm_provider.py atom_harness_knowledge.py `
  atom_harness_runtime.py atom_harness_side_view.py atom_harness_experiment.py `
  tests/test_atom_language_harness_integration.py `
  tests/test_atom_causal_live_integration.py

python -m unittest `
  tests.test_atom_language_harness_integration `
  tests.test_atom_causal_live_integration -v
```

Rust workspace:

```powershell
Push-Location atom_causal_memory_rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
Pop-Location
```

Then run one real provider request. Verify all of the following in the retained
output:

- `passed` and `answerable` have the expected values;
- citations are members of the evidence packet;
- store hashes before and after are identical;
- artifact, packet, graph, trace, and workflow hashes match;
- the side view is bound to that exact artifact;
- provider secrets do not occur in generated files; and
- `git status --short` remains empty.

The real-provider baseline on 2026-07-29 used
`mistralai/mistral-small-3.2-24b-instruct` through OpenRouter. Its two-pass
request produced five packet-valid citations and a hash-bound side view without
changing Atom memory. That ignored local output is evidence for that exact
revision, not a substitute for a fresh run.

During the 2026-07-30 publication verification, all 20 active Python integration
tests and all 43 Rust tests passed after the documentation change, together
with formatting, lint, compilation, contract, and launcher checks. A repeat
cloud request was intentionally not sent because publishing the repository did
not itself authorize transmitting its private evidence packet to a third
party. An on-device probe with the available
`NVIDIA-Nemotron-3-Nano-4B-UD-Q6_K_XL.gguf` loaded successfully but did not
return a JSON object under the required intent schema. That model is therefore
not a verified local provider for this harness. Treat every provider claim as
model-, backend-, and revision-specific.

## 11. Safe extension points

### New Atom query vocabulary

Add vocabulary through the wiki graph, extend the protocol role deliberately,
update validation, and test both accepted and unknown values. Never let a
provider-created string bypass wiki membership.

### New evidence type

Give the evidence stable provenance and an ID, include it in graph traversal,
bound it into the packet hash, render it as untrusted data, and constrain
citations to packet membership.

### New output surface

Consume the validated artifact rather than provider output directly. Check all
canonical bindings before rendering. Keep the existing side view working until
the runtime declarations and integration tests explicitly select a
replacement.

### New tool

Do not attach a tool directly to the LLM. Model the operation as a typed Atom
request, authorize it at L3, route it through the Spiderweb layers, validate
both ramps, and bind the result as evidence before language rendering.

### New memory behavior

Language requests are currently read-only. Any learning or feedback path must
remain a separate, explicit Atom-owned transaction with provenance,
idempotency, conflict handling, and its own tests. It must not be smuggled into
the renderer pass.

## 12. Developer handoff checklist

Before merging or publishing a change:

- confirm `language-harness` is still the declared active runtime;
- confirm wiki graph, graph RAG, and side view remain runtime-wired;
- confirm no Rust crate exceeds 4,000 Rust source lines;
- inspect the staged diff for secrets and generated artifacts;
- run Python lint, format, compilation, and both integration suites;
- run Rust format, Clippy with warnings denied, and all workspace tests;
- run a real configured provider when provider/runtime behavior changed;
- inspect the real side view, not only the JSON;
- verify memory immutability and every hash binding;
- update architecture and developer notes when a boundary changes; and
- verify the exact remote commit after pushing.

If any authority boundary, real-provider path, runtime declaration, knowledge
integration, side-view binding, warning gate, or test remains unresolved,
report it as unresolved. Do not describe the harness as complete on the
strength of a scaffold or deterministic provider alone.
