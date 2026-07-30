# Atom Causal Experience Memory

Atom Causal Experience Memory extends the Primitive Forge store from a static
primitive catalog into an append-only memory of events and causal
relationships. The Forge describes what can be composed. The experience layer
records what the causal world actually produced, what relationships survived
consolidation, and how later outcomes should alter structural recall.

This is not a vector database and it does not retrieve passages. An experience
is represented by durable `(experience, role, value)` motifs inside the Rust
Atom DB substrate. Cause, effect, context, domain, direction, state, measured
properties, and ordered seven-root composition are graph structure. Source
hashes and provenance remain available for audit but are barred from the
retrieval membrane.

## What enters memory

The current workflow binds three exact artifacts:

- the 69-glyph Primitive Forge graph;
- 2,304 immutable causal-world observation revisions;
- 395 consolidated causal laws.

The evidence file contains 845 recurring source labels. Those labels are not
used as primary keys. Each complete observation payload receives its own
content identity, so all 2,304 revisions survive instead of overwriting one
another.

Observations and consolidated laws enter through separate atomic batches. Each
batch records the SHA-256 identity of its source artifact. Replaying an
unchanged batch is idempotent: the runtime reports that no commit occurred and
retains the existing snapshot.

## Structural recall

A query is a small causal graph, not a sentence embedding. It can constrain:

- cause, effect, and contextual relationships;
- domain, direction, kind, and lifecycle state;
- delay, magnitude, support, confidence, and contradiction bands;
- ordered composition from the seven root primitives.

Required constraints are fail-closed. If the store has not observed the
required topology, the response contains no hits and explicitly reports
`insufficient_evidence=true`. Optional constraints can rank competing
relationships without allowing an absent required relationship to pass.

Every returned hit exposes the exact motifs, weights, conductance, and feedback
counts that produced its score. Recall opens the store read-only.

## Outcome-driven adaptation

Adaptation is explicit. An outcome event supplies the experience that should
have been selected and the one the field actually selected.

- A correct outcome strengthens matching motifs on the expected experience.
- A wrong outcome weakens the misleading path and strengthens the expected
  path in the same atomic update.

The adjustment survives process restart. The experiment proves this by
reopening the Rust store and repeating the same structural query. Merely
recalling a relationship does not alter it.

This is the first mechanism by which use can change future behavior, but it is
not unrestricted self-training. A trusted observation path still decides what
outcome enters memory. That boundary prevents retrieval alone from turning its
own output into evidence.

## Wiki graph, RAG, and side view

`atom_causal_experience_knowledge.py` builds the runtime wiki graph from the
exact records present in the durable catalog. Its RAG path calls structural
Atom DB recall and returns linked graph context. Consolidated laws point to
their supporting observation revisions with explicit `supported_by` edges.

`atom_causal_experience_side_view.py` renders the real experience artifact at
the side of the runtime surface. Before rendering, it validates the report,
inventory, workflow, knowledge graph, and store bindings. The visible result
shows record counts, domain and state distributions, the selected causal
relationship, feedback movement, and abstention behavior.

## Run it

Run the causal experience workflow:

```powershell
py -3.13 atom_causal_experience_experiment.py --output-dir causal_experience_outputs
```

Run its integration suite:

```powershell
py -3.13 -m unittest tests.test_atom_causal_experience_integration -v
```

Run the Rust substrate checks:

```powershell
cd atom_causal_memory_rust
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
```

The runtime emits the `.atomdb` store, inventory, queries, feedback workflow,
hash-bound report, wiki/RAG graph, and user-visible HTML side view.

## Current measurement boundary

The current experiment measures exact preservation, structural retrieval,
explicit abstention, deterministic replay, and durable outcome feedback over
the saved causal-world corpus. It does not establish open-domain language
ability, universal physical understanding, or autonomous truth acquisition.
Those abilities require the next layer to continuously turn trusted
interaction outcomes into new observation batches and to evaluate competing
causal explanations before consolidation.

That next layer is now exercised by `atom_causal_live_experiment.py`. It makes
a prediction before the outcome, accepts only independently certified outcome
evidence, appends the interaction, applies idempotent feedback inside Rust, and
uses the new experience during a second session. See `ATOM_CAUSAL_LIVE.md` and
`atom-causal-live-architecture.json`.
