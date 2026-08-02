# Atom Causal Memory

Atom Causal Memory is the persistent world-memory between the open Primitive
Forge and the later reasoning and language layers.

It is a local fork of the Atom DB fact substrate. The Lucerna Labs repository
retains its lexical evidence-retrieval purpose. This project imports the
content-addressed atom, ternary bond, atomic-cell, causal-order, root, snapshot,
and observer laws into a separate Rust crate and gives that crate a different
retrieval membrane.

## What is stored

Each Primitive Forge record becomes a causal glyph. A glyph is not a text
chunk. It is a topological object containing:

- typed input and output ports;
- dimensions;
- ordered composition relationships;
- complete reduction to the seven immutable roots;
- invariants, symmetries, boundaries, and scales;
- lifecycle state;
- evidence, counterexamples, and provenance.

Every `(glyph, role, value)` relationship also receives its own content-named
motif atom. This allows learning to adjust an exact structural relationship
without globally promoting a word, passage, or shared value.

An import is one atomic cell. The active catalog is a named root pointing at
the exact Primitive Forge graph hash. Importing the same graph again changes
nothing. Importing a changed graph creates a causally versioned catalog while
retaining earlier facts.

## Structural resonance

A query supplies typed and topological constraints. The Rust runtime currently
accepts:

- domain and glyph kind;
- recipe mode;
- ordered component relationships;
- ordered seven-root expansion;
- input and output types and dimensions;
- invariants, symmetries, boundaries, and scales.

Aliases, document passages, evidence prose, provenance prose, embedding
vectors, and cosine similarity cannot enter this retrieval path. Matching
motifs inject role-specific activation. Direct composition and root-expansion
relationships carry more weight than descriptive metadata. Every result
contains the exact motif identities that supported it.

If required topology is absent, the runtime returns
`insufficient_evidence=true`.

## Prediction-driven learning

Learning consumes an observation with an expected glyph and the glyph selected
by the causal field.

- A correct prediction strengthens the matched motifs on that glyph.
- A wrong prediction weakens the matched motifs that supported the selected
  glyph and strengthens the matching structure on the expected glyph.

The adjustments are committed together in an atomic cell. They survive process
restart. Retrieval opens the store read-only and does not learn implicitly.

## Runtime boundary

Python remains the research orchestrator and validates the hash-bound Primitive
Forge artifact. It emits a deterministic hexadecimal line protocol. The Rust
runtime imports that protocol, owns the durable store, performs structural
retrieval, and commits prediction observations.

The transport is intentionally not the memory representation. After import,
the active data is reconstructed from Atom DB atoms, bonds, motifs, roots, and
feedback facts.

## Commands

Run the full real-artifact workflow:

```powershell
py -3.13 atom_causal_memory_experiment.py --output-dir causal_memory_outputs
```

Run the Python integration path:

```powershell
py -3.13 -m unittest tests.test_atom_causal_memory_integration -v
```

Run the Rust crate directly:

```powershell
cd atom_causal_memory_rust
cargo test -p atom-causal-memory
cargo clippy -p atom-causal-memory --all-targets -- -D warnings
```

The experiment writes the durable `.atomdb` store, inventory, query envelopes,
retrieval and learning report, runtime wiki graph, workflow binding, and HTML
side view.

## Causal-world experience layer

The same Rust substrate now accepts the causal world's complete observation
history and its consolidated laws. This moves the memory boundary beyond a
one-time Forge import:

- all 2,304 observation revisions receive immutable content identities, even
  when their source label repeats;
- 395 consolidated laws are stored separately and link directly to their
  supporting observations;
- observations and laws append as independently hash-bound atomic batches;
- structural recall operates over cause, effect, context, domain, direction,
  measured bands, and ordered seven-root composition;
- explicit outcome feedback adjusts exact motif paths and survives process
  restart;
- unknown required structure produces an explicit abstention.

Run this larger workflow with:

```powershell
py -3.13 atom_causal_experience_experiment.py --output-dir causal_experience_outputs
py -3.13 -m unittest tests.test_atom_causal_experience_integration -v
```

See `ATOM_CAUSAL_EXPERIENCE.md` and
`atom-causal-experience-architecture.json` for the batch, retrieval,
adaptation, wiki/RAG, and side-view contracts.

## Measurement boundary

The experiment measures structural retrieval and durable feedback over the
bounded Primitive Forge graph. It does not establish general language
understanding, complete physics, or universal reasoning.
