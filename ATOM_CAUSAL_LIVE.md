# Trusted Live Causal Learning

The live layer turns Atom Causal Memory from a saved-world archive into a
runtime that can change through use. It does not train a dense model and it
does not treat its own answer as evidence. A prediction is made first. An
independent authority then supplies the observed outcome. That outcome becomes
an immutable structural experience and changes the conductance of the causal
paths used during the next interaction.

## One interaction cycle

Each event contains a situation and a trusted outcome.

The situation is the information the runtime may use before the outcome:

- domain and cause;
- contextual relationships;
- causal direction;
- ordered composition from the seven universe roots.

The trusted outcome adds:

- the effect that actually occurred;
- delay, magnitude, and invariant error;
- an authority kind and identity;
- a SHA-256 binding to the source evidence.

The accepted authority classes are operator, simulator, instrument, and
verified test, but the class label is not enough. The authority identity must
also appear in the runtime's explicit policy, and a separate canonical
evidence receipt must match the event fields and the declared SHA-256.
Retrieval output is not an authority class. This prevents the system from
remembering a guess merely because it generated that guess.

The runtime recalls prior experience without seeing the outcome and selects the
highest supported non-retired path. After the outcome arrives, it appends a
new observation carrying the complete situation, outcome, prediction, session,
and authority provenance.

- If the selected path predicted the observed effect, that path is
  strengthened.
- If it predicted a different effect, the selected path is weakened and the
  newly observed path is strengthened.
- If memory abstained, the observation is retained without inventing a
  prediction to reward or punish.

## Crash and replay behavior

Each outcome is keyed by the canonical SHA-256 of its event manifest. The Rust
command `observe-experience-once` stores that key, the expected and selected
paths, the outcome event, and all motif adjustments in one atomic cell.

Repeating the exact event returns `committed=false`, performs no adjustments,
and leaves the store byte-for-byte unchanged. Reusing the same key with a
different query or decision fails closed. If a process stops after the
observation batch but before feedback, replay can still apply the missing
outcome cell once because prediction provenance is stored on the observation.

## Inspectable memory

The Rust inventory now exposes every experience's exact structural features.
That surface is for inspection, recovery, and graph construction. Audit roles
such as session, authority, source hashes, and prediction provenance remain
forbidden as recall inputs.

The runtime wiki graph is rebuilt from the post-interaction inventory. Live
observations connect to their session through `observed_in` and to their
outcome authority through `certified_by`. Structural RAG calls the Rust recall
path and returns the exact motifs behind each context.

The live side view displays all three passes used by the experiment:

1. a prior path predicts the novel outcome incorrectly;
2. exact replay changes nothing;
3. a second session predicts the new effect from the first session's durable
   experience and receives positive feedback.

## Run it

```powershell
py -3.13 atom_causal_live_experiment.py --output-dir causal_live_outputs
py -3.13 -m unittest tests.test_atom_causal_live_integration -v
```

The output contains the inherited causal-world store, both event manifests,
cycle receipts, post-interaction inventory, wiki/RAG graph, hash-bound report
and workflow, query wires, and the user-visible right-side HTML view.

## Current measurement boundary

This experiment measures durable two-session adaptation on top of the full
saved causal world. A previously unseen effect is learned from a trusted
simulator result, selected in the next session, preserved after process
reopen, and protected from duplicate feedback. It does not establish that
arbitrary external information is true. Truth admission remains an explicit
authority boundary.
