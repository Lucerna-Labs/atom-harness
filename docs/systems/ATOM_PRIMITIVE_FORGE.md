# Atom Open Primitive Forge

Atom now has an open mathematical primitive layer above its seven root
operators. The roots remain radiation, dissipation, gravitation,
attraction/repulsion, nucleation, conservation, and decay. They are the
immutable generative substrate, not a closed list of everything the runtime
may understand.

The hierarchy is:

```text
seven immutable roots
    -> unbounded recursively discovered primitives
        -> structures, laws, and systems
            -> later optional coding, Rust, and frontend projections
```

The Primitive Forge does not contain a handwritten enum of higher concepts.
It proposes typed compositions algorithmically, gives each normalized
composition a deterministic identity, quarantines it, and promotes it only
after repeated predictions agree with observations from distinct contexts.
A newly crystallized primitive is immediately available as an input to deeper
compositions.

Every derived record retains:

- its typed domain and dimensional signature;
- its direct composition recipe and equivalent derivations;
- its complete expansion to the seven roots;
- invariants, symmetries, boundaries, and scale metadata;
- supporting evidence and counterexamples;
- confidence, persistence, lifecycle status, and provenance.

Serial composition requires each output type and dimension to match the next
input. Parallel composition requires identical signatures. Feedback requires
a closed compatible dimension. Unknown references and cycles fail closed.
Equivalent parallel orderings and associative rewrites merge into the same
canonical identity while preserving every derivation and provenance path.

## Continual use

New structures enter `quarantined`, become `candidate` after support, and
become `crystallized` only after repeated predictive evidence. Contradictions
move a crystallized structure to `revised`; unsupported persistence decays
until the structure is `retired`. The seven roots cannot be mutated, trained,
revised, decayed, retired, or replaced.

The normal experiment exercises that lifecycle. A persisted graph can also
accept a hash-bound use observation:

```powershell
py -3.13 atom_primitive_experiment.py --model primitive_forge_outputs/atom_primitive_graph.json --observation observation.json --updated-model updated-primitive-graph.json
```

`build_use_observation_request()` creates the request envelope. The runtime
rejects a modified request hash, corrupt model hash, duplicate JSON keys,
unknown primitive, incompatible type or dimension, and cyclic graph.

## Discovery evidence

The executable discovery experiment searches ordered root pairs and then
reuses selected discoveries in a deeper search. It does not enumerate desired
named outcomes. Candidates are executed in a bounded scalar-field world and
checked against a separate flat evaluator operating on their root expansion.
Evaluation includes unseen controls from the calibration family and a
counterfactual family with changed drive, polarity, loss, attraction,
nucleation, conservation budget, and retention.

Run it with:

```powershell
py -3.13 atom_primitive_experiment.py --output-dir primitive_forge_outputs
py -3.13 -m unittest tests.test_atom_primitive_forge_integration -v
```

The output includes the hash-bound primitive graph, report, workflow request
and response, dynamic knowledge graph, and the user-visible HTML side view.
The wiki is generated from the actual inventory. RAG starts at matching graph
nodes and follows direct components and dependents; retrieved context includes
the complete root lineage.

## Claim boundary

The current measurements establish recursive composition, evidence gating,
serialization integrity, graph retrieval, and transfer inside the declared
bounded mathematical simulation. They do not establish complete physics,
quantum-mechanical understanding, perfect mathematics, consciousness, or a
complete model of the universe. Those are future experimental questions for
the same open composition architecture, not claims encoded into this result.
