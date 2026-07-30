# Atom Language Harness

## Active direction

The active product is now a local language harness around the causal Atom
runtime. Atom owns facts, causal memory, the runtime wiki graph, graph RAG,
tool-routing policy, and abstention. A replaceable local LLM supplies natural
language intent parsing and evidence-grounded answer rendering only.

The causal-live runtime remains the evidence and learning kernel. The language
model has no Atom DB write path, cannot promote its own output into evidence,
and cannot override `insufficient_evidence`. See
`ATOM_LANGUAGE_HARNESS.md` for the runnable path and trust boundaries.

The former 227M student / SmolLM2-1.7B teacher distillation pipeline is retired.
Its fixed multi-billion-token schedule reproduced the conventional
train-a-language-model bottleneck this project is intended to escape. Its
source, generated Kaggle runners, tests, and compact evidence are preserved
only for audit under `retired/generative-english-distillation/`; they are not
active runtime or training entrypoints.

The historical technical and operator records remain in
`ATOM_GENERATIVE_ENGLISH_DEV_NOTES.md` and
`ATOM_GENERATIVE_ENGLISH_USER_NOTES.md`.

## Atom-native construction language

Atom is the primary language of this architecture. It is not a helper syntax
around Python or Rust. `atom_language.py` defines a strict language, parser,
typed causal intermediate representation, reference interpreter, and
Atom-native intervention trainer. Every program starts from exactly seven root
mechanics, composes those roots into mathematical primitives, binds primitives
to learned capabilities, and arranges the result through the four Spiderweb
layers.

The current toolchain has three deliberately unequal language roles:

1. **Atom** owns construction, meaning, learning, memory, validation, and
   composition.
2. **Rust** is a generated native execution projection. The generated Cargo
   project has no third-party dependencies, embeds the Atom program hash, runs
   behavioral tests for every declared capability, and tests invalid requests
   for fail-closed behavior.
3. **Svelte with TypeScript** is a thin generated view. Its control calls an
   explicit Atom-to-Rust bridge; routing, projection, and other platform
   behavior do not live in the component.

`atom_native_experiment.py` is the integrated runtime. It learns capability
bindings from treated/control Atom interventions, compiles unseen programs to
Atom and Rust, validates Rust with both isolated hidden probes and Cargo,
validates the Svelte/TypeScript projection with the real compilers, retrieves
supporting context from the coding wiki graph and RAG path, and renders the
real artifacts together in the side view.

Run the experiment:

```powershell
py -3.13 atom_native_experiment.py --output-dir atom_native_outputs
```

Run its integration boundary:

```powershell
py -3.13 -m unittest tests.test_atom_native_language_integration -v
```

The architecture contract is recorded in
`atom-native-language-architecture.json`. Atom is intended to absorb more of
the execution and interface stack as its vocabulary and compiler grow. Rust
and Svelte/TypeScript are therefore useful, tested targets, but they are not the
semantic authority.

## Massive causal-world architecture

`atom_causal_world_experiment.py` is the graph-native architecture that replaces
the sequence-backbone direction explored earlier. Its persistent authority is a
causal phase hypergraph: nodes represent state, context, and mechanisms; edges
represent conditional executable laws with direction, delay, confidence,
contradiction counts, persistence, and provenance. English is only the query and
answer surface. It cannot manufacture facts that are absent from the graph.

Every query executes the full seven-component loop:

1. `CausalGraph` stores what depends on what.
2. `MolecularRecognition` selects laws whose binding sites fit the context.
3. `PhaseLockedLoop` synchronizes compatible laws over causal time.
4. `PhaseMixer` combines mutually supporting paths and cancels conflict.
5. `TopologicalPersistence` keeps reproducible structure and weakens contradicted
   structure.
6. `ThermalAnnealing` searches near the order/disorder boundary without changing
   the persistent graph during inference.
7. `ProjectiveMeasurement` emits a grounded discrete claim or `unknown`.

The runtime now also contains a typed formal-domain substrate for logic,
algebra, geometry, calculus, chemistry, biology, and information theory. Its
initial registry contains 15 executable primitives with explicit input
signatures, output types, numerical precision, invariants, and compositions of
the seven root mechanics. Runtime solutions and sealed evaluator solutions use
separate implementations. Exact supported results are marked `proven`, false
candidates are marked `contradicted`, and invalid or unsupported serialized
operations fail closed. Deterministic curricula divide generated cases into
demonstration, validation, and held-out partitions. Five typed programs already
compose results across calculus-to-algebra, geometry-to-algebra,
chemistry-to-logic, biology-to-algebra, and information-to-algebra while
retaining every stage's proof trace. A full accelerator run evaluates 7,680
formal cases, including 1,920 held-out cases, and binds the registry, truth,
responses, cross-domain programs, wiki/RAG nodes, and user-visible side view
into the experiment report.

The world compiler procedurally creates physical, chemical, biological,
ecological, agent, social, symbolic, and language systems from the seven root
mechanics. Learning sees matched treated and baseline outcomes. It does not see
the simulator's mechanics names, answer labels, or domain-mechanism names. An
active experiment scheduler chooses interventions that reduce uncertainty, and
independent evidence partitions keep a large accelerator shard from collapsing
billions of interactions into one statistical vote. Evidence records the signed
causal derivative rather than the raw outcome sign, so increasing and decreasing
interventions reinforce the same relationship when they should. Stable context
conditions create separate law variants, allowing genuine conditional behavior
without turning every contextual reversal into a false contradiction. Each
intervention retains eight distinct effect channels rather than only its loudest
consequence. The runtime can then compose compatible crystallized edges into
transient causal paths up to four links deep; every derived path carries the
ordered IDs and provenance of its source laws and never becomes an invented
persistent edge.

Semantic scale now comes from a compositional curriculum rather than a larger
dense state vector. A world program combines two distinct members of the seven
root mechanics with nine independent regimes: scale, resources, signal,
relations, time, topology, phase, energy, and boundary behavior. This creates
52,500,000 lazily decoded regimes, or 420,000,000 domain-conditioned worlds,
without allocating a lookup table. The selected program changes initial state,
relation topology and channel strength, and root-mechanic gains. Its eleven-part
stable signature is bound to every evidence record and causal-law key, then
rendered and parsed in ordinary English so contradictory regimes cannot be
silently averaged into one answer. Training evidence retains raw simulator
trace binding. Independently regenerated evaluator truth uses rounded semantic
measurements plus program and replica identity, so numerically equivalent
causal evidence has the same identity across accelerator and local replay.

The `tpu-massive` profile contains 16 cumulative shards, 8,192 worlds per shard,
256 entities per world, 12 relations per entity, 64 time steps, and 64 candidate
interventions. Across the profile that is 2,147,483,648 entity updates and
25,769,803,776 relation updates before counting the paired treated/baseline
rollouts. Dense world evolution stays on JAX/XLA; compact persistent graph
consolidation stays on the host. On Kaggle, the generated TPU and GPU scripts
execute all 16 shards in one process by default, so the fixed-shape XLA program
is compiled once and reused. A multi-device TPU assignment uses `pmap`; a
single GPU assignment uses `jit`. Both paths require the requested accelerator
and reject a CPU-only or mismatched runtime instead of silently falling back.
Each 512-world microbatch is split evenly across the assigned devices and
recombined in deterministic world order. After every shard, atomic model and
cursor writes bind the
ordered shard list, per-shard evidence hashes, every intermediate model hash,
the cumulative evidence count, and the next legal shard. A resumed run rejects
altered, skipped, duplicated, reordered, or model-mismatched state.
Each massive shard now cycles through four independently reduced curriculum
programs. The 16-shard run therefore exercises 64 real regimes and emits up to
65,536 evidence rows per shard (1,048,576 across the run), with four independent
world partitions behind each intervention/domain/regime hypothesis.

Unseen-regime inference is evaluated separately from the training workflow.
The evaluator selects world programs absent from training, regenerates each
causal direction from two independent treated/control simulations, excludes
self-effects, balances increasing and decreasing cross-feature relations, and
seals those labels away from inference. Exact retrieval is compared with a
direction-neutral context-factor graph. Persistent laws contribute a direction
prior, eleven singleton-condition likelihoods, and all 55 pairwise condition
motifs. Projection requires at least three independent source regimes and emits
the complete factor trace with every contextual claim or rejection.

The metaplastic governor makes one expensive inference pass over validation,
then reprojects those immutable factor traces across 5,000 combinations of
direction-prior power, pair-motif power, and separate decreasing/increasing
consensus thresholds. A policy is eligible only when its 95% Wilson
selective-error upper bound is at most 10% overall and at most 15% in either
direction, while preserving paraphrase invariance, direction accuracy, and
utility relative to the default projection. Wilson values, condition
log-likelihoods, and consensus use 80-digit decimal functions with a
twelve-decimal half-even projection before either thresholding or hashing. This
removes operating-system math-library tails without changing the declared risk
resolution. Validation now uses 24 programs disjoint from both training and the
final 12-program evaluation. The selected policy is hash-bound to the learned
graph and frozen before evaluator questions. Its replay digest covers the
complete 5,000-policy projection lattice: every searchable policy paired with
the evaluation hash produced from the shared factor probe. The artifact
preserves every source-law ID and returns
`unknown` when the selected risk boundary is not met. These bounds constrain
policy selection; they are not a guarantee about unseen distributions.
Structured graph-RAG uses the parsed domain/cause/effect fields to restrict
learned-law retrieval while the wiki graph remains present on every turn.

Before the independent run, the stricter balanced development set against the
version-6 graph contained 288 cross-feature cases and 576 English paraphrases.
The axis-conditioned runtime answered 54.51% with 95.54% selective accuracy, a
2.43% false-assertion rate, and 100% paraphrase consistency; exact condition
matching answered none of the unseen regimes. Structured graph-RAG reduced the
576-turn replay from 159.8 seconds to 7.14 seconds.

Kaggle version 8 then evaluated a fresh seed over 12 world programs absent from
the 64-program training schedule. Its separately regenerated truth contained
288 balanced cross-feature cases and two English paraphrases per case. Exact
matching again had zero coverage. Axis-conditioned transfer asserted 328 of 576
turns (56.94% coverage), with 310 correct assertions, 18 false assertions,
94.51% selective accuracy, a 3.125% false-assertion rate, 50.69% safe-direction
utility, and 100% paraphrase consistency. The independent Windows verifier
regenerated the truth and reproduced both exact and contextual responses
byte-for-byte. This is evidence for transfer inside the simulated causal-world
distribution, not evidence for unrestricted English or open-world factuality.

Kaggle version 9 measured the metaplastic runtime end to end. Validation used
12 programs absent from training, evaluated 500 policies, and selected prior
power `0.4`, a decreasing threshold of `0.75`, and an increasing threshold of
`0.95`. The policy was then frozen and evaluated on 12 different programs
absent from both training and validation. That balanced 576-turn evaluation
reached 64.93% coverage, 45.14% decreasing coverage, 92.51% selective accuracy,
a 4.86% false-assertion rate, 55.21% safe-direction utility, and 100% paraphrase
consistency. The independent Windows verifier regenerated both truth sets, all
500 policy evaluations, the selected policy, and both response artifacts, with
`59/59` checks passing. Detailed hashes and measurements are recorded in
`KAGGLE_CAUSAL_WORLD_V9_EVIDENCE.md`.

Versions 10 and 11 expanded validation to 24 disjoint programs and the
metaplastic search to 5,000 policies, including pair-motif strength and separate
decreasing/increasing consensus controls. Version 10 exposed accelerator-local
evaluator provenance; version 11 corrected that and independently regenerated
both truth sets, but its policy identity still included numerically irrelevant
raw factor-trace tails. Version 11 passed `65/66` audit checks, with only exact
policy regeneration rejected. The measurements and exact difference are
recorded in `KAGGLE_CAUSAL_WORLD_V10_EVIDENCE.md` and
`KAGGLE_CAUSAL_WORLD_V11_EVIDENCE.md`.

The separate private GPU kernel is
<https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-gpu-v1>.
GPU version 1 assigned one Tesla P100-PCIE-16GB and executed the same
million-evidence workload through one cached JAX/XLA `jit` executor in
`117.83772524400365` accelerator seconds. That was `2.86623416585501` times
faster than TPU version 11's measured accelerator section. The GPU run
reproduced the same aggregate held-out behavior but retained version 11's raw
trace policy digest, so its independent audit correctly rejected two dependent
replay checks. `KAGGLE_CAUSAL_WORLD_GPU_V1_EVIDENCE.md` records the measured
boundary.

GPU version 2 bound the complete 5,000-policy projection lattice and again
reproduced every persisted control, validation measurement, answer decision,
RAG artifact, and side view. Its `64/66` audit isolated the remaining mismatch
to unrounded Wilson-score square-root tails inside non-selected lattice
evaluations. A counterfactual replay showed the observed values converge exactly
at twelve decimal places. The diagnosis is recorded in
`KAGGLE_CAUSAL_WORLD_GPU_V2_EVIDENCE.md`. Policy runtime v5 applies that
normalization before risk thresholding and hashing. An exhaustive check of all
166,752 possible count pairs through 576 turns then found five values close
enough to a decimal half-step that binary square-root rounding was still
ambiguous. Policy runtime v6 removes that residual boundary by computing the
Wilson statistic with fixed-precision decimal arithmetic.

GPU version 3 confirmed exact regeneration of validation truth and the complete
5,000-policy artifact. Its audit passed `67/68` checks; the only mismatch was
two pair-motif log-likelihood diagnostics, each separated by
`0.000000000001`, while every policy decision and answer agreed. The exact
fields are recorded in `KAGGLE_CAUSAL_WORLD_GPU_V3_EVIDENCE.md`. Context-factor
runtime v2 and policy runtime v7 move condition logarithms and consensus onto
the same fixed-precision decimal substrate so the diagnostic response remains
strictly bound instead of being ignored.

GPU version 4 verified the deterministic decimal Wilson path: its full policy
artifact reproduced exactly and the audit passed `68/69` checks. It retained
the same two response-trace log tails by design, isolating risk math from
condition math. The measured run and hashes are recorded in
`KAGGLE_CAUSAL_WORLD_GPU_V4_EVIDENCE.md`.

GPU version 5 moved direction priors, singleton and pair-motif likelihoods, and
consensus transforms onto the fixed-precision decimal substrate. One P100
generated `1048576` evidence rows in `117.67405132099918` accelerator seconds,
learned `50989` laws with `6350` crystallized, and passed `30/30` experiment,
`15/15` policy, and `18/18` transfer gates. Independent Windows reconstruction
then passed `69/69` checks, including exact policy and contextual-response
regeneration, graph-RAG, and the bound side-view artifact. This closes the
cross-platform replay boundary observed in GPU versions 1 through 4. The
measured run, hashes, and scope boundary are recorded in
`KAGGLE_CAUSAL_WORLD_GPU_V5_EVIDENCE.md`.

GPU version 7 added the seven-domain typed formal substrate to that full
causal-world workload and corrected the registry's Python-tuple versus
serialized-JSON-list boundary. One P100 generated `1048576` evidence rows in
`117.6007801620004` accelerator seconds, learned `50989` laws with `6350`
crystallized, evaluated `7680` formal cases including `1920` held out, and
passed `35/35` experiment, `15/15` policy, `18/18` transfer, and `8/8` formal
gates. Independent reconstruction from the downloaded executed source passed
`72/72` checks, including exact formal-artifact, oracle, graph-RAG, workflow,
and side-view reproduction. The measured run and hashes are recorded in
`KAGGLE_CAUSAL_WORLD_GPU_V7_EVIDENCE.md`.

Mass, energy, and resources remain globally conserved when an entity expires.
The world retains at least one carrier, normalizes both the previous and proposed
local distributions to the original budget, and lets the conservation root
control retention versus redistribution between those two valid distributions.
This keeps death and decay from numerically annihilating matter while preserving
the conservation atom as an active local dynamic rather than a reporting-only
constraint.

The architecture contract is recorded in `causal-world-architecture.json`.
The runtime knowledge and side-view contracts are declared in
`ai-runtime-knowledge.json` and `ai-artifact-side-view.json`.

The private full-scale Kaggle run is at
<https://www.kaggle.com/code/jessealicea/atom-massive-causal-world-v1>.
Version 9 exercised the `tpu-massive` profile on eight TPU v5-lite devices
through `pmap`: 16 shards, 64 unique programs, 1,048,576 evidence rows,
4,294,967,296 paired entity updates, and 51,539,607,552 paired relation updates.
The accelerator section took `337.384543` seconds, learned 50,992 laws with
6,352 crystallized, reproduced every first microbatch, passed `28/28`
experiment gates, `11/11` metaplastic-policy gates, and `18/18` transfer gates,
and preserved the runtime wiki graph, graph-RAG, and bound right-side artifact
view. The canonical source SHA-256 is
`ac19e3967e15bab8c2930698ba9c7a5db5af1188f9e4c90dd7ccf878fedc3922`.

Version 3 exercised the exact `tpu-massive` profile on eight TPU v5-lite
devices through `pmap`: all 16 contiguous shards, 131,072 worlds/evidence
rows, 4,294,967,296 paired entity updates, and 51,539,607,552 paired relation
updates. The accelerator section took `336.883607` seconds, or approximately
12.75 million entity updates and 152.99 million relation updates per second.
It constructed one cached XLA executor, reproduced the first microbatch of all
16 shards exactly, learned 412 laws (105 crystallized), answered six direct and
one genuinely two-link causal query, withheld two unsupported claims, and
passed all 21 recorded experiment gates. The report hash is
`d2fc337fa98bc73ad13aa97efef455eebc867ad66e6a68a3f966f9a24b0cc95a`;
the model hash is
`09104bc3a755df807ca3b3732db756565c390879a7ebecd7623e404215f612ec`.

Version 2 supplied the useful negative result that shaped the fix: all 16
sequential shards and 131,072 evidence rows were safely persisted, but its
poorer accelerator-world dynamics produced no genuine crystallized two-hop
chain and finalization raised after `3122.501386` seconds. Version 3 ports the
full domain cascades into the accelerator world, preserves a failed gate/report
instead of raising when a composition is absent, and reached its report at
`439.993110` seconds, a measured 7.10x end-to-end improvement. The complete
artifact hashes, replay checks, cursor bindings, and platform diagnostics are
recorded in `KAGGLE_CAUSAL_WORLD_V3_EVIDENCE.md`.

Version 4 is retained as a negative large-run result. It executed all 16 shards
and emitted 524,288 condition-bound evidence rows, but its 64-program schedule
covered only two of five scales and three of four boundary regimes, and several
accelerator worlds lost their entire conserved budget after entity expiration.
The fail-closed verifier rejects that download for exactly axis coverage,
conservation, and the aggregate experiment gate. Its machine-readable audit is
`kaggle-results/version-4/verification.json`; the measured diagnosis and file
hashes are recorded in `KAGGLE_CAUSAL_WORLD_V4_EVIDENCE.md`.

Version 5 exercised the corrected schedule across every value of all nine axes,
all seven roots in primary and secondary roles, 524,288 evidence rows, and the
full paired 4.29-billion entity / 51.54-billion relation update workload. It
still predates the conservation repair and is rejected only for conservation
and its aggregate experiment gate. Its machine-readable audit is
`kaggle-results/version-5/verification.json`; measured execution and hashes are
recorded in `KAGGLE_CAUSAL_WORLD_V5_EVIDENCE.md`.

Version 6 ran the same full workload on eight TPU v5-lite devices with the
conservation repair. Maximum invariant error fell from `1.0` to
`3.824358714155096e-07`; all 25 experiment gates, all nine curriculum axes,
524,288 evidence rows, 16 deterministic shard replays, and the 9/9 persisted
workflow passed the independent verifier. Its machine-readable audit is
`kaggle-results/version-6/verification.json`; measured execution and hashes are
recorded in `KAGGLE_CAUSAL_WORLD_V6_EVIDENCE.md`.

Version 7 doubled the retained evidence to 1,048,576 rows and supplied the first
massive held-out transfer measurement, but its strict audit exposed a one-bit
Linux/Windows floating-point representation difference in a diagnostic
similarity value. All directions, sources, scores, and truth labels agreed, yet
the contextual response did not reproduce byte-for-byte, so version 7 was not
accepted as deterministic evidence. Version 8 projects persisted transfer
diagnostics to a stable 12-decimal representation. It retained the same measured
behavior while passing exact response replay, independent truth regeneration,
all transfer gates, graph-RAG, side-view binding, and every pre-existing causal
world check. Its machine-readable audit is
`kaggle-results/version-8/verification.json`; the measured record and artifact
hashes are in `KAGGLE_CAUSAL_WORLD_V8_EVIDENCE.md`.

```powershell
py -3 atom_causal_world_experiment.py --self-test
py -3 atom_causal_world_experiment.py --profile local --backend numpy
py -3 -m unittest tests.test_atom_causal_world_integration -v
py -3 scripts/build_kaggle_causal_world_bundle.py
py -3 scripts/build_kaggle_causal_world_bundle.py --accelerator gpu
py -3 kaggle/causal-world-v1/atom_causal_world_kaggle.py --self-test
py -3 scripts/verify_kaggle_causal_world_run.py kaggle-results/version-N --source-file kaggle-results/version-N-source/atom-massive-causal-world-v1.py --expected-bundle-sha256 SHA256 --require-transfer-benchmark --expected-accelerator tpu
py -3 scripts/verify_kaggle_causal_world_run.py kaggle-results/gpu-version-N --source-file kaggle-results/gpu-version-N-source/atom-massive-causal-world-gpu-v1.py --expected-bundle-sha256 SHA256 --require-transfer-benchmark --expected-accelerator gpu
# Example continuation after four accelerator shards:
py -3 atom_causal_world_experiment.py --profile tpu-massive --backend jax-xla --shard-index 4 --shards-per-run 4 --resume-from .\causal_world_outputs
```

## Evidence-bound natural English

`atom_english_language_experiment.py` adds an ordinary-English shell to the
consequence-grounded neural field. The user can ask requests such as "please
spread and gather, then report the node with greatest signal." A semantic-free
codec removes English function words but does not translate action words into
operators. The learner must still infer that words such as `spread`, `gather`,
and previously unseen synonyms such as `disperse` or `collect` denote physical
processes by comparing the field before and after each event. Answers are
rendered back as normal English.

The codec reduces the active vocabulary from 66 words in the first English
attempt to 35 content tokens. The resulting Atom model has 25,161 parameters;
the matched flat neural control has 48,865. All seven root primitives remain
causally active in both the recurrent text field and frozen world field. The
nucleation branch uses a small mixed pre-crystallization phase so short English
requests cannot switch that primitive completely off.

Adaptation now includes a metaplastic policy-selection layer. The adaptive
thermal/phase governor and the simple fixed schedule train independently. Two
of every six recovery cases are held out, giving 42 consequence cases that
select the policy by joint accuracy, state accuracy, continuous error, and
update cost. The untouched 96-case transfer-composition set is never consulted
by the selector. In the current deterministic run both candidates scored
`0.952381` on selection; the secondary criteria selected the adaptive policy.

The selected 25,161-parameter model scored `0.916667` joint accuracy on the 96
untouched English transfer compositions, retained `0.833333` accuracy on 192
base-language compositions, and reached `0.828125` on the same composition set
after training on only 252 of 1,008 base examples. The flat control scored
`0.000000` on base composition. The evidence gate covered all 96 supported
transfer cases with `0.906250` assertion accuracy, made no assertion on any of
48 grounded-but-unlearned requests, and also rejects arbitrary out-of-vocabulary
English before recurrent execution. Evidence preflight and the one-tick fast
path reduced recurrent work by `0.523810` across the supported and unsupported
sets.

The serialized eight-turn workflow contains six supported English requests and
two unsupported requests. It scored 8/8, supplied wiki-graph/RAG context on
every turn, and bound every answer to the saved model and right-side artifact
view. All 24 experiment gates passed, five corrupt outer/nested model variants
failed closed, deterministic retraining reproduced model hash
`4a2887bc2e338eede2434b0d717dd76f59c57b2a1d1a330f0cb0841c26ca77d1`,
and the modular source and generated single-file bundle produced identical
Windows artifacts. Kaggle's Linux runtime produced different floating-point
weights and one different base-retention decision (`0.828125` versus local
`0.833333`), so weight hashes are intentionally not treated as portable. The
untouched transfer decisions and eight workflow decisions were identical, with
cross-platform behavior hash
`67b2cbc5861e887fd08f68ef1108acda6745ce41b51e2daa296bdcdd67dc4f02`.

The private Kaggle notebook is
<https://www.kaggle.com/code/jessealicea/atom-evidence-bound-english-v1>.
Its generated source is retained locally at
`kaggle/english-language-v1/atom_english_language_kaggle.py`; the exact source
downloaded from Kaggle version 2 is under
`backups/kaggle-notebook-atom-evidence-bound-english-v1-v2-20260722-081514`.

This is a bounded synthetic field-language experiment, not an open-domain
chatbot. It learns 28 English action surfaces over seven known root mechanics,
six query types, and 13 answer tokens. Its abstention guarantee is internal: it
prevents an unsupported internal candidate from becoming an assertion, but it
does not establish that supplied training consequences are externally true or
solve unrestricted factual hallucination.

```powershell
py -3 atom_english_language_experiment.py --self-test
py -3 atom_english_language_experiment.py --output-dir english_language_outputs
py -3 -m unittest tests.test_atom_english_language_integration -v
py -3 scripts/build_kaggle_english_language_bundle.py
py -3 kaggle/english-language-v1/atom_english_language_kaggle.py --self-test
```

## Lifelong neural language field

`atom_neural_language_experiment.py` is the first integrated neural-language
architecture in this repository. It does not receive operator labels,
translations, language identities, query meanings, simulator controls, or
family names at runtime. Each event contains an opaque utterance, a graph field
before the event, the observed field afterward, and an opaque response. Human
labels and controls exist only in a separate evaluator artifact.

The architecture is a 30,012-parameter Atom field assembled in eight interacting
layers. A consequence inducer replays candidate root dynamics against observed
before/after fields. An operator memory crystallizes opaque surface tokens into
the seven root controls. A recurrent text field processes every utterance
through radiation, gravitation, attraction/repulsion, dissipation, nucleation,
conservation, and decay branches simultaneously. A frozen root executor applies
the induced controls to the world graph. A factorized cognitive memory discovers
six opaque query meanings but stores answer surfaces by language, answer kind,
and semantic value rather than duplicating them for every query. An
Atom-composed homeostatic governor controls adaptation. An evidence gate permits
an assertion only when operator, execution, query, and answer-surface support
form a complete path, while an adaptive compute gate skips unsupported requests
and reduces crystallized requests from three text-field ticks to one.

The generated curriculum has 2,230 events across four disjoint opaque
languages. Two languages teach 21 one-, two-, and three-operator composition
families. A third language supplies coherent adaptation, an incoherent noise
stream, recovery evidence, and four held-out compositions containing four to
seven operators. A fourth, entirely unseen language is retained as a negative
control. The flat neural comparison has 56,727 parameters and sees the same
base training rows.

The current local v2 run scored `0.937500` joint accuracy on 96 held-out
transfer compositions. The fixed schedule scored `0.364583`, the 56,727-
parameter flat comparison scored `0.000000` on base composition, base-language
retention was `0.885417`, and the unseen-language control remained `0.000000`.
The evidence policy asserted all 96 supported cases with `0.937500` joint
assertion accuracy, made zero assertions on all 48 unsupported-language cases,
and the flat generator emitted a candidate for all 48. Preflight rejection plus
the one-tick supported fast path reduced recurrent text/field updates by
`0.523810` across those 144 cases. Factorization reduced the surface-memory
table from six answer channels to two, a two-thirds cell reduction.

A coverage-balanced curriculum retained one world variant for each
language/composition/question cell. It used 252 of the normal 1,008 base
training events and reached `0.885417` joint accuracy on 192 unseen base
compositions after the same five epochs. This selection uses evaluator metadata
to preserve curriculum coverage; it measures redundancy removal, not autonomous
example selection.

The eight-turn serialized workflow exercised six supported derivations and two
unsupported requests through the model, wiki graph, graph RAG, and model-bound
side view. The supported turns used one text tick and four field ticks with
three-entry evidence paths; both unsupported turns returned `unknown` without
executing either recurrent field. All 33 experiment gates passed, seven corrupt
serialized variants were rejected, and deterministic retraining reproduced
model hash
`ca4a4b45afa8b0fec6da86320fc74f472ccf8ed1f5126387d00528d1131ca316`.

The existing private Kaggle kernel at
<https://www.kaggle.com/code/jessealicea/atom-lifelong-neural-language-field-v1>
records the preceding v1 architecture. The v2 evidence, compute, and factorized
memory changes have only been exercised locally at this point.

The measured boundary is deliberately sharp. This is synthetic consequence-
grounded field language, not free text. The seven root mechanics and the finite
candidate composition lattice are known to the consequence inducer, and the
observable question inventory has six forms. The zero-shot language result is
intentionally negative: the architecture adapts a new language from coherent
experience, but it does not infer a completely disjoint language without any
grounding events. These results test compositional consequence induction,
lifelong lexical adaptation, selective forgetting, and controlled chaos inside
that domain. The evidence gate proves internal support lineage, not external
world truth, so these measurements do not establish unrestricted language
understanding, universal hallucination prevention, or general intelligence.

```powershell
py -3 atom_neural_language_experiment.py --self-test
py -3 atom_neural_language_experiment.py --output-dir neural_language_v2_outputs
py -3 -m unittest tests.test_atom_neural_language_integration -v
py -3 scripts/build_kaggle_neural_language_bundle.py
py -3 kaggle/neural-language-v1/atom_neural_language_kaggle.py --self-test
```

## Atom-composed homeostatic governor

`atom_homeostatic_experiment.py` turns chaos from a fixed training schedule into
a deterministic feedback-controlled quantity. The learner sees one identical
232-event stream through either the adaptive governor or a fixed cooling
schedule. Runtime rows contain only an opaque cue, opaque effect, event ID, and
salience; regime names and expected effects remain evaluator-only data.

The governor observes surprise, conflict coherence, uphill acceptance, order,
free binding mass, nucleation rate, churn, and energy. It independently adjusts
temperature, phase strength, and the nucleation threshold. The controller is
itself composed from the seven universe primitives: radiation supplies bounded
reheating and phase pulses, gravitation aggregates field observables,
attraction/repulsion forms target-band errors, dissipation cools and damps,
nucleation commits laws and sustained control changes, conservation enforces
the evidence and chaos budgets, and decay removes stale windows and raw
evidence.

On the deterministic stream, the adaptive field learned all four replacement
laws while the fixed schedule retained the four obsolete laws: 4/4 versus 0/4
on the final evaluation. The adaptive field rejected all 40 incoherent-noise
events without replacing a law and scored 32/32 during final consolidation.
All 27 declared experiment gates passed, including seven causal primitive
ablations, seven strict corruption rejections, exact replay, graph-RAG use on
all four serialized workflow turns, and removal of all raw events and raw
evidence.

Across 29 controller windows, temperature ranged from `0.04` to
`0.623492938975`, phase strength from `0.01` to `0.166711871754`, and the
nucleation threshold from `0.56` to `0.91`. The maximum chaos load was exactly
the conserved `1.25` budget. The field accepted 24 and rejected 79 of 103
uphill proposals, an acceptance ratio of `0.23300970873786409`. Its serialized
model hash is
`3da98ac9339584b15a09ab7725b117e104de728c703a44d0c68e7dfdbe664a58`.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-homeostatic-criticality-governor-v1>.
Kaggle reproduced the 27/27 gates, 4/4 adaptive result, 0/4 fixed result, 40/40
noise retention, 32/32 consolidation, 4/4 serialized workflow, and the same
model, workflow response, and side-view bytes. The model-bound side view was
also inspected in the in-app browser at desktop and 390-by-844 layouts: all 27
rendered gates passed with no viewport overflow, interactive controls, browser
warnings, or browser errors.

The measured boundary is one seeded, deterministic, four-law synthetic stream
with hand-designed target bands. These results do not establish an edge-of-
chaos regime, self-organized criticality, Lyapunov stability, global
optimality, open-ended learning, or transfer to natural language. They isolate
the narrower claim that atom-composed feedback can distinguish incoherent
disturbance from a sustained coherent law change better than the matched fixed
schedule in this stream.

```powershell
py -3 atom_homeostatic_experiment.py --self-test
py -3 atom_homeostatic_experiment.py --output-dir homeostatic_outputs
py -3 atom_homeostatic_experiment.py --model homeostatic_outputs\atom_homeostatic_model.json --request homeostatic_outputs\atom_homeostatic_workflow_request.json --response homeostatic_outputs\atom_homeostatic_workflow_response_replay.json
py -3 scripts/build_kaggle_homeostatic_bundle.py --output-dir kaggle\homeostatic-v1
py -3 -m unittest tests.test_atom_homeostatic_governor_integration -v
```

## Emergent ontology discovery

`atom_ontology_experiment.py` removes the last supplied world-schema names from
the transition experiment. Runtime observations contain only opaque entity IDs,
opaque relation aliases, utterances, and before/after relation pairs. They do
not contain `locations`, `holders`, agent/object/location labels, participant
roles, transition names, or evaluator meanings. The relation aliases are also
disjoint across splits: training uses `q7x`/`v2m`, validation uses `z4u`/`a9k`,
and held-out evaluation uses `b3x`/`y8o`.

The learner induces three entity-type atoms from key/value participation and
nullability, then induces two relation atoms from those structural types. That
alias-independent ontology becomes the substrate for grounding 15 opaque
entity lexemes and nucleating five simultaneous effect laws. The seven universe
primitives remain the only mutation boundary; learn, remember, retrieve,
forget, and abstract are graph-resolved cognitive compositions. Phase mixing
and thermal annealing perturb the learning trajectory within conserved bounds.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-emergent-ontology-discovery-v1>.
Version 3 passed all 31 declared gates. It scored 15/15 validation and 25/25
held-out cases independently for world execution, stable law identity, and
surface generation; the exchange and release families scored 10/10. Exact
surface memory, relation-alias memory, and a fixed named-schema baseline each
scored 0/25. The three split-specific schemas produced the same structural
ontology signature,
`fba0cd90d8c9c69e37bbf97a73c2619a7d2bca0ec269f89939ef9de0e180eaa8`.

The five-turn serialized workflow bound held-out aliases at runtime and scored
5/5 while exercising both the wiki graph and graph RAG path. Five corrupt model
variants were rejected, every one-primitive ablation produced a causal signal,
raw episode/evidence counts ended at zero, and deterministic replay reproduced
model hash
`25e75e7a340c405dea73d17f6c6ddb1abe0acf7029b1afa626316bf2032d6164`.
The downloaded Kaggle source matched the generated bundle after normalizing
Kaggle's LF-to-CRLF conversion; the generated bundle's SHA-256 is
`593368dac323c4221fe788099fdaf911901f5c531f77e1c738d61279dd46d329`.
Its model-bound side view was inspected at desktop and 390-by-844 layouts with
no page or table overflow, clipped elements, interactive controls, or browser
console errors.

This experiment isolates alias-invariant schema transfer, not unrestricted
ontology induction. Each split still has exactly two binary relations over the
same 15 entity IDs: one total relation and one nullable relation. Variable
relation counts, arity, topology, and previously unseen entities remain outside
this experiment's measured boundary.

```powershell
py -3 atom_ontology_experiment.py --self-test
py -3 atom_ontology_experiment.py --output-dir ontology_outputs
py -3 atom_ontology_experiment.py --model atom_ontology_model.json --request atom_ontology_workflow_request.json --response response.json
py -3 scripts/build_kaggle_ontology_bundle.py --output-dir kaggle/ontology-v1
py -3 -m unittest tests.test_atom_ontology_discovery_integration -v
```

## Emergent transition-law discovery

`atom_transition_experiment.py` removes the fixed semantic predicate inventory
from the learning runtime. The field receives only opaque utterances, a world
before each utterance, and the world after it. Human names for the five tested
transitions remain in evaluator-only maps. The learner first grounds 15 entity
surfaces from repeated causal participation, then canonicalizes world deltas
into typed slots and executable effect atoms. Repeated structures nucleate as
stable opaque identities such as `law-ac8f88d55633fb09`; the runtime never
receives labels such as MOVE, TAKE, GIVE, exchange, or release.

The deterministic micro-world contains 50 training, 15 validation, and 25
held-out episodes. Every held-out utterance is surface-novel. Three transition
structures overlap the capabilities of the earlier fixed delta parser, while
two require genuinely new effect programs: simultaneous exchange of two
agents' locations and release of a held object to null. The exchange law uses
two simultaneous old-cell reads rather than a named operation.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-emergent-transition-law-discovery-v1>.
Version 2 passed all 23 declared gates. It crystallized all 15 entity lexemes
and five executable laws, then scored 15/15 validation and 25/25 held-out cases
for world execution, stable law identity, and surface generation. The two new
transition families scored 10/10. Exact-surface memory covered 0/25 held-out
utterances; the fixed three-predicate delta baseline recognized 15/25 overall
and 0/10 new-family cases.

The serialized five-turn workflow applies relocation, location exchange,
acquisition, transfer, and release against one evolving world and scored 5/5.
Its graph-native retrieval path runs on every turn. Four corrupt-model cases
failed closed, deterministic replay reproduced the model and history hashes,
and all seven single-primitive ablations produced their declared causal
signal. Phase mixing accumulated `5.830341` energy and thermal annealing
accepted 19 bounded non-improving moves. Cognitive forgetting reduced raw
episodes and raw evidence to zero while preserving all held-out behavior. The
model-bound side view was checked at desktop and 390-by-844 layouts without
horizontal overflow or clipped elements. The model hash is
`51e88f27d08157e4c369cb1b280de6ddb7bcced3eef1a8b35c46d408d1083c98`.

This experiment is confined to a typed `locations`/`holders` world. It learns
effect programs from consequences, but entity type markers and the two world
collections are still supplied. Its effect language currently covers direct
slot assignment, null assignment, typed preconditions, and copying an old
cell. It does not establish open-ended ontology formation, unrestricted
language, or general reasoning.

```powershell
py -3 atom_transition_experiment.py --self-test
py -3 atom_transition_experiment.py --output-dir transition_outputs
py -3 scripts/build_kaggle_transition_bundle.py
py -3 -m unittest tests.test_atom_transition_discovery_integration -v
```

## Opaque compositional grammar induction

`atom_language_opaque_experiment.py` removes the inherited English surface
entirely. A fresh universe-core field receives 48 grounded episodes containing
12 unseen concepts and 12 opaque grammar tokens. Their combined 24-token
surface vocabulary has zero overlap with the earlier language. Role order is
also changed: destinations precede MOVE agents, TAKE patients precede agents,
and GIVE begins with the recipient. Truth responses use the initially
meaningless forms `aya` and `nox` instead of English answer words.

The answer learner does not receive a translation table. It derives the answer
meaning from the question and observable world consequence, then crystallizes
the supplied surface form as a reusable answer law. Gold meanings and family
labels remain outside all 48 training observations. The evaluation uses 20
validation and 52 held-out episodes, including action combinations absent from
training, opaque assertions and questions, true and false world queries,
three-role GIVE compositions, and bidirectional generation.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-opaque-compositional-grammar-v1>.
Version 1 passed all 26 declared gates. Both independent fields scored 20/20
validation and 52/52 held-out grounded cases, plus 52/52 semantic generation
round trips. The frozen English field scored zero. Each field crystallized all
12 lexical laws and all 24 required parse/speak laws without assigning a
grammar token to a world concept. Both generated `aya` for true and `nox` for
false, and the serialized opaque workflow scored 8/8. Character-span F1 was
`0.959184`.

Abstraction removed all raw episodes and evidence while retaining the measured
behavior. Deterministic replay reproduced the word model and history hashes;
all seven single-primitive ablations produced their declared causal signal.
Phase mixing accumulated `8.633872` word and `8.937923` character phase energy,
while annealing accepted 59 and 58 bounded non-improving moves. The word and
character model hashes are
`91be96c78ac1c42c529d3e6f2eecc1fd8a77309cf71a75bfc2d0b0dcdcbc5ab6` and
`48c6156a0f9062c128d8dbeb9d581aef2076f3d54dc3add705386f20a02c6a87`.

This establishes grounded induction of an opaque compositional surface in a
small executable world. The semantic predicate inventory is still supplied by
the runtime, every tested grammar pattern appears somewhere in training, and
the result does not establish free-text language understanding or discovery of
entirely new predicates.

```powershell
py -3 atom_language_opaque_experiment.py --self-test
py -3 atom_language_opaque_experiment.py --output-dir opaque_outputs
py -3 scripts/build_kaggle_opaque_bundle.py
py -3 -m unittest tests.test_atom_language_opaque_integration -v
```

## Disjoint-lexicon compositional transfer

`atom_language_transfer_experiment.py` tests whether the retained laws from the
grounded language field can acquire a new miniature language without replaying
the original episodes. The base word and character models are abstracted to
lexical, frame, reference, and character-span laws, serialized, restored, and
then exposed to 12 grounded action demonstrations using nine previously unseen
concepts and nine previously unseen surface forms. Two transient disturbances
test correction and forgetting: a low-salience false `lumi` binding and the
one-off noise form `florp`.

The 48 evaluator-only cases contain no demonstration utterances. They test
unseen MOVE and TAKE pairings, GIVE triples absent from adaptation, assertions,
questions, true and false world queries, short-lived `they` and `it` context,
and bidirectional generation. The base and transfer concept identifiers are
disjoint, held-out action surfaces have zero overlap with adaptation, and gold
meanings remain outside the grounding rows.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-language-disjoint-transfer-v1>.
Version 4 passed all 25 declared experiment gates. Both stages scored 48/48
grounded held-out cases and 48/48 semantic generation round trips, while the
frozen base and exact-demonstration baselines scored zero transfer coverage.
Both adapted fields learned all 9/9 new bindings, corrected `lumi`, rejected
`florp`, retained 48/48 original-language behavior, and preserved every base
lexical and frame law. Character-span F1 was `0.972376`.

Abstraction removed all raw episodes and evidence without changing transfer
accuracy. The seven-turn serialized workflow scored 7/7, deterministic replay
reproduced the word model and adaptation-history hashes, and all seven universe
primitives ran during adaptation. Phase mixing accumulated `2.909432` word and
`3.021227` character phase energy; thermal annealing accepted 39 and 45 bounded
non-improving moves. The final word and character model hashes are
`dad3406477e6b55978a79a9ad5d9dc40d6650178ddeb82f16bdfea5078a1d4a7` and
`020b5dc1e07b4c3189f18c5a309bbb7111a3d721cc640ce14162c6000e681394`.

This establishes disjoint vocabulary acquisition and compositional reuse in a
small executable world. It does not establish open-domain language, learning
from ungrounded text, semantic discovery without observable consequences, or
scaling to unrestricted conversation.

```powershell
py -3 atom_language_transfer_experiment.py --self-test
py -3 atom_language_transfer_experiment.py --output-dir transfer_outputs
py -3 scripts/build_kaggle_transfer_bundle.py
py -3 -m unittest tests.test_atom_language_transfer_integration -v
```

## Grounded language-field experiment

`atom_language_experiment.py` is a from-scratch language substrate whose only
state-transition authority is the seven-primitive `UniverseLanguageKernel`.
It contains no pretrained model, neural-network framework, gradient path,
backpropagation, or trainable weight matrix. Radiation exposes word or
character pulses; gravitation accumulates grounded evidence; attraction and
repulsion bind surface forms to world roles; nucleation forms reusable lexical,
frame, and reference laws; conservation bounds their semantic mass;
dissipation cools the field; and decay removes episode evidence after its
reusable laws have formed.

The deterministic program contains exactly 192 grounded episodes: 120 train,
24 validation, and 48 held out. Training receives utterances, optional context
and grounded paraphrases, before/after worlds, answer consequences, and
salience. Gold meanings and family labels remain in a separate evaluator map.
Word-pulse and fresh character-pulse fields train independently. The language
surface covers commands, assertions, questions, bidirectional generation,
world mutation, answers, and short-lived `they`/`it` reference context over a
latent world of four agents, three objects, and three locations.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-emergent-language-field-v1>.
Kaggle version 6 scored 48/48 held-out grounded episodes for both independent
stages, 48/48 semantic generation round trips, all six context-reference cases,
and all seven turns in the serialized stateful workflow. Character-span F1 was
`0.975881`. After abstraction, both models retained 10 lexical laws, 50 frame
laws, and 2 reference laws while retaining zero raw episodes or raw evidence.
All seven single-primitive ablations produced their declared causal effect.
Phase mixing changed the learned trajectory, thermal annealing cooled from
`1.4` to `0.2` while accepting 53 bounded non-improving moves, and a repeated
training run produced the same model and history hashes.

The resulting character model hash was
`76b79e23bdff718c45453a128a784b1d8c8825c0648fc1ad90a75b54ad0056f9`.
The strict JSON loader validates schema, finite values, model hash, runtime and
knowledge markers, semantic roles, evidence removal, and unknown fields. The
HTML side view is bound to the actual character-model hash and shows a held-out
interaction, semantic-role field, learned lexemes, controlled-chaos state, and
all experiment gates. This result is intentionally limited to the generated
ten-concept micro-world; it is not evidence of unrestricted natural-language
competence or general intelligence.

```powershell
py -3 atom_language_experiment.py --self-test
py -3 atom_language_experiment.py --output-dir language_outputs
py -3 atom_language_experiment.py --model atom_language_character_model.json --request atom_language_workflow_request.json --response response.json
py -3 scripts/build_kaggle_language_bundle.py
```

## Emergent phase-law experiment

`atom_phase_law_experiment.py` is the current from-scratch substrate test. It
does not begin with a language model, neural-network layer, gradient path, or
trainable weight matrix. Eight opaque symbols occupy a cyclic phase lattice,
and four opaque operators are inferred as reusable phase transformations. The
seven universe primitives are the only code paths allowed to replace state:
radiation, dissipation, gravitation, attraction/repulsion, nucleation,
conservation, and decay.

The higher cognitive operations are compositions of that core. Phase mixing
creates bounded interference during retrieval, thermal annealing admits both
improving and controlled non-improving lattice moves while cooling, and the
learn/remember/forget/retrieve/abstract operations control evidence formation,
law crystallization, trace loss, and inference. A runtime wiki graph and graph
retriever resolve each composition back to the seven primitive leaves.

The private Kaggle kernel is
<https://www.kaggle.com/code/jessealicea/atom-emergent-phase-law-v3>.
Kaggle version 4 formed four persistent operator laws from 21 training traces,
reduced lattice energy from `0.333333` to `0`, and removed all 21 raw traces
after abstraction. The resulting four laws answered 7/7 held-out single-step,
128/128 unseen two-step, and 512/512 unseen three-step cases. All seven
single-primitive ablations changed behavior or broke the relevant invariant.
The serialized 39-query response downloaded from Kaggle replays byte-for-byte
on Windows with model hash
`c73dc62ae74865468271a05c3522fc856a8711da98bfb83f96c3c2a3753afebd`.

The generated HTML side view binds that model hash to the learned eight-node
phase ring, crystallized laws, causal gates, and controlled-chaos measurements.
This experiment demonstrates compact law recovery and composition in one tiny
synthetic cyclic world. It does not demonstrate language ability, open-world
reasoning, consciousness, or general intelligence.

```powershell
py -3 atom_phase_law_experiment.py --self-test
py -3 atom_phase_law_experiment.py --output-dir phase_law_outputs
py -3 atom_phase_law_experiment.py --model atom_phase_law_model.json --request request.json --response response.json
py -3 scripts/build_kaggle_phase_bundle.py --output-dir kaggle/phase-law-v3
```

## Universe-core composition experiment

`atom_universe_composition.py` makes the seven universe primitives the sole
state-transition authority. The substrate and its traces are immutable. Every
state replacement is structurally confined to `UniverseKernel`; cognitive
atoms receive no direct write path.

The seven core operations are radiation, dissipation, gravitation,
attraction/repulsion, nucleation, conservation, and decay. Higher atoms are
declarative compositions:

- phase mixing = radiation + gravitation with bounded seeded interference;
- thermal annealing = dissipation + conservation with monotonic cooling;
- attention = phase mixing + attraction/repulsion;
- forget = thermal annealing + decay;
- learn = attention + nucleation + conservation;
- remember = forget + learn;
- retrieve = attention;
- revise = remember under contradictory evidence;
- abstract = remember across repeated related evidence.

The experiment performs a structural mutation-boundary audit and disables each
core primitive independently. Every ablation must produce a measurable loss of
behavior or violation of the conservation invariant.

The private Kaggle run is at
<https://www.kaggle.com/code/jessealicea/atom-universe-core-composition-v2>.
It processed 38 experiences into 10 retained structures, removed two
unsupported structures, and answered all 31 fixed-program queries. Every core
primitive was exercised and every single-primitive ablation produced its
declared causal signal.

Version 2 adds deterministic phase mixing at strength `0.035` and thermal
annealing from `1.35` toward a `0.72` floor. The Kaggle run ended at `0.7397`;
its largest single-step phase energy was `0.00708`. A four-level phase-strength
sweep produced four distinct transition trajectories while retaining the same
31 correct responses.

```powershell
python atom_universe_composition.py --self-test
python atom_universe_composition.py --output-dir universe_core_outputs
python atom_universe_composition.py --run-request request.json --response response.json
```

## Cognitive memory experiment

`atom_cognitive_experiment.py` tests whether cognitive and universe atoms can
produce useful learning behavior without a conventional neural network,
gradient descent, or trainable weights. Experiences reshape a bounded field of
causal traces through attention, association, reinforcement, nucleation,
conservation, contradiction, dissipation, and decay.

The deterministic tiny program exercises:

- association learning from repeated experience;
- context-dependent memory;
- reconstruction from partial cues;
- revision after contradictory evidence;
- retention of reinforced traces;
- disappearance of unsupported traces;
- compression of repeated events into shared field structures;
- state serialization and a strict JSON request/response path.

The experiment compares the Atom field with exact-address storage and a raw
nearest-memory baseline. Kaggle is the intended execution surface.

The private Kaggle v2 run is at
<https://www.kaggle.com/code/jessealicea/atom-cognitive-memory-experiment-v2>.
It retained 10 supported associations from 38 experiences, removed 2 one-off
traces, and answered all 31 full, partial, contextual, correction, and
noise-rejection queries in the fixed tiny program. The exact-address and raw
nearest-memory comparison scores were 0.6800 and 0.7067; the Atom system scored
1.0000 under the same task metric.

```powershell
python atom_cognitive_experiment.py --self-test
python atom_cognitive_experiment.py --output-dir cognitive_outputs
python atom_cognitive_experiment.py --run-request request.json --response response.json
```

## Field dynamics experiment

This experiment starts without a pretrained language model or LoRA. It asks a smaller, cleaner question: can a purpose-built neural field learn a universe-first state-transition calculus from a tiny executable dataset?

The architecture updates a graph synchronously over repeated field ticks. Seven operator branches are structurally different:

1. radiation propagates state along directed paths;
2. dissipation attenuates unsupported state;
3. gravitation aggregates state toward learned attractors;
4. attraction/repulsion uses signed relations to bind or separate;
5. nucleation gates structure formation at a threshold;
6. conservation projects closed-system mass back onto its invariant budget;
7. decay retires unsupported, expired state.

The branches act independently and simultaneously; they do not compete for a unit attention budget. Sixteen learned values calibrate their rates, thresholds, and field mixtures. They start deliberately miscalibrated, and the proof gate requires the tiny dataset to improve validation behavior by at least 10%. A normalized route is exposed only for auxiliary supervision and observability. The process names never appear in model inputs; they are evaluation metadata.

## Tiny proof

The script deterministically generates:

- 140 training cases;
- 40 validation cases;
- 36 held-out cases made from composition families absent from training.

Gold states come from an executable simulator. A matched flat MLP is trained on the same examples. Evaluation includes continuous-state error, active/structure accuracy, closed-system conservation error, held-out composition performance, routing behavior, and seven operator ablations.

## Commands

Static proof checks only:

```powershell
python atom_field_proof.py --self-test
```

Full training run (the intended execution surface is the private Kaggle kernel):

```powershell
python atom_field_proof.py --output-dir outputs
```

Kaggle writes the report, generated split files, model weights, manifest, and a fresh serialized request/response run to `/kaggle/working`.

Run the saved model against another validated JSON request:

```powershell
python atom_field_proof.py --infer-request request.json --weights atom_field_state.pt --response response.json
```

The request path rejects unknown/missing keys, wrong tensor shapes, non-finite values, out-of-range controls, negative mass, and adjacency self-loops before inference.

This is an architectural experiment. A positive result would show learnable observable state dynamics on a tiny synthetic world; it would not establish language competence or prove a claim about private reasoning.

## Open mathematical Primitive Forge

`atom_primitive_forge.py` replaces a closed higher-primitive vocabulary with
an open, recursively compositional graph. The seven universe operators remain
immutable roots; discovered primitives retain typed signatures, recipes,
complete root expansions, invariants, symmetries, boundary and scale metadata,
evidence, counterexamples, confidence, persistence, lifecycle state, and
provenance.

`atom_primitive_experiment.py` generates compositions algorithmically, tests
them in a bounded scalar-field simulation, reuses crystallized discoveries at
the next depth, and measures unseen and counterfactual transfer against a flat
root-expansion evaluator. Normal use observations enter quarantine and require
repeated predictive support before crystallization. Contradictions revise
structures, unsupported structures decay, and roots cannot mutate.

The runtime produces a dynamic graph-native wiki/RAG artifact and a
user-visible side view bound to the exact primitive graph. Coding, Rust, and
frontend languages are later optional projections, not the discovery
ontology. See `ATOM_PRIMITIVE_FORGE.md` for the architecture, commands, live
observation envelope, and explicit claim boundary.

```powershell
py -3.13 atom_primitive_experiment.py --output-dir primitive_forge_outputs
py -3.13 -m unittest tests.test_atom_primitive_forge_integration -v
```
## Causal coding and mathematical platform synthesis

The coding experiment learns which mathematical primitives cause executable
platform behaviors. It does not select a memorized platform template. Training
uses isolated treated/control executions, persists the resulting causal laws,
and composes every supported cause needed by an unseen specification.

The nine platform primitives are identity, directed relation, composition,
conservation, ordering, feedback, bounded fixed point, topology, and
projection. They generate a four-layer Spiderweb platform with ground
transport, typed messages, parallel promotion, explicit off-ramps, preloading,
flow-created threads and intersections, backpressure vibrations, bounded
recovery, and discrete output.

Run the full experiment:

```powershell
py -3.13 atom_coding_experiment.py --output-dir coding_harness_outputs
```

Run its integration suite:

```powershell
py -3.13 -m unittest tests.test_atom_coding_harness_integration -v
```

The output directory contains the persistent causal model, benchmark report,
live workflow request and response, generated executable platform, and a
right-side HTML view bound to that exact artifact.

The private Kaggle GPU bundle is generated with:

```powershell
py -3.13 scripts/build_kaggle_coding_bundle.py
```

The current remote run and local evidence are recorded in
`KAGGLE_CODING_PLATFORM_V2_EVIDENCE.md`.

## Causal Atom Memory

`atom_causal_memory_rust` is an isolated local fork of the Lucerna Labs Atom DB
substrate. The upstream implementation remains the lexical evidence/RAG
system. The new `atom-causal-memory` Rust crate stores the Primitive Forge
inventory as typed causal glyphs and exact `(glyph, role, value)` motif atoms.

Retrieval is structural. Queries use composition topology, root lineage, type
and dimension signatures, invariants, symmetries, boundaries, and scales.
Aliases, passages, provenance prose, embedding vectors, and cosine similarity
cannot enter the causal-resonance path. Results expose the exact motif
identities and conductance values supporting every match and return an explicit
insufficient-evidence state when required topology has not been observed.

Prediction observations provide durable metaplasticity. Correct selections
strengthen the matched motifs. Wrong selections weaken the misleading motif
paths and strengthen the expected structure in one atomic cell. Queries remain
read-only, feedback survives process restart, and importing an unchanged Forge
graph is idempotent.

```powershell
py -3.13 atom_causal_memory_experiment.py --output-dir causal_memory_outputs
py -3.13 -m unittest tests.test_atom_causal_memory_integration -v
```

See `ATOM_CAUSAL_MEMORY.md` and
`atom-causal-memory-architecture.json` for the storage schema, retrieval law,
learning event, fork boundary, and measurement scope.

## Persistent Causal Experience

The causal Atom DB store now retains the full saved causal-world history, not
only the Primitive Forge catalog. It preserves 2,304 immutable observation
revisions and 395 consolidated laws as 2,699 structural experience records.
Repeated observation labels remain separate revisions. Laws retain
graph-native links to their supporting evidence.

Queries resonate over cause, effect, context, domain, direction, measured
bands, and ordered seven-root composition. Required structure is fail-closed:
an unobserved causal topology returns no hits and explicitly reports
insufficient evidence. Outcome feedback strengthens the expected motif path,
weakens a misleading selected path, and persists across process restart.
Recall itself remains read-only.

The runtime wiki graph and structural RAG are built from the exact durable
catalog. The right-side HTML view validates the report, inventory, workflow,
knowledge graph, and store bindings before rendering.

```powershell
py -3.13 atom_causal_experience_experiment.py --output-dir causal_experience_outputs
py -3.13 -m unittest tests.test_atom_causal_experience_integration -v
```

See `ATOM_CAUSAL_EXPERIENCE.md` and
`atom-causal-experience-architecture.json` for the detailed contracts and
measurement boundary.

## Trusted Live Causal Learning

The next runtime layer learns from interaction without treating its own output
as truth. It recalls a prediction from the situation alone, receives an
independently certified outcome, appends that observation to Atom DB, and
updates the exact structural paths responsible for the prediction.

Every outcome carries a canonical SHA-256 idempotency key enforced by the Rust
store. Exact replay performs no additional mutation. Reusing a key for a
different decision fails closed. Live observations retain session, authority,
source-evidence, and prediction provenance; those audit fields remain barred
from structural recall. Outcome admission also requires an allowlisted
authority identity and a separate canonical evidence receipt whose SHA-256 and
event fields match.

The full workflow first predicts a novel effect incorrectly, learns from the
trusted result, proves that replay changes nothing, and then predicts the new
effect correctly in a second session after reopening the store.

```powershell
py -3.13 atom_causal_live_experiment.py --output-dir causal_live_outputs
py -3.13 -m unittest tests.test_atom_causal_live_integration -v
```

See `ATOM_CAUSAL_LIVE.md` and
`atom-causal-live-architecture.json` for the event schema, trust boundary,
atomic replay law, and runtime graph bindings.
