# Atom Universal Knowledge

## Purpose

Phase 7 gives Atom Harness a multidisciplinary knowledge foundation without
turning the language model into the source of truth. The runtime still uses the
official Qwen 4B model only for language rendering. Atom owns routing,
retrieval, graph formation, citations, claim identity, epistemic status,
abstention, permissions, and committed artifacts.

The first pack is intentionally a foundation, not a claim that the repository
contains all human knowledge. It provides a verified seed across 15 broad
disciplines, proves the full ingestion and retrieval boundary, and defines the
format through which deeper packs can be admitted later.

The active pack is:

```text
knowledge_packs/universal-foundation-v1/manifest.json
```

Its current contents are:

- 15 declared domains
- 45 Atom-authored claim records
- 22 source records
- 454 wiki graph nodes
- 650 wiki graph edges

## Knowledge lanes

Atom keeps two distinct knowledge lanes.

### Causal experience lane

The existing causal lane contains verified experimental records in the Rust
Atom database. It answers questions for which a saved relationship and bounded
experience packet exist. This lane retains the V3 causal response contract,
causal graph, database snapshot, contradiction policy, and evidence side view.

### Multidisciplinary reference lane

The new multidisciplinary lane contains sourced reference claims. It routes a
question to one or more declared domains, creates a query-specific Spiderweb
thread across the relevant graph ground lanes, retrieves bounded claim records,
and gives the language model only that packet. The response must reproduce the
primary claim identity, claim type, epistemic status, and statement hash.

The two lanes are not merged into a single confidence score. A scientific
model, formal definition, health limitation, fictional fact, literary context,
and writing heuristic remain different kinds of records even when they share
concepts.

## Domain foundation

The initial taxonomy declares these domains:

1. Formal mathematics
2. Computer science
3. Physics and quantum science
4. Astronomy and space science
5. Chemistry and materials science
6. Earth and environmental science
7. Biology
8. Medicine and health science
9. Engineering
10. Agriculture and veterinary science
11. Social and behavioral science
12. Linguistics
13. Literature
14. Writing and creative writing
15. Research practice

Every declared domain has at least one routable and citable claim. The Phase 7
integration test queries every domain through the real router and graph RAG
path. A domain may not be declared with an empty claim set.

## Pack layout

The pack is content addressed and read only during a session.

```text
knowledge_packs/universal-foundation-v1/
  manifest.json
  taxonomy.json
  sources.json
  claims/
    formal-physical.jsonl
    earth-life-health.jsonl
    engineering-social-linguistics.jsonl
    literature-writing.jsonl
    research.jsonl
```

`manifest.json` declares the pack identity, version, files, rights policy,
security policy, and SHA-256 of every data file. The runtime verifies the
manifest schema, rejects links and path escapes, verifies all data hashes, and
then retains the manifest SHA-256 for the life of the session. A change to the
manifest or any referenced file closes the lane rather than silently reloading
different knowledge.

`taxonomy.json` declares domains, aliases, subdomains, routing concepts, and
relationships to other domains.

`sources.json` records source identity, title, publisher, canonical URL,
license or usage terms, rights lane, acquisition mode, source class, access
date, and limitations. It does not bundle source text from citation-only
sources.

Each JSONL shard contains one strict claim object per line. Claim records have
bounded text, explicit source IDs, domain and subdomain, claim type, epistemic
status, concepts, keywords, limitations, and a fictional flag. Unknown fields
and invalid enum values are rejected.

## Claim and epistemic types

The claim type explains what kind of statement Atom is handling. Current
examples include:

- `definition`
- `formal-result`
- `scientific-model`
- `scientific-law`
- `research-method`
- `literary-context`
- `interpretation`
- `craft-principle`

The epistemic status explains how the statement should be read. A formal
result is not presented as a writing suggestion, and a writing suggestion is
not presented as an experimentally established law. Fictional records must
remain in the literary lane and carry `fictional: true`. Craft guidance is
marked as a heuristic, not as universal fact.

Every claim receives a canonical content hash. Every response grounding object
must bind:

- `source_claim_id`
- `domain_id`
- `claim_type`
- `epistemic_status`
- `statement_sha256`

The validator rejects a response that cites an ID absent from its packet,
changes any grounding field, or invents support. If the packet is not
answerable, Atom emits a deterministic abstention.

The constrained answer is bounded to 1,024 characters. This matches the
certified causal response contract and the grammar limit verified against the
packaged llama.cpp runtime. Longer exposition must be decomposed into another
grounded request rather than weakening structured generation.

## Source and rights policy

The pack separates the right to cite a source from the right to redistribute
its text.

- Green sources are rights-clear or public-domain-compatible after the exact
  object and attribution requirements are verified.
- Amber sources carry collection-specific or share-alike obligations and must
  stay isolated so those obligations are not lost.
- Yellow sources are citation only. Atom stores metadata, a URL, an
  Atom-authored factual summary, and limitations. It does not bundle the source
  text.
- Unknown or all-rights-reserved source text is not admitted without explicit
  permission.

The initial pack uses only Atom-authored claim statements. Source records give
the operator enough metadata to inspect the underlying authority. A generated
summary is never treated as independent evidence for itself.

Before adding a source, verify the exact page or object rather than relying on
the general reputation of its publisher. Record collection-specific terms,
jurisdictional public-domain caveats, attribution requirements, and access
date. If the rights status is uncertain, use citation-only metadata or do not
admit the source.

## Spiderweb retrieval flow

The multidisciplinary lane follows the project Spiderweb doctrine.

1. Ground lanes are preloaded for domains, claims, sources, and relationships.
2. The deterministic router observes the question and scores declared domains.
   It admits the reference lane only at the declared minimum score of 8, which
   prevents a single generic token from hijacking a causal request.
3. A temporary thread forms only for the domains and concepts activated by
   that flow.
4. Typed intersections connect related claims and source provenance where the
   query creates useful overlap.
5. The on-ramp admits a `BoundedKnowledgeQuery`.
6. Graph-first retrieval selects a bounded packet and preloads adjacent domain
   manifests.
7. The off-ramp emits `BoundedKnowledgeEvidence`.
8. The language provider receives the bounded packet and strict response
   schema.
9. Atom validates the output, commits the transaction, and renders the real
   artifact in the side view.

The graph is not a hidden prompt index. The committed transaction contains the
graph identity, selected domains, claim IDs, source IDs, intersections,
preload evidence, and the exact passage packet used for the answer.

## Prompt-injection boundary

Source metadata, retrieved claims, operator questions, model output, and tool
results are untrusted data. None can:

- grant tool permission
- invoke a capability
- write causal memory
- change a claim or source record
- promote generated text into evidence
- override an epistemic status
- disable abstention
- mutate the active pack

The language model has no object reference through which it could perform one
of those actions. Permissioned hands remain a sibling Phase 6 lane. A model may
propose a registered capability, but the exact prepared manifest still waits
for a human, one-time approval.

## Runtime artifacts

An evidence transaction now snapshots both knowledge systems:

- the causal Atom database and causal wiki graph
- the multidisciplinary pack manifest and every hashed data file
- `atom_multidisciplinary_wiki_graph.json`
- the bounded evidence packet
- the language response
- the Spiderweb trace
- the user-visible side view

The side view displays the selected knowledge lane. For a multidisciplinary
answer it shows the primary claim, claim type, epistemic status, fictional
marker where applicable, domain, source links, rights metadata, limitations,
and exact grounding object. It is bound to the real committed artifact rather
than reconstructed from chat text.

Tool transactions also snapshot the multidisciplinary identities and verify
that the pack did not change while an approved action executed. Tool output
still cannot add to either knowledge lane.

## Adding or extending a pack

Use this order for an additive update:

1. Define the intended domain and source scope.
2. Verify each source object, canonical URL, rights lane, and usage terms.
3. Write concise Atom-authored claim statements with explicit limitations.
4. Select the correct claim type and epistemic status.
5. Link every claim to one or more registered sources.
6. Add taxonomy aliases and cross-domain relations only where they improve
   deterministic routing.
7. Recompute the SHA-256 values in the pack manifest.
8. Load the pack through `load_multidisciplinary_knowledge`.
9. Query every new domain through the real router and retrieval path.
10. Inspect the rendered side view and transaction snapshots.
11. Rerun the Phase 7 integration, certification, desktop verification, and
    full regression suite.

Do not edit a released pack in place. Create a new versioned pack directory and
update the runtime declaration after review. This preserves old transaction
replay and makes a knowledge change visible as a new hash-bound identity.

## Verification

Run the mandatory Phase 7 integration:

```powershell
py -3.13 -m unittest discover -s tests `
  -p "test_atom_universal_knowledge_integration.py" -v
```

Run the source and declaration verifier:

```powershell
py -3.13 scripts/verify_atom_harness_v7.py --source-only
```

Create a local certificate, or promote the repository certificate after the
source is final:

```powershell
py -3.13 scripts/certify_atom_universal_knowledge.py
py -3.13 scripts/certify_atom_universal_knowledge.py --promote
```

The release build performs these checks again before creating the portable ZIP
and per-user MSI. The installed-layout verifier rehashes the complete
application and separately opens and verifies the packaged knowledge pack.

## Capability floor and current limits

Phase 7 is additive to the Phase 6 permissioned-hands experiment and preserves
the Ornith 1.0 capability floor. Coding, builds, simulations, documents,
workspace management, bounded web reads, permission review, causal evidence,
wiki graph, graph RAG, and both artifact side views remain available.

The initial foundation is broad but deliberately shallow. It proves trusted
acquisition, epistemic separation, deterministic routing, graph retrieval,
grounding, packaging, and user-visible provenance. Deeper disciplinary packs,
contradiction sets, temporal source refresh, and expert review records can be
added through the same versioned boundary without enlarging the model or giving
it knowledge authority.
