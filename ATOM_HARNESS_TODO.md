# Atom Harness Canonical TODO and Roadmap

This file is the canonical list of known unfinished work for Atom Harness. It
is deliberately more precise than a feature wish list. Every item says why it
exists, what completion means, and what evidence must be produced before the
checkbox can be closed.

The audit baseline for this list is source commit
`58e0cbb59d5fabdec3bff45e2e785860f9a806d4` on `main`, inspected on
2026-07-31 local time. At that baseline, the repository was clean, the Lucerna
Labs remote matched the local commit, the latest Phase 7 CI run passed, and the
private GitHub repository had no tags, releases, open issues, or open pull
requests.

This is the complete known backlog, not a claim that future experiments can
never reveal more work. When new work is discovered, add it here with an ID,
priority, dependency, acceptance criteria, and verification method before
implementation begins.

## 1. How to use this file

Priority meanings:

| Priority | Meaning |
| --- | --- |
| P0 | Required before an external release or a stronger safety claim |
| P1 | Required for the next experimental phase |
| P2 | Important robustness, maintainability, or operator experience work |
| P3 | Optional expansion that must not delay the core experiment |

Status meanings:

| Marker | Meaning |
| --- | --- |
| `[ ]` | Open |
| `[~]` | In progress, with evidence still incomplete |
| `[x]` | Complete and linked to fresh verification evidence |
| `[!]` | Externally blocked, with the blocker recorded |

Rules for closing an item:

1. Implement the behavior and its failure path.
2. Add or update machine-readable contracts when authority, data, runtime, or
   release behavior changes.
3. Add deterministic tests and an integration path through the real runtime.
4. Preserve the runtime wiki graph, graph RAG, real artifact side views,
   permission boundary, and Ornith 1.0 capability floor.
5. Update the developer guide, user guide when user behavior changes, focused
   subsystem record, and chronological developer notes.
6. Regenerate source-bound certificates instead of editing their hashes.
7. Rerun all relevant checks after the final edit.
8. Record the commit, commands, results, artifacts, and remaining claim
   boundary beside the closed item.

An item is not complete because a scaffold exists, a unit test passes, or the
happy path worked once. Completion requires the acceptance evidence named in
the item.

## 2. Confirmed foundation that should not be reopened as unfinished work

These are established Phase 7 capabilities. They remain regression obligations
but are not open tasks by themselves.

- [x] BASE-001: The active runtime is `language-harness-v6`, presented by Atom
  Harness Desktop Phase 7 version `7.0.0`.
- [x] BASE-002: The Qwen3 4B Instruct Q8_0 artifact is the certified resident
  local language membrane. Model escalation is measurement gated.
- [x] BASE-003: The causal wiki graph and graph RAG lane are runtime wired.
- [x] BASE-004: The multidisciplinary wiki graph and graph RAG lane are runtime
  wired through a content-addressed pack.
- [x] BASE-005: The knowledge foundation covers 15 declared domains with 45
  claim records and 22 source records. This proves the boundary but is not an
  encyclopedia.
- [x] BASE-006: Evidence and tool results render in real user-visible side
  views bound to committed artifacts.
- [x] BASE-007: Coding, builds, simulations, documents, workspace management,
  and bounded public web reads are exposed through registered capabilities.
- [x] BASE-008: Every tool execution requires explicit human approval of one
  exact, hashed, expiring, non-replayable manifest.
- [x] BASE-009: LLM output, source text, web content, process output, and tool
  results remain untrusted and cannot grant permission or write Atom memory.
- [x] BASE-010: Source, Python, .NET, Rust, frozen-backend, local-model,
  installation, package, process-shutdown, and artifact-side-view evidence has
  been produced for the certified local Phase 7 package.
- [x] BASE-011: A per-user MSI and portable Windows x64 ZIP have been built and
  hash recorded from one verified staged layout.
- [x] BASE-012: The source is preserved in the private
  `Lucerna-Labs/atom-harness` repository and the active Phase 7 workflow passes.

## 3. Recommended next milestone: Phase 8

Phase 8 should be the Knowledge Acquisition and Adversarial Evaluation phase.
Its purpose is to deepen the evidence fabric and measure the experiment against
outside influence at scale without giving the model authority. It must not be
presented as universal knowledge or universal prompt-injection resistance.

### 3.1 Phase 8 contract and migration

- [ ] P8-CONTRACT-001, P1: Write the Phase 8 experiment contract before adding
  data.
  - Define the question being tested, non-goals, threat actors, attack surfaces,
    measurable outcomes, failure conditions, and claim boundary.
  - Preserve explicit approval before every tool execution.
  - Acceptance: a machine-readable architecture contract, focused design
    record, and integration-test declaration agree on one runtime identity.

- [ ] P8-CONTRACT-002, P1: Define knowledge-pack schema evolution.
  - Add explicit compatibility rules for pack schema, pack version, taxonomy,
    claims, sources, reviewer records, supersession, and contradiction data.
  - Never silently reinterpret a Phase 7 pack under new rules.
  - Acceptance: old packs either open identically or fail with a precise
    migration error; migration fixtures prove both paths.

- [ ] P8-CONTRACT-003, P1: Add a pack-diff and migration report.
  - Show added, changed, superseded, retracted, and removed claims and sources.
  - Bind the report to old and new manifest hashes.
  - Acceptance: a user-visible artifact and machine-readable report reproduce
    the exact transition between two test packs.

- [ ] P8-CONTRACT-004, P1: Define admission roles and separation of duties.
  - Candidate acquisition, rights review, scientific or editorial review, pack
    approval, and runtime admission must be distinct recorded stages.
  - The LLM may assist with candidate extraction but may approve none of them.
  - Acceptance: no single untrusted result can advance a candidate from source
    discovery to admitted Atom knowledge.

### 3.2 Acquisition, provenance, rights, and freshness

- [ ] P8-DATA-001, P1: Build a candidate-source staging pipeline.
  - Accept exact source metadata and permitted content into a quarantine area.
  - Record origin URL or persistent ID, retrieval time, content hash, media
    type, byte count, acquisition method, and parser identity.
  - Acceptance: staged data cannot enter the active graph until every later
    admission gate passes.

- [ ] P8-DATA-002, P1: Add deterministic candidate extraction.
  - Convert permitted source material into bounded candidate claims with exact
    source spans or source-level citation evidence.
  - Treat extracted text as untrusted data and never as instructions.
  - Acceptance: the same source bytes and extractor version reproduce the same
    candidate set and hashes.

- [ ] P8-DATA-003, P1: Add an item-level rights-review queue.
  - Record rights lane, license, license URL, allowed acquisition mode,
    redistribution decision, reviewer, review time, and limitations.
  - Reject unknown rights by default. Citation-only sources must not smuggle
    source text into the pack.
  - Acceptance: tampered or missing rights records fail pack admission.

- [ ] P8-DATA-004, P1: Add reviewer and approval provenance.
  - Store review identity or pseudonymous reviewer ID, review role, decision,
    timestamp, reviewed hashes, and notes.
  - Approval records must be append-only and content addressed.
  - Acceptance: changing a reviewed claim or source invalidates the approval.

- [ ] P8-DATA-005, P1: Add temporal source semantics.
  - Distinguish publication date, effective date, retrieval date, review date,
    expiration or recheck date, and supersession date.
  - Acceptance: the router can abstain or warn when a time-sensitive claim is
    stale for the question being asked.

- [ ] P8-DATA-006, P1: Add correction, retraction, and supersession records.
  - Never erase the existence of a formerly admitted claim.
  - Route current answers to the active record while rendering the historical
    relationship in the side view.
  - Acceptance: tests cover correction, full retraction, replacement, and a
    question explicitly asking for historical state.

- [ ] P8-DATA-007, P1: Add contradiction and disagreement sets.
  - Represent conflicting empirical results, interpretive disagreements,
    competing models, and unresolved evidence without collapsing them into one
    confidence score.
  - Acceptance: the answer and side view name the disagreement and sources
    instead of choosing a winner without declared authority.

- [ ] P8-DATA-008, P1: Add a freshness review scheduler that creates review
  work but never updates knowledge automatically.
  - It may identify records whose review date has arrived.
  - It may not fetch, approve, rewrite, or publish without the relevant human
    permissions.
  - Acceptance: a due item creates a bounded review artifact and no knowledge
    mutation.

- [ ] P8-DATA-009, P1: Establish backup, restore, and disaster-recovery rules
  for admitted packs and review records.
  - Acceptance: a clean checkout can restore an admitted pack from its declared
    source artifacts and verify every content hash.

### 3.3 Knowledge depth and domain quality

- [ ] P8-KNOW-001, P1: Define depth targets for every one of the 15 domains.
  - Use declared subdomain and question-type coverage, not an arbitrary bulk
    claim count.
  - Acceptance: a coverage manifest exposes empty or weak subdomains and the
    certification gate rejects undeclared coverage gaps.

- [ ] P8-KNOW-002, P1: Deepen formal mathematics.
  - Represent definitions, assumptions, theorems, proof sketches, counterexamples,
    and dependency relationships separately.
  - Acceptance: formal statements cannot be rendered as empirical findings or
    writing heuristics.

- [ ] P8-KNOW-003, P1: Deepen computer and information science.
  - Bind version-sensitive API or language claims to exact versions and dates.
  - Separate executable examples from general claims and run examples only
    through permissioned hands.
  - Acceptance: outdated version claims are detectable and executable examples
    never run during retrieval.

- [ ] P8-KNOW-004, P1: Deepen physics, quantum science, astronomy, chemistry,
  earth science, biology, engineering, agriculture, and social science.
  - Preserve measurement conditions, scale, uncertainty, model scope, and
    source limitations.
  - Acceptance: evaluation cases cover definitions, mechanisms, quantitative
    limitations, competing models, and abstention in each domain.

- [ ] P8-KNOW-005, P1: Harden medicine and health content.
  - Add date, jurisdiction where relevant, evidence grade, population scope,
    contraindication or limitation fields, and an educational-use boundary.
  - Acceptance: clinical or personal treatment requests cannot be mistaken for
    certified medical authority.

- [ ] P8-KNOW-006, P1: Deepen linguistics across language families and analysis
  levels.
  - Cover phonetics, phonology, morphology, syntax, semantics, pragmatics,
    sociolinguistics, historical linguistics, writing systems, and language
    documentation without treating one language as universal.
  - Acceptance: Unicode, normalization, script-direction, and multilingual
    routing cases pass.

- [ ] P8-KNOW-007, P1: Deepen literature and creative-writing material without
  violating rights.
  - Keep public-domain text, licensed excerpts, citation-only scholarship,
    fictional facts, interpretations, and craft heuristics in explicit lanes.
  - Acceptance: copyrighted full text cannot enter through a citation-only
    record, and fictional claims cannot route as scientific facts.

- [ ] P8-KNOW-008, P1: Add source diversity and independence measurements.
  - Count distinct institutions, authors, publication classes, geographic
    origins where known, and citation dependencies.
  - Acceptance: multiple records derived from one upstream source are not
    misreported as independent corroboration.

- [ ] P8-KNOW-009, P1: Add graph-scale performance tests.
  - Measure pack open time, graph build time, query latency, memory, candidate
    count, path count, and side-view size as claims and edges grow.
  - Acceptance: declared budgets pass at the target Phase 8 scale and overflow
    fails closed with a bounded error.

### 3.4 Retrieval and answer evaluation

- [ ] P8-EVAL-001, P1: Create a versioned multidisciplinary benchmark.
  - Include answerable, partially answerable, unsupported, ambiguous,
    adversarial, time-sensitive, and cross-domain questions.
  - Keep development and held-out partitions distinct.
  - Acceptance: benchmark bytes, partitions, expected evidence, and scorer are
    content addressed.

- [ ] P8-EVAL-002, P1: Measure retrieval separately from language rendering.
  - Report routing, claim recall, source recall, citation closure, graph-path
    validity, primary-claim fidelity, abstention, and rendering quality.
  - Acceptance: a fluent answer cannot hide a retrieval or grounding failure.

- [ ] P8-EVAL-003, P1: Add multi-hop graph questions.
  - Require retrieval across related claims or domains while keeping every
    cited statement inside the packet.
  - Acceptance: tests reject unsupported bridge claims introduced by the LLM.

- [ ] P8-EVAL-004, P1: Add ambiguity and clarification evaluation.
  - Detect questions with multiple defensible interpretations before selecting
    evidence.
  - Acceptance: the runtime asks for clarification or renders bounded
    alternatives instead of silently guessing.

- [ ] P8-EVAL-005, P1: Add calibrated abstention evaluation.
  - Measure false answers, false abstentions, partial-evidence behavior, and
    abstention reason quality.
  - Acceptance: every unsupported answer path is a test failure even if prose
    sounds plausible.

- [ ] P8-EVAL-006, P1: Add temporal and contradiction evaluation.
  - Ask for current state, historical state, superseded guidance, and disputed
    topics.
  - Acceptance: answers expose date and disagreement boundaries from Atom
    records.

- [ ] P8-EVAL-007, P1: Add reproducible regression thresholds.
  - Define floors and maximum regressions before comparing implementations.
  - Acceptance: a release cannot pass by improving an average while violating
    a critical safety case.

### 3.5 Prompt-injection and outside-influence experiment

- [ ] P8-ADV-001, P1: Create a content-addressed adversarial corpus.
  - Cover direct prompts, retrieved source text, files, web pages, process
    output, tool output, document metadata, nested encodings, Unicode tricks,
    citation spoofing, authority spoofing, and cross-turn influence.
  - Acceptance: each case declares the forbidden authority transition and the
    expected observable result.

- [ ] P8-ADV-002, P1: Test attacks against knowledge retrieval.
  - Attempt to change routing, fabricate claims, alter citations, bypass
    abstention, collapse epistemic types, and promote untrusted text.
  - Acceptance: no attack can create Atom evidence or an unbound citation.

- [ ] P8-ADV-003, P1: Test attacks against permissioned hands.
  - Attempt hidden actions, argument substitution, manifest tampering,
    permission replay, approval spoofing, expiry bypass, multi-action smuggling,
    and follow-up execution from tool output.
  - Acceptance: every execution remains bound to the exact human-approved
    manifest and every rejected attempt leaves inspectable evidence.

- [ ] P8-ADV-004, P1: Test cross-lane attacks.
  - Attempt to move causal evidence into reference knowledge, fiction into
    science, model prose into memory, and tool output into either graph.
  - Acceptance: lane identity and authority remain intact through the complete
    runtime and side view.

- [ ] P8-ADV-005, P1: Add structured mutation and fuzz testing.
  - Mutate JSON, JSONL, HTTP envelopes, manifests, paths, hashes, permission
    receipts, source metadata, HTML artifacts, and Unicode text.
  - Acceptance: parsers reject malformed or oversized inputs without crashes,
    hangs, partial publication, or authority widening.

- [ ] P8-ADV-006, P1: Add multi-turn and long-context attack sequences.
  - Test delayed instructions, gradual authority claims, conflicting prior
    approvals, context flooding, and attacks hidden far from the final request.
  - Acceptance: old untrusted content cannot become a grant or trusted memory.

- [ ] P8-ADV-007, P1: Add attack outcome metrics.
  - Report attack success, safe refusal, false refusal, permission prompt rate,
    unauthorized side effects, evidence corruption, citation corruption, and
    residual process activity.
  - Acceptance: the certificate exposes raw case outcomes and does not replace
    critical failures with one average score.

- [ ] P8-ADV-008, P1: Preserve and minimize every discovered failure.
  - A real failure becomes a smallest reproducible regression fixture before
    its fix is considered complete.
  - Acceptance: the failing fixture fails on the old behavior and passes only
    after the fix.

### 3.6 Phase 8 completion gate

- [ ] P8-GATE-001, P1: Add a Phase 8 exact integration test that exercises both
  wiki graphs, both graph RAG lanes, both real artifact side views, provider
  privacy, permission denial, exact approval, tool-result quarantine, candidate
  staging, rights review, contradiction handling, and adversarial cases.

- [ ] P8-GATE-002, P1: Add a source-only Phase 8 policy verifier with required
  files, runtime markers, hashes, rights rules, action pins, and capability-floor
  checks.

- [ ] P8-GATE-003, P1: Generate a source-bound Phase 8 certificate from the
  final source. The certificate must report all passed and failed cases, pack
  hashes, benchmark hashes, runtime identities, timings, and claim boundary.

- [ ] P8-GATE-004, P1: Run a fresh live-model trial after the final Phase 8
  change. It must include supported, unsupported, contradictory, adversarial,
  and permissioned-hands cases with graceful shutdown.

- [ ] P8-GATE-005, P1: Keep Qwen3 4B as the default unless the expanded
  certification shows a measured language-only failure. If it fails, compare
  the declared 8B candidate under identical evidence, prompts, limits, and
  hardware before changing the contract.

- [ ] P8-GATE-006, P1: Update the reconstruction guide and operator guide so a
  new developer can rebuild Phase 8 and reproduce its evidence from a clean
  checkout.

Phase 8 is complete only when every `P8-GATE` item and all Phase 8 items on
which they depend are closed with fresh evidence.

## 4. Security and authority hardening backlog

- [ ] SEC-001, P0 before stronger autonomy claims: Evaluate an operating-system
  containment boundary for `process.run` and `simulation.run`.
  - The current user-space workspace and environment controls do not prevent an
    approved process from reaching every resource available to the Windows
    account.
  - Compare restricted tokens, AppContainer, Windows Sandbox, a dedicated
    worker account, and virtual-machine isolation.
  - Acceptance: select a boundary from measured tests or document why the
    experiment remains attended and user-account scoped.

- [ ] SEC-002, P1: Decide whether approval remains per multi-action manifest or
  becomes per action or per dependency stage.
  - Acceptance: the UI makes sequential effects unmistakable, and a later
    action cannot run after an earlier action changes its declared assumptions
    unless the approved contract explicitly permits it.

- [ ] SEC-003, P1: Add post-execution effect reconciliation.
  - Compare declared writes, moves, processes, network connections, and outputs
    with observed effects.
  - Acceptance: undeclared observable effects fail the run and are rendered as
    a security event without becoming trusted evidence.

- [ ] SEC-004, P1: Expand filesystem boundary tests.
  - Cover junctions, reparse points, hard links, alternate data streams,
    network shares, case aliases, device paths, reserved names, path-length
    edges, concurrent replacement, and antivirus file locks.
  - Acceptance: every escape or ambiguous target fails before mutation.

- [ ] SEC-005, P1: Expand web-fetch boundary tests.
  - Cover DNS rebinding, address-set drift, IPv4 and IPv6 aliases, proxy
    variables, certificate failure, chunked bodies, compression bombs, MIME
    confusion, slow responses, and connection reuse.
  - Acceptance: connections use only permission-bound public addresses and
    bounded verified bytes.

- [ ] SEC-006, P1: Add executable policy profiles.
  - Decide which exact interpreters, compilers, build tools, and binaries may be
    proposed for each workspace.
  - Acceptance: an executable remains path and SHA-256 bound, with arguments,
    working directory, environment, timeout, and effects visible before
    approval.

- [ ] SEC-007, P1: Add secret-content detection and operator warnings.
  - The runtime already strips provider environment secrets, but a user can
    still place a secret in a prompt, file, process argument, or document.
  - Acceptance: high-confidence secret patterns trigger a local warning and
    redacted diagnostic path without silently blocking legitimate work.

- [ ] SEC-008, P1: Add an integrity-linked security event log.
  - Include denials, tampering, replay, drift, path rejection, blocked network
    targets, process termination, and artifact verification failure.
  - Acceptance: events are locally inspectable, bounded, privacy aware, and
    hash chained or otherwise tamper evident.

- [ ] SEC-009, P1: Add approval-surface spoofing tests.
  - Cover overlay attempts, misleading plan prose, clipped arguments, hidden
    scroll areas, Unicode lookalikes, focus changes, and stale manifests.
  - Acceptance: trusted controls always show the canonical manifest and final
    hash independently of model-authored prose.

- [ ] SEC-010, P2: Commission an independent security review after Phase 8 is
  source complete.
  - Acceptance: every finding has a disposition, regression test where
    applicable, and re-review result. A list of unfixed findings is not a pass.

## 5. Reliability, performance, and data-lifecycle backlog

- [ ] REL-001, P1: Diagnose the installed live-response performance gap.
  - Resident certification measured roughly 73 to 94 generated tokens per
    second, while the recorded installed multidisciplinary transaction reported
    3.051 tokens per second and about 69.9 seconds end to end.
  - Separate intent generation, retrieval, prompt processing, response
    generation, artifact commit, side-view generation, desktop transport, and
    recovery overhead.
  - Acceptance: a repeatable profile explains the gap and records cold-load and
    warm-request timing separately.

- [ ] REL-002, P1: Create a supported hardware matrix.
  - Measure CPU-only behavior, available GPU backends, low-memory failure,
    driver mismatch, and the meaning of `gpu-layers=auto` on tested systems.
  - Acceptance: the UI and docs report actual placement and honest expected
    performance for every supported configuration.

- [ ] REL-003, P1: Add sustained queue and concurrency tests.
  - Exercise the single resident generation slot, eight-request queue,
    cancellation, timeouts, retries, UI pressure, and shutdown under load.
  - Acceptance: no request is lost, duplicated, published partially, or left
    running after cancellation.

- [ ] REL-004, P1: Run long-duration operator and desktop soak tests.
  - Include repeated evidence queries, abstentions, permission proposals,
    denials, approved tools, model restarts, and side-view changes.
  - Acceptance: declare a duration and workload, then prove bounded memory,
    handle count, disk growth, queue depth, and process count.

- [ ] REL-005, P1: Expand fault injection.
  - Cover power-loss-style interruption, disk full, read-only state, corrupted
    journal, truncated artifact, locked file, killed model, killed backend,
    killed desktop, and interrupted updater.
  - Acceptance: recovery never publishes partial work or loses the last known
    good installation.

- [ ] REL-006, P1: Define state and journal schema migrations.
  - Acceptance: each supported prior version has a tested read, migrate,
    rollback, or explicit refusal path. No update silently discards user data.

- [ ] REL-007, P2: Add user-controlled retention, export, and deletion.
  - Cover sessions, request artifacts, tool artifacts, logs, quarantined items,
    review records, model files, and rollback installations separately.
  - Acceptance: each action previews exact targets, preserves immutable evidence
    rules, and verifies the result.

- [ ] REL-008, P2: Add a redacted diagnostic export.
  - Include runtime versions, health, hashes, recent bounded errors, release
    identity, and requested artifacts without model bytes, grants, prompts,
    unrelated workspace data, or secrets.
  - Acceptance: automated tests seed secrets and prove they are absent.

- [ ] REL-009, P2: Add graph and transaction database maintenance tests.
  - Measure reopen, integrity scan, backup, restore, compaction if introduced,
    and corruption handling at projected Phase 8 scale.
  - Acceptance: maintenance cannot alter claim identity or evidence history.

## 6. Permissioned-hands capability backlog

- [ ] HAND-001, P1: Add a permissioned restore-from-quarantine capability.
  - Acceptance: it restores only the exact hash-bound item to an unoccupied
    approved destination and records the full reversible transition.

- [ ] HAND-002, P1: Add content and filesystem diff previews before approval.
  - Acceptance: writes, patches, moves, quarantine actions, and generated
    documents show bounded canonical before and after evidence.

- [ ] HAND-003, P1: Add transactional multi-file workspace changes.
  - Acceptance: either every approved write is committed with a manifest or the
    workspace returns to its verified starting state.

- [ ] HAND-004, P2: Add a dedicated Git capability family instead of relying
  only on generic process execution.
  - Separate status, diff, stage, commit, branch, fetch, push, and destructive
    operations by risk.
  - Acceptance: remote writes always require exact explicit permission and no
    destructive Git fallback is available.

- [ ] HAND-005, P2: Add first-class build and test profiles.
  - Profiles may wrap pinned Python, Rust, .NET, and Node commands while keeping
    executable hashes, arguments, workspace, environment, timeouts, and output
    limits visible.
  - Acceptance: profiles reduce planner ambiguity without hiding effects.

- [ ] HAND-006, P2: Strengthen simulation artifacts.
  - Add declared input matrix, seed, environment, executable identity,
    per-case timing, exit status, measurements, and comparison summary.
  - Acceptance: a simulation can be reproduced from its committed manifest.

- [ ] HAND-007, P2: Add post-tool continuation as a new proposal only.
  - The current result must remain untrusted and cannot authorize the next
    action.
  - Acceptance: the operator sees a fresh exact manifest and must approve it
    independently.

- [ ] HAND-008, P3: Evaluate richer document output such as DOCX and PDF.
  - This is optional and must use format-specific validators and a real rendered
    side view.
  - Acceptance: no general office automation or macro execution enters through
    the document capability.

- [ ] HAND-009, P3: Evaluate a bounded multi-fetch research capability.
  - It must preserve explicit destination visibility, address binding, rights
    metadata, response limits, and one-time approval.
  - Acceptance: it is not an unrestricted browser and cannot perform login,
    posting, account mutation, or hidden redirects.

## 7. Desktop and operator experience backlog

- [ ] UX-001, P2: Add first-run onboarding that explains Atom, Qwen, evidence,
  abstention, permissions, untrusted results, and the side view before the first
  tool proposal.

- [ ] UX-002, P2: Improve the knowledge-coverage view.
  - Show available domains, pack version, review date, known depth limits, and
    why a question abstained.

- [ ] UX-003, P2: Improve permission review for long or multi-action manifests.
  - Add canonical grouping, risk filters, changed-since-last-view indication,
    and guaranteed visibility of every effect before approval.

- [ ] UX-004, P2: Add session management.
  - Support named sessions, search, export, exact deletion, storage size, and
    recovery status without turning conversation history into evidence.

- [ ] UX-005, P2: Add quarantine management.
  - Show reason, original path, current recovery path, hash, time, related
    proposal, restore eligibility, and explicit purge controls.

- [ ] UX-006, P2: Complete an accessibility audit.
  - Cover keyboard-only operation, focus order, screen readers, labels, high
    contrast, zoom, color independence, reduced motion, error announcement,
    large permission manifests, and both side views.
  - Acceptance: automated checks and a documented manual Windows test matrix
    pass.

- [ ] UX-007, P2: Improve 4.28 GB model provisioning.
  - Add preflight disk-space checks, resume support, clear progress, hash-check
    progress, cancellation, recovery, and precise network errors.
  - Acceptance: interrupted downloads cannot be mistaken for admitted model
    bytes.

- [ ] UX-008, P2: Add an operator-visible performance panel.
  - Show cold model load, prompt processing, generation, queue wait, retrieval,
    commit, side-view render, GPU placement, restart count, and current health.

- [ ] UX-009, P3: Evaluate localization after Unicode and accessibility work.
  - Keep runtime identities, canonical hashes, manifests, and security terms
    unambiguous across translated display text.

## 8. Engineering and continuous-integration backlog

- [ ] ENG-001, P1: Remove the two stale V5 labels in
  `atom_harness_operator_server.py`.
  - The CLI description still says `Operator V5`, and the default output folder
    still uses `operator-v5` even though the active authority runtime is V6 and
    the desktop is Phase 7.
  - Because this file is certificate bound, the change requires complete Phase
    7 or Phase 8 recertification and new package evidence.

- [ ] ENG-002, P1: Add the full Python regression suite to an automatic CI
  surface.
  - The active V7 workflow runs the exact Phase 7 integration but not every
    historical and research Python test.
  - Acceptance: a scheduled or appropriately scoped workflow runs the full
    suite without weakening the required Phase 7 gate.

- [ ] ENG-003, P2: Add code coverage and critical-path mapping.
  - Coverage is diagnostic, not a substitute for behavior tests.
  - Acceptance: report unexercised authority, parser, recovery, update, and
    artifact-binding branches, then add tests for critical gaps.

- [ ] ENG-004, P2: Add dependency-update automation with review gates.
  - The repository currently has no Dependabot configuration.
  - Preserve exact pins, lock files, full-SHA GitHub Actions, model hashes, and
    mandatory recertification for behavior-sensitive updates.

- [ ] ENG-005, P2: Harden repository Actions policy.
  - The current repository setting allows all Actions and does not enforce SHA
    pinning, although the active workflow pins third-party actions manually.
  - Acceptance: restrict allowed actions and enable enforceable SHA pinning if
    the account plan supports it, or document and test the local verifier as
    the compensating control.

- [ ] ENG-006, P2: Add protected-main governance when the repository plan or
  visibility permits it.
  - Current GitHub branch protection and rulesets are unavailable for this
    private repository under the current account plan.
  - Acceptance: required Phase 7 or later CI, review rules, and force-push and
    deletion protections are active, or the limitation remains documented.

- [ ] ENG-007, P2: Add a trusted release workflow.
  - It should build from a tag, verify source policy, run tests, generate
    certificates, package one staged layout, sign approved artifacts, produce
    hashes and SBOM, and upload only after every gate passes.
  - Acceptance: the release job cannot publish from an unverified or dirty
    source state.

- [ ] ENG-008, P2: Add a live-model certification path on controlled hardware.
  - GitHub-hosted CI intentionally omits the 4.28 GB model and GPU trial.
  - Acceptance: a trusted runner or documented local release ceremony binds
    live evidence to the exact commit and package without uploading private
    prompts or secrets.

- [ ] ENG-009, P2: Add clean-machine bootstrap validation.
  - Detect exact Python, .NET, Rust, Node, WiX, WebView2, llama.cpp, and package
    versions before a long build begins.
  - Acceptance: missing or wrong prerequisites produce actionable errors.

- [ ] ENG-010, P2: Decide whether byte-reproducible ZIP and MSI output is a
  requirement.
  - Functional reconstruction is currently certified, but archive timestamps
    and native tool output can change package hashes.
  - Acceptance: either implement deterministic packaging and compare bytes, or
    preserve the current honest functional-reproduction boundary.

## 9. External release backlog

This section is required only when the operator authorizes distribution beyond
the current private local experiment. None of these tasks grants permission to
publish.

- [ ] RELEASE-001, P0: Decide the distribution boundary.
  - The repository is private, so its `releases/latest/download` URL is not an
    unauthenticated update service.
  - Choose an authorized public GitHub release location or another HTTPS
    endpoint, and document access, retention, and incident ownership.

- [ ] RELEASE-002, P0: Decide public version semantics.
  - Determine whether internal Phase 7 version `7.0.0` is published as a
    prerelease, experimental release, or replaced by a separate public version.
  - Acceptance: application, MSI, ZIP, feed, tag, release notes, docs, and
    evidence agree.

- [ ] RELEASE-003, P0: Add Authenticode signing.
  - The current tree contains no code-signing or timestamping step.
  - Obtain and protect an appropriate certificate, sign desktop, updater,
    backend and relevant native executables, and MSI, then verify signatures and
    timestamps from a clean machine.

- [ ] RELEASE-004, P0: Add software-bill-of-materials and third-party notices.
  - Cover Python packages, .NET packages, Rust crates, Node tooling used in the
    build, WebView2, llama.cpp, WiX inputs, Qwen model license, and bundled
    native libraries.
  - Acceptance: SBOM components bind to the released artifact and license
    review has a recorded disposition.

- [ ] RELEASE-005, P0: Add root governance documents.
  - The repository currently has no root `LICENSE`, `SECURITY`, `CHANGELOG`,
    `CONTRIBUTING`, or third-party notice file.
  - Acceptance: the operator approves the project license and disclosure,
    contribution, release-history, and support boundaries.

- [ ] RELEASE-006, P0: Add release security scans.
  - Include secret scanning, dependency vulnerability review, static analysis,
    malware or antivirus inspection of packaged files, and signature
    verification.
  - Acceptance: findings are fixed or explicitly accepted with scope and
    expiration. Unknown critical findings block publication.

- [ ] RELEASE-007, P0: Create an authorized tag and GitHub or HTTPS release.
  - There are currently no tags or releases.
  - Upload the verified ZIP, MSI, update feed, release notes, hashes, SBOM, and
    signatures from the exact approved commit.

- [ ] RELEASE-008, P0: Exercise the real published update path.
  - From the prior installed version, request the feed, review the offer,
    consent to download, verify bytes and SHA-256, consent to install, exit the
    app, replace files, verify the new layout, and retain rollback.
  - Acceptance: the exact test uses the published feed and succeeds after the
    final release change.

- [ ] RELEASE-009, P0: Exercise update failures and rollback.
  - Cover unavailable feed, oversized feed, wrong platform, stale version,
    wrong length, wrong hash, interrupted download, locked install, failed new
    layout verification, updater crash, and operator cancellation.
  - Acceptance: the old installation remains runnable and no silent install
    occurs.

- [ ] RELEASE-010, P0: Run independent clean-machine install and uninstall.
  - Test a supported Windows x64 machine without the source checkout or
    development toolchain.
  - Acceptance: first launch, model provisioning, evidence, denial, approved
    harmless tool, side views, shutdown, update, rollback, and uninstall are
    recorded.

- [ ] RELEASE-011, P0: Publish support, privacy, retention, and incident
  boundaries.
  - State what stays local, what network requests occur, what users should not
    submit, how diagnostics are shared, what is retained, and how a compromised
    release is withdrawn.

- [ ] RELEASE-012, P0: Create a release checklist and two-person verification
  ceremony for signing and publication credentials.
  - Acceptance: no single model output or unattended script can authorize an
    external publication.

## 10. Optional expansions that are not blockers

- [ ] OPTIONAL-001, P3: Linux desktop or service support.
- [ ] OPTIONAL-002, P3: macOS desktop support.
- [ ] OPTIONAL-003, P3: An explicitly consented cloud language provider.
- [ ] OPTIONAL-004, P3: Multi-user or organizational policy administration.
- [ ] OPTIONAL-005, P3: General office-document editing.
- [ ] OPTIONAL-006, P3: Authenticated browsing or account actions.
- [ ] OPTIONAL-007, P3: External publishing capabilities.
- [ ] OPTIONAL-008, P3: Additional local language models after identical
  certification, not benchmark reputation alone.
- [ ] OPTIONAL-009, P3: Additional knowledge packs maintained by independent
  reviewers under the same content, rights, and authority contract.

Every optional capability must still ask permission before any external or
mutating action. Optional work cannot remove or bypass an existing safety gate.

## 11. Explicit non-goals and boundaries that must remain

The following are not TODO items unless the operator explicitly changes the
experiment. Future developers should not mark them as missing features and
quietly add them.

- Do not let the LLM approve or execute its own tool proposal.
- Do not reuse an old permission for a changed, expired, denied, or completed
  manifest.
- Do not let model prose, source text, tool output, web content, or conversation
  history write Atom memory or promote itself into knowledge.
- Do not add unattended execution as a convenience mode.
- Do not add a raw shell fallback outside the registered capability contract.
- Do not silently route prompts or evidence to a cloud provider.
- Do not make updates automatic, silent, or hash optional.
- Do not turn bounded public web fetch into unrestricted browsing.
- Do not import copyrighted full text when the rights record allows only a
  citation or metadata.
- Do not collapse formal, empirical, interpretive, fictional, and craft records
  into one undifferentiated confidence score.
- Do not merge the causal experience lane and multidisciplinary reference lane
  into one mutable store.
- Do not remove either runtime wiki graph, either graph RAG path, or either real
  artifact side view.
- Do not narrow the platform below Ornith 1.0 capability parity.
- Do not describe the current system as exhaustive human knowledge, universal
  prompt-injection resistance, clinical authority, or safe unattended
  autonomy.
- Do not revive conventional Atom generative-English distillation as the
  product direction unless the operator expressly reverses its retirement.

## 12. Backlog maintenance checklist

At the end of every phase or release:

- [ ] Reinspect active source for `TODO`, `FIXME`, placeholders, unimplemented
  branches, stale phase labels, and claim-boundary drift.
- [ ] Reconcile this file with the user guide's current limits.
- [ ] Reconcile this file with architecture contracts and certificates.
- [ ] Reconcile this file with GitHub issues, pull requests, tags, releases,
  workflow status, repository rules, and published assets.
- [ ] Move completed items to a dated evidence section without deleting their
  history.
- [ ] Add newly discovered work with dependencies and acceptance evidence.
- [ ] Rerun all relevant verification after the final documentation edit.
