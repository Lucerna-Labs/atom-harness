# Atom Harness Developer Notes

## Phase 7 multidisciplinary knowledge engineering record

Phase 7 adds a separate, immutable reference-knowledge fabric while preserving
the causal evidence kernel and Phase 6 permissioned hands. The active registry
identity is `language-harness-v6`. The desktop identity is
`atom-harness-desktop-v7` version `7.0.0`. The mandatory Phase 7 integration is
`tests/test_atom_universal_knowledge_integration.py`.

The selected `Qwen/Qwen3-4B-Instruct-2507` Q8_0 artifact does not become a
knowledge store. It receives one bounded Atom evidence packet and renders a
schema-constrained response. Atom remains authoritative for routing, graph
retrieval, claim identity, source identity, epistemic status, grounding,
abstention, memory, permission, and transaction publication.

The first versioned pack is
`knowledge_packs/universal-foundation-v1/manifest.json`. It contains 15 seeded
domains, 45 Atom-authored claims, 22 source records, 454 graph nodes, and 650
graph edges. This is a verified foundation and an ingestion contract, not a
claim of exhaustive human knowledge. `ATOM_UNIVERSAL_KNOWLEDGE.md` is the full
operator and pack-authoring reference.

### 1. Architecture decision

The causal and multidisciplinary records stay in distinct lanes. The causal
lane answers from saved Atom experiences in the Rust database. The reference
lane answers from sourced claims in a content-addressed pack. This separation
prevents a literary interpretation, writing heuristic, scientific model,
formal result, and causal intervention record from collapsing into the same
kind of confidence.

Routing is deterministic. An exact saved causal relationship retains the V3
two-stage intent and response path. A question matching the multidisciplinary
taxonomy enters the new graph-first path and needs only one language
completion. An unresolved question produces a bounded abstention rather than
allowing the model to answer from resident weights as hidden evidence.

The lane arbiter requires a multidisciplinary route score of at least 8. This
prevents one generic token such as `relation`, `map`, or `topology` from
diverting an older causal request into an unrelated reference claim. Exact
cause-to-effect grammar, including compact `cause-to-effect` spelling, keeps
causal precedence. Retrieval still enforces its independent passage threshold,
so route admission does not itself license an answer.

### 2. Runtime modules

`atom_multidisciplinary_knowledge.py` owns strict pack loading, path and link
checks, file and manifest SHA-256 verification, source, domain, and claim
validation, deterministic domain routing, graph construction, query-created
Spiderweb threads, intersections, neighbor preload, and bounded retrieval.

`atom_knowledge_protocol.py` owns the multidisciplinary response schema. The
grounding object must exactly repeat the packet's claim ID, domain, claim type,
epistemic status, and statement hash. Citation IDs must exist in the same
packet. Invalid or unsupported output fails closed.

`atom_harness_runtime.py` selects the causal, multidisciplinary, or unresolved
lane before provider generation. It binds the request to the active knowledge
hash, supplies only the selected packet, validates the response, verifies that
both knowledge stores stayed unchanged, and emits lane-specific Spiderweb
evidence.

`atom_harness_experiment.py` commits the pack, multidisciplinary wiki graph,
evidence packet, response, workflow bindings, and side view into the same
atomic transaction. `atom_tool_fabric.py` records the same immutable knowledge
identities around permissioned execution without giving tool output a write
path into knowledge.

`atom_harness_side_view.py` validates and renders either artifact shape. A
multidisciplinary card shows the exact claim, type, epistemic status, fictional
marker, domain, sources, rights metadata, limitations, citations, grounding,
and graph/thread identities. The runtime binding marker remains
`render_atom_harness_artifact`.

`atom_harness_operator_server.py` and `atom_harness_operator_ui.py` expose both
new runtime identities through the authenticated loopback health and operator
surface. The left-side primary surface and right-side real artifact view remain
separate. Phase 6 permission controls remain unchanged.

### 3. Pack and rights boundary

The manifest names every pack file and its SHA-256. The loader rejects unknown
schema fields, missing or duplicate identities, invalid enum values, source
references that do not exist, unseeded domains, path escapes, symbolic links,
file hash changes, and manifest changes during the session.

Source records use explicit rights lanes. Citation-only records carry metadata
and links but no copied source text. All initial claim statements are
Atom-authored summaries. Fiction must remain in a literary claim type and carry
the fictional flag. Craft advice is a heuristic. Generated prose never becomes
an independent source or a knowledge mutation.

### 4. Spiderweb topology

Domain, claim, source, and relationship ground lanes preload before admission.
A question creates a temporary thread from observed routing flow. Related
domains become typed intersections only when the query activates them. The
on-ramp is `BoundedKnowledgeQuery`; the off-ramp is
`BoundedKnowledgeEvidence`. The committed trace retains the selected ground
lanes, intersections, preloaded manifests, provider route, and artifact
publication vibration.

This topology is additive to the causal and tool highways. It does not replace
them with a central agent loop. L3 policy continues to belong to Atom and the
operator, never the language provider.

### 5. Desktop and packaging changes

Desktop V7 validates the new wiki and RAG markers during backend startup. The
PyInstaller specification includes the complete versioned pack. The installed
layout verifier locates the packaged manifest, rejects reparse points, verifies
every declared file hash, parses the taxonomy, source registry, and JSONL
claims, and enforces floors of 15 domains, 45 claims, and 20 sources.

The build produces `Atom-Harness-7.0.0-windows-x64.zip` and
`Atom-Harness-7.0.0-windows-x64.msi`. The update contract remains schema 1,
explicit opt-in, byte-count and SHA-256 verified, staged outside the install
directory, and applied only after the application exits.

### 6. Verification ownership

`scripts/verify_atom_harness_v7.py` checks the declarations, runtime markers,
pack, CI, promoted certificate, opt-in update policy, and release evidence.
`scripts/certify_atom_universal_knowledge.py` reruns the exact integration and
binds the relevant source files into a canonical certificate.

The Phase 7 integration exercises a real committed multidisciplinary artifact,
all 15 routes, citation and epistemic separation, prompt-injection boundaries,
manifest and shard tampering, unresolved knowledge abstention, Phase 6 hands,
and both user-visible side views. Historical causal, operator, provider,
transaction, desktop, Rust AtomDB, and Rust native language tests remain part
of the full regression pass.

Phase 7 preserves the Ornith 1.0 capability floor. The broad tool vocabulary,
human permission boundary, coding, builds, simulations, documents, workspace
management, web reads, causal evidence, graph RAG, and artifact views remain
available. The new knowledge lane adds breadth without narrowing those
capabilities or transferring authority to Qwen.

### 7. Live llama.cpp grammar boundary

The first real-model Phase 7 probe exposed a backend grammar constraint that
the scripted provider could not reveal. The installed llama.cpp build rejected
`answer.maxLength` values above 1,024 while compiling the response JSON Schema.
The failure appeared as HTTP 400 before generation, the provider fabric marked
the route as an admission failure, Atom emitted a deterministic abstention, and
the side-view validator refused to publish the failed artifact. No ungrounded
answer escaped.

The multidisciplinary answer field now uses the same 1,024-character bound as
the certified causal response protocol. Claim IDs, packet-local citation
enums, exact grounding fields, source provenance, statement hashes, and all
post-generation validators remain unchanged. A fresh Qwen run then produced a
committed quantum-superposition answer with one model load, one constrained
completion, the exact claim grounding, a verified transaction, and the real
side view.

The installed desktop visual probe exposed a second integration issue. An
empty iframe sandbox gives the child an opaque origin, so WebView2 did not send
the path-scoped, SameSite artifact cookie and rendered the server's denied
response instead of the artifact. Granting `allow-same-origin` did not resolve
that behavior consistently and would have widened the frame unnecessarily.

The final design keeps the empty sandbox. Trusted top-level operator code uses
the in-memory token header to fetch the exact artifact route, reads the HTML,
and assigns it to the iframe's `srcdoc`. The token never enters the URL or
artifact. The frame receives no scripts, same-origin identity, forms, popups,
navigation, or downloads. Artifact HTML retains `default-src 'none'`, contains
no executable content, and the route remains independently protected by the
same-origin server policy. The path-scoped HttpOnly cookies remain a compatible
authenticated route option, but visible rendering no longer depends on
WebView2 cookie behavior inside an opaque sandbox.

### 8. Successor regression findings

The first complete historical test pass found that weak single-token matches
could divert legacy causal requests into the new reference lane. A request for
the saved `trust-to-belief` relation matched generic multidisciplinary terms
such as `relation` and `map`, then retrieved unrelated claims. The final lane
arbiter rejects those low-confidence scores and recognizes hyphenated causal
grammar before selecting a lane. The V1 causal language suite and the Phase 7
domain suite both pass the repaired boundary.

The same pass found that the Phase 6 unit fixture implemented only the former
causal knowledge interface. Phase 7 tool artifacts now snapshot and hash both
the causal database and the immutable multidisciplinary pack, so the fixture
was advanced to the active two-lane contract. The permissioned-hands suite now
exercises that real pack identity while retaining all denial, hash-drift,
injection, process-tree, document, simulation, and side-view checks.

### 9. Phase 7 closeout and virtualized artifact recovery

The final installed-app probe reproduced a blank artifact pane even though the
transaction files were complete and `verify_committed_run` accepted them. The
failure depended on launch context. When a packaged Windows application such
as Codex launched Atom Harness, Windows redirected LocalAppData writes into the
launcher's package-local cache. The operator journal retained the logical
LocalAppData path while `Path.resolve()` could expose the physical redirected
path. Comparing those two filesystem-resolved spellings caused a valid run to
look as if it had escaped its run root. Both recovered artifacts and a newly
committed quantum-superposition artifact returned `artifact-not-available`.

`atom_run_transaction.bind_recorded_run_directory` is the common repair for
evidence and tool artifacts. A journal path is now an assertion, never path
authority. The binder requires the exact 32-character hexadecimal request or
proposal identity, derives `request-<id>` or `proposal-<id>` under the runtime's
own runs root, compares normalized absolute paths lexically without resolving
the Windows virtualization layer, rejects symlinks and junctions, and returns
the deterministic runtime-owned path. Transaction verification still checks
every committed byte before either side view is served. A changed journal path,
invalid identity, linked directory, missing transaction, or mismatched file
continues to fail closed.

`AtomHarnessOperator.side_view_path` and
`PermissionedToolFabric.side_view_path` both use this binder. They translate a
transaction-integrity failure into their typed state errors, so the loopback
server returns a bounded conflict instead of dropping a request handler. This
keeps the Phase 6 permissioned-hands floor and its tool artifact under the same
repair as the evidence view.

The operator UI now performs three bounded artifact fetch attempts with 0,
250, and 750 millisecond delays. A persistent failure renders an explicit
in-frame unavailable state and stops the 800 millisecond refresh loop from
hammering the artifact endpoint. Selecting the completed item clears that
single failure marker and retries. Raw backend errors never enter the iframe.
The successful path still assigns only the verified HTML response to an empty
`srcdoc` sandbox.

The exact Phase 7 integration now includes
`test_virtualized_run_binding_stays_on_the_logical_runtime_path`. It prevents a
future implementation from reintroducing filesystem resolution, checks the
deterministic binding, and rejects both an escaped path and a malformed
identity. The existing Phase 6 integration continues to exercise real evidence
and tool side-view endpoints through the same binder.

The closeout package and installed evidence are:

- portable ZIP: 138,822,129 bytes, SHA-256
  `00f1100e343cd6f1fc0704df3e5ad9e3ed080dacc7fcf763fcbe3ff93a7d3b0c`;
- MSI: 120,257,743 bytes, SHA-256
  `27b866700d6d41b876b23827ebb80935caeee0bf4a01c44ec905d996b8df0ed9`;
- package file count: 165;
- installed-layout report SHA-256
  `06b894902a363c3da08db3657cc9a56c1c816f1bb0af7b48678b58173c1b9a91`;
- source-bound knowledge certificate SHA-256
  `85767430543163a9c5d715572596a23956e6590d5c54eb2a2a3981aed3570174`;
- live multidisciplinary transaction
  `7c26e38b6e7cd2eddfaaff22964cdc37decaeea1dcae4665fa48d038d86f098e`;
- live artifact SHA-256
  `4918b2d5b56c26fc6e0025d7a20c70ba8db42b1d88ea265216fea8c89dd860d1`;
- visible side-view SHA-256
  `0d96175c4b320eb237360f70b80f5cce84c606384984136f3cbda7f8ed62935c`.

After installing that MSI, Atom Harness launched with one resident Qwen model
load, restored the completed quantum transaction, rendered its real
multidisciplinary artifact on the right, displayed one exact citation and the
bound transaction identity, and shut down without leaving the desktop,
backend, or llama-server process running. The release evidence records this
probe while retaining the narrower claim boundary: it does not claim
exhaustive knowledge, universal injection resistance, unattended autonomy
safety, medical authority, or a published update feed.

## Phase 6 permissioned-hands engineering record

Phase 6 changes Atom Harness from a language-and-evidence operator into an
adversarial capability experiment. The model can propose useful work, and the
runtime can perform real work, but these are deliberately separate powers.
The model never receives a callable tool object, never creates permission, and
never decides that an outside instruction is trusted. The operator must approve
the exact prepared manifest before each execution.

The active registry identity is `language-harness-v5`. The desktop identity is
`atom-harness-desktop-v6` version `6.0.3`. The mandatory Phase 6 integration is
`tests/test_atom_permissioned_hands_integration.py`. The historical V4 operator
and V3 evidence runtime remain registered and testable below this active layer.

### 1. Experimental question

The experiment is not whether a language model can call a function. It is
whether broad, useful capability can be placed behind a boundary that remains
observable and fail-closed when prompts, files, process output, documents, or
web content try to influence the model.

Phase 6 therefore tests these separations:

1. Language planning is not execution.
2. A model proposal is not permission.
3. Outside content is data, not authority.
4. A previous tool result is not permission for a next tool.
5. A visible plan is not the executable object until it is prepared and hashed.
6. One approval cannot be replayed for another execution.
7. Tool output cannot enter Atom memory or become causal evidence.
8. Every completed result must be bound into a real, user-visible artifact.

This is a falsifiable experiment. A bypass of any separation above is a Phase
6 failure, even if the requested task happened to complete.

### 2. Threat model

The runtime assumes that all of the following can contain hostile or misleading
instructions:

- the operator's task text;
- a file read from the workspace;
- search matches;
- process standard output and standard error;
- simulation output;
- generated documents;
- public web content;
- a previous tool artifact supplied to a follow-up plan;
- the language model's own proposal and rationale.

The trusted computing base is intentionally smaller. It includes the local
Python runtime, the typed validators, the capability adapters, the exact
manifest builder, the in-memory permission key, the authenticated loopback
server, the trusted permission controls, the transaction verifier, and the
operator who reviews the manifest.

Phase 6 does not assume the model is honest. It assumes only that valid model
output can be reduced to a registered capability contract. Invalid output is
rejected without granting a capability. Safe reduction may assign Atom's
canonical action ID and omit fields that the chosen capability does not
recognize. Every such change is displayed, included in the exact manifest, and
covered by the manifest hash before the operator can approve it.

### 2.1 Planner-candidate normalization

The resident llama.cpp grammar binds the proposal to the exact task hash,
context hash, registered capability names, and bounded action IDs. A single
JSON Schema union that duplicated all twelve capability argument objects was
tested against the packaged llama-server and rejected at grammar admission.
The production design therefore keeps model generation structurally strict and
performs capability-specific reduction in Atom after generation.

Reduction is deliberately one way. Atom assigns `action-1`, `action-2`, and so
on in array order and removes argument fields that are absent from the selected
capability's schema. It never invents a missing required field, changes a
capability name, repairs an invalid value, adds an action, lowers risk, or
executes the result. Missing required arguments, unknown capabilities, invalid
values, path escapes, and preparation failures still fail closed before a
permission request exists.

The execution manifest carries `planner_normalizations`. The trusted UI shows
each canonicalized action ID and omitted field beside the exact arguments,
effects, risk, and action hash. This prevents a model's broader candidate from
being mistaken for the authoritative manifest.

The live Qwen probe that motivated this boundary selected
`workspace.write_text` correctly but also supplied the `format` field belonging
to `document.create`. Atom omitted only that unsupported field, retained the
required path, content, mode, and existing-file hash binding, prepared the
exact write effect, and stopped at permission. No file was created during the
probe.

### 3. Spiderweb topology

The tool path follows the project Spiderweb doctrine rather than a single
opaque agent loop.

- L0 is the authenticated local transport and bounded planning or execution
  queue.
- L1 contains typed messages such as `ToolTask`, `ToolProposal`,
  `ApprovedToolManifest`, `QuarantinedToolResult`, and `CommittedToolArtifact`.
- L2 is the permissioned action flow. It forms proposal threads, pauses at the
  permission intersection, promotes approved work onto the capability lane,
  and demotes results into an artifact off-ramp.
- L3 contains Atom policy and trusted local operator authority. The model is
  not present at this layer.

Flow events are durable vertical vibrations. Planning, backpressure,
permission request, grant, denial, expiry, capability on-ramp, capability
off-ramp, cancellation, artifact publication, and restart recovery are recorded
with canonical hashes. There is one bounded worker highway, so action order is
deterministic and pressure is visible rather than hidden in unbounded threads.

The capability registry is preloaded before the operator opens admission. A
proposal can reference only names in that registry. This is the same
preload-before-traffic rule used by the resident language lane and the wiki and
RAG catalog.

### 4. Modules and ownership

`atom_tool_protocol.py` owns the planner protocol. It normalizes task text,
bounds and marks prior context as untrusted, detects observable injection-like
phrases, defines the strict proposal schema, validates proposal identity and
hashes, and constructs the planner payload. The planner system message states
that it may propose but cannot execute or grant permission.

`atom_tool_capabilities.py` owns capability preparation and execution. Every
adapter has a JSON argument schema, deterministic preparation step, declared
effects, base and dynamic risk, and execution function. Preparation resolves
paths and programs and computes content bindings before permission. Execution
accepts only the prepared representation.

`atom_tool_fabric.py` owns queueing, proposal lifecycle, permission grants,
execution, recovery, journals, and atomic artifact publication. It is the only
component that can call a capability adapter. Its grant key is random process
memory and is never serialized.

`atom_tool_side_view.py` renders the real committed tool artifact. It validates
the artifact, workflow, wiki graph, and binding identities before escaping and
rendering values. The binding marker is `render_atom_tool_artifact` and the
runtime is `atom-permissioned-hands-side-view-v1`.

`atom_harness_operator.py` hosts the evidence queue and tool fabric as sibling
lanes. A failure while preloading either lane closes both resources before the
operator enters the failed state. Shutdown stops admission, drains or cancels
according to the explicit request, closes the tool lane, and then closes the
resident provider session.

`atom_harness_operator_server.py` exposes the trusted local API. It creates a
separate HttpOnly, SameSite cookie scoped to tool artifact routes, maintains the
existing origin, host, token, content type, and request size checks, and maps
permission conflicts to HTTP 409 without leaking raw error text.

`atom_harness_operator_ui.py` owns the trusted review controls. It renders
untrusted values only through `textContent` and DOM text nodes. It does not use
`innerHTML`. The same sandboxed right-side frame loads either the evidence
artifact or the tool artifact from an authenticated same-origin route.

### 5. Capability registry

The first Phase 6 registry contains twelve capabilities.

| Capability | Purpose | Important preparation boundary |
| --- | --- | --- |
| `workspace.list` | Bounded directory inventory | Workspace-contained path, entry limit |
| `workspace.read_text` | Bounded UTF-8 file read | Regular file, maximum bytes |
| `workspace.search_text` | Bounded literal text search | Root, glob, query, result limit |
| `workspace.write_text` | Create or replace UTF-8 text | Create refuses overwrite, replace binds old SHA-256 |
| `workspace.patch_text` | Exact text replacement | Old SHA-256 and occurrence count |
| `workspace.make_directory` | Create a directory | Exact path and parents flag |
| `workspace.move` | Move a file or tree | Full file or tree hash and absent destination |
| `workspace.quarantine` | Recoverably remove an item from active workspace | Full hash and private recovery destination |
| `process.run` | Run one exact program | Resolved executable and SHA-256, argument array, cwd, timeout, stdin, bounded streamed output |
| `simulation.run` | Run bounded named cases | Resolved executable and SHA-256, base arguments, exact cases, per-case timeout, bounded streamed output |
| `document.create` | Create Markdown, text, HTML, or JSON | Format-extension match and write hash rules |
| `web.fetch` | Read bounded public HTTP or HTTPS content | Public address set bound into permission, pinned connection, no credentials or redirects, byte and time limit |

There is no generic delete capability. Quarantine is recoverable and stores the
recovery path in the result. There is no implicit shell command string. Process
execution always uses an argument array and `shell=False`. If the exact program
is itself a shell, the manifest labels the action critical and displays that
program and every argument before approval.

Only a small allowlist of operating-system variables is forwarded to a child
process. Provider API keys and arbitrary parent environment values are not
forwarded. Standard output and standard error are drained through bounded
in-memory previews while full byte counts and SHA-256 values are retained. They
are never accumulated in unbounded temporary files. Timeout or cancellation
terminates the process group before the action closes.

Workspace paths are relative to an explicit existing root. Absolute paths,
parent traversal, root mutation, and symbolic-link crossings are rejected.
The default Windows workspace is `C:\Projects`, and the launcher exposes
`-ToolWorkspace` so a narrower experimental workspace can be selected.

### 6. Proposal and permission state machine

The durable states are:

`planning -> awaiting-permission -> approved -> executing -> completed`

Terminal alternatives are `no-actions`, `denied`, `expired`, `cancelled`,
`failed`, `failed-closed`, and `interrupted`.

The transition details are:

1. `submit_task` validates and hashes the task, optionally loads only a
   completed parent result as bounded untrusted context, creates a cancellation
   token, emits `hands-planning-thread-formed`, and queues planning.
2. The provider returns strict JSON for `tool.plan`. The validator requires the
   exact task and context hashes and only registered capability names. Atom
   then reduces the untrusted candidate to the selected capability schemas and
   records every safe normalization.
3. Each reduced action is prepared. Preparation converts model strings into
   resolved, bounded arguments and declared effects. Dynamic risk can raise but
   never lower the registered risk.
4. The fabric builds `atom-exact-tool-execution-manifest-v1`, including the
   workspace root, capability registry hash, every action hash, maximum risk,
   creation time, and expiry. The complete core is canonically hashed.
5. The UI receives the exact prepared manifest and a random decision nonce.
   No grant exists while the record is `awaiting-permission`.
6. Approval must echo the proposal ID, manifest hash, and decision nonce. A
   mismatch leaves the proposal waiting and produces no side effect.
7. The fabric creates a random grant ID and an HMAC signature using its
   process-only key. The durable permission receipt contains only a grant hash,
   never the key or signature.
8. Execution atomically pops the in-memory grant, reconstructs the exact
   actions from the approved manifest, verifies the registry, manifest, and
   action hashes again, and marks the grant consumed before the first action.
9. A second approval attempt fails because the state is no longer awaiting and
   the grant no longer exists. A restart also destroys all grants and marks
   nonterminal records interrupted.

Denial uses the same manifest and nonce binding but creates no grant. Expiry
removes any grant and closes the record. Cancellation propagates through the
provider or action lane and cannot publish a partial run as passed.

### 7. Time-of-check to time-of-use controls

Manifest approval must bind what will execute, not merely a model's prose.
Replacement and patch actions record the current file SHA-256. Patch actions
also record the exact expected occurrence count. Move and quarantine record a
deterministic file or full tree hash. Each is recomputed immediately before the
mutation. A changed target fails without overwriting the changed data.

Programs are resolved and hashed during preparation. Both the executable path
and SHA-256 are stored in the manifest and rechecked before spawn. Web hosts
are resolved to public addresses during preparation
and resolved again during execution. The connection bypasses ambient proxy
configuration and opens only to one of the exact approved addresses while
retaining the reviewed host for HTTP and TLS certificate verification. A
changed address set, non-public address, credential-bearing URL, redirect,
unsupported port, response over the approved limit, or timeout fails closed.

These controls reduce common drift and rebinding paths. They do not claim that
all operating-system races are eliminated. Directory entries can still change
between a final user-space check and the kernel operation. This is one reason
Phase 6 remains explicitly experimental.

### 8. Results, memory, and artifacts

Every result uses `atom-quarantined-tool-result-v1` and includes the approved
action ID and action hash, status, elapsed time, `untrusted-tool-output` trust
label, bounded output, and a canonical result hash. A failed action stops the
sequence. Later actions are not attempted.

Before execution the fabric loads the resident `HarnessKnowledge`, hashes the
Atom store, and records the wiki graph hash. It repeats both checks after the
actions. The artifact passes only if Atom memory and the graph are unchanged,
every result belongs to the corresponding approved action, execution stopped
after a failure, all requested actions completed, and every result is marked
untrusted.

The run transaction stages and atomically publishes:

- `atom_tool_artifact.json`;
- `atom_tool_workflow.json`;
- `atom_tool_permission.json`;
- `atom_tool_results.json`;
- `atom_harness_knowledge.json`;
- `atom_harness_wiki_graph.json`;
- `atom_tool_side_view.html`;
- the exact immutable Atom database snapshot.

The transaction is non-overwriting and has a committed file manifest. The
workflow binds the manifest, permission receipt, every result, knowledge hash,
graph hash, wiki runtime, RAG runtime, transaction ID, and side-view runtime.
The right-side UI fetches this committed file. It does not re-render model text
into an unbound preview.

### 9. Loopback API and cookies

The Phase 6 API adds:

- `POST /api/tools/propose`;
- `GET /api/tools/proposals/{proposal_id}`;
- `POST /api/tools/approve`;
- `POST /api/tools/deny`;
- `POST /api/tools/cancel`;
- `GET /api/tool-artifacts/{proposal_id}/side-view`.

Control routes require the in-memory `X-Atom-Operator-Token`, exact loopback
origin, valid host, JSON content type, bounded content length, and exact request
fields. Evidence and tool artifacts use separate cookies with distinct scoped
paths and also accept the trusted in-memory header. Tokens are not placed in
artifact URLs. The operator fetches the exact artifact HTML with that header
and renders it in an empty, script-disabled `srcdoc` sandbox.

### 10. Adversarial and integration coverage

`tests/test_atom_permissioned_hands.py` exercises five core attacks and flows:

- a forged manifest hash cannot approve and a bound denial creates no file;
- a Qwen-shaped mixed argument candidate is reduced to the chosen capability,
  the omitted field and canonical ID are visible, and denial still creates no
  file;
- a real code, simulation, and document workflow requires one exact approval,
  commits a verified artifact, preserves Atom memory, and rejects replay;
- an injected file that claims permission remains tainted, and a follow-up
  workspace escape is rejected before any new permission can be created;
- a file changed after approval fails the hash check and preserves the newer
  external content;
- a changed process executable fails before spawn, while process output is
  streamed into bounded previews rather than unbounded files; and
- public web execution connects only to the address set bound into the exact
  permission manifest.

`tests/test_atom_permissioned_hands_integration.py` uses the real Atom catalog,
wiki graph, graph RAG, provider fabric, operator, authenticated loopback server,
trusted permission route, capability execution, atomic transaction verifier,
evidence side view, and tool side view. It first completes a grounded evidence
request, then proves that a tampered permission gets HTTP 409 with no side
effect, approves the exact manifest, runs generated Python simulation cases,
writes a document, verifies both artifact bindings, and confirms that the
planner request has `model_may_execute` false.

`tests/test_atom_harness_desktop_v6_integration.py` invokes that exact chain
through the declared desktop gate and checks the installed runtime and opt-in
update contracts.

`scripts/certify_atom_permissioned_hands.py` runs all three suites and writes a
source-bound report below `local-results`. It records normalized source hashes,
the source-manifest hash, observed adversarial cases, command, timings, and a
report hash. Its claim is deliberately narrow: it certifies the deterministic
and loopback test matrix, not universal prompt-injection resistance.

`scripts/verify_atom_harness_v6.py` checks the machine declarations, active and
historical runtimes, wiki and RAG wiring, both side-view bindings, permission
rules, capability source markers, update contract, desktop runtime, pinned CI
actions, Git candidate safety, and the 4,000-line Rust crate ceiling.

### 11. Desktop and packaging boundary

The Phase 6 desktop remains a thin .NET shell. It expects loopback server V2,
operator UI V5, the permissioned-hands fabric, and the tool side-view runtime in
the backend startup record. It does not implement a second permission engine.
All approval controls are served by the authenticated Python runtime inside the
existing WebView2 surface.

The updater boundary is unchanged and remains opt-in. `lucerna-update.json`
schema 1 prohibits automatic download and automatic install, requires explicit
consent, SHA-256, external staging, application exit before replacement, and a
rollback backup. Phase 6 packages are version `6.0.3` and the build output uses
`local-results/desktop-v6-package-*`.

### 11.1 Phase 6.0.3 completion evidence

The final local package completed at `2026-07-31T15:59:33.7968270Z` with 157
manifest-bound application files.

- Portable ZIP: 138,764,940 bytes, SHA-256
  `ede0d697dbb3351f513632fe572b68ea84010fb3f2bdcd97dd01594abed5fb63`.
- Per-user MSI: 120,206,546 bytes, SHA-256
  `2cb27e6ea84810b21935ee08418cc9aeadc117d3ca90e7cc38a7bcbf39656dc3`.
- Bundled llama server: SHA-256
  `2ab5559be6a09d9372fd107d7318eb6265eecf1761cdea62674667c752851639`.

The MSI installed without elevation and created the desktop and Start menu
shortcuts. The installed-layout verifier passed the complete release manifest
with WebView2 `150.0.4078.105`. Its report SHA-256 was
`300d7caa5107af7e10e92a32eb4b8859b274cc1ea0257993669772b88d534e58`.

The installed application completed the real local Qwen question
`How can repeated verification turn trust into a stable belief?` in 7,409 ms.
It used 2,737 wiki nodes, retrieved seven passages, emitted one bound citation,
performed no LLM memory write, and used no cloud evidence. Intent generation
measured 89.467 tokens per second and grounded response generation measured
74.158 tokens per second. The model stayed resident with one load and zero
restarts. The committed transaction SHA-256 was
`ed4c5881cb29d15fa6d383cb20eee7318cf8aa71842a7e7311a59e371a6eb101`;
the logical artifact SHA-256 was
`478e70eab7478dd993ad5d19e138c6e2251bb7c425fe768001f559fc7ad1f71b`;
and the real right-side view SHA-256 was
`42f413efaa03c1f74f5fbf9f7b41b83170df978ac8dc429712bccf1b823e0198`.

The live hands probe asked Qwen to write
`phase6-v603-live-denied.txt`. Qwen selected `workspace.write_text` and supplied
one extra `format` field. Atom displayed that field's omission, the exact
prepared arguments, effects, risk, action hash, and manifest hash before any
permission could exist. The operator used Deny. The durable result had zero
tool results, no persisted grant or grant secret, and no proposed file at
either the configured or application-virtualized workspace path. Closing and
reopening the installed application recovered the same denied proposal and
normalization record. The manifest SHA-256 was
`2aa800abc5f60c08bf0d26e02774a6fb095a13b844bc8c16e15ec515b55257f5`;
the permission receipt SHA-256 was
`bc71e7d4c6efa6dfa019bdf85e2441425ee0b5946a1791696dd56840b7ba4333`;
and the recovered journal SHA-256 was
`75ec33b012ec6eaf8df606163b12897bc75e5933e10b1aef5819f9c8a5c61075`.

The 6.0.3 hardening audit also exercised four resource and outside-influence
boundaries. Public web connections use only a permission-bound public address,
so the HTTP layer cannot perform a later ambient DNS lookup. A changed
executable hash fails before spawn. Standard output larger than 512 KiB is
fully drained and hashed but only a bounded preview is retained, with no
unbounded output file. A timed-out Python parent and its sleeping child were
both gone before the action returned.

`atom-permissioned-hands-certification.json` binds the adversarial, loopback,
desktop, live-model, and denial evidence to normalized hashes of the exact
Phase 6 source. Its source-manifest SHA-256 is
`5e73d71146bfc105f421cb47c917393c33ae543d3ea50744b0b4b2fd8751d9b0`
and its canonical report hash is
`93e38dd9191f8f77f27971ae07a864bd241dca41882e71392087616ce066e548`.
Changing a bound runtime, declaration, verifier, or test invalidates this
certificate until the full adversarial certificate is regenerated.

### 12. Adding a capability safely

A new capability is incomplete until all of these are true:

1. It has one focused adapter with a strict JSON argument schema.
2. Preparation resolves all implicit state into exact manifest fields.
3. Declared effects name every read, write, process, network, or external
   consequence the operator must evaluate.
4. Base and dynamic risk are honest and cannot be lowered by model output.
5. Execution accepts only prepared arguments and propagates cancellation.
6. Output is bounded, hash-bound, and marked untrusted.
7. The capability cannot mutate Atom memory or access provider secrets.
8. A denial, tamper, replay, expiry, path escape, and relevant time-of-check
   drift test exists.
9. The exact loopback integration exercises the capability and both mandatory
   knowledge and artifact surfaces after the latest change.
10. Documentation and the machine-readable capability contract are updated.

Do not add a raw fallback. If an operation cannot be expressed with exact
arguments and visible effects, it is not ready for the registry.

### 13. Known limits and honest claim boundary

Phase 6 does not solve prompt injection. It makes the tested authority boundary
explicit and gives the operator a chance to stop a harmful plan. A user can
still approve a harmful exact command. A shell executable can still interpret
its approved arguments as a script. A process can modify any resource its
operating-system account can reach, even though the declared workspace and
sanitized environment reduce ambient reach. The web connection is pinned to
the permission-bound public address set, but public DNS and filesystem checks
remain user-space defenses, not kernel capabilities. The current permission is
per manifest, so one approved multi-action plan can contain several sequential
effects.

For that reason the product remains an adversarial experiment, not unattended
production autonomy. The correct Phase 6 claim is: the harness provides broad
registered hands, requires an explicit exact one-time approval for every
execution, fails closed on the tested tamper and drift paths, quarantines tool
results, preserves Atom memory, and renders the real result beside the primary
surface.

## V4 interactive operator engineering record

This section is the preserved V4 engineering record. V5 is now active.

V4 is a persistent operator around the certified V3 harness. It does not
replace the V3 evidence, provider, transaction, or artifact rules. The V4 host
keeps session resources resident, accepts bounded interactive work, exposes
operator controls, and delegates every question to the full V3 transaction
path.

### Product and authority boundary

The V4 registry identity is `language-harness-v4`, with
`atom_harness_operator_server.py` as its runtime entrypoint. The underlying
answer runtime remains `atom-language-harness-v3`. Atom still owns the
immutable database, wiki graph, graph-first RAG, primary claim, citation
closure, machine grounding, and deterministic abstention. Qwen is still a
language-only membrane. Conversation history is operational state, not
evidence, and cannot mutate Atom DB.

The operator is local-only. V4 constructs
`AtomHarnessSession.official_local`, whose fabric policy admits only
`ProviderLocation.LOCAL`. The server CLI has no cloud flag, fallback provider
argument, or credential input.

### Preload lifecycle

`ResidentLanguageLane.preload()` starts `llama-server`, waits for health,
executes the strict warmup schema, and returns hash-bound non-secret evidence
without incrementing the user request count.
`LlamaCppResidentJsonLanguageModel.preload()` binds that lane evidence to the
admitted model identity. `ProviderFabric.preload_runtime()` invokes preload
only for policy-admitted providers. Non-resident test providers retain an
honest `manifest-only` preload mode.

`AtomHarnessSession.preload_knowledge()` creates or reopens one durable
immutable catalog below `.atom-operator-runtime/knowledge`. Reopen reconstructs
and validates the graph against the current evidence and model files. A
mismatch fails closed. `preload_runtime()` opens knowledge first, then warms
the provider fabric. `AtomHarnessOperator.start()` stays in `preloading` and
does not expose its on-ramp until that complete operation succeeds.

The ordinary V3 session does not automatically warm the model in its
constructor. This preserves the historical live certification semantics where
the first user completion is the measured cold path. Operator V4 calls the new
explicit preload operation before traffic.

### Immutable catalog binding

`run_atom_language_harness` accepts an optional preloaded `HarnessKnowledge`.
When supplied, retrieval runs against that exact verified session object
instead of constructing a new database. The per-request transaction still
contains `runtime/atom_harness_knowledge.atomdb` and requires it before seal.

`RunTransaction.snapshot_file()` first attempts a same-volume hard link to the
immutable session database. This avoids physically copying roughly 57 MB for
every operator request. If the filesystem refuses a hard link, the method
falls back to a bounded streaming copy, flushes the file, fsyncs it, and
atomically replaces the staged target. In both cases the transaction manifest
calculates the complete SHA-256 and verification sees the same transaction
path. No behavior depends on hard-link support.

### Durable operator core

`AtomHarnessOperator` owns one session, one bounded `queue.Queue`, one worker
thread, cancellation tokens, and an atomic JSON journal. The default queue has
eight waiting slots and one active worker. A request record contains a random
operator request ID, bounded question, parent request ID, attempt number,
state timestamps, output directory, a compact committed artifact reference,
and a safe error envelope.

Journal states are `queued`, `running`, `completed`, `cancelled`, `failed`, and
`interrupted`. Every journal update is written to a unique temporary file,
flushed, fsynced, and replaced. The complete canonical payload receives
`journal_hash`. Startup rejects malformed, oversized, or hash-mismatched
journals. Any prior `queued` or `running` record becomes `interrupted` on
reopen, with a hashed recovery reason. It can be retried but is never silently
resumed or marked complete.

The journal is capped at 1,000 requests and retains at most 1,000 flow events.
Only committed artifact references include answer text and citation IDs. Raw
provider errors are reduced to type, safe kind, and a SHA-256 identity. The
browser token, provider keys, prompts, and raw provider output are never
persisted.

The worker marks a request complete only after `session.answer()` returns and
`verify_committed_run()` validates the published directory. The journal binds
artifact hash, transaction ID, harness request ID, side-view path, answer,
citations, limitations, knowledge hashes, total time, and provider route
hashes. Failure or cancellation cannot publish a completed record.

### Controls and recovery behavior

Queued cancellation marks the record terminal immediately and lets the worker
skip it. Active cancellation triggers the shared `CancellationToken`; the
provider layer decides how to preempt its transport. The resident lane stops
its child on active cancellation so private work cannot continue after the
operator returns.

Retry accepts only a failed, cancelled, or interrupted request and creates a
new request with the old ID as `parent_request_id` and an incremented attempt.
It never overwrites history or reuses an output directory.

Resident restart requires a ready and idle operator. Admission pauses, every
provider with `terminate_lane_for_recovery` is stopped, the full session
preload path runs, and admission reopens only after warmup succeeds.

Graceful shutdown stops new admission and places a sentinel after existing
queue work. The default waits for open work. A caller may explicitly cancel
pending work. Once the worker exits, the provider fabric closes and final
journal state becomes `closed`.

### Spiderweb operator flow

The operator does not flatten the existing four-layer runtime. Its typed
on-ramp is `OperatorQuestion` and its off-ramp is
`CommittedAtomArtifact`. Submission creates an operator thread from observed
flow. Promotion creates the resident knowledge-language intersection only
when a worker begins that shared path. Queue depth emits vertical backpressure
from L0 transport through L2 flow and L3 orchestration. Cancellation, restart
recovery, resident restart, and artifact demotion also emit canonical
hash-bound vertical events.

This operator trace complements the per-artifact V3 Spiderweb trace. The V3
artifact remains the authoritative fine-grained record of intent, retrieval,
provider routes, and rendering.

### Loopback API security

`AtomOperatorHTTPServer` may bind only to `127.0.0.1`. Port zero is the default.
The expected Host header is the exact loopback address and selected port.
Operator status, control, and artifact routes require a 256-bit random token
generated in server memory and delivered only to the local page in memory.
The health route exposes only liveness and declared runtime identities. Every
POST additionally requires the exact loopback Origin. CORS is absent.

Request bodies must declare JSON, include a decimal content length, fit within
16 KiB, and decode to an object. Control routes accept exact field sets.
Errors return stable safe codes rather than exception text. Server logging is
disabled. Responses disable caching, MIME sniffing, framing, referrers, and
unneeded browser permissions.

The root page receives the access token only as an in-memory JavaScript value.
The startup JSON printed to the console never contains that token.

### Two-pane artifact binding

`atom_harness_operator_ui.py` declares
`ATOM_HARNESS_OPERATOR_UI_RUNTIME` and the exact binding marker
`render_operator_surface`. The declared V4 integration imports and executes
that renderer.

The left pane uses text nodes for questions, answers, state, metrics, and
citations. It does not inject model strings with `innerHTML`. The right pane
fetches the exact committed V3 side view through an authenticated request,
then assigns its returned HTML to a script-disabled `srcdoc` iframe. The root
response sets an HttpOnly, SameSite session cookie scoped only to
`/api/artifacts/`; the ordinary JavaScript access token remains in memory.
Artifact responses allow same-origin framing for direct authenticated clients,
and every other response denies framing. The token does not appear in a query
string, fragment, artifact file, iframe URL, `srcdoc`, or journal.

The page has no third-party assets. A per-response nonce permits only its
bundled style and script. The Content Security Policy otherwise defaults to
none and restricts connections and frames to self. The
artifact remains the existing V3 renderer, so all prior hash and grounding
validation stays in force.

### V4 module map

| Surface | V4 responsibility |
| --- | --- |
| `atom_harness_operator.py` | Resident lifecycle, queue, cancellation, retry, restart, journal, recovery, and Spiderweb flow |
| `atom_harness_operator_server.py` | Exact loopback host, in-memory token, same-origin typed API, safe errors, startup, and shutdown |
| `atom_harness_operator_ui.py` | Two-pane controls and authenticated real-artifact binding |
| `atom_harness_session.py` | Session-resident immutable knowledge and explicit full runtime preload |
| `atom_resident_language_lane.py` | Explicit pre-traffic model and schema warmup plus process supervision |
| `atom_provider_fabric.py` | Policy-aware provider runtime preload |
| `atom_run_transaction.py` | Immutable snapshot hard link with fsynced copy fallback |
| `run-atom-harness-operator.ps1` | Contract-validating Windows launcher |
| `START-ATOM-HARNESS-OPERATOR.cmd` | Double-click Windows entrypoint |
| `scripts/certify_atom_harness_operator.py` | Mixed endurance, cancellation, retry, restart, artifact, journal, memory, process, GPU, and immutability certificate |
| `tests/test_atom_harness_operator.py` | Queue, capacity, cancellation, retry, raw-error hashing, and crash-journal recovery |
| `tests/test_atom_language_harness_v4_integration.py` | Real wiki and RAG, operator API, security, transaction, side view, restart, journal, and declarations |
| `scripts/verify_atom_harness_v4.py` | Fail-closed V4 declarations, source wiring, CI, Git, secrets, model, and crate policy |

### Certification contract

Deterministic mode requires at least 32 full operator requests and defaults to
120. It mixes all eight known Atom domains with unsupported and adversarial
prompts. It exercises queue buildup, active cancellation, retry, resident
restart, journal verification, artifact verification, real side-view
resolution, immutable store hashing, Python allocation growth, process
working-set growth, and GPU-memory growth when available. Working-set samples
are grouped by resident process generation. Each generation receives twelve
settling samples before the certificate compares its first and last
three-sample medians. This prevents the intentional restart from turning a
different Windows file-mapping baseline into a false growth result while still
failing sustained settled growth above 1 GiB or any 24 GiB process ceiling.
The raw resource samples and per-generation calculations remain in the report.

Live mode defaults to 3,600 seconds and requires at least 100 full requests. It
uses the official Qwen GGUF and real `llama-server`. Requests are paced across
the full interval so the certificate proves a resident extended session
instead of a short burst followed by idle claims. It requires exactly one
preload before user traffic, mixed grounded and unsupported results,
successful cancellation and retry, an explicit idle restart and rewarm,
bounded resource growth, an unchanged Atom store, a valid durable journal, no
persisted secrets, and clean shutdown.

Reports and progress files live under `local-results` and are excluded from
Git. A report is evidence only for the exact source, model, backend, GPU
policy, and machine state that produced it.

### Phase 4 live certification evidence

The final live certificate completed at
`2026-07-31T04:08:42.615388+00:00` and passed every declared check. It ran 100
mixed grounded and unsupported requests over 3,601.941 seconds with the
official Qwen Q8_0 artifact, `llama-server` version `10173 (e9fa0781f)`, and
`--gpu-layers all`. The local report is:

```text
C:\Projects\atom-harness\local-results\operator-live-certification-phase4-final-20260730-230836-9980270\atom_harness_operator_certification.json
```

The report SHA-256 is
`43d93ccfa6e5f22ddce55a630712bf67d0b1bcf3ee6f96884362eb6067bbc237`.
Its canonical report hash is
`7db8fb04a05c0c084e31b8a1f11d4ec619ed4d80ddc104ae5e5881b15cd0a552`.
The fifteen-file runtime and evaluator source binding normalizes UTF-8 text to
LF endings and has canonical hash
`523918483823494d73d4a854c6f6f161946ade329a72d83f7b4215eef905f3f9`;
the complete portable file map is recorded in
`atom-language-harness-architecture.json`.

Preload completed with process generation 1, model-load count 1, cold start
5,270 ms, and schema warmup 156 ms before user traffic. The active
cancellation probe produced `ProviderCancelledError`; its retry completed and
retained the parent request binding. The idle restart advanced to generation
2, model-load count 2, and restart count 1, then rewarmed in 139 ms. There were
no provider failures. All 100 requested artifacts passed answerability,
citation, transaction, side-view, and knowledge-binding checks. Total
per-request latency was 1,681 ms minimum, 9,120.5 ms median, and 10,749 ms
maximum.

The Atom knowledge database stayed byte-identical at SHA-256
`f9998147145d14e1a6b406d6e16cb6deaceb320f1c012d7b366235a7608fc352`.
The journal hash, no-secret rule, local-only provider rule, and clean operator
shutdown all passed. Python traced growth was 37,361,336 bytes with a
98,902,902-byte peak. The report retained 96 working-set samples across both
model generations. Generation 1 settled growth was 365,768,704 bytes and
generation 2 settled growth was 483,008,512 bytes, both below the 1 GiB limit;
the maximum observed working set was 13,238,329,344 bytes, below the 24 GiB
ceiling.

Per-process GPU memory was not observable because NVIDIA WDDM returned
`used_memory` as `N/A`, so this certificate makes no numeric GPU-memory-growth
claim. The exact `all` offload policy, two local GPU identities, model
integrity, provider route, and generation lifecycle remain recorded
separately. The machine exposed an NVIDIA GeForce RTX 5070 Ti with 16,303 MiB
and an RTX 3060 with 12,288 MiB under driver 610.88.

### Packaging and update boundary

V4 is a local browser host launched from source. It is not packaged as an
installer or desktop binary in this phase. Therefore no release updater is
activated. If a later phase packages the operator as a user-facing desktop
application, that work must first add `lucerna-update.json` schema 1 with
explicit consent, SHA-256 verification, out-of-install staging, and
replacement only after the running app exits.

## V3 resident language-lane engineering record

V3 changes the local provider lifecycle, not the product authority boundary.
Qwen remains a language-only membrane. Atom still owns evidence, causal
memory, the wiki graph, graph RAG, primary-claim selection, citation closure,
grounding validation, and abstention. The language process receives no Atom DB
handle and has no evidence-write path.

### Side-view binding verification correction

The initial V3 publication produced and validated the real side-view file, but
the declared V3 integration test referred to the renderer only indirectly
through `run_atom_language_harness`. The global completion audit correctly
required the exact declared binding marker, `render_atom_harness_artifact`, to
be exercised by that test.

The V3 integration test now reloads the committed artifact, workflow, and wiki
graph, calls the declared renderer directly, and requires its result to match
the committed HTML exactly. The V3 repository policy mirrors the global audit:
it requires the declared runtime marker in the entrypoint, the binding marker
in the side-view module, and both markers in the declared integration test.
This turns a release-time convention into a CI-enforced repository invariant.

### Why the resident lane exists

The V2 local adapter started `llama-completion` once for every intent or
grounded-response pass. That was simple and isolated, but it reloaded the same
4.28 GB model for every completion. V3 keeps one supervised `llama-server`
process alive for a session. The first admitted request loads the model and
warms the schema-constrained inference path. Later requests reuse that process
and report zero cold-start time. An injected process loss is not hidden: the
active request receives a typed transport failure, the lane enters a stopped
state, and the next request creates a new process generation and performs the
same warmup before traffic resumes.

The resident lane is represented as an elevated permanent Spiderweb lane. It
does not flatten the four layers or replace the provider fabric.
`JsonGenerationRequest` is the typed on-ramp and `JsonGenerationResult` is the
typed off-ramp. The bounded admission queue produces vertical backpressure
vibrations. Cold start and restart propagate from L0 transport through L2 flow
and L3 orchestration. The intersection is recorded only after real resident
flow occurs.

### Process and transport boundary

`atom_resident_language_lane.py` owns the child-process lifecycle. Admission
requires an executable whose basename is exactly `llama-server` and a local
GGUF. The server is bound to `127.0.0.1` on an ephemeral port. A random API key
is generated in memory and supplied only through the child environment. The
key, port, prompts, raw model output, and temporary log paths do not enter
provider manifests or committed artifacts.

Every loopback request uses an explicit no-proxy opener. Process-level HTTP
proxy settings therefore cannot redirect private prompts or health checks away
from the local server.

The server starts with continuous batching, prompt caching, metrics and slot
endpoints, no web UI, disabled logs, reasoning off, a fixed context limit, and
the configured GPU-layer policy. Standard input is closed. Standard output and
error are redirected to a private temporary directory with hard byte limits.
HTTP health, error, and completion bodies also have hard limits. The
completion envelope is SHA-256 bound before its schema-constrained content is
parsed.

The lifecycle state exposed to operators contains only non-secret evidence:
state, liveness, process generation, model-load count, restart count, forced
termination count, request counters, active and queued counts, last cold-start
and warmup durations, and the last exit code. `close()` is idempotent and
stops the child before the temporary authentication state is removed.

### Admission, pressure, cancellation, and recovery

The resident lane has a bounded semaphore and a separately bounded waiting
queue. The default contract is one active slot and eight queued requests.
Excess work fails with `ProviderCapacityError`; a bounded wait records
`resident-language-lane-backpressure`. Cancellation and request timeout stop
the child process so a background HTTP request cannot continue using private
evidence after the caller has left. Unexpected process exit becomes
`ProviderTransportError`. The next admitted request performs a supervised
restart and reports the new generation.

`ProviderFabric` validates the full lane envelope before it can enter a route.
The validator checks runtime, stage, integer counters, possible load/restart
relationships, reuse semantics, exact on-ramp and off-ramp shapes, optional
transport hash, bounded vibrations, and propagation targets. Lane evidence is
included in the route hash, completion manifest, Spiderweb trace, final
artifact hash, workflow binding, and side-view binding.

### Exact vocabulary assistance

The broad domain evaluation found an important 4B-model failure mode: the
model sometimes abstained from a valid request because it did not know the
answer, and sometimes guessed a direction while parsing a question that asked
Atom to retrieve that direction. V3 separates literal parsing assistance from
semantic authority.

`atom_harness_runtime.py` computes exact, boundary-aware vocabulary anchors
from the already preloaded Atom wiki. A narrow grammar recognizes explicit
`from X to Y`, `how X affects Y`, and `does X affect Y` forms. It can assign
cause and effect roles, including self-relations where X and Y are identical.
It never adds a value absent from the wiki and never claims that a relation
exists. If the exact grammar is unambiguous, the final intent is retrieval and
Atom graph RAG decides whether evidence exists. A direction emitted by the
model is removed unless the user explicitly supplied that exact direction
value. Exact user-stated anchors are promoted to required query features.

The assistance record is committed as
`atom-exact-vocabulary-anchor-v1`. It contains the anchors, narrow proposal,
model action, final action, a false semantic-authority marker, and its own
canonical hash. The side view verifies that hash and shows the anchor count
and model-to-final intent path. This makes deterministic assistance visible
instead of silently correcting model output.

### Session host

`AtomHarnessSession` owns one `ProviderFabric` across multiple calls to the
full harness. It assigns a unique hash-derived output directory to every
question, counts started, completed, and failed requests, exposes resident
snapshots without secrets, and closes the fabric exactly once. Explicit output
directories still increment session counts.

`atom_harness_session_cli.py` accepts repeated `--question` values or a bounded
JSON string array. `run-atom-harness-session.ps1` is the public Windows
launcher. The final `atom_harness_session.json` binds every artifact hash,
transaction ID, output directory, process generation, and model-load count.
The single-question launcher remains available, but a process naturally ends
after that one question and therefore cannot provide cross-question reuse.

### V3 module map

| Surface | V3 responsibility |
| --- | --- |
| `atom_resident_language_lane.py` | Authenticated loopback server supervision, warmup, bounded admission, metrics, cancellation, shutdown, and restart |
| `atom_llm_provider.py` | Exact GGUF admission plus the resident JSON provider; the V2 one-shot provider remains for compatibility tests |
| `atom_provider_fabric.py` | Hash-bound lane-envelope validation, route propagation, pressure vibration propagation, and provider lifecycle closure |
| `atom_harness_runtime.py` | Exact vocabulary assistance, typed resident ramps, resident intersection, cold/warm trace evidence, and unchanged Atom authority |
| `atom_harness_session.py` | Reusable multi-request fabric and output allocation |
| `atom_harness_session_cli.py` | Bounded multi-question command-line host and session report |
| `run-atom-harness-session.ps1` | Non-secret Windows entrypoint using contract defaults |
| `atom_harness_side_view.py` | User-visible cold start, warm reuse, process generation, model loads, restart count, queue wait, and intent-assistance path |
| `scripts/certify_resident_language_lane.py` | Live domain matrix, sequential soak, concurrency pressure, injected crash, and recovery certification |
| `tests/test_atom_resident_language_lane.py` | Deterministic reuse, queue, capacity, crash, route binding, closure, and session tests without model weights |
| `tests/test_atom_language_harness_v3_integration.py` | Runtime-wired resident lane, Spiderweb ramps, wiki/RAG, side view, exact anchors, transaction, and declarations |
| `scripts/verify_atom_harness_v3.py` | Fail-closed V3 repository, declaration, CI, secret, Git, model-contract, and crate-size policy |

### Live certification contract

The V3 live gate is intentionally broader than the V2 three-case adoption
smoke. It requires:

1. Direct and paraphrased known-relation requests in all eight crystallized
   Atom domains.
2. Four unsupported or adversarial open-world requests that must abstain.
3. Twenty full harness cases and thirty-six language completions before fault
   injection.
4. Exactly one process generation and one model load throughout the sequential
   soak.
5. Exactly one cold completion and thirty-five warm completions.
6. Two concurrent schema-bound requests with a real bounded-queue vibration
   and no reload.
7. A process termination during active generation that surfaces as a typed
   transport failure.
8. A new process generation, exactly one supervised restart, and a successful
   full harness answer after the injected failure.
9. Fresh wiki graph and graph RAG, primary-claim grounding, closed-world
   citations, immutable Atom memory, atomic transaction verification, and the
   user-visible side view for every applicable case.

The exact report hash, llama.cpp identity, cold-start latency, warm latency,
throughput, and pass flags are copied into `atom-language-model.json` only
after a successful live run. Any later source, declaration, or documentation
edit still requires the entire relevant check set and live certification to be
rerun before publication.

The resident adoption baseline completed at
`2026-07-30T22:47:42.767934+00:00` with llama.cpp 10173
(`e9fa0781f`) and GPU layers set to `all`. Its report SHA-256 is
`5f6242672e1e6b29b244b0ff6ccd6a795924d57c29645b943d33e63217d6924f`.
All 20 cases, all eight domains, and all 36 pre-fault completions passed. The
pre-fault snapshot recorded generation one, one model load, no restart, and no
failed request. One completion carried a 4,864 ms cold start; the other 35
were warm. Warm request latency ranged from 921 to 3,993 ms with a 2,077 ms
median. Generation throughput ranged from 73.330 to 93.661 tokens per second
with an 89.922 median. The concurrency probe completed both requests without a
reload and recorded a 944 ms resident queue wait. The injected crash surfaced
as `ProviderTransportError`; the recovery request passed on process generation
two with model-load count two and restart count one.

### Operational boundaries

- The resident lane is local by default. OpenRouter remains opt-in and does not
  share this local process.
- The API key proves local transport admission; it does not turn model output
  into evidence.
- A successful restart is expected to increase model-load count. The
  single-load claim applies to a healthy pre-fault process generation.
- Metrics are revision-specific. Model, llama.cpp, GPU policy, prompt, schema,
  and cache changes invalidate performance comparisons.
- The default parallel slot count is one because the current certification
  prioritizes bounded memory and observable pressure over throughput.
- The server has no silent installer or updater. The harness is a development
  application and the GGUF remains outside Git.
- Retired generative-English distillation remains retired. V3 does not restore
  a teacher/student training pipeline.

## V2 engineering record

V2 keeps the V1 product identity and hardens the operational shell around it.
Atom still decides meaning, retrieval, citations, and abstention. No provider
can write Atom DB, invent evidence, weaken a schema, or override
`insufficient_evidence`.

The V2 changes are:

- `atom_llm_protocol.py` defines provider locations, capability manifests,
  cancellation, data sensitivity, typed provider failures, and route-bearing
  results.
- `atom_provider_fabric.py` preloads capabilities before question data moves,
  admits only policy-compatible providers, routes an ordered chain, retries
  only typed retryable failures with bounded exponential backoff, opens
  circuits, bounds concurrent calls, and emits Spiderweb vibrations for retry,
  fallback, privacy blocking, circuits, and backpressure.
- `atom_llm_provider.py` binds every local model to its expected size and
  SHA-256, requires the non-interactive `llama-completion` executable, applies
  an explicit prompt transport, enforces exactly one JSON object, accepts only
  the backend's fixed end-of-text sentinel, rejects duplicate keys and all
  other surrounding text, separates load latency from generation throughput,
  maps backend failures to typed errors, and never persists API keys, raw
  backend errors, prompts, model paths, or temporary files in provider
  manifests.
- `atom_run_transaction.py` takes an exact-target lock, stages the whole
  bundle, atomically writes each file, seals a SHA-256 file manifest, publishes
  with one directory rename, refuses overwrite, and recovers or quarantines
  dead-process state.
- `atom_harness_runtime.py` treats the provider fabric as an executable part of
  the Spiderweb. Provider outcomes alter routing and vibrations; they are not
  trace decoration. Atom selects a primary claim from graph RAG, constrains the
  response grounding object to that exact claim, and rejects mismatches.
  Exhaustion degrades to a fixed Atom abstention while cancellation aborts the
  transaction.
- `atom_harness_side_view.py` renders the real artifact beside evidence and
  exposes route attempts, selected provider, privacy policy, failure classes,
  timings, degraded status, the primary Atom claim, and transaction identity.
- `scripts/verify_atom_harness_v2.py` enforces runtime declarations, wiki/RAG
  and side-view wiring, provider and transaction controls, Git artifact safety,
  credential-shaped text checks, the 100 MiB Git file ceiling, and the 4,000
  Rust source-line ceiling for every Cargo package.
- `.github/workflows/atom-harness-v2-ci.yml` runs the policy, Python format,
  lint, compile, focused integration gates, the full Python regression suite,
  PowerShell parsing, Rust format, Clippy with warnings denied, Rust tests, and
  a privacy-blocked launcher run on Windows. The historical Svelte validator is
  restored from its lockfile first. Official GitHub actions are pinned to exact
  commit identities.

The active runtime name is `language-harness-v2`. The V1 integration suite is
retained as a regression boundary. The V2 suite adds adversarial provider,
privacy, concurrency, cancellation, corruption, locking, and crash-recovery
coverage.

### Provider route state

Every provider is classified as `local`, `private`, or `cloud`. The policy
defaults to local and private locations. Merely having
`OPENROUTER_API_KEY` in the environment does not authorize egress.
`-AllowCloud` or `ATOM_ALLOW_CLOUD_DATA=1` is required before a cloud provider
can receive a question or evidence packet.

A provider attempt can complete, fail, be blocked, be skipped by an open
circuit, or be cancelled. Only typed transport, capacity, and timeout failures
are retryable. Retry delay grows exponentially from the configured bounded
base and remains cancellation-aware. Schema and boundary failures can move to
the next admitted provider, but never through a relaxed contract. All
persisted errors are represented by type, failure class, retryability, elapsed
time, and a SHA-256 identity. Raw backend text is not retained.

The route hash covers every attempt and vibration. The workflow binds both
intent and response route hashes to the final artifact. This makes fallback
behavior auditable without treating the provider as semantic authority.

### Run transaction state

The transaction moves through `preparing`, `sealed`, and `committed` states.
The public output directory does not exist until all required files, including
the runtime Atom store and side view, have been written and validated. The
committed transaction manifest binds each file path, byte count, and SHA-256
hash. `verify_committed_run` detects deletion, insertion, truncation, and
content tampering.

An exact output target can have only one owner. Existing output directories are
never overwritten. On startup, a dead-process committed stage is published
only after full manifest verification. Incomplete, invalid, or corrupt stages
are moved under `.atom-harness-v2/recovery` for inspection. Cancellation and
runtime exceptions do not publish partial runs. Relative-path, symbolic-link,
junction, and control-directory escape attempts fail closed.

### Release evidence rule

Verification claims are revision-specific. After the latest source or
documentation edit, rerun every relevant check and the privacy-blocked launcher
path. A prior cloud or local-model result is useful history, but it is not
fresh V2 evidence and never authorizes a new cloud transfer.

### Official local language model selection

The selected default language membrane is
`Qwen/Qwen3-4B-Instruct-2507`. The admitted artifact is the official ggml-org
Q8_0 conversion:

| Field | Contract value |
| --- | --- |
| Repository | `ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF` |
| File | `qwen3-4b-instruct-2507-q8_0.gguf` |
| Size | 4,280,403,520 bytes |
| SHA-256 | `ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1` |
| License | Apache-2.0 |
| Backend | `llama-completion` 10173 (`e9fa0781f`) |
| Prompt transport | `qwen-chatml-manual-v1` through temporary files |
| Harness context | 32,768 tokens |
| Reasoning | off |
| Sampling | temperature 0, seed 1 |
| Adoption | certified local default |

The source model is a 4B dense causal language model with 3.6B non-embedding
parameters, native 262,144-token context, and non-thinking-only behavior. Those
properties match this harness better than a 35B class agentic model:

- Atom already owns causal evidence, memory, wiki graph, graph RAG, policy,
  citation validity, and abstention.
- The LLM has only two bounded jobs: map language to a schema and render a
  bounded packet back into language.
- Q8_0 spends local memory on fidelity for instruction following and strict
  schema generation while remaining small enough for the available local GPU.
- Non-thinking-only output matches the explicit reasoning-off boundary and
  avoids hidden reasoning blocks crossing the one-object JSON transport.

This selection passed live certification for the exact weight, llama.cpp
revision, prompt transport, 32K context policy, machine-grounding boundary,
GPU-layer route, and all declared cases. The machine-readable source of truth
is `atom-language-model.json`. `install-atom-language-model.ps1` reads that
contract, downloads and verifies the weight outside Git, and refuses an
integrity mismatch. `scripts/certify_atom_language_model.py` requires direct
and paraphrased known-relation answers, an unsupported open-world abstention,
both schema passes, wiki/RAG, closed-world citations, exact primary-claim
grounding, the real side view, and separate model-load and
generation-throughput metrics.

Do not move to the 8B candidate merely because it exists. Escalate only after a
recorded certification failure attributable to model capacity. A 35B class
model is not the default escalation target because the harness does not grant
the language membrane 35B parameters worth of semantic authority.

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

The active runtime is declared as `language-harness-v2` in
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
- **Cloud is explicit.** Cloud providers are blocked before request data is
  sent unless the current process has explicit cloud-data consent.
- **Provider failure is typed.** Retry, fallback, circuits, cancellation, and
  backpressure are policy-controlled and cannot weaken a schema.
- **Publication is atomic.** A run becomes visible only after its full file
  manifest passes integrity checks; existing outputs are never overwritten.
- **Artifacts are hash-bound.** The artifact, packet, Spiderweb trace, graph,
  workflow, and rendered side view must all refer to the same run.
- **The side view is a real runtime surface.** It renders the produced answer
  beside its evidence; it is not a mock or a separate demo.
- **The Spiderweb remains layered.** Do not flatten the runtime into a generic
  queue, actor system, or ordinary pub/sub pipeline.

These constraints are enforced in code and in both harness integration suites.
Documentation alone is not evidence that they still hold.

## 4. Source map

| File | Responsibility |
| --- | --- |
| `atom_harness_experiment.py` | CLI entrypoint, provider construction, runtime execution, artifact writing, hash checks |
| `atom-language-model.json` | Official local model identity, artifact hash, runtime policy, certification surfaces, and escalation boundary |
| `atom_language_model_contract.py` | Fail-closed loader, default model-store resolution, and explicit custom-GGUF integrity policy |
| `atom_harness_runtime.py` | Two-pass language flow, Atom authority gate, memory immutability check, Spiderweb trace |
| `atom_harness_knowledge.py` | Wiki graph construction, vocabulary preload, graph-first RAG, bounded evidence packets |
| `atom_llm_protocol.py` | Provider-neutral request/result protocol, strict intent and response schemas, boundary validation |
| `atom_llm_provider.py` | OpenRouter and llama.cpp production adapters plus deterministic test adapter |
| `atom_provider_fabric.py` | Capability admission, privacy, ordered fallback, typed retry, circuits, concurrency, cancellation, and vibrations |
| `atom_run_transaction.py` | Exact-target locks, staged writes, manifests, atomic publication, integrity checks, and recovery |
| `atom_harness_side_view.py` | Hash validation and two-column user-visible artifact/evidence rendering |
| `atom_causal_experience.py` | Loading and interpreting the committed causal evidence corpus |
| `atom_causal_memory.py` | Python bridge to the dependency-free Rust Atom memory binary |
| `atom_causal_memory_rust/` | Atom DB, causal memory, retrieval field, and CLI workspace |
| `run-atom-harness.ps1` | Windows launcher with Python/NumPy discovery and provider selection |
| `install-atom-language-model.ps1` | Resumable official-weight download with byte-count and SHA-256 verification |
| `scripts/certify_atom_language_model.py` | Live three-case certification and separate load-latency and generation-throughput evidence |
| `ai-runtime-knowledge.json` | Required runtime wiki and RAG declaration |
| `ai-artifact-side-view.json` | Required user-visible side-view declaration |
| `ai-provider-fabric.json` | Required provider admission and resilience declaration |
| `ai-run-transaction.json` | Required atomic run-publication declaration |
| `atom-language-harness-architecture.json` | Machine-readable identity, trust, provider, bus, and claim boundary |
| `tests/test_atom_language_harness_integration.py` | Full harness contract and adversarial boundary tests |
| `tests/test_atom_language_harness_v2_integration.py` | V2 resilience, privacy, concurrency, cancellation, transaction, wiki/RAG, and side-view integration |
| `tests/test_atom_provider_protocol_v2.py` | Strict transport parsing, secret redaction, privacy admission, internal-error hashing, and transaction path safety |
| `tests/test_atom_causal_live_integration.py` | Preserved causal learning and evidence-kernel contract |
| `scripts/verify_atom_harness_v2.py` | Repository, declaration, secret, artifact, and crate-size release policy |

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
unchanged. The experiment writes the packet, graph, artifact, workflow, HTML
side view, and runtime store inside a transaction stage. Canonical hashes bind
the semantic files. The renderer validates those bindings before producing
HTML. The transaction then seals byte counts and SHA-256 hashes for the entire
bundle before one atomic publication. Stale, mixed-run, partial, or corrupt
inputs fail instead of creating a plausible-looking public run.

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
contains parsed JSON plus a non-secret completion manifest. Production runtime
code calls providers through `ProviderFabric`, not directly.

Every provider must expose a capability manifest before request data is routed.
The manifest declares provider identity, model identity, location, strict JSON
support, context and output limits, cancellation support, cost tier, and
whether the adapter is test-only. Admission estimates the complete request
against both context and output limits before a provider call. Request,
response, route, and manifest byte sizes are bounded. Admission is fail-closed.

### OpenRouter

`OpenRouterJsonLanguageModel` optionally reads `OPENROUTER_API_KEY` from the process
environment, sends a schema-constrained chat completion, uses temperature zero,
and requires provider parameter support. It is classified as cloud and cannot
receive Atom data without explicit cloud consent. There is no implicit cloud
model. The operator must provide the exact current provider model ID.

The standard-library OpenRouter adapter reports
`supports_cancellation = false` because an in-flight synchronous HTTP request
cannot be preempted reliably. The fabric checks cancellation before and after
the call, and the adapter enforces a timeout and bounded response size. The
local llama.cpp adapter can terminate its child process and reports true
cancellation support. Do not claim that every admitted provider is preemptible;
the aggregate capability and side view expose the real state.

The key itself must never be copied into a config file. `.env.example` contains
names and non-secret defaults only.

### Local llama.cpp

`LlamaCppJsonLanguageModel` is the configured default. It invokes
`llama-completion`, not the interactive `llama-cli`, with a local GGUF, a JSON
schema file, deterministic temperature, configured context length, reasoning
disabled, performance timings enabled, and configurable GPU-layer count.
Prompt and schema data are passed through temporary files to avoid Windows
command-line length limits and private-prompt exposure in process arguments.
Stdin is closed so the child cannot attach to Codex or another parent console.

The Qwen adapter uses `qwen-chatml-manual-v1`: separate system and user roles
are written into the prompt file, payload JSON escapes ChatML delimiter
characters, and the assistant prefix is present before grammar-constrained
generation starts. The adapter refuses unknown prompt transports and refuses
`llama-cli`. A custom GGUF therefore requires both an expected SHA-256 and an
explicit supported chat-template declaration.

`llama-completion` writes a fixed `[end of text]` display sentinel after the
generated object. The parser strips exactly one copy at the end and then
applies the ordinary one-object boundary. Arbitrary suffixes, duplicate keys,
prose, banners, and malformed JSON remain rejected. Performance parsing
supports both current `common_perf_print` and prior
`llama_perf_context_print` timing lines without retaining backend logs.

Every local GGUF must have an expected SHA-256 before it is admitted. The
official filename automatically resolves to the byte count and SHA-256 in
`atom-language-model.json`; a custom filename requires an explicit expected
hash. Byte-count mismatch and hash mismatch fail before question or Atom
evidence data reaches the model. The provider records the verified content
hash, not its machine-local path.

A GGUF file being present does not prove compatibility. The previously probed
`Ternary-Bonsai-4B-Q2_0_g64.gguf` failed against the installed llama.cpp build
with `invalid ggml type 42`. Use a GGUF supported by the installed llama.cpp
version and run a real two-pass request before claiming that local provider is
verified. The installed Qwen artifact was promoted only after the declared
three-case certification, not merely because it loaded.

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
5. Classify the provider location and report honest capability limits.
6. Map transport, capacity, timeout, cancellation, and internal errors to typed
   provider failures.
7. Give the provider no Atom DB or generic tool access.
8. Add it to the ordered provider-chain parser and explicit policy admission.
9. Add deterministic boundary, fallback, privacy, and cancellation tests.
10. Update the architecture contract and these notes.

## 8. Generated run artifacts

Each successful run atomically publishes an ignored directory under
`atom_harness_outputs/`:

| Artifact | Meaning |
| --- | --- |
| `atom_harness_artifact.json` | Validated question, intent, answer, citations, routes, timings, memory hashes, provider manifests, checks |
| `atom_harness_evidence_packet.json` | Exact bounded evidence supplied to the rendering pass |
| `atom_harness_wiki_graph.json` | Runtime wiki graph used by retrieval |
| `atom_harness_workflow.json` | Cross-file, provider-route, transaction, and source/store bindings |
| `atom_harness_transaction.json` | Committed state plus byte and SHA-256 manifest for every run file |
| `atom_harness_side_view.html` | User-visible answer beside graph, evidence, route, privacy, timing, and transaction details |
| `runtime/atom_harness_knowledge.atomdb` | Run-local Atom knowledge store, read-only during language execution |

Do not commit these generated directories. A useful report should identify the
output directory and verification results, not add runtime state to Git.

## 9. Setup and operation

Prerequisites:

- Windows PowerShell;
- Python 3.13 with the versions pinned by `requirements-dev.txt` for release
  verification;
- Rust 1.96.0 with Clippy and rustfmt, pinned by `rust-toolchain.toml`;
- `llama-completion` plus the hash-verified official GGUF for the default
  path; or
- an explicitly configured and authorized OpenRouter process.

Install the Python runtime dependency:

```powershell
python -m pip install -r requirements-harness.txt
```

For the exact CI and release-verification environment:

```powershell
py -3.13 -m pip install -r requirements-dev.txt
```

The release set pins NumPy, Ruff, and PyTorch. PyTorch is required by the
retained field and neural regression surfaces even though it is not on the
Atom Language Harness V2 production path.

Install the official local model outside Git:

```powershell
.\install-atom-language-model.ps1
```

Run the default local path:

```powershell
.\run-atom-harness.ps1 `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Certify the exact model and backend through all live surfaces:

```powershell
py -3.13 scripts\certify_atom_language_model.py
```

Run with OpenRouter only after selecting its exact model ID:

```powershell
$env:OPENROUTER_API_KEY = '<secret from your secret manager>'
.\run-atom-harness.ps1 `
  -Provider openrouter `
  -LlmModel '<explicit OpenRouter model ID>' `
  -AllowCloud `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Run with a custom llama.cpp model:

```powershell
.\run-atom-harness.ps1 `
  -Provider llama-cpp `
  -ModelPath 'C:\models\compatible-model.gguf' `
  -ModelSha256 '<expected SHA-256>' `
  -ChatTemplate raw-prompt-v1 `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Run an ordered local-to-cloud chain only when cloud transfer is intended:

```powershell
.\run-atom-harness.ps1 `
  -ProviderChain 'llama-cpp,openrouter' `
  -ModelPath 'C:\models\compatible-model.gguf' `
  -ModelSha256 '<expected SHA-256>' `
  -ChatTemplate raw-prompt-v1 `
  -LlmModel '<explicit OpenRouter model ID>' `
  -AllowCloud `
  -Question 'In the language domain, what is the direction from trust to belief?'
```

Use `-OutputDir` when a stable evidence location is needed. Without it, the
Python entrypoint creates a timestamped output directory. The target must not
already exist.

## 10. Verification

Run all relevant checks after the latest change, not before it.

Python formatting, linting, and integration:

```powershell
ruff check `
  atom_language_model_contract.py atom_llm_protocol.py atom_llm_provider.py `
  atom_provider_fabric.py `
  atom_run_transaction.py atom_harness_knowledge.py atom_harness_runtime.py `
  atom_harness_side_view.py atom_harness_experiment.py `
  scripts/certify_atom_language_model.py `
  tests/test_atom_language_harness_integration.py `
  tests/test_atom_language_harness_v2_integration.py `
  tests/test_atom_provider_protocol_v2.py `
  tests/test_atom_causal_live_integration.py

ruff format --check `
  atom_language_model_contract.py atom_llm_protocol.py atom_llm_provider.py `
  atom_provider_fabric.py `
  atom_run_transaction.py atom_harness_knowledge.py atom_harness_runtime.py `
  atom_harness_side_view.py atom_harness_experiment.py `
  scripts/certify_atom_language_model.py `
  tests/test_atom_language_harness_integration.py `
  tests/test_atom_language_harness_v2_integration.py `
  tests/test_atom_provider_protocol_v2.py `
  tests/test_atom_causal_live_integration.py

py -3.13 scripts/verify_atom_harness_v2.py
py -3.13 -m unittest discover -s tests `
  -p 'test_atom_language_harness_v2_integration.py' -v
py -3.13 -m unittest discover -s tests `
  -p 'test_atom_provider_protocol_v2.py' -v
py -3.13 -m unittest discover -s tests `
  -p 'test_atom_language_harness_integration.py' -v
py -3.13 -m unittest discover -s tests `
  -p 'test_atom_causal_live_integration.py' -v

# Requires the installed 4.28 GB weight.
py -3.13 scripts/certify_atom_language_model.py
```

Rust workspace:

```powershell
Push-Location atom_causal_memory_rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
Pop-Location
```

Then run the privacy-blocked launcher path, which proves no cloud call occurs,
and a real provider request only when its data transfer has been authorized.
Verify all of the following in the retained output:

- `passed` and `answerable` have the expected values;
- citations are members of the evidence packet;
- every answerable response grounding object exactly matches the packet's
  primary Atom claim and cites its source experience;
- the admitted model byte count and SHA-256 match the contract;
- load latency and generation throughput are reported separately;
- store hashes before and after are identical;
- artifact, packet, graph, trace, and workflow hashes match;
- the side view is bound to that exact artifact;
- provider route hashes, location policy, and transaction identity match;
- every committed file matches the transaction manifest;
- provider secrets do not occur in generated files; and
- `git status --short` remains empty.

The pre-V2 real-provider baseline on 2026-07-29 used
`mistralai/mistral-small-3.2-24b-instruct` through OpenRouter. Its two-pass
request produced five packet-valid citations and a hash-bound side view without
changing Atom memory. That ignored local output is evidence for that exact
revision, not a substitute for a fresh V2 run.

During the pre-V2 2026-07-30 publication verification, all 20 active Python integration
tests and all 43 Rust tests passed after the documentation change, together
with formatting, lint, compilation, contract, and launcher checks. A repeat
cloud request was intentionally not sent because publishing the repository did
not itself authorize transmitting its private evidence packet to a third
party. An on-device probe with the available
`NVIDIA-Nemotron-3-Nano-4B-UD-Q6_K_XL.gguf` loaded successfully but did not
return a JSON object under the required intent schema. That model is therefore
not a verified local provider for this harness. Treat every provider claim as
model-, backend-, and revision-specific.

### Certified Qwen adoption evidence

The grounded adoption run completed at
`2026-07-30T21:05:16.824630+00:00` with llama.cpp 10173
(`e9fa0781f`), `llama-completion`, `qwen-chatml-manual-v1`, a 32,768-token
context, and `--gpu-layers all`. The exact
`qwen3-4b-instruct-2507-q8_0.gguf` was 4,280,403,520 bytes and matched
SHA-256
`ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1`.

All three isolated cases passed:

- `direct-known-relation` answered with the exact primary claim, one
  packet-local citation, artifact SHA-256
  `0c0f7779a5cbcdae12ee1248cf6b2ba72f56febd0d02f142cf116e22f75039e7`,
  and side-view SHA-256
  `421cb4b9ddb6ae43f1069f449229851f2ef1e131b3e00bd04874042dd7e8079f`.
- `paraphrased-known-relation` answered with the same machine-grounded claim,
  one packet-local citation, artifact SHA-256
  `edcb18732d50d9efe9fb99547b657f9b21b7287673d06dd8ae6c6aaaa367ab5f`,
  and side-view SHA-256
  `c27846fcd09e58d64dd88c8d2f0d8988a90149c507ef0b58ada844dfdc320a94`.
- `unsupported-open-world-question` returned Atom's deterministic abstention
  with no citation or grounding claim, artifact SHA-256
  `094ea4c2ae424f2b2afbb8c45e2d1bf9c98d23ce6555fb5a145d8b3ea061179a`,
  and side-view SHA-256
  `1d138fb9b7d04e6ed2f64afa4db3fddc3064a7fc00b65a4f68411bef7d058fe4`.

The five live completions recorded model-load latency from 3,817.31 to
4,464.56 ms with a 3,968.90 ms median. Generation throughput ranged from
74.24 to 93.55 tokens per second with a 93.33 median. Every case also passed
wiki graph and RAG execution, exact machine grounding, citation closure,
memory immutability, committed transaction verification, selected-model
binding, and user-visible side-view binding.

The retained local report is
`C:\tmp\atom-harness-live-cert-20260730-210409-7837501\atom_language_model_certification.json`.
Its SHA-256 is
`48a23b7c9e3e18ea3cdfb87f912e4a7c73141977e895ea71824a19ed8bd3cbdd`.
The report path is machine-local evidence and is not committed. Its summary
and hash are recorded in `atom-language-model.json`. A later source change
still requires a fresh live rerun before a release claim.

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

- confirm `language-harness-v4` is still preserved as a historical runtime and
  `language-harness-v3` remains registered as historical;
- confirm wiki graph, graph RAG, and side view remain runtime-wired;
- confirm provider admission, privacy, and run transaction declarations remain
  runtime-wired;
- confirm no Rust crate exceeds 4,000 Rust source lines;
- inspect the staged diff for secrets and generated artifacts;
- confirm every GitHub action remains pinned by full commit SHA and uses the
  current Node 24 action runtime;
- run Python lint, format, compilation, policy, and all declared integration
  suites;
- run Rust format, Clippy with warnings denied, and all workspace tests;
- parse the PowerShell launcher and run its privacy-blocked end-to-end path;
- run a real configured provider when provider/runtime behavior changed;
- verify the model byte count and SHA-256 before provider admission;
- record load latency separately from generation throughput;
- inspect the real side view, not only the JSON;
- verify memory immutability and every hash binding;
- update architecture and developer notes when a boundary changes; and
- verify the exact remote commit after pushing.

If any authority boundary, real-provider path, runtime declaration, knowledge
integration, side-view binding, warning gate, or test remains unresolved,
report it as unresolved. Do not describe the harness as complete on the
strength of a scaffold or deterministic provider alone.

## 13. Historical Desktop Phase 5 release

### 13.1 Product boundary

This section preserves the Phase 5 packaging record. Phase 5 packaged the
certified Operator V4 as a native per-user Windows
desktop application. It deliberately does not create a new authority runtime.
The .NET shell owns only installation-facing concerns: native windowing,
single-instance admission, model discovery and provisioning, child-process
supervision, safe diagnostics, installed-layout verification, and explicit
update consent. Operator V4 continues to own the authenticated loopback API,
persistent request journal, provider fabric, transaction state machine, wiki
graph, graph RAG, artifact publication, and right-side artifact renderer.

The authority chain remains:

```text
AtomHarness.Desktop.exe
  -> frozen atom-harness-backend.exe
    -> Atom Harness Operator V4
      -> certified wiki graph and graph RAG
      -> resident language-only llama-server
      -> committed Atom artifact
      -> real Operator V4 side view inside WebView2
```

The desktop process contains the runtime markers
`ATOM_HARNESS_OPERATOR_RUNTIME`, `ATOM_HARNESS_WIKI_RUNTIME`,
`ATOM_HARNESS_RAG_RUNTIME`, `ATOM_HARNESS_OPERATOR_UI_RUNTIME`, and
`render_operator_surface`. The Phase 5 integration invokes the declared V4
test that exercises the wiki, RAG, operator API, committed artifact, and real
side view together. This prevents the native shell from satisfying the
desktop gate with an unrelated mock view.

### 13.2 Source layout

- `desktop/AtomHarness.Desktop` is the .NET 9 WinForms and WebView2 shell.
- `desktop/AtomHarness.Desktop.Core` contains the model, release-manifest,
  integrity, release-feed, and safe-update contracts.
- `desktop/AtomHarness.Updater` is the out-of-process replacement helper.
- `desktop/AtomHarness.Desktop.Tests` contains the managed safety tests.
- `atom_harness_desktop_backend.py` is the frozen entrypoint for Operator V4.
- `atom-harness-backend.spec` binds the required compact Atom data and the
  release Rust causal-memory executable into the frozen backend.
- `desktop/packaging/AtomHarness.wxs` defines the per-user MSI and shortcuts.
- `desktop/packaging/PerUserHarvest.xslt` converts harvested file components
  to per-user registry keypaths and supplies nested directory cleanup.
- `scripts/build_atom_harness_desktop.ps1` is the complete release builder.
- `lucerna-update.json` is the required schema 1 opt-in update declaration.
- `atom-harness-desktop-architecture.json` is the machine-readable Phase 5
  boundary.
- `atom-harness-desktop-release-evidence.json` records the package and live
  installed-runtime proof without committing the large binaries.

Dependency versions are centrally pinned. Every .NET project has a package
lock file, restores in locked mode, treats warnings as errors, enables the
latest analyzers, and produces deterministic managed builds. The desktop and
updater are self-contained Windows x64 applications. WebView2 is pinned at
build time while the installed runtime is checked at verification time.
`global.json` pins SDK 9.0.305 with roll-forward disabled because the SDK
selects implicit framework packages such as `Microsoft.NET.ILLink.Tasks`;
allowing a newer servicing SDK would invalidate an otherwise exact lockfile.

### 13.3 Process lifecycle and recovery

The shell creates a named per-user mutex before any backend starts. It assigns
the frozen backend to a Windows job object configured with
`JobObjectLimitKillOnJobClose`. Operator V4 starts its own local
`llama-server` beneath that process. Window close first requests the
authenticated V4 graceful-shutdown route and waits. Disposing the job object
then provides a fail-safe process-tree termination boundary.

Operator V4 journal durability is unchanged. A failed first request remained
visible after application restart. Selecting that row and using Retry produced
a completed second attempt. After another full close and restart, both the
failed attempt and completed attempt were recovered, and the completed
transaction restored its real artifact in the right-side view.

The frozen backend must not invoke Cargo on an installed machine. The package
therefore includes the locked release build of `atom-causal-memory.exe`. The
frozen entrypoint verifies that binary is present, binds
`atom_causal_memory.RELEASE_BINARY`, replaces the build lookup with the
bundled path, and only then imports Operator V4. This import order matters
because the V4 knowledge module captures the release-binary provider during
module import.

### 13.4 Model provisioning

Model weights are not bundled. The application searches the configured path
and known local locations for
`qwen3-4b-instruct-2507-q8_0.gguf`. Admission requires exactly
4,280,403,520 bytes and SHA-256
`ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1`.

If no valid local model exists, the application presents the download size and
asks for consent. It writes a partial file only in the per-user model staging
area outside the installation. The final file is exposed to the runtime only
after incremental SHA-256 and exact-length verification both pass. A short,
wrong, or tampered download is rejected and its partial file is removed.

### 13.5 Update safety

`lucerna-update.json` schema 1 is part of the installed application and is
validated at startup. Automatic download and automatic installation are both
false. The release client accepts only HTTPS feed and artifact URLs, bounds
the feed and artifact sizes, validates application and platform identity, and
requires stable three-part versions.

Update installation is a separate process so the running application cannot
replace itself. The helper verifies the outer package SHA-256, rejects path
traversal, Windows device names, alternate data stream paths, trailing-dot and
trailing-space aliases, and oversized expansion, verifies every file through
`atom-harness-release-manifest.json`, waits for the desktop process to exit,
moves the existing application to a timestamped backup, and moves the fully
staged application into place. If the second move fails, it restores the old
directory. A receipt binds the installed manifest version, package digest,
install location, and rollback location.

The repository is private, so its raw GitHub release URL is not an
unauthenticated production feed. Phase 5 certifies the updater behavior and
local package. Publishing a future release ZIP and feed at an authorized HTTPS
endpoint remains a separate operator-controlled distribution action.

### 13.6 Packaging evidence

The final local build completed at `2026-07-31T06:50:03.9058760Z` with 157
manifest-bound application files.

- Portable ZIP: 138,684,711 bytes, SHA-256
  `808f3699fb7cb12d55894e034770ffecadb8e16f5ffc48e258ae7cb3c0d90cee`.
- Per-user MSI: 120,141,010 bytes, SHA-256
  `f71ed80614d70f002cc0da19cedba2afce312c004162c3de2860ac0b826ffce2`.
- Bundled llama server: SHA-256
  `2ab5559be6a09d9372fd107d7318eb6265eecf1761cdea62674667c752851639`.

The MSI installed under
`%LOCALAPPDATA%\Programs\Lucerna Labs\Atom Harness` without elevation and
created both requested shortcuts. Installed-layout verification checked every
release-manifest entry, the model and update contracts, and WebView2
`150.0.4078.105`. The verification report hash was
`535c71ddc859c50b36dc17b5a891bbca2982061dd5c6ec017354d2a96cc55c77`.

The installed application then completed the question
`How can repeated verification turn trust into a stable belief?` in 7,028 ms.
Transaction SHA-256 was
`c03acd4a7f12d6d5838a5b04e3a97caae1192b97a3c7704ef901a0db87356655`.
Artifact SHA-256 was
`6b0900631d73cd16f92876a98b24c4df9542305febafb45774e11a7557fcc858`,
and the real side-view SHA-256 was
`2e4f69c4b5800a765c0d4bd3418e63db0d2e7d2ca40f2c15ebe47ed597726cf4`.
The runtime used 2,737 wiki nodes, retrieved seven passages, emitted one bound
citation, performed no LLM memory write, used no cloud evidence, and rendered
the committed artifact in the right-side view. Intent generation measured
89.204 tokens per second and grounded response generation measured 72.490
tokens per second. The model stayed resident with one load and zero restarts.

### 13.7 Release verification

The required local release sequence is:

1. Run Python formatting, lint, and compilation checks.
2. Parse every changed PowerShell entrypoint.
3. Run `scripts/verify_atom_harness_v5.py`.
4. Run the exact declared Phase 5 integration.
5. Restore all .NET projects in locked mode.
6. Verify .NET formatting and build the shell and updater with no warnings.
7. Run all managed update and integrity tests.
8. Run Rust format, Clippy with warnings denied, workspace tests, and the
   locked release build.
9. Run the complete Python integration suite.
10. Recompute ZIP, MSI, model, installed-report, and installed-file hashes.
11. Run the installed layout verifier.
12. Launch the installed application, complete a real local-model request,
    inspect the side view, close it, confirm child-process cleanup, relaunch,
    and confirm journal plus artifact recovery.
13. Push the exact reviewed commit and wait for the V3, V4, and V5 workflows to
    finish successfully.

`.github/workflows/atom-harness-v5-ci.yml` independently enforces the Phase 5
policy, exact wiki and RAG plus side-view integration, locked .NET graph,
managed safety tests, warnings-denied Rust graph, bundled causal-memory build,
and a real frozen-backend startup probe. Every external action is pinned to a
full commit SHA.
