# Atom Harness Operator V5

Atom Harness Operator V5 is the persistent local interface for the certified
Atom language harness. It keeps the official Qwen language model, the Atom wiki
graph, graph RAG, provider fabric, and operator queue available for an extended
session. It does not change who owns meaning. Atom remains the sole authority
for evidence, retrieval, grounding, citations, memory, and abstention.

V5 adds an experimental permissioned-hands lane. The language model may draft
an exact plan using registered capabilities, but it cannot execute, approve,
or manufacture a grant. Every action waits at a trusted permission surface.
One approval is bound to one manifest and is consumed once.

## Start

From PowerShell:

```powershell
cd C:\Projects\atom-harness
.\run-atom-harness-operator.ps1
```

Or double-click:

```text
START-ATOM-HARNESS-OPERATOR.cmd
```

The initial wait is deliberate. The operator does not accept a question until
the immutable knowledge catalog is open and the local Qwen model has completed
its schema warmup. When both are ready, the launcher opens the browser.

Useful optional launcher arguments:

```powershell
.\run-atom-harness-operator.ps1 `
  -OutputRoot C:\Projects\atom-harness-runtime\my-session `
  -ToolWorkspace C:\Projects `
  -GpuLayers all `
  -MaxQueueDepth 8 `
  -Port 0
```

Port zero selects an available loopback port. `-NoBrowser` starts the host
without opening a browser. The startup line prints the local origin and output
root, but never prints the access token.

## What the two panes show

The left pane has Evidence and Permissioned hands tabs. Evidence shows each
question, its current state, its
answer when committed, citation count, attempt number, queue occupancy, model
load count, model restart count, and the latest total runtime.

Permissioned hands shows the task, outside-influence warnings, exact proposed
capabilities and arguments, declared effects, risk, workspace root, expiry,
manifest hash, and trusted approve, deny, and cancel controls. A proposed plan
does not mean approval. The approve control submits both the current manifest
hash and its one-time decision nonce. If Atom canonicalizes an action ID or
omits an argument the selected capability does not support, the pane names that
normalization and makes the resulting exact manifest authoritative.

The artifact pane on the right is not a summary invented by the UI. For
evidence, it renders the actual `atom_harness_side_view.html` from the selected
committed transaction. For hands, it renders the actual
`atom_tool_side_view.html`. These views bind the real output, permission
receipt, manifest, quarantined results, wiki and RAG identity, Spiderweb trace,
provider route, privacy state, and transaction ID.

## Controls

- Ask Atom submits a typed `OperatorQuestion` to the bounded queue.
- Cancel active requests cancellation through the full provider path. A
  cancelled partial transaction is never published as complete.
- Retry selected creates a new request with a parent ID and incremented attempt
  number. The earlier record remains unchanged.
- Restart model is enabled only when no request is active or queued. It stops
  the current resident lane, starts a new process generation, warms it, and
  reopens admission.
- Shut down performs a graceful close. It stops new admission, lets open work
  finish by default, closes the provider process, and writes final journal
  state.
- Plan tools asks Qwen for a schema-valid proposal against the current
  capability registry. Planning itself has no side effect.
- Approve exact actions creates an in-memory, HMAC-bound, one-time grant for
  the displayed manifest and schedules that exact manifest for execution.
- Deny records a bound denial receipt and leaves the workspace unchanged.
- Cancel tool work propagates cancellation to planning or execution and never
  publishes an incomplete run as successful.

## Permission and outside-influence boundary

Tool tasks, workspace files, web responses, process output, and prior tool
results are untrusted input. Injection-like phrases are surfaced as warnings,
not treated as authority. A previous result enters a follow-up planning request
only when the operator deliberately continues from that completed proposal,
and it remains marked `untrusted-tool-output`.

The current registry covers bounded workspace listing, UTF-8 reads and search,
hash-bound create, replace, and patch, directory creation, hash-bound move and
recoverable quarantine, exact process execution without shell expansion,
multi-case simulations, Markdown, text, HTML, and JSON documents, and
credential-free public web GETs with redirects disabled. Process arguments and
resolved executable paths are visible before approval. Environment forwarding
is allowlisted and excludes provider credentials. The executable SHA-256 is
bound into the manifest and checked again before spawn. Output is drained into
bounded previews without unbounded temporary files, and timeout or cancellation
terminates the process group. Public web connections bypass ambient proxy
settings and connect only to an address in the public address set bound into
the exact permission manifest. HTTPS still verifies the reviewed hostname
through TLS.

All workspace paths are resolved below the configured root and symbolic-link
crossings are rejected. Replacement, patch, move, and quarantine actions bind
the current content hash so a time-of-check to time-of-use change fails closed.
Public web destinations are resolved and checked before permission and again
before execution. The HTTP or HTTPS socket is pinned to that exact address set,
so a later resolver answer cannot silently redirect the approved request. Any
redirect or destination change requires a new plan and new permission.

## Durable session state

The operator writes `atom_harness_operator_journal.json` below the selected
output root. The whole journal is canonically hash-bound after every state
transition. It contains bounded questions, request state, safe hashed error
identities, parent and attempt relationships, Spiderweb flow events, and
references to committed artifacts.

It does not contain:

- the browser access token;
- provider API keys;
- raw provider errors;
- model prompts or raw model output;
- a write path from conversation into Atom evidence.

If a process ends while a request is queued or running, the next operator
instance marks that record `interrupted`. It never pretends an incomplete
request finished. The interrupted record can then be retried.

## Local security boundary

The operator binds only to IPv4 loopback at `127.0.0.1`. The HTTP host header
must match the exact selected port. Full status, control, and artifact routes
require a random in-memory token. The health route exposes only liveness and
declared runtime identities. Every modifying request also requires the exact
same origin, JSON content type, a declared bounded content length, and a body
no larger than 16 KiB. CORS is not enabled.

The UI uses a nonce-based Content Security Policy, no remote scripts, no
external fonts, no inline event handlers, and a sandboxed artifact frame. An
HttpOnly, SameSite session cookie is scoped only to artifact routes, allowing
the frame to authenticate without placing a secret in its URL. The
resident `llama-server` has its own separate random in-memory API key, no web
UI, and an explicit no-proxy loopback transport.

Cloud providers are unavailable through the V5 operator entrypoint. This is a
local evidence and permissioned-capability runtime.

## Knowledge and artifact storage

One session-resident Atom database backs the wiki graph and graph RAG view.
Every retrieval hashes the database before and after use and rejects a change.
Each committed artifact transaction binds the exact same database snapshot at
`runtime/atom_harness_knowledge.atomdb`.

On the same volume, the transaction uses an NTFS hard link to avoid duplicating
the roughly 57 MB immutable catalog for every request. If a hard link is not
available, it falls back to an fsynced byte-for-byte copy. The transaction
manifest hashes the file either way, so verification is identical.

## Verification

The exact Phase 6 integration gate is:

```powershell
python -m unittest discover -s tests `
  -p "test_atom_permissioned_hands_integration.py" -v
```

Operator lifecycle and journal tests are:

```powershell
python -m unittest discover -s tests `
  -p "test_atom_harness_operator.py" -v
```

Repository policy and the adversarial certificate are:

```powershell
python scripts\verify_atom_harness_v6.py
python scripts\certify_atom_permissioned_hands.py
```

The deterministic endurance certificate uses at least 32 requests. The
default uses 120:

```powershell
python scripts\certify_atom_harness_operator.py --mode scripted
```

The live release certificate keeps the real local model resident for one hour
and completes at least 100 mixed requests:

```powershell
python scripts\certify_atom_harness_operator.py `
  --mode live `
  --duration-seconds 3600 `
  --requests 100 `
  --gpu-layers all
```

The live report separately records operator duration, request count, model
process generations, load and restart counts, cancellation and retry,
transaction and side-view verification, Atom-store immutability, journal
integrity, Python memory growth, process working-set growth, and GPU-memory
growth. Process working-set evidence is evaluated within each model process
generation after a declared settling window, and the raw samples remain in the
report.

The latest local live certificate passed 100 requests over 3,601.941 seconds.
It verified cancellation and parent-bound retry, idle restart and rewarm,
unchanged Atom knowledge, transaction and side-view integrity, journal
integrity, local-only routing, bounded Python growth, settled working-set
growth across both process generations, and clean shutdown. The report
SHA-256 is
`43d93ccfa6e5f22ddce55a630712bf67d0b1bcf3ee6f96884362eb6067bbc237`.
See `DEVELOPER_NOTES.md` for the complete evidence and the explicit WDDM
GPU-measurement limitation.
