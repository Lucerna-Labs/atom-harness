# Atom Harness Operator V4

Atom Harness Operator V4 is the persistent local interface for the certified
Atom language harness. It keeps the official Qwen language model, the Atom wiki
graph, graph RAG, provider fabric, and operator queue available for an extended
session. It does not change who owns meaning. Atom remains the sole authority
for evidence, retrieval, grounding, citations, memory, and abstention.

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
  -GpuLayers all `
  -MaxQueueDepth 8 `
  -Port 0
```

Port zero selects an available loopback port. `-NoBrowser` starts the host
without opening a browser. The startup line prints the local origin and output
root, but never prints the access token.

## What the two panes show

The conversation pane on the left shows each question, its current state, its
answer when committed, citation count, attempt number, queue occupancy, model
load count, model restart count, and the latest total runtime.

The artifact pane on the right is not a summary invented by the UI. It renders
the actual `atom_harness_side_view.html` from the selected committed
transaction. That artifact binds the answer, citations, Atom primary claim,
evidence packet, wiki graph, graph RAG identity, Spiderweb trace, provider
routes, privacy state, model metrics, and transaction ID.

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

Cloud providers are unavailable through the V4 operator entrypoint. This is a
local language-only runtime.

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

The exact integration gate is:

```powershell
python -m unittest discover -s tests `
  -p "test_atom_language_harness_v4_integration.py" -v
```

Operator lifecycle and journal tests are:

```powershell
python -m unittest discover -s tests `
  -p "test_atom_harness_operator.py" -v
```

Repository policy is:

```powershell
python scripts\verify_atom_harness_v4.py
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
