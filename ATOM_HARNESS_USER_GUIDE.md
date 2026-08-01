# Atom Harness User Guide

This is the practical guide for Atom Harness Desktop Phase 7. It explains how
to install the application, set up its local language model, ask evidence
questions, use permissioned hands safely, understand the artifact panel, keep
your data, update the application, and recover from common problems.

## 1. What Atom Harness is

Atom Harness is a local evidence and capability experiment. It combines:

1. Atom's causal evidence and memory system.
2. A verified multidisciplinary reference pack.
3. A local Qwen language model that turns grounded results into readable
   language and can propose tool actions.
4. A permission system that requires you to approve every exact tool manifest.
5. A right-side artifact view that shows the real saved output of each answer
   or tool run.

The language model is not the authority. Atom decides which claims and sources
support an answer, when the system lacks evidence, and what a tool proposal is
allowed to contain. The language model cannot approve its own proposal, cannot
give itself a tool, and cannot write into Atom memory.

Atom Harness is experimental. It is not a promise of complete human knowledge,
universal prompt-injection resistance, unattended autonomy safety, or medical
authority.

## 2. What runs on your computer

Atom Harness Desktop runs locally on Windows x64. It starts three cooperating
parts:

1. The desktop window.
2. The Atom backend and operator.
3. A local `llama-server` process that keeps the Qwen model resident.

The interface is served only on `127.0.0.1`, which is your own machine. The
browser access token is created in memory for each launch and is not saved in a
settings file.

Operator V6 does not use a cloud model. A public web request is possible only
through the `web.fetch` permissioned tool, and only after you approve the exact
URL and resolved address set shown in the manifest.

## 3. Requirements

You need:

- A 64-bit Windows computer.
- Microsoft Edge WebView2 Runtime.
- Enough storage for the application, its working data, and the exact
  4,280,403,520-byte model.
- Enough memory to load and run the model through the bundled `llama-server`.
- A compatible GPU is helpful, but the actual usable configuration depends on
  your llama.cpp build and system memory.
- Internet access only if you choose to download the model, check for an
  update, or approve a public web fetch.

As a practical first-install allowance, keep at least 10 GB free. This leaves
room for the model, a staged download, the application, and initial session
data. Long-running sessions and many committed artifacts can require more.

## 4. Install Atom Harness

### 4.1 MSI installation

The per-user MSI is the simplest installation method.

1. Obtain the verified `Atom-Harness-7.0.0-windows-x64.msi` package.
2. If a SHA-256 value was provided with the package, compare it before opening
   the installer.
3. Open the MSI.
4. Follow the installer prompts.
5. Start Atom Harness from the Desktop shortcut or the Lucerna Labs folder in
   the Start Menu.

The MSI does not require administrator rights. It installs to:

```text
%LOCALAPPDATA%\Programs\Lucerna Labs\Atom Harness
```

The certified Phase 7 MSI has this identity:

```text
Bytes:   120,257,743
SHA-256: 27b866700d6d41b876b23827ebb80935caeee0bf4a01c44ec905d996b8df0ed9
```

Use those values only for the Phase 7.0.0 package. A newer release must publish
its own size and hash.

### 4.2 Portable ZIP

The portable package contains the same verified application layout without the
MSI installer.

1. Obtain `Atom-Harness-7.0.0-windows-x64.zip`.
2. Verify the provided SHA-256.
3. Extract the entire archive into one ordinary local folder.
4. Do not run the executable from inside the ZIP viewer.
5. Open `AtomHarness.Desktop.exe` from the extracted folder.

The certified Phase 7 portable ZIP has this identity:

```text
Bytes:   138,822,129
SHA-256: 00f1100e343cd6f1fc0704df3e5ad9e3ed080dacc7fcf763fcbe3ff93a7d3b0c
```

Do not move individual DLLs, the backend, or the `runtime` directory away from
the desktop executable. The application verifies its installed layout before
starting.

## 5. First launch and model setup

The language model is about 4.28 GB and is not bundled with the application.
This keeps the application package much smaller and allows the model to be
verified separately.

On first launch, Atom Harness searches expected locations for the exact model.
If it does not find a verified copy, it offers three choices:

1. Download the exact model.
2. Browse to an existing local copy.
3. Cancel and close Atom Harness.

### 5.1 Download the model

Choose the download option only when you want Atom Harness to retrieve the
certified file. The application stages the download outside its install
directory and verifies both the exact byte count and SHA-256 before admitting
it.

The admitted model is:

```text
Model:        Qwen/Qwen3-4B-Instruct-2507
File:         qwen3-4b-instruct-2507-q8_0.gguf
Quantization: Q8_0
Bytes:        4,280,403,520
SHA-256:      ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1
```

A partial, corrupted, or different GGUF will be rejected.

### 5.2 Use an existing model file

Choose Browse if you already have the exact file. Selecting a file does not
bypass verification. Atom Harness checks its byte count and SHA-256 before
saving the path.

### 5.3 Wait for preload

At launch, the status bar moves through installed-layout checks, model
verification, knowledge loading, and local-model preload. The application does
not accept requests until the model and both knowledge lanes are resident.

The ready message is:

```text
Ready, local model resident
```

First launch is slower than a warm request because it has to start the backend
and load the model. Do not open multiple copies to make startup faster. The
desktop is single-instance.

## 6. Understand the window

The main window has a trusted control area on the left and a committed artifact
area on the right.

### 6.1 Left side

The left side contains two tabs:

- `Evidence` for questions answered from Atom evidence.
- `Permissioned hands` for tasks that may read, write, build, simulate, create
  documents, manage workspace items, run an exact process, or fetch an approved
  public URL.

The top metrics show the model and runtime state, queue activity, request
timing, and restart count.

### 6.2 Right side

The right side displays the real HTML artifact saved in the completed
transaction. It is not a second answer invented by the desktop window.

For evidence, it can show:

- The selected knowledge lane.
- The primary Atom claim.
- Claim type and epistemic status.
- Whether the record is fictional or interpretive.
- Source identities, licenses, and citation links.
- Graph and retrieval details.
- Limitations and abstention information.
- Model load and generation timing.
- Transaction and artifact identity.

For tool runs, it can show:

- The task and proposal identity.
- The exact approved manifest.
- Permission receipt.
- Each action and its result.
- Failure, cancellation, or completion state.
- Proof that tool output remained untrusted.
- Proof that Atom memory and knowledge did not change.

The artifact frame cannot run scripts or use the access token. This prevents a
generated artifact from becoming a hidden control surface.

## 7. Ask an evidence question

1. Select `Evidence`.
2. Type one clear question.
3. Select `Ask Atom`.
4. Watch the request move from queued to running to completed.
5. Select the completed request if it is not already selected.
6. Read the answer on the left and the evidence artifact on the right.

Good questions identify the concept or relationship you want. Examples:

```text
What does quantum superposition mean in quantum mechanics?

What is the difference between syntax and semantics in linguistics?

What does reproducibility mean in scientific research?

What is a graph in mathematics?

What is dramatic irony in literature?
```

The first knowledge pack seeds 15 areas:

- Mathematics and formal science
- Computer and information science
- Physics and quantum science
- Astronomy and space science
- Chemistry and materials science
- Earth and environmental science
- Biological science
- Medical and health science
- Engineering and technology
- Agricultural and veterinary science
- Social and behavioral science
- Linguistics
- Literature and language arts
- Writing and creative practice
- Research practice and philosophy of science

Coverage is intentionally bounded. A domain being listed does not mean every
question in that domain is already answerable.

## 8. Read an evidence result correctly

Use the following order:

1. Read the primary Atom claim.
2. Check the knowledge lane.
3. Check the claim type.
4. Check the epistemic status.
5. Check the cited source record and rights information.
6. Read the language-model rendering.
7. Read limitations or abstention notes.

Important status examples:

| Status | Plain meaning |
| --- | --- |
| `formal` | Bound to a formal definition, axiom, proof, or theorem context |
| `established` | Treated as well-established within the cited scope |
| `consensus` | Represents a supported consensus, not an exceptionless truth |
| `provisional` | Evidence can change with new research |
| `contextual` | Meaning depends on historical, literary, or disciplinary context |
| `interpretive` | One supported interpretation, not a measured fact |
| `heuristic` | Practical guidance, not a law or theorem |

Fictional facts and literary interpretations must remain visibly marked. A
craft principle is advice for practice, not scientific evidence.

## 9. What abstention means

Atom Harness is supposed to refuse unsupported answers. An abstention means the
loaded evidence packet did not contain enough support for a grounded response.
It does not necessarily mean the subject is unknowable.

Do not repeatedly rephrase an unsupported question until the language model
produces confident prose. Instead:

1. Check whether the question is narrower than the available claim.
2. Ask for a concept that is actually represented in the pack.
3. Treat live weather, current sports, breaking news, and other open-world facts
   as outside the static evidence pack unless a separately approved tool flow
   obtains data.
4. For important medical, legal, financial, or safety decisions, consult an
   appropriate current authority even if the system provides background
   evidence.

Abstention is a safety property, not a failure of fluency.

## 10. Evidence controls

### Ask Atom

Submits a new question to the bounded queue.

### Cancel active

Requests cancellation through the operator, provider fabric, and model lane.
The final record may show `cancelled` after the current cancellation boundary is
reached.

### Retry selected

Creates a new attempt linked to a failed, cancelled, or interrupted request.
It does not overwrite the prior record.

### Restart model

Stops and reloads the resident model. It is available only while the operator
is idle. Use it after a model-process problem, not as a normal step between
questions.

### Shut down

Stops new work, closes the operator, and terminates the local process tree.
Use this control or close the desktop window normally. Avoid ending only the
desktop process in Task Manager because graceful journals and cleanup are
preferable.

## 11. Use permissioned hands

Permissioned hands can perform real work inside the configured workspace, but
planning and execution are separate.

### 11.1 Plan a task

1. Select `Permissioned hands`.
2. Describe a specific task.
3. Select `Plan exact actions`.
4. Wait for the proposal to reach `awaiting-permission`.
5. Review the exact manifest before choosing Approve or Deny.

Planning itself does not run a tool.

Examples of appropriately specific tasks:

```text
Create a Markdown document named notes.md in this workspace with the following three headings...

Run the test executable at this exact path with these arguments and a 120 second timeout.

Search the src directory for the literal text TODO and return at most 100 matches.

Run the simulation program over cases baseline, high-load, and recovery, with a 60 second timeout per case.
```

Avoid vague tasks such as `clean everything`, `fix my whole computer`, or
`download whatever is needed`. A precise task produces a smaller manifest that
is easier to inspect.

### 11.2 Review the exact manifest

Before approval, check:

- Capability name
- Risk level
- Exact workspace root
- Exact relative paths
- Create, replace, move, or quarantine mode
- Expected hashes for files, trees, or executables
- Exact process program and argument array
- Working directory
- Timeout
- Input sent to a process
- Exact public URL and resolved address set
- Maximum returned bytes
- Predicted effects
- Candidate normalizations or omitted unsupported fields
- Action count
- Manifest hash
- Expiry

If any field is surprising, deny the proposal and write a clearer task. Do not
approve because the prose description sounds harmless while the manifest does
something different.

### 11.3 Risk levels

| Risk | Typical examples | Review guidance |
| --- | --- | --- |
| low | Bounded directory listing | Confirm the path and limit |
| medium | Read or search text, create a directory | Confirm scope and that sensitive content is not unintentionally exposed to later planning |
| high | Write, patch, move, create a document, public web fetch | Read every path, content, hash, URL, and effect |
| critical | Quarantine, run a process, run simulations | Verify executable identity, arguments, targets, timeouts, reversibility, and downstream effects |

Risk labels guide attention. They are not automatic permission.

### 11.4 Approve exact actions

Approval applies only to the displayed manifest. The grant:

- Is created in memory.
- Is tied to the exact manifest hash and decision nonce.
- Expires.
- Can be consumed only once.
- Cannot be reused for a changed action.

The system rechecks time-sensitive facts immediately before execution. A file,
tree, executable, or network address that changed after approval causes the run
to fail closed.

### 11.5 Deny

Deny records a receipt and leaves the workspace unchanged. Use Deny whenever
the task, effect, risk, or normalization is not exactly what you intended.

### 11.6 Cancel selected task

Cancellation is available during planning and execution. For a running process
or simulation, Atom Harness attempts to terminate the complete process tree.
Already completed effects are recorded in the artifact; cancellation does not
pretend they never happened.

### 11.7 Continue from a selected result

A completed tool result may be used as untrusted context for a follow-up plan.
The next plan still requires a new exact manifest and a new approval. Tool
output cannot approve the next action, even if it contains text that says to
continue automatically.

## 12. Available hands

The current registry supports:

| Ability | What it can do |
| --- | --- |
| List workspace | Show a bounded tree without following links |
| Read text | Read one bounded UTF-8 text file |
| Search text | Search regular files for one literal query |
| Write text | Create or hash-guardedly replace a text file |
| Patch text | Replace an exact fragment under a required file hash |
| Create directory | Create a specified workspace directory |
| Move item | Move one hash-bound file or tree without overwrite |
| Quarantine item | Reversibly move a hash-bound item outside the active workspace |
| Run process | Execute one exact program and argument list without a command shell |
| Run simulation | Run bounded named cases and collect measurements |
| Create document | Create Markdown, plain text, HTML, or JSON |
| Fetch public web resource | Read one exact public URL without redirects or credentials |

There is no generic hidden shell action. A process action names one executable
and one argument array. Public web fetches do not follow redirects, do not send
credentials, and cannot access private or loopback addresses.

## 13. Protect yourself from outside influence

Prompts, source text, documents, web pages, process output, and tool output can
contain instructions. Atom Harness treats them as untrusted data, but you
should still review proposals carefully.

Red flags include:

- A document says to ignore the permission screen.
- Tool output asks for another action unrelated to your task.
- A web page asks the system to reveal tokens or environment variables.
- A proposal changes from reading to writing.
- A process is different from the one you expected.
- A path is broader than the project you intended.
- A write uses replace mode when you expected create mode.
- A quarantine or move targets a directory rather than one file.
- A public URL resolves to an unexpected address.
- A proposal expires and immediately asks you to approve a different hash.

The correct response is to deny and inspect. Do not treat urgency in generated
text as a reason to approve.

## 14. Saved data

The application is installed separately from your state.

Application files:

```text
%LOCALAPPDATA%\Programs\Lucerna Labs\Atom Harness
```

User state:

```text
%LOCALAPPDATA%\Lucerna Labs\Atom Harness
```

Important folders are:

```text
Data\Sessions\default   Answer, tool, journal, and transaction data
Data\Logs               Desktop and backend diagnostics
Data\Updates            Download and update staging
Models                   Managed model storage
WebView2                 Embedded browser profile data
settings.json            Model path and GPU-layer settings
```

Select `Open data` in the desktop toolbar to open the state location.

Each completed request or tool run is a separate committed directory. Do not
edit its JSON, HTML, knowledge snapshot, or transaction manifest if you want it
to remain verifiable.

## 15. Back up and restore

### 15.1 Back up

1. Shut down Atom Harness and wait for the model process to exit.
2. Open the data folder.
3. Copy `%LOCALAPPDATA%\Lucerna Labs\Atom Harness` to your backup location.
4. Keep the folder structure intact.
5. Record the application version used with the backup.

The model is large. You may exclude `Models` if you are willing to download or
select the exact model again. Do not exclude `Data\Sessions\default` if you want
to preserve request and permission history.

### 15.2 Restore

1. Install the same or a compatible Atom Harness version.
2. Keep the application closed.
3. Preserve the current state folder as a rollback copy.
4. Restore the backed-up state to the same per-user location.
5. Start Atom Harness.
6. Confirm history, artifacts, and model selection.

If integrity checks reject a restored artifact, keep it for audit and do not
manually change its hashes. A copied artifact is useful evidence only while its
transaction remains verifiable.

## 16. Updates

Atom Harness never downloads or installs an application update silently.

1. Select `Check for updates`.
2. If a newer release is offered, read its version and notes.
3. Approve or decline the download.
4. If approved, wait for download and verification.
5. Review the installation confirmation.
6. Approve or decline installation.
7. If approved, Atom Harness closes before the external updater replaces files.
8. The updater retains the previous installation for rollback.

The update is rejected when its app identity, platform, version, byte count, or
SHA-256 does not match the signed-off feed contract.

If the feed has not yet been published, update checking can report that no
usable feed is available. This does not affect the local runtime. Install a
newer verified package manually when one is distributed.

## 17. Privacy and networking

Normal evidence questions stay on the local Atom and Qwen runtime. The operator
does not activate cloud model routing.

Network activity can occur when you explicitly choose one of these actions:

- Download the model.
- Check for an application update.
- Download an approved update.
- Approve a `web.fetch` tool manifest for one public URL.

The permissioned process adapter does not forward provider environment secrets.
The web adapter sends no credentials and follows no redirects. Still, do not
place passwords, access tokens, or private keys into prompts, documents, or
files merely because the system is local.

## 18. Troubleshooting

### The application says the installed runtime is invalid

One or more installed files differ from the release manifest. Reinstall from a
verified MSI or re-extract the complete verified ZIP. Do not copy a missing DLL
from another application.

### The model is not found

Choose Download or Browse when prompted. If you moved the model, browse to the
new exact file. The old saved path will not be silently redirected.

### The model is rejected

The file has the wrong byte count or SHA-256. It may be incomplete, corrupted,
or a different conversion. Obtain the exact admitted Q8_0 artifact.

### Startup appears stuck

The first model load can take time. Watch the status bar. If startup exceeds the
declared timeout or reports failure, open `Data\Logs` and preserve the newest
diagnostic record before restarting.

### The window is blank or WebView2 fails

Install or repair Microsoft Edge WebView2 Runtime. If the runtime is installed,
close Atom Harness, back up the state directory, and investigate the `WebView2`
profile and logs. Do not delete the entire Atom Harness data directory as a
first step.

### A request stays queued

The resident lane has one parallel slot. Wait for the active request, cancel it,
or inspect whether the model process failed. The queue is intentionally
bounded.

### A request failed or was interrupted

Select it and use `Retry selected`. The retry becomes a new attempt linked to
the original. The original record remains for audit.

### The artifact panel says it is unavailable

Select the completed item again. The UI retries short commit conflicts. If it
continues to fail, preserve the session data and logs. Do not edit the artifact
or transaction manifest.

### Restart model is disabled

A request is active or queued. Cancel or wait for current work. Model restart is
allowed only while idle.

### A tool proposal has no actions

The planned candidate did not reduce to the registered capabilities. Make the
task smaller and more exact. Do not ask for a generic shell or hidden autonomy.

### Approval reports a conflict

The manifest, nonce, expiry, or a checked file, executable, or address changed.
Create a fresh plan and review it again. The system intentionally does not reuse
the old approval.

### A process or simulation timed out

The tool attempted to terminate the process tree and records the bounded result.
Inspect the tool artifact. If a longer run is appropriate, request a new plan
with a clear timeout and approve that new manifest.

### A web fetch failed

The URL may redirect, resolve to a private address, exceed the size limit, fail
TLS validation, or exceed the timeout. Use an exact public final URL and plan a
new fetch.

### Check for updates fails

The update feed may not be published, the network may be unavailable, or the
feed may fail identity validation. Continue using the installed local version
or install a separately verified newer package.

### Atom Harness closed but a model process remains

Wait briefly for graceful cleanup. If a process persists, record its name and
command line, preserve logs, and then end that exact process tree. Repeated
leaks are a defect and should be reported with the application version and log
time.

## 19. Uninstall

Use Windows Settings, Apps, Installed apps, Atom Harness, Uninstall.

The application and per-user state are separate. Uninstalling the MSI may leave
your sessions, logs, settings, model, and update staging under:

```text
%LOCALAPPDATA%\Lucerna Labs\Atom Harness
```

Back up anything you want to keep. Remove the remaining state only when you
intend to delete your local history and model. A 4.28 GB model is expensive to
download again.

For a portable installation, close Atom Harness and remove only the exact
folder into which you extracted the portable package. The separate per-user
state folder remains until you intentionally remove it.

## 20. Current limits

The current release has these honest limits:

- The knowledge pack is a 45-claim foundation across 15 domains, not an
  encyclopedia.
- Static reference knowledge is not automatically current.
- The Qwen model is used for language and proposals, not as a trusted fact
  store.
- The operator uses one resident generation slot.
- Cloud language routing is disabled in Operator V6.
- Tool execution is experimental and must remain attended.
- Documents are Markdown, plain text, HTML, or JSON. There is no general office
  document editor in the capability registry.
- Public web access is a single approved fetch, not unrestricted browsing.
- Tool outputs remain untrusted and cannot automatically trigger follow-up
  work.
- Prompt-injection defenses are tested against known classes, not proven
  universal.
- Medical material is educational reference evidence, not diagnosis or
  treatment advice.
- The private update feed may not be published even when a local release has
  been certified.

The developer-facing list of planned work, release gates, and optional
extensions is maintained in `ATOM_HARNESS_TODO.md`. Current safety boundaries
are not missing features and remain in force unless the experiment is
explicitly redesigned.

## 21. Glossary

| Term | Meaning |
| --- | --- |
| Atom | The project's evidence, memory, graph, policy, and abstention authority |
| Language membrane | The local Qwen model used to understand wording and render grounded responses without owning facts |
| Causal lane | Evidence records about causal relationships, transitions, and interventions |
| Multidisciplinary lane | Reference claims spanning science, mathematics, linguistics, literature, writing, and research practice |
| Wiki graph | The typed graph of claims, domains, sources, and relationships |
| Graph RAG | Retrieval that follows graph structure and produces a bounded evidence packet |
| Artifact | The real saved evidence or tool result produced by a committed transaction |
| Side view | The right-side frame that renders the real artifact |
| Manifest | The exact canonical set of proposed actions, arguments, risks, effects, and hashes |
| Permission grant | A one-time, expiring, memory-only approval bound to one exact manifest |
| Untrusted result | Data that can be displayed or reviewed but cannot grant permission or become evidence by itself |
| Abstention | A deliberate refusal to answer when Atom lacks sufficient evidence |
| Quarantine | A reversible move of an item out of the active workspace |
| Transaction | A staged, hash-manifested, atomically published run directory |

## 22. What to include in a problem report

Provide:

1. Atom Harness version.
2. MSI or portable installation.
3. Windows version and x64 confirmation.
4. WebView2 Runtime version if the window is affected.
5. The exact status message.
6. The approximate time of the failure.
7. Whether the model had previously loaded successfully.
8. Whether the issue occurred in Evidence or Permissioned hands.
9. Request or proposal ID when visible.
10. The relevant files from `Data\Logs`.
11. The affected committed run directory, kept unchanged.
12. Whether a retry, idle model restart, or full application restart changed the
    result.

Do not send model files, private project content, passwords, tokens, or unrelated
session data. Preserve originals and share only the smallest material needed to
reproduce the defect.
