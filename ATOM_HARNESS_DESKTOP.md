# Atom Harness Desktop Phase 6

Atom Harness Desktop is the installed Windows form of the certified Atom
Harness Operator V5. The desktop application is deliberately a shell, not a
second reasoning system. Atom still owns evidence, the wiki graph, graph RAG,
causal memory, policy, abstention, and the committed artifact. The selected
Qwen model supplies language only.

## Install and start

The Phase 6 release has two Windows x64 formats:

- `Atom-Harness-6.0.3-windows-x64.msi` installs per user without administrator
  rights.
- `Atom-Harness-6.0.3-windows-x64.zip` is the portable and update payload.

The MSI installs to:

```text
%LOCALAPPDATA%\Programs\Lucerna Labs\Atom Harness
```

It creates `Atom Harness` shortcuts on the Desktop and in the Start Menu.
Start the application from either shortcut. Only one desktop instance can run
at a time.

On first start, the application searches the configured and known local model
locations for the exact certified Qwen artifact. If it cannot find the model,
it explains the required 4.28 GB download and asks for permission. A download
is staged outside the application directory and admitted only after both the
exact byte count and SHA-256 match the model contract.

## Everyday operation

The left side of the window contains Evidence and Permissioned hands tabs,
request status, retry and cancel controls, exact permission controls, and model
lifecycle information. The right side is the real artifact side view produced
by the runtime. It displays either a committed evidence artifact or a committed
tool artifact. It is not a preview assembled by the desktop shell.

No tool executes merely because the model proposed it. The user sees the exact
manifest, including paths, executable, arguments, effects, risk, workspace,
hash, and expiry, then approves or denies it. Approval is single-use. Tool
output is displayed as untrusted and cannot approve more work.

The application starts a private loopback backend and a private loopback
`llama-server`. Neither service is exposed to the network. Cloud routing
remains unavailable in the desktop product. Closing the window first requests
a graceful runtime shutdown. A Windows job object then guarantees cleanup of
the backend and model process tree if graceful shutdown cannot finish.

Session history is durable. An interrupted or failed request remains visible
after restart and can be retried. A completed request and its bound side-view
artifact also reappear after restart.

## Update behavior

Atom Harness never downloads or installs an update silently.

1. The user selects the update control.
2. The application retrieves the HTTPS release feed only on request.
3. If a newer version exists, the application shows its version and notes.
4. The user explicitly approves the download.
5. The complete ZIP is checked against the feed byte count and SHA-256.
6. The user explicitly approves installation.
7. The updater stages and verifies the full internal release manifest outside
   the install directory.
8. The updater waits for Atom Harness to exit before replacing files.
9. The previous installation is retained as a rollback directory.
10. The new application is started only after replacement succeeds.

A private repository does not provide an unauthenticated update feed. The
local updater implementation is complete and verified, but Lucerna Labs must
publish the release ZIP and feed at an authorized HTTPS endpoint before the
in-application update check can distribute a future version.

## Build a release

Prerequisites are Python 3.13, Rust 1.96.0, the exact .NET SDK 9.0.305, WiX
Toolset 3.14, and the certified llama.cpp Windows runtime. `global.json`
disables SDK roll-forward so implicit framework packages cannot silently
rewrite the checked-in lock graph. The build script performs policy
validation, locked dependency restoration, Rust and .NET builds, .NET tests,
the frozen Python backend build, full-file manifest creation, ZIP creation,
and per-user MSI creation.

```powershell
.\scripts\build_atom_harness_desktop.ps1
```

Build outputs are written under `local-results` and remain machine-local. The
source repository records their hashes in
`atom-harness-desktop-release-evidence.json`.

To verify an installed layout without starting the interactive runtime:

```powershell
& "$env:LOCALAPPDATA\Programs\Lucerna Labs\Atom Harness\AtomHarness.Desktop.exe" `
  --verify-install "$env:TEMP\atom-harness-install-verification.json"
```

This checks every file in the installed release manifest, the update and model
contracts, the installed WebView2 runtime, and the declared authority runtime.

## Failure recovery

- If model verification fails, no model is admitted. Select or download the
  exact certified artifact.
- If backend startup fails, the desktop records a hash-identified diagnostic
  without copying prompts, answers, model data, or secrets into the log.
- If a request fails, select its history row and use the retry control.
- If an update fails before replacement, the installed application is
  unchanged.
- If replacement fails after the old directory moves, the updater restores the
  previous directory.
- If the application is closed during work, the job object cleans up all child
  processes and the persistent journal drives recovery at the next start.

## Authority boundary

Phase 6 retains installation, lifecycle supervision, verified model
provisioning, safe updates, and the native window. It gives the model a
proposal vocabulary for registered tools, not an execution handle or
permission authority. The model still has no evidence authority, Atom DB write
access, cloud provider access, or permission to override abstention. Wiki graph
and graph RAG execution remain mandatory, and every produced artifact remains
visibly bound into the right-side view.
