# Start Here

A one-page map of the Atom Harness repo by role, time budget, and depth.
If you have read this and the [`README.md`](README.md), you have everything
you need to navigate the rest.

## If you have 5 minutes

1. Read the top of [`README.md`](README.md) — the *Active direction*
   paragraph tells you what this is and what's currently running.
2. Open [`contracts/atom-language-model.json`](contracts/atom-language-model.json)
   and skim the `runtime_policy` block. The LLM is a replaceable membrane;
   that JSON is the contract.
3. Skim [`VERSIONS.md`](VERSIONS.md). The Kaggle version table is the
   trail of measured evidence behind every claim in the codebase.

## If you have 1 hour

1. Read [`docs/roles/ATOM_HARNESS_USER_GUIDE.md`](docs/roles/ATOM_HARNESS_USER_GUIDE.md) if
   you want to *use* Atom Harness.
2. Read [`docs/roles/ATOM_HARNESS_DEVELOPER_GUIDE.md`](docs/roles/ATOM_HARNESS_DEVELOPER_GUIDE.md)
   if you want to *rebuild* Atom Harness from scratch on a clean machine.
3. Read [`docs/notes/DEVELOPER_NOTES.md`](docs/notes/DEVELOPER_NOTES.md) for
   the chronological engineering record.

## If you want to build from scratch

[`docs/roles/ATOM_HARNESS_DEVELOPER_GUIDE.md`](docs/roles/ATOM_HARNESS_DEVELOPER_GUIDE.md)
is the canonical reconstruction guide. It covers architecture, authority
boundaries, every active module, exact prerequisites, model admission, source
launch, API, knowledge schemas, capability security, transactions, desktop
packaging, CI, and a clean-machine acceptance checklist.

The first thing the guide will tell you to do is the clean-machine
acceptance checklist. **Run that before anything else.** If it passes, the
runtime is intact. If it fails, something is wrong with the local environment
and no amount of reading will fix it.

## If you are auditing

The audit chain is:

1. [`docs/versions/kaggle/`](docs/versions/kaggle/) — every accelerator run
   has its own evidence file with measured timings, hashes, audit checks,
   and the per-version replay verification.
2. [`docs/notes/DEVELOPER_NOTES.md`](docs/notes/DEVELOPER_NOTES.md) — the
   engineering journal. Certification runs and their pass/fail are recorded
   here.
3. [`contracts/`](contracts/) — the machine-readable architecture contracts.
   The `runtime_policy` and `certification` blocks in
   [`contracts/atom-language-model.json`](contracts/atom-language-model.json)
   are the enforceable source of truth.

The hash-bound evidence is regenerable: every published run is reproducible
on a different OS as long as the same model and source are used. The
verifier scripts (`scripts/verify_kaggle_*.py`) do the byte-level replay.

## If you are extending

[`docs/roles/ATOM_HARNESS_TODO.md`](docs/roles/ATOM_HARNESS_TODO.md) is the
canonical unfinished list. Every item has a priority, dependency,
acceptance condition, and the boundary it must not weaken. Read it before
adding anything. If your idea isn't on the list and isn't blocked by a
listed boundary, add it to the list before adding the code.

The architecture contracts in [`contracts/`](contracts/) are what you extend
*against*, not what you extend. The language-membrane contract is the one
new LLM providers must satisfy. The causal-world contract is what
data-shape additions have to match. The capability registry is the gate
for new tools.

## Where things live

See [`STRUCTURE.md`](STRUCTURE.md) for the full layout. The short version:

- Python source is at the root (`atom_*.py`) — that is intentional, do not
  move it into a `src/` package. Every Kaggle script imports from the root.
- Role docs (user, developer, operator, TODO, desktop) are in
  [`docs/roles/`](docs/roles/).
- System docs (causal-memory, causal-live, language, primitive-forge,
  generative-english, universal-knowledge) are in [`docs/systems/`](docs/systems/).
- Every Kaggle version's evidence is in
  [`docs/versions/kaggle/`](docs/versions/kaggle/).
- Engineering notes are in [`docs/notes/`](docs/notes/).
- Every machine-readable architecture contract is in
  [`contracts/`](contracts/).
- The Python entry point is `atom_harness_runtime.py`. The launchers are
  `run-atom-harness.ps1` (operator), `run-atom-harness-session.ps1`
  (batch), and `START-ATOM-HARNESS-OPERATOR.cmd` (double-click).

## What the repo is *not*

This is not a chatbot framework, not a fine-tuning toolkit, not an
instruction-tuning dataset, and not a general-purpose agent. Atom Harness
is a *per-user Windows shell* that pairs a small local language model with
a typed causal-memory runtime, a multidisciplinary knowledge fabric, and
a permissioned tool layer. The model renders language and proposes tool
calls. Atom owns facts, memory, evidence, authority, and abstention. The
human operator is the sole execution-permission authority.

If a capability you want is not in the repo, the right move is *not* to
add a generic tool. The right move is to write an Atom capability that
satisfies the existing contracts, then register it in the capability
registry. The architecture is closed against ad-hoc additions on purpose.
