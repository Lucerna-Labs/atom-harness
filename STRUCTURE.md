# Structure

Where things live in the repo, and why. Read this when you can't find
something or when you're about to add something new and want to put it in
the right place.

## Top level

```
atom-harness/
├── README.md                  # orientation — read first
├── START_HERE.md              # one-page map by role, time budget, depth
├── VERSIONS.md                # every Kaggle / architecture version, one line each
├── STRUCTURE.md               # this file
│
├── docs/
│   ├── roles/                 # what you need based on who you are
│   ├── systems/               # what you need based on what you're studying
│   ├── versions/
│   │   └── kaggle/            # measured evidence per accelerator run
│   └── notes/                 # chronological engineering record
│
├── contracts/                 # machine-readable architecture contracts
│
├── atom_*.py                  # Python source (entry point: atom_harness_runtime.py)
├── atom_causal_memory_rust/   # generated Rust crate
├── desktop/                   # Windows desktop shell
│
├── scripts/                   # Python helper scripts (build, verify, certify)
├── tests/                     # Python tests
├── knowledge_packs/           # data
├── tooling/                   # dev tooling
│
├── causal_world_outputs/       # runtime outputs (causal-world experiments)
├── primitive_forge_outputs/    # runtime outputs (primitive-forge)
│
├── retired/                   # historical, audit-only
│
├── run-atom-harness*.ps1       # launchers
├── START-ATOM-HARNESS-OPERATOR.cmd  # double-click launcher
│
├── requirements-*.txt          # Python dependencies
├── rust-toolchain.toml
├── .env.example
├── .gitignore, .gitattributes
└── .github/                   # CI workflows
```

## Why the Python source is at the root

Every Kaggle script imports Atom modules as flat top-level packages:
`from atom_causal_world_schema import canonical_hash`. If we ever
moved the source into `src/atom_harness/...`, every Kaggle kernel and
local replay would need its sys.path adjusted.

**Rule:** Python source stays at the root. `tests/`, `scripts/`, and
`desktop/` are separate Python projects with their own internal structure,
and that's fine.

## Why docs/ is split into roles, systems, versions, notes

- **`docs/roles/`** — answers the question *"who are you?"* User, developer,
  operator, TODO tracker, desktop-specific. Read the one that matches.
- **`docs/systems/`** — answers *"what subsystem are you reading about?"*
  Causal memory, causal live, language harness, primitive forge,
  generative English, universal knowledge.
- **`docs/versions/kaggle/`** — answers *"what was the measured result of
  this particular run?"* Every accelerator run has its own evidence file
  with timings, hashes, audit checks, and replay verification.
- **`docs/notes/`** — answers *"what was the engineering story?"*
  Chronological journal, fork lineage, GitHub publication policy.

The split is by reader's question, not by document topic. If you have a
new document, decide which question it answers before deciding the file
name.

## Why contracts/ is its own directory

The JSON files in `contracts/` are the *machine-readable* architecture.
They are what code reads at runtime, what verifiers hash against, and
what new components must satisfy. They are not documentation in the
narrative sense — they are *interfaces*.

**Rule:** anything in `contracts/` should be a JSON file with a `schema`
field at the top. Anything in `docs/` should be a Markdown file a human
reads. Don't mix.

## Why retired/ is at the top level

It used to live inside other directories and was always hard to find.
Moving it to the top level makes "this is preserved but not active"
visible at a glance. The README, the user guide, and the developer guide
all explicitly point into `retired/` for the historical record.

## Adding new things

| You want to add... | Put it in... |
|---|---|
| A new Python source file | the root, named `atom_<thing>.py` |
| A new test | `tests/test_<thing>.py` |
| A new helper script | `scripts/<verb>_<thing>.py` |
| A new role doc (operator, user, etc.) | `docs/roles/ATOM_HARNESS_<NAME>.md` |
| A new subsystem doc (causal, language, etc.) | `docs/systems/ATOM_<NAME>.md` |
| A new architecture contract | `contracts/<thing>.json` with a `schema` field |
| A new accelerator run | `docs/versions/kaggle/KAGGLE_<NAME>_EVIDENCE.md` |
| A new engineering-journal entry | append to `docs/notes/DEVELOPER_NOTES.md` |
| A new generated Rust crate | `atom_causal_memory_rust/` (or a sibling crate) |
| A new desktop component | `desktop/` |
| A new data file | `knowledge_packs/` (or a new subdir there) |
| Something you want to keep but not run | `retired/<thing>/` |

If the table doesn't fit, that's a sign either (a) the thing shouldn't be
in this repo, or (b) we need a new directory and the right move is to
open an issue or add an entry to `docs/roles/ATOM_HARNESS_TODO.md`
proposing it.
