# Fork and Publication Boundary

This checkout is the local Atom Language Harness V2 fork:

```text
C:\Projects\atom-harness
```

It was created from the active source and required causal artifacts in
`C:\Projects\atom lora`. The source checkout was not modified.

This fork has its own Git history. On 2026-07-30, the operator explicitly
authorized publication as the private Lucerna Labs repository:

```text
https://github.com/Lucerna-Labs/atom-harness
```

That repository is the harness remote. Do not push these commits into
`atom-lora`, `atom-vibe-coder`, or another related repository without a
separate operator decision and a reviewed synchronization diff.

Large Kaggle results, historical runtime outputs, backups, dependency caches,
model weights, and Rust build products were not duplicated. The original
checkout retains those artifacts. This fork keeps the compact causal-world and
Primitive Forge artifacts required to bootstrap the runtime wiki, graph RAG,
and Atom DB evidence store.

See `DEVELOPER_NOTES.md` for the architecture, trust boundary, module map,
provider contract, transaction and recovery behavior, operation, verification,
and handoff checklist. The active runtime is `language-harness-v2`.
