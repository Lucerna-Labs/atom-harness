# Local Fork Boundary

This checkout is the local Atom language-harness fork:

```text
C:\Projects\atom-harness
```

It was created from the active source and required causal artifacts in
`C:\Projects\atom lora`. The source checkout was not modified.

This fork has its own local Git history and intentionally has no GitHub remote.
Do not add, push, or publish a remote unless the operator explicitly changes
that boundary.

Large Kaggle results, historical runtime outputs, backups, dependency caches,
model weights, and Rust build products were not duplicated. The original
checkout retains those artifacts. This fork keeps the compact causal-world and
Primitive Forge artifacts required to bootstrap the runtime wiki, graph RAG,
and Atom DB evidence store.
