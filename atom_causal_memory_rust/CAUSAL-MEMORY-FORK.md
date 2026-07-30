# Causal Memory Fork

This local workspace was fetched from:

`https://github.com/Lucerna-Labs/atom-db`

The imported base is commit:

`b9a199bf994ffaa6811c59b7fa0f27dab41c7949`

The `origin` fetch URL remains available for comparing upstream fixes. Its push
URL is deliberately set to `DISABLED`, preventing this project from changing
the Lucerna Labs repository.

The existing `atom-db` library and `atom-retrieval-field` crate retain their
upstream behavior. The original command-line target is registered as the
separate `atom-db-cli` package so each compiled Rust crate remains below the
workspace's 4,000-source-line ceiling. The CLI still uses the same source and
commands.

This fork adds the focused `atom-causal-memory` crate. It uses the storage
substrate but does not use the lexical passage retriever. Its persisted objects
are Primitive Forge causal glyphs, causal-world experience records, and exact
structural motifs.

Experience batches append immutable observations and consolidated laws while
retaining the SHA-256 identity of their source artifact. The inventory exposes
the exact role-value features needed to reconstruct the runtime wiki graph and
recover live prediction provenance. Audit features remain forbidden at the
recall membrane.

The `observe-experience-once` command gives live outcomes a durable SHA-256
idempotency key. The key, decision, outcome event, and all motif adjustments
commit in one atomic cell. An exact replay returns without changing the store;
reusing the key for a conflicting decision fails closed.
