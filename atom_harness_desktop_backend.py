"""Frozen desktop entrypoint for Atom Harness Phase 6 Operator V5."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ATOM_HARNESS_DESKTOP_BACKEND_RUNTIME = "atom-harness-desktop-backend-v6"
ATOM_HARNESS_BUNDLED_CAUSAL_RUNTIME = "atom-harness-bundled-causal-memory-v1"


def _runtime_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parent


def _bind_bundled_causal_memory(runtime_root: Path) -> None:
    if getattr(sys, "_MEIPASS", None) is None:
        return

    binary = runtime_root / "rust" / "atom-causal-memory.exe"
    if not binary.is_file():
        raise FileNotFoundError(
            "The bundled Atom causal-memory runtime is missing.",
            binary,
        )

    import atom_causal_memory

    def bundled_release_binary() -> Path:
        return binary

    atom_causal_memory.RELEASE_BINARY = binary
    atom_causal_memory.build_release_binary = bundled_release_binary

    import atom_harness_knowledge

    atom_harness_knowledge.build_release_binary = bundled_release_binary


def main() -> int:
    runtime_root = _runtime_root()
    os.chdir(runtime_root)
    _bind_bundled_causal_memory(runtime_root)
    from atom_harness_operator_server import main as operator_main

    return operator_main()


if __name__ == "__main__":
    raise SystemExit(main())
