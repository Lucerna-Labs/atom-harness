"""Thin Svelte and TypeScript projection generated from Atom typed IR."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash
from atom_language import AtomProgram


ATOM_FRONTEND_TARGET_RUNTIME = "atom-thin-svelte-projection-v1"
ATOM_FRONTEND_VALIDATOR_RUNTIME = "atom-svelte-typescript-validation-v1"


def compile_atom_to_frontend(program: AtomProgram) -> dict[str, str]:
    """Generate a presentation-only frontend from one Atom program."""

    program.validate()
    manifest_json = json.dumps(
        program.manifest(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )
    platform_typescript = f"""export const atomPlatform = {manifest_json} as const;

export type AtomPlatform = typeof atomPlatform;

export function summarizePlatform(): string {{
  return `${{atomPlatform.name}}:${{atomPlatform.program_hash}}`;
}}
"""
    bridge_typescript = """export interface AtomRequest {
  readonly action: string;
  readonly payload: Readonly<Record<string, unknown>>;
}

export interface AtomExecutionResult {
  readonly status: "derived" | "unknown" | "contradicted";
  readonly value: unknown;
}

export interface AtomRustBridge {
  execute(request: AtomRequest): Promise<AtomExecutionResult>;
}
"""
    component = """<script lang="ts">
  import { atomPlatform } from "./atom-platform";
  import type { AtomExecutionResult, AtomRustBridge } from "./bridge";

  let { bridge }: { bridge: AtomRustBridge } = $props();
  let execution = $state<AtomExecutionResult | null>(null);
  let running = $state(false);

  async function runProjection(): Promise<void> {
    running = true;
    try {
      execution = await bridge.execute({
        action: "project",
        payload: {
          programHash: atomPlatform.program_hash,
          candidates: [
            { value: "reject", support: 0.2 },
            { value: "accept", support: 0.9 }
          ]
        }
      });
    } finally {
      running = false;
    }
  }
</script>

<main aria-labelledby="atom-title">
  <header>
    <p>Atom causal platform</p>
    <h1 id="atom-title">{atomPlatform.name}</h1>
    <code>{atomPlatform.program_hash}</code>
  </header>

  <section aria-labelledby="capabilities-title">
    <h2 id="capabilities-title">Capabilities</h2>
    <ul>
      {#each atomPlatform.capabilities as binding}
        <li>
          <strong>{binding.capability}</strong>
          <span>{binding.primitives.join(" + ")}</span>
        </li>
      {/each}
    </ul>
  </section>

  <section aria-labelledby="layers-title">
    <h2 id="layers-title">Spiderweb layers</h2>
    <ol>
      {#each atomPlatform.layers as layer}
        <li>
          <strong>{layer.name}</strong>
          <span>{layer.primitives.join(" + ") || "root substrate"}</span>
        </li>
      {/each}
    </ol>
  </section>

  <button type="button" onclick={runProjection} disabled={running}>
    {running ? "Running Atom projection" : "Run through Rust"}
  </button>

  {#if execution}
    <output aria-live="polite">
      {execution.status}: {JSON.stringify(execution.value)}
    </output>
  {/if}
</main>

<style>
  :global(body) {
    margin: 0;
    background: #081018;
    color: #e5f2f7;
    font-family: Inter, system-ui, sans-serif;
  }

  main {
    max-width: 72rem;
    margin: 0 auto;
    padding: 2rem;
  }

  header,
  section {
    padding: 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid #29465b;
    border-radius: 0.75rem;
    background: #0f1d28;
  }

  ul,
  ol {
    display: grid;
    gap: 0.65rem;
    padding-left: 1.25rem;
  }

  li {
    display: grid;
    grid-template-columns: minmax(12rem, 1fr) 2fr;
    gap: 1rem;
  }

  span,
  code {
    color: #93cbbb;
  }

  button {
    padding: 0.8rem 1rem;
    border: 0;
    border-radius: 0.5rem;
    background: #80e0c2;
    color: #07110f;
    font-weight: 700;
    cursor: pointer;
  }
</style>
"""
    return {
        "src/atom-platform.ts": platform_typescript,
        "src/bridge.ts": bridge_typescript,
        "src/AtomPlatform.svelte": component,
    }


def write_frontend_project(
    program: AtomProgram,
    output_dir: Path,
) -> dict[str, Any]:
    files = compile_atom_to_frontend(program)
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, source in files.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
    core = {
        "runtime": ATOM_FRONTEND_TARGET_RUNTIME,
        "program_hash": program.manifest()["program_hash"],
        "files": {
            relative: canonical_hash({"source": source})
            for relative, source in sorted(files.items())
        },
    }
    return {**core, "project_hash": canonical_hash(core)}


def validate_frontend_artifacts(
    files: Mapping[str, str],
    validator_dir: Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    expected = {
        "src/atom-platform.ts",
        "src/bridge.ts",
        "src/AtomPlatform.svelte",
    }
    if set(files) != expected:
        raise ValueError("frontend artifact fields are invalid")
    validator_dir = validator_dir.resolve()
    validator = validator_dir / "validate.mjs"
    if not validator.is_file():
        raise FileNotFoundError("Svelte validator script is missing")
    with tempfile.TemporaryDirectory(prefix="atom-frontend-target-") as temporary:
        root = Path(temporary)
        for relative, source in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8", newline="\n")
        process = subprocess.run(
            [
                "node",
                str(validator),
                str(root / "src/AtomPlatform.svelte"),
                str(root / "src/atom-platform.ts"),
                str(root / "src/bridge.ts"),
            ],
            cwd=validator_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    parsed: Mapping[str, Any] | None = None
    if process.returncode == 0:
        try:
            candidate = json.loads(process.stdout)
            if isinstance(candidate, Mapping):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = None
    core = {
        "runtime": ATOM_FRONTEND_VALIDATOR_RUNTIME,
        "return_code": process.returncode,
        "passed": bool(
            process.returncode == 0
            and parsed is not None
            and parsed.get("svelteCompiled") is True
            and parsed.get("typescriptPassed") is True
        ),
        "result": dict(parsed) if parsed is not None else None,
        "stderr": process.stderr.strip(),
    }
    return {**core, "validation_hash": canonical_hash(core)}


def frontend_target_self_test(
    program: AtomProgram,
    validator_dir: Path,
) -> dict[str, bool]:
    files = compile_atom_to_frontend(program)
    validation = validate_frontend_artifacts(files, validator_dir)
    component = files["src/AtomPlatform.svelte"]
    return {
        "same_program_hash_is_rendered": (
            program.manifest()["program_hash"]
            in files["src/atom-platform.ts"]
        ),
        "frontend_has_no_platform_logic": "pub fn execute" not in component,
        "rust_bridge_is_explicit": "AtomRustBridge" in component,
        "user_control_is_wired": "onclick={runProjection}" in component,
        "svelte_and_typescript_validate": validation["passed"],
    }
