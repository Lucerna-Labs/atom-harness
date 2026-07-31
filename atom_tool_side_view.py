"""User-visible side view for real permissioned tool artifacts."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash


ATOM_TOOL_SIDE_VIEW_RUNTIME = "atom-permissioned-hands-side-view-v1"
ATOM_TOOL_ARTIFACT_BINDING = "render_atom_tool_artifact"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pretty(value: Any) -> str:
    return html.escape(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        quote=False,
    )


def render_atom_tool_artifact(
    artifact: Mapping[str, Any],
    workflow: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
) -> str:
    """Render the exact committed tool artifact without executable content."""

    if artifact.get("artifact_hash") != canonical_hash(
        {key: value for key, value in artifact.items() if key != "artifact_hash"}
    ):
        raise ValueError("tool artifact hash is invalid")
    if workflow.get("artifact_hash") != artifact["artifact_hash"]:
        raise ValueError("tool workflow is not bound to the artifact")
    if workflow.get("graph_knowledge_hash") != graph_manifest.get("knowledge_hash"):
        raise ValueError("tool workflow is not bound to the wiki graph")

    permission = artifact["permission"]
    manifest = artifact["execution_manifest"]
    actions = manifest["actions"]
    results_by_id = {
        str(item["action_id"]): item for item in artifact.get("results", [])
    }
    action_rows = []
    for action in actions:
        result = results_by_id.get(str(action["action_id"]))
        status = result.get("status", "not-run") if result else "not-run"
        action_rows.append(
            '<article class="action">'
            f'<div class="action-head"><strong>{_escape(action["action_id"])}</strong>'
            f'<span class="risk risk-{_escape(action["risk"])}">{_escape(action["risk"])}</span>'
            f'<span class="status">{_escape(status)}</span></div>'
            f"<h3>{_escape(action['capability'])}</h3>"
            f"<p>{_escape(action['rationale'])}</p>"
            f"<details><summary>Exact approved arguments</summary><pre>{_pretty(action['arguments'])}</pre></details>"
            f"<details><summary>Declared effects</summary><pre>{_pretty(action['effects'])}</pre></details>"
            + (
                f"<details open><summary>Quarantined result</summary><pre>{_pretty(result)}</pre></details>"
                if result is not None
                else ""
            )
            + "</article>"
        )

    signals = artifact.get("injection_signals", [])
    signal_markup = (
        "".join(f"<li>{_escape(item)}</li>" for item in signals)
        if signals
        else "<li>None detected. Detection is observability, not the security boundary.</li>"
    )
    outcome_class = "passed" if artifact.get("passed") else "failed"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Atom Permissioned Hands Artifact</title>
  <style>
    :root {{ color-scheme: dark; --ink:#f5f4ee; --muted:#aaa99f; --line:#343b30; --panel:#171a15; --green:#9fe870; --amber:#f1c75b; --red:#ff7d7d; --blue:#79b8ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:24px; background:#0b0d0a; color:var(--ink); font:14px/1.5 Inter,system-ui,sans-serif; }}
    h1,h2,h3,p {{ margin-top:0; }}
    header,.panel,.action {{ margin-bottom:14px; padding:16px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }}
    header {{ border-color:#456a35; }}
    .eyebrow {{ color:var(--green); font-size:11px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }}
    .fact {{ padding:10px; border:1px solid var(--line); border-radius:8px; background:#10120f; }}
    .fact span {{ display:block; color:var(--muted); font-size:10px; text-transform:uppercase; }}
    .fact strong {{ display:block; overflow-wrap:anywhere; margin-top:4px; }}
    .passed {{ color:var(--green); }} .failed {{ color:var(--red); }}
    .notice {{ border-left:4px solid var(--amber); }}
    .action-head {{ display:flex; gap:8px; align-items:center; }}
    .action-head .status {{ margin-left:auto; color:var(--green); }}
    .risk {{ padding:2px 7px; border-radius:999px; background:#282d24; font-size:10px; text-transform:uppercase; }}
    .risk-high,.risk-critical {{ color:var(--amber); }} .risk-critical {{ border:1px solid #764c30; }}
    details {{ margin-top:9px; }} summary {{ cursor:pointer; color:var(--blue); }}
    pre {{ max-height:360px; overflow:auto; padding:12px; border:1px solid var(--line); border-radius:8px; background:#0d0f0c; color:#d8d9d2; white-space:pre-wrap; overflow-wrap:anywhere; }}
    code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body data-runtime="{ATOM_TOOL_SIDE_VIEW_RUNTIME}" data-binding="{ATOM_TOOL_ARTIFACT_BINDING}">
  <header>
    <div class="eyebrow">Real permissioned execution artifact</div>
    <h1>Atom Hands Experiment</h1>
    <p>{_escape(artifact["task"])}</p>
    <strong class="{outcome_class}">{_escape(artifact["status"])}</strong>
  </header>
  <section class="panel notice">
    <h2>Authority boundary</h2>
    <p>The language model proposed actions. The trusted interface supplied a one-time permission grant bound to the exact manifest below. Every result remains untrusted data and cannot grant another permission.</p>
  </section>
  <section class="panel">
    <h2>Permission receipt</h2>
    <div class="grid">
      <div class="fact"><span>Decision</span><strong>{_escape(permission["decision"])}</strong></div>
      <div class="fact"><span>Authority</span><strong>{_escape(permission["authority"])}</strong></div>
      <div class="fact"><span>Manifest</span><strong>{_escape(permission["manifest_hash"])}</strong></div>
      <div class="fact"><span>Grant</span><strong>{_escape(permission["grant_hash"])}</strong></div>
      <div class="fact"><span>Workspace</span><strong>{_escape(manifest["workspace_root"])}</strong></div>
      <div class="fact"><span>Transaction</span><strong>{_escape(artifact["transaction"]["transaction_id"])}</strong></div>
    </div>
  </section>
  <section class="panel">
    <h2>Outside influence signals</h2>
    <ul>{signal_markup}</ul>
  </section>
  <section>
    <h2>Exact actions and results</h2>
    {"".join(action_rows)}
  </section>
  <section class="panel">
    <h2>Atom integrity</h2>
    <div class="grid">
      <div class="fact"><span>Wiki runtime</span><strong>{_escape(artifact["knowledge"]["wiki_runtime"])}</strong></div>
      <div class="fact"><span>RAG runtime</span><strong>{_escape(artifact["knowledge"]["rag_runtime"])}</strong></div>
      <div class="fact"><span>Graph hash</span><strong>{_escape(artifact["knowledge"]["graph_knowledge_hash"])}</strong></div>
      <div class="fact"><span>Memory unchanged</span><strong>{_escape(artifact["memory"]["unchanged"])}</strong></div>
    </div>
  </section>
  <section class="panel">
    <h2>Workflow binding</h2>
    <pre>{_pretty(workflow)}</pre>
  </section>
</body>
</html>"""
