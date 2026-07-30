"""User-visible side view bound to a real Atom harness artifact."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash
from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_harness_runtime import (
    ATOM_LANGUAGE_HARNESS_RUNTIME,
    ATOM_SPIDERWEB_TRACE_RUNTIME,
)


ATOM_HARNESS_SIDE_VIEW_RUNTIME = "atom-language-harness-side-view-v1"


def _validate_hash(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> None:
    if field not in payload:
        raise ValueError(f"{label} has no {field}")
    core = {key: payload[key] for key in sorted(payload) if key != field}
    if payload[field] != canonical_hash(core):
        raise ValueError(f"{label} hash mismatch")


def _validate_binding(
    artifact: Mapping[str, Any],
    workflow: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> None:
    _validate_hash(artifact, "artifact_hash", "harness artifact")
    _validate_hash(workflow, "workflow_hash", "harness workflow")
    _validate_hash(
        artifact["evidence_packet"],
        "packet_hash",
        "evidence packet",
    )
    _validate_hash(
        artifact["spiderweb_trace"],
        "trace_hash",
        "Spiderweb trace",
    )
    _validate_hash(graph, "knowledge_hash", "wiki graph")
    if artifact["runtime"] != ATOM_LANGUAGE_HARNESS_RUNTIME:
        raise ValueError("harness artifact runtime is invalid")
    if artifact["spiderweb_trace"]["runtime"] != ATOM_SPIDERWEB_TRACE_RUNTIME:
        raise ValueError("Spiderweb trace runtime is invalid")
    if workflow["artifact_hash"] != artifact["artifact_hash"]:
        raise ValueError("side view is detached from harness artifact")
    if workflow["evidence_packet_hash"] != artifact["evidence_packet"]["packet_hash"]:
        raise ValueError("side view is detached from evidence packet")
    if workflow["graph_knowledge_hash"] != graph["knowledge_hash"]:
        raise ValueError("side view is detached from wiki graph")
    if artifact["knowledge"]["graph_knowledge_hash"] != graph["knowledge_hash"]:
        raise ValueError("artifact is detached from wiki graph")
    if workflow["side_view_runtime"] != ATOM_HARNESS_SIDE_VIEW_RUNTIME:
        raise ValueError("side view workflow runtime is invalid")
    if artifact["knowledge"]["wiki_runtime"] != ATOM_HARNESS_WIKI_RUNTIME:
        raise ValueError("harness wiki runtime is invalid")
    if artifact["knowledge"]["rag_runtime"] != ATOM_HARNESS_RAG_RUNTIME:
        raise ValueError("harness RAG runtime is invalid")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _passage_card(passage: Mapping[str, Any]) -> str:
    facts = "".join(
        (f'<span class="fact">{_escape(item["role"])}: {_escape(item["value"])}</span>')
        for item in passage["facts"][:10]
    )
    return f"""
<article class="passage">
  <div class="passage-top">
    <span class="score">score {_escape(passage["score"])}</span>
    <code>{_escape(passage["experience_id"])}</code>
  </div>
  <p>{_escape(passage["summary"])}</p>
  <div class="facts">{facts}</div>
  <small>{len(passage["wiki_paths"])} graph paths ·
  {_escape(passage["coverage_per_million"])} ppm coverage</small>
</article>
"""


def render_atom_harness_artifact(
    artifact: Mapping[str, Any],
    workflow: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    """Render the actual answer beside its Atom evidence and bus trace."""

    _validate_binding(artifact, workflow, graph)
    response = artifact["response"]
    packet = artifact["evidence_packet"]
    status_class = "answerable" if response["answerable"] else "abstained"
    status_label = "Evidence-grounded" if response["answerable"] else "Abstained"
    citations = "".join(
        f"<li><code>{_escape(item)}</code></li>" for item in response["citations"]
    )
    if not citations:
        citations = "<li>No evidence citations emitted.</li>"
    passages = "".join(_passage_card(item) for item in packet["passages"])
    if not passages:
        passages = (
            '<div class="empty">Atom returned insufficient evidence. '
            "The language model was not allowed to fill the gap.</div>"
        )
    layers = "".join(
        (
            '<div class="layer">'
            f"<strong>{_escape(item['layer'])}</strong>"
            f"<span>{_escape(item['name'])}</span>"
            "</div>"
        )
        for item in artifact["spiderweb_trace"]["layers"]
    )
    model = artifact["language_model"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atom Language Harness</title>
<style>
:root {{
  color-scheme: dark;
  --ink: #f4f3ef;
  --muted: #aaa9a2;
  --panel: #171918;
  --panel-2: #202320;
  --line: #343a35;
  --lime: #b7f36b;
  --amber: #ffc865;
  --cyan: #75dbe8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #0c0e0d;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}}
header {{
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 24px 28px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(110deg, #111512, #171b18);
}}
h1, h2, h3, p {{ margin-top: 0; }}
h1 {{ margin-bottom: 4px; font-size: 22px; }}
header p {{ margin: 0; color: var(--muted); }}
.badge {{
  align-self: center;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 7px 12px;
  color: var(--lime);
  white-space: nowrap;
}}
.workspace {{
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(340px, 1fr);
  min-height: calc(100vh - 86px);
}}
main, aside {{ padding: 28px; }}
main {{ border-right: 1px solid var(--line); }}
aside {{ background: #101210; }}
.eyebrow {{
  color: var(--cyan);
  font-size: 12px;
  letter-spacing: .13em;
  text-transform: uppercase;
}}
.question {{
  font-size: clamp(22px, 3vw, 38px);
  line-height: 1.15;
  max-width: 900px;
}}
.answer {{
  margin-top: 28px;
  padding: 24px;
  border: 1px solid var(--line);
  border-left: 4px solid var(--lime);
  background: var(--panel);
  font-size: 18px;
  line-height: 1.65;
}}
.answer.abstained {{ border-left-color: var(--amber); }}
.answer small {{ display: block; color: var(--muted); margin-top: 16px; }}
.meta-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 18px;
}}
.meta {{
  padding: 14px;
  border: 1px solid var(--line);
  background: var(--panel);
}}
.meta span {{ display: block; color: var(--muted); font-size: 12px; }}
.meta strong {{ display: block; margin-top: 5px; overflow-wrap: anywhere; }}
code {{
  color: var(--lime);
  font-family: "Cascadia Code", Consolas, monospace;
  overflow-wrap: anywhere;
}}
.passage {{
  margin: 14px 0;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--panel);
}}
.passage-top {{ display: flex; flex-direction: column; gap: 7px; }}
.score {{ color: var(--cyan); font-size: 12px; }}
.passage p {{ margin: 14px 0; line-height: 1.45; }}
.passage small {{ color: var(--muted); }}
.facts {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
.fact {{
  padding: 4px 7px;
  border-radius: 5px;
  background: var(--panel-2);
  color: #d8dad5;
  font-size: 11px;
}}
.layers {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; }}
.layer {{
  padding: 10px;
  border: 1px solid var(--line);
  background: var(--panel);
}}
.layer strong, .layer span {{ display: block; }}
.layer strong {{ color: var(--amber); }}
.layer span {{ color: var(--muted); font-size: 11px; margin-top: 3px; }}
.empty {{
  padding: 18px;
  border: 1px dashed var(--amber);
  color: var(--muted);
}}
ul {{ padding-left: 20px; }}
@media (max-width: 900px) {{
  .workspace {{ grid-template-columns: 1fr; }}
  main {{ border-right: 0; border-bottom: 1px solid var(--line); }}
}}
</style>
</head>
<body>
<header>
  <div>
    <h1>Atom Language Harness</h1>
    <p>Atom owns evidence. The LLM supplies language.</p>
  </div>
  <div class="badge">{_escape(status_label)}</div>
</header>
<div class="workspace">
  <main>
    <p class="eyebrow">User request</p>
    <h2 class="question">{_escape(artifact["question"])}</h2>
    <section class="answer {status_class}">
      {_escape(response["answer"])}
      <small>{_escape(response["limitations"])}</small>
    </section>
    <h3>Citations</h3>
    <ul>{citations}</ul>
    <div class="meta-grid">
      <div class="meta"><span>Model</span>
        <strong>{_escape(model["model"])}</strong></div>
      <div class="meta"><span>Atom snapshot</span>
        <strong>{_escape(packet["snapshot_sequence"])}</strong></div>
      <div class="meta"><span>Wiki nodes</span>
        <strong>{_escape(graph["node_count"])}</strong></div>
      <div class="meta"><span>Memory writes by LLM</span>
        <strong>none</strong></div>
    </div>
    <h3>Spiderweb route</h3>
    <div class="layers">{layers}</div>
    <p><small>Thread <code>{
        _escape(artifact["spiderweb_trace"]["thread"]["thread_id"])
    }</code> formed from observed flow.</small></p>
  </main>
  <aside>
    <p class="eyebrow">Bound evidence · side view</p>
    <h2>{len(packet["passages"])} retrieved passages</h2>
    <p>{_escape(packet["untrusted_data_notice"])}</p>
    {passages}
    <p><small>Packet <code>{_escape(packet["packet_hash"])}</code></small></p>
  </aside>
</div>
<!-- {ATOM_HARNESS_SIDE_VIEW_RUNTIME} -->
</body>
</html>
"""
