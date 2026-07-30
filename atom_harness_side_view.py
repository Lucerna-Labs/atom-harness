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
from atom_provider_fabric import ATOM_PROVIDER_FABRIC_RUNTIME


ATOM_HARNESS_SIDE_VIEW_RUNTIME = "atom-language-harness-side-view-v3"


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


def _validate_provider_state(payload: Mapping[str, Any]) -> None:
    identity_core = {
        "schema": payload["schema"],
        "runtime": payload.get("runtime", ATOM_PROVIDER_FABRIC_RUNTIME),
        "protocol": payload["protocol"],
        "ordered": payload["ordered"],
        "policy": payload["policy"],
        "providers": [
            {key: value for key, value in provider.items() if key != "circuit"}
            for provider in payload["providers"]
        ],
    }
    state_core = {
        **identity_core,
        "providers": payload["providers"],
    }
    if payload["preload_hash"] != canonical_hash(identity_core):
        raise ValueError("provider preload identity hash mismatch")
    if payload["state_hash"] != canonical_hash(state_core):
        raise ValueError("provider preload state hash mismatch")


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
    _validate_hash(
        artifact["intent_assistance"],
        "assistance_hash",
        "intent assistance",
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
    if workflow["transaction_id"] != artifact["transaction"]["transaction_id"]:
        raise ValueError("side view is detached from run transaction")
    if workflow["transaction_runtime"] != artifact["transaction"]["runtime"]:
        raise ValueError("side view transaction runtime is invalid")
    if (
        artifact["transaction"]["atomic_publication"] is not True
        or artifact["transaction"]["overwrite_allowed"] is not False
    ):
        raise ValueError("side view transaction contract is invalid")
    if workflow["provider_route_hashes"] != [
        item["route_hash"] for item in artifact["provider_routes"]
    ]:
        raise ValueError("side view is detached from provider routes")
    for route in artifact["provider_routes"]:
        _validate_hash(route, "route_hash", "provider route")
    routes_by_stage = {
        route["stage"]: route
        for route in artifact["provider_routes"]
        if route.get("completed") is True
    }
    for completion in artifact["completions"]:
        lane = completion.get("language_lane")
        route = routes_by_stage.get(completion["stage"])
        if isinstance(lane, Mapping) and lane:
            if route is None or route.get("language_lane") != lane:
                raise ValueError("side view is detached from resident lane evidence")
    if workflow["graph_knowledge_hash"] != graph["knowledge_hash"]:
        raise ValueError("side view is detached from wiki graph")
    if workflow["knowledge_hash"] != artifact["knowledge"]["knowledge_hash"]:
        raise ValueError("side view is detached from knowledge manifest")
    if workflow["store_sha256"] != artifact["memory"]["store_sha256_after"]:
        raise ValueError("side view is detached from Atom memory")
    if artifact["memory"]["unchanged"] is not True:
        raise ValueError("side view cannot render mutated Atom memory")
    if workflow["model_manifest_hash"] != canonical_hash(artifact["language_model"]):
        raise ValueError("side view is detached from provider fabric")
    if (
        artifact["provider_preload"]["preload_hash"]
        != artifact["language_model"]["preload_hash"]
    ):
        raise ValueError("side view is detached from provider preload")
    _validate_provider_state(artifact["provider_preload"])
    _validate_provider_state(artifact["language_model"])
    if artifact["knowledge"]["graph_knowledge_hash"] != graph["knowledge_hash"]:
        raise ValueError("artifact is detached from wiki graph")
    if workflow["side_view_runtime"] != ATOM_HARNESS_SIDE_VIEW_RUNTIME:
        raise ValueError("side view workflow runtime is invalid")
    if artifact["knowledge"]["wiki_runtime"] != ATOM_HARNESS_WIKI_RUNTIME:
        raise ValueError("harness wiki runtime is invalid")
    if artifact["knowledge"]["rag_runtime"] != ATOM_HARNESS_RAG_RUNTIME:
        raise ValueError("harness RAG runtime is invalid")
    if artifact["evidence_packet"]["graph_knowledge_hash"] != graph["knowledge_hash"]:
        raise ValueError("evidence packet is detached from wiki graph")
    request_id = artifact["request_id"]
    if (
        artifact["evidence_packet"]["request_id"] != request_id
        or artifact["spiderweb_trace"]["request_id"] != request_id
    ):
        raise ValueError("side view combines different requests")
    if artifact["passed"] is not True or not all(artifact["checks"].values()):
        raise ValueError("side view refuses a failed harness artifact")
    allowed_citations = {
        item["experience_id"] for item in artifact["evidence_packet"]["passages"]
    }
    response = artifact["response"]
    if not set(response["citations"]) <= allowed_citations:
        raise ValueError("side view refuses a citation outside the evidence packet")
    if response["answerable"] and not response["citations"]:
        raise ValueError("side view refuses an uncited answer")
    if response["answerable"] and response.get("grounding") != artifact[
        "evidence_packet"
    ].get("primary_claim"):
        raise ValueError("side view refuses an answer detached from Atom's claim")
    if not response["answerable"] and response.get("grounding") is not None:
        raise ValueError("side view refuses grounding on an abstention")
    if artifact["evidence_packet"]["insufficient_evidence"] and response["answerable"]:
        raise ValueError("side view refuses an answer over insufficient evidence")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _yes_no(value: Any) -> str:
    return "yes" if value is True else "no"


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
  <small>{len(passage["wiki_paths"])} graph paths &middot;
  {_escape(passage["coverage_per_million"])} ppm coverage</small>
</article>
"""


def _provider_route_card(route: Mapping[str, Any]) -> str:
    selected = route.get("selected_provider")
    selected_label = (
        f"{selected['provider_id']} / {selected['model']}"
        if isinstance(selected, Mapping)
        else "none"
    )
    attempts = "".join(
        (
            '<li class="attempt">'
            f"<strong>{_escape(item['provider_id'])}</strong>"
            f"<span>{_escape(item['location'])} &middot; "
            f"attempt {_escape(item['attempt'])} &middot; "
            f"{_escape(item['outcome'])}"
            + (
                f" &middot; {_escape(item['failure_kind'])}"
                if item.get("failure_kind")
                else ""
            )
            + (
                f" &middot; circuit {_escape(item['circuit_after']['state'])}"
                if isinstance(item.get("circuit_after"), Mapping)
                else ""
            )
            + "</span></li>"
        )
        for item in route["attempts"]
    )
    return f"""
<article class="route">
  <div class="route-top">
    <strong>{_escape(route["stage"])}</strong>
    <span>{_escape(route["disposition"])}</span>
  </div>
  <p>Selected: <code>{_escape(selected_label)}</code></p>
  <ul>{attempts}</ul>
  <small>{_escape(route["elapsed_ms"])} ms &middot;
  <code>{_escape(route["route_hash"])}</code></small>
</article>
"""


def _performance_card(completion: Mapping[str, Any]) -> str:
    performance = completion.get("performance")
    lane = completion.get("language_lane")
    if not isinstance(performance, Mapping) or not performance:
        detail = "Backend performance metrics unavailable."
    else:
        cold_start = performance.get("cold_start_ms")
        load = performance.get("load_ms")
        throughput = performance.get("generation_tokens_per_second")
        tokens = performance.get("generated_tokens")
        parts = [
            (
                f"cold start {_escape(cold_start)} ms"
                if cold_start not in (None, 0)
                else (
                    "resident model reused"
                    if performance.get("warm_request") is True
                    else (
                        f"load {_escape(load)} ms"
                        if load is not None
                        else "load unavailable"
                    )
                )
            ),
            (
                f"generation {_escape(throughput)} tok/s"
                if throughput is not None
                else "generation throughput unavailable"
            ),
        ]
        if tokens is not None:
            parts.append(f"{_escape(tokens)} generated tokens")
        if isinstance(lane, Mapping) and lane:
            parts.extend(
                (
                    f"lane generation {_escape(lane.get('process_generation'))}",
                    f"model loads {_escape(lane.get('model_load_count'))}",
                    f"restarts {_escape(lane.get('restart_count'))}",
                    f"queue wait {_escape(lane.get('queue_wait_ms'))} ms",
                )
            )
        detail = " &middot; ".join(parts)
    return f"""
<article class="route performance">
  <div class="route-top">
    <strong>{_escape(completion["stage"])}</strong>
    <span>{_escape(completion["model"])}</span>
  </div>
  <p>{detail}</p>
  <small>Total provider call: {_escape(completion["elapsed_ms"])} ms</small>
</article>
"""


def _selected_model_label(artifact: Mapping[str, Any]) -> str:
    selected = {
        str(route["selected_provider"]["model"])
        for route in artifact["provider_routes"]
        if isinstance(route.get("selected_provider"), Mapping)
    }
    if selected:
        return ", ".join(sorted(selected))
    return str(artifact["language_model"]["model"])


def render_atom_harness_artifact(
    artifact: Mapping[str, Any],
    workflow: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> str:
    """Render the actual answer beside its Atom evidence and bus trace."""

    _validate_binding(artifact, workflow, graph)
    response = artifact["response"]
    packet = artifact["evidence_packet"]
    status_class = (
        "answerable"
        if response["answerable"]
        else ("degraded" if artifact["degraded"] else "abstained")
    )
    status_label = (
        "Evidence-grounded"
        if response["answerable"]
        else ("Degraded safely" if artifact["degraded"] else "Abstained")
    )
    citations = "".join(
        f"<li><code>{_escape(item)}</code></li>" for item in response["citations"]
    )
    if not citations:
        citations = "<li>No evidence citations emitted.</li>"
    grounding = response.get("grounding")
    if isinstance(grounding, Mapping):
        grounding_view = (
            '<section class="authority-claim">'
            "<strong>Primary Atom claim</strong>"
            f"<p>{_escape(grounding['status'])} {_escape(grounding['kind'])}: "
            f"{_escape(grounding['cause'])} &rarr; {_escape(grounding['effect'])} "
            f"(direction {_escape(grounding['direction'])}) in "
            f"{_escape(grounding['domain'])}.</p>"
            f"<code>{_escape(grounding['source_experience_id'])}</code>"
            "</section>"
        )
    else:
        grounding_view = (
            '<section class="authority-claim"><strong>Primary Atom claim</strong>'
            "<p>No claim was licensed.</p></section>"
        )
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
    intent_assistance = artifact["intent_assistance"]
    anchor_count = sum(
        len(values) for values in intent_assistance["lexical_anchors"].values()
    )
    policy = model["policy"]
    timings = artifact["timings"]
    selected_model = _selected_model_label(artifact)
    routes = "".join(_provider_route_card(item) for item in artifact["provider_routes"])
    if not routes:
        routes = '<div class="empty">No provider route was recorded.</div>'
    performance = "".join(_performance_card(item) for item in artifact["completions"])
    if not performance:
        performance = '<div class="empty">No language completion was performed.</div>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none';
  form-action 'none'">
<title>Atom Language Harness V3</title>
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
.answer.degraded {{ border-left-color: #ff7d7d; }}
.answer small {{ display: block; color: var(--muted); margin-top: 16px; }}
.authority-claim {{
  margin: 16px 0 24px;
  padding: 14px 16px;
  border: 1px solid var(--line);
  background: rgba(70, 226, 195, 0.06);
}}
.authority-claim strong {{ color: var(--cyan); }}
.authority-claim p {{ margin: 8px 0; }}
.authority-claim code {{ overflow-wrap: anywhere; }}
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
.route {{
  margin: 12px 0;
  padding: 14px;
  border: 1px solid var(--line);
  background: var(--panel);
}}
.route-top {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
}}
.route-top span {{ color: var(--cyan); }}
.route ul {{ margin: 10px 0; }}
.attempt {{ margin: 7px 0; }}
.attempt strong, .attempt span {{ display: block; }}
.attempt span {{ color: var(--muted); font-size: 12px; }}
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
    <h1>Atom Language Harness V3</h1>
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
    {grounding_view}
    <h3>Citations</h3>
    <ul>{citations}</ul>
    <div class="meta-grid">
      <div class="meta"><span>Model</span>
        <strong>{_escape(selected_model)}</strong></div>
      <div class="meta"><span>Outcome</span>
        <strong>{_escape(artifact["outcome"])}</strong></div>
      <div class="meta"><span>Privacy locations</span>
        <strong>{_escape(", ".join(policy["allowed_locations"]))}</strong></div>
      <div class="meta"><span>Cloud evidence allowed</span>
        <strong>{_yes_no(policy["allow_cloud_data"])}</strong></div>
      <div class="meta"><span>All admitted providers cancellable</span>
        <strong>{_yes_no(model["capabilities"]["supports_cancellation"])}</strong></div>
      <div class="meta"><span>Atom snapshot</span>
        <strong>{_escape(packet["snapshot_sequence"])}</strong></div>
      <div class="meta"><span>Exact vocabulary anchors</span>
        <strong>{_escape(anchor_count)}</strong></div>
      <div class="meta"><span>Intent path</span>
        <strong>{_escape(intent_assistance["model_action"])} to {
        _escape(intent_assistance["final_action"])
    }</strong></div>
      <div class="meta"><span>Wiki nodes</span>
        <strong>{_escape(graph["node_count"])}</strong></div>
      <div class="meta"><span>Memory writes by LLM</span>
        <strong>none</strong></div>
      <div class="meta"><span>Answer runtime</span>
        <strong>{_escape(timings["total_ms"])} ms</strong></div>
      <div class="meta"><span>Retrieval runtime</span>
        <strong>{_escape(timings["retrieval_ms"])} ms</strong></div>
      <div class="meta"><span>Atomic transaction</span>
        <strong>{_escape(artifact["transaction"]["transaction_id"])}</strong></div>
      <div class="meta"><span>Recovery events before run</span>
        <strong>{
        _escape(artifact["transaction"]["recovery_event_count"])
    }</strong></div>
    </div>
    <h3>Spiderweb route</h3>
    <div class="layers">{layers}</div>
    <p><small>Thread <code>{
        _escape(artifact["spiderweb_trace"]["thread"]["thread_id"])
    }</code> formed from observed flow.</small></p>
    <h3>Provider fabric</h3>
    {routes}
    <h3>Language performance</h3>
    {performance}
  </main>
  <aside>
    <p class="eyebrow">Bound evidence &middot; side view</p>
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
