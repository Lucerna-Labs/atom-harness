"""User-visible side view for trusted live causal learning cycles."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
)
from atom_causal_live import CAUSAL_LIVE_RUNTIME
from atom_causal_world_schema import canonical_hash

CAUSAL_LIVE_SIDE_VIEW_RUNTIME = "atom-causal-live-side-view-v1"


def _validate_binding(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> None:
    if report.get("passed") is not True:
        raise ValueError("live causal report did not pass its gates")
    core = {
        key: report[key]
        for key in sorted(report)
        if key != "report_hash"
    }
    if report.get("report_hash") != canonical_hash(core):
        raise ValueError("live causal report hash mismatch")
    workflow_core = {
        key: workflow[key]
        for key in sorted(workflow)
        if key != "workflow_hash"
    }
    if workflow.get("workflow_hash") != canonical_hash(workflow_core):
        raise ValueError("live causal workflow hash mismatch")
    if workflow.get("report_hash") != report["report_hash"]:
        raise ValueError("live causal workflow is detached from its report")
    if workflow.get("inventory_hash") != canonical_hash(inventory):
        raise ValueError("live causal inventory binding is invalid")
    if workflow.get("knowledge_hash") != knowledge.get(
        "knowledge_hash"
    ):
        raise ValueError("live causal knowledge binding is invalid")
    if workflow.get("store_sha256") != report.get("store_sha256"):
        raise ValueError("live causal store binding is invalid")
    if knowledge.get("inventory_hash") != canonical_hash(inventory):
        raise ValueError("live causal knowledge inventory is detached")
    if knowledge.get("catalog_identity") != inventory.get(
        "catalog_identity"
    ):
        raise ValueError("live causal knowledge catalog is detached")
    contract = report.get("side_view_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("runtime") != CAUSAL_LIVE_SIDE_VIEW_RUNTIME
        or contract.get("artifact_binding_marker")
        != "render_causal_live_artifact"
        or contract.get("placement") != "side"
        or contract.get("user_visible") is not True
    ):
        raise ValueError("live causal side-view contract is invalid")


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(str(value))
        + "</strong></div>"
    )


def render_causal_live_artifact(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> str:
    """Render live prediction, observation, feedback, and replay evidence."""

    _validate_binding(report, inventory, workflow, knowledge)
    cycles = report["cycles"]
    rows = []
    for label in ("first_novel_outcome", "replay", "second_outcome"):
        cycle = cycles[label]
        feedback = cycle["feedback"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td><code>{html.escape(str(cycle['selected_experience']))}</code></td>"
            f"<td>{html.escape(str(cycle['selected_effect']))}</td>"
            f"<td>{html.escape(str(cycle['observed_effect']))}</td>"
            f"<td>{html.escape(str(cycle['prediction_correct']))}</td>"
            f"<td>{html.escape(str(cycle['ingest']['committed']))}</td>"
            f"<td>{html.escape(str(feedback['committed']) if feedback else 'none')}</td>"
            f"<td>{html.escape(str(cycle['replayed']))}</td>"
            "</tr>"
        )
    checks = "".join(
        "<li>"
        + html.escape(name)
        + ": "
        + ("pass" if value else "fail")
        + "</li>"
        for name, value in sorted(report["checks"].items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Causal Learning</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #08100f; color: #e8fff5; }}
aside {{ min-height: 100vh; width: min(760px, 100%); margin-left: auto;
  box-sizing: border-box; padding: 26px; border-left: 1px solid #24594b;
  background: linear-gradient(160deg, #0b1714, #101b24); }}
h1, h2 {{ margin: 0 0 12px; }} h2 {{ margin-top: 24px; color: #8fe8c6; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.metric {{ padding: 12px; border: 1px solid #24594b; border-radius: 10px;
  background: #10231d; }} .metric span {{ display: block; color: #9dc2b6;
  font-size: .78rem; }} .metric strong {{ font-size: 1.15rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
th, td {{ padding: 8px; border-bottom: 1px solid #244139; text-align: left;
  vertical-align: top; }} code {{ overflow-wrap: anywhere; color: #9ff4d2; }}
.hash {{ overflow-wrap: anywhere; font-size: .76rem; color: #a9c8be; }}
li {{ margin: 5px 0; }}
</style>
</head>
<body><aside data-runtime="{CAUSAL_LIVE_SIDE_VIEW_RUNTIME}">
<h1>Live causal learning</h1>
<p>A trusted outcome entered structural Atom memory, changed the next
prediction, survived process reopen, and replayed without another mutation.</p>
<div class="metrics">
{_metric("Experiences", len(inventory["experiences"]))}
{_metric("Batches", len(inventory["batches"]))}
{_metric("Live sessions", report["live_session_count"])}
{_metric("Wiki nodes", knowledge["node_count"])}
{_metric("Wiki edges", knowledge["edge_count"])}
{_metric("RAG contexts", report["rag_context_count"])}
</div>
<h2>Interaction cycles</h2>
<table><thead><tr><th>Cycle</th><th>Selected</th><th>Predicted effect</th>
<th>Observed effect</th><th>Correct</th><th>Ingest</th><th>Feedback</th>
<th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Runtime gates</h2><ul>{checks}</ul>
<h2>Bound artifact</h2>
<p class="hash">Store: {html.escape(report["store_sha256"])}<br>
Report: {html.escape(report["report_hash"])}<br>
Knowledge: {html.escape(knowledge["knowledge_hash"])}<br>
Workflow: {html.escape(workflow["workflow_hash"])}</p>
<p>Wiki: <code>{CAUSAL_EXPERIENCE_WIKI_RUNTIME}</code><br>
RAG: <code>{CAUSAL_EXPERIENCE_RAG_RUNTIME}</code><br>
Runtime: <code>{CAUSAL_LIVE_RUNTIME}</code><br>
Side view: <code>{CAUSAL_LIVE_SIDE_VIEW_RUNTIME}</code></p>
</aside></body></html>"""
