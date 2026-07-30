"""User-visible artifact side view for the causal coding experiment."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash


ATOM_CODING_SIDE_VIEW_RUNTIME = "atom-coding-artifact-side-view-v1"


def _metric(label: str, value: Any) -> str:
    rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(rendered)
        + "</strong></div>"
    )


def _validate_binding(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    artifact_source: str,
) -> tuple[str, str]:
    model_hash = model.get("model_hash")
    report_hash = report.get("report_hash")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("coding side view requires a model hash")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError("coding side view requires a report hash")
    if report.get("model_hash") != model_hash:
        raise ValueError("coding report is not bound to the causal model")
    if workflow.get("model_hash") != model_hash:
        raise ValueError("coding workflow is not bound to the causal model")
    if workflow.get("report_hash") != report_hash:
        raise ValueError("coding workflow is not bound to the report")
    if canonical_hash({"source": artifact_source}) != workflow.get("artifact_hash"):
        raise ValueError("coding side view source is not the workflow artifact")
    contract = report.get("side_view_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("coding report has no side-view contract")
    if contract.get("runtime") != ATOM_CODING_SIDE_VIEW_RUNTIME:
        raise ValueError("coding side-view runtime mismatch")
    if contract.get("artifact_binding_marker") != "render_coding_artifact":
        raise ValueError("coding side-view marker mismatch")
    return model_hash, report_hash


def render_coding_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    artifact_source: str,
) -> str:
    """Render the generated platform beside its measured causal evidence."""

    model_hash, report_hash = _validate_binding(
        model,
        report,
        workflow,
        artifact_source,
    )
    benchmark = report["benchmark"]
    learned = [
        law
        for law in model["laws"]
        if law["status"] == "crystallized"
    ]
    law_rows = "".join(
        "<tr><td>"
        + html.escape(law["requirement"])
        + "</td><td>"
        + html.escape(law["primitive"])
        + "</td><td>"
        + html.escape(f'{law["confidence"]:.3f}')
        + "</td><td>"
        + html.escape(f'{law["persistence"]:.3f}')
        + "</td></tr>"
        for law in learned
    )
    rag_items = "".join(
        "<li><strong>"
        + html.escape(hit["name"])
        + "</strong> - "
        + html.escape(hit["description"])
        + "</li>"
        for hit in workflow["knowledge_context"]
    )
    proof_rows = "".join(
        "<tr><td>"
        + html.escape(item["platform_primitive"])
        + "</td><td>"
        + html.escape(item["formal_primitive"])
        + "</td><td>"
        + html.escape(item["claim_status"])
        + "</td></tr>"
        for item in workflow["blueprint"]["proof_trace"]
    )
    metrics = "".join(
        (
            _metric("Baseline capability score", benchmark["baseline_capability_score"]),
            _metric("Atom capability score", benchmark["atom_capability_score"]),
            _metric("Improvement", benchmark["improvement"]),
            _metric("Atom full passes", benchmark["atom_full_passes"]),
            _metric("Crystallized laws", len(learned)),
            _metric("Training interventions", model["observation_count"]),
        )
    )
    source_excerpt = artifact_source[:8000]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Coding Artifact</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #091018; color: #e7f0f7; }}
.surface {{ min-height: 100vh; display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 38vw); }}
main {{ padding: 28px; }}
aside {{ padding: 28px; background: #101c28; border-left: 1px solid #294056; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; }}
.metric {{ padding: 12px; background: #142434; border: 1px solid #294056; border-radius: 8px; }}
.metric span {{ display: block; color: #95b1c8; font-size: 12px; }}
.metric strong {{ display: block; margin-top: 4px; font-size: 20px; color: #8de4ca; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th, td {{ border-bottom: 1px solid #294056; padding: 8px; text-align: left; vertical-align: top; }}
pre {{ padding: 16px; overflow: auto; background: #050a0f; border: 1px solid #294056; border-radius: 8px; }}
code {{ color: #c8f6e7; }}
.hash {{ word-break: break-all; color: #8198aa; font-size: 11px; }}
</style>
</head>
<body>
<div class="surface">
<main>
<h1>Generated mathematical-primitive platform</h1>
<p>This is the real executable artifact selected by the learned causal graph.</p>
<section class="metrics">{metrics}</section>
<h2>Generated source</h2>
<pre><code>{html.escape(source_excerpt)}</code></pre>
<p class="hash">Artifact {html.escape(workflow["artifact_hash"])}</p>
</main>
<aside aria-label="Causal artifact evidence">
<h2>Causal side view</h2>
<p>Model and artifact evidence are rendered beside the primary generated surface.</p>
<h3>Learned requirement laws</h3>
<table><thead><tr><th>Capability</th><th>Primitive</th><th>Confidence</th><th>Persistence</th></tr></thead>
<tbody>{law_rows}</tbody></table>
<h3>Formal primitive trace</h3>
<table><thead><tr><th>Platform primitive</th><th>Formal primitive</th><th>Status</th></tr></thead>
<tbody>{proof_rows}</tbody></table>
<h3>Graph-RAG context</h3>
<ul>{rag_items}</ul>
<p class="hash">Model {html.escape(model_hash)}<br>Report {html.escape(report_hash)}</p>
</aside>
</div>
</body>
</html>"""
