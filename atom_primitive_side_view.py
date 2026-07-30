"""User-visible side view bound to the real Primitive Forge graph artifact."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash
from atom_primitive_forge import PrimitiveForge
from atom_primitive_knowledge import validate_knowledge_manifest


PRIMITIVE_SIDE_VIEW_RUNTIME = "atom-primitive-artifact-side-view-v1"


def _validate_binding(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge_manifest: Mapping[str, Any],
) -> PrimitiveForge:
    forge = PrimitiveForge.from_model_payload(model)
    graph_hash = forge.graph_hash
    if report.get("graph_hash") != graph_hash:
        raise ValueError("primitive report is detached from the graph artifact")
    if workflow.get("graph_hash") != graph_hash:
        raise ValueError("primitive workflow is detached from the graph artifact")
    artifact_hash = canonical_hash(model)
    if workflow.get("artifact_hash") != artifact_hash:
        raise ValueError("side view is not bound to the exact model artifact")
    report_hash = report.get("report_hash")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError("primitive report hash is invalid")
    report_core = {key: report[key] for key in report if key != "report_hash"}
    if canonical_hash(report_core) != report_hash:
        raise ValueError("primitive report hash mismatch")
    if workflow.get("report_hash") != report_hash:
        raise ValueError("primitive workflow is detached from the report")
    response_hash = workflow.get("response_hash")
    if not isinstance(response_hash, str) or len(response_hash) != 64:
        raise ValueError("primitive workflow hash is invalid")
    workflow_core = {
        key: workflow[key] for key in workflow if key != "response_hash"
    }
    if canonical_hash(workflow_core) != response_hash:
        raise ValueError("primitive workflow hash mismatch")
    validate_knowledge_manifest(knowledge_manifest, forge)
    if (
        workflow.get("knowledge_hash")
        != knowledge_manifest.get("knowledge_hash")
    ):
        raise ValueError("primitive workflow is detached from graph knowledge")
    contract = report.get("side_view_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("runtime") != PRIMITIVE_SIDE_VIEW_RUNTIME
        or contract.get("artifact_binding_marker")
        != "render_primitive_forge_artifact"
        or contract.get("placement") != "side"
        or contract.get("user_visible") is not True
    ):
        raise ValueError("primitive side-view contract is invalid")
    return forge


def _metric(label: str, value: Any) -> str:
    rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(rendered)
        + "</strong></div>"
    )


def render_primitive_forge_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge_manifest: Mapping[str, Any],
) -> str:
    """Render the generated primitive inventory and retrieved graph context."""

    forge = _validate_binding(model, report, workflow, knowledge_manifest)
    summary = report["inventory"]
    evaluation = report["evaluation"]
    roots = "".join(
        f"<li><code>{html.escape(root)}</code></li>" for root in forge.root_ids
    )
    rows = []
    for record in forge.derived_records:
        assert record.recipe is not None
        expansion = " → ".join(forge.expand_to_roots(record.primitive_id))
        rows.append(
            "<tr><td><code>"
            + html.escape(record.primitive_id)
            + "</code></td><td>"
            + html.escape(record.recipe.mode)
            + "</td><td>"
            + html.escape(record.status)
            + "</td><td>"
            + html.escape(expansion)
            + "</td></tr>"
        )
    rag_rows = []
    for item in workflow["knowledge_context"]:
        rag_rows.append(
            "<article><h3><code>"
            + html.escape(item["primitive_id"])
            + "</code></h3><p>"
            + html.escape(item["description"])
            + "</p><p class=\"lineage\">"
            + html.escape(" → ".join(item["root_expansion"]))
            + "</p></article>"
        )
    metrics = "".join(
        (
            _metric("Immutable roots", summary["root_count"]),
            _metric("Derived primitives", summary["derived_count"]),
            _metric("Recursive primitives", summary["recursive_count"]),
            _metric("Crystallized", summary["crystallized_count"]),
            _metric("Held-out accuracy", evaluation["heldout_accuracy"]),
            _metric(
                "Counterfactual accuracy",
                evaluation["counterfactual_accuracy"],
            ),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Open Primitive Forge</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #071018; color: #ecf6f1; }}
.surface {{ display: grid; grid-template-columns: minmax(0,1.45fr) minmax(390px,.55fr); min-height: 100vh; }}
main, aside {{ padding: 26px; }}
aside {{ border-left: 1px solid #315265; background: #0c1b24; }}
.metrics {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; }}
.metric {{ padding: 12px; border: 1px solid #315265; border-radius: 9px; background: #112733; }}
.metric span {{ display: block; color: #9fb4be; font-size: 12px; }}
.metric strong {{ display: block; color: #8fe0bd; font-size: 20px; margin-top: 4px; }}
.roots {{ display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 7px; padding: 0; list-style: none; }}
.roots li, article {{ border: 1px solid #315265; border-radius: 8px; padding: 10px; background: #0e202b; }}
.inventory {{ max-height: 58vh; overflow: auto; border: 1px solid #315265; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
td, th {{ padding: 8px; border-bottom: 1px solid #284657; text-align: left; vertical-align: top; }}
article {{ margin-bottom: 10px; }}
article h3 {{ margin: 0; font-size: 12px; }}
article p {{ color: #bfd0d7; font-size: 12px; }}
.lineage, .hash {{ color: #8fa5af; overflow-wrap: anywhere; font-size: 11px; }}
</style>
</head>
<body>
<div class="surface">
<main>
<p>Universe-first mathematical composition runtime</p>
<h1>Open Primitive Forge</h1>
<section class="metrics">{metrics}</section>
<h2>Immutable generative substrate</h2>
<ul class="roots">{roots}</ul>
<h2>Discovered recursive inventory</h2>
<div class="inventory"><table>
<thead><tr><th>Canonical identity</th><th>Recipe</th><th>State</th><th>Root expansion</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
</main>
<aside aria-label="Primitive artifact side view">
<h2>Graph-native wiki and RAG</h2>
<p>This side view is bound to the serialized graph that generated these nodes.</p>
{''.join(rag_rows)}
<h2>Scope</h2>
<p>{html.escape(report["claim_scope"])}</p>
<p class="hash">Graph {html.escape(forge.graph_hash)}<br>
Artifact {html.escape(workflow["artifact_hash"])}<br>
Knowledge {html.escape(knowledge_manifest["knowledge_hash"])}</p>
</aside>
</div>
</body>
</html>"""
