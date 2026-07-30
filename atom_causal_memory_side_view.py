"""User-visible side view bound to the persisted causal-memory artifact."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_memory_knowledge import (
    validate_causal_memory_knowledge,
)
from atom_causal_world_schema import canonical_hash


CAUSAL_MEMORY_SIDE_VIEW_RUNTIME = "atom-causal-memory-side-view-v1"


def _validate_binding(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> None:
    graph_hash = report.get("source_graph_hash")
    catalog_identity = inventory.get("catalog_identity")
    if not isinstance(graph_hash, str) or len(graph_hash) != 64:
        raise ValueError("causal-memory report graph hash is invalid")
    if (
        inventory.get("source_graph_hash") != graph_hash
        or workflow.get("source_graph_hash") != graph_hash
    ):
        raise ValueError("causal-memory artifacts use different graphs")
    if (
        not isinstance(catalog_identity, str)
        or workflow.get("catalog_identity") != catalog_identity
    ):
        raise ValueError("causal-memory artifacts use different catalogs")
    report_hash = report.get("report_hash")
    report_core = {
        key: report[key] for key in report if key != "report_hash"
    }
    if (
        not isinstance(report_hash, str)
        or canonical_hash(report_core) != report_hash
    ):
        raise ValueError("causal-memory report hash mismatch")
    if workflow.get("report_hash") != report_hash:
        raise ValueError("causal-memory workflow is detached from its report")
    workflow_hash = workflow.get("workflow_hash")
    workflow_core = {
        key: workflow[key] for key in workflow if key != "workflow_hash"
    }
    if (
        not isinstance(workflow_hash, str)
        or canonical_hash(workflow_core) != workflow_hash
    ):
        raise ValueError("causal-memory workflow hash mismatch")
    validate_causal_memory_knowledge(
        knowledge,
        graph_hash=graph_hash,
        catalog_identity=catalog_identity,
    )
    if workflow.get("knowledge_hash") != knowledge.get("knowledge_hash"):
        raise ValueError(
            "causal-memory workflow is detached from runtime knowledge"
        )
    contract = report.get("side_view_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("runtime") != CAUSAL_MEMORY_SIDE_VIEW_RUNTIME
        or contract.get("artifact_binding_marker")
        != "render_causal_memory_artifact"
        or contract.get("placement") != "side"
        or contract.get("user_visible") is not True
    ):
        raise ValueError("causal-memory side-view contract is invalid")


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(str(value))
        + "</strong></div>"
    )


def render_causal_memory_artifact(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> str:
    """Render the real structural-query and feedback artifacts at the side."""

    _validate_binding(report, inventory, workflow, knowledge)
    before = report["retrieval"]["before_feedback"]
    persisted = report["retrieval"]["persisted"]
    feedback = report["learning"]
    rows = []
    for hit in persisted["hits"]:
        paths = "<br>".join(
            (
                f"{html.escape(item['role'])} "
                f"→ {html.escape(item['value'])} "
                f"@ {item['conductance_per_mille']}"
            )
            for item in hit["motifs"]
        )
        rows.append(
            "<tr><td><code>"
            + html.escape(hit["primitive_id"])
            + "</code></td><td>"
            + str(hit["score"])
            + "</td><td>"
            + f"{hit['coverage_per_million'] / 10_000:.2f}%"
            + "</td><td>"
            + paths
            + "</td></tr>"
        )
    adjustments = []
    for event_name in ("wrong_prediction", "correct_prediction"):
        event = feedback[event_name]
        adjustments.append(
            "<article><h3>"
            + html.escape(event_name.replace("_", " ").title())
            + "</h3><p>"
            + html.escape(event["selected_glyph"])
            + " → expected "
            + html.escape(event["expected_glyph"])
            + "</p><p>"
            + str(len(event["adjustments"]))
            + " motif conductance changes committed atomically.</p></article>"
        )
    metrics = "".join(
        (
            _metric("Durable glyphs", len(inventory["glyphs"])),
            _metric(
                "Structural motifs",
                report["storage"]["motif_count"],
            ),
            _metric("Immutable roots", report["storage"]["root_count"]),
            _metric("Top match", persisted["hits"][0]["primitive_id"]),
            _metric(
                "Score increase",
                report["retrieval"]["target_score_increase"],
            ),
            _metric(
                "Unknown query",
                (
                    "abstained"
                    if report["retrieval"]["unknown"][
                        "insufficient_evidence"
                    ]
                    else "answered"
                ),
            ),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Causal Memory</title>
<style>
:root {{ color-scheme: dark; font-family: Inter,system-ui,sans-serif; }}
body {{ margin: 0; background: #07110f; color: #effbf5; }}
.surface {{ display:grid; grid-template-columns:minmax(0,1.55fr) minmax(390px,.45fr); min-height:100vh; }}
main,aside {{ padding:26px; }}
aside {{ border-left:1px solid #355e50; background:#0b1d18; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
.metric,article {{ border:1px solid #355e50; border-radius:9px; background:#102a22; padding:11px; }}
.metric span {{ display:block; color:#9ab9ae; font-size:12px; }}
.metric strong {{ display:block; color:#8ee5bd; font-size:17px; margin-top:5px; overflow-wrap:anywhere; }}
.inventory {{ margin-top:18px; max-height:66vh; overflow:auto; border:1px solid #355e50; border-radius:9px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ padding:8px; border-bottom:1px solid #284b40; text-align:left; vertical-align:top; }}
article {{ margin-bottom:10px; }}
article h3 {{ margin:0 0 6px; }}
.hash {{ color:#8ea89f; overflow-wrap:anywhere; font-size:11px; }}
</style>
</head>
<body>
<div class="surface">
<main>
<p>Persistent topological memory over immutable Atom DB facts</p>
<h1>Causal Atom Memory</h1>
<section class="metrics">{metrics}</section>
<h2>Structural resonance after durable learning</h2>
<p>The query contains typed ports, composition topology, root lineage,
invariants, and symmetries. It contains no aliases, passage text, or
embedding vector.</p>
<div class="inventory"><table>
<thead><tr><th>Glyph</th><th>Activation</th><th>Coverage</th><th>Exact motif paths</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
</main>
<aside aria-label="Causal-memory artifact side view">
<h2>Prediction-driven metaplasticity</h2>
{''.join(adjustments)}
<h2>Fail-closed result</h2>
<p>Unobserved required topology returned
<strong>{html.escape(str(report['retrieval']['unknown']['insufficient_evidence']).lower())}</strong>
for insufficient evidence.</p>
<h2>Artifact binding</h2>
<p class="hash">Graph {html.escape(report['source_graph_hash'])}<br>
Catalog {html.escape(inventory['catalog_identity'])}<br>
Store {html.escape(workflow['store_sha256'])}<br>
Knowledge {html.escape(knowledge['knowledge_hash'])}<br>
Initial query sequence {before['snapshot_sequence']}<br>
Persisted query sequence {persisted['snapshot_sequence']}</p>
</aside>
</div>
</body>
</html>"""
