"""User-visible side view for the persistent causal-experience artifact."""

from __future__ import annotations

import html
from collections import Counter
from typing import Any, Mapping

from atom_causal_experience_knowledge import (
    CAUSAL_EXPERIENCE_RAG_RUNTIME,
    CAUSAL_EXPERIENCE_WIKI_RUNTIME,
)
from atom_causal_world_schema import canonical_hash

CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME = (
    "atom-causal-experience-side-view-v1"
)


def _validate_binding(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> None:
    if not report.get("passed"):
        raise ValueError("experience report did not pass its runtime gates")
    if report.get("report_hash") != canonical_hash(
        {
            key: report[key]
            for key in sorted(report)
            if key != "report_hash"
        }
    ):
        raise ValueError("experience report hash mismatch")
    if report["experience"]["catalog_identity"] != inventory.get(
        "catalog_identity"
    ):
        raise ValueError("side view inventory catalog is detached")
    if workflow.get("report_hash") != report["report_hash"]:
        raise ValueError("side view workflow report binding is invalid")
    if workflow.get("knowledge_hash") != knowledge.get("knowledge_hash"):
        raise ValueError("side view workflow knowledge binding is invalid")
    if workflow.get("inventory_hash") != canonical_hash(inventory):
        raise ValueError("side view workflow inventory binding is invalid")
    if knowledge.get("catalog_identity") != inventory.get(
        "catalog_identity"
    ):
        raise ValueError("side view knowledge catalog is detached")
    if knowledge.get("inventory_hash") != canonical_hash(inventory):
        raise ValueError("side view knowledge inventory is detached")
    contract = report.get("side_view_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("runtime")
        != CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME
        or contract.get("artifact_binding_marker")
        != "render_causal_experience_artifact"
        or contract.get("placement") != "side"
        or contract.get("user_visible") is not True
    ):
        raise ValueError("experience side-view contract is invalid")


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(str(value))
        + "</strong></div>"
    )


def render_causal_experience_artifact(
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    workflow: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> str:
    """Render the real online-memory artifact and its causal evidence."""

    _validate_binding(report, inventory, workflow, knowledge)
    experiences = inventory["experiences"]
    kind_counts = Counter(str(item["kind"]) for item in experiences)
    status_counts = Counter(str(item["status"]) for item in experiences)
    domain_counts = Counter(str(item["domain"]) for item in experiences)
    recall = report["recall"]["before_feedback"]
    persisted = report["recall"]["persisted"]
    target = report["recall"]["target_experience"]
    target_hit = next(
        item for item in persisted["hits"] if item["experience_id"] == target
    )
    motif_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['role'])}</td>"
        f"<td>{html.escape(item['value'])}</td>"
        f"<td>{item['conductance_per_mille']}</td>"
        f"<td>{item['contribution']}</td>"
        "</tr>"
        for item in target_hit["motifs"]
    )
    batch_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['batch_id']))}</td>"
        f"<td>{item['experience_count']}</td>"
        f"<td><code>{html.escape(str(item['source_artifact_hash']))}</code></td>"
        "</tr>"
        for item in inventory["batches"]
    )
    domain_rows = "".join(
        f"<tr><td>{html.escape(domain)}</td><td>{count}</td></tr>"
        for domain, count in sorted(domain_counts.items())
    )
    checks = "".join(
        '<li class="pass">✓ '
        + html.escape(name.replace("_", " "))
        + "</li>"
        for name, passed in sorted(report["checks"].items())
        if passed
    )
    metrics = "".join(
        [
            _metric("Observation revisions", kind_counts["observation"]),
            _metric("Consolidated laws", kind_counts["law"]),
            _metric("Durable batches", len(inventory["batches"])),
            _metric("Crystallized", status_counts["crystallized"]),
            _metric(
                "Target score increase",
                report["recall"]["target_score_increase"],
            ),
            _metric(
                "Unknown topology",
                "abstained"
                if report["recall"]["unknown"][
                    "insufficient_evidence"
                ]
                else "answered",
            ),
        ]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom causal experience memory</title>
<style>
:root {{ color-scheme:dark; --bg:#07110f; --panel:#10241e;
--line:#2c6553; --text:#e8fff5; --muted:#9bc7b7; --accent:#79f2bd; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at 20% 0,#173c31,var(--bg) 48%);
color:var(--text); font:14px/1.45 ui-monospace,Consolas,monospace; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 390px;
min-height:100vh; }}
article {{ padding:28px; }}
aside {{ border-left:1px solid var(--line); background:#091814ee;
padding:22px; position:sticky; top:0; height:100vh; overflow:auto; }}
h1,h2 {{ margin-top:0; letter-spacing:.02em; }}
h1 {{ color:var(--accent); }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
gap:10px; margin:18px 0; }}
.metric {{ border:1px solid var(--line); background:var(--panel);
border-radius:8px; padding:12px; }}
.metric span {{ display:block; color:var(--muted); font-size:11px; }}
.metric strong {{ display:block; margin-top:5px; font-size:18px; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 22px; }}
th,td {{ padding:8px; border-bottom:1px solid #254a40; text-align:left;
vertical-align:top; }}
th {{ color:var(--accent); }}
code {{ color:#b8ffe0; word-break:break-all; }}
ul {{ padding-left:20px; }}
.pass {{ color:#b8ffe0; margin:5px 0; }}
.hash {{ color:var(--muted); word-break:break-all; font-size:11px; }}
@media(max-width:900px) {{ main {{ grid-template-columns:1fr; }}
aside {{ position:static; height:auto; border-left:0; border-top:1px solid var(--line); }} }}
</style></head>
<body><main><article>
<h1>Persistent causal experience</h1>
<p>This is the live Atom DB side view. It renders observation revisions,
consolidated laws, structural recall paths, and outcome-driven conductance
from the durable store—not an embedding index or document search.</p>
<div class="metrics">{metrics}</div>
<h2>Append-only batches</h2>
<table><thead><tr><th>Batch</th><th>Records</th><th>Source hash</th></tr></thead>
<tbody>{batch_rows}</tbody></table>
<h2>World domains</h2>
<table><thead><tr><th>Domain</th><th>Durable experiences</th></tr></thead>
<tbody>{domain_rows}</tbody></table>
<h2>Persisted target motifs</h2>
<p><code>{html.escape(target)}</code></p>
<table><thead><tr><th>Role</th><th>Value</th><th>Conductance</th>
<th>Contribution</th></tr></thead><tbody>{motif_rows}</tbody></table>
</article><aside>
<h2>Runtime bindings</h2>
<p>Wiki: <code>{CAUSAL_EXPERIENCE_WIKI_RUNTIME}</code><br>
RAG: <code>{CAUSAL_EXPERIENCE_RAG_RUNTIME}</code><br>
Side view: <code>{CAUSAL_EXPERIENCE_SIDE_VIEW_RUNTIME}</code></p>
<p>Initial snapshot {recall['snapshot_sequence']}<br>
Persisted snapshot {persisted['snapshot_sequence']}<br>
Knowledge nodes {knowledge['node_count']}<br>
Knowledge edges {knowledge['edge_count']}</p>
<h2>Runtime gates</h2><ul>{checks}</ul>
<h2>Artifact identities</h2>
<p class="hash">Catalog {html.escape(str(inventory['catalog_identity']))}<br>
Report {html.escape(str(report['report_hash']))}<br>
Knowledge {html.escape(str(knowledge['knowledge_hash']))}<br>
Store {html.escape(str(workflow['store_sha256']))}</p>
</aside></main></body></html>"""
