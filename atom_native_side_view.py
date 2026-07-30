"""User-visible Atom-first artifact view with Rust and frontend projections."""

from __future__ import annotations

import html
from typing import Any, Mapping

from atom_causal_world_schema import canonical_hash


ATOM_NATIVE_SIDE_VIEW_RUNTIME = "atom-native-artifact-side-view-v1"


def _validate_binding(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    atom_source: str,
    rust_source: str,
    frontend_component: str,
) -> None:
    model_hash = model.get("model_hash")
    report_hash = report.get("report_hash")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("native side view requires a model hash")
    if report.get("model_hash") != model_hash:
        raise ValueError("native report is not bound to the model")
    if workflow.get("model_hash") != model_hash:
        raise ValueError("native workflow is not bound to the model")
    if workflow.get("report_hash") != report_hash:
        raise ValueError("native workflow is not bound to the report")
    expected = {
        "atom": canonical_hash({"source": atom_source}),
        "rust": canonical_hash({"source": rust_source}),
        "frontend": canonical_hash({"source": frontend_component}),
    }
    if workflow.get("artifact_hashes") != expected:
        raise ValueError("native side view artifacts are not workflow-bound")
    contract = report.get("side_view_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("runtime") != ATOM_NATIVE_SIDE_VIEW_RUNTIME
        or contract.get("artifact_binding_marker")
        != "render_atom_native_artifact"
    ):
        raise ValueError("native side-view contract is invalid")


def _metric(label: str, value: Any) -> str:
    rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(rendered)
        + "</strong></div>"
    )


def render_atom_native_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
    atom_source: str,
    rust_source: str,
    frontend_component: str,
) -> str:
    """Render Atom as primary and its replaceable projections beside it."""

    _validate_binding(
        model,
        report,
        workflow,
        atom_source,
        rust_source,
        frontend_component,
    )
    benchmark = report["benchmark"]
    metrics = "".join(
        (
            _metric("Atom capability score", benchmark["atom_capability_score"]),
            _metric(
                "Baseline capability score",
                benchmark["baseline_capability_score"],
            ),
            _metric("Rust target passes", benchmark["rust_full_passes"]),
            _metric("Atom programs", benchmark["spec_count"]),
            _metric("Causal laws", report["learning"]["crystallized_laws"]),
            _metric("Atom interventions", report["learning"]["interventions"]),
        )
    )
    laws = "".join(
        "<tr><td>"
        + html.escape(law["requirement"])
        + "</td><td>"
        + html.escape(law["primitive"])
        + "</td><td>"
        + html.escape(law["status"])
        + "</td></tr>"
        for law in model["laws"]
        if law["status"] == "crystallized"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Native Construction</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #071018; color: #e7f1f5; }}
.surface {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(420px, .75fr); min-height: 100vh; }}
main, aside {{ padding: 28px; }}
aside {{ border-left: 1px solid #29485b; background: #0d1a24; }}
.metrics {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; }}
.metric {{ padding: 12px; border: 1px solid #29485b; border-radius: 8px; background: #112331; }}
.metric span {{ display: block; color: #91abba; font-size: 12px; }}
.metric strong {{ display: block; margin-top: 5px; color: #89e3c4; font-size: 20px; }}
pre {{ max-height: 42vh; overflow: auto; padding: 15px; background: #03080c; border: 1px solid #29485b; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
td, th {{ padding: 7px; border-bottom: 1px solid #29485b; text-align: left; }}
.hash {{ color: #8199a8; word-break: break-all; font-size: 11px; }}
</style>
</head>
<body>
<div class="surface">
<main>
<p>Primary construction language</p>
<h1>Atom</h1>
<section class="metrics">{metrics}</section>
<h2>Typed causal source</h2>
<pre><code>{html.escape(atom_source)}</code></pre>
<h2>Learned causal bindings</h2>
<table><thead><tr><th>Capability</th><th>Primitive</th><th>State</th></tr></thead>
<tbody>{laws}</tbody></table>
<p class="hash">Program {html.escape(workflow["program_hash"])}</p>
</main>
<aside aria-label="Replaceable target projections">
<h2>Rust execution projection</h2>
<pre><code>{html.escape(rust_source[:6000])}</code></pre>
<h2>Thin Svelte projection</h2>
<pre><code>{html.escape(frontend_component[:5000])}</code></pre>
<p>Both projections are generated from the same Atom program hash.</p>
<p class="hash">Model {html.escape(model["model_hash"])}<br>Report {html.escape(report["report_hash"])}</p>
</aside>
</div>
</body>
</html>"""
