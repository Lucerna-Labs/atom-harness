"""Static model-bound side view for the homeostatic governor experiment."""

from __future__ import annotations

import html
import json
import math
from typing import Any, Mapping, Sequence


ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME = "atom-homeostatic-side-view-v1"
ATOM_HOMEOSTATIC_ARTIFACT_BINDING = "render_homeostatic_artifact"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _metric(label: str, value: Any) -> str:
    if isinstance(value, float):
        rendered = f"{value:.3f}"
    else:
        rendered = str(value)
    return (
        '<div class="metric">'
        f"<span>{_escape(label)}</span><strong>{_escape(rendered)}</strong>"
        "</div>"
    )


def _points(
    values: Sequence[float],
    minimum: float,
    maximum: float,
    width: float = 760.0,
    height: float = 210.0,
) -> str:
    if not values:
        return ""
    span = max(maximum - minimum, 1e-12)
    denominator = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = 18.0 + index * (width - 36.0) / denominator
        normalized = (value - minimum) / span
        y = height - 18.0 - normalized * (height - 36.0)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _trajectory(model: Mapping[str, Any]) -> str:
    history = model["controller"]["history"]
    config = model["config"]
    temperatures = [
        _finite(row["temperature_after"], "temperature") for row in history
    ]
    phases = [_finite(row["phase_after"], "phase") for row in history]
    thresholds = [
        _finite(row["threshold_after"], "threshold") for row in history
    ]
    return f"""
    <svg class="trajectory" viewBox="0 0 760 210" role="img" aria-label="Controller trajectory across evidence windows">
      <line x1="18" y1="192" x2="742" y2="192" class="axis"/>
      <line x1="18" y1="18" x2="18" y2="192" class="axis"/>
      <polyline points="{_points(temperatures, float(config['minimum_temperature']), float(config['maximum_temperature']))}" class="temp"/>
      <polyline points="{_points(phases, float(config['minimum_phase_strength']), float(config['maximum_phase_strength']))}" class="phase"/>
      <polyline points="{_points(thresholds, float(config['minimum_nucleation_threshold']), float(config['maximum_nucleation_threshold']))}" class="threshold"/>
    </svg>
    <div class="legend"><span class="temp-key">temperature</span><span class="phase-key">phase strength</span><span class="threshold-key">nucleation threshold</span></div>
    """


def _control_rows(model: Mapping[str, Any]) -> str:
    rows = []
    for item in model["controller"]["history"]:
        rows.append(
            "<tr>"
            f"<td>{_escape(item['window'])}</td>"
            f"<td>{_escape(item['action'])}</td>"
            f"<td>{float(item['surprise_rate']):.3f}</td>"
            f"<td>{float(item['coherence']):.3f}</td>"
            f"<td>{float(item['acceptance_ratio']):.3f}</td>"
            f"<td>{float(item['order_parameter']):.3f}</td>"
            f"<td>{float(item['chaos_load']):.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _law_cards(model: Mapping[str, Any]) -> str:
    return "".join(
        (
            '<article class="law">'
            f"<span>{_escape(row['cue'])}</span>"
            f"<strong>{_escape(row['effect'])}</strong>"
            f"<small>strength {float(row['strength']):.3f}</small>"
            "</article>"
        )
        for row in model["laws"]
    )


def _primitive_rows(report: Mapping[str, Any]) -> str:
    return "".join(
        (
            '<div class="primitive">'
            f"<span>{_escape(name)}</span><p>{_escape(description)}</p>"
            "</div>"
        )
        for name, description in report["atomic_governor"].items()
    )


def _workflow_rows(workflow: Mapping[str, Any]) -> str:
    return "".join(
        (
            '<article class="turn">'
            f"<span>{_escape(turn['turn_id'])}</span>"
            f"<strong>{_escape(turn['cue'])} &rarr; {_escape(turn['effect'])}</strong>"
            f"<small>{len(turn['knowledge_context'])} graph context nodes</small>"
            "</article>"
        )
        for turn in workflow["turns"]
    )


def _gate_rows(report: Mapping[str, Any]) -> str:
    return "".join(
        (
            '<div class="gate"><i></i>'
            f"<span>{_escape(name.replace('_', ' '))}</span>"
            f"<strong>{'PASS' if passed else 'FAIL'}</strong></div>"
        )
        for name, passed in sorted(report["experiment_gates"]["checks"].items())
    )


def render_homeostatic_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    """Render the learned laws and real controller trajectory into one document."""

    if workflow["runtime"]["model_hash"] != model["model_hash"]:
        raise ValueError("side-view workflow is not bound to the model")
    adaptive = report["comparison"]["adaptive"]
    fixed = report["comparison"]["fixed"]
    metrics = "".join(
        (
            _metric("adaptive final", adaptive["final"]["accuracy"]),
            _metric("fixed final", fixed["final"]["accuracy"]),
            _metric("noise retention", adaptive["prequential"]["noise_burst"]["accuracy"]),
            _metric("final consolidation", adaptive["prequential"]["consolidation"]["accuracy"]),
            _metric("reheats", model["training"]["reheats"]),
            _metric("cools", model["training"]["cools"]),
            _metric("chaos load", model["controller"]["chaos_load"]),
            _metric("windows", len(model["controller"]["history"])),
        )
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atom Homeostatic Governor</title>
  <style>
    :root {{ color-scheme:dark; --ink:#f7f5ef; --muted:#a8b8c4; --panel:#101923;
      --line:#294151; --cyan:#53e1d3; --amber:#ffbd69; --violet:#ac96ff; --red:#ff718c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-width:0; color:var(--ink); font:15px/1.5 Inter,Segoe UI,sans-serif;
      background:radial-gradient(circle at 10% 4%,#183b40 0,#071018 38%,#05080d 100%); }}
    .shell {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(320px,400px); gap:20px;
      max-width:1500px; margin:auto; padding:24px; align-items:start; }}
    main,.side {{ min-width:0; }} .side {{ position:sticky; top:20px; }}
    .hero,.panel {{ min-width:0; padding:22px; margin-bottom:20px; border:1px solid var(--line);
      border-radius:18px; background:linear-gradient(145deg,rgba(16,25,35,.97),rgba(7,13,20,.97));
      box-shadow:0 18px 55px rgba(0,0,0,.28); }} .hero {{ padding:28px; }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:800; }}
    h1 {{ margin:.25rem 0 .5rem; font-size:clamp(31px,5vw,58px); line-height:1.02; }}
    h2 {{ margin:0 0 14px; font-size:21px; }} p,small {{ color:var(--muted); }}
    code {{ color:#d6faf4; overflow-wrap:anywhere; }} .hash {{ display:block; margin-top:16px; padding:10px 12px;
      border:1px solid var(--line); border-radius:10px; background:#050a0f; color:#91a9b8; overflow-wrap:anywhere; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:22px; }}
    .metric {{ min-width:0; padding:12px; border:1px solid var(--line); border-radius:12px; background:#09131c; }}
    .metric span {{ display:block; color:var(--muted); font-size:10px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:3px; font-size:21px; color:var(--amber); overflow-wrap:anywhere; }}
    .trajectory {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:12px; background:#07111a; }}
    .axis {{ stroke:#38505e; stroke-width:1; }} polyline {{ fill:none; stroke-width:3; vector-effect:non-scaling-stroke; }}
    .temp {{ stroke:var(--amber); }} .phase {{ stroke:var(--cyan); }} .threshold {{ stroke:var(--violet); }}
    .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:10px; color:var(--muted); font-size:12px; }}
    .legend span::before {{ content:""; display:inline-block; width:12px; height:3px; margin:0 6px 3px 0; }}
    .temp-key::before {{ background:var(--amber); }} .phase-key::before {{ background:var(--cyan); }}
    .threshold-key::before {{ background:var(--violet); }}
    .laws {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .law {{ min-width:0; padding:14px; border:1px solid var(--line); border-radius:12px; background:#09131c; }}
    .law span,.law strong,.law small {{ display:block; overflow-wrap:anywhere; }} .law span {{ color:var(--cyan); }}
    .law strong {{ margin:5px 0; font-size:20px; }}
    .table-wrap {{ overflow-x:auto; }} table {{ width:100%; min-width:790px; border-collapse:collapse; }}
    th,td {{ padding:9px; text-align:left; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:10px; text-transform:uppercase; }} td:nth-child(2) {{ color:var(--cyan); }}
    .primitive {{ padding:11px 0; border-bottom:1px solid var(--line); }} .primitive span {{ color:var(--amber); font-weight:700; }}
    .primitive p {{ margin:3px 0 0; }} .turn {{ display:grid; gap:3px; padding:11px 0; border-bottom:1px solid var(--line); }}
    .turn span {{ color:var(--amber); font-size:10px; text-transform:uppercase; }} .turn strong {{ overflow-wrap:anywhere; }}
    .gate {{ display:grid; grid-template-columns:10px minmax(0,1fr) auto; gap:9px; padding:8px 0; border-bottom:1px solid var(--line); }}
    .gate i {{ width:8px; height:8px; margin-top:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan); }}
    .gate span {{ overflow-wrap:anywhere; }} .gate strong {{ color:var(--cyan); font-size:10px; }}
    .runtime {{ color:#6f8494; font-size:11px; overflow-wrap:anywhere; }}
    @media (max-width:980px) {{ .shell {{ grid-template-columns:1fr; }} .side {{ position:static; }}
      .metrics,.laws {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:620px) {{ .shell {{ padding:12px; gap:12px; }} .hero,.panel {{ padding:16px; border-radius:14px; }}
      .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .laws {{ grid-template-columns:1fr; }}
      h1 {{ font-size:34px; }} .metric strong {{ font-size:18px; }} .table-wrap {{ overflow:visible; }}
      table {{ display:block; min-width:0; }} thead {{ display:none; }} tbody {{ display:grid; gap:12px; }}
      tr {{ display:grid; gap:7px; padding:12px; border:1px solid var(--line); border-radius:12px; background:#09131c; }}
      td {{ display:block; min-width:0; padding:0; border:0; overflow-wrap:anywhere; }}
      td::before {{ display:block; color:var(--muted); font-size:9px; font-weight:700; text-transform:uppercase; }}
      td:nth-child(1)::before {{ content:"window"; }} td:nth-child(2)::before {{ content:"action"; }}
      td:nth-child(3)::before {{ content:"surprise"; }} td:nth-child(4)::before {{ content:"coherence"; }}
      td:nth-child(5)::before {{ content:"acceptance"; }} td:nth-child(6)::before {{ content:"order"; }}
      td:nth-child(7)::before {{ content:"chaos load"; }} }}
  </style>
</head>
<body data-runtime="{_escape(ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME)}" data-binding="{_escape(ATOM_HOMEOSTATIC_ARTIFACT_BINDING)}">
  <div class="shell">
    <main>
      <section class="hero">
        <span class="eyebrow">homeostasis &middot; atom-composed metaplasticity</span>
        <h1>Chaos becomes a controlled quantity.</h1>
        <p>The field distinguishes incoherent disturbance from coherent law change, then regulates exploration, perturbation, and commitment without receiving regime labels.</p>
        <code class="hash">model {_escape(model['model_hash'])}</code>
        <div class="metrics">{metrics}</div>
      </section>
      <section class="panel"><h2>Deterministic controller trajectory</h2>{_trajectory(model)}</section>
      <section class="panel"><h2>Crystallized final laws</h2><div class="laws">{_law_cards(model)}</div></section>
      <section class="panel"><h2>Window observables and actions</h2><div class="table-wrap"><table><thead><tr><th>window</th><th>action</th><th>surprise</th><th>coherence</th><th>acceptance</th><th>order</th><th>chaos load</th></tr></thead><tbody>{_control_rows(model)}</tbody></table></div></section>
    </main>
    <aside class="side">
      <section class="panel"><h2>Seven-primitive governor</h2>{_primitive_rows(report)}</section>
      <section class="panel"><h2>Serialized workflow</h2>{_workflow_rows(workflow)}</section>
      <section class="panel"><h2>Experiment gates</h2>{_gate_rows(report)}</section>
      <p class="runtime">{_escape(ATOM_HOMEOSTATIC_SIDE_VIEW_RUNTIME)} &middot; {_escape(ATOM_HOMEOSTATIC_ARTIFACT_BINDING)}</p>
    </aside>
  </div>
</body>
</html>"""
    if json.dumps(model["laws"], sort_keys=True) in document:
        raise AssertionError("side view must render, not dump, learned laws")
    return document
