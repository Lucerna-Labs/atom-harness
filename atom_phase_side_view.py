"""User-visible side view for the learned Atom phase-law artifact."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ATOM_SIDE_VIEW_RUNTIME = "atom-phase-artifact-side-view-v1"
ATOM_ARTIFACT_BINDING = "render_phase_artifact"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span class="metric-label">'
        + _escape(label)
        + "</span><strong>"
        + _escape(value)
        + "</strong></div>"
    )


def _ring_svg(symbols: Sequence[Mapping[str, Any]], size: int) -> str:
    center = 260.0
    radius = 190.0
    pieces = [
        '<svg viewBox="0 0 520 520" role="img" aria-label="Learned phase lattice">',
        '<circle cx="260" cy="260" r="190" class="orbit"/>',
    ]
    for symbol in sorted(symbols, key=lambda row: int(row["slot"])):
        slot = int(symbol["slot"])
        angle = -math.pi / 2.0 + 2.0 * math.pi * slot / size
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        pieces.append(
            f'<line x1="{center:.1f}" y1="{center:.1f}" '
            f'x2="{x:.1f}" y2="{y:.1f}" class="spoke"/>'
        )
        pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="30" class="node"/>')
        pieces.append(
            f'<text x="{x:.1f}" y="{y + 5.0:.1f}" '
            f'class="node-label">{_escape(symbol["name"])}</text>'
        )
        pieces.append(
            f'<text x="{x:.1f}" y="{y + 48.0:.1f}" '
            f'class="slot-label">phase {slot}</text>'
        )
    pieces.append('<text x="260" y="252" class="center-title">Atom phase field</text>')
    pieces.append(
        '<text x="260" y="276" class="center-subtitle">opaque symbols self-arranged</text>'
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _law_rows(laws: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for law in sorted(laws, key=lambda row: str(row["operator"])):
        coherence = _finite(law["coherence"], "law coherence")
        rows.append(
            "<tr>"
            f'<td><span class="law-name">{_escape(law["operator"])}</span></td>'
            f"<td>+{int(law['shift'])}</td>"
            f"<td>{coherence:.3f}</td>"
            f"<td>{int(law['evidence_count'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_phase_artifact(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Bind real run artifacts into a side-by-side inspectable HTML view."""

    required_report = {
        "experiment",
        "model_hash",
        "training",
        "evaluation",
        "controlled_chaos",
        "experiment_gates",
    }
    required_model = {"schema_version", "symbols", "laws", "lattice_size"}
    if set(report) < required_report:
        missing = sorted(required_report - set(report))
        raise ValueError(f"report is missing side-view fields: {missing}")
    if set(model) < required_model:
        missing = sorted(required_model - set(model))
        raise ValueError(f"model is missing side-view fields: {missing}")
    symbols = model["symbols"]
    laws = model["laws"]
    size = int(model["lattice_size"])
    if not isinstance(symbols, list) or len(symbols) != size:
        raise ValueError("model symbols must cover the phase lattice")
    if not isinstance(laws, list) or not laws:
        raise ValueError("model laws must be a non-empty list")

    evaluation = report["evaluation"]
    training = report["training"]
    chaos = report["controlled_chaos"]
    gates = report["experiment_gates"]
    heldout = evaluation["heldout_single_step"]
    two_step = evaluation["unseen_two_step"]
    three_step = evaluation["unseen_three_step"]
    cards = "".join(
        (
            _metric_card(
                "Training energy", f"{_finite(training['final_energy'], 'energy'):.4f}"
            ),
            _metric_card(
                "Held-out one step",
                f"{100.0 * _finite(heldout['accuracy'], 'accuracy'):.1f}%",
            ),
            _metric_card(
                "Unseen two step",
                f"{100.0 * _finite(two_step['accuracy'], 'accuracy'):.1f}%",
            ),
            _metric_card(
                "Unseen three step",
                f"{100.0 * _finite(three_step['accuracy'], 'accuracy'):.1f}%",
            ),
            _metric_card("Persistent laws", len(laws)),
            _metric_card("Raw traces", training["raw_traces_after_abstraction"]),
        )
    )
    gate_rows = "".join(
        f'<li class="{("pass" if passed else "fail")}">'
        f"{_escape(name)}: {('pass' if passed else 'fail')}</li>"
        for name, passed in sorted(gates["gates"].items())
    )
    manifest_json = _escape(
        json.dumps(
            {
                "side_view_runtime": ATOM_SIDE_VIEW_RUNTIME,
                "artifact_binding": ATOM_ARTIFACT_BINDING,
                "experiment": report["experiment"],
                "model_hash": report["model_hash"],
            },
            sort_keys=True,
        )
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atom Phase-Law Run</title>
<style>
:root {{ color-scheme: dark; --ink:#eef4ff; --muted:#9eacc5; --panel:#101827;
  --line:#263552; --cyan:#62e6ff; --violet:#ae8cff; --green:#65e2a3; --red:#ff7c91; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; background:radial-gradient(circle at 20% 0%,#19243f,#080c14 52%);
  color:var(--ink); font:15px/1.5 Inter,Segoe UI,Arial,sans-serif; }}
main {{ max-width:1480px; margin:0 auto; padding:28px; }}
header {{ display:flex; justify-content:space-between; gap:20px; align-items:end; margin-bottom:22px; }}
h1 {{ margin:0; font-size:clamp(26px,4vw,48px); letter-spacing:-.035em; }}
.kicker {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:12px; }}
.hash {{ color:var(--muted); font-family:Consolas,monospace; max-width:430px; overflow-wrap:anywhere; }}
.surface {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(480px,1fr); gap:20px; align-items:start; }}
.primary,.side {{ background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line);
  border-radius:20px; padding:22px; box-shadow:0 18px 70px #0008; }}
.side {{ position:sticky; top:20px; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:18px 0 24px; }}
.metric {{ padding:14px; border:1px solid var(--line); border-radius:12px; background:#0c1321; }}
.metric-label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:5px; }}
.metric strong {{ font-size:20px; }}
h2 {{ margin:20px 0 10px; font-size:18px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:10px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.law-name {{ color:var(--violet); font-weight:700; }}
ul {{ padding:0; margin:0; list-style:none; display:grid; grid-template-columns:1fr 1fr; gap:7px 14px; }}
li {{ padding-left:17px; position:relative; }} li::before {{ content:' '; width:8px; height:8px; border-radius:50%;
  position:absolute; left:0; top:.55em; background:var(--red); }} li.pass::before {{ background:var(--green); }}
.orbit {{ fill:none; stroke:#293959; stroke-width:2; stroke-dasharray:5 8; }}
.spoke {{ stroke:#1f3456; stroke-width:1.4; }} .node {{ fill:#111c30; stroke:var(--cyan); stroke-width:2; }}
.node-label,.slot-label,.center-title,.center-subtitle {{ text-anchor:middle; fill:var(--ink); }}
.node-label {{ font-weight:700; font-size:12px; }} .slot-label,.center-subtitle {{ fill:var(--muted); font-size:11px; }}
.center-title {{ font-size:18px; font-weight:700; }}
.chaos {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }} .chaos div {{ padding:12px; background:#0c1321; border-radius:10px; }}
.manifest {{ margin-top:18px; color:var(--muted); font:11px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
@media (max-width:980px) {{ .surface {{ grid-template-columns:1fr; }} .side {{ position:static; }} }}
</style>
</head>
<body><main>
<header><div><div class="kicker">From-scratch universe-core experiment</div>
<h1>Emergent phase-law learner</h1></div><div class="hash">model {_escape(report["model_hash"])}</div></header>
<div class="surface">
<section class="primary" aria-label="Run results">
<div class="metrics">{cards}</div>
<h2>Crystallized operator laws</h2>
<table><thead><tr><th>Opaque operator</th><th>Phase shift</th><th>Coherence</th><th>Evidence</th></tr></thead>
<tbody>{_law_rows(laws)}</tbody></table>
<h2>Causal gates</h2><ul>{gate_rows}</ul>
<h2>Controlled chaos</h2><div class="chaos">
<div>Temperature<br><strong>{_finite(chaos["initial_temperature"], "temperature"):.3f} to {_finite(chaos["final_temperature"], "temperature"):.3f}</strong></div>
<div>Phase energy<br><strong>{_finite(chaos["cumulative_phase_energy"], "phase energy"):.6f}</strong></div>
</div>
</section>
<aside class="side" aria-label="Learned artifact side view">
{_ring_svg(symbols, size)}
<div class="manifest">{manifest_json}</div>
</aside>
</div></main></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    return output_path
