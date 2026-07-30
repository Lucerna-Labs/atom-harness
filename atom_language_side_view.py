"""User-visible side view for the grounded Atom language artifact."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ATOM_LANGUAGE_SIDE_VIEW_RUNTIME = "atom-language-artifact-side-view-v1"
ATOM_LANGUAGE_ARTIFACT_BINDING = "render_language_artifact"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span>'
        + _escape(label)
        + "</span><strong>"
        + _escape(value)
        + "</strong></div>"
    )


def _law_rows(laws: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for law in sorted(laws, key=lambda row: str(row["surface"])):
        rows.append(
            "<tr>"
            f'<td class="surface-word">{_escape(law["surface"])}</td>'
            f"<td>{_escape(law['concept'])}</td>"
            f"<td>{_finite(law['support'], 'support'):.3f}</td>"
            f"<td>{int(law['evidence_count'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _binding_svg(frame: Mapping[str, Any]) -> str:
    roles = list(sorted(frame.get("roles", {}).items()))
    width = 540
    center_x = 270
    center_y = 245
    radius = 175
    pieces = [
        f'<svg viewBox="0 0 {width} 490" role="img" aria-label="Learned language meaning field">',
        '<circle cx="270" cy="245" r="175" class="orbit"/>',
        f'<text x="{center_x}" y="235" class="center-title">{_escape(frame.get("predicate", "unknown"))}</text>',
        f'<text x="{center_x}" y="260" class="center-subtitle">{_escape(frame.get("speech_act", "unknown"))}</text>',
    ]
    count = max(1, len(roles))
    for index, (role, concept) in enumerate(roles):
        angle = -math.pi / 2 + 2 * math.pi * index / count
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        pieces.append(
            f'<line x1="{center_x}" y1="{center_y}" x2="{x:.1f}" y2="{y:.1f}" class="spoke"/>'
        )
        pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="44" class="node"/>')
        pieces.append(
            f'<text x="{x:.1f}" y="{y - 4:.1f}" class="node-label">{_escape(role)}</text>'
        )
        pieces.append(
            f'<text x="{x:.1f}" y="{y + 15:.1f}" class="node-value">{_escape(concept)}</text>'
        )
    pieces.append("</svg>")
    return "".join(pieces)


def render_language_artifact(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    output_path: Path,
) -> Path:
    required_report = {
        "experiment",
        "primary_model_hash",
        "stages",
        "experiment_gates",
        "side_view_interaction",
        "controlled_chaos",
    }
    required_model = {
        "model_hash",
        "stage",
        "lexeme_laws",
        "frame_laws",
        "reference_laws",
    }
    if set(report) < required_report:
        raise ValueError(
            f"report is missing language side-view fields: {sorted(required_report - set(report))}"
        )
    if set(model) < required_model:
        raise ValueError(
            f"model is missing language side-view fields: {sorted(required_model - set(model))}"
        )
    if report["primary_model_hash"] != model["model_hash"]:
        raise ValueError("language side view requires the report's primary model")

    word = report["stages"]["word"]["heldout"]
    character = report["stages"]["character"]["heldout"]
    spans = report["stages"]["character"]["character_spans"]
    interaction = report["side_view_interaction"]
    metrics = "".join(
        (
            _metric(
                "Word understanding",
                f"{100 * _finite(word['grounded_accuracy'], 'word accuracy'):.1f}%",
            ),
            _metric(
                "Character understanding",
                f"{100 * _finite(character['grounded_accuracy'], 'character accuracy'):.1f}%",
            ),
            _metric(
                "Character generation",
                f"{100 * _finite(character['generation_roundtrip_accuracy'], 'generation accuracy'):.1f}%",
            ),
            _metric("Lexical-span F1", f"{_finite(spans['f1'], 'span f1'):.3f}"),
            _metric("Persistent lexemes", len(model["lexeme_laws"])),
            _metric("Raw episodes", model["raw_episode_count"]),
        )
    )
    gate_rows = "".join(
        f'<li class="{("pass" if passed else "fail")}">{_escape(name)}: {("pass" if passed else "fail")}</li>'
        for name, passed in sorted(report["experiment_gates"]["gates"].items())
    )
    manifest = _escape(
        json.dumps(
            {
                "side_view_runtime": ATOM_LANGUAGE_SIDE_VIEW_RUNTIME,
                "artifact_binding": ATOM_LANGUAGE_ARTIFACT_BINDING,
                "experiment": report["experiment"],
                "model_hash": model["model_hash"],
            },
            sort_keys=True,
        )
    )
    frame = interaction["meaning"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Language Field Run</title>
<style>
:root {{ color-scheme:dark; --ink:#eef4ff; --muted:#9eacc5; --panel:#101827;
  --line:#263552; --cyan:#62e6ff; --violet:#ae8cff; --green:#65e2a3; --red:#ff7c91; }}
* {{ box-sizing:border-box; }} html,body {{ max-width:100%; overflow-x:hidden; }} body {{ margin:0; min-height:100vh;
  background:radial-gradient(circle at 18% 0%,#1c2848,#080c14 54%); color:var(--ink);
  font:15px/1.5 Inter,Segoe UI,Arial,sans-serif; }} main {{ max-width:1500px; margin:auto; padding:28px; }}
header {{ display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:22px; }}
header > div {{ min-width:0; }}
h1 {{ margin:0; font-size:clamp(28px,4vw,50px); letter-spacing:-.04em; }}
.kicker {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:12px; }}
.hash {{ flex:0 1 430px; min-width:280px; max-width:430px; color:var(--muted); font:12px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
.surface {{ display:grid; grid-template-columns:minmax(0,1.08fr) minmax(460px,.92fr); gap:20px; align-items:start; }}
.primary,.side {{ min-width:0; background:color-mix(in srgb,var(--panel) 93%,transparent); border:1px solid var(--line);
  border-radius:20px; padding:22px; box-shadow:0 18px 70px #0008; }} .side {{ position:sticky; top:20px; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
.metric {{ padding:14px; border:1px solid var(--line); border-radius:12px; background:#0c1321; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; }} .metric strong {{ font-size:20px; }}
h2 {{ margin:22px 0 10px; font-size:18px; }} table {{ width:100%; border-collapse:collapse; }}
.table-scroll {{ max-width:100%; overflow-x:auto; }}
th,td {{ padding:9px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
.surface-word {{ color:var(--violet); font-weight:700; }} .turn {{ display:grid; grid-template-columns:100px 1fr; gap:8px;
  padding:12px; margin:8px 0; background:#0c1321; border-radius:12px; }} .turn span {{ color:var(--muted); }}
ul {{ padding:0; list-style:none; columns:2; column-gap:18px; }} li {{ break-inside:avoid; padding:3px 0 3px 16px; position:relative; }}
li::before {{ content:' '; width:8px; height:8px; border-radius:50%; position:absolute; left:0; top:.65em; background:var(--red); }}
li.pass::before {{ background:var(--green); }} .orbit {{ fill:none; stroke:#293959; stroke-width:2; stroke-dasharray:5 8; }}
.spoke {{ stroke:#1f3456; stroke-width:1.5; }} .node {{ fill:#111c30; stroke:var(--cyan); stroke-width:2; }}
.node-label,.node-value,.center-title,.center-subtitle {{ text-anchor:middle; fill:var(--ink); }} .node-label {{ font-weight:700; font-size:13px; }}
.node-value,.center-subtitle {{ fill:var(--muted); font-size:10px; }} .center-title {{ font-size:22px; font-weight:800; }}
.manifest {{ margin-top:14px; color:var(--muted); font:11px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
@media(max-width:1100px) {{ header {{ align-items:start; flex-direction:column; }}
  .hash {{ flex:none; min-width:0; max-width:100%; }} }}
@media(max-width:980px) {{ .surface {{ grid-template-columns:1fr; }} .side {{ position:static; }} }}
@media(max-width:650px) {{ main {{ padding:16px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  ul {{ columns:1; }} }}
</style></head><body><main>
<header><div><div class="kicker">Grounded universe-core language experiment</div><h1>Atom Language Field</h1></div>
<div class="hash">model {_escape(model["model_hash"])}</div></header>
<div class="surface"><section class="primary" aria-label="Language run results">
<div class="metrics">{metrics}</div><h2>Real held-out interaction</h2>
<div class="turn"><span>utterance</span><strong>{_escape(interaction["utterance"])}</strong></div>
<div class="turn"><span>meaning</span><strong>{_escape(frame["speech_act"])} / {_escape(frame["predicate"])}</strong></div>
<div class="turn"><span>answer</span><strong>{_escape(interaction.get("answer") or "world changed")}</strong></div>
<h2>Crystallized lexical laws</h2><div class="table-scroll"><table><thead><tr><th>surface</th><th>grounded atom</th><th>support</th><th>evidence</th></tr></thead>
<tbody>{_law_rows(model["lexeme_laws"])}</tbody></table></div><h2>Causal and language gates</h2><ul>{gate_rows}</ul>
</section><aside class="side" aria-label="Learned language artifact side view">{_binding_svg(frame)}
<div class="turn"><span>temperature</span><strong>{_finite(report["controlled_chaos"]["initial_temperature"], "initial temperature"):.3f} -> {_finite(report["controlled_chaos"]["final_temperature"], "final temperature"):.3f}</strong></div>
<div class="turn"><span>semantic mass</span><strong>{_escape(interaction["semantic_mass"])}</strong></div>
<div class="manifest">{manifest}</div></aside></div></main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    return output_path
