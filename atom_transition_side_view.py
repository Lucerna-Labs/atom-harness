"""User-visible side view for an emergent transition-law artifact."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ATOM_TRANSITION_SIDE_VIEW_RUNTIME = "atom-transition-law-side-view-v1"
ATOM_TRANSITION_ARTIFACT_BINDING = "render_transition_artifact"


def _transition_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _transition_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _transition_metric(label: str, value: Any) -> str:
    return (
        '<div class="metric"><span>'
        + _transition_escape(label)
        + "</span><strong>"
        + _transition_escape(value)
        + "</strong></div>"
    )


def _effect_text(effects: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"{effect['collection']}[slot {effect['key_slot']}] "
        f"{effect['before']} -> {effect['after']}"
        for effect in effects
    )


def _transition_law_rows(
    laws: Sequence[Mapping[str, Any]],
    evaluator_labels: Mapping[str, str],
) -> str:
    rows: list[str] = []
    for law in sorted(laws, key=lambda value: str(value["law_id"])):
        rows.append(
            "<tr>"
            f'<td class="law-id">{_transition_escape(law["law_id"])}</td>'
            f"<td>{_transition_escape(' '.join(law['pattern']))}</td>"
            f"<td>{_transition_escape(_effect_text(law['effects']))}</td>"
            f"<td>{_transition_escape(evaluator_labels.get(str(law['law_id']), 'unlabeled'))}</td>"
            f"<td>{int(law['evidence_count'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _effect_svg(interaction: Mapping[str, Any]) -> str:
    effects = list(interaction["effects"])
    height = max(360, 190 + 118 * len(effects))
    pieces = [
        f'<svg viewBox="0 0 560 {height}" role="img" aria-label="Emergent executable effect program">',
        '<rect x="150" y="30" width="260" height="78" rx="20" class="law-node"/>',
        f'<text x="280" y="63" class="law-title">{_transition_escape(interaction["law_id"])}</text>',
        '<text x="280" y="88" class="law-subtitle">learned identity, not a supplied predicate</text>',
    ]
    for index, effect in enumerate(effects):
        y = 166 + index * 118
        pieces.append(
            f'<line x1="280" y1="108" x2="280" y2="{y}" class="effect-line"/>'
        )
        pieces.append(
            f'<rect x="74" y="{y}" width="412" height="82" rx="16" class="effect-node"/>'
        )
        pieces.append(
            f'<text x="280" y="{y + 31}" class="effect-title">{_transition_escape(effect["collection"])}[slot {_transition_escape(effect["key_slot"])}]</text>'
        )
        pieces.append(
            f'<text x="280" y="{y + 57}" class="effect-subtitle">{_transition_escape(effect["before"])} to {_transition_escape(effect["after"])}</text>'
        )
    pieces.append("</svg>")
    return "".join(pieces)


def render_transition_artifact(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    output_path: Path,
) -> Path:
    required_report = {
        "experiment",
        "primary_model_hash",
        "evaluation",
        "experiment_gates",
        "side_view_interaction",
        "controlled_chaos",
        "evaluator_law_mapping",
    }
    required_model = {
        "model_hash",
        "surface_laws",
        "transition_laws",
        "raw_episode_count",
        "raw_evidence_count",
    }
    if not required_report <= set(report):
        raise ValueError(
            "report is missing transition side-view fields: "
            + str(sorted(required_report - set(report)))
        )
    if not required_model <= set(model):
        raise ValueError(
            "model is missing transition side-view fields: "
            + str(sorted(required_model - set(model)))
        )
    if report["primary_model_hash"] != model["model_hash"]:
        raise ValueError("transition side view requires the report's primary model")
    law_to_label = {
        str(law_id): str(label)
        for label, law_id in report["evaluator_law_mapping"].items()
    }
    heldout = report["evaluation"]["heldout"]
    novel = report["evaluation"]["novel_transitions"]
    metrics = "".join(
        (
            _transition_metric("Discovered laws", len(model["transition_laws"])),
            _transition_metric(
                "Held-out execution",
                f"{100 * _transition_finite(heldout['execution_accuracy'], 'held-out execution'):.1f}%",
            ),
            _transition_metric(
                "Held-out law identity",
                f"{100 * _transition_finite(heldout['law_accuracy'], 'held-out law identity'):.1f}%",
            ),
            _transition_metric(
                "Novel transition execution",
                f"{novel['execution_correct']}/{novel['cases']}",
            ),
            _transition_metric("Grounded lexemes", len(model["surface_laws"])),
            _transition_metric(
                "Raw memory retained",
                model["raw_episode_count"] + model["raw_evidence_count"],
            ),
        )
    )
    gate_rows = "".join(
        f'<li class="{("pass" if passed else "fail")}">{_transition_escape(name)}: {("pass" if passed else "fail")}</li>'
        for name, passed in sorted(report["experiment_gates"]["gates"].items())
    )
    interaction = report["side_view_interaction"]
    manifest = _transition_escape(
        json.dumps(
            {
                "side_view_runtime": ATOM_TRANSITION_SIDE_VIEW_RUNTIME,
                "artifact_binding": ATOM_TRANSITION_ARTIFACT_BINDING,
                "experiment": report["experiment"],
                "model_hash": model["model_hash"],
            },
            sort_keys=True,
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Emergent Transition Laws</title>
<style>
:root {{ color-scheme:dark; --ink:#f2f6ff; --muted:#9caac1; --panel:#101723;
  --line:#26344c; --cyan:#5fe1f6; --amber:#ffc66d; --green:#69dfa5; --red:#ff7d94; }}
* {{ box-sizing:border-box; }} html,body {{ max-width:100%; overflow-x:hidden; }}
body {{ margin:0; min-height:100vh; color:var(--ink); font:15px/1.5 Inter,Segoe UI,Arial,sans-serif;
  background:radial-gradient(circle at 10% 0%,#21324c,#080b12 52%); }}
main {{ max-width:1520px; margin:auto; padding:28px; }} header {{ display:flex; justify-content:space-between;
  align-items:end; gap:24px; margin-bottom:22px; }} header>div {{ min-width:0; }}
.kicker {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:12px; }}
h1 {{ margin:0; font-size:clamp(28px,4vw,52px); letter-spacing:-.045em; }}
.hash {{ flex:0 1 440px; max-width:440px; min-width:280px; color:var(--muted);
  font:12px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
.surface {{ display:grid; grid-template-columns:minmax(0,1.13fr) minmax(430px,.87fr); gap:20px; align-items:start; }}
.primary,.side {{ min-width:0; padding:22px; border:1px solid var(--line); border-radius:20px;
  background:color-mix(in srgb,var(--panel) 94%,transparent); box-shadow:0 18px 70px #0008; }}
.side {{ position:sticky; top:20px; }} .metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
.metric {{ padding:14px; border:1px solid var(--line); border-radius:12px; background:#0b111c; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; }} .metric strong {{ font-size:20px; }}
h2 {{ margin:22px 0 10px; font-size:18px; }} .table-scroll {{ max-width:100%; overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:800px; }} th,td {{ padding:9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }} .law-id {{ color:var(--cyan); font:12px Consolas,monospace; }}
.turn {{ display:grid; grid-template-columns:112px minmax(0,1fr); gap:8px; padding:12px; margin:8px 0;
  background:#0b111c; border-radius:12px; }} .turn span {{ color:var(--muted); }} .turn strong {{ overflow-wrap:anywhere; }}
ul {{ padding:0; list-style:none; columns:2; column-gap:18px; }} li {{ break-inside:avoid; padding:3px 0 3px 16px; position:relative; overflow-wrap:anywhere; }}
li::before {{ content:' '; width:8px; height:8px; border-radius:50%; position:absolute; left:0; top:.65em; background:var(--red); }}
li.pass::before {{ background:var(--green); }} svg {{ width:100%; height:auto; }} .law-node {{ fill:#17253a; stroke:var(--cyan); stroke-width:2; }}
.effect-node {{ fill:#151c29; stroke:var(--amber); stroke-width:2; }} .effect-line {{ stroke:#3c506e; stroke-width:2; }}
.law-title,.law-subtitle,.effect-title,.effect-subtitle {{ text-anchor:middle; fill:var(--ink); }} .law-title,.effect-title {{ font-weight:800; }}
.law-subtitle,.effect-subtitle {{ fill:var(--muted); font-size:12px; }} .manifest {{ margin-top:14px; color:var(--muted); font:11px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
@media(max-width:1060px) {{ header {{ align-items:start; flex-direction:column; }} .hash {{ flex:none; min-width:0; max-width:100%; }}
  .surface {{ grid-template-columns:1fr; }} .side {{ position:static; }} }}
@media(max-width:650px) {{ main {{ padding:16px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} ul {{ columns:1; }} }}
</style></head><body><main>
<header><div><div class="kicker">Universe-core latent law induction</div><h1>Emergent Transition Field</h1></div>
<div class="hash">model {_transition_escape(model["model_hash"])}</div></header>
<div class="surface"><section class="primary" aria-label="Transition discovery results">
<div class="metrics">{metrics}</div><h2>Held-out execution</h2>
<div class="turn"><span>opaque input</span><strong>{_transition_escape(interaction["utterance"])}</strong></div>
<div class="turn"><span>discovered law</span><strong>{_transition_escape(interaction["law_id"])}</strong></div>
<div class="turn"><span>external label</span><strong>{_transition_escape(interaction["evaluator_label"])}</strong></div>
<div class="turn"><span>world delta</span><strong>{_transition_escape(_effect_text(interaction["effects"]))}</strong></div>
<h2>Crystallized executable laws</h2><div class="table-scroll"><table><thead><tr><th>emergent id</th><th>surface law</th><th>effect program</th><th>evaluator only</th><th>evidence</th></tr></thead>
<tbody>{_transition_law_rows(model["transition_laws"], law_to_label)}</tbody></table></div>
<h2>Causal and execution gates</h2><ul>{gate_rows}</ul></section>
<aside class="side" aria-label="Learned transition artifact side view">{_effect_svg(interaction)}
<div class="turn"><span>temperature</span><strong>{_transition_finite(report["controlled_chaos"]["initial_temperature"], "initial temperature"):.3f} to {_transition_finite(report["controlled_chaos"]["final_temperature"], "final temperature"):.3f}</strong></div>
<div class="turn"><span>phase energy</span><strong>{_transition_finite(report["controlled_chaos"]["cumulative_phase_energy"], "phase energy"):.6f}</strong></div>
<div class="manifest">{manifest}</div></aside></div></main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    return output_path
