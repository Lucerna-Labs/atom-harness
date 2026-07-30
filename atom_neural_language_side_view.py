"""Model-bound side view for the lifelong Atom neural language field."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME = "atom-neural-language-side-view-v2"


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _metric(label: str, value: str) -> str:
    return (
        '<div class="metric"><span>'
        + html.escape(label)
        + "</span><strong>"
        + html.escape(value)
        + "</strong></div>"
    )


def _validate_binding(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    model_hash = model.get("model_hash")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("model hash is missing from the side-view artifact")
    if report.get("model_hash") != model_hash:
        raise ValueError("report is not bound to the supplied model")
    if workflow.get("model_hash") != model_hash:
        raise ValueError("workflow is not bound to the supplied model")
    if (
        report.get("side_view_contract", {}).get("runtime")
        != ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME
    ):
        raise ValueError("side-view runtime contract does not match")
    return model_hash


def render_neural_language_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    model_hash = _validate_binding(model, report, workflow)
    evaluations = report["evaluations"]
    base = evaluations["base"]["base_composition"]
    adaptive = evaluations["adaptive"]["transfer_composition"]
    fixed = evaluations["fixed"]["transfer_composition"]
    retention = evaluations["adaptive"]["base_composition"]
    zero_shot = evaluations["adaptive"]["zero_shot_composition"]
    flat = evaluations["flat"]["base_composition"]
    controller = report["controller"]
    evidence = report["evidence_boundary"]
    sample_efficiency = report["sample_efficiency"]
    gates = report["experiment_gates"]["checks"]

    metrics = "".join(
        (
            _metric("base composition", _number(base["joint_accuracy"])),
            _metric("adaptive transfer", _number(adaptive["joint_accuracy"])),
            _metric("fixed transfer", _number(fixed["joint_accuracy"])),
            _metric("base retention", _number(retention["joint_accuracy"])),
            _metric("zero-shot control", _number(zero_shot["joint_accuracy"])),
            _metric("flat composition", _number(flat["joint_accuracy"])),
            _metric(
                "unsupported abstention",
                _number(evidence["unsupported"]["correct_abstention_rate"]),
            ),
            _metric(
                "recurrent work saved",
                _number(evidence["compute"]["reduction"]),
            ),
            _metric(
                "quarter-data composition",
                _number(sample_efficiency["evaluation"]["joint_accuracy"]),
            ),
        )
    )
    layers = (
        (
            "1",
            "Consequence inducer",
            "replays root mechanics against observed before/after fields",
        ),
        (
            "2",
            "Operator lexicon",
            "crystallizes opaque tokens into the seven reusable controls",
        ),
        (
            "3",
            "Atom text field",
            "phase-mixes language through seven simultaneous neural branches",
        ),
        (
            "4",
            "Root field executor",
            "applies the fixed universe mechanics to the supplied world graph",
        ),
        (
            "5",
            "Query/surface memory",
            "discovers opaque questions and emits grounded response tokens",
        ),
        (
            "6",
            "Homeostatic governor",
            "accepts coherent novelty and cools inconsistent evidence",
        ),
        (
            "7",
            "Evidence-bound claim gate",
            "asserts only when operator, derivation, query, and surface laws form a complete path",
        ),
        (
            "8",
            "Adaptive latent compute",
            "skips unsupported work and uses one text-field tick on crystallized requests",
        ),
    )
    layer_html = "".join(
        '<article class="layer"><b>'
        + number
        + "</b><div><strong>"
        + html.escape(name)
        + "</strong><span>"
        + html.escape(description)
        + "</span></div></article>"
        for number, name, description in layers
    )
    stage_rows = "".join(
        "<tr><td>"
        + html.escape(stage)
        + "</td><td>"
        + str(values["windows"])
        + "</td><td>"
        + str(values["accepted"])
        + "</td><td>"
        + str(values["rejected"])
        + "</td></tr>"
        for stage, values in controller["stage_counts"].items()
    )
    ablations = report["ablations"]
    ablation_html = "".join(
        '<article class="ablation"><strong>'
        + html.escape(name)
        + "</strong><span>text &Delta; "
        + _number(ablations["text"][name]["maximum_delta"], 4)
        + "</span><span>field &Delta; "
        + _number(ablations["field"][name]["maximum_delta"], 4)
        + "</span></article>"
        for name in ablations["text"]
    )
    workflow_html = "".join(
        '<article class="turn"><span>'
        + html.escape(str(turn["turn_id"]))
        + "</span><strong>"
        + html.escape(
            str(turn["artifact"]["response"])
            if turn["artifact"]["response"] is not None
            else "ABSTAIN"
        )
        + "</strong><small>"
        + html.escape(str(turn["artifact"]["claim_status"]))
        + " &middot; text ticks "
        + str(turn["artifact"]["reasoning"]["text_ticks_used"])
        + " &middot; graph context "
        + str(len(turn["knowledge_context"]))
        + " nodes</small></article>"
        for turn in workflow["turns"]
    )
    gate_html = "".join(
        '<article class="gate"><span>'
        + html.escape(name.replace("_", " "))
        + "</span><strong>"
        + ("PASS" if passed else "FAIL")
        + "</strong></article>"
        for name, passed in sorted(gates.items())
    )
    architecture_payload = html.escape(
        json.dumps(
            {
                "architecture": report["architecture"],
                "dataset_stages": report["dataset"]["stages"],
                "model_hash": model_hash,
                "parameters": report["parameter_counts"],
            },
            sort_keys=True,
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Lifelong Neural Language Field</title>
<style>
:root{{--bg:#081414;--panel:#0d1d20;--line:#264044;--ink:#f4f0e5;--muted:#9eafb0;--mint:#55e0c1;--gold:#f2b65d;--violet:#a98fff}}
*{{box-sizing:border-box}}html,body{{margin:0;background:radial-gradient(circle at 15% 5%,#123839 0,#081414 34%);color:var(--ink);font:15px/1.48 Inter,Segoe UI,sans-serif}}
.shell{{display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:20px;padding:22px;max-width:1500px;margin:auto}}
main,.side{{min-width:0}}.panel,.hero,.side section{{background:linear-gradient(145deg,rgba(15,33,36,.98),rgba(8,20,23,.98));border:1px solid var(--line);border-radius:20px;box-shadow:0 18px 55px #0005}}
.hero{{padding:30px}}.eyebrow{{color:var(--mint);font-size:.78rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}}
h1{{font-size:clamp(2.1rem,5vw,4.8rem);line-height:.98;letter-spacing:-.045em;margin:.35em 0 .25em;max-width:900px}}h2{{font-size:1.35rem;margin:0 0 16px}}p{{color:var(--muted);max-width:850px}}
.hash{{display:block;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:#061012;color:#a8c5c6;overflow-wrap:anywhere}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px;margin-top:20px}}.metric{{border:1px solid var(--line);border-radius:13px;padding:13px;background:#091719}}.metric span{{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase}}.metric strong{{display:block;color:var(--gold);font-size:1.35rem;margin-top:3px}}
.panel{{padding:22px;margin-top:20px}}.layers{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}.layer{{display:flex;gap:12px;padding:14px;border:1px solid var(--line);border-radius:13px}}.layer>b{{display:grid;place-items:center;min-width:32px;height:32px;border-radius:50%;background:#173f3d;color:var(--mint)}}.layer strong,.layer span{{display:block}}.layer span{{color:var(--muted);font-size:.85rem;margin-top:3px}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--mint);font-size:.72rem;text-transform:uppercase}}
.ablations{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}}.ablation{{border:1px solid var(--line);border-radius:12px;padding:12px}}.ablation strong,.ablation span{{display:block}}.ablation span{{color:var(--muted);font-size:.78rem;margin-top:4px}}
.side{{display:flex;flex-direction:column;gap:18px}}.side section{{padding:21px}}.turn,.gate{{display:grid;grid-template-columns:1fr auto;gap:5px 12px;padding:10px 0;border-bottom:1px solid var(--line)}}.turn small{{grid-column:1/-1;color:var(--muted)}}.turn strong{{color:var(--gold)}}.gate strong{{color:var(--mint);font-size:.76rem}}.runtime{{color:var(--muted);font-size:.76rem;overflow-wrap:anywhere}}
@media(max-width:900px){{.shell{{grid-template-columns:1fr;padding:12px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.layers{{grid-template-columns:1fr}}.ablations{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:520px){{.hero,.panel,.side section{{padding:16px}}h1{{font-size:2.25rem}}.ablations{{grid-template-columns:1fr}}table,thead,tbody,tr,th,td{{display:block}}thead{{position:absolute;left:-9999px}}tr{{padding:9px 0;border-bottom:1px solid var(--line)}}td{{display:grid;grid-template-columns:1fr 1fr;border:0;padding:5px}}td:before{{color:var(--muted);font-size:.72rem}}td:nth-child(1):before{{content:'stage'}}td:nth-child(2):before{{content:'windows'}}td:nth-child(3):before{{content:'accepted'}}td:nth-child(4):before{{content:'rejected'}}}}
</style>
</head>
<body>
<div class="shell">
<main>
<section class="hero">
<div class="eyebrow">lifelong language &middot; consequence-grounded neural field</div>
<h1>Language becomes executable field control.</h1>
<p>Opaque utterances are grounded through observed consequences and composed into the seven root mechanics. A factorized surface memory, evidence gate, and adaptive recurrent budget separate fluent candidates from supported assertions while avoiding work on unsupported requests.</p>
<code class="hash">model {html.escape(model_hash)}</code>
<div class="metrics">{metrics}</div>
</section>
<section class="panel"><h2>Architecture composed outward</h2><div class="layers">{layer_html}</div></section>
<section class="panel"><h2>Homeostatic adaptation stream</h2><div class="table-wrap"><table><thead><tr><th>stage</th><th>windows</th><th>accepted</th><th>rejected</th></tr></thead><tbody>{stage_rows}</tbody></table></div></section>
<section class="panel"><h2>Dual-layer causal ablations</h2><div class="ablations">{ablation_html}</div></section>
</main>
<aside class="side">
<section><h2>Serialized workflow</h2>{workflow_html}</section>
<section><h2>Experiment gates</h2>{gate_html}</section>
<section><h2>Bound artifact</h2><p class="runtime">{architecture_payload}</p><p class="runtime">{ATOM_NEURAL_LANGUAGE_SIDE_VIEW_RUNTIME} &middot; render_neural_language_artifact</p></section>
</aside>
</div>
</body>
</html>"""
