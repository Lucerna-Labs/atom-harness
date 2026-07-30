"""Model-bound side view for the natural-English Atom language shell."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME = "atom-english-language-side-view-v1"


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
) -> str:
    model_hash = model.get("model_hash")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ValueError("English side view requires a model hash")
    if report.get("model_hash") != model_hash:
        raise ValueError("English report is not bound to the model")
    if workflow.get("model_hash") != model_hash:
        raise ValueError("English workflow is not bound to the model")
    contract = report.get("side_view_contract", {})
    if contract.get("runtime") != ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME:
        raise ValueError("English side-view runtime does not match")
    return model_hash


def render_english_language_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    model_hash = _validate_binding(model, report, workflow)
    evaluations = report["evaluations"]
    evidence = report["evidence_boundary"]
    sample = report["sample_efficiency"]
    metrics = "".join(
        (
            _metric(
                "English transfer",
                evaluations["selected"]["transfer_composition"]["joint_accuracy"],
            ),
            _metric(
                "Fixed transfer",
                evaluations["fixed"]["transfer_composition"]["joint_accuracy"],
            ),
            _metric(
                "Base retention",
                evaluations["selected"]["base_composition"]["joint_accuracy"],
            ),
            _metric(
                "Flat composition",
                evaluations["flat"]["base_composition"]["joint_accuracy"],
            ),
            _metric("Quarter-data English", sample["evaluation"]["joint_accuracy"]),
            _metric(
                "Unknown abstention",
                evidence["unsupported"]["correct_abstention_rate"],
            ),
            _metric("Recurrent work saved", evidence["compute"]["reduction"]),
            _metric("Atom parameters", report["parameter_counts"]["atom"]),
            _metric("Selected policy", report["selection"]["policy"]),
        )
    )
    layers = (
        ("English codec", "normalizes user English and renders grounded answers"),
        ("Consequence inducer", "learns action words from world changes"),
        ("Atom recurrent field", "composes learned actions through seven branches"),
        ("Factorized memory", "separates query meaning from answer language"),
        ("Homeostatic adaptation", "learns coherent synonyms and rejects noise"),
        ("Evidence gate", "keeps unsupported candidates from becoming assertions"),
        ("Adaptive compute", "uses one text tick or skips the field entirely"),
    )
    layer_html = "".join(
        '<article class="layer"><strong>'
        + html.escape(name)
        + "</strong><span>"
        + html.escape(description)
        + "</span></article>"
        for name, description in layers
    )
    turns = "".join(
        '<article class="turn"><p>'
        + html.escape(str(turn["artifact"]["user_utterance"]))
        + "</p><strong>"
        + html.escape(str(turn["artifact"]["answer"]))
        + "</strong><small>"
        + html.escape(str(turn["artifact"]["claim_status"]))
        + " &middot; text ticks "
        + str(turn["artifact"]["reasoning"]["text_ticks_used"])
        + " &middot; evidence entries "
        + str(len(turn["artifact"]["evidence_path"]))
        + "</small></article>"
        for turn in workflow["turns"]
    )
    gates = "".join(
        '<article class="gate"><span>'
        + html.escape(name.replace("_", " "))
        + "</span><strong>"
        + ("PASS" if passed else "FAIL")
        + "</strong></article>"
        for name, passed in sorted(report["experiment_gates"]["checks"].items())
    )
    binding = html.escape(
        json.dumps(
            {
                "architecture": report["architecture"],
                "behavior_sha256": report["behavior_contract"]["behavior_sha256"],
                "core_model_hash": report["core_model_hash"],
                "model_hash": model_hash,
                "parameters": report["parameter_counts"],
            },
            sort_keys=True,
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Evidence-Bound English</title>
<style>
:root{{--bg:#071216;--panel:#0c1c22;--line:#24414a;--ink:#f6f1e7;--muted:#9eb2b8;--mint:#5ce1bd;--gold:#f1b85b}}
*{{box-sizing:border-box}}html,body{{margin:0;background:radial-gradient(circle at 8% 3%,#173a3c,var(--bg) 35%);color:var(--ink);font:15px/1.5 Inter,Segoe UI,sans-serif}}
.shell{{display:grid;grid-template-columns:minmax(0,2fr) minmax(340px,1fr);gap:20px;max-width:1500px;margin:auto;padding:22px}}main,aside{{min-width:0}}
.hero,.panel,aside section{{background:linear-gradient(145deg,#10252aee,#09171cee);border:1px solid var(--line);border-radius:20px;box-shadow:0 20px 55px #0005}}
.hero,.panel,aside section{{padding:24px}}.panel{{margin-top:20px}}aside{{display:flex;flex-direction:column;gap:18px}}
.eyebrow{{color:var(--mint);font-size:.75rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase}}h1{{font-size:clamp(2.3rem,5vw,4.9rem);line-height:.98;letter-spacing:-.05em;margin:.3em 0}}h2{{margin:0 0 14px}}p{{color:var(--muted)}}
.hash{{display:block;padding:11px;border:1px solid var(--line);background:#061014;border-radius:10px;overflow-wrap:anywhere}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:9px;margin-top:18px}}.metric{{border:1px solid var(--line);border-radius:12px;padding:12px;background:#08161b}}.metric span{{display:block;color:var(--muted);font-size:.67rem;text-transform:uppercase}}.metric strong{{display:block;color:var(--gold);font-size:1.25rem;margin-top:4px}}
.layers{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}}.layer{{padding:13px;border:1px solid var(--line);border-radius:12px}}.layer strong,.layer span{{display:block}}.layer span{{color:var(--muted);font-size:.83rem;margin-top:4px}}
.turn,.gate{{padding:11px 0;border-bottom:1px solid var(--line)}}.turn p{{margin:0 0 5px;font-size:.84rem}}.turn strong,.turn small{{display:block}}.turn strong{{color:var(--gold)}}.turn small{{color:var(--muted);margin-top:4px}}.gate{{display:grid;grid-template-columns:1fr auto;gap:12px}}.gate strong{{color:var(--mint);font-size:.74rem}}.runtime{{font-size:.75rem;overflow-wrap:anywhere}}
@media(max-width:900px){{.shell{{grid-template-columns:1fr;padding:12px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:520px){{.hero,.panel,aside section{{padding:16px}}.layers,.metrics{{grid-template-columns:1fr}}h1{{font-size:2.35rem}}}}
</style></head><body><div class="shell"><main>
<section class="hero"><div class="eyebrow">natural English &middot; evidence-bound Atom cognition</div><h1>English is the interface. Evidence controls the answer.</h1><p>The compact shell learns action vocabulary from consequences, composes it through the Atom field, and renders ordinary English only when the model can trace a complete support path.</p><code class="hash">model {html.escape(model_hash)}</code><div class="metrics">{metrics}</div></section>
<section class="panel"><h2>Hybrid architecture</h2><div class="layers">{layer_html}</div></section>
</main><aside><section><h2>Real English workflow</h2>{turns}</section><section><h2>Experiment gates</h2>{gates}</section><section><h2>Bound artifact</h2><p class="runtime">{binding}</p><p class="runtime">{ATOM_ENGLISH_LANGUAGE_SIDE_VIEW_RUNTIME} &middot; render_english_language_artifact</p></section></aside>
</div></body></html>"""
