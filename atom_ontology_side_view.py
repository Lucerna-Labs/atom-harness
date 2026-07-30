"""User-visible side artifact for emergent ontology discovery."""

from __future__ import annotations

import html
import json
import math
from typing import Any, Mapping, Sequence


ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME = "atom-ontology-side-view-v1"
ATOM_ONTOLOGY_ARTIFACT_BINDING = "render_ontology_artifact"


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
    if isinstance(value, float):
        rendered = f"{value:.3f}"
    else:
        rendered = str(value)
    return (
        '<div class="metric"><span>'
        + _escape(label)
        + "</span><strong>"
        + _escape(rendered)
        + "</strong></div>"
    )


def _type_cards(types: Sequence[Mapping[str, Any]]) -> str:
    cards: list[str] = []
    for atom in types:
        roles = "".join(f"<li>{_escape(role)}</li>" for role in atom["roles"])
        cards.append(
            '<article class="atom type-atom"><span class="eyebrow">type atom</span>'
            f"<h3>{_escape(atom['type_id'])}</h3><ul>{roles}</ul></article>"
        )
    return "".join(cards)


def _relation_cards(relations: Sequence[Mapping[str, Any]]) -> str:
    cards: list[str] = []
    for atom in relations:
        badge = "null-bearing" if atom["nullable"] else "total"
        cards.append(
            '<article class="atom relation-atom"><span class="eyebrow">relation atom</span>'
            f"<h3>{_escape(atom['relation_id'])}</h3>"
            f"<p><code>{_escape(atom['domain_type'])}</code><b> &rarr; </b>"
            f"<code>{_escape(atom['range_type'])}</code></p>"
            f'<span class="badge">{_escape(badge)}</span></article>'
        )
    return "".join(cards)


def _law_rows(laws: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for law in laws:
        pattern = " ".join(str(piece) for piece in law["pattern"])
        effects = " ; ".join(
            f"{effect['relation_id']}[slot {effect['key_slot']}]: "
            f"{effect['before']} -> {effect['after']}"
            for effect in law["effects"]
        )
        rows.append(
            "<tr>"
            f"<td><code>{_escape(law['law_id'])}</code></td>"
            f"<td><code>{_escape(pattern)}</code></td>"
            f'<td class="effect">{_escape(effects)}</td>'
            f"<td>{_escape(law['evidence_count'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _gate_rows(gates: Mapping[str, Any]) -> str:
    return "".join(
        '<div class="gate"><span class="dot"></span><span>'
        + _escape(name.replace("_", " "))
        + "</span><strong>"
        + ("PASS" if passed else "FAIL")
        + "</strong></div>"
        for name, passed in sorted(gates["checks"].items())
    )


def _workflow_rows(workflow: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for turn in workflow["turns"]:
        binding = ", ".join(
            f"{alias} -> {relation}"
            for alias, relation in sorted(turn["ontology_binding"].items())
        )
        rows.append(
            '<article class="turn">'
            f'<span class="turn-id">{_escape(turn["turn_id"])}</span>'
            f"<h3>{_escape(turn['text'])}</h3>"
            f"<p><code>{_escape(turn['law_id'])}</code></p>"
            f'<p class="binding">local bind: {_escape(binding)}</p>'
            "</article>"
        )
    return "".join(rows)


def render_ontology_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    """Render the actual learned model and exercised held-out workflow."""

    if workflow["runtime"]["model_hash"] != model["model_hash"]:
        raise ValueError("side artifact workflow is not bound to the model")
    evaluation = report["evaluation"]
    validation = evaluation["validation"]
    heldout = evaluation["heldout"]
    phase_energy = _finite(
        report["controlled_chaos"]["cumulative_phase_energy"],
        "cumulative phase energy",
    )
    metrics = "".join(
        (
            _metric("types", len(model["ontology"]["types"])),
            _metric("relations", len(model["ontology"]["relations"])),
            _metric("effect laws", len(model["transition_laws"])),
            _metric("held-out execution", heldout["execution_accuracy"]),
            _metric("held-out law ID", heldout["law_accuracy"]),
            _metric("held-out generation", heldout["generation_accuracy"]),
            _metric("validation execution", validation["execution_accuracy"]),
            _metric("phase energy", phase_energy),
        )
    )
    aliases = report["dataset"]["relation_aliases"]
    alias_rows = "".join(
        f"<div><span>{_escape(split)}</span><code>{_escape(', '.join(values))}</code></div>"
        for split, values in sorted(aliases.items())
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atom Ontology Field</title>
  <style>
    :root {{ color-scheme: dark; --ink:#f4f2eb; --muted:#aeb8c2; --panel:#111923;
      --line:#2b3d4f; --cyan:#51e5d4; --amber:#ffbd69; --violet:#a997ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-width:0; background:radial-gradient(circle at 12% 8%,#17323d 0,#071018 38%,#05080d 100%);
      color:var(--ink); font:15px/1.5 Inter,Segoe UI,sans-serif; }}
    .shell {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(310px,390px); gap:20px;
      max-width:1500px; margin:auto; padding:24px; align-items:start; }}
    main,.side {{ min-width:0; }} .side {{ position:sticky; top:20px; }}
    .hero,.panel {{ background:linear-gradient(145deg,rgba(17,25,35,.96),rgba(8,14,21,.96));
      border:1px solid var(--line); border-radius:18px; box-shadow:0 18px 55px rgba(0,0,0,.28); }}
    .hero {{ padding:28px; margin-bottom:20px; }} .panel {{ padding:20px; margin-bottom:20px; }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:800; }}
    h1 {{ margin:.25rem 0 .5rem; font-size:clamp(30px,5vw,56px); line-height:1.02; }}
    h2 {{ margin:0 0 14px; font-size:21px; }} h3 {{ margin:5px 0 8px; overflow-wrap:anywhere; }}
    p {{ color:var(--muted); }} code {{ color:#d6f9f4; overflow-wrap:anywhere; }}
    .hash {{ display:block; margin-top:16px; padding:10px 12px; border:1px solid var(--line); border-radius:10px;
      background:#050a0f; color:#91a9b8; overflow-wrap:anywhere; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:22px; }}
    .metric {{ min-width:0; padding:12px; border:1px solid var(--line); border-radius:12px; background:#09131c; }}
    .metric span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:3px; font-size:22px; color:var(--amber); overflow-wrap:anywhere; }}
    .atom-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .atom {{ min-width:0; padding:15px; border:1px solid var(--line); border-radius:14px; background:#0a141e; }}
    .type-atom {{ border-top:3px solid var(--cyan); }} .relation-atom {{ border-top:3px solid var(--violet); }}
    ul {{ margin:8px 0 0; padding-left:18px; color:var(--muted); }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#2b2140; color:#cfc3ff; font-size:12px; }}
    .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; min-width:760px; }}
    th,td {{ padding:10px; text-align:left; vertical-align:top; border-bottom:1px solid var(--line); }}
    th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }} .effect {{ max-width:430px; overflow-wrap:anywhere; }}
    .aliases div {{ display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-bottom:1px solid var(--line); }}
    .aliases span {{ color:var(--muted); }} .aliases code {{ text-align:right; }}
    .gate {{ display:grid; grid-template-columns:10px minmax(0,1fr) auto; gap:9px; align-items:start; padding:8px 0; border-bottom:1px solid var(--line); }}
    .gate .dot {{ width:8px; height:8px; margin-top:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan); }}
    .gate span:nth-child(2) {{ overflow-wrap:anywhere; }} .gate strong {{ color:var(--cyan); font-size:11px; }}
    .turn {{ padding:12px 0; border-bottom:1px solid var(--line); }} .turn-id {{ color:var(--amber); font-size:11px; text-transform:uppercase; }}
    .turn h3 {{ font-size:17px; }} .turn p {{ margin:4px 0; }} .binding {{ overflow-wrap:anywhere; }}
    .runtime {{ font-size:11px; color:#6f8494; overflow-wrap:anywhere; }}
    @media (max-width:980px) {{ .shell {{ grid-template-columns:1fr; }} .side {{ position:static; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:620px) {{ .shell {{ padding:12px; gap:12px; }} .hero,.panel {{ padding:16px; border-radius:14px; }}
      .atom-grid {{ grid-template-columns:1fr; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .metric strong {{ font-size:18px; }} h1 {{ font-size:34px; }}
      .table-wrap {{ overflow:visible; }} table {{ display:block; min-width:0; }} thead {{ display:none; }}
      tbody {{ display:grid; gap:12px; }} tr {{ display:grid; gap:8px; padding:12px; border:1px solid var(--line); border-radius:12px; background:#09131c; }}
      td {{ display:block; min-width:0; padding:0; border:0; overflow-wrap:anywhere; }}
      td::before {{ display:block; color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase; }}
      td:nth-child(1)::before {{ content:"law"; }} td:nth-child(2)::before {{ content:"surface pattern"; }}
      td:nth-child(3)::before {{ content:"simultaneous effect"; }} td:nth-child(4)::before {{ content:"evidence"; }} }}
  </style>
</head>
<body data-runtime="{_escape(ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME)}" data-binding="{_escape(ATOM_ONTOLOGY_ARTIFACT_BINDING)}">
  <div class="shell">
    <main>
      <section class="hero">
        <span class="eyebrow">structural induction &middot; no relation-name carryover</span>
        <h1>Ontology from position, not labels.</h1>
        <p>The field observes opaque tables, discovers reusable type and relation atoms, then executes learned effects after every relation has been renamed.</p>
        <code class="hash">model {_escape(model["model_hash"])}</code>
        <div class="metrics">{metrics}</div>
      </section>
      <section class="panel"><h2>Emergent type atoms</h2><div class="atom-grid">{_type_cards(model["ontology"]["types"])}</div></section>
      <section class="panel"><h2>Emergent relation atoms</h2><div class="atom-grid">{_relation_cards(model["ontology"]["relations"])}</div></section>
      <section class="panel"><h2>Executable effect laws</h2><div class="table-wrap"><table><thead><tr><th>law</th><th>surface pattern</th><th>simultaneous effect</th><th>evidence</th></tr></thead><tbody>{_law_rows(model["transition_laws"])}</tbody></table></div></section>
    </main>
    <aside class="side">
      <section class="panel"><h2>Alias firewall</h2><p>Each split uses a disjoint local vocabulary for the same inferred structure.</p><div class="aliases">{alias_rows}</div></section>
      <section class="panel"><h2>Held-out workflow</h2>{_workflow_rows(workflow)}</section>
      <section class="panel"><h2>Experiment gates</h2>{_gate_rows(report["experiment_gates"])}</section>
      <p class="runtime">{_escape(ATOM_ONTOLOGY_SIDE_VIEW_RUNTIME)} &middot; {_escape(ATOM_ONTOLOGY_ARTIFACT_BINDING)}</p>
    </aside>
  </div>
</body>
</html>"""
    if json.dumps(model["ontology"], sort_keys=True) in document:
        raise AssertionError("side artifact should render, not dump, the ontology JSON")
    return document
