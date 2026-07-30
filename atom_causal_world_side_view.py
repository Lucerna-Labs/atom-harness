"""User-visible right-side artifact view for the causal-world runtime."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME = "atom-causal-world-side-view-v10"

CURRICULUM_AXIS_ORDER = (
    "scale",
    "resources",
    "signal",
    "relations",
    "time",
    "topology",
    "phase_regime",
    "energy_regime",
    "boundary",
)


def _metric(label: str, value: Any) -> str:
    if isinstance(value, float):
        rendered = f"{value:,.3f}"
    elif isinstance(value, int):
        rendered = f"{value:,}"
    else:
        rendered = str(value)
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
        raise ValueError("causal side view requires a model hash")
    if report.get("model_hash") != model_hash:
        raise ValueError("causal report is not bound to the graph model")
    if workflow.get("model_hash") != model_hash:
        raise ValueError("causal workflow is not bound to the graph model")
    contract = report.get("side_view_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("causal report has no side-view contract")
    if contract.get("runtime") != ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME:
        raise ValueError("causal side-view runtime mismatch")
    if contract.get("artifact_binding_marker") != "render_causal_world_artifact":
        raise ValueError("causal side-view artifact marker mismatch")
    transfer = report.get("transfer_benchmark")
    if not isinstance(transfer, Mapping) or transfer.get("model_hash") != model_hash:
        raise ValueError("causal transfer benchmark is not bound to the graph model")
    calibration = transfer.get("metaplastic_calibration")
    risk_contract = (
        calibration.get("risk_contract")
        if isinstance(calibration, Mapping)
        else None
    )
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("passed") is not True
        or not isinstance(risk_contract, Mapping)
        or not isinstance(transfer.get("transfer_policy_hash"), str)
        or len(transfer["transfer_policy_hash"]) != 64
    ):
        raise ValueError("causal transfer calibration is not bound to the graph model")
    formal = report.get("formal_domains")
    if (
        not isinstance(formal, Mapping)
        or formal.get("passed") is not True
        or not isinstance(formal.get("registry_hash"), str)
        or len(formal["registry_hash"]) != 64
        or not isinstance(formal.get("report_hash"), str)
        or len(formal["report_hash"]) != 64
    ):
        raise ValueError("formal-domain artifact is not bound to the causal report")
    return model_hash


def render_causal_world_artifact(
    model: Mapping[str, Any],
    report: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> str:
    """Render the real graph artifact beside the primary runtime surface."""

    model_hash = _validate_binding(model, report, workflow)
    graph = model["graph"]
    scale = report["world"]["scale"]
    curriculum = report["world"]["curriculum"]
    evaluation = report["evaluation"]
    transfer = report["transfer_benchmark"]
    transfer_evaluation = transfer["contextual_transfer"]
    calibration = transfer["metaplastic_calibration"]
    formal = report["formal_domains"]
    selected_policy = calibration["selected_policy"]
    validation_evaluation = calibration["selected_validation_evaluation"]
    metric_items = [
        _metric("Graph laws", graph["law_count"]),
        _metric("Crystallized", report["learning"]["crystallized_laws"]),
        _metric("Evidence rows", report["learning"]["evidence_count"]),
        _metric("Derived answers", evaluation["derived_answers"]),
        _metric("Honest unknowns", evaluation["unknown_answers"]),
        _metric("Worlds in scale", scale["worlds"]),
        _metric("Entity updates", scale["entity_updates"]),
        _metric("Relation updates", scale["relation_updates"]),
        _metric("Possible regimes", curriculum["procedural_program_space"]),
        _metric("Regimes exercised", curriculum["execution_program_count"]),
        _metric("Held-out regimes", transfer["heldout_program_count"]),
        _metric("Validation regimes", transfer["validation_program_count"]),
        _metric("Policies evaluated", calibration["evaluated_policy_count"]),
        _metric("Formal domains", formal["domain_count"]),
        _metric("Formal primitives", formal["primitive_count"]),
        _metric("Formal cases", formal["case_count"]),
        _metric("Formal held-out", formal["partition_counts"]["heldout"]),
        _metric("Transfer coverage", transfer_evaluation["coverage"]),
        _metric("Selective accuracy", transfer_evaluation["selective_accuracy"]),
        _metric(
            "Selective error upper",
            transfer_evaluation["selective_error_upper_bound"],
        ),
        _metric("Safe transfer utility", transfer_evaluation["safe_direction_utility"]),
        _metric("False assertion rate", transfer_evaluation["false_assertion_rate"]),
        _metric(
            "Negative / positive truth",
            (
                f"{transfer['truth_direction_counts']['-1']} / "
                f"{transfer['truth_direction_counts']['1']}"
            ),
        ),
    ]
    execution = report.get("execution")
    if isinstance(execution, Mapping):
        accelerator = execution.get("accelerator")
        if isinstance(accelerator, Mapping):
            metric_items.extend(
                (
                    _metric("Executed shards", execution["shards_executed"]),
                    _metric("Devices used", accelerator["devices_used"]),
                    _metric("Executor", accelerator["executor_mode"]),
                    _metric("Runtime seconds", accelerator["elapsed_seconds"]),
                    _metric("Paired entity updates", accelerator["entity_updates"]),
                    _metric("Paired relation updates", accelerator["relation_updates"]),
                )
            )
    metrics = "".join(metric_items)
    components = "".join(
        '<article class="component"><strong>'
        + html.escape(component.replace("_", " "))
        + "</strong><span>"
        + html.escape(runtime)
        + "</span></article>"
        for component, runtime in sorted(model["runtimes"].items())
        if component != "active_experiment"
    )
    top_laws = sorted(
        graph["laws"],
        key=lambda law: (
            -float(law["persistence"]),
            -float(law["confidence"]),
            str(law["law_id"]),
        ),
    )[:18]
    laws = "".join(
        '<article class="law"><div><strong>'
        + html.escape(str(law["cause_feature"]))
        + (" &uarr; " if int(law["direction"]) > 0 else " &darr; ")
        + html.escape(str(law["effect_feature"]))
        + "</strong><span>"
        + html.escape(str(law["domain"]))
        + "</span></div><small>support "
        + str(law["support"])
        + " &middot; confidence "
        + f"{float(law['confidence']):.3f}"
        + " &middot; persistence "
        + f"{float(law['persistence']):.3f}"
        + "<br><em>"
        + html.escape(
            ", ".join(
                context
                for context in sorted(law["contexts"])
                if context.split(":", 1)[0]
                in {
                    "scale",
                    "resources",
                    "signal",
                    "relations",
                    "time",
                    "topology",
                    "phase_regime",
                    "energy_regime",
                    "boundary",
                    "primary_root",
                    "secondary_root",
                }
            )
        )
        + "</em>"
        + "</small></article>"
        for law in top_laws
    )
    turns = "".join(
        '<article class="turn"><p>'
        + html.escape(str(turn["request"]["text"]))
        + "</p><strong>"
        + html.escape(str(turn["answer"]))
        + "</strong><small>"
        + html.escape(str(turn["artifact"]["claim_status"]))
        + " &middot; causal entries "
        + str(len(turn["artifact"]["evidence_path"]))
        + " &middot; RAG hits "
        + str(len(turn["knowledge_context"]))
        + "</small></article>"
        for turn in workflow["turns"]
    )
    transfer_samples = "".join(
        '<article class="turn"><p>'
        + html.escape(str(sample["question"]))
        + "</p><strong>expected "
        + ("increase" if int(sample["expected_direction"]) > 0 else "decrease")
        + " &middot; predicted "
        + (
            "unknown"
            if sample["predicted_direction"] is None
            else (
                "increase" if int(sample["predicted_direction"]) > 0 else "decrease"
            )
        )
        + "</strong><small>"
        + html.escape(str(sample["derivation_kind"]).replace("_", " "))
        + " &middot; source laws "
        + str(sample["source_count"])
        + " &middot; pair motifs "
        + str(sample["pair_motif_count"])
        + "</small></article>"
        for sample in transfer["samples"]
    )
    calibration_view = "".join(
        (
            _metric(
                "Direction prior power",
                selected_policy["direction_prior_power"],
            ),
            _metric(
                "Pair motif power",
                selected_policy["pair_motif_power"],
            ),
            _metric(
                "Negative acceptance",
                selected_policy["consensus_thresholds"]["-1"],
            ),
            _metric(
                "Positive acceptance",
                selected_policy["consensus_thresholds"]["1"],
            ),
            _metric(
                "Validation safe utility",
                validation_evaluation["safe_direction_utility"],
            ),
            _metric(
                "Validation selective accuracy",
                validation_evaluation["selective_accuracy"],
            ),
            _metric(
                "Validation false assertion rate",
                validation_evaluation["false_assertion_rate"],
            ),
            _metric(
                "Validation error upper",
                validation_evaluation["selective_error_upper_bound"],
            ),
            _metric(
                "Declared overall risk limit",
                calibration["risk_contract"][
                    "overall_selective_error_upper_limit"
                ],
            ),
            _metric(
                "Declared directional risk limit",
                calibration["risk_contract"][
                    "direction_selective_error_upper_limit"
                ],
            ),
            _metric(
                "Projection lattice digest",
                calibration["probe_response_hashes"][
                    "policy_neutral_projection_lattice"
                ][:12],
            ),
        )
    )
    domain_rows = "".join(
        '<span class="domain">'
        + html.escape(domain)
        + " <strong>"
        + str(count)
        + "</strong></span>"
        for domain, count in sorted(report["world"]["domain_counts"].items())
    )
    formal_domain_rows = "".join(
        '<article class="component"><strong>'
        + html.escape(domain.replace("_", " "))
        + "</strong><span>"
        + str(values["correct"])
        + " / "
        + str(values["cases"])
        + " independently verified</span></article>"
        for domain, values in sorted(formal["per_domain"].items())
    )
    formal_programs = "".join(
        '<article class="turn"><p>'
        + html.escape(program["program_id"].replace("-", " "))
        + "</p><strong>"
        + html.escape(" \u2192 ".join(program["domains"]))
        + "</strong><small>"
        + str(program["stage_count"])
        + " typed stages &middot; "
        + html.escape(program["claim_status"])
        + " &middot; "
        + ("PASS" if program["passed"] else "FAIL")
        + "</small></article>"
        for program in formal["cross_domain_programs"]
    )
    curriculum_axes = "".join(
        '<article class="component"><strong>'
        + html.escape(axis.replace("_", " "))
        + "</strong><span>"
        + html.escape(
            " | ".join(
                str(value).replace("_", " ")
                for value in curriculum["axes"][axis]
            )
        )
        + "</span></article>"
        for axis in CURRICULUM_AXIS_ORDER
    )
    execution_program_ids = set(curriculum["execution_program_ids"])
    execution_programs = [
        program
        for program in curriculum["schedule"]
        if program["program_id"] in execution_program_ids
    ]
    curriculum_programs = "".join(
        '<article class="law"><div><strong>regime '
        + str(program["program_id"])
        + "</strong><span>"
        + html.escape(
            str(program["primary_root"]).replace("_", " ")
            + " + "
            + str(program["secondary_root"]).replace("_", " ")
        )
        + "</span></div><small>"
        + html.escape(
            " | ".join(
                str(program[key]).replace("_", " ")
                for key in (
                    "scale",
                    "resources",
                    "signal",
                    "relations",
                    "time",
                    "topology",
                    "phase_regime",
                    "energy_regime",
                    "boundary",
                )
            )
        )
        + "</small></article>"
        for program in execution_programs[:16]
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
                "architecture": model["architecture"],
                "model_hash": model_hash,
                "report_hash": report["report_hash"],
                "workflow_hash": workflow["workflow_hash"],
                "transfer_report_hash": transfer["report_hash"],
                "transfer_policy_hash": transfer["transfer_policy_hash"],
                "validation_truth_hash": transfer["validation_truth_hash"],
                "projection_lattice_digest": calibration["probe_response_hashes"][
                    "policy_neutral_projection_lattice"
                ],
                "risk_contract": calibration["risk_contract"],
                "formal_registry_hash": formal["registry_hash"],
                "formal_report_hash": formal["report_hash"],
                "formal_truth_hash": formal["truth_hash"],
                "artifact_binding_marker": "render_causal_world_artifact",
            },
            sort_keys=True,
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atom Causal World</title>
<style>
:root{{--bg:#060912;--panel:#0c1422;--line:#243752;--ink:#f4f7fb;--muted:#9bb0c9;--cyan:#58d5ff;--violet:#ba92ff;--green:#61e6a8;--amber:#ffc56b}}
*{{box-sizing:border-box}}html,body{{margin:0;min-height:100%;background:radial-gradient(circle at 12% 4%,#172d55,var(--bg) 36%);color:var(--ink);font:15px/1.5 Inter,Segoe UI,sans-serif}}
.shell{{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(380px,1fr);gap:20px;max-width:1640px;margin:auto;padding:22px}}main,aside{{min-width:0}}aside{{display:flex;flex-direction:column;gap:18px}}
.hero,.panel,aside section{{background:linear-gradient(145deg,#111e32f5,#090f1cf5);border:1px solid var(--line);border-radius:20px;box-shadow:0 22px 60px #0007;padding:24px}}.panel{{margin-top:20px}}
.eyebrow{{color:var(--cyan);font-size:.74rem;font-weight:900;letter-spacing:.17em;text-transform:uppercase}}h1{{font-size:clamp(2.5rem,5.4vw,5.4rem);line-height:.93;letter-spacing:-.055em;margin:.3em 0}}h2{{margin:0 0 14px}}p{{color:var(--muted)}}
.hash{{display:block;padding:11px;border:1px solid var(--line);background:#050a12;border-radius:10px;overflow-wrap:anywhere}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(115px,1fr));gap:9px;margin-top:18px}}.metric{{border:1px solid var(--line);border-radius:12px;padding:12px;background:#09111e}}.metric span{{display:block;color:var(--muted);font-size:.65rem;text-transform:uppercase}}.metric strong{{display:block;color:var(--amber);font-size:1.12rem;margin-top:4px;overflow-wrap:anywhere}}
.components{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}}.component{{padding:13px;border:1px solid var(--line);border-radius:12px}}.component strong,.component span{{display:block}}.component strong{{color:var(--violet);text-transform:capitalize}}.component span{{color:var(--muted);font-size:.72rem;margin-top:5px;overflow-wrap:anywhere}}
.domains{{display:flex;flex-wrap:wrap;gap:7px}}.domain{{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted)}}.domain strong{{color:var(--cyan)}}
.law,.turn,.gate{{padding:11px 0;border-bottom:1px solid var(--line)}}.law div{{display:flex;justify-content:space-between;gap:12px}}.law strong{{color:var(--green)}}.law span,.law small,.turn small{{color:var(--muted)}}.law small,.turn strong,.turn small{{display:block}}.turn p{{margin:0 0 5px;font-size:.84rem}}.turn strong{{color:var(--amber)}}.turn small{{margin-top:4px}}.gate{{display:grid;grid-template-columns:1fr auto;gap:12px}}.gate strong{{color:var(--green);font-size:.73rem}}.runtime{{font-size:.72rem;overflow-wrap:anywhere}}
@media(max-width:980px){{.shell{{grid-template-columns:1fr;padding:12px}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:560px){{.hero,.panel,aside section{{padding:16px}}.metrics,.components{{grid-template-columns:1fr}}h1{{font-size:2.5rem}}}}
</style></head><body><div class="shell"><main>
<section class="hero"><div class="eyebrow">pure causal graph &middot; procedural universe</div><h1>The graph learns laws, not strings.</h1><p>A persistent executable causal hypergraph learns from matched interventions across physical, chemical, biological, ecological, agent, social, symbolic, and language worlds. A typed formal substrate adds independently checked logic, algebra, geometry, calculus, chemistry, biology, and information theory. Phase dynamics operate only on retrieved working subgraphs; unsupported projections remain unknown.</p><code class="hash">graph {html.escape(model_hash)}</code><div class="metrics">{metrics}</div></section>
<section class="panel"><h2>Seven-component causal core</h2><div class="components">{components}</div></section>
<section class="panel"><h2>World domains</h2><div class="domains">{domain_rows}</div></section>
<section class="panel"><h2>Compositional world curriculum</h2><p>{curriculum["procedural_program_space"]:,} regimes are generated on demand from seven root mechanics and nine independent environmental axes.</p><div class="components">{curriculum_axes}</div></section>
<section class="panel"><h2>Typed formal domains</h2><p>Deduction, oracle evaluation, and epistemic projection remain separate. Exact results are proven, false candidates are contradicted, and unsupported operations remain unknown.</p><div class="components">{formal_domain_rows}</div></section>
</main><aside><section><h2>Measured causal artifact</h2>{turns}</section><section><h2>Cross-domain formal programs</h2>{formal_programs}</section><section><h2>Context-factor risk governor</h2><p>Singleton conditions and pairwise world motifs are composed from one direction-neutral factor trace. Prior, motif strength, and direction-specific acceptance bands are selected on validation worlds disjoint from training and final evaluation. A policy is eligible only inside its declared 95% selective-error bounds, and the evaluator cannot alter it. Wilson risk, condition log-likelihoods, and consensus projection use 80-digit decimal arithmetic with a twelve-decimal half-even boundary, so both the complete 5,000-policy lattice and the rendered diagnostic trace have one operating-system-independent identity.</p><div class="metrics">{calibration_view}</div></section><section><h2>Held-out causal transfer</h2><p>Exact-match retrieval is compared with risk-limited contextual transfer on balanced increasing/decreasing, cross-feature relations from regimes excluded from training and validation. Evaluator directions remain sealed from inference, and unsupported projections remain unknown.</p>{transfer_samples}</section><section><h2>Exercised world regimes</h2>{curriculum_programs}</section><section><h2>Persistent conditional laws</h2>{laws}</section><section><h2>Runtime gates</h2>{gates}</section><section><h2>Bound artifact</h2><p class="runtime">{binding}</p><p class="runtime">{ATOM_CAUSAL_WORLD_SIDE_VIEW_RUNTIME} &middot; render_causal_world_artifact</p></section></aside>
</div></body></html>"""
