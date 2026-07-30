"""User-visible side view for generated English and training evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from atom_english_knowledge import (
    retrieve_english_knowledge,
    validate_english_knowledge_graph,
)

ATOM_ENGLISH_SIDE_VIEW_RUNTIME = "atom-english-side-view-v1"
SUPPORTED_ARTIFACT_RUNTIMES = {
    "atom-english-kaggle-runner-v1",
    "atom-generative-english-chat-v1",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _sample_cards(samples: list[Mapping[str, Any]]) -> str:
    cards: list[str] = []
    for index, sample in enumerate(samples):
        cards.append(
            """
            <article class="sample">
              <div class="sample-index">Dialogue {index}</div>
              <div class="bubble user">{prompt}</div>
              <div class="bubble assistant">{response}</div>
              <div class="tokens">{prompt_tokens} prompt tokens -
              {response_tokens} generated tokens</div>
            </article>
            """.format(
                index=index + 1,
                prompt=_escape(sample.get("prompt", "")),
                response=_escape(sample.get("response", "")),
                prompt_tokens=_escape(sample.get("prompt_tokens", 0)),
                response_tokens=_escape(sample.get("response_tokens", 0)),
            )
        )
    return "\n".join(cards)


def render_english_generation_artifact(
    run_summary: Mapping[str, Any],
    knowledge_graph: Mapping[str, Any],
) -> str:
    """Render the actual generated-language artifact beside run evidence."""

    validate_english_knowledge_graph(knowledge_graph)
    if run_summary.get("runtime") not in SUPPORTED_ARTIFACT_RUNTIMES:
        raise ValueError("English side view received a foreign run")
    samples_value = run_summary.get("samples")
    if not isinstance(samples_value, list):
        raise ValueError("English run samples are invalid")
    samples = [dict(sample) for sample in samples_value if isinstance(sample, Mapping)]
    contexts = retrieve_english_knowledge(
        knowledge_graph,
        "English causal graph tokenizer corpus teacher evaluation run",
        limit=8,
    )
    mode = str(run_summary.get("mode", "unknown"))
    parameter_count = run_summary.get("parameter_count")
    if parameter_count is None:
        parameter_count = run_summary.get("evaluation", {}).get(
            "model_parameter_count", "unknown"
        )
    state = run_summary.get("training_report", {}).get("training_state", {})
    evaluation = run_summary.get("evaluation", {})
    gate = evaluation.get("gate", {})
    status = (
        f"{state.get('consumed_tokens', 0):,} supervised tokens"
        if mode == "train"
        else (
            "external language gate passed"
            if gate.get("passed")
            else "external language gate not passed"
        )
    )
    context_rows = "\n".join(
        """
        <tr><td>{kind}</td><td>{label}</td><td>{score}</td></tr>
        """.format(
            kind=_escape(item["node"]["kind"]),
            label=_escape(item["node"]["label"]),
            score=_escape(item["score"]),
        )
        for item in contexts
    )
    embedded = _escape(
        json.dumps(
            {
                "run": run_summary,
                "knowledge": knowledge_graph,
                "rag_context": contexts,
            },
            sort_keys=True,
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Atom Generative English</title>
  <style>
    :root {{
      color-scheme: dark;
      --ink: #e8f0f2;
      --muted: #94aeb2;
      --line: #26464d;
      --panel: #102329;
      --accent: #62e5bd;
      --user: #173e4a;
      --assistant: #24352f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #071418;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }}
    main {{ padding: 32px; }}
    aside {{
      border-left: 1px solid var(--line);
      background: #0b1b20;
      padding: 28px 22px;
    }}
    .eyebrow {{
      color: var(--accent);
      font: 700 12px/1.4 ui-monospace, monospace;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 8px 0 4px; font-size: clamp(30px, 5vw, 58px); }}
    .subtitle {{ color: var(--muted); margin-bottom: 28px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .metric, .sample {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
    }}
    .metric {{ padding: 16px; }}
    .metric strong {{ display: block; font-size: 21px; margin-top: 4px; }}
    .sample {{ padding: 18px; margin: 0 0 16px; }}
    .sample-index, .tokens {{
      color: var(--muted);
      font: 12px/1.5 ui-monospace, monospace;
    }}
    .bubble {{
      max-width: 88%;
      padding: 12px 14px;
      border-radius: 14px;
      margin: 12px 0;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    .user {{ background: var(--user); margin-left: auto; }}
    .assistant {{ background: var(--assistant); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{
      padding: 9px 5px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--accent); }}
    .marker {{
      margin-top: 24px;
      color: var(--muted);
      font: 11px/1.5 ui-monospace, monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 900px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <div class="eyebrow">Causal graph language artifact</div>
      <h1>Generative English</h1>
      <div class="subtitle">Free-form model output, not a bounded command parser.</div>
      <section class="metrics">
        <div class="metric"><span>Mode</span><strong>{_escape(mode)}</strong></div>
        <div class="metric"><span>Parameters</span><strong>{_escape(parameter_count)}</strong></div>
        <div class="metric"><span>Evidence</span><strong>{_escape(status)}</strong></div>
      </section>
      <section>{_sample_cards(samples)}</section>
    </main>
    <aside>
      <div class="eyebrow">Runtime wiki and RAG</div>
      <h2>Bound knowledge</h2>
      <table>
        <thead><tr><th>Kind</th><th>Node</th><th>Score</th></tr></thead>
        <tbody>{context_rows}</tbody>
      </table>
      <div class="marker">
        {ATOM_ENGLISH_SIDE_VIEW_RUNTIME}<br>
        render_english_generation_artifact<br>
        {embedded}
      </div>
    </aside>
  </div>
</body>
</html>
"""


def write_english_generation_side_view(
    path: Path,
    run_summary: Mapping[str, Any],
    knowledge_graph: Mapping[str, Any],
) -> None:
    path.write_text(
        render_english_generation_artifact(run_summary, knowledge_graph),
        encoding="utf-8",
    )
