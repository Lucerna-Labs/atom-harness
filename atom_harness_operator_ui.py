"""Two-pane operator surface bound to committed Atom harness artifacts."""

from __future__ import annotations

import json

from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)


ATOM_HARNESS_OPERATOR_UI_RUNTIME = "atom-language-harness-operator-ui-v4"
ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING = "render_operator_surface"


def _javascript_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False).replace("<", "\\u003c")


def render_operator_surface(*, access_token: str, nonce: str) -> str:
    """Render the live controls beside the selected real artifact side view."""

    token = _javascript_string(access_token)
    ui_runtime = _javascript_string(ATOM_HARNESS_OPERATOR_UI_RUNTIME)
    wiki_runtime = _javascript_string(ATOM_HARNESS_WIKI_RUNTIME)
    rag_runtime = _javascript_string(ATOM_HARNESS_RAG_RUNTIME)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Atom Harness Operator</title>
  <style nonce="{nonce}">
    :root {{
      color-scheme: dark;
      --ink: #f5f4ee;
      --muted: #aaa99f;
      --panel: #151713;
      --panel-2: #1c201a;
      --line: #343b30;
      --green: #9fe870;
      --amber: #f1c75b;
      --red: #ff7d7d;
      --blue: #79b8ff;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; }}
    body {{
      background: #0d0f0c;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      overflow: hidden;
    }}
    header {{
      height: 64px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: #10120f;
    }}
    header h1 {{ margin: 0; font-size: 17px; letter-spacing: .04em; }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .pill.ready {{ color: var(--green); border-color: #456a35; }}
    .pill.failed {{ color: var(--red); border-color: #743f3f; }}
    .grow {{ flex: 1; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(360px, 42%) minmax(480px, 58%);
      height: calc(100% - 64px);
    }}
    .conversation {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }}
    .metric {{
      min-width: 0;
      padding: 10px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 9px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .metric strong {{
      display: block;
      overflow: hidden;
      margin-top: 5px;
      color: var(--green);
      font-size: 14px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    #history {{
      overflow: auto;
      padding: 14px;
    }}
    .empty {{
      margin: 24px;
      color: var(--muted);
      line-height: 1.6;
      text-align: center;
    }}
    .request {{
      width: 100%;
      margin: 0 0 10px;
      padding: 12px;
      color: var(--ink);
      text-align: left;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
      cursor: pointer;
    }}
    .request:hover, .request.selected {{ border-color: var(--green); }}
    .request .question {{ font-size: 13px; line-height: 1.45; }}
    .request .answer {{
      margin-top: 9px;
      color: #d2d5cb;
      font-size: 12px;
      line-height: 1.45;
    }}
    .request .meta {{
      display: flex;
      gap: 8px;
      margin-top: 9px;
      color: var(--muted);
      font-size: 10px;
    }}
    .status-completed {{ color: var(--green); }}
    .status-failed, .status-cancelled {{ color: var(--red); }}
    .status-running, .status-queued {{ color: var(--amber); }}
    form {{
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #10120f;
    }}
    textarea {{
      width: 100%;
      min-height: 78px;
      resize: vertical;
      padding: 12px;
      color: var(--ink);
      background: #0d0f0c;
      border: 1px solid var(--line);
      border-radius: 9px;
      font: inherit;
      line-height: 1.45;
    }}
    textarea:focus {{ outline: 2px solid #486f37; border-color: var(--green); }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    button {{
      padding: 8px 12px;
      color: var(--ink);
      background: #242a20;
      border: 1px solid var(--line);
      border-radius: 7px;
      cursor: pointer;
    }}
    button.primary {{ color: #0d120a; background: var(--green); border-color: var(--green); font-weight: 700; }}
    button.danger {{ color: var(--red); }}
    button:disabled {{ cursor: not-allowed; opacity: .45; }}
    .artifact {{
      display: grid;
      grid-template-rows: 48px 1fr;
      min-width: 0;
      background: #0b0d0a;
    }}
    .artifact-bar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}
    #artifact-frame {{ width: 100%; height: 100%; border: 0; background: #111; }}
    #notice {{ color: var(--amber); font-size: 11px; }}
    @media (max-width: 920px) {{
      body {{ overflow: auto; }}
      .layout {{ grid-template-columns: 1fr; height: auto; }}
      .conversation {{ min-height: 720px; border-right: 0; }}
      .artifact {{ min-height: 760px; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body data-runtime="{ATOM_HARNESS_OPERATOR_UI_RUNTIME}"
      data-binding="{ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING}"
      data-wiki="{ATOM_HARNESS_WIKI_RUNTIME}"
      data-rag="{ATOM_HARNESS_RAG_RUNTIME}">
  <header>
    <h1>ATOM HARNESS OPERATOR</h1>
    <span id="runtime-state" class="pill">preloading</span>
    <span id="model-state" class="pill">model waiting</span>
    <span class="grow"></span>
    <span id="notice"></span>
  </header>
  <main class="layout">
    <section class="conversation" aria-label="Conversation and controls">
      <div class="metrics">
        <div class="metric"><span>Queue</span><strong id="queue">0 / 8</strong></div>
        <div class="metric"><span>Model loads</span><strong id="loads">0</strong></div>
        <div class="metric"><span>Restarts</span><strong id="restarts">0</strong></div>
        <div class="metric"><span>Last time</span><strong id="timing">none</strong></div>
      </div>
      <div id="history">
        <p class="empty">The resident model and Atom graph are loading. Your committed answers will appear here, with the real evidence artifact beside them.</p>
      </div>
      <form id="ask-form">
        <label for="question">Ask the Atom evidence harness</label>
        <textarea id="question" maxlength="4096" required
          placeholder="Ask a causal question grounded in Atom evidence"></textarea>
        <div class="actions">
          <button class="primary" id="ask" type="submit">Ask Atom</button>
          <button id="cancel" type="button">Cancel active</button>
          <button id="retry" type="button">Retry selected</button>
          <button id="restart" type="button">Restart model</button>
          <button class="danger" id="shutdown" type="button">Shut down</button>
        </div>
      </form>
    </section>
    <section class="artifact" aria-label="Committed artifact side view">
      <div class="artifact-bar">
        <strong>REAL ARTIFACT SIDE VIEW</strong>
        <span id="artifact-label">Select a completed answer</span>
      </div>
      <iframe id="artifact-frame" title="Atom evidence artifact"
        sandbox="" referrerpolicy="no-referrer"
        srcdoc="<style>body{{background:#0b0d0a;color:#aaa;font:16px system-ui;display:grid;place-items:center;height:90vh}}</style><p>The committed evidence artifact will render here.</p>"></iframe>
    </section>
  </main>
  <script nonce="{nonce}">
    "use strict";
    const accessToken = {token};
    const uiRuntime = {ui_runtime};
    const wikiRuntime = {wiki_runtime};
    const ragRuntime = {rag_runtime};
    const headers = {{
      "X-Atom-Operator-Token": accessToken,
      "Content-Type": "application/json"
    }};
    const state = {{
      snapshot: null,
      selected: null,
      loadingArtifactId: null,
      loadedArtifactId: null
    }};
    const byId = (id) => document.getElementById(id);

    async function api(path, options = {{}}) {{
      const response = await fetch(path, {{
        cache: "no-store",
        credentials: "same-origin",
        ...options,
        headers: {{...headers, ...(options.headers || {{}})}}
      }});
      if (!response.ok) {{
        let message = `Request failed (${{response.status}})`;
        try {{ message = (await response.json()).error || message; }} catch (_) {{}}
        throw new Error(message);
      }}
      return response;
    }}

    function lane(snapshot) {{
      const providers = snapshot?.session?.providers || [];
      return providers.find((item) => item.lane)?.lane || null;
    }}

    function setNotice(message) {{
      byId("notice").textContent = message;
      if (message) setTimeout(() => {{
        if (byId("notice").textContent === message) byId("notice").textContent = "";
      }}, 5000);
    }}

    function requestButton(record) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "request" + (state.selected === record.request_id ? " selected" : "");
      button.addEventListener("click", async () => {{
        await selectRequest(record.request_id);
        if (state.snapshot) render(state.snapshot);
      }});
      const question = document.createElement("div");
      question.className = "question";
      question.textContent = record.question;
      button.appendChild(question);
      if (record.artifact?.answer) {{
        const answer = document.createElement("div");
        answer.className = "answer";
        answer.textContent = record.artifact.answer;
        button.appendChild(answer);
      }}
      const meta = document.createElement("div");
      meta.className = "meta";
      const status = document.createElement("span");
      status.className = "status-" + record.status;
      status.textContent = record.status;
      meta.appendChild(status);
      const attempt = document.createElement("span");
      attempt.textContent = "attempt " + record.attempt;
      meta.appendChild(attempt);
      if (record.artifact?.citations) {{
        const citations = document.createElement("span");
        citations.textContent = record.artifact.citations.length + " citations";
        meta.appendChild(citations);
      }}
      button.appendChild(meta);
      return button;
    }}

    function render(snapshot) {{
      state.snapshot = snapshot;
      const runtimeState = byId("runtime-state");
      runtimeState.textContent = snapshot.state;
      runtimeState.className = "pill " + snapshot.state;
      const currentLane = lane(snapshot);
      byId("model-state").textContent = currentLane?.alive
        ? "model resident"
        : (snapshot.preload ? "model available" : "model waiting");
      byId("model-state").className = "pill " + (currentLane?.alive ? "ready" : "");
      byId("queue").textContent = snapshot.queue_depth + " / " + snapshot.max_queue_depth;
      byId("loads").textContent = currentLane?.model_load_count ?? 0;
      byId("restarts").textContent = currentLane?.restart_count ?? 0;
      const completed = snapshot.requests.filter((item) => item.status === "completed");
      const latest = completed[completed.length - 1];
      byId("timing").textContent = latest?.artifact?.total_ms
        ? latest.artifact.total_ms + " ms"
        : "none";
      byId("ask").disabled = !snapshot.accepting;
      byId("cancel").disabled = !snapshot.active_request_id;
      const selected = snapshot.requests.find((item) => item.request_id === state.selected);
      byId("retry").disabled = !selected || !["failed", "cancelled", "interrupted"].includes(selected.status);
      byId("restart").disabled = !snapshot.accepting || !!snapshot.active_request_id || snapshot.queue_depth > 0;
      const history = byId("history");
      history.replaceChildren();
      if (!snapshot.requests.length) {{
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "The runtime is ready. Ask a question to produce a transaction-bound answer and evidence view.";
        history.appendChild(empty);
      }} else {{
        [...snapshot.requests].reverse().forEach((record) => history.appendChild(requestButton(record)));
      }}
      if (!state.selected && latest) {{
        selectRequest(latest.request_id);
      }} else if (
        selected?.status === "completed" &&
        state.loadedArtifactId !== selected.request_id &&
        state.loadingArtifactId !== selected.request_id
      ) {{
        selectRequest(selected.request_id);
      }}
    }}

    async function selectRequest(requestId) {{
      state.selected = requestId;
      const record = state.snapshot?.requests.find((item) => item.request_id === requestId);
      if (!record || record.status !== "completed") return;
      if (
        state.loadedArtifactId === requestId ||
        state.loadingArtifactId === requestId
      ) return;
      state.loadingArtifactId = requestId;
      try {{
        state.loadedArtifactId = requestId;
        const artifactFrame = byId("artifact-frame");
        artifactFrame.removeAttribute("srcdoc");
        artifactFrame.src =
          `/api/artifacts/${{encodeURIComponent(requestId)}}/side-view`;
        byId("artifact-label").textContent =
          record.artifact.citations.length + " citations, transaction " +
          record.artifact.transaction_id.slice(0, 12);
      }} catch (error) {{
        setNotice(error.message);
      }} finally {{
        state.loadingArtifactId = null;
      }}
    }}

    async function refresh() {{
      try {{
        const response = await api("/api/status", {{headers: {{"Accept": "application/json"}}}});
        render(await response.json());
      }} catch (error) {{
        setNotice(error.message);
      }}
    }}

    byId("ask-form").addEventListener("submit", async (event) => {{
      event.preventDefault();
      const question = byId("question").value.trim();
      if (!question) return;
      try {{
        const response = await api("/api/ask", {{
          method: "POST",
          body: JSON.stringify({{question}})
        }});
        const record = await response.json();
        state.selected = record.request_id;
        byId("question").value = "";
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("cancel").addEventListener("click", async () => {{
      const requestId = state.snapshot?.active_request_id;
      if (!requestId) return;
      try {{
        await api("/api/cancel", {{method: "POST", body: JSON.stringify({{request_id: requestId}})}});
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("retry").addEventListener("click", async () => {{
      if (!state.selected) return;
      try {{
        const response = await api("/api/retry", {{
          method: "POST",
          body: JSON.stringify({{request_id: state.selected}})
        }});
        const record = await response.json();
        state.selected = record.request_id;
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("restart").addEventListener("click", async () => {{
      try {{
        await api("/api/restart", {{method: "POST", body: "{{}}"}});
        setNotice("Resident model restarted and warmed.");
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("shutdown").addEventListener("click", async () => {{
      try {{
        await api("/api/shutdown", {{method: "POST", body: JSON.stringify({{cancel_pending: false}})}});
        setNotice("Graceful shutdown started. Open work will finish first.");
      }} catch (error) {{ setNotice(error.message); }}
    }});
    if (!uiRuntime || !wikiRuntime || !ragRuntime) throw new Error("runtime markers absent");
    refresh();
    setInterval(refresh, 800);
  </script>
</body>
</html>
"""
