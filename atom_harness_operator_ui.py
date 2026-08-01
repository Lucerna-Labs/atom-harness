"""Two-pane evidence and permissioned-hands operator surface."""

from __future__ import annotations

import json

from atom_harness_knowledge import (
    ATOM_HARNESS_RAG_RUNTIME,
    ATOM_HARNESS_WIKI_RUNTIME,
)
from atom_tool_fabric import ATOM_PERMISSIONED_HANDS_RUNTIME
from atom_tool_side_view import (
    ATOM_TOOL_ARTIFACT_BINDING,
    ATOM_TOOL_SIDE_VIEW_RUNTIME,
)
from atom_multidisciplinary_knowledge import (
    ATOM_MULTIDISCIPLINARY_RAG_RUNTIME,
    ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME,
)


ATOM_HARNESS_OPERATOR_UI_RUNTIME = "atom-language-harness-operator-ui-v6"
ATOM_HARNESS_OPERATOR_ARTIFACT_BINDING = "render_operator_surface"


def _javascript_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False).replace("<", "\\u003c")


def render_operator_surface(*, access_token: str, nonce: str) -> str:
    """Render the live controls beside the selected real artifact side view."""

    token = _javascript_string(access_token)
    ui_runtime = _javascript_string(ATOM_HARNESS_OPERATOR_UI_RUNTIME)
    wiki_runtime = _javascript_string(ATOM_HARNESS_WIKI_RUNTIME)
    rag_runtime = _javascript_string(ATOM_HARNESS_RAG_RUNTIME)
    multidisciplinary_wiki_runtime = _javascript_string(
        ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME
    )
    multidisciplinary_rag_runtime = _javascript_string(
        ATOM_MULTIDISCIPLINARY_RAG_RUNTIME
    )
    hands_runtime = _javascript_string(ATOM_PERMISSIONED_HANDS_RUNTIME)
    tool_side_view_runtime = _javascript_string(ATOM_TOOL_SIDE_VIEW_RUNTIME)
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
      grid-template-rows: auto auto 1fr auto;
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
    #hands-history {{ overflow: auto; padding: 14px; }}
    .mode-tabs {{ display: flex; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
    .mode-tabs button.active {{ color: #0d120a; background: var(--green); border-color: var(--green); font-weight: 700; }}
    [hidden] {{ display: none !important; }}
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
    .tool-request {{ cursor: default; }}
    .tool-request.selected {{ border-color: var(--blue); }}
    .tool-request .request-select {{ width: 100%; padding: 0; color: inherit; text-align: left; background: transparent; border: 0; }}
    .tool-actions {{ margin-top: 10px; padding-top: 9px; border-top: 1px solid var(--line); }}
    .tool-action {{ margin: 7px 0; padding: 9px; border: 1px solid var(--line); border-radius: 7px; background: #10120f; }}
    .tool-action-head {{ display: flex; align-items: center; gap: 8px; }}
    .risk {{ padding: 2px 6px; border: 1px solid var(--line); border-radius: 999px; color: var(--amber); font-size: 9px; text-transform: uppercase; }}
    .exact {{ max-height: 180px; overflow: auto; margin: 7px 0 0; padding: 8px; color: #d5d8ce; background: #090a08; border-radius: 6px; font: 10px/1.4 ui-monospace,monospace; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .warning {{ margin-top: 8px; padding: 8px; color: var(--amber); background: #292313; border: 1px solid #5d4c22; border-radius: 7px; font-size: 11px; }}
    .permission {{ margin-top: 10px; padding: 10px; border: 1px solid #5d4c22; border-radius: 8px; background: #18150d; }}
    .permission strong {{ display: block; margin-bottom: 7px; color: var(--amber); }}
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
    .status-planning, .status-approved, .status-executing, .status-awaiting-permission {{ color: var(--amber); }}
    .status-denied, .status-expired, .status-interrupted, .status-no-actions {{ color: var(--muted); }}
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
      data-rag="{ATOM_HARNESS_RAG_RUNTIME}"
      data-multidisciplinary-wiki="{ATOM_MULTIDISCIPLINARY_WIKI_RUNTIME}"
      data-multidisciplinary-rag="{ATOM_MULTIDISCIPLINARY_RAG_RUNTIME}"
      data-hands="{ATOM_PERMISSIONED_HANDS_RUNTIME}"
      data-tool-side-view="{ATOM_TOOL_SIDE_VIEW_RUNTIME}"
      data-tool-binding="{ATOM_TOOL_ARTIFACT_BINDING}">
  <header>
    <h1>ATOM HARNESS EXPERIMENT</h1>
    <span id="runtime-state" class="pill">preloading</span>
    <span id="model-state" class="pill">model waiting</span>
    <span class="pill">15 knowledge disciplines</span>
    <span class="grow"></span>
    <span id="notice"></span>
  </header>
  <main class="layout">
    <section class="conversation" aria-label="Conversation and controls">
      <div class="metrics">
        <div class="metric"><span>Queues</span><strong id="queue">0 / 8</strong></div>
        <div class="metric"><span>Model loads</span><strong id="loads">0</strong></div>
        <div class="metric"><span>Restarts</span><strong id="restarts">0</strong></div>
        <div class="metric"><span>Last time</span><strong id="timing">none</strong></div>
      </div>
      <div class="mode-tabs" role="tablist" aria-label="Experiment mode">
        <button id="mode-evidence" class="active" type="button" role="tab">Evidence</button>
        <button id="mode-hands" type="button" role="tab">Permissioned hands</button>
      </div>
      <div id="history">
        <p class="empty">The resident model and Atom graph are loading. Your committed answers will appear here, with the real evidence artifact beside them.</p>
      </div>
      <div id="hands-history" hidden>
        <p class="empty">Tool proposals will appear here. Nothing can execute until you approve the exact displayed manifest.</p>
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
      <form id="tool-form" hidden>
        <label for="tool-task">Give Atom a task that may require hands</label>
        <textarea id="tool-task" maxlength="4096" required
          placeholder="Create code, build a simulation, write a document, or manage a project"></textarea>
        <div class="actions">
          <button class="primary" id="plan-tools" type="submit">Plan exact actions</button>
          <button id="continue-tools" type="button">Continue from selected result</button>
          <button id="cancel-tools" type="button">Cancel selected task</button>
        </div>
        <p class="warning">Planning does not grant permission. Every exact action must be approved here before any tool runs.</p>
      </form>
    </section>
    <section class="artifact" aria-label="Committed artifact side view">
      <div class="artifact-bar">
        <strong>REAL ARTIFACT SIDE VIEW</strong>
        <span id="artifact-label">Select a completed answer or tool run</span>
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
    const multidisciplinaryWikiRuntime = {multidisciplinary_wiki_runtime};
    const multidisciplinaryRagRuntime = {multidisciplinary_rag_runtime};
    const handsRuntime = {hands_runtime};
    const toolSideViewRuntime = {tool_side_view_runtime};
    const headers = {{
      "X-Atom-Operator-Token": accessToken,
      "Content-Type": "application/json"
    }};
    const state = {{
      snapshot: null,
      mode: "evidence",
      selected: null,
      selectedTool: null,
      loadingArtifactId: null,
      loadedArtifactId: null,
      failedArtifactId: null
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

    async function renderArtifact(path, artifactKey) {{
      const retryDelays = [0, 250, 750];
      let failure = new Error("artifact-not-available");
      for (const delay of retryDelays) {{
        if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
        try {{
          const response = await api(path, {{
            headers: {{"Accept": "text/html"}}
          }});
          const artifactHtml = await response.text();
          const artifactFrame = byId("artifact-frame");
          artifactFrame.removeAttribute("src");
          artifactFrame.srcdoc = artifactHtml;
          state.failedArtifactId = null;
          state.loadedArtifactId = artifactKey;
          return;
        }} catch (error) {{
          failure = error;
        }}
      }}
      throw failure;
    }}

    function renderArtifactFailure(artifactKey, kind) {{
      const artifactFrame = byId("artifact-frame");
      artifactFrame.removeAttribute("src");
      artifactFrame.srcdoc =
        "<style>body{{background:#0b0d0a;color:#ddd;font:16px system-ui;" +
        "display:grid;place-items:center;height:90vh;margin:0}}" +
        "main{{max-width:34rem;padding:2rem}}h2{{color:#ffca73}}</style>" +
        "<main><h2>Artifact unavailable</h2><p>The committed " + kind +
        " artifact could not be verified. Select the completed item to retry.</p></main>";
      byId("artifact-label").textContent =
        "Artifact unavailable, select the completed item to retry";
      state.failedArtifactId = artifactKey;
      state.loadedArtifactId = artifactKey;
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
        await selectRequest(record.request_id, true);
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

    function setMode(mode) {{
      state.mode = mode;
      const hands = mode === "hands";
      byId("mode-evidence").classList.toggle("active", !hands);
      byId("mode-hands").classList.toggle("active", hands);
      byId("history").hidden = hands;
      byId("hands-history").hidden = !hands;
      byId("ask-form").hidden = hands;
      byId("tool-form").hidden = !hands;
    }}

    function appendText(parent, className, text) {{
      const item = document.createElement("div");
      item.className = className;
      item.textContent = text;
      parent.appendChild(item);
      return item;
    }}

    function toolRequestCard(record) {{
      const card = document.createElement("article");
      card.className = "request tool-request" +
        (state.selectedTool === record.proposal_id ? " selected" : "");
      const select = document.createElement("button");
      select.type = "button";
      select.className = "request-select";
      select.addEventListener("click", async () => {{
        await selectTool(record.proposal_id, true);
        if (state.snapshot) render(state.snapshot);
      }});
      appendText(select, "question", record.task);
      if (record.summary) appendText(select, "answer", record.summary);
      const meta = document.createElement("div");
      meta.className = "meta";
      const status = document.createElement("span");
      status.className = "status-" + record.status;
      status.textContent = record.status;
      meta.appendChild(status);
      const count = document.createElement("span");
      count.textContent = record.action_count + " actions";
      meta.appendChild(count);
      if (record.maximum_risk) {{
        const risk = document.createElement("span");
        risk.textContent = "max risk " + record.maximum_risk;
        meta.appendChild(risk);
      }}
      select.appendChild(meta);
      card.appendChild(select);

      if (record.injection_signals?.length) {{
        appendText(
          card,
          "warning",
          "Outside influence signals: " + record.injection_signals.join(", ")
        );
      }}

      if (record.planner_normalizations?.length) {{
        const normalized = record.planner_normalizations.map((item) => {{
          const fields = item.fields?.length ? " (" + item.fields.join(", ") + ")" : "";
          return item.action_id + ": " + item.kind + fields;
        }});
        appendText(
          card,
          "warning",
          "Atom normalized the untrusted model proposal before permission: " +
            normalized.join("; ") + ". The exact manifest below is authoritative."
        );
      }}

      if (record.error) {{
        const failure = record.error.operator_message ||
          "Planning failed closed. No tool action ran.";
        const kinds = (record.error.provider_failures || [])
          .map((item) => item.failure_kind)
          .filter(Boolean);
        appendText(
          card,
          "warning",
          failure + (kinds.length ? " Provider boundary: " + kinds.join(", ") + "." : "")
        );
      }}

      if (record.actions?.length) {{
        const actions = document.createElement("div");
        actions.className = "tool-actions";
        record.actions.forEach((action) => {{
          const item = document.createElement("section");
          item.className = "tool-action";
          const head = document.createElement("div");
          head.className = "tool-action-head";
          const name = document.createElement("strong");
          name.textContent = action.action_id + "  " + action.capability;
          head.appendChild(name);
          const risk = document.createElement("span");
          risk.className = "risk";
          risk.textContent = action.risk;
          head.appendChild(risk);
          item.appendChild(head);
          appendText(item, "answer", action.rationale);
          appendText(item, "exact", JSON.stringify({{
            arguments: action.arguments,
            effects: action.effects,
            action_hash: action.action_hash
          }}, null, 2));
          actions.appendChild(item);
        }});
        card.appendChild(actions);
      }}

      if (record.status === "awaiting-permission") {{
        const permission = document.createElement("div");
        permission.className = "permission";
        const label = document.createElement("strong");
        label.textContent = "Permission required for this exact manifest";
        permission.appendChild(label);
        appendText(permission, "exact", record.manifest_hash);
        const controls = document.createElement("div");
        controls.className = "actions";
        const approve = document.createElement("button");
        approve.type = "button";
        approve.className = "primary";
        approve.textContent = "Approve exact actions";
        approve.addEventListener("click", async () => {{
          try {{
            await api("/api/tools/approve", {{
              method: "POST",
              body: JSON.stringify({{
                proposal_id: record.proposal_id,
                manifest_hash: record.manifest_hash,
                decision_nonce: record.decision_nonce
              }})
            }});
            setNotice("One-time permission granted to the exact manifest.");
            await refresh();
          }} catch (error) {{ setNotice(error.message); }}
        }});
        const deny = document.createElement("button");
        deny.type = "button";
        deny.className = "danger";
        deny.textContent = "Deny";
        deny.addEventListener("click", async () => {{
          try {{
            await api("/api/tools/deny", {{
              method: "POST",
              body: JSON.stringify({{
                proposal_id: record.proposal_id,
                manifest_hash: record.manifest_hash,
                decision_nonce: record.decision_nonce
              }})
            }});
            setNotice("Tool proposal denied. No action ran.");
            await refresh();
          }} catch (error) {{ setNotice(error.message); }}
        }});
        controls.append(approve, deny);
        permission.appendChild(controls);
        card.appendChild(permission);
      }}
      return card;
    }}

    function render(snapshot) {{
      state.snapshot = snapshot;
      const hands = snapshot.hands || {{enabled: false, proposals: []}};
      const runtimeState = byId("runtime-state");
      runtimeState.textContent = snapshot.state;
      runtimeState.className = "pill " + snapshot.state;
      const currentLane = lane(snapshot);
      byId("model-state").textContent = currentLane?.alive
        ? "model resident"
        : (snapshot.preload ? "model available" : "model waiting");
      byId("model-state").className = "pill " + (currentLane?.alive ? "ready" : "");
      byId("queue").textContent = snapshot.queue_depth + " / " + snapshot.max_queue_depth +
        " | hands " + (hands.queue_depth ?? 0) + " / " + (hands.max_queue_depth ?? 0);
      byId("loads").textContent = currentLane?.model_load_count ?? 0;
      byId("restarts").textContent = currentLane?.restart_count ?? 0;
      const completed = snapshot.requests.filter((item) => item.status === "completed");
      const latest = completed[completed.length - 1];
      byId("timing").textContent = latest?.artifact?.total_ms
        ? latest.artifact.total_ms + " ms"
        : "none";
      byId("ask").disabled = !snapshot.accepting;
      byId("plan-tools").disabled = !snapshot.accepting || hands.accepting === false;
      byId("cancel").disabled = !snapshot.active_request_id;
      const selected = snapshot.requests.find((item) => item.request_id === state.selected);
      byId("retry").disabled = !selected || !["failed", "cancelled", "interrupted"].includes(selected.status);
      byId("restart").disabled = !snapshot.accepting || !!snapshot.active_request_id || snapshot.queue_depth > 0;
      const selectedTool = (hands.proposals || []).find(
        (item) => item.proposal_id === state.selectedTool
      );
      byId("continue-tools").disabled = !selectedTool || selectedTool.status !== "completed";
      byId("cancel-tools").disabled = !selectedTool || [
        "completed", "failed", "cancelled", "denied", "interrupted", "expired", "no-actions"
      ].includes(selectedTool.status);
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
        state.loadedArtifactId !== "answer:" + selected.request_id &&
        state.loadingArtifactId !== "answer:" + selected.request_id
      ) {{
        selectRequest(selected.request_id);
      }}

      const handsHistory = byId("hands-history");
      handsHistory.replaceChildren();
      if (!(hands.proposals || []).length) {{
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "The capability registry is ready. Submit a task to produce a permission request. No tool can run before approval.";
        handsHistory.appendChild(empty);
      }} else {{
        [...hands.proposals].reverse().forEach(
          (record) => handsHistory.appendChild(toolRequestCard(record))
        );
      }}
      const latestTool = [...(hands.proposals || [])].reverse().find(
        (item) => item.artifact
      );
      if (state.mode === "hands" && !state.selectedTool && latestTool) {{
        selectTool(latestTool.proposal_id);
      }} else if (
        state.mode === "hands" && selectedTool?.artifact &&
        state.loadedArtifactId !== "tool:" + selectedTool.proposal_id &&
        state.loadingArtifactId !== "tool:" + selectedTool.proposal_id
      ) {{
        selectTool(selectedTool.proposal_id);
      }}
    }}

    async function selectRequest(requestId, forceRetry = false) {{
      state.selected = requestId;
      const artifactKey = "answer:" + requestId;
      const record = state.snapshot?.requests.find((item) => item.request_id === requestId);
      if (!record || record.status !== "completed") return;
      if (forceRetry && state.failedArtifactId === artifactKey) {{
        state.failedArtifactId = null;
        state.loadedArtifactId = null;
      }}
      if (
        state.loadedArtifactId === artifactKey ||
        state.loadingArtifactId === artifactKey
      ) return;
      state.loadingArtifactId = artifactKey;
      try {{
        await renderArtifact(
          `/api/artifacts/${{encodeURIComponent(requestId)}}/side-view`,
          artifactKey
        );
        byId("artifact-label").textContent =
          record.artifact.citations.length + " citations, transaction " +
          record.artifact.transaction_id.slice(0, 12);
      }} catch (error) {{
        renderArtifactFailure(artifactKey, "evidence");
        setNotice("Artifact verification failed. Select the completed answer to retry.");
      }} finally {{
        state.loadingArtifactId = null;
      }}
    }}

    async function selectTool(proposalId, forceRetry = false) {{
      state.selectedTool = proposalId;
      const artifactKey = "tool:" + proposalId;
      const record = state.snapshot?.hands?.proposals?.find(
        (item) => item.proposal_id === proposalId
      );
      if (!record?.artifact) return;
      if (forceRetry && state.failedArtifactId === artifactKey) {{
        state.failedArtifactId = null;
        state.loadedArtifactId = null;
      }}
      if (
        state.loadedArtifactId === artifactKey ||
        state.loadingArtifactId === artifactKey
      ) return;
      state.loadingArtifactId = artifactKey;
      try {{
        await renderArtifact(
          `/api/tool-artifacts/${{encodeURIComponent(proposalId)}}/side-view`,
          artifactKey
        );
        byId("artifact-label").textContent =
          record.action_count + " permissioned actions, transaction " +
          record.artifact.transaction_id.slice(0, 12);
      }} catch (error) {{
        renderArtifactFailure(artifactKey, "tool");
        setNotice("Artifact verification failed. Select the completed tool run to retry.");
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

    byId("mode-evidence").addEventListener("click", () => {{
      setMode("evidence");
      if (state.selected) selectRequest(state.selected);
    }});
    byId("mode-hands").addEventListener("click", () => {{
      setMode("hands");
      if (state.selectedTool) selectTool(state.selectedTool);
    }});

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
    byId("tool-form").addEventListener("submit", async (event) => {{
      event.preventDefault();
      const task = byId("tool-task").value.trim();
      if (!task) return;
      try {{
        const response = await api("/api/tools/propose", {{
          method: "POST",
          body: JSON.stringify({{task}})
        }});
        const record = await response.json();
        state.selectedTool = record.proposal_id;
        byId("tool-task").value = "";
        setNotice("Planning started. No tool has permission to run.");
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("continue-tools").addEventListener("click", async () => {{
      const task = byId("tool-task").value.trim();
      if (!task || !state.selectedTool) {{
        setNotice("Enter the next objective before continuing from a result.");
        return;
      }}
      try {{
        const response = await api("/api/tools/propose", {{
          method: "POST",
          body: JSON.stringify({{
            task,
            parent_proposal_id: state.selectedTool
          }})
        }});
        const record = await response.json();
        state.selectedTool = record.proposal_id;
        byId("tool-task").value = "";
        setNotice("Prior results were supplied as untrusted context. Review the new manifest before approval.");
        await refresh();
      }} catch (error) {{ setNotice(error.message); }}
    }});
    byId("cancel-tools").addEventListener("click", async () => {{
      if (!state.selectedTool) return;
      try {{
        await api("/api/tools/cancel", {{
          method: "POST",
          body: JSON.stringify({{proposal_id: state.selectedTool}})
        }});
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
    if (!uiRuntime || !wikiRuntime || !ragRuntime || !multidisciplinaryWikiRuntime || !multidisciplinaryRagRuntime || !handsRuntime || !toolSideViewRuntime) throw new Error("runtime markers absent");
    setMode("evidence");
    refresh();
    setInterval(refresh, 800);
  </script>
</body>
</html>
"""
