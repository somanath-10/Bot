import json
import os
import re
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


app = FastAPI()
START = time.time()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
MAX_ACTIONS_PER_TICK = 20

# In-memory stores. This is enough for the challenge harness; the process is not
# expected to restart during a test run.
contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
conversations: Dict[str, List[Dict[str, str]]] = {}
conversation_contexts: Dict[str, Dict[str, Any]] = {}
seen_messages: Dict[str, int] = {}
ended_conversations: set[str] = set()
sent_suppression_keys: set[str] = set()
conversation_seq = count(1)
last_tick_actions: List[Dict[str, Any]] = []


# Kept for compatibility with the original generator import path.
COMPOSER_SYSTEM_PROMPT = "Deterministic local Vera composer."
REPLY_SYSTEM_PROMPT = "Deterministic local Vera reply handler."


class LocalLLMProvider:
    """Compatibility shim for old imports; the bot does not need network LLMs."""

    def compose(self, prompt: str, system_prompt: str) -> str:
        return "{}"


llm = LocalLLMProvider()


async def _parse_json_model(request: Request, model_cls):
    try:
        data = await request.json()
    except Exception:
        raw = (await request.body()).decode("utf-8").strip()
        try:
            import json

            data = json.loads(raw) if raw else {}
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    try:
        return model_cls(**data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
@app.get("/ui")
async def root():
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-store">
  <title>Vera Bot Console</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f7;
      color: #182230;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee8;
      padding: 18px 22px;
    }
    h1 { margin: 0; font-size: 23px; font-weight: 750; letter-spacing: 0; }
    .sub { margin: 4px 0 0; color: #667085; font-size: 13px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      border: 1px solid #cfd7e4;
      border-radius: 999px;
      padding: 8px 11px;
      background: #f8fafc;
      font-size: 13px;
      color: #344054;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; }
    .dot.ok { background: #16a34a; }
    .dot.bad { background: #dc2626; }
    main {
      display: grid;
      grid-template-columns: 280px minmax(340px, 1fr) minmax(320px, 440px);
      gap: 14px;
      padding: 14px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section {
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      min-width: 0;
    }
    .panel { padding: 14px; }
    h2 { margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }
    .stack { display: grid; gap: 8px; }
    button, select, textarea {
      font: inherit;
      letter-spacing: 0;
    }
    button {
      border: 1px solid #c9d3df;
      background: #ffffff;
      color: #182230;
      padding: 10px 11px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
      text-align: left;
    }
    button.primary { background: #175cd3; border-color: #175cd3; color: #ffffff; }
    button.success { background: #067647; border-color: #067647; color: #ffffff; }
    button:hover { border-color: #175cd3; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .stat {
      border: 1px solid #e0e6ef;
      border-radius: 6px;
      padding: 9px;
      background: #f8fafc;
    }
    .label { color: #667085; font-size: 12px; }
    .value { margin-top: 3px; font-weight: 700; font-size: 18px; }
    .actions {
      display: grid;
      gap: 8px;
      max-height: 280px;
      overflow: auto;
      padding-right: 2px;
    }
    .action {
      border: 1px solid #d8dee8;
      border-radius: 7px;
      padding: 10px;
      background: #ffffff;
      cursor: pointer;
    }
    .action.active { border-color: #175cd3; box-shadow: inset 3px 0 0 #175cd3; }
    .action strong { display: block; font-size: 13px; margin-bottom: 5px; }
    .action span { display: block; color: #667085; font-size: 12px; line-height: 1.35; }
    .chat {
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      min-height: calc(100vh - 104px);
    }
    .chat-head {
      border-bottom: 1px solid #d8dee8;
      padding: 12px 14px;
    }
    .chat-head select {
      width: 100%;
      border: 1px solid #c9d3df;
      border-radius: 6px;
      padding: 9px;
      background: #ffffff;
      color: #182230;
    }
    .messages {
      padding: 14px;
      overflow: auto;
      background: #f7f9fc;
    }
    .msg {
      max-width: 88%;
      padding: 10px 11px;
      border-radius: 8px;
      margin: 0 0 10px;
      line-height: 1.42;
      font-size: 14px;
      word-break: break-word;
    }
    .msg.bot { background: #ffffff; border: 1px solid #dde4ee; }
    .msg.merchant { background: #dbeafe; margin-left: auto; }
    .msg small { display: block; margin-bottom: 4px; color: #667085; font-size: 11px; }
    .composer {
      border-top: 1px solid #d8dee8;
      padding: 12px;
      background: #ffffff;
    }
    .quick { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 8px; }
    .quick button { padding: 7px 9px; }
    textarea {
      width: 100%;
      min-height: 78px;
      resize: vertical;
      border: 1px solid #c9d3df;
      border-radius: 6px;
      padding: 10px;
      line-height: 1.45;
    }
    .send-row {
      display: flex;
      justify-content: flex-end;
      margin-top: 8px;
    }
    pre {
      min-height: calc(100vh - 210px);
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      padding: 13px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
    }
    code { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace; }
    .muted { color: #667085; font-size: 12px; line-height: 1.45; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 300px 1fr; }
      .json-panel { grid-column: 1 / -1; }
      pre { min-height: 280px; }
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; padding: 15px; }
      main { grid-template-columns: 1fr; padding: 10px; }
      .chat { min-height: 620px; }
      .msg { max-width: 96%; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Vera Bot Console</h1>
      <p class="sub">magicpin bot API and chat tester</p>
    </div>
    <div class="pill"><span id="statusDot" class="dot"></span><span id="live">Checking server</span></div>
  </header>
  <main>
    <section class="panel">
      <h2>Run</h2>
      <div class="stack">
        <button class="primary" onclick="loadDemo()">Load Demo Dataset</button>
        <button class="success" onclick="runTick()">Run Tick</button>
        <button onclick="runHealth()">Health</button>
        <button onclick="runMetadata()">Metadata</button>
      </div>
      <div class="stats">
        <div class="stat"><div class="label">Categories</div><div id="catCount" class="value">0</div></div>
        <div class="stat"><div class="label">Merchants</div><div id="merchantCount" class="value">0</div></div>
        <div class="stat"><div class="label">Customers</div><div id="customerCount" class="value">0</div></div>
        <div class="stat"><div class="label">Triggers</div><div id="triggerCount" class="value">0</div></div>
      </div>
      <h2>Tick Actions</h2>
      <div id="actions" class="actions"></div>
      <p class="muted">Submit the public URL for judging. This page is only a manual console.</p>
    </section>

    <section class="chat">
      <div class="chat-head">
        <select id="conversationSelect" onchange="selectConversation(this.value)">
          <option value="">No conversation selected</option>
        </select>
      </div>
      <div id="messages" class="messages"></div>
      <div class="composer">
        <div class="quick">
          <button onclick="quickReply('Yes, send it')">Yes</button>
          <button onclick="quickReply('What is the price?')">Price?</button>
          <button onclick="quickReply('I am busy, later')">Later</button>
          <button onclick="quickReply('Stop messaging')">Stop</button>
          <button onclick="quickReply('Thank you for contacting us. We will respond shortly.')">Auto</button>
        </div>
        <textarea id="replyText" placeholder="Type merchant reply here"></textarea>
        <div class="send-row">
          <button class="primary" onclick="sendReply()">Send Reply</button>
        </div>
      </div>
    </section>

    <section class="panel json-panel">
      <h2>JSON</h2>
      <pre id="output">Load demo data, run tick, then reply from the chat panel.</pre>
    </section>
  </main>
  <script>
    const out = document.getElementById("output");
    const live = document.getElementById("live");
    const statusDot = document.getElementById("statusDot");
    const actionsEl = document.getElementById("actions");
    const messagesEl = document.getElementById("messages");
    const conversationSelect = document.getElementById("conversationSelect");
    const replyText = document.getElementById("replyText");

    let actions = [];
    let conversations = {};
    let activeConversationId = "";

    function show(data) {
      out.textContent = JSON.stringify(data, null, 2);
    }

    async function call(path, options = {}) {
      const res = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { status: res.status, body: text }; }
      if (!res.ok) throw data;
      return data;
    }

    function setCounts(counts) {
      document.getElementById("catCount").textContent = counts.category || 0;
      document.getElementById("merchantCount").textContent = counts.merchant || 0;
      document.getElementById("customerCount").textContent = counts.customer || 0;
      document.getElementById("triggerCount").textContent = counts.trigger || 0;
    }

    function addTurn(conversationId, from, msg) {
      if (!conversations[conversationId]) conversations[conversationId] = [];
      conversations[conversationId].push({ from, msg });
      renderConversationList();
      if (activeConversationId === conversationId) renderMessages();
    }

    function renderConversationList() {
      const known = Object.keys(conversations);
      const options = ['<option value="">No conversation selected</option>'];
      for (const id of known) {
        options.push(`<option value="${id}" ${id === activeConversationId ? "selected" : ""}>${id}</option>`);
      }
      conversationSelect.innerHTML = options.join("");
    }

    function selectConversation(id) {
      activeConversationId = id;
      renderActions();
      renderConversationList();
      renderMessages();
    }

    function renderMessages() {
      const turns = conversations[activeConversationId] || [];
      if (!activeConversationId) {
        messagesEl.innerHTML = '<p class="muted">Run a tick and select an action to start chatting.</p>';
        return;
      }
      messagesEl.innerHTML = turns.map((turn) => {
        const cls = turn.from === "merchant" ? "merchant" : "bot";
        const from = turn.from === "merchant" ? "Merchant" : "Vera";
        return `<div class="msg ${cls}"><small>${from}</small>${escapeHtml(turn.msg || "")}</div>`;
      }).join("");
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    function renderActions() {
      if (!actions.length) {
        actionsEl.innerHTML = '<p class="muted">No tick actions yet.</p>';
        return;
      }
      actionsEl.innerHTML = actions.map((action) => {
        const active = action.conversation_id === activeConversationId ? " active" : "";
        const body = escapeHtml((action.body || "").slice(0, 145));
        return `<div class="action${active}" onclick="selectConversation('${action.conversation_id}')">
          <strong>${escapeHtml(action.merchant_id || action.conversation_id)}</strong>
          <span>${escapeHtml(action.trigger_id || "")}</span>
          <span>${body}</span>
        </div>`;
      }).join("");
    }

    function importState(data) {
      setCounts(data.contexts_loaded || {});
      actions = data.last_actions || actions;
      conversations = {};
      for (const item of data.conversations || []) {
        conversations[item.conversation_id] = item.history || [];
      }
      if (!activeConversationId && actions.length) {
        activeConversationId = actions[0].conversation_id;
      }
      renderActions();
      renderConversationList();
      renderMessages();
    }

    async function refreshState() {
      const data = await call("/ui/state");
      importState(data);
      return data;
    }

    async function loadDemo() {
      try {
        const data = await call("/ui/load-demo-data", { method: "POST" });
        show(data);
        await refreshState();
      } catch (err) { show(err); }
    }

    async function runTick() {
      try {
        const state = await refreshState();
        const triggerIds = (state.triggers || []).map((item) => item.id);
        const data = await call("/v1/tick", {
          method: "POST",
          body: JSON.stringify({
            now: new Date().toISOString(),
            available_triggers: triggerIds
          })
        });
        actions = data.actions || [];
        for (const action of actions) {
          conversations[action.conversation_id] = [{ from: "bot", msg: action.body }];
        }
        activeConversationId = actions[0]?.conversation_id || "";
        renderActions();
        renderConversationList();
        renderMessages();
        show(data);
      } catch (err) { show(err); }
    }

    function quickReply(text) {
      replyText.value = text;
      replyText.focus();
    }

    async function sendReply() {
      if (!activeConversationId) {
        show({ error: "Run tick and choose a conversation first." });
        return;
      }
      const message = replyText.value.trim();
      if (!message) {
        show({ error: "Type a merchant reply first." });
        return;
      }
      const action = actions.find((item) => item.conversation_id === activeConversationId) || {};
      addTurn(activeConversationId, "merchant", message);
      replyText.value = "";
      try {
        const data = await call("/v1/reply", {
          method: "POST",
          body: JSON.stringify({
            conversation_id: activeConversationId,
            merchant_id: action.merchant_id || null,
            customer_id: action.customer_id || null,
            from_role: "merchant",
            message,
            received_at: new Date().toISOString(),
            turn_number: (conversations[activeConversationId] || []).length
          })
        });
        addTurn(activeConversationId, "bot", data.body || data.action || "");
        show(data);
      } catch (err) { show(err); }
    }

    async function runHealth() {
      try {
        const data = await call("/v1/healthz");
        show(data);
        setCounts(data.contexts_loaded || {});
      } catch (err) { show(err); }
    }

    async function runMetadata() {
      try { show(await call("/v1/metadata")); }
      catch (err) { show(err); }
    }

    refreshState()
      .then(() => {
        live.textContent = "Server running";
        statusDot.classList.add("ok");
      })
      .catch(() => {
        live.textContent = "Server not reachable";
        statusDot.classList.add("bad");
      });
  </script>
</body>
</html>
        """,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


def _one_line(text: str) -> str:
    cleaned = str(text or "")
    replacements = {
        "\u00e2\u201a\u00b9": "Rs. ",
        "\u20b9": "Rs. ",
        "\u00e2\u20ac\u201d": "-",
        "\u2014": "-",
        "\u00e2\u20ac\u201c": "-",
        "\u2013": "-",
        "\u00e2\u2020\u2019": "->",
        "\u2192": "->",
        "\u00e2\u20ac\u00a6": "...",
        "\u2026": "...",
        "\u00e2\u02dc\u2026": "star",
        "\u2605": "star",
        "\u00f0\u0178\u00a6\u00b7": "",
        "\U0001f9b7": "",
    }
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r"Rs\.\s+", "Rs. ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _num(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any, signed: bool = True) -> str:
    try:
        pct = float(value) * 100
    except (TypeError, ValueError):
        return ""
    sign = "+" if signed and pct > 0 else ""
    return f"{sign}{pct:.0f}%"


def _ctr(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def _date_label(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.minute:
            hour = parsed.strftime("%I").lstrip("0") or "0"
            return f"{parsed.strftime('%d %b %Y')}, {hour}:{parsed.strftime('%M%p')}"
        return parsed.strftime("%d %b %Y")
    except Exception:
        return text.split("T", 1)[0]


def _identity(merchant: Dict[str, Any]) -> Dict[str, Any]:
    return merchant.get("identity", {}) if isinstance(merchant, dict) else {}


def _business_name(merchant: Dict[str, Any]) -> str:
    ident = _identity(merchant)
    return ident.get("name") or merchant.get("merchant_id") or "your business"


def _owner_name(merchant: Dict[str, Any]) -> str:
    ident = _identity(merchant)
    owner = ident.get("owner_first_name") or ident.get("name") or "there"
    return _one_line(owner)


def _salutation(category: Dict[str, Any], merchant: Dict[str, Any]) -> str:
    owner = _owner_name(merchant)
    if category.get("slug") == "dentists" and not owner.lower().startswith("dr."):
        return f"Dr. {owner}"
    return f"Hi {owner}"


def _customer_name(customer: Optional[Dict[str, Any]]) -> str:
    if not customer:
        return "there"
    name = customer.get("identity", {}).get("name") or "there"
    return _one_line(name)


def _active_offer(merchant: Dict[str, Any], category: Dict[str, Any]) -> str:
    for offer in merchant.get("offers", []) or []:
        if offer.get("status") == "active" and offer.get("title"):
            return _one_line(offer["title"])
    catalog = category.get("offer_catalog", []) or []
    if catalog:
        return _one_line(catalog[0].get("title", ""))
    return ""


def _peer_stat(category: Dict[str, Any], key: str) -> Any:
    return (category.get("peer_stats") or {}).get(key)


def _performance_line(merchant: Dict[str, Any], category: Dict[str, Any]) -> str:
    perf = merchant.get("performance", {}) or {}
    parts = []
    if perf.get("views") is not None:
        parts.append(f"{_num(perf.get('views'))} views")
    if perf.get("calls") is not None:
        parts.append(f"{_num(perf.get('calls'))} calls")
    if perf.get("directions") is not None:
        parts.append(f"{_num(perf.get('directions'))} directions")
    if perf.get("ctr") is not None:
        peer_ctr = _peer_stat(category, "avg_ctr")
        if peer_ctr is not None:
            parts.append(f"CTR {_ctr(perf.get('ctr'))} vs peer {_ctr(peer_ctr)}")
        else:
            parts.append(f"CTR {_ctr(perf.get('ctr'))}")
    return "30d profile: " + ", ".join(parts) if parts else "your current profile data"


def _customer_value_line(customer: Dict[str, Any]) -> str:
    rel = customer.get("relationship", {}) or {}
    bits = []
    if rel.get("last_visit"):
        bits.append(f"last visit {_date_label(rel.get('last_visit'))}")
    if rel.get("visits_total") is not None:
        bits.append(f"{_num(rel.get('visits_total'))} visits")
    if rel.get("lifetime_value") is not None:
        bits.append(f"Rs. {_num(rel.get('lifetime_value'))} history")
    return ", ".join(bits)


def _clean_topic(value: Any) -> str:
    return _one_line(str(value or "").replace("_", " "))


def _find_digest(category: Dict[str, Any], trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = trigger.get("payload", {}) or {}
    digest = category.get("digest", []) or []
    wanted_ids = [
        payload.get("top_item_id"),
        payload.get("digest_item_id"),
        payload.get("alert_id"),
    ]
    for wanted in wanted_ids:
        if not wanted:
            continue
        for item in digest:
            if item.get("id") == wanted:
                return item

    kind = trigger.get("kind", "")
    preferred_kinds = {
        "research_digest": {"research", "trend", "tech"},
        "regulation_change": {"compliance"},
        "cde_opportunity": {"cde"},
        "supply_alert": {"alert", "supply"},
        "category_seasonal": {"seasonal"},
        "festival_upcoming": {"seasonal", "trend"},
        "seasonal_perf_dip": {"seasonal"},
    }.get(kind, set())
    for item in digest:
        if item.get("kind") in preferred_kinds:
            return item
    return digest[0] if kind == "research_digest" and digest else None


def _digest_sentence(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return ""
    title = _one_line(item.get("title", ""))
    source = _one_line(item.get("source", ""))
    if source:
        return f"{title} ({source})"
    return title


def _rationale(kind: str, merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    source = trigger.get("source", "context")
    return (
        f"Uses {source} {kind} trigger, merchant performance, category voice, "
        f"and a single low-friction CTA for {_business_name(merchant)}."
    )


def _finish(
    body: str,
    cta: str,
    send_as: str,
    trigger: Dict[str, Any],
    rationale: str,
) -> Dict[str, Any]:
    body = _one_line(body)
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": _one_line(rationale),
    }


def _compose_research(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    item = _find_digest(category, trigger)
    agg = merchant.get("customer_aggregate", {}) or {}
    cohort = ""
    if agg.get("high_risk_adult_count"):
        cohort = f" You have {_num(agg['high_risk_adult_count'])} high-risk adult patients in context."
    elif agg.get("total_unique_ytd"):
        cohort = f" You have {_num(agg['total_unique_ytd'])} unique customers YTD."
    return (
        f"{_salutation(category, merchant)}, worth a look: {_digest_sentence(item)}."
        f"{cohort} Want me to pull the 2-min brief and draft a WhatsApp/post angle for "
        f"{_business_name(merchant)}? Reply YES or STOP."
    )


def _compose_compliance(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    item = _find_digest(category, trigger)
    deadline = _date_label(payload.get("deadline_iso"))
    summary = _one_line((item or {}).get("summary", ""))
    detail = f" Deadline: {deadline}." if deadline else ""
    return (
        f"{_salutation(category, merchant)}, compliance heads-up: {_digest_sentence(item)}."
        f" {summary[:190]}{detail} Want me to draft a 5-point SOP/checklist you can review? "
        f"Reply YES or STOP."
    )


def _compose_cde(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    item = _find_digest(category, trigger)
    credits = payload.get("credits") or (item or {}).get("credits")
    fee = _clean_topic(payload.get("fee") or (item or {}).get("actionable", ""))
    date = _date_label((item or {}).get("date"))
    detail = ", ".join([p for p in [f"{credits} credits" if credits else "", fee, date] if p])
    detail = f" ({detail})" if detail else ""
    return (
        f"{_salutation(category, merchant)}, CDE opportunity: {_digest_sentence(item)}{detail}. "
        f"{_performance_line(merchant, category)}. Want me to send the agenda summary and a patient-post idea after it? "
        f"Reply YES or STOP."
    )


def _compose_perf(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], spike: bool = False) -> str:
    payload = trigger.get("payload", {}) or {}
    perf = merchant.get("performance", {}) or {}
    delta_7d = perf.get("delta_7d", {}) or {}
    metric = payload.get("metric")
    if not metric:
        candidates = [(k[:-4], v) for k, v in delta_7d.items() if k.endswith("_pct")]
        if candidates:
            metric, _ = max(candidates, key=lambda item: item[1]) if spike else min(candidates, key=lambda item: item[1])
        else:
            metric = "calls"
    delta = payload.get("delta_pct")
    if delta is None:
        delta = delta_7d.get(f"{metric}_pct")
    window = payload.get("window", "7d")
    current = perf.get(metric)
    current_line = f"; current 30d {metric} {_num(current)}" if current is not None else ""
    offer = _active_offer(merchant, category)
    offer_line = f" Active offer: {offer}." if offer else ""
    if delta is None:
        movement = f"{metric} needs a performance check over {window}"
    elif spike:
        movement = f"{metric} are up {_pct(delta)} over {window}"
    elif float(delta) < 0:
        movement = f"{metric} are down {_pct(abs(float(delta)), signed=False)} over {window}"
    else:
        movement = f"{metric} dip follow-up is active; {metric} moved {_pct(delta)} over {window}, so conversion needs a quick audit"
    cta_verb = "lock this gain" if spike else "recover the drop"
    return (
        f"{_salutation(category, merchant)}, {movement}{current_line}. "
        f"{_performance_line(merchant, category)}.{offer_line} Want me to draft a 2-step plan to {cta_verb}? "
        f"Reply YES or STOP."
    )


def _compose_renewal(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    sub = merchant.get("subscription", {}) or {}
    days = payload.get("days_remaining") or sub.get("days_remaining")
    amount = payload.get("renewal_amount")
    amount_line = f" Rs. {_num(amount)} renewal" if amount else " renewal"
    day_line = f" in {_num(days)} days" if days is not None else " soon"
    return (
        f"{_salutation(category, merchant)}, Pro{amount_line} is due{day_line}. "
        f"Before you decide, {_performance_line(merchant, category)}. Want me to make a 3-point ROI summary "
        f"plus the next offer/post recommendation? Reply YES or STOP."
    )


def _compose_festival(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    festival = payload.get("festival", "festival season")
    date = _date_label(payload.get("date"))
    days = payload.get("days_until")
    beat = ""
    for item in category.get("seasonal_beats", []) or []:
        if "Oct" in item.get("month_range", "") or "Nov" in item.get("month_range", ""):
            beat = item.get("note", "")
            break
    when = f" on {date}" if date else ""
    count = f" ({_num(days)} days out)" if days is not None else ""
    offer = _active_offer(merchant, category)
    offer_line = f" Your current hook is {offer}." if offer else ""
    return (
        f"{_salutation(category, merchant)}, {festival}{when}{count} is a planning window for {_business_name(merchant)}. "
        f"{beat}{offer_line} Want me to draft the first festival post with one clean YES/STOP CTA? Reply YES or STOP."
    )


def _compose_winback(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    sub = merchant.get("subscription", {}) or {}
    days = payload.get("days_since_expiry") or sub.get("days_since_expiry") or payload.get("days_since_last_merchant_message")
    dip = payload.get("perf_dip_pct")
    lapsed = payload.get("lapsed_customers_added_since_expiry")
    facts = []
    if days is not None:
        facts.append(f"{_num(days)} days since last active touch")
    if dip is not None:
        facts.append(f"performance down {_pct(dip, signed=False)}")
    if lapsed is not None:
        facts.append(f"{_num(lapsed)} lapsed customers added")
    if not facts:
        facts.append(_performance_line(merchant, category))
    return (
        f"{_salutation(category, merchant)}, quick winback check: {', '.join(facts)}. "
        f"Want me to draft a low-pressure return message and a profile recovery plan for {_business_name(merchant)}? "
        f"Reply YES or STOP."
    )


def _compose_review_theme(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    theme = payload.get("theme")
    occurrences = payload.get("occurrences_30d")
    quote = payload.get("common_quote")
    if not theme and merchant.get("review_themes"):
        rt = merchant["review_themes"][0]
        theme = rt.get("theme")
        occurrences = rt.get("occurrences_30d")
        quote = rt.get("common_quote")
    theme_text = _clean_topic(theme or "review pattern")
    occ = f" appeared {_num(occurrences)} times in 30d" if occurrences is not None else " is showing up in reviews"
    quote_line = f"; quote: \"{_one_line(quote)}\"" if quote else ""
    return (
        f"{_salutation(category, merchant)}, review signal: {theme_text}{occ}{quote_line}. "
        f"Want me to draft the public reply plus a short ops note to prevent repeats? Reply YES or STOP."
    )


def _compose_milestone(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    metric = _clean_topic(payload.get("metric") or "profile activity")
    value_now = payload.get("value_now")
    milestone = payload.get("milestone_value")
    if value_now is not None and milestone is not None:
        fact = f"{metric} is at {_num(value_now)}, just {_num(int(milestone) - int(value_now))} short of {_num(milestone)}"
    else:
        fact = _performance_line(merchant, category)
    return (
        f"{_salutation(category, merchant)}, milestone moment: {fact}. "
        f"Want me to turn it into a thank-you post and a review ask for recent customers? Reply YES or STOP."
    )


def _compose_planning(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    topic = _clean_topic(payload.get("intent_topic") or "your plan")
    last = _one_line(payload.get("merchant_last_message", ""))
    offer = _active_offer(merchant, category)
    agg = merchant.get("customer_aggregate", {}) or {}
    repeat = agg.get("repeat_customer_pct")
    lead = merchant.get("performance", {}).get("leads")

    if "corporate" in topic or "thali" in topic:
        draft = (
            f"Use {offer or 'your weekday offer'} as base: 10+ plates, order cutoff 11am, "
            "office delivery, add curd/sweet as paid add-on"
        )
    elif "kids" in topic or "yoga" in topic:
        draft = (
            f"Package {offer or 'trial class'} into a 4-week kids batch: Sat morning, parent WhatsApp updates, "
            "first class free, paid monthly seat after trial"
        )
    else:
        draft = f"Start from {offer or 'your best-performing service'} and turn it into one clear post + offer"

    proof = []
    if lead is not None:
        proof.append(f"{_num(lead)} leads in 30d")
    if repeat is not None:
        proof.append(f"{_pct(repeat, signed=False)} repeat customers")
    proof_line = f" Context: {', '.join(proof)}." if proof else ""
    last_line = f" You said: \"{last}\"." if last else ""
    return (
        f"{_salutation(category, merchant)},{last_line} Here is the draft direction for {topic}: {draft}."
        f"{proof_line} Want me to write the exact customer-facing copy now? Reply YES or STOP."
    )


def _compose_category_seasonal(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    trends = payload.get("trends") or []
    item = _find_digest(category, trigger)
    if trends:
        fact = ", ".join(_clean_topic(t) for t in trends[:4])
    else:
        fact = _digest_sentence(item) or "seasonal demand is shifting"
    return (
        f"{_salutation(category, merchant)}, seasonal demand shift: {fact}. "
        f"{_performance_line(merchant, category)}. Want me to draft the shelf/menu/profile update for this week? "
        f"Reply YES or STOP."
    )


def _compose_gbp(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    uplift = payload.get("estimated_uplift_pct")
    path = _clean_topic(payload.get("verification_path") or "phone or postcard")
    uplift_line = f" estimated discovery uplift {_pct(uplift, signed=False)}" if uplift is not None else " better discovery"
    return (
        f"{_salutation(category, merchant)}, your Google profile verification is still pending. "
        f"Path: {path};{uplift_line} once completed. Want me to send the exact verification checklist? Reply YES or STOP."
    )


def _compose_supply(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    item = _find_digest(category, trigger)
    molecule = payload.get("molecule", "affected stock")
    batches = ", ".join(payload.get("affected_batches", []) or [])
    maker = payload.get("manufacturer")
    maker_line = f" from {maker}" if maker else ""
    batch_line = f" Batches: {batches}." if batches else ""
    return (
        f"{_salutation(category, merchant)}, stock alert: {molecule}{maker_line}. {_digest_sentence(item)}."
        f"{batch_line} Want me to draft the shelf-pull checklist and affected-customer WhatsApp? Reply YES or STOP."
    )


def _compose_competitor(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    name = payload.get("competitor_name")
    distance = payload.get("distance_km")
    their_offer = payload.get("their_offer")
    opened = _date_label(payload.get("opened_date"))
    own_offer = _active_offer(merchant, category)
    if name:
        fact = f"{name} opened"
        if distance is not None:
            fact += f" {distance} km away"
        if opened:
            fact += f" on {opened}"
        if their_offer:
            fact += f" with {their_offer}"
    else:
        locality = _identity(merchant).get("locality", "your locality")
        fact = f"a nearby competitor signal is active in {locality}"
    own_line = f" Your current hook: {own_offer}." if own_offer else f" {_performance_line(merchant, category)}."
    return (
        f"{_salutation(category, merchant)}, {fact}.{own_line} Want me to draft a counter-positioning post that uses your real strengths? "
        f"Reply YES or STOP."
    )


def _compose_ipl(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    payload = trigger.get("payload", {}) or {}
    match = payload.get("match", "tonight's match")
    venue = payload.get("venue")
    time_label = _date_label(payload.get("match_time_iso"))
    offer = _active_offer(merchant, category)
    place = f" near {venue}" if venue else ""
    offer_line = f" You already have {offer}." if offer else ""
    return (
        f"{_salutation(category, merchant)}, {match}{place} at {time_label} is a same-day demand window."
        f"{offer_line} Want me to draft a delivery-first match-night WhatsApp/post before orders start? Reply YES or STOP."
    )


def _compose_curious(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    category_word = {
        "dentists": "treatment",
        "salons": "service",
        "restaurants": "dish or combo",
        "gyms": "class or program",
        "pharmacies": "product category",
    }.get(category.get("slug"), "service")
    body = (
        f"{_salutation(category, merchant)}, quick check: {_performance_line(merchant, category)}. "
        f"What {category_word} is most in demand this week? Reply with the name; I will turn it into one post/offer draft. STOP to pause."
    )
    return _finish(body, "open_ended", "vera", trigger, _rationale("curious_ask_due", merchant, trigger))


def _compose_customer(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not customer:
        body = (
            f"{_salutation(category, merchant)}, this is a customer trigger but customer context is missing. "
            f"Want me to prepare the merchant-facing draft once the customer profile arrives? Reply YES or STOP."
        )
        return _finish(body, "binary YES/STOP", "vera", trigger, _rationale(trigger.get("kind", "customer"), merchant, trigger))

    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {}) or {}
    cust = _customer_name(customer)
    business = _business_name(merchant)
    offer = _active_offer(merchant, category)
    offer_line = f" {offer} is available." if offer else ""
    value_line = _customer_value_line(customer)
    value_sentence = f" ({value_line})" if value_line else ""
    base = f"Hi {cust}, {business} here."

    if kind == "appointment_tomorrow":
        body = (
            f"{base} Reminder for your appointment tomorrow{value_sentence}. "
            f"Reply C to confirm, R to reschedule, or STOP."
        )
        return _finish(body, "open_ended", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    if kind == "recall_due":
        service = _clean_topic(payload.get("service_due") or "follow-up")
        last = _date_label(payload.get("last_service_date") or customer.get("relationship", {}).get("last_visit"))
        due = _date_label(payload.get("due_date"))
        slots = payload.get("available_slots") or []
        if slots:
            slot_labels = " / ".join(_one_line(s.get("label")) for s in slots[:2])
            slot_line = f" Slots ready: {slot_labels}."
        else:
            slot_line = " Reply YES and we will share this week's slots."
        due_line = f" due on {due}" if due else " due now"
        last_line = f" Last visit: {last}." if last else ""
        body = f"{base} Your {service}{due_line}.{last_line}{slot_line}{offer_line} Reply YES/slot number or STOP."
        return _finish(body, "open_ended", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    if kind in {"customer_lapsed_soft", "customer_lapsed_hard"}:
        state = _clean_topic(customer.get("state") or kind)
        body = (
            f"{base} We have not seen you since {_date_label(customer.get('relationship', {}).get('last_visit'))}. "
            f"You are marked {state} with {value_line or 'past visit history'}.{offer_line} Reply YES for available slots/offers, or STOP."
        )
        return _finish(body, "binary YES/STOP", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    if kind == "chronic_refill_due":
        molecules = payload.get("molecule_list") or []
        stock_out = _date_label(payload.get("stock_runs_out_iso"))
        if category.get("slug") == "pharmacies" or molecules:
            meds = ", ".join(molecules) if molecules else "your regular medicines"
            address = " Saved address is on file." if payload.get("delivery_address_saved") else ""
            body = (
                f"{base} Your refill for {meds} is due"
                f"{' before ' + stock_out if stock_out else ''}.{address} Reply YES for delivery/packing, or STOP."
            )
        else:
            body = (
                f"{base} Your follow-up is due based on your last visit{value_sentence}. "
                f"Reply YES for available slots, or STOP."
            )
        return _finish(body, "binary YES/STOP", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    if kind == "trial_followup":
        trial = _date_label(payload.get("trial_date") or customer.get("relationship", {}).get("last_visit"))
        options = payload.get("next_session_options") or []
        next_slot = _one_line(options[0].get("label")) if options else "the next available slot"
        body = (
            f"{base} Thanks for trying us on {trial}. Next option: {next_slot}."
            f"{offer_line} Reply YES to hold the slot, or STOP."
        )
        return _finish(body, "binary YES/STOP", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    if kind == "wedding_package_followup":
        wedding = _date_label(payload.get("wedding_date"))
        trial = _date_label(payload.get("trial_completed"))
        window = _clean_topic(payload.get("next_step_window_open") or "prep window")
        body = (
            f"{base} Your bridal trial was on {trial}, and wedding date is {wedding}. "
            f"The {window} is open now. Reply YES and we will share the prep plan, or STOP."
        )
        return _finish(body, "binary YES/STOP", "merchant_on_behalf", trigger, _rationale(kind, merchant, trigger))

    body = (
        f"{base} Quick follow-up from your last visit{value_sentence}.{offer_line} "
        f"Reply YES if you want us to share the best slot/offer for this week, or STOP."
    )
    return _finish(body, "binary YES/STOP", "merchant_on_behalf", trigger, _rationale(kind or "customer_followup", merchant, trigger))


def _compose_merchant(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    kind = trigger.get("kind", "")
    builder_body = None
    cta = "binary YES/STOP"

    if kind == "research_digest":
        builder_body = _compose_research(category, merchant, trigger)
    elif kind == "regulation_change":
        builder_body = _compose_compliance(category, merchant, trigger)
    elif kind == "cde_opportunity":
        builder_body = _compose_cde(category, merchant, trigger)
    elif kind == "perf_dip":
        builder_body = _compose_perf(category, merchant, trigger, spike=False)
    elif kind == "perf_spike":
        builder_body = _compose_perf(category, merchant, trigger, spike=True)
    elif kind == "renewal_due":
        builder_body = _compose_renewal(category, merchant, trigger)
    elif kind == "festival_upcoming":
        builder_body = _compose_festival(category, merchant, trigger)
    elif kind in {"winback_eligible", "dormant_with_vera"}:
        builder_body = _compose_winback(category, merchant, trigger)
    elif kind == "review_theme_emerged":
        builder_body = _compose_review_theme(category, merchant, trigger)
    elif kind == "milestone_reached":
        builder_body = _compose_milestone(category, merchant, trigger)
    elif kind == "active_planning_intent":
        builder_body = _compose_planning(category, merchant, trigger)
    elif kind == "seasonal_perf_dip":
        builder_body = _compose_category_seasonal(category, merchant, trigger)
    elif kind == "category_seasonal":
        builder_body = _compose_category_seasonal(category, merchant, trigger)
    elif kind == "gbp_unverified":
        builder_body = _compose_gbp(category, merchant, trigger)
    elif kind == "supply_alert":
        builder_body = _compose_supply(category, merchant, trigger)
    elif kind == "competitor_opened":
        builder_body = _compose_competitor(category, merchant, trigger)
    elif kind == "ipl_match_today":
        builder_body = _compose_ipl(category, merchant, trigger)
    elif kind == "curious_ask_due":
        return _compose_curious(category, merchant, trigger)
    else:
        offer = _active_offer(merchant, category)
        offer_line = f" Best current hook: {offer}." if offer else ""
        builder_body = (
            f"{_salutation(category, merchant)}, quick Vera check for {_business_name(merchant)}: "
            f"{_performance_line(merchant, category)}.{offer_line} Want me to draft one timely WhatsApp/post for this week? "
            f"Reply YES or STOP."
        )

    return _finish(builder_body, cta, "vera", trigger, _rationale(kind or "general", merchant, trigger))


def compose(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic challenge composer. It uses only provided context and returns
    the schema expected by the submission and HTTP judge.
    """
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    if trigger.get("scope") == "customer" or customer:
        return _compose_customer(category, merchant, trigger, customer)
    return _compose_merchant(category, merchant, trigger)


def _trigger_priority(
    trigger: Dict[str, Any],
    merchant: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
) -> int:
    kind = trigger.get("kind", "")
    priority = int(trigger.get("urgency") or 0) * 100
    priority += {
        "supply_alert": 90,
        "active_planning_intent": 85,
        "regulation_change": 80,
        "chronic_refill_due": 75,
        "recall_due": 74,
        "appointment_tomorrow": 72,
        "customer_lapsed_hard": 70,
        "customer_lapsed_soft": 68,
        "trial_followup": 66,
        "perf_dip": 62,
        "renewal_due": 60,
        "review_theme_emerged": 58,
        "competitor_opened": 55,
        "gbp_unverified": 54,
        "ipl_match_today": 52,
        "category_seasonal": 48,
        "festival_upcoming": 44,
        "cde_opportunity": 40,
        "research_digest": 38,
        "seasonal_perf_dip": 35,
        "winback_eligible": 34,
        "dormant_with_vera": 32,
        "milestone_reached": 25,
        "perf_spike": 22,
        "curious_ask_due": 12,
    }.get(kind, 20)

    signals = set(merchant.get("signals", []) or [])
    if any("engaged_in_last" in signal for signal in signals):
        priority += 20
    if "perf_dip_severe" in signals or any("below_peer" in signal for signal in signals):
        priority += 15
    subscription = merchant.get("subscription", {}) or {}
    if subscription.get("status") in {"expired", "trial"}:
        priority += 10
    days_remaining = subscription.get("days_remaining")
    if isinstance(days_remaining, int) and days_remaining <= 14:
        priority += 10
    if customer and (customer.get("preferences", {}) or {}).get("reminder_opt_in"):
        priority += 8
    return priority


def generate_composer_prompt(category, merchant, trigger, customer):
    # Backward-compatible helper; retained for notebooks/scripts that imported it.
    return {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
    }


def generate_reply_prompt(conversation_history, latest_message, from_role):
    lines = [f"[{turn['from']}]: {turn['msg']}" for turn in conversation_history]
    return "\n".join(lines + [f"Latest from {from_role}: {latest_message}"])


def _contexts_loaded_counts() -> Dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _ in contexts.keys():
        if scope in counts:
            counts[scope] += 1
    return counts


def _demo_context_id(scope: str, payload: Dict[str, Any], fallback: str) -> str:
    if scope == "category":
        return str(payload.get("slug") or fallback)
    if scope == "merchant":
        return str(payload.get("merchant_id") or fallback)
    if scope == "customer":
        return str(payload.get("customer_id") or fallback)
    if scope == "trigger":
        return str(payload.get("id") or fallback)
    return fallback


def _load_demo_contexts() -> Dict[str, int]:
    base = Path(__file__).resolve().parent / "dataset" / "expanded"
    folders = {
        "category": "categories",
        "merchant": "merchants",
        "customer": "customers",
        "trigger": "triggers",
    }
    if not base.exists():
        raise HTTPException(status_code=404, detail="dataset/expanded is missing")

    loaded = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, folder in folders.items():
        source_dir = base / folder
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            context_id = _demo_context_id(scope, payload, path.stem)
            contexts[(scope, context_id)] = {"version": 1, "payload": payload}
            loaded[scope] += 1
    return loaded


@app.get("/ui/state")
async def ui_state():
    triggers = []
    for (scope, context_id), record in contexts.items():
        if scope != "trigger":
            continue
        payload = record.get("payload") or {}
        triggers.append(
            {
                "id": context_id,
                "kind": payload.get("kind"),
                "merchant_id": payload.get("merchant_id"),
                "customer_id": payload.get("customer_id"),
                "urgency": payload.get("urgency"),
            }
        )
    triggers.sort(key=lambda item: item["id"])

    return {
        "contexts_loaded": _contexts_loaded_counts(),
        "triggers": triggers,
        "last_actions": last_tick_actions[-MAX_ACTIONS_PER_TICK:],
        "conversations": [
            {"conversation_id": conv_id, "history": history[-30:]}
            for conv_id, history in list(conversations.items())[-30:]
        ],
    }


@app.post("/ui/load-demo-data")
async def ui_load_demo_data():
    contexts.clear()
    sent_suppression_keys.clear()
    conversations.clear()
    conversation_contexts.clear()
    ended_conversations.clear()
    seen_messages.clear()
    last_tick_actions.clear()
    loaded = _load_demo_contexts()
    return {
        "accepted": True,
        "loaded": loaded,
        "contexts_loaded": _contexts_loaded_counts(),
    }


@app.post("/ui/reset")
async def ui_reset():
    contexts.clear()
    sent_suppression_keys.clear()
    conversations.clear()
    conversation_contexts.clear()
    ended_conversations.clear()
    seen_messages.clear()
    last_tick_actions.clear()
    return {
        "accepted": True,
        "contexts_loaded": _contexts_loaded_counts(),
    }


@app.get("/healthz")
@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": _contexts_loaded_counts(),
    }


@app.get("/metadata")
@app.get("/v1/metadata")
async def metadata():
    team_members = [
        member.strip()
        for member in os.environ.get("TEAM_MEMBERS", "Soman").split(",")
        if member.strip()
    ]
    return {
        "team_name": os.environ.get("TEAM_NAME", "Team Soman"),
        "team_members": team_members or ["Soman"],
        "model": "deterministic-local-composer",
        "approach": "FastAPI stateful bot with ranked trigger selection and deterministic context-grounded templates",
        "contact_email": os.environ.get("CONTACT_EMAIL", ""),
        "version": os.environ.get("BOT_VERSION", "2.0.1"),
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/context")
@app.post("/v1/context")
async def push_context(request: Request):
    body = await _parse_json_model(request, CtxBody)
    if body.scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope", "details": body.scope}
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": cur["version"],
        }
    if cur and cur["version"] == body.version:
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z",
            "idempotent": True,
        }
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


@app.post("/tick")
@app.post("/v1/tick")
async def tick(request: Request):
    global last_tick_actions
    body = await _parse_json_model(request, TickBody)
    actions = []
    candidates = []
    for order, trg_id in enumerate(body.available_triggers):
        trigger = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trigger:
            continue
        suppression_key = trigger.get("suppression_key") or f"trigger:{trg_id}"
        if suppression_key in sent_suppression_keys:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        if not merchant:
            continue

        category = contexts.get(("category", merchant.get("category_slug")), {}).get("payload") or {}
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None
        priority = _trigger_priority(trigger, merchant, customer)
        candidates.append((priority, -order, trg_id, trigger, merchant_id, customer_id, merchant, category, customer))

    candidates.sort(reverse=True)

    for _, _, trg_id, trigger, merchant_id, customer_id, merchant, category, customer in candidates[:MAX_ACTIONS_PER_TICK]:
        msg = compose(category, merchant, trigger, customer)
        conv_id = f"conv_{next(conversation_seq)}_{merchant_id}_{trg_id}"

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": msg["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v2",
            "template_params": [],
            "body": msg["body"],
            "cta": msg["cta"],
            "suppression_key": msg["suppression_key"],
            "rationale": msg["rationale"],
        }
        actions.append(action)
        sent_suppression_keys.add(msg["suppression_key"] or f"trigger:{trg_id}")
        conversations.setdefault(conv_id, []).append({"from": "bot", "msg": msg["body"]})
        conversation_contexts[conv_id] = {
            "category": category,
            "merchant": merchant,
            "trigger": trigger,
            "customer": customer,
            "action": action,
        }

    last_tick_actions = actions
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


AUTO_REPLY_PATTERNS = (
    "thank you for contacting",
    "we will respond shortly",
    "will respond shortly",
    "business hours",
    "automated",
    "auto-reply",
    "autoreply",
    "away message",
)

HOSTILE_PATTERNS = (
    "stop messaging",
    "stop msg",
    "unsubscribe",
    "spam",
    "useless",
    "not interested",
    "do not message",
    "don't message",
    "leave me",
    "remove me",
)

OUT_OF_SCOPE_PATTERNS = (
    "gst",
    "tax filing",
    "file my tax",
    "itr",
    "loan",
    "legal notice",
    "accounting",
)

INTENT_PATTERNS = (
    "lets do it",
    "let's do it",
    "go ahead",
    "proceed",
    "yes",
    "ok do",
    "okay do",
    "send it",
    "please do",
    "what next",
    "whats next",
    "what's next",
)


def _reply_send(body: str, cta: str, rationale: str) -> Dict[str, Any]:
    return {
        "action": "send",
        "body": _one_line(body),
        "cta": cta,
        "rationale": _one_line(rationale),
    }


@app.post("/reply")
@app.post("/v1/reply")
async def reply(request: Request):
    body = await _parse_json_model(request, ReplyBody)
    if body.conversation_id in ended_conversations:
        return {
            "action": "end",
            "rationale": "Conversation was already closed; no further messages sent.",
        }

    conv_history = conversations.setdefault(body.conversation_id, [])
    conv_history.append({"from": body.from_role, "msg": body.message})

    latest = _one_line(body.message)
    low = latest.lower()
    user_msgs = [
        _one_line(turn["msg"]).lower()
        for turn in conv_history
        if turn.get("from") == body.from_role
    ]
    repeated_three = len(user_msgs) >= 3 and user_msgs[-1] == user_msgs[-2] == user_msgs[-3]
    repeated_two = len(user_msgs) >= 2 and user_msgs[-1] == user_msgs[-2]

    if any(pattern in low for pattern in HOSTILE_PATTERNS) or low.strip() in {"stop", "no stop"}:
        ended_conversations.add(body.conversation_id)
        return {
            "action": "end",
            "rationale": "User asked to stop or signaled hostility; ending gracefully.",
        }

    is_auto_reply = any(pattern in low for pattern in AUTO_REPLY_PATTERNS)
    if repeated_three:
        ended_conversations.add(body.conversation_id)
        return {
            "action": "end",
            "rationale": "Same message repeated three times; treating as auto-reply loop and closing.",
        }
    if is_auto_reply or repeated_two:
        return {
            "action": "wait",
            "wait_seconds": 14400,
            "rationale": "Likely WhatsApp Business auto-reply; backing off for four hours.",
        }

    if any(pattern in low for pattern in ("later", "tomorrow", "busy", "after some time")):
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "User asked for time, so Vera backs off for 30 minutes.",
        }

    ctx = conversation_contexts.get(body.conversation_id, {})
    merchant = ctx.get("merchant") or contexts.get(("merchant", body.merchant_id), {}).get("payload") or {}
    business = _business_name(merchant)

    if any(pattern in low for pattern in OUT_OF_SCOPE_PATTERNS):
        response = (
            "I will leave that to your CA or specialist, since it is outside Vera's scope. "
            f"Coming back to {business}: I can continue with the approval-ready draft from the current trigger."
        )
        conv_history.append({"from": "bot", "msg": response})
        return _reply_send(response, "open_ended", "Politely declined out-of-scope ask and redirected to Vera's current task.")

    if any(pattern in low for pattern in INTENT_PATTERNS):
        if "abstract" in low or "patient" in low or "whatsapp" in low:
            response = (
                f"Sending the brief next for {business}. I will also keep the patient WhatsApp draft approval-ready "
                "with one clear CTA, using only the context we have."
            )
            conv_history.append({"from": "bot", "msg": response})
            return _reply_send(response, "none", "Merchant accepted and asked for concrete material; moving directly to action.")
        response = (
            f"Done. I will proceed with the draft for {business} now. "
            f"Next step: I will keep one approval-ready message here with the exact offer/post copy."
        )
        conv_history.append({"from": "bot", "msg": response})
        return _reply_send(response, "none", "Explicit intent detected; switched to action mode immediately.")

    if "price" in low or "cost" in low or "charge" in low:
        category = ctx.get("category") or {}
        offer = _active_offer(merchant, category)
        response = (
            f"Here is the grounded option I have from context: {offer or 'no active price offer is listed yet'}. "
            f"I can proceed with that, or keep the copy price-free."
        )
        conv_history.append({"from": "bot", "msg": response})
        return _reply_send(response, "open_ended", "Answered the pricing question using only available context.")

    response = (
        f"Got it. Here is the clean next move for {business}: I will keep the message specific, use only your profile data, "
        f"and send one approval-ready draft instead of more questions."
    )
    conv_history.append({"from": "bot", "msg": response})
    return _reply_send(response, "open_ended", "Continues conversation with a concrete next step.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
