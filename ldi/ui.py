"""A chat page on this Mac. Nothing on the page is loaded from the network."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from ldi.chat import SYSTEM
from ldi.desk import Desk
from ldi.guard import GuardError

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local chat</title>
<style>
  :root {
    color-scheme: light;
    --ink: #1c1915;
    --muted: #5c564c;
    --line: #e4dccf;
    --paper: #f6f3ec;
    --card: #fffdf8;
    --green: #0f6e56;
    --green-ink: #ffffff;
    --danger: #8c2f2f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 16px/1.45 ui-sans-serif, system-ui, sans-serif;
    color: var(--ink);
    background: var(--paper);
  }
  header, main { padding: 20px; }
  header { padding-bottom: 0; }
  h1, h2, h3, h4 { margin: 0 0 8px; font-weight: 600; }
  h1 { font-size: 1.6rem; }
  h2 { font-size: 1.05rem; }
  p { margin: 0 0 8px; }
  .eyebrow { color: var(--muted); font-size: 0.85rem; margin-bottom: 4px; }
  #status, #template-note, #kept, #notice { color: var(--muted); }
  #notice { color: var(--danger); min-height: 1.2em; }
  .check { color: var(--muted); margin: 4px 0 0; }
  .models { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
  .split { display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.2fr); gap: 16px; align-items: start; }
  article, .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px;
  }
  .panel + .panel, article + article { margin-top: 12px; }
  label { display: block; font-size: 0.85rem; color: var(--muted); margin: 8px 0 4px; }
  textarea, pre {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fff;
    color: var(--ink);
    font: 0.92rem/1.4 ui-monospace, SFMono-Regular, monospace;
    padding: 10px;
  }
  textarea { resize: vertical; }
  #base { min-height: 4.5rem; }
  #rendered, .passage { max-height: 16rem; overflow: auto; white-space: pre-wrap; }
  #message { min-height: 4.5rem; font-family: ui-sans-serif, system-ui, sans-serif; }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; margin-top: 8px; }
  .fields label { margin: 0; }
  input[type="number"] {
    display: block;
    width: 100%;
    margin-top: 4px;
    padding: 6px 8px;
    border: 1px solid var(--line);
    border-radius: 8px;
    font: inherit;
    color: var(--ink);
    background: #fff;
  }
  button, input[type="file"] { font: inherit; }
  button {
    background: var(--green);
    color: var(--green-ink);
    border: 0;
    border-radius: 8px;
    padding: 8px 14px;
    cursor: pointer;
  }
  button:disabled { background: #b7b1a6; cursor: default; }
  button.quiet { background: transparent; color: var(--ink); border: 1px solid var(--line); }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  #log { min-height: 12rem; max-height: 28rem; overflow: auto; }
  #log article h3 { font-size: 0.8rem; color: var(--muted); }
  #log article p { white-space: pre-wrap; }
  @media (max-width: 800px) {
    .models, .split { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <p class="eyebrow">On this Mac</p>
  <h1>Local chat</h1>
  <p id="status">Reading models on this Mac… reduced retrieval mode</p>
</header>
<main>
  <div id="models" class="models"></div>
  <div class="split">
    <div>
      <section class="panel">
        <h2>System prompt</h2>
        <label for="base">Base sentence</label>
        <textarea id="base">__SYSTEM__</textarea>
        <div class="row">
          <button id="update-prompt" type="button">Update prompt</button>
        </div>
        <p id="template-note"></p>
        <h3>What the model receives</h3>
        <pre id="rendered">__SYSTEM__</pre>
        <p id="kept"></p>
      </section>
      <section class="panel">
        <h2>Documents</h2>
        <p>Longer attached text is cut from the end. The limit is Attached text under Generation.</p>
        <div id="passages"><p>No file is attached.</p></div>
        <label for="file">Attach a .txt, .md, .pdf, or image file</label>
        <input id="file" type="file" accept=".txt,.md,.markdown,.rst,.pdf,.png,.jpg,.jpeg,.webp">
      </section>
    </div>
    <section class="panel">
      <h2>Chat</h2>
      <div id="log"></div>
      <h3>Generation</h3>
      <div class="fields">
        <label>Temperature
          <input id="temperature" type="number" min="0" max="2" step="0.1" value="0">
        </label>
        <label>Output tokens
          <input id="max_tokens" type="number" min="1" max="1024" step="16" value="256">
        </label>
        <label>Top-p
          <input id="top_p" type="number" min="0" max="1" step="0.05" value="0">
        </label>
        <label>Top-k
          <input id="top_k" type="number" min="0" max="200" step="1" value="0">
        </label>
        <label>Min-p
          <input id="min_p" type="number" min="0" max="1" step="0.05" value="0">
        </label>
        <label>Attached text
          <input id="document_tokens" type="number" min="64" max="2048" step="64" value="2048">
        </label>
        <label>Messages kept
          <input id="history_messages" type="number" min="0" max="24" step="1" value="6">
        </label>
      </div>
      <p id="generation-note">Temperature 0 picks the most likely next word. The model can stop before the output token limit. Top-p, top-k, and min-p apply when temperature is above 0.</p>
      <button id="update-generation" type="button">Update generation</button>
      <p id="notice" role="status"></p>
      <form id="talk">
        <label for="message">Message</label>
        <textarea id="message" placeholder="Ask a question or just talk"></textarea>
        <div class="row">
          <button id="send" type="submit">Send</button>
          <button id="clear" class="quiet" type="button">Clear chat</button>
        </div>
      </form>
    </section>
  </div>
</main>
<script>
const status = document.querySelector("#status");
const models = document.querySelector("#models");
const base = document.querySelector("#base");
const rendered = document.querySelector("#rendered");
const templateNote = document.querySelector("#template-note");
const kept = document.querySelector("#kept");
const passages = document.querySelector("#passages");
const fileInput = document.querySelector("#file");
const log = document.querySelector("#log");
const notice = document.querySelector("#notice");
const form = document.querySelector("#talk");
const message = document.querySelector("#message");
const sendButton = document.querySelector("#send");

function render(state) {
  const loaded = state.models.find((model) => model.loaded);
  const mode = state.retrieval_mode ? ` ${state.retrieval_mode}.` : "";
  status.textContent = loaded
    ? `Loaded ${loaded.label}. Memory pressure: ${state.memory}.${mode}`
    : `No model is loaded. Memory pressure: ${state.memory}.${mode}`;
  if (document.activeElement !== base) base.value = state.base;
  rendered.textContent = state.rendered;
  templateNote.textContent = state.template_note;
  kept.textContent = state.kept || "";
  fillGeneration(state.generation);
  renderModels(state);
  renderPassages(state);
  renderLog(state);
}

const generationFields = [
  "temperature",
  "max_tokens",
  "top_p",
  "top_k",
  "min_p",
  "document_tokens",
  "history_messages",
];

function fillGeneration(settings) {
  if (!settings) return;
  for (const name of generationFields) {
    const input = document.querySelector("#" + name);
    if (input && document.activeElement !== input) input.value = settings[name];
  }
}

function generationPayload() {
  const payload = {};
  for (const name of generationFields) {
    payload[name] = Number(document.querySelector("#" + name).value);
  }
  return payload;
}

function renderModels(state) {
  models.replaceChildren();
  for (const model of state.models) {
    const card = document.createElement("article");
    const title = document.createElement("h2");
    title.textContent = model.label;
    const detail = document.createElement("p");
    detail.textContent = `Metal cap ${model.metal}. ${model.detail}`;
    card.append(title, detail);
    if (model.loaded) {
      const button = document.createElement("button");
      button.type = "button";
      button.id = "unload";
      button.textContent = "Unload";
      button.addEventListener("click", () => {
        button.disabled = true;
        post("/api/unload").then(render).catch(fail);
      });
      card.append(button);
    } else if (model.can_load) {
      const button = document.createElement("button");
      button.type = "button";
      button.id = "load-" + model.key;
      button.textContent = "Load";
      button.addEventListener("click", () => {
        button.disabled = true;
        status.textContent = `Loading ${model.label}…`;
        post("/api/load", {model: model.key}).then(render).catch(fail);
      });
      card.append(button);
    }
    models.append(card);
  }
}

function renderPassages(state) {
  passages.replaceChildren();
  if (!state.passages.length) {
    const empty = document.createElement("p");
    empty.textContent = "No file is attached.";
    passages.append(empty);
    return;
  }
  for (const item of state.passages) {
    const block = document.createElement("article");
    const title = document.createElement("h3");
    title.textContent = `${item.title} · ${item.tokens} tokens`;
    const body = document.createElement("pre");
    body.className = "passage";
    body.textContent = item.text;
    const detach = document.createElement("button");
    detach.type = "button";
    detach.className = "quiet";
    detach.textContent = `Remove ${item.title}`;
    detach.addEventListener("click", () => {
      post("/api/detach", {name: item.title}).then(render).catch(fail);
    });
    block.append(title, body, detach);
    passages.append(block);
  }
}

function renderLog(state) {
  log.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement("p");
    empty.textContent = "The thread is empty.";
    log.append(empty);
    return;
  }
  for (const item of state.messages) {
    const block = document.createElement("article");
    const who = document.createElement("h3");
    who.textContent = item.role === "user" ? "You" : "Model";
    const body = document.createElement("p");
    body.textContent = item.content;
    block.append(who, body);
    for (const check of item.checks || []) {
      const line = document.createElement("p");
      line.className = "check";
      const wrote = check.wrote || check.detail || "";
      line.textContent = `${check.kind}: ${check.status}. Model wrote ${wrote}. Python computed ${check.computed}.`;
      block.append(line);
    }
    log.append(block);
  }
  log.scrollTop = log.scrollHeight;
}

async function post(path, body) {
  notice.textContent = "";
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "The request failed.");
  return payload;
}

let refreshing = false;

function fail(error) {
  notice.textContent = error.message || "The request failed.";
  if (refreshing) return;
  refreshing = true;
  fetch("/api/state")
    .then((response) => response.json())
    .then((state) => {
      const message = notice.textContent;
      render(state);
      notice.textContent = message;
    })
    .catch(() => {})
    .finally(() => {
      refreshing = false;
    });
}

function refresh() {
  return fetch("/api/state").then((response) => response.json()).then(render).catch(fail);
}

async function attachText(name, text) {
  render(await post("/api/attach", {name, text}));
}

document.querySelector("#update-prompt").addEventListener("click", () => {
  post("/api/prompt", {text: base.value}).then(render).catch(fail);
});

fileInput.addEventListener("change", async () => {
  const chosen = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!chosen) return;
  try {
    const lower = chosen.name.toLowerCase();
    const binaryFile = lower.endsWith(".pdf") || /[.](png|jpe?g|webp)$/.test(lower);
    if (binaryFile) {
      const bytes = new Uint8Array(await chosen.arrayBuffer());
      let binary = "";
      const size = 8192;
      for (let index = 0; index < bytes.length; index += size) {
        binary += String.fromCharCode(...bytes.subarray(index, index + size));
      }
      render(await post("/api/attach", {name: chosen.name, file_base64: btoa(binary)}));
    } else {
      await attachText(chosen.name, await chosen.text());
    }
  } catch (error) {
    fail(error);
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = message.value.trim();
  if (!text) return;
  sendButton.disabled = true;
  post("/api/chat", {message: text}).then((state) => {
    message.value = "";
    render(state);
  }).catch(fail).finally(() => {
    sendButton.disabled = false;
  });
});

document.querySelector("#update-generation").addEventListener("click", () => {
  post("/api/generation", generationPayload()).then(render).catch(fail);
});

document.querySelector("#clear").addEventListener("click", () => {
  post("/api/clear").then(render).catch(fail);
});

window.addEventListener("pagehide", () => {
  fetch("/api/unload", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: "{}",
    keepalive: true,
  });
});

refresh();
</script>
</body>
</html>
"""


def page() -> str:
    return _PAGE.replace("__SYSTEM__", html.escape(SYSTEM))


class Handler(BaseHTTPRequestHandler):
    server: LocalServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._bytes(200, page().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._run(self.server.desk.state)
            return
        self._json(404, {"error": "That page is not on this server."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        desk = self.server.desk
        routes = {
            "/api/load": lambda: desk.load(str(payload.get("model", ""))),
            "/api/unload": desk.unload,
            "/api/prompt": lambda: desk.set_prompt(str(payload.get("text", ""))),
            "/api/generation": lambda: desk.set_generation(payload),
            "/api/attach": lambda: desk.attach_payload(payload),
            "/api/detach": lambda: desk.detach(str(payload.get("name", ""))),
            "/api/clear": desk.clear,
            "/api/chat": lambda: desk.ask(str(payload.get("message", ""))),
        }
        action = routes.get(path)
        if action is None:
            self._json(404, {"error": "That page is not on this server."})
            return
        self._run(action)

    def log_message(self, fmt: str, *args) -> None:
        return

    def _run(self, action) -> None:
        try:
            action()
        except GuardError as exc:
            self._json(409, {"error": str(exc)})
            return
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        try:
            self._json(200, self.server.desk.state())
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length < 0 or length > 40 * 1024 * 1024:
            raise ValueError("That upload is larger than 32 MB.")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("The request was not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("The request was not a JSON object.")
        return payload

    def _json(self, status: int, payload: dict) -> None:
        self._bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'none'; font-src 'none'",
        )
        self.end_headers()
        self.wfile.write(body)


class LocalServer(HTTPServer):
    desk: Desk

    def __init__(self, desk: Desk, port: int) -> None:
        self.desk = desk
        super().__init__(("127.0.0.1", port), Handler)


def serve(port: int = 8765) -> None:
    desk = Desk()
    try:
        server = LocalServer(desk, port)
    except OSError as exc:
        raise SystemExit(f"127.0.0.1:{port} is already in use.") from exc
    host, actual = server.server_address
    print(f"Local chat is at http://{host}:{actual}", flush=True)
    print("Closing this page, or stopping this command, unloads the model.", flush=True)
    try:
        server.serve_forever()
    finally:
        desk.unload()
        server.server_close()
        print("The model is unloaded.", flush=True)
