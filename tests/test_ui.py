import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ldi.chat import SYSTEM
from ldi.config import MODELS
from ldi.desk import (
    Desk,
    clip_documents,
    document_kwargs,
    question_text,
    system_text,
    uses_documents,
)
from ldi.guard import GuardError
from ldi.ui import LocalServer, page
from ldi.weights import local_snapshot


class _Plain:
    chat_template = "You are a helpful assistant."


class _Documents:
    chat_template = "{% if documents %}{{ documents }}{% endif %}"


def test_page_has_no_network_assets():
    html = page()
    assert "System prompt" in html
    assert "What the model receives" in html
    assert SYSTEM in html
    assert "http://" not in html
    assert "https://" not in html
    assert "reduced retrieval mode" in html


def test_system_text_reads_the_granite_system_role():
    rendered = (
        "<|start_of_role|>system<|end_of_role|>"
        "You are a helpful assistant. Keep answers short and direct."
        "\n\nDocument text"
        "<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>"
    )
    assert system_text(rendered) == (
        "You are a helpful assistant. Keep answers short and direct.\n\nDocument text"
    )


def test_clip_documents_cuts_from_the_end():
    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(ids: list[int]) -> str:
        return "".join(chr(item) for item in ids)

    passages, note = clip_documents([("note.md", "abcdefghij")], 4, encode, decode)
    assert passages == [{"title": "note.md", "text": "abcd", "tokens": 4}]
    assert "kept 4 of 10" in note

    passages, note = clip_documents(
        [("a.md", "abcd"), ("b.md", "ef")],
        4,
        encode,
        decode,
    )
    assert [item["title"] for item in passages] == ["a.md"]
    assert "b.md" in note


def test_granite_slot_and_plain_template_differ():
    passages = [{"title": "sample.md", "text": "SEAL-441"}]
    assert uses_documents(_Documents())
    assert document_kwargs(_Documents(), passages)["documents"][0]["text"] == "SEAL-441"
    assert question_text("What is the seal?", passages, _Documents()) == "What is the seal?"
    assert document_kwargs(_Plain(), passages) == {}
    assert "SEAL-441" in question_text("What is the seal?", passages, _Plain())


def test_ui_command_parses():
    from ldi.cli import _parser

    args = _parser().parse_args(["ui", "--port", "9"])
    assert args.command == "ui"
    assert args.port == 9


def test_load_requires_weights_on_disk(monkeypatch):
    monkeypatch.setattr("ldi.desk.local_snapshot", lambda repo: None)
    with pytest.raises(ValueError, match="No weights on disk yet"):
        Desk().load("1b")


def test_question_is_refused_until_a_model_is_loaded(monkeypatch):
    monkeypatch.setattr("ldi.desk.local_snapshot", lambda repo: None)
    desk = Desk()
    with pytest.raises(GuardError, match="Load a model"):
        desk.ask("Hello")
    desk.attach_text("sample.md", "The replacement seal is part SEAL-441.")
    state = desk.state()
    assert state["loaded"] is None
    assert state["rendered"] == SYSTEM
    assert state["passages"][0]["text"] == "The replacement seal is part SEAL-441."
    with pytest.raises(ValueError, match="txt"):
        desk.attach_text("notes.exe", "hello")


def test_localhost_page_refuses_chat_when_unloaded(monkeypatch):
    monkeypatch.setattr("ldi.desk.local_snapshot", lambda repo: None)
    server = LocalServer(Desk(), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        assert server.server_address[0] == "127.0.0.1"
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            html = response.read().decode()
        assert "System prompt" in html
        with urlopen(f"http://127.0.0.1:{port}/api/state") as response:
            state = json.load(response)
        assert [model["key"] for model in state["models"]] == ["1b", "3b", "7b"]
        assert state["loaded"] is None
        request = Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=json.dumps({"message": "Hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as caught:
            urlopen(request)
        assert caught.value.code == 409
        assert "Load a model" in caught.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_generation_defaults_and_bounds(monkeypatch):
    monkeypatch.setattr("ldi.desk.local_snapshot", lambda repo: None)
    desk = Desk()
    state = desk.state()
    assert state["generation"] == {
        "temperature": 0.0,
        "max_tokens": 256,
        "top_p": 0.0,
        "top_k": 0,
        "min_p": 0.0,
        "document_tokens": 2048,
        "history_messages": 6,
    }
    desk.set_generation(
        {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "document_tokens": 512,
            "history_messages": 2,
        }
    )
    assert desk.generation.temperature == 0.7
    assert desk.generation.max_tokens == 128
    assert desk.generation.top_k == 40
    with pytest.raises(ValueError, match="Temperature"):
        desk.set_generation({"temperature": 3})
    with pytest.raises(ValueError, match="Output tokens"):
        desk.set_generation({"max_tokens": 0})


def test_page_shows_generation_controls():
    html = page()
    assert "Temperature" in html
    assert "Output tokens" in html
    assert "Top-p" in html
    assert "Top-k" in html
    assert "Min-p" in html
    assert 'id="document_tokens"' in html
    assert 'id="history_messages"' in html


def test_desk_does_not_download_or_stream():
    source = Path(__file__).resolve().parents[1].joinpath("ldi", "desk.py").read_text()
    assert "HF_HUB_OFFLINE" in source
    assert "generate_step" in source
    assert "stream_generate" not in source
    assert "snapshot_download" not in source
    load = source.split("def load(", 1)[1].split("def unload(", 1)[0]
    assert load.index("self.unload()") < load.index("self._load_weights")


def test_granite_template_appends_the_document():
    if local_snapshot(MODELS["1b"].repo) is None:
        pytest.skip("Granite weights are not on disk")
    desk = Desk()
    desk.attach_text("sample.md", "The replacement seal is part SEAL-441.")
    state = desk.state()
    assert state["template_model"] == "1b"
    assert state["uses_documents"] is True
    assert state["rendered"].startswith(SYSTEM)
    assert "SEAL-441" in state["rendered"]
    assert "strictly aligning with the facts" in state["rendered"]
    assert "Hello" not in state["rendered"]
