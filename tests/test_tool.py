import sys
from pathlib import Path

import pytest

from ldi.chunk import chunk_text
from ldi.cli import main
from ldi.config import (
    DEFAULT_MODEL,
    EMBEDDINGS,
    MODELS,
    OCR_MODELS,
    catalog_cap_from_peak,
    setup_models,
)
from ldi.guard import (
    GuardError,
    assess_download,
    assess_load,
    parse_on_battery,
    parse_swap_used,
    parse_vm_stat,
    reclaimable_from,
)
from ldi.ingest import ingest
from ldi.prompt import build_messages
from ldi.store import Hit, Index, match_query


VM_STAT = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     1000.
Pages inactive:                                10000.
Pages speculative:                               100.
Pages purgeable:                                5000.
"""


def test_catalog_keeps_3b_as_the_default():
    assert set(MODELS) == {"1b", "3b", "7b"}
    assert DEFAULT_MODEL == "3b"
    assert MODELS["1b"].repo == "mlx-community/granite-4.0-1b-4bit"
    assert MODELS["1b"].metal_limit_bytes == 1280 * 1024**2
    assert MODELS["1b"].metal_limit_bytes == catalog_cap_from_peak(1166512822)
    assert MODELS["3b"].metal_limit_bytes == 4 * 1024**3
    assert MODELS["7b"].metal_limit_bytes == 7 * 1024**3
    assert MODELS["3b"].repo == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert MODELS["7b"].repo == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert EMBEDDINGS["bge"].repo == "mlx-community/bge-small-en-v1.5-8bit"
    assert EMBEDDINGS["bge"].metal_limit_bytes == 512 * 1024**2
    assert "ocr" not in MODELS
    assert setup_models()["ocr"] is OCR_MODELS["ocr"]
    assert OCR_MODELS["ocr"].repo == "jmbarrancoidener/PaddleOCR-VL-1.6-mlx-8bit"
    assert OCR_MODELS["ocr"].metal_limit_bytes == 2 * 1024**3
    assert "mlx_vlm" not in Path("ldi/ocr.py").read_text()
    assert "mlx-vlm" not in Path("pyproject.toml").read_text()


def test_small_model_may_load_while_swap_is_in_use():
    spec = MODELS["1b"]
    assess_load(spec, 1, spec.metal_limit_bytes, swap_used=3 * 1024**3)
    assess_load(EMBEDDINGS["bge"], 1, EMBEDDINGS["bge"].metal_limit_bytes, swap_used=3 * 1024**3)
    with pytest.raises(GuardError, match="red"):
        assess_load(spec, 4, spec.metal_limit_bytes, swap_used=3 * 1024**3)


def test_1b_cap_comes_from_the_measured_peak():
    spec = MODELS["1b"]
    assert spec.metal_limit_bytes <= 2 * 1024**3
    assess_load(spec, 1, spec.metal_limit_bytes, swap_used=2 * 1024**3)
    with pytest.raises(GuardError, match="red"):
        assess_load(spec, 4, spec.metal_limit_bytes, swap_used=0)


def test_embedding_runtime_stays_uninstalled(capsys):
    from ldi.cli import _parser

    source = Path("ldi/embed.py").read_text()
    project = Path("pyproject.toml").read_text()
    assert "mlx_embeddings" not in source
    assert "mlx_vlm" not in source
    assert "mlx-embeddings" not in project
    assert "mlx-vlm" not in project
    args = _parser().parse_args(["setup", "--model", "bge"])
    assert args.model == "bge"
    with pytest.raises(SystemExit):
        _parser().parse_args(["chat", "--model", "bge"])
    assert main(["embed"]) == 2
    assert "will not install it" in capsys.readouterr().err


def test_catalog_cap_rounds_up_inside_2gb():
    assert catalog_cap_from_peak(0) == 256 * 1024**2
    assert catalog_cap_from_peak(300 * 1024**2) == 512 * 1024**2
    assert catalog_cap_from_peak(512 * 1024**2) == 768 * 1024**2
    assert catalog_cap_from_peak(3 * 1024**3) == 2 * 1024**3


def test_chat_keeps_the_recent_turns():
    from ldi.chat import trim_history

    messages = [{"role": "system", "content": "stay"}]
    for turn in range(5):
        messages.append({"role": "user", "content": f"user {turn}"})
        messages.append({"role": "assistant", "content": f"model {turn}"})
    trimmed = trim_history(messages, limit=4)
    assert trimmed[0] == {"role": "system", "content": "stay"}
    assert [item["content"] for item in trimmed[1:]] == [
        "user 3",
        "model 3",
        "user 4",
        "model 4",
    ]


def test_chat_stays_offline():
    source = Path("ldi/chat.py").read_text()
    assert "HF_HUB_OFFLINE" in source
    assert "generate_step" in source
    assert "stream_generate" not in source
    source = Path("ldi/probe.py").read_text()
    assert "HF_HUB_OFFLINE" in source
    assert "generate_step" in source
    assert "stream_generate" not in source
    assert "set_memory_limit" not in source or "_cap_metal" in source


def test_reclaimable_memory_skips_purgeable_double_count():
    page, pages = parse_vm_stat(VM_STAT)
    assert page == 16384
    assert reclaimable_from(page, pages) == 16384 * (1000 + 10000 + 100)


def test_swap_and_power_parsers():
    assert parse_swap_used("vm.swapusage: total = 4096.00M  used = 2462.81M  free = 1.00M") == int(
        2462.81 * 1024**2
    )
    assert parse_on_battery("Now drawing from 'AC Power'") is False
    assert parse_on_battery("Now drawing from 'Battery Power'") is True


def test_download_keeps_25gb_free():
    assess_download(5 * 1024**3, 31 * 1024**3)
    with pytest.raises(GuardError):
        assess_download(5 * 1024**3, 29 * 1024**3)


def test_load_waits_for_green_or_yellow_and_enough_memory():
    spec = MODELS["3b"]
    assess_load(spec, 1, spec.metal_limit_bytes)
    assess_load(spec, 2, spec.metal_limit_bytes)
    assess_load(spec, 1, spec.metal_limit_bytes, swap_used=200 * 1024**2)
    with pytest.raises(GuardError, match="Swap"):
        assess_load(spec, 1, spec.metal_limit_bytes, swap_used=2 * 1024**3)
    with pytest.raises(GuardError):
        assess_load(spec, 4, spec.metal_limit_bytes)
    with pytest.raises(GuardError):
        assess_load(spec, 1, spec.metal_limit_bytes - 1)
    with pytest.raises(GuardError):
        assess_load(spec, None, spec.metal_limit_bytes)


def test_chunks_cover_a_long_note_and_keep_paragraphs():
    text = "A" * 5000
    parts = chunk_text(text, size=1200, overlap=200)
    assert parts[0].startswith("A")
    assert parts[-1].endswith("A")
    assert "".join(parts).count("A") >= 5000
    assert len(parts) < 20

    mixed = ("north pump " * 30) + "\n\n" + ("south pump " * 30)
    pieces = chunk_text(mixed, size=400, overlap=40)
    assert any("north" in piece and "south" not in piece for piece in pieces)
    assert any("south" in piece for piece in pieces)


def test_prompt_stays_short():
    hits = [Hit("notes.md", 2, "x" * 5000) for _ in range(4)]
    messages = build_messages("Where is the seal?", hits)
    message = messages[1]["content"]
    assert "notes.md, page 2" in message
    assert message.startswith("Evidence:")
    assert "[E1]" in message
    assert "Passages:" not in message
    assert "exact words from the file" not in messages[0]["content"]
    assert len(message) < 5000


def test_ingest_search_and_skip_hidden_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "notes.md").write_text(
        "The north pump was serviced on 12 March. The replacement seal is SEAL-441.\n"
    )
    (folder / ".secret.md").write_text("hidden part SEAL-999\n")
    nested = folder / ".venv"
    nested.mkdir()
    (nested / "skip.md").write_text("venv part SEAL-000\n")

    assert main(["ingest", str(folder)]) == 0
    assert main(["search", "replacement seal"]) == 0
    found = capsys.readouterr().out
    assert "SEAL-441" in found
    assert "SEAL-999" not in found
    assert "SEAL-000" not in found

    assert main(["ingest", str(folder)]) == 0
    again = capsys.readouterr().out
    assert "Left 1 unchanged" in again

    assert main(["search", "???"]) == 2


def test_pdf_page_is_searchable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    pdf = tmp_path / "record.pdf"
    pdf.write_bytes(_pdf_with_text("SEAL-441"))
    assert main(["ingest", str(pdf)]) == 0
    assert main(["search", "SEAL-441"]) == 0
    output = capsys.readouterr().out
    assert "record.pdf, page 1" in output
    assert "SEAL-441" in output


def test_ask_does_not_load_or_download_without_weights(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    note = tmp_path / "notes.md"
    note.write_text("The replacement seal is SEAL-441.\n")
    assert main(["ingest", str(note)]) == 0
    monkeypatch.setattr("ldi.ask.local_snapshot", lambda repo: None)
    assert main(["ask", "replacement seal"]) == 3
    captured = capsys.readouterr()
    assert "SEAL-441" in captured.out
    assert "ldi setup" in captured.err
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)


def test_ask_module_has_no_network_client():
    source = Path("ldi/ask.py").read_text()
    assert "HF_HUB_OFFLINE" in source
    assert "snapshot_download" not in source
    for banned in ("openai", "anthropic", "httpx", "requests"):
        assert banned not in source


def test_refuses_the_disk_root_and_a_larger_model(tmp_path):
    with Index(tmp_path / "index.sqlite") as index:
        with pytest.raises(ValueError, match="documents folder"):
            ingest(Path("/"), index)
    with pytest.raises(SystemExit):
        main(["setup", "--model", "32b"])


def test_oversized_file_is_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    monkeypatch.setattr("ldi.ingest.MAX_FILE_BYTES", 20)
    note = tmp_path / "big.md"
    note.write_text("The replacement seal is SEAL-441 and this line is longer than twenty bytes.\n")
    assert main(["ingest", str(note)]) == 0
    captured = capsys.readouterr()
    assert "larger than 32 MB" in captured.err
    assert main(["search", "seal"]) == 0
    missed = capsys.readouterr().out
    assert missed.startswith("I could not find supporting evidence in the indexed text")
    assert "big.md" in missed
    assert "skipped" in missed


def test_match_query_quotes_operator_words():
    assert match_query('seal OR "north"') == '"seal" AND "north"'
    assert match_query("seal NOT north") == '"seal" AND "NOT" AND "north"'


def _pdf_with_text(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET\n".encode("ascii")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    objects[3] = b"4 0 obj\n" + objects[3] + b"endobj\n"
    content = b"%PDF-1.4\n"
    offsets = []
    for item in objects:
        offsets.append(len(content))
        content += item
    xref_at = len(content)
    content += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    content += b"0000000000 65535 f \n"
    for offset in offsets:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return content
