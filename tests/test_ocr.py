import base64
from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from ldi.cli import main
from ldi.config import MODELS
from ldi.desk import Desk
from ldi.ocr import classify_page


def test_a_short_page_is_text_and_a_blank_page_is_not_a_scan():
    assert classify_page("Only a few lines.", 0) == "text"
    assert classify_page("   ", 0) == "blank"
    assert classify_page("", 2) == "scan"


def test_text_ingest_does_not_start_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    called = []
    monkeypatch.setattr("ldi.ingest.recognize", lambda scans: called.append(scans))
    note = tmp_path / "note.md"
    note.write_text("The replacement seal is SEAL-441.\n")
    assert main(["ingest", str(note)]) == 0
    assert called == []


def test_a_blank_pdf_does_not_start_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    called = []
    monkeypatch.setattr("ldi.ingest.recognize", lambda scans: called.append(scans))
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(_pdf_with_text(""))
    assert main(["ingest", str(pdf)]) == 0
    assert called == []


def test_an_image_only_page_is_queued(tmp_path, monkeypatch):
    monkeypatch.setenv("LDI_ROOT", str(tmp_path))
    seen = []

    def fake(scans):
        seen.append(scans)
        return ["scanned line"]

    monkeypatch.setattr("ldi.ingest.recognize", fake)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(_pdf_with_image())
    assert main(["ingest", str(pdf)]) == 0
    assert len(seen) == 1
    assert seen[0][0].page == 1


def test_attach_unloads_ocr_then_restores_the_chat_model(monkeypatch):
    desk = Desk()
    desk.model = object()
    desk.tokenizer = object()
    desk.spec = MODELS["1b"]
    order = []

    def unload():
        order.append("unload")
        desk.model = None
        desk.tokenizer = None
        desk.spec = None

    def load(key):
        order.append(("load", key))
        assert desk.model is None
        desk.spec = MODELS[key]
        desk.model = object()
        desk.tokenizer = object()

    def fake(scans):
        order.append("ocr")
        assert desk.model is None
        assert scans[0].page == 1
        return ["SEAL-441"]

    monkeypatch.setattr(desk, "unload", unload)
    monkeypatch.setattr(desk, "load", load)
    monkeypatch.setattr("ldi.desk.recognize", fake)
    payload = {
        "name": "scan.pdf",
        "file_base64": base64.b64encode(_pdf_with_image()).decode(),
    }
    desk.attach_payload(payload)
    assert order == ["unload", "ocr", ("load", "1b")]
    assert desk.spec.key == "1b"
    assert "SEAL-441" in desk.files[0][1]


def test_chat_cannot_select_the_ocr_model():
    from ldi.cli import _parser

    args = _parser().parse_args(["setup", "--model", "ocr"])
    assert args.model == "ocr"
    with pytest.raises(SystemExit):
        _parser().parse_args(["ask", "hello", "--model", "ocr"])


def _pdf_with_text(text: str) -> bytes:
    from tests.test_tool import _pdf_with_text as build

    return build(text)


def _pdf_with_image() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    image = DecodedStreamObject()
    image.set_data(bytes([255, 0, 0]))
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    resources = page["/Resources"].get_object()
    xobjects = resources.get("/XObject")
    if xobjects is None:
        xobjects = DictionaryObject()
        resources[NameObject("/XObject")] = xobjects
    else:
        xobjects = xobjects.get_object()
    xobjects[NameObject("/Im0")] = writer._add_object(image)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
