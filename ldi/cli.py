"""Command line for local document questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ldi import __version__
from ldi.ask import ModelNotReady, answer
from ldi.chat import chat
from ldi.config import (
    CHAT_MODEL,
    DENSE_ENABLED,
    DEFAULT_MODEL,
    EMBEDDINGS,
    MAX_TOP_CHUNKS,
    MODELS,
    OCR_MODELS,
    REDUCED_RETRIEVAL_MODE,
    RESERVE_BYTES,
    TOP_CHUNKS,
    gb,
    index_path,
    setup_models,
)
from ldi.guard import (
    GuardError,
    assess_setup,
    download_block_reason,
    load_block_reason,
    machine_snapshot,
    pressure_label,
)
from ldi.ingest import format_coverage, ingest
from ldi.probe import probe
from ldi.prompt import format_passages
from ldi.retrieve import retrieve
from ldi.store import Index
from ldi.verify import format_not_found
from ldi.weights import download, local_snapshot


def main(argv: list[str] | None = None) -> int:
    try:
        return _dispatch(argv)
    except (GuardError, ModelNotReady, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2 if isinstance(exc, (GuardError, ValueError)) else 3
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


def _dispatch(argv: list[str] | None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        _status()
    elif args.command == "models":
        _models()
    elif args.command == "setup":
        spec = setup_models()[args.model]
        if spec.key == "ocr":
            from ldi.ocr import setup_ocr

            setup_ocr()
        else:
            download(spec)
    elif args.command == "embed":
        from ldi.embed import embed_check

        embed_check()
    elif args.command == "ingest":
        _ingest(args.path, reset=args.reset)
    elif args.command == "search":
        _search(args.question, _top(args.top))
    elif args.command == "ask":
        answer(args.question, args.model, _top(args.top))
    elif args.command == "chat":
        chat(args.model, once=args.once)
    elif args.command == "probe":
        probe()
    elif args.command == "eval":
        from ldi.evalrun import evaluate

        evaluate()
    elif args.command == "ui":
        from ldi.ui import serve

        serve(args.port)
    else:
        parser.error("Choose a command.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ldi",
        description="Ask questions about documents on this Mac. The model stays here.",
    )
    parser.add_argument("--version", action="version", version=f"ldi {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="Show disk, memory, and whether a model may load")
    commands.add_parser("models", help="Show the models this tool will run")
    commands.add_parser(
        "probe",
        help="Load the 1B 4-bit weights once and measure memory",
    )
    commands.add_parser(
        "eval",
        help="Score the 1B model on five skill conditions, then unload it",
    )
    chat_cmd = commands.add_parser("chat", help="Talk with the local Granite 4.0 1B model")
    chat_cmd.add_argument("--model", choices=tuple(MODELS), default=CHAT_MODEL)
    chat_cmd.add_argument("--once", help="Send one message, then unload the model")

    ui_cmd = commands.add_parser("ui", help="Serve the local chat page on this Mac")
    ui_cmd.add_argument("--port", type=int, default=8765)

    setup = commands.add_parser("setup", help="Download weights once. Does not load them")
    setup.add_argument("--model", choices=tuple(setup_models()), default=DEFAULT_MODEL)
    commands.add_parser(
        "embed",
        help="Embed one passage and one question with BGE, then unload it",
    )

    ingest_cmd = commands.add_parser("ingest", help="Index text, PDF, Word, Excel, and PNG files")
    ingest_cmd.add_argument("path", type=Path)
    ingest_cmd.add_argument("--reset", action="store_true", help="Clear the index first")

    search = commands.add_parser("search", help="Show matching passages. Does not load a model")
    search.add_argument("question")
    search.add_argument("--top", type=int, default=TOP_CHUNKS)

    ask = commands.add_parser("ask", help="Answer from indexed passages with a local model")
    ask.add_argument("question")
    ask.add_argument("--model", choices=tuple(MODELS), default=DEFAULT_MODEL)
    ask.add_argument("--top", type=int, default=TOP_CHUNKS)
    return parser


def _top(value: int) -> int:
    if not 1 <= value <= MAX_TOP_CHUNKS:
        raise ValueError(f"Ask for 1 to {MAX_TOP_CHUNKS} passages.")
    return value


def _ingest(path: Path, *, reset: bool) -> None:
    with Index(index_path()) as index:
        report = ingest(path, index, reset=reset)
    if report.files == 0 and report.unchanged == 0 and report.skipped == 0:
        print("No supported documents were found.")
        return
    files = "file" if report.files == 1 else "files"
    chunks = "chunk" if report.chunks == 1 else "chunks"
    print(
        f"Indexed {report.files} {files}, {report.chunks} {chunks}. "
        f"Left {report.unchanged} unchanged."
    )
    printed = format_coverage(report)
    if printed:
        print(printed)


def _search(question: str, top: int) -> None:
    path = index_path()
    if not path.exists():
        raise ValueError("The index is empty. Run ldi ingest on a folder of documents.")
    with Index(path) as index:
        found = retrieve(index, question, limit=top)
        gaps = index.list_gaps()
    if not found.hits:
        print(format_not_found(gaps))
    else:
        print(format_passages(found.hits))
    if not DENSE_ENABLED:
        print(REDUCED_RETRIEVAL_MODE)


def _status() -> None:
    snap = machine_snapshot()
    print(f"Memory pressure: {pressure_label(snap.pressure)}")
    if snap.reclaimable is None:
        print("Reclaimable memory: unreadable")
    else:
        print(f"Reclaimable memory: {gb(snap.reclaimable)}")
    print(f"Disk free: {gb(snap.disk_free)}. Keep {gb(RESERVE_BYTES)} free.")
    if snap.on_battery is True:
        print("Power: battery. Plug in for a long answer.")
    elif snap.on_battery is False:
        print("Power: plugged in.")
    if snap.swap_used is not None:
        print(f"Swap used: {gb(snap.swap_used)}")
        if snap.swap_used > 1024**3:
            print("Swap is in use. A restart clears a long session before the first model load.")
    print()
    db = index_path()
    if db.exists():
        with Index(db) as index:
            files, chunks = index.stats()
        file_word = "file" if files == 1 else "files"
        chunk_word = "chunk" if chunks == 1 else "chunks"
        print(f"Index: {files} {file_word}, {chunks} {chunk_word}. {db}")
    else:
        print(f"Index: empty. It will be created at {db}")
    print()
    _model_lines(snap.pressure, snap.reclaimable, snap.disk_free, snap.swap_used)


def _models() -> None:
    snap = machine_snapshot()
    _model_lines(snap.pressure, snap.reclaimable, snap.disk_free, snap.swap_used)


def _model_lines(
    pressure: int | None,
    reclaimable: int | None,
    disk_free: int,
    swap_used: int | None,
) -> None:
    print("Models")
    for spec in MODELS.values():
        _print_spec(spec, pressure, reclaimable, disk_free, swap_used)
    print()
    print("Embeddings")
    for spec in EMBEDDINGS.values():
        _print_spec(spec, pressure, reclaimable, disk_free, swap_used)
        print("      It loads, encodes, and unloads before a chat model loads.")
    print()
    print("OCR")
    for spec in OCR_MODELS.values():
        _print_spec(spec, pressure, reclaimable, disk_free, swap_used)
        print("      It loads for a scanned page and unloads before a chat model loads.")


def _print_spec(
    spec,
    pressure: int | None,
    reclaimable: int | None,
    disk_free: int,
    swap_used: int | None,
) -> None:
    print(f"  {spec.key}  {spec.label}")
    print(f"      {spec.repo}")
    local = local_snapshot(spec.repo)
    if local is None:
        print("      weights are not on disk")
        disk_reason = download_block_reason(spec.approx_bytes, disk_free)
        if disk_reason is not None:
            print(f"      {disk_reason}")
        else:
            try:
                assess_setup(pressure)
            except GuardError as exc:
                print(f"      {exc}")
            else:
                print("      a download is allowed. It does not load the model.")
    else:
        print(f"      weights are on disk: {local}")
    reason = load_block_reason(spec, pressure, reclaimable, swap_used)
    if reason is None:
        print("      a load is allowed")
    else:
        print(f"      {reason}")
