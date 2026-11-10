"""Replay the live answer path against local corpora and disposable indexes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import date
from functools import partial
from pathlib import Path

from eval.config import CASE_TIMEOUT_S, PRODUCT_HITS, STALE_ROOTS
from eval.floor_arms import CORPORA, _unseen, _write_unseen
from eval.goldload import SuiteCase, synthetic_cases, task1_open_cases
from eval.metrics import GoldView, is_fabricated, metric_for
from eval.provenance import collapse
from eval.traps import PASSING, all_traps, judge
from ldi import config
from ldi.ask import _cap_metal, _complete, format_stale, settle_answer, stale_paths
from ldi.evidence import attach_source, build_evidence, format_evidence
from ldi.guard import assess_load, machine_snapshot
from ldi.ingest import ingest
from ldi.retrieve import retrieve
from ldi.store import Hit, Index, sha256_file
from ldi.verify import format_not_found
from ldi.weights import local_snapshot


class FreshReplyNeeded(Exception):
    pass


def _build(root: Path, source: Path) -> None:
    with Index(root / "data/index.sqlite") as index, contextlib.redirect_stderr(io.StringIO()):
        ingest(source, index, ocr=False)


def _build_stale(root: Path, source: Path, *, edited: bool) -> None:
    local_docs = root / "docs"
    shutil.copytree(source, local_docs)
    if edited:
        db = source.parent / "data/index.sqlite"
        with contextlib.closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)) as original:
            row = original.execute("SELECT body FROM chunks WHERE path = 'docs/policy.txt'").fetchone()
        (local_docs / "policy.txt").write_text(row[0] + "\n")
    old_root = os.environ.get("LDI_ROOT")
    os.environ["LDI_ROOT"] = str(root)
    try:
        _build(root, local_docs)
    finally:
        if old_root is None:
            os.environ.pop("LDI_ROOT")
        else:
            os.environ["LDI_ROOT"] = old_root
    if edited:
        shutil.copyfile(source / "policy.txt", local_docs / "policy.txt")


def _covers(anchors, texts) -> bool | None:
    if not anchors:
        return None
    pieces = tuple(collapse(text) for text in texts)
    return all(any(collapse(anchor) in text for text in pieces) for anchor in anchors)


def _replay(question, anchors, root, today, generate, recorded=None) -> dict:
    os.environ["LDI_ROOT"] = str(root)
    started = time.perf_counter()
    with Index(config.index_path()) as index:
        found = retrieve(index, question, PRODUCT_HITS, today=today, diagnostic=True)
        hits = attach_source(question, found.hits, index.spans_for)
        gaps = index.list_gaps()
        extraction = _covers(anchors, (text for (text,) in index.db.execute(
            "SELECT body FROM chunks UNION ALL SELECT text FROM spans"
        )))
    elapsed = time.perf_counter() - started
    items = build_evidence(question, hits)
    saved_items = [
        {"id": item.id, "path": item.path, "locator": item.locator, "text": item.text}
        for item in items
    ]
    replay_calls = (
        recorded.get("calls", []) if recorded is not None and recorded.get("items") == saved_items else []
    )
    row = {
        "extraction": extraction,
        "candidates": _covers(anchors, (hit.text for hit in found.candidates)),
        "hits": _covers(anchors, (text for hit in hits for text in (hit.text, *hit.source_lines))),
        "evidence": _covers(anchors, (item.text for item in items)),
        "search_s": sum(found.log[key] for key in ("retrieve_s", "fuse_s", "gate_s")),
        "prepare_s": elapsed,
        "evidence_chars": len(format_evidence(items)),
        "items": saved_items,
        "calls": [],
    }
    if hits and (stale := stale_paths(hits)):
        row["shown"] = format_stale(stale)
    elif not items:
        row["shown"] = format_not_found(gaps)
    elif generate is not None or replay_calls:
        def answer(reason):
            started = time.perf_counter()
            offset = len(row["calls"])
            reused = offset < len(replay_calls) and replay_calls[offset]["reason"] == reason
            if reused:
                reply = replay_calls[offset]["reply"]
            elif generate is not None:
                reply = generate(question, hits, reason, items=items)
            else:
                raise FreshReplyNeeded
            row["calls"].append({
                "reason": reason, "reply": reply, "reused": reused,
                "seconds": None if reused else time.perf_counter() - started,
            })
            return reply

        try:
            _, released = settle_answer(answer, hits, question, items=items)
        except FreshReplyNeeded:
            return row
        row["kind"] = released.kind
        row["timeout"] = sum(call["seconds"] or 0 for call in row["calls"]) > CASE_TIMEOUT_S
        stale = stale_paths([Hit(path, None, "") for path in released.paths]) if released.paths else []
        row["shown"] = (
            format_stale(stale) if stale else format_not_found(gaps)
            if released.kind in {"not_found", "retry"} else released.text
        )
    return row


def _run(args, scratch: Path, generate) -> None:
    case_list = synthetic_cases() + task1_open_cases() + _unseen()
    traps = all_traps()
    sources = dict(CORPORA)
    indexes = {}
    if args.cases:
        payload = json.loads(args.cases.read_text())
        sources.update({key: Path(value) for key, value in payload["corpora"].items()})
        indexes = {key: Path(value) for key, value in payload.get("indexes", {}).items()}
        case_list = [
            SuiteCase(
                row["id"], row.get("category", "reported"), row["question"],
                GoldView(row.get("abstain", False), tuple(row.get("anchors", ())), True),
                row["root"],
            )
            for row in payload["cases"]
        ]
        traps = []
    recorded = {}
    if args.recorded:
        with args.recorded.open() as source:
            recorded = {
                row["id"]: row for row in map(json.loads, source) if "id" in row
            }
    if args.ids:
        selected = set(args.ids.split(","))
        case_list = [case for case in case_list if case.id in selected]
        traps = [trap for trap in traps if trap.id in selected]
        missing = selected - {case.id for case in case_list} - {trap.id for trap in traps}
        if missing:
            raise SystemExit(f"Unknown cases: {', '.join(sorted(missing))}")
    if any(case.root_name == "unseen" for case in case_list):
        sources["unseen"] = scratch / "unseen-docs"
        _write_unseen(sources["unseen"], None)
    manifest = {}
    for name in sorted({case.root_name for case in case_list}):
        source = sources[name]
        dest = scratch / name
        if name in indexes:
            target = dest / "data/index.sqlite"
            target.parent.mkdir(parents=True)
            with contextlib.closing(sqlite3.connect(f"{indexes[name].as_uri()}?mode=ro", uri=True)) as original:
                with contextlib.closing(sqlite3.connect(target)) as copy:
                    original.backup(copy)
        elif name in STALE_ROOTS.values():
            _build_stale(dest, source, edited=name == STALE_ROOTS["STALE-03"])
        else:
            _build(dest, source)
        manifest[name] = {
            str(path.relative_to(source)): sha256_file(path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix.lower() in config.SUPPORTED_SUFFIXES
        }
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"])).hexdigest(),
        "date": str(args.date),
        "extractor": config.EXTRACTOR_VERSION,
        "chunker": config.CHUNKER_VERSION,
        "model": str(args.snapshot) if generate is not None else None,
        "product_sources": {
            str(path): sha256_file(path) for path in sorted(Path("ldi").glob("*.py"))
        },
        "document_first": args.document_first,
        "corpora": manifest,
    }
    with args.output.open("x") as out:
        out.write(json.dumps({"metadata": metadata}) + "\n")
        for case in case_list:
            row = _replay(
                case.question, case.gold.anchors, scratch / case.root_name, args.date, generate,
                recorded.get(case.id),
            )
            shown = row.get("shown")
            row.update(id=case.id, metric=metric_for(case.gold, shown) if shown is not None else None)
            row["fabricated"] = is_fabricated(shown) if shown is not None else None
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(case.id, row["metric"], flush=True)
        for trap in traps:
            root = scratch / trap.id
            source = root / "docs"
            source.mkdir(parents=True)
            for name, content in trap.files:
                if name in trap.selected:
                    (source / name).write_text(content)
            _build(root, source)
            row = _replay(trap.question, (), root, args.date, generate, recorded.get(trap.id))
            shown = row.get("shown")
            verdict = judge(trap, shown) if shown is not None else None
            row.update(id=trap.id, metric=verdict, trap_pass=verdict in PASSING)
            row["fabricated"] = is_fabricated(shown) if shown is not None else None
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(trap.id, verdict, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--ids", help="Comma-separated case ids")
    parser.add_argument("--cases", type=Path, help="Local JSON containing extra corpora and cases")
    parser.add_argument("--model", action="store_true", help="Generate fresh replies with local Granite")
    parser.add_argument("--recorded", type=Path, help="Reuse replies only when the evidence and retry match")
    parser.add_argument("--document-first", action="store_true", help="Evaluate the document-first route")
    args = parser.parse_args()
    config.DOCUMENT_FIRST_RECONCILIATION = args.document_first
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    with tempfile.TemporaryDirectory(prefix="ldi-replay-") as scratch:
        if not args.model:
            _run(args, Path(scratch), None)
            return
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        spec = config.MODELS["1b"]
        args.snapshot = local_snapshot(spec.repo)
        if args.snapshot is None:
            raise SystemExit("Local Granite weights are required.")
        snap = machine_snapshot()
        assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
        _cap_metal(mx, spec.metal_limit_bytes)
        model, tokenizer = load(str(args.snapshot), trust_remote_code=False)
        generate = partial(_complete, mx, generate_step, model, tokenizer, make_sampler(temp=0.0))
        _run(args, Path(scratch), generate)
        print(f"Peak Metal memory: {mx.get_peak_memory()} bytes")


if __name__ == "__main__":
    main()
