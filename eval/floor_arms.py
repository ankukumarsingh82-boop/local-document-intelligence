"""Score the floor fixes through ingest and ldi ask.

Each arm is cumulative. The model, top-k, and evidence budget stay the same.
Indexes are temporary. The suite indexes are not opened. OCR is not run.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    PYTHONDONTWRITEBYTECODE="1",
)
sys.dont_write_bytecode = True

from eval.config import PRODUCT_HITS, RESULTS_DIR, SUITE_ROOT  # noqa: E402
from eval.goldload import SuiteCase, synthetic_cases, task1_open_cases  # noqa: E402
from eval.metrics import GoldView, anchors_covered, is_fabricated, metric_for  # noqa: E402
from eval.splits import assign, read_splits  # noqa: E402
from ldi import config  # noqa: E402
from ldi.ask import (  # noqa: E402
    _complete,
    _gaps,
    _hits,
    format_stale,
    settle_answer,
    stale_paths,
)
from ldi.chunker import uncovered_lines  # noqa: E402
from ldi.config import MODELS, TOP_CHUNKS, index_path  # noqa: E402
from ldi.evidence import EVIDENCE_LIMIT, build_evidence  # noqa: E402
from ldi.guard import assess_load, machine_snapshot  # noqa: E402
from ldi.ingest import ingest  # noqa: E402
from ldi.spans import PassageRecord  # noqa: E402
from ldi.store import Hit, Index  # noqa: E402
from ldi.verify import format_not_found  # noqa: E402
from ldi.weights import local_snapshot  # noqa: E402

WORK = RESULTS_DIR / "floor-arms"
SHOWN_PATH = WORK / "shown.jsonl"
REPORT_PATH = WORK / "report.json"
CORPORA = {
    "main": SUITE_ROOT / "corpus",
    "task1": SUITE_ROOT / "task1" / "documents",
    "stale1": SUITE_ROOT / "idx" / "stale1" / "docs",
    "stale3": SUITE_ROOT / "idx" / "stale3" / "docs",
}
BASE = {
    "carry": False,
    "sentence": False,
    "source": False,
    "cover": False,
    "target": 1200,
    "rows": 3,
}


def main() -> None:
    if PRODUCT_HITS != TOP_CHUNKS:
        raise SystemExit("PRODUCT_HITS must match the ask path's TOP_CHUNKS.")
    if EVIDENCE_LIMIT != 12:
        raise SystemExit("The evidence budget for this measurement is 12 lines.")
    sets = _sets()
    _expect_sizes(sets)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    spec = MODELS["1b"]
    local = local_snapshot(spec.repo)
    if local is None:
        raise SystemExit("The 1b weights are not on disk. This run does not download them.")
    snap = machine_snapshot()
    assess_load(spec, snap.pressure, snap.reclaimable, snap.swap_used)
    print(
        json.dumps(
            {
                "commit": commit,
                "model": spec.key,
                "weights": str(local),
                "top_k": PRODUCT_HITS,
                "evidence_lines": EVIDENCE_LIMIT,
                "pressure": snap.pressure,
                "reclaimable": snap.reclaimable,
            }
        ),
        flush=True,
    )
    jobs = _jobs()
    uncovered = {}
    for job in jobs:
        if job["reuse"]:
            continue
        uncovered[job["name"]] = _ingest_job(job, sets)
    _print_uncovered(uncovered)
    answers = _load_shown()
    import mlx.core as mx
    from ldi.ask import _cap_metal
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    _cap_metal(mx, spec.metal_limit_bytes)
    model, tokenizer = load(str(local), trust_remote_code=False)
    sampler = make_sampler(temp=0.0)
    try:
        for job in jobs:
            _answer_job(
                job,
                sets,
                answers,
                mx,
                generate_step,
                model,
                tokenizer,
                sampler,
            )
    finally:
        del model
        del tokenizer
    norm = _norm_compare(answers)
    report = _report(sets, jobs, answers, uncovered, commit, spec.key, str(local), norm)
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)


def _sets() -> dict[str, list[SuiteCase]]:
    cases = [
        case
        for case in synthetic_cases() + task1_open_cases()
        if case.root_name in CORPORA
    ]
    splits = read_splits()
    missing = [(case.id, case.category) for case in cases if case.id not in splits]
    splits.update(assign(missing))
    dev = [
        case
        for case in cases
        if splits.get(case.id) == "dev"
        and case.gold.scored
        and case.gold.anchors
        and not case.gold.abstain
    ]
    regression = [
        case
        for case in cases
        if splits.get(case.id) == "heldout"
        and case.gold.scored
        and case.gold.anchors
        and not case.gold.abstain
    ]
    held_rest = [
        case
        for case in cases
        if splits.get(case.id) == "heldout" and case not in regression
    ]
    floor_ids = set(json.loads((ROOT / "eval" / "floor_ids.json").read_text()))
    by_id = {case.id: case for case in cases}
    floor = [by_id[case_id] for case_id in sorted(floor_ids)]
    task1 = [case for case in cases if case.root_name == "task1"]
    unseen = _unseen()
    return {
        "dev": dev,
        "regression": regression,
        "floor": floor,
        "task1": task1,
        "unseen": unseen,
        "held_rest": held_rest,
    }


def _expect_sizes(sets: dict[str, list[SuiteCase]]) -> None:
    wanted = {"dev": 48, "regression": 24, "floor": 52, "held_rest": 9}
    for name, size in wanted.items():
        if len(sets[name]) != size:
            raise SystemExit(f"{name} has {len(sets[name])} cases, wanted {size}.")
    if len(sets["unseen"]) < 10:
        raise SystemExit("The unseen set needs at least 10 cases.")


def _unseen() -> list[SuiteCase]:
    payload = json.loads((ROOT / "eval" / "unseen_cases.json").read_text())
    loaded = []
    for row in payload["cases"]:
        anchors = tuple(row["anchors"])
        loaded.append(
            SuiteCase(
                id=row["id"],
                category="unseen",
                question=row["question"],
                gold=GoldView(
                    abstain=bool(row["abstain"]),
                    anchors=() if row["abstain"] else anchors,
                    scored=True,
                ),
                root_name="unseen",
            )
        )
    return loaded


def _jobs() -> list[dict]:
    final = {**BASE, "sentence": True, "source": True, "cover": True}
    jobs = [
        {"name": "1", "flags": dict(BASE), "reuse": None},
        {"name": "1+2", "flags": {**BASE, "sentence": True}, "reuse": None},
        {"name": "1+2+3", "flags": {**BASE, "sentence": True, "source": True}, "reuse": None},
        {"name": "1+2+3+4", "flags": dict(final), "reuse": None},
        {"name": "1+2+3+4-repeat", "flags": dict(final), "reuse": "1+2+3+4"},
        {
            "name": "size-800",
            "flags": {**final, "target": 800, "table": 800},
            "reuse": None,
        },
        {
            "name": "size-1600",
            "flags": {**final, "target": 1600, "table": 1600},
            "reuse": None,
        },
        {"name": "rows-2", "flags": {**final, "rows": 2}, "reuse": "1+2+3+4"},
        {"name": "rows-4", "flags": {**final, "rows": 4}, "reuse": "1+2+3+4"},
        {"name": "nfc", "flags": dict(final), "reuse": None, "norm": "NFC", "only": "unseen"},
        {"name": "nfd", "flags": dict(final), "reuse": None, "norm": "NFD", "only": "unseen"},
    ]
    return jobs


def _apply(flags: dict) -> None:
    config.CARRY_AFTER_FINAL_CUT = flags["carry"]
    config.SENTENCE_NEEDS_BOUNDARY = flags["sentence"]
    config.SOURCE_EVIDENCE = flags["source"]
    config.COVER_EVERY_LINE = flags["cover"]
    config.PASSAGE_TARGET_CHARS = flags["target"]
    config.PASSAGE_MAX_CHARS = flags["target"] + 200
    config.TABLE_EVIDENCE_CHARS = flags.get("table", flags["target"] + 200)
    config.EVIDENCE_MATCHED_ROWS = flags["rows"]


def _ingest_job(job: dict, sets: dict[str, list[SuiteCase]]) -> dict[str, list[str]]:
    _apply(job["flags"])
    found: dict[str, list[str]] = {}
    only = job.get("only")
    roots = _roots_for(sets, only)
    for name, source in roots.items():
        dest = _root_dir(job["name"], name)
        os.environ["LDI_ROOT"] = str(dest)
        if name == "unseen":
            _write_unseen(dest / "docs", job.get("norm"))
            source = dest / "docs"
        print(f"ingest {job['name']} {name}", flush=True)
        log_path = WORK / "ingest.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log, contextlib.redirect_stderr(log):
            with Index(index_path()) as index:
                ingest(source, index, reset=True, ocr=False)
        found[name] = _uncovered()
    return found


def _roots_for(sets: dict[str, list[SuiteCase]], only: str | None) -> dict[str, Path]:
    if only == "unseen":
        return {"unseen": Path()}
    names = {case.root_name for group in sets.values() for case in group}
    roots = {name: CORPORA[name] for name in sorted(names) if name in CORPORA}
    roots["unseen"] = Path()
    return roots


def _root_dir(job: str, name: str) -> Path:
    dest = WORK / "idx" / job / name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _write_unseen(folder: Path, form: str | None) -> None:
    payload = json.loads((ROOT / "eval" / "unseen_cases.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    for doc in payload["documents"]:
        text = doc["text"]
        if form:
            text = unicodedata.normalize(form, text)
        (folder / doc["name"]).write_text(text)


def _uncovered() -> list[str]:
    missing: list[str] = []
    with Index(index_path()) as index:
        for (path,) in index.db.execute("SELECT path FROM files"):
            spans = index.spans_for(path)
            rows = index.db.execute(
                """
                SELECT passage_id, span_ids, text, source_type
                FROM passages WHERE path = ?
                """,
                (path,),
            ).fetchall()
            passages = [
                PassageRecord(
                    passage_id=row[0],
                    span_ids=row[1] or "",
                    text=row[2] or "",
                    source_type=row[3] or "",
                )
                for row in rows
            ]
            missing.extend(uncovered_lines(spans, passages))
    return missing


def _print_uncovered(found: dict[str, dict[str, list[str]]]) -> None:
    for job, roots in found.items():
        lines = [line for group in roots.values() for line in group]
        print(f"uncovered {job} {len(lines)}", flush=True)
        for line in lines:
            print(line, flush=True)


def _load_shown() -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    if not SHOWN_PATH.exists():
        return found
    for line in SHOWN_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[(row["arm"], row["id"])] = row["shown"]
    return found


def _answer_job(job, sets, answers, mx, generate_step, model, tokenizer, sampler) -> None:
    _apply(job["flags"])
    index_job = job["reuse"] or job["name"]
    only = job.get("only")
    groups = [only] if only else ["dev", "regression", "floor", "task1", "unseen", "held_rest"]
    seen: set[str] = set()
    ordered: list[SuiteCase] = []
    for name in groups:
        for case in sets[name]:
            if case.id in seen:
                continue
            seen.add(case.id)
            ordered.append(case)
    for case in ordered:
        key = (job["name"], case.id)
        if key in answers:
            continue
        os.environ["LDI_ROOT"] = str(_root_dir(index_job, case.root_name))
        shown = _rendered(case.question, mx, generate_step, model, tokenizer, sampler)
        answers[key] = shown
        metric = metric_for(case.gold, shown)
        print(f"{job['name']} {case.id} {metric}", flush=True)
        _append_shown(job["name"], case.id, shown)


def _rendered(question, mx, generate_step, model, tokenizer, sampler) -> str:
    hits = _hits(question, PRODUCT_HITS)
    gaps = _gaps()
    if not hits:
        return format_not_found(gaps)
    stale = stale_paths(hits)
    if stale:
        return format_stale(stale)
    items = build_evidence(question, hits)
    if not items:
        return format_not_found(gaps)

    def generate(reason: str) -> str:
        return _complete(
            mx,
            generate_step,
            model,
            tokenizer,
            sampler,
            question,
            hits,
            reason,
        )

    _raw, released = settle_answer(generate, hits, question)
    if released.paths:
        cited = stale_paths([Hit(path, None, "") for path in released.paths])
        if cited:
            return format_stale(cited)
    if released.kind in {"not_found", "retry"}:
        return format_not_found(gaps)
    return released.text


def _append_shown(arm: str, case_id: str, shown: str) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with SHOWN_PATH.open("a") as handle:
        handle.write(json.dumps({"arm": arm, "id": case_id, "shown": shown}) + "\n")


def _norm_compare(answers: dict[tuple[str, str], str]) -> dict:
    ids = sorted({case_id for (arm, case_id) in answers if arm in {"nfc", "nfd"}})
    differed = []
    for case_id in ids:
        left = answers.get(("nfc", case_id))
        right = answers.get(("nfd", case_id))
        if left != right:
            differed.append(case_id)
    return {"cases": len(ids), "differed": differed}


def _report(sets, jobs, answers, uncovered, commit, model, weights, norm) -> dict:
    names = [job["name"] for job in jobs if job.get("only") is None]
    rows = []
    for name in names:
        rows.append({"arm": name, **_tally(name, sets, answers)})
    flips = {
        "1 -> 1+2": _flips("1", "1+2", sets, answers),
        "1+2 -> 1+2+3": _flips("1+2", "1+2+3", sets, answers),
        "1+2+3 -> 1+2+3+4": _flips("1+2+3", "1+2+3+4", sets, answers),
        "1 -> 1+2+3+4": _flips("1", "1+2+3+4", sets, answers),
    }
    repeat = _repeat(answers)
    covered = {
        name: {root: len(lines) for root, lines in roots.items()}
        for name, roots in uncovered.items()
    }
    return {
        "commit": commit,
        "model": model,
        "weights": weights,
        "top_k": PRODUCT_HITS,
        "evidence_lines": EVIDENCE_LIMIT,
        "arms": rows,
        "flips": flips,
        "repeat": repeat,
        "normalization": norm,
        "uncovered": covered,
        "held_rest": _held_rest(sets, answers),
    }


def _tally(arm: str, sets, answers) -> dict:
    out = {}
    for name in ("dev", "regression", "floor", "task1", "unseen"):
        counts: Counter[str] = Counter()
        fabricated = 0
        missing = []
        for case in sets[name]:
            shown = answers.get((arm, case.id))
            if shown is None:
                missing.append(case.id)
                continue
            metric = metric_for(case.gold, shown)
            counts[metric] += 1
            if is_fabricated(shown):
                fabricated += 1
        total = len(sets[name])
        out[name] = {
            "correct": counts["correct"],
            "total": total,
            "wrong": counts["wrong"],
            "false_not_found": counts["false_not_found"],
            "false_accept": counts["false_accept"],
            "unscored": counts["unscored"] + len(missing),
            "unscored_ids": [case_id for case_id in missing],
            "fabrication": fabricated,
        }
    return out


def _flips(left: str, right: str, sets, answers) -> list[dict]:
    seen: set[str] = set()
    rows = []
    for name in ("dev", "regression", "floor", "task1", "unseen"):
        for case in sets[name]:
            if case.id in seen:
                continue
            seen.add(case.id)
            before = answers.get((left, case.id))
            after = answers.get((right, case.id))
            if before is None or after is None:
                continue
            old = metric_for(case.gold, before)
            new = metric_for(case.gold, after)
            if old == new:
                continue
            if new == "correct" and old != "correct":
                kind = "fixed"
            elif old == "correct" and new != "correct":
                kind = "newly broken"
            else:
                kind = "changed"
            cited = anchors_covered(case.gold.anchors, after) if case.gold.anchors else new == "correct"
            rows.append(
                {
                    "id": case.id,
                    "kind": kind,
                    "reason": f"{old} to {new}; anchor cited={cited}",
                }
            )
    return rows


def _repeat(answers) -> dict:
    ids = sorted({case_id for (arm, case_id) in answers if arm == "1+2+3+4"})
    differed = []
    for case_id in ids:
        left = answers.get(("1+2+3+4", case_id))
        right = answers.get(("1+2+3+4-repeat", case_id))
        if left != right:
            differed.append(case_id)
    return {"cases": len(ids), "differed": differed}


def _held_rest(sets, answers) -> list[dict]:
    rows = []
    for case in sets["held_rest"]:
        shown = answers.get(("1+2+3+4", case.id), "")
        rows.append(
            {
                "id": case.id,
                "scored": case.gold.scored,
                "abstain": case.gold.abstain,
                "metric": metric_for(case.gold, shown) if shown else "missing",
            }
        )
    return rows


def _print_report(report: dict) -> None:
    print(f"commit {report['commit']}")
    print(f"model {report['model']} {report['weights']}")
    print(f"top-k {report['top_k']} evidence {report['evidence_lines']}")
    header = (
        "arm",
        "dev",
        "regression",
        "floor",
        "task1",
        "unseen",
        "fabrication",
        "unscored",
    )
    print("\t".join(header))
    for row in report["arms"]:
        cells = [row["arm"]]
        unscored = 0
        fabricated = 0
        for name in ("dev", "regression", "floor", "task1", "unseen"):
            item = row[name]
            cells.append(f"{item['correct']}/{item['total']}")
            unscored += item["unscored"]
            fabricated += item["fabrication"]
        cells.append(str(fabricated))
        cells.append(str(unscored))
        print("\t".join(cells))
    for label, rows in report["flips"].items():
        print(f"flips {label} {len(rows)}")
        for item in rows:
            print(f"  {item['kind']} {item['id']} {item['reason']}")
    print(
        f"repeat {report['repeat']['cases']} differed {report['repeat']['differed'] or 'none'}"
    )
    print(
        f"nfc-nfd {report['normalization']['cases']} differed "
        f"{report['normalization']['differed'] or 'none'}"
    )
    for row in report["held_rest"]:
        print(f"held-rest {row['id']} scored={row['scored']} abstain={row['abstain']} {row['metric']}")


if __name__ == "__main__":
    main()
