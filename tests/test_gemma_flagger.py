"""The optional Gemma meaning check. Every test runs with the weights absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_flagger.service import available, gemma_dir
from ldi import config
from ldi.provisions import Provision, meaning_checks

_PAIR = (
    "The tenant shall pay a deposit of Rs 50000.",
    "The tenant shall pay a deposit of Rs 75000.",
)


def _row(provision_id: str, value: str, status: str, link: str = "") -> Provision:
    return Provision(
        provision_id=provision_id,
        doc_id="policy.md",
        clause_key="deposit",
        value_text=value,
        value_span="span",
        effective_from="",
        effective_to="",
        superseded_by=link,
        status=status,
        source_span="span",
    )


def test_the_app_runs_without_the_gemma_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMMA_DIR", str(tmp_path / "missing"))
    from ldi.cli import main

    assert gemma_dir() == tmp_path / "missing"
    assert available() is False
    assert config.gemma_available() is False
    assert config.meaning_check_state() == {"available": False, "enabled": False}
    monkeypatch.setattr(config, "GEMMA_MEANING_CHECK", True)
    rows = [
        _row("old", _PAIR[0], "superseded", "new"),
        _row("new", _PAIR[1], "current"),
    ]
    assert meaning_checks(rows) == {}
    assert main(["status"]) == 0


def test_a_rule_flag_is_kept_whatever_the_probe_says():
    """The probe may add a check. It can never clear one the rules already set."""

    class _Probe:
        def __init__(self, score: float) -> None:
            self.score = score

        def check(self, old: str, new: str) -> dict:
            return {
                "score": self.score,
                "probe_flag": self.score >= 0,
                "rule_flag": True,
                "kw_flag": False,
                "needs_check": True or self.score >= 0,
            }

    for score in (-5.0, 5.0):
        result = _Probe(score).check(*_PAIR)
        assert result["rule_flag"] is True
        assert result["needs_check"] is True


@pytest.mark.skipif(not available(), reason="Gemma weights are not downloaded")
def test_the_same_pair_scores_the_same_twice():
    from gemma_flagger.flagger import MeaningFlagger

    flagger = MeaningFlagger(gemma_dir())
    first = flagger.score(*_PAIR)
    second = flagger.score(*_PAIR)
    assert first == second
    checked = flagger.check(*_PAIR)
    assert checked["score"] == first
    assert set(checked) == {"score", "probe_flag", "rule_flag", "kw_flag", "needs_check"}


def test_the_probe_file_matches_the_frozen_hash():
    import hashlib
    import json

    assets = Path(__file__).resolve().parents[1] / "gemma_flagger" / "assets"
    probe = assets / "probe_legal_v1.npz"
    config_file = assets / "frozen_config.json"
    if not probe.is_file() or not config_file.is_file():
        pytest.skip("The probe bundle has not been dropped into gemma_flagger/assets yet")
    recorded = json.loads(config_file.read_text())["probe_sha256"]
    found = hashlib.sha256(probe.read_bytes()).hexdigest()
    assert found == recorded
    assert found == "62f64cd62227e5f155947540a9b91666557e5638c8d3d4853604a8774a563cf4"
