"""Phone-coverage tests on fake transcript sets — no audio needed for the pure G2P path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vox_dataset import cli, phones

pytestmark = pytest.mark.skipif(not phones.have_pronouncing(),
                                reason="pronouncing (cmudict) not installed")

SYLLABARY = Path(__file__).parent / "fixtures" / "syllabary.txt"


def test_transcript_stats_counts_phones_and_syllables():
    stats = phones.transcript_stats("machine")
    # machine -> M AH0 SH IY1 N : 2 syllables, phones present.
    assert stats["syllables"] == 2
    assert stats["phone_counts"].get("SH", 0) >= 1
    assert stats["phone_counts"].get("N", 0) >= 1
    assert stats["words"] == 1
    assert stats["words_in_dict"] == 1


def test_coverage_flags_missing_phones():
    # A tiny transcript that cannot cover the whole inventory.
    counts: dict[str, int] = {}
    for word in ["the", "cat", "sat"]:
        for ph, n in phones.transcript_stats(word)["phone_counts"].items():
            counts[ph] = counts.get(ph, 0) + n
    cov = phones.coverage(counts)
    assert cov["n_present"] < cov["n_inventory"]
    assert "ZH" in cov["missing"]  # ZH is rare and won't appear
    # Weighted coverage must be a percentage in [0, 100].
    assert 0.0 <= cov["weighted_pct"] <= 100.0
    assert cov["raw_pct"] <= 100.0


def test_full_inventory_present_scores_100():
    # Force every inventory phone to be present.
    counts = {p: 5 for p in phones.ARPABET_INVENTORY}
    cov = phones.coverage(counts)
    assert cov["missing"] == []
    assert cov["raw_pct"] == 100.0
    assert cov["weighted_pct"] == pytest.approx(100.0, abs=0.1)


def test_rare_phones_flagged():
    counts = {p: 5 for p in phones.ARPABET_INVENTORY}
    counts["ZH"] = 1  # thin
    cov = phones.coverage(counts, rare_threshold=2)
    assert "ZH" in cov["rare"]


def test_missing_ranked_by_frequency():
    # Missing a common phone (N) and a rare one (ZH); the common one ranks first.
    counts = {p: 5 for p in phones.ARPABET_INVENTORY if p not in ("N", "ZH")}
    cov = phones.coverage(counts)
    assert cov["missing"][0] == "N"
    assert cov["missing"].index("N") < cov["missing"].index("ZH")


# --------------------------------------------------------------------- text-only script coverage
def test_syllabary_script_covers_the_whole_inventory():
    rep = phones.script_coverage(SYLLABARY.read_text(encoding="utf-8"))
    cov = rep["coverage"]
    assert cov["n_present"] == 39
    assert cov["missing"] == []
    assert rep["oov"] == []
    assert cov["raw_pct"] == 100.0
    assert cov["weighted_pct"] == pytest.approx(100.0, abs=0.1)


def test_script_coverage_reports_oov_words():
    rep = phones.script_coverage("calm xqzvbl calm")
    assert rep["oov"] == ["xqzvbl"]
    assert rep["words"] == 3
    assert rep["words_in_dict"] == 2
    assert rep["words_out"] == 1
    # The OOV token contributes no phones: 'calm' twice == K AA L M twice.
    assert rep["counts"]["K"] == 2


def test_script_weighted_pct_matches_rubric_metric():
    # The rubric reads phones.coverage(...)['weighted_pct']; the text-only path must be identical.
    rep = phones.script_coverage("the cat sat on the measure")
    assert rep["coverage"]["weighted_pct"] == phones.coverage(rep["counts"])["weighted_pct"]


def test_cli_coverage_script(capsys):
    rc = cli.main(["--no-color", "coverage", "--script", str(SYLLABARY)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "39/39" in out
    assert "missing:" not in out
    assert "OOV: none" in out
    assert "-> PASS" in out


def test_cli_coverage_words(capsys):
    rc = cli.main(["--no-color", "coverage", "--words",
                   SYLLABARY.read_text(encoding="utf-8")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "39/39" in out
    assert "missing:" not in out


def test_cli_coverage_script_json(capsys):
    rc = cli.main(["--no-color", "coverage", "--script", str(SYLLABARY), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["coverage"]["n_present"] == 39
    assert data["oov"] == []
    assert data["target_weighted_pct"] == 95


def test_cli_coverage_requires_a_directory_or_text():
    with pytest.raises(SystemExit) as exc:
        cli.main(["coverage"])
    assert exc.value.code != 0
