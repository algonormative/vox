"""Tests for the singing-warp op (warp.py) — the alignment + WORLD-warp that puts syllable onsets
on the grid and stretches held notes to note length.

The whisper-dependent path (``align_words`` / ``warp_to_score``) is gated behind
``whisper_available`` (faster-whisper is a heavy optional dep). The pure-DSP pieces — syllable
anchoring, the ratio clamp, and join continuity — are tested with hand-built synthetic audio so
they run everywhere.

Run (siblings on PYTHONPATH):

    cd tools/vox-tongue && uv run pytest tests/test_warp.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from vox_tongue import compile as compile_mod
from vox_tongue import warp as warp_mod

SR = 44_100


def _tone(f0: float, dur_s: float, sr: int = SR, gap_s: float = 0.0) -> np.ndarray:
    """A voiced harmonic tone (6 partials + fades), optional trailing silence — WORLD-trackable."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    x = sum((1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 7))
    fade = int(0.01 * sr)
    if fade > 1 and n > 2 * fade:
        x[:fade] *= np.linspace(0.0, 1.0, fade)
        x[-fade:] *= np.linspace(1.0, 0.0, fade)
    x = (x / (np.max(np.abs(x)) or 1.0) * 0.9)
    if gap_s > 0:
        x = np.concatenate([x, np.zeros(int(sr * gap_s))])
    return x.astype("float64")


# ---------------------------------------------------------------------------
# syllable_anchors — proportional phone-count split (no whisper needed).
# ---------------------------------------------------------------------------
def test_syllable_anchors_split_by_phone_count():
    # "machine" -> 2 syllables, phones ["M","AH0"] (2) + ["SH","IY1","N"] (3): a 2:3 split.
    score = compile_mod.compile(["machine"], ["A3"], bpm=120)
    assert len(score["syllables"]) == 2
    words = [{"word": "machine", "t0": 0.0, "t1": 1.0, "matched": True}]
    anchors = warp_mod.syllable_anchors(words, score)
    assert len(anchors) == 2
    # First syllable ~2/5 of the span, second ~3/5.
    assert anchors[0]["start"] == pytest.approx(0.0, abs=1e-6)
    assert anchors[0]["end"] == pytest.approx(0.4, abs=1e-3)
    assert anchors[1]["start"] == pytest.approx(0.4, abs=1e-3)
    assert anchors[1]["end"] == pytest.approx(1.0, abs=1e-6)


def test_syllable_anchors_multiword_grouping():
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.2, "matched": True},
        {"word": "low", "t0": 1.2, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert [a["word"] for a in anchors] == ["I", "lay", "low"]
    assert anchors[1]["start"] == pytest.approx(0.5, abs=1e-6)
    assert anchors[2]["end"] == pytest.approx(2.0, abs=1e-6)


def test_syllable_anchors_lowconf_span_clamped_to_neighbours():
    """A word whisper never matched (matched=False) whose interpolated span is a long outlier is
    capped to LOWCONF_SYL_DUR_CAP x the median per-syllable span of the confident words."""
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    # "i" and "low" are confident single-syllable words at 0.5 s each -> median per-syllable = 0.5 s.
    # "lay" was interpolated to a 1.0 s span (0.5..1.5) -> capped to 1.5 * 0.5 = 0.75 s.
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.5, "matched": False},
        {"word": "low", "t0": 1.5, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert anchors[1]["start"] == pytest.approx(0.5, abs=1e-6)
    assert anchors[1]["end"] == pytest.approx(1.25, abs=1e-3)  # clamped, not the raw 1.5


def test_syllable_anchors_lowconf_within_cap_unchanged():
    """A low-confidence word already within the neighbour cap keeps its full span (no shrink)."""
    score = compile_mod.compile(["I lay low"], ["A3", "C4", "E4"], bpm=120)
    words = [
        {"word": "i", "t0": 0.0, "t1": 0.5, "matched": True},
        {"word": "lay", "t0": 0.5, "t1": 1.1, "matched": False},  # 0.6 s < 0.75 s cap
        {"word": "low", "t0": 1.5, "t1": 2.0, "matched": True},
    ]
    anchors = warp_mod.syllable_anchors(words, score)
    assert anchors[1]["end"] == pytest.approx(1.1, abs=1e-6)


# ---------------------------------------------------------------------------
# _warp_segment — WORLD elastic time-scale + the ratio clamp.
# ---------------------------------------------------------------------------
def test_warp_segment_ratio_clamped_high():
    from vox_larynx import world

    seg = _tone(180.0, 0.20)  # 0.2 s source, ask for a 2.0 s note -> raw ratio 10, clamp to 4.
    warped, ratio = warp_mod._warp_segment(seg, SR, 2.0, world.note_to_hz("A3"), world)
    assert ratio == pytest.approx(warp_mod.TIME_RATIO_MAX)
    # Achieved dur is actual*clamped_ratio (~0.8 s), NOT the impossible 2.0 s — honest clamp.
    assert len(warped) / SR == pytest.approx(0.8, abs=0.05)


def test_warp_segment_ratio_clamped_low():
    from vox_larynx import world

    seg = _tone(180.0, 2.0)  # 2 s source squeezed toward 0.1 s -> raw ratio 0.05, clamp to 0.25.
    warped, ratio = warp_mod._warp_segment(seg, SR, 0.1, world.note_to_hz("A3"), world)
    assert ratio == pytest.approx(warp_mod.TIME_RATIO_MIN)


def test_warp_segment_imposes_pitch():
    from vox_larynx import world

    seg = _tone(180.0, 0.6)
    warped, _ratio = warp_mod._warp_segment(seg, SR, 0.6, world.note_to_hz("A3"), world)  # A3=220
    import pyworld as pw

    x = np.ascontiguousarray(warped)
    f0, t = pw.harvest(x, SR, f0_floor=80.0, f0_ceil=500.0, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, SR)
    med = float(np.median(f0[f0 > 0]))
    assert med == pytest.approx(220.0, rel=0.06)  # re-voiced to A3, not the 180 Hz source


# ---------------------------------------------------------------------------
# Vowel-sustain crossfade loop — holds past the 4x WORLD cap (vault-1gha).
# ---------------------------------------------------------------------------
def _capped_hold(source_s: float = 0.25, target_s: float = 2.0):
    """A short vowel stretched at the WORLD cap toward a long hold — the fixture this feature
    exists for. Returns ``(warped, target_s)`` where ``warped`` is ~4x the source, still short."""
    from vox_larynx import world

    seg = _tone(180.0, source_s)
    warped, ratio = warp_mod._warp_segment(seg, SR, target_s, None, world)
    assert ratio == pytest.approx(warp_mod.TIME_RATIO_MAX)  # the cap really did bind
    return np.ascontiguousarray(warped, dtype="float64"), target_s


def test_sustain_loop_fills_hold_past_world_cap():
    """The un-fixed path caps at ~4x the source and comes out SHORT of the note; the sustain loop
    closes the rest, landing on the target duration."""
    warped, target = _capped_hold(source_s=0.25, target_s=2.0)
    capped_dur = len(warped) / SR
    assert capped_dur == pytest.approx(0.25 * warp_mod.TIME_RATIO_MAX, abs=0.05)
    assert capped_dur < target - 0.5  # ~1.0 s of a 2.0 s note — the bug being fixed

    out, info = warp_mod._sustain_loop(warped, SR, target)
    assert info["sustained"] is True
    assert info["loops"] >= 1
    assert len(out) / SR == pytest.approx(target, abs=1e-3)  # target duration achieved


def test_sustain_loop_noop_when_target_already_met():
    """No deficit (or one smaller than a single seam) ⇒ no loop, and the row says so."""
    warped, _target = _capped_hold(source_s=0.25, target_s=2.0)
    dur = len(warped) / SR
    for ask in (dur * 0.5, dur, dur + 0.001):
        out, info = warp_mod._sustain_loop(warped, SR, ask)
        assert info == {"sustained": False, "loops": 0}
        assert np.array_equal(out, warped)


def test_sustain_loop_seams_are_continuous():
    """Seam continuity: no click. Two rulers — the worst inter-sample step must stay inside the
    source's own step distribution, and the 5 ms sliding RMS must not jump >6 dB anywhere across
    the sustained body (the house `test_join_continuity_no_step` gate)."""
    warped, target = _capped_hold(source_s=0.25, target_s=2.0)
    out, info = warp_mod._sustain_loop(warped, SR, target)
    assert info["sustained"] is True

    # (a) sample-to-sample step, relative to the signal's own distribution: a click is a step far
    # outside what the un-looped material already contains.
    src_step = float(np.max(np.abs(np.diff(warped))))
    out_step = float(np.max(np.abs(np.diff(out))))
    assert out_step <= src_step * 1.5

    # (b) RMS-delta over a sliding 5 ms window — a discontinuity or a phase-cancelling crossfade
    # shows up as a notch. Measured against the UN-LOOPED segment's own worst step, so the gate is
    # "the seams added no ripple the material didn't already have", not an arbitrary threshold. The
    # first/last 100 ms are excluded from both: the fixture's attack/release are legitimate moves.
    def _max_rms_step_db(x):
        win = int(0.005 * SR)
        rms = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same") + 1e-12)
        lo, hi = int(0.1 * SR), len(x) - int(0.1 * SR)
        frames = rms[lo:hi:win // 2]
        return float(np.max(20.0 * np.log10(np.maximum(frames[1:], frames[:-1]) /
                                            np.minimum(frames[1:], frames[:-1]))))

    assert _max_rms_step_db(out) <= _max_rms_step_db(warped) + 1.0
    assert _max_rms_step_db(out) < 6.0  # and inside the house join-continuity bound outright


def test_sustain_loop_keeps_onset_and_offset_intact():
    """Only INTERIOR material is looped: the reserved head (onset consonant) and tail (offset
    consonant / release) come through sample-identical to the stretched segment."""
    warped, target = _capped_hold(source_s=0.25, target_s=2.0)
    out, info = warp_mod._sustain_loop(warped, SR, target)
    assert info["sustained"] is True

    a, b = warp_mod._steady_state_region(len(warped))
    assert a > 0 and b < len(warped)  # a real edge margin was reserved
    assert np.array_equal(out[:a], warped[:a])              # onset untouched
    assert np.array_equal(out[len(out) - (len(warped) - b):], warped[b:])  # offset untouched
    # And the growth is all interior: everything added sits between the two.
    assert len(out) - len(warped) == pytest.approx(int(round(target * SR)) - len(warped))


def test_sustain_loop_skipped_when_interior_too_short():
    """A segment whose steady state can't host a loop is left alone (and reports it) rather than
    stitching a seam onto a few milliseconds of material."""
    tiny = _tone(180.0, 0.03)  # 30 ms -> ~15 ms interior, under the 2x-crossfade floor
    out, info = warp_mod._sustain_loop(tiny, SR, 1.0)
    assert info == {"sustained": False, "loops": 0}
    assert np.array_equal(out, tiny)


def test_sustain_not_triggered_in_onset_modes(monkeypatch):
    """'onsets'/'median-pitch' cap the stretch at ONSET_RATIO_CAP on purpose — the sustain loop must
    NOT engage there. The note-filling modes ('full', 'time-only') do."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=30)  # 2 s per beat: a long hold
    vocal = _clicky_word_audio()  # ~0.4 s per word -> raw ratio ~5, past the 4x cap

    def fake_align(v, sr, expected_text, **kw):
        return ([{"word": "lay", "t0": 0.0, "t1": 0.4, "matched": True},
                 {"word": "low", "t0": 0.6, "t1": 1.0, "matched": True}],
                {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                 "mismatches": [], "transcript": "lay low"})

    monkeypatch.setattr(warp_mod, "align_words", fake_align)

    for m in ("full", "time-only"):
        _out, rep = warp_mod.warp_to_score(vocal, SR, score, bpm=30, mode=m)
        assert all(r["sustained"] for r in rep["syllables"]), m
        assert all(r["sustain_loops"] >= 1 for r in rep["syllables"]), m
        # The slot is actually filled now (was ~1.6 s of a 2.0 s note under the bare cap).
        for r in rep["syllables"]:
            assert r["achieved_dur"] >= r["target_dur"]

    for m in ("onsets", "median-pitch"):
        _out, rep = warp_mod.warp_to_score(vocal, SR, score, bpm=30, mode=m)
        assert not any(r["sustained"] for r in rep["syllables"]), m
        assert all(r["sustain_loops"] == 0 for r in rep["syllables"]), m


# ---------------------------------------------------------------------------
# Join continuity — the equal-power crossfade must not leave a step at a seam.
# ---------------------------------------------------------------------------
def test_join_continuity_no_step():
    # Two abutting grid slots; assembling must not produce a >6 dB inter-sample jump at the seam.
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=120)  # beats 0 and 1, 1 s each
    # Build a synthetic "vocal": two 0.4 s tones with silence, whose word spans we pin directly.
    words = [
        {"word": "lay", "t0": 0.0, "t1": 0.45, "matched": True},
        {"word": "low", "t0": 0.5, "t1": 0.95, "matched": True},
    ]
    _ = words  # spans below are pinned via the mocked aligner so the SEAM is the shipped crossfade
    vocal = np.concatenate([_tone(200.0, 0.45), np.zeros(int(0.05 * SR)), _tone(160.0, 0.45)])

    def fake_align(v, sr, expected_text, **kw):
        return ([{"word": "lay", "t0": 0.0, "t1": 0.45, "matched": True},
                 {"word": "low", "t0": 0.5, "t1": 0.95, "matched": True}],
                {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                 "mismatches": [], "transcript": "lay low"})

    from unittest import mock
    with mock.patch.object(warp_mod, "align_words", fake_align):
        out, report = warp_mod.warp_to_score(vocal, SR, score, bpm=120)

    # The interior seam is the 2nd syllable's grid onset (beat 1 = 0.5 s). Check the smoothed
    # amplitude envelope across a window straddling it — no >6 dB step (a factor of 2.0) means the
    # equal-power crossfade held (no notch/click). The global attack/release are excluded on
    # purpose: a note starting from silence is a legitimate >6 dB rise, not a seam click.
    seam_s = report["syllables"][1]["target_t"]
    win = int(0.005 * SR)
    rms = np.sqrt(np.convolve(out.astype("float64") ** 2, np.ones(win) / win, mode="same") + 1e-12)
    lo, hi = int((seam_s - 0.03) * SR), int((seam_s + 0.03) * SR)
    frames = rms[lo:hi:win // 2]
    step_db = 20.0 * np.log10(np.maximum(frames[1:], frames[:-1]) /
                              np.minimum(frames[1:], frames[:-1]))
    assert float(np.max(step_db)) < 6.0


# ---------------------------------------------------------------------------
# align_words + full warp_to_score — whisper-gated end-to-end.
# ---------------------------------------------------------------------------
whisper_gate = pytest.mark.skipif(not warp_mod.whisper_available(),
                                  reason="faster-whisper not installed")


def _clicky_word_audio():
    """Synthesise 'lay low' as two clearly separated voiced tones — whisper won't transcribe tones
    reliably, so this fixture is only for the syllable-anchor/warp math with a MOCKED aligner."""
    return np.concatenate([_tone(200.0, 0.4, gap_s=0.2), _tone(160.0, 0.4, gap_s=0.2)])


def test_warp_to_score_places_on_grid_with_mocked_alignment(monkeypatch):
    """Grid placement correctness WITHOUT whisper: mock align_words to return known word spans and
    assert every syllable onset lands within 30 ms of its grid slot (the acceptance metric)."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=90)
    vocal = _clicky_word_audio()  # word1 ~[0,0.4], word2 ~[0.6,1.0]

    def fake_align(v, sr, expected_text, **kw):
        words = [
            {"word": "lay", "t0": 0.0, "t1": 0.4, "matched": True},
            {"word": "low", "t0": 0.6, "t1": 1.0, "matched": True},
        ]
        return words, {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                       "mismatches": [], "transcript": "lay low"}

    monkeypatch.setattr(warp_mod, "align_words", fake_align)
    out, report = warp_mod.warp_to_score(vocal, SR, score, bpm=90)
    spb = 60.0 / 90.0
    for i, row in enumerate(report["syllables"]):
        target = i * spb
        assert row["target_t"] == pytest.approx(target, abs=1e-3)
        assert abs(row["onset_err_ms"]) <= 30.0  # onset on the grid within 30 ms


@whisper_gate
def test_align_words_matches_expected_sequence():
    """faster-whisper on a real spoken clip — say-render 'lay low' and confirm the aligner returns
    two ordered word spans matched to the expected sequence."""
    from vox_tongue.render import say_available

    if not say_available():
        pytest.skip("`say` unavailable")
    from vox_tongue.render import _say_render

    clip = _say_render("lay low", "Fred", SR)
    words, report = warp_mod.align_words(clip, SR, "lay low")
    assert [w["word"] for w in words] == ["lay", "low"]
    assert words[0]["t1"] <= words[1]["t0"] + 0.05  # ordered, non-overlapping-ish
    assert report["n_expected"] == 2


# ---------------------------------------------------------------------------
# Warp MODES — the diction↔grid aggression dial (beads I2).
# ---------------------------------------------------------------------------
def test_warp_mode_unknown_raises():
    """An unknown mode fails loudly (before any whisper pass) rather than silently doing 'full'."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=120)
    with pytest.raises(ValueError):
        warp_mod.warp_to_score(_tone(180.0, 0.4), SR, score, bpm=120, mode="bogus")


def test_warp_mode_pitch_imposition_dispatch(monkeypatch):
    """Only 'full' imposes the score note per syllable; 'time-only'/'onsets'/'median-pitch' keep the
    take's native contour (pitch_imposed False) and echo their mode in the report."""
    score = compile_mod.compile(["lay low"], ["A3", "C4"], bpm=90)
    vocal = _clicky_word_audio()

    def fake_align(v, sr, expected_text, **kw):
        return ([{"word": "lay", "t0": 0.0, "t1": 0.4, "matched": True},
                 {"word": "low", "t0": 0.6, "t1": 1.0, "matched": True}],
                {"n_expected": 2, "n_recognised": 2, "n_matched": 2,
                 "mismatches": [], "transcript": "lay low"})

    monkeypatch.setattr(warp_mod, "align_words", fake_align)

    _out, rep_full = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode="full")
    assert rep_full["mode"] == "full"
    assert all(r["pitch_imposed"] for r in rep_full["syllables"])

    for m in ("time-only", "onsets", "median-pitch"):
        _out, rep = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode=m)
        assert rep["mode"] == m
        assert not any(r["pitch_imposed"] for r in rep["syllables"])
    # median-pitch attaches the global-shift record (contour-preserving register correction).
    _out, rep_mp = warp_mod.warp_to_score(vocal, SR, score, bpm=90, mode="median-pitch")
    assert set(rep_mp["global_shift"]) == {"take_median_hz", "score_median_hz", "shift_semitones"}


def test_warp_mode_onsets_ratio_capped():
    """'onsets'/'median-pitch' clamp the time-ratio to ONSET_RATIO_CAP — the interior is only nudged,
    never note-filled (the onset lands on grid by placement, not by a 4x stretch)."""
    from vox_larynx import world

    cap = warp_mod.ONSET_RATIO_CAP
    # 0.2 s source asked to fill a 2.0 s note: full mode would stretch ~10x (clamp 4x); onsets caps 1.5x.
    seg = _tone(180.0, 0.2)
    _warped, ratio = warp_mod._warp_segment(seg, SR, 2.0, None, world,
                                            ratio_min=1.0 / cap, ratio_max=cap)
    assert ratio == pytest.approx(cap)
    assert len(_warped) / SR == pytest.approx(0.2 * cap, abs=0.03)  # only the capped stretch applied
    # And the compress side clamps to 1/cap.
    seg2 = _tone(180.0, 1.0)
    _w2, ratio2 = warp_mod._warp_segment(seg2, SR, 0.1, None, world,
                                         ratio_min=1.0 / cap, ratio_max=cap)
    assert ratio2 == pytest.approx(1.0 / cap)


def test_median_shift_semitones_math():
    """The global register shift is 12*log2(target/take) — a constant interval that preserves the
    take's contour. Octaves are exact; missing/non-positive inputs mean no shift."""
    assert warp_mod._median_shift_semitones(110.0, 220.0) == pytest.approx(12.0)
    assert warp_mod._median_shift_semitones(220.0, 110.0) == pytest.approx(-12.0)
    assert warp_mod._median_shift_semitones(130.81, 130.81) == pytest.approx(0.0, abs=1e-6)
    # Guards: any missing/zero/negative operand yields 0.0 (leave the take alone).
    assert warp_mod._median_shift_semitones(None, 220.0) == 0.0
    assert warp_mod._median_shift_semitones(110.0, None) == 0.0
    assert warp_mod._median_shift_semitones(0.0, 220.0) == 0.0
    assert warp_mod._median_shift_semitones(-5.0, 220.0) == 0.0


def test_score_median_note_hz():
    """The register target is the median of the score's pitched notes (rests skipped)."""
    from vox_tongue.render import note_to_hz_any

    score = compile_mod.compile(["I lay low"], ["A2", "A2", "A4"], bpm=120)
    hzs = [note_to_hz_any(s["note"]) for s in score["syllables"]
           if note_to_hz_any(s["note"])]
    assert warp_mod._score_median_note_hz(score) == pytest.approx(float(np.median(hzs)))


@whisper_gate
def test_warp_mode_full_vs_time_only_real(monkeypatch):
    """Real end-to-end mode comparison: full mode re-voices the take to the score note; time-only
    keeps the take near its native register. (Whisper- + say-gated; measures f0 with pyworld.)"""
    from vox_tongue.render import say_available

    if not say_available():
        pytest.skip("`say` unavailable")
    import pyworld as pw
    from vox_tongue.render import _say_render

    def _median_f0(x):
        xx = np.ascontiguousarray(x, dtype="float64")
        f0, t = pw.harvest(xx, SR, f0_floor=80.0, f0_ceil=700.0, frame_period=5.0)
        f0 = pw.stonemask(xx, f0, t, SR)
        v = f0[f0 > 0]
        return float(np.median(v)) if v.size else 0.0

    clip = _say_render("lay low", "Fred", SR)
    score = compile_mod.compile(["lay low"], ["A4", "A4"], bpm=90)  # 440 Hz target, far from `say`
    target = 440.0

    out_full, rep_full = warp_mod.warp_to_score(clip, SR, score, bpm=90, mode="full")
    out_to, rep_to = warp_mod.warp_to_score(clip, SR, score, bpm=90, mode="time-only")

    # full lands near the score note; time-only stays closer to the take's native register.
    assert abs(_median_f0(out_full) - target) < abs(_median_f0(out_to) - target)
    assert all(r["pitch_imposed"] for r in rep_full["syllables"])
    assert not any(r["pitch_imposed"] for r in rep_to["syllables"])
