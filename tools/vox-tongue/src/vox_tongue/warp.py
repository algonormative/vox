"""warp — the SINGING-WARP op: warp markers for a synthesised voice.

The diagnosis: a neural TTS voice decides its own timing — *spoken*
prosody. Music needs syllable onsets ON grid positions and held notes stretched to note lengths.
Every DSP piece already exists: ``vox_larynx.world.render`` does formant-preserving F0 imposition
AND elastic time-scale in one call. The missing piece was **alignment** — knowing WHERE each
syllable actually sits in the rendered clip so it can be moved to where the score says it belongs.

This module supplies the three alignment stages plus the warp assembler:

  1. :func:`align_words`      — faster-whisper word timestamps, greedily matched to the expected
                               words (tolerates whisper merges/splits/misses; documents mismatches).
  2. :func:`syllable_anchors` — split each word's ``[t0, t1]`` proportionally by its syllables'
                               phone counts (taken from the TONGUE score) -> per-syllable actual span.
  3. :func:`warp_to_score`    — the warp map: per syllable, WORLD-render the actual span to the
                               score's target start/dur (elastic time-scale) at the score's note Hz,
                               equal-power-crossfade the segments onto the grid timeline.

The whole path is deterministic given the source clip + the whisper model. faster-whisper is a heavy
optional dep (imported lazily, gated by :func:`whisper_available`); pyworld comes in via
``vox_larynx`` on PYTHONPATH.
"""

from __future__ import annotations

import re

import numpy as np

from .concat import _equal_power_fades, _resample
from .render import note_to_hz_any
from .schema import load_score

# WORLD elastic time-scale bounds — a syllable may be stretched 4x (a short spoken vowel filling a
# long held note) or compressed to 1/4 (a drawled word squeezed onto a 16th); past that the WORLD
# artifacts dominate, so we clamp and record the clamp honestly in the warp report.
TIME_RATIO_MIN = 0.25
TIME_RATIO_MAX = 4.0

# ---------------------------------------------------------------------------
# Warp modes — the aggression DIAL (beads I2). Round-7 measured the diction tax of the full
# per-syllable time+pitch imposition (StyleTTS2 clone 0.957 native → 0.617 warped). These modes
# trade grid-lock for diction so a role can pick where on the curve it sits:
#
#   'full'         — existing behaviour: per-syllable elastic time to the note length AND constant
#                    F0 imposition to the score note. Maximum grid+pitch lock, maximum diction tax.
#   'time-only'    — per-syllable elastic time to the note length, NO pitch imposition (native
#                    contour survives). Full grid-lock, keeps the take's own melody/inflection.
#   'onsets'       — pin syllable ONSETS to the grid, small local stretch only (ratio capped to
#                    ONSET_RATIO_CAP), interiors untouched, no pitch. Grid feel, minimal diction tax.
#   'median-pitch' — 'onsets' timing + ONE global pitch shift moving the take's median f0 to the
#                    score's median note (contour preserved, register corrected). Onset feel + a
#                    register fix without the per-syllable pitch cost.
WARP_MODES = ("full", "time-only", "onsets", "median-pitch")

# 'onsets'/'median-pitch' cap the per-syllable time-stretch to a small local nudge — the onset lands
# on the grid via segment placement, so the interior only needs a gentle stretch, never the 4x/0.25x
# a note-filling warp demands. 1.5x keeps WORLD's formant/timbre artifacts inaudible.
ONSET_RATIO_CAP = 1.5

# ---------------------------------------------------------------------------
# Vowel-sustain crossfade loop — holds PAST the 4x WORLD cap (beads vault-1gha).
# ---------------------------------------------------------------------------
# The longest melisma holds ask for more than TIME_RATIO_MAX on a short spoken vowel: WORLD
# stretches to the 4x cap and the syllable STILL comes out short of its note. Rather than push
# WORLD past the point its artifacts dominate, we stretch at the cap and then LOOP the vowel's
# steady-state interior with equal-power crossfades until the slot is full — the classic sampler
# sustain loop. Onset and offset consonants stay untouched: every looped copy is drawn from the
# interior only, so the segment's head and tail are the stretched material, sample for sample.
#
# Only the NOTE-FILLING clamp sustains ('full' / 'time-only'). 'onsets'/'median-pitch' deliberately
# cap the stretch at ONSET_RATIO_CAP because the onset lands on the grid by placement — filling the
# interior there would defeat the whole point of the mode.
SUSTAIN_XFADE_MS = 15.0   # equal-power crossfade at each loop seam
SUSTAIN_EDGE_FRAC = 0.25  # fraction of the stretched segment reserved at EACH edge (consonants)

# A word whisper never recognised (``matched=False``) has its span *interpolated* from neighbours
# (:func:`_fill_spans`). That interpolation can hand back a wildly long span (a ~289 ms outlier when a
# gap is spread over one word), which then drives an extreme WORLD warp ratio + audible artifacts. So
# when confidence is low we cap the word's per-syllable source span to this multiple of the median
# per-syllable span of the *confident* (matched) words — a neighbour-relative sanity clamp.
LOWCONF_SYL_DUR_CAP = 1.5

# Equal-power crossfade at every segment join (ms). Small enough not to move a grid onset past the
# 30 ms acceptance window, long enough to kill the seam click (the join-continuity test gate).
_XFADE_MS = 6.0

# WORLD needs enough samples to analyse; below this we skip pitch imposition and linear-resample.
_MIN_WORLD_SAMPLES = 1024

_WHISPER_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Text normalisation + whisper availability.
# ---------------------------------------------------------------------------
def _norm(word: str) -> str:
    """Lowercase, strip everything but letters/digits/apostrophe — the token match key."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _get_whisper(model_size: str):
    if model_size not in _WHISPER_CACHE:
        from faster_whisper import WhisperModel

        # int8 on CPU — fast + deterministic enough for word timings; the model is a skeleton ruler,
        # not the render, so quantisation error here is immaterial.
        _WHISPER_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _WHISPER_CACHE[model_size]


# ---------------------------------------------------------------------------
# 1. align_words — faster-whisper word timestamps, greedily matched to expected words.
# ---------------------------------------------------------------------------
def _transcribe_words(vocal: np.ndarray, sr: int, model_size: str):
    """Run faster-whisper and return a flat list of ``{word, start, end}`` (seconds)."""
    model = _get_whisper(model_size)
    # faster-whisper wants float32 mono at 16 kHz.
    x = np.ascontiguousarray(vocal, dtype="float64")
    x16 = _resample(x, int(sr), 16000).astype("float32")
    segments, _info = model.transcribe(x16, language="en", word_timestamps=True,
                                       beam_size=1, vad_filter=False)
    rec = []
    for seg in segments:
        for w in (seg.words or []):
            token = _norm(w.word)
            if not token:
                continue
            rec.append({"word": token, "raw": w.word.strip(),
                        "start": float(w.start), "end": float(w.end)})
    return rec


def _fill_spans(spans, total_dur):
    """Interpolate any ``None`` span (an expected word whisper never emitted) across the gap
    between its known neighbours, so every expected word gets a usable ``[t0, t1]``."""
    n = len(spans)
    # Anchor endpoints so leading/trailing Nones have something to interpolate against.
    known = [(i, s) for i, s in enumerate(spans) if s is not None]
    if not known:
        # Whisper produced nothing usable — spread the whole clip evenly (documented as mismatch).
        step = total_dur / max(n, 1)
        return [(i * step, (i + 1) * step, False) for i in range(n)]
    out = list(spans)
    # Leading Nones -> [0, first.t0].
    first_i, first_s = known[0]
    for i in range(first_i):
        frac0 = i / (first_i + 1)
        frac1 = (i + 1) / (first_i + 1)
        out[i] = (first_s[0] * frac0, first_s[0] * frac1, False)
    # Trailing Nones -> [last.t1, total].
    last_i, last_s = known[-1]
    for k, i in enumerate(range(last_i + 1, n)):
        span_len = (total_dur - last_s[1]) / (n - last_i)
        out[i] = (last_s[1] + k * span_len, last_s[1] + (k + 1) * span_len, False)
    # Interior Nones between two known anchors -> even split of the gap.
    ki = 0
    while ki < len(known) - 1:
        a_i, a_s = known[ki]
        b_i, b_s = known[ki + 1]
        gap = b_i - a_i
        if gap > 1:
            lo, hi = a_s[1], b_s[0]
            step = (hi - lo) / gap
            for j in range(1, gap):
                out[a_i + j] = (lo + (j - 1) * step, lo + j * step, False)
        ki += 1
    return [(s[0], s[1], (s[2] if len(s) > 2 else True)) for s in out]


def align_words(vocal, sr, expected_text, *, model_size="base.en"):
    """Align faster-whisper word timestamps to the ``expected_text`` word sequence.

    Returns ``(words, report)`` where ``words`` is one entry PER EXPECTED WORD, in order::

        {"word": str, "t0": float, "t1": float, "matched": bool}

    and ``report`` documents the alignment: ``{"n_expected", "n_recognised", "n_matched",
    "mismatches": [{"expected", "recognised", "op"}], "transcript"}``.

    Matching is a ``difflib`` opcode pass over the normalised token lists. ``equal`` blocks map
    one-to-one; ``replace`` blocks distribute the recognised time-span proportionally across the
    expected words (whisper merged/split); ``delete`` blocks (expected words whisper missed) are
    interpolated from neighbours by :func:`_fill_spans`. Every non-``equal`` op is a documented
    mismatch — the caller can see exactly where the ruler was uncertain.
    """
    from difflib import SequenceMatcher

    vocal = np.ascontiguousarray(vocal, dtype="float64")
    total_dur = len(vocal) / float(sr)
    expected = [_norm(w) for w in str(expected_text).split() if _norm(w)]
    rec = _transcribe_words(vocal, sr, model_size)
    rec_tokens = [r["word"] for r in rec]

    spans = [None] * len(expected)  # per expected word: (t0, t1, matched) or None
    mismatches = []

    sm = SequenceMatcher(a=expected, b=rec_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                r = rec[j1 + k]
                spans[i1 + k] = (r["start"], r["end"], True)
        elif tag == "replace":
            # An expected block mapped to a recognised block of possibly different length: share the
            # recognised time-span proportionally across the expected words (merge OR split case).
            n_exp = i2 - i1
            if j2 > j1:
                lo, hi = rec[j1]["start"], rec[j2 - 1]["end"]
            else:  # pragma: no cover — replace with empty b-block is degenerate
                lo = hi = total_dur * (i1 / max(len(expected), 1))
            step = (hi - lo) / max(n_exp, 1)
            for k in range(n_exp):
                spans[i1 + k] = (lo + k * step, lo + (k + 1) * step, False)
                mismatches.append({"expected": expected[i1 + k],
                                   "recognised": " ".join(rec_tokens[j1:j2]) or None, "op": "replace"})
        elif tag == "delete":
            # Expected words with no recognised counterpart — leave None, _fill_spans interpolates.
            for k in range(i2 - i1):
                mismatches.append({"expected": expected[i1 + k], "recognised": None, "op": "delete"})
        elif tag == "insert":
            # Recognised words with no expected counterpart (whisper hallucinated/split) — dropped;
            # their time is absorbed by the surrounding equal/replace spans. Recorded for audit.
            mismatches.append({"expected": None,
                               "recognised": " ".join(rec_tokens[j1:j2]) or None, "op": "insert"})

    filled = _fill_spans(spans, total_dur)
    words = [{"word": expected[i], "t0": float(filled[i][0]), "t1": float(filled[i][1]),
              "matched": bool(filled[i][2])} for i in range(len(expected))]
    report = {
        "n_expected": len(expected),
        "n_recognised": len(rec),
        "n_matched": sum(1 for w in words if w["matched"]),
        "mismatches": mismatches,
        "transcript": " ".join(r["raw"] for r in rec),
    }
    return words, report


# ---------------------------------------------------------------------------
# 2. syllable_anchors — split each word span by its syllables' phone counts.
# ---------------------------------------------------------------------------
def _word_groups(score: dict):
    """Group a score's syllables into consecutive same-word runs.

    Returns a list of ``(word, [syllable_dicts])`` in score order. Adjacent identical word strings
    are treated as one word (rare in lyrics; documented) — the grouping only exists to line syllables
    up against the aligned per-word spans.
    """
    groups = []
    for syl in score["syllables"]:
        w = syl.get("word", "")
        if groups and groups[-1][0] == w:
            groups[-1][1].append(syl)
        else:
            groups.append((w, [syl]))
    return groups


def _syl_weight(syl: dict) -> float:
    """Proportional weight of a syllable within its word — its phone count (>=1). A held vowel with
    a long coda gets more of the word's duration than a bare CV syllable."""
    phones = syl.get("phones")
    return float(len(phones)) if phones else 1.0


def _confident_per_syl(words, groups, n):
    """Median per-syllable span (seconds) over the matched words, or ``None`` if there are none.

    The neighbour reference for the low-confidence clamp in :func:`syllable_anchors`: only words
    whisper actually recognised (``matched=True``) contribute, so an interpolated outlier can't
    poison its own bound."""
    per_syl = []
    for gi in range(n):
        if words[gi].get("matched", True):
            dur = words[gi]["t1"] - words[gi]["t0"]
            if dur > 0:
                per_syl.append(dur / max(len(groups[gi][1]), 1))
    return float(np.median(per_syl)) if per_syl else None


def syllable_anchors(words, score):
    """Split each aligned word's ``[t0, t1]`` proportionally by its syllables' phone counts.

    ``words`` is the per-word list from :func:`align_words`; ``score`` is a validated TONGUE score.
    Returns one ``{"start": float, "end": float, "word": str, "text": str}`` per score syllable, in
    score order — the *actual* per-syllable span in the source clip. When the word-group count and
    the aligned-word count disagree (a heuristic split difference), the shorter length governs and
    the tail is spanned from the last known word (documented, never silently mis-indexed).
    """
    score = load_score(score)
    groups = _word_groups(score)
    anchors = []
    n = min(len(groups), len(words))
    # Neighbour reference: median per-syllable span over the CONFIDENT (matched) words. A low-
    # confidence word (matched=False, span interpolated) is capped to LOWCONF_SYL_DUR_CAP x this,
    # so a bad interpolation can't feed an extreme warp ratio. No matched words -> no clamp.
    ref_per_syl = _confident_per_syl(words, groups, n)
    for gi in range(n):
        _word, syls = groups[gi]
        t0, t1 = words[gi]["t0"], words[gi]["t1"]
        total = max(t1 - t0, 1e-6)
        if ref_per_syl is not None and not words[gi].get("matched", True):
            total = min(total, LOWCONF_SYL_DUR_CAP * ref_per_syl * max(len(syls), 1))
        weights = [_syl_weight(s) for s in syls]
        wsum = sum(weights) or 1.0
        cursor = t0
        for s, wt in zip(syls, weights):
            dur = total * (wt / wsum)
            anchors.append({"start": float(cursor), "end": float(cursor + dur),
                            "word": s.get("word", ""), "text": s.get("text", "")})
            cursor += dur
    # Any trailing groups beyond the aligned words (rare) hang off the last word's end point.
    if len(groups) > n:
        tail_t = words[-1]["t1"] if words else 0.0
        for gi in range(n, len(groups)):
            _word, syls = groups[gi]
            for s in syls:
                anchors.append({"start": float(tail_t), "end": float(tail_t),
                                "word": s.get("word", ""), "text": s.get("text", "")})
    return anchors


# ---------------------------------------------------------------------------
# 3. warp_to_score — the warp map: WORLD-render each syllable onto its grid slot.
# ---------------------------------------------------------------------------
def _extract(vocal: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(int(round(start_s * sr)), 0)
    b = min(int(round(end_s * sr)), len(vocal))
    if b <= a:
        return np.zeros(0, dtype="float64")
    return np.ascontiguousarray(vocal[a:b], dtype="float64")


def _warp_segment(seg: np.ndarray, sr: int, target_dur_s: float, hz, world,
                  *, ratio_min: float = TIME_RATIO_MIN, ratio_max: float = TIME_RATIO_MAX):
    """Time-scale ``seg`` to ``target_dur_s`` (WORLD elastic, ratio-clamped to
    ``[ratio_min, ratio_max]``) and, if ``hz`` given, impose that constant F0 (formants kept).
    Returns ``(warped float64, ratio_applied)``.

    The default clamp is the note-filling ``[TIME_RATIO_MIN, TIME_RATIO_MAX]`` (0.25–4x); the
    onset-oriented modes pass a tighter ``[1/ONSET_RATIO_CAP, ONSET_RATIO_CAP]`` so the interior is
    only gently nudged (the onset is placed on the grid by the assembler, not by stretching).

    Short segments (< ~23 ms at 44.1k) can't be WORLD-analysed reliably; those fall back to a plain
    linear resample to the target length with no pitch imposition."""
    actual_dur = len(seg) / float(sr)
    if actual_dur <= 0:
        return np.zeros(max(int(round(target_dur_s * sr)), 1), dtype="float64"), 1.0
    raw_ratio = target_dur_s / actual_dur
    ratio = float(np.clip(raw_ratio, ratio_min, ratio_max))

    if len(seg) >= _MIN_WORLD_SAMPLES:
        try:
            y, _params = world.render(seg, sr, to_hz=hz, time_ratio=ratio)
            return np.ascontiguousarray(y, dtype="float64"), ratio
        except Exception:  # noqa: BLE001 — WORLD can choke on near-silent/degenerate frames.
            pass
    # Fallback: linear-resample to the clamped target length, no pitch imposition.
    target_n = max(int(round(len(seg) * ratio)), 1)
    idx = np.linspace(0, len(seg) - 1, target_n)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(seg) - 1)
    fr = idx - lo
    y = seg[lo] * (1 - fr) + seg[hi] * fr
    return np.ascontiguousarray(y, dtype="float64"), ratio


def _steady_state_region(n: int) -> tuple[int, int]:
    """The steady-state interior window ``[a, b)`` of an ``n``-sample stretched syllable.

    **Selection rule: a CENTRED window with :data:`SUSTAIN_EDGE_FRAC` of the segment reserved at
    each edge** (so the middle 50% by default). Why a fixed fraction rather than a measured vowel
    detector: the segment handed here is one syllable that has already been trimmed to its voicing
    (:func:`_trim_lead`) and elastically stretched, so its onset consonant / attack sits in the
    leading fraction and its offset consonant / release in the trailing fraction — both scale WITH
    the stretch, which a fixed millisecond margin would not. It is deterministic and needs no pitch
    or formant analysis. Everything outside ``[a, b)`` is never copied, which is exactly what keeps
    the onset/offset intact.
    """
    edge = int(round(n * SUSTAIN_EDGE_FRAC))
    return edge, n - edge


def _xfade_join(acc: np.ndarray, seg: np.ndarray, xf: int) -> np.ndarray:
    """Append ``seg`` to ``acc`` over an ``xf``-sample equal-power (sin/cos) crossfade.

    ``sin^2 + cos^2 = 1`` ⇒ constant power across the seam, so a loop point between two stretches of
    the same vowel joins without a step. The result is ``len(acc) + len(seg) - xf`` samples long —
    the join's *gain* is ``len(seg) - xf``, which is how :func:`_sustain_loop` hits an exact target.
    """
    xf = int(min(xf, len(acc), len(seg)))
    if xf <= 1:
        return np.concatenate([acc, seg])
    fin = np.sin(np.linspace(0.0, np.pi / 2.0, xf))
    fout = np.cos(np.linspace(0.0, np.pi / 2.0, xf))
    mixed = acc[-xf:] * fout + seg[:xf] * fin
    return np.concatenate([acc[:-xf], mixed, seg[xf:]])


def _sustain_loop(warped: np.ndarray, sr: int, target_dur_s: float):
    """Sustain ``warped`` to ``target_dur_s`` by crossfade-looping its vowel steady state.

    The gap this closes is the one :func:`_warp_segment` reports honestly and cannot fill: when a
    hold demands more than :data:`TIME_RATIO_MAX`, WORLD stretches to the cap and the syllable is
    still short of its note. Here the remaining deficit is made up by re-playing the steady-state
    interior (:func:`_steady_state_region`), never the edges.

    Construction — every looped copy is a SUFFIX of the interior, ending at the interior's end
    ``b``: a copy planned to add ``g`` samples is ``warped[b - g - x : b]``, joined over an
    ``x``-sample equal-power crossfade, so it adds exactly ``len(seg) - x = g`` and the total lands
    on ``target_dur_s`` to the sample. Because each copy ends at ``b``, its tail is the original
    material that ran into the offset consonant, and the LAST copy therefore joins the untouched
    tail as an exact sample continuation — no seam there at all.

    On the seams themselves: the accumulator invariantly ends at interior index ``b``, so a join
    blends ``warped[b - x : b]`` against ``warped[b - g - x : b - g]`` — a content lag of exactly
    ``g``, fixed by the copy's planned gain and by nothing else. There is no loop-point phase
    search here and adding one would be inert: sliding the copy's start ``s`` while holding the
    gain exact necessarily slides the crossfade length with it, leaving the blended lag at ``g``.
    What keeps the seam click-free is simply the equal-power crossfade over
    :data:`SUSTAIN_XFADE_MS` — measured on the synthetic-vowel fixture, the looped output's worst
    sliding-RMS step is identical to the un-looped segment's (3.47 dB either way: the seams add no
    ripple at all), while butt-splicing the same copies roughly doubles the worst inter-sample step
    on that fixture and multiplies it by up to ~25x on shorter sources.

    Returns ``(out float64, info)`` with ``info = {"sustained": bool, "loops": int}``. No-ops (and
    reports ``sustained=False``) when the segment already reaches the target, when the deficit is
    smaller than one crossfade, or when the interior is too short to loop — the honest-clamp ethos:
    a hold that could not be sustained says so in its report row rather than pretending.
    """
    warped = np.ascontiguousarray(warped, dtype="float64")
    n = len(warped)
    n_target = int(round(target_dur_s * sr))
    info = {"sustained": False, "loops": 0}
    if n == 0 or n_target <= n:
        return warped, info

    x = max(int(SUSTAIN_XFADE_MS / 1000.0 * sr), 1)
    a, b = _steady_state_region(n)
    # Max samples one copy can add: the interior, less the crossfade the join eats.
    g_max = (b - a) - x
    deficit = n_target - n
    if g_max < 2 * x or deficit < x:
        return warped, info  # interior too short to loop, or gap not worth a seam

    # Spread the deficit over as few copies as possible, near-equally. Every gain then lands in
    # [x, g_max]: with one copy the gain is the deficit, which the guard put at >= x; with k >= 2
    # copies, k = ceil(deficit/g_max) implies deficit > (k-1)*g_max >= k*g_max/2, so each gain is
    # floor(deficit/k) >= g_max/2 >= x by the g_max >= 2x guard. Gain >= x in turn keeps
    # s = b - g - x >= a, and leaves the accumulator ending in g >= x samples of pristine interior.
    n_extra = int(np.ceil(deficit / g_max))
    base, rem = divmod(deficit, n_extra)
    gains = [base + (1 if i < rem else 0) for i in range(n_extra)]

    out = warped[:b].copy()          # onset + first pass through the interior, untouched
    tail = warped[b:]                # offset consonant / release, untouched
    for g in gains:
        seg = warped[b - g - x:b]    # interior suffix; adds exactly len(seg) - x == g
        out = _xfade_join(out, seg, x)
    out = np.concatenate([out, tail])
    return np.ascontiguousarray(out, dtype="float64"), {"sustained": True, "loops": len(gains)}


def _trim_lead(seg: np.ndarray, sr: int, thresh_frac: float = 0.08, preroll_ms: float = 8.0):
    """Trim leading near-silence from an extracted syllable so its VOICING lands on the grid, not
    the dead air a whisper word-start bracket often includes. Keeps a short pre-roll so a consonant
    onset isn't clipped. Returns ``(trimmed, trimmed_seconds)`` — the seconds removed feed the report
    so the reported ``actual_t_after_warp`` still reflects the true onset placement."""
    if seg.size == 0:
        return seg, 0.0
    env = np.abs(seg)
    peak = float(env.max()) or 1.0
    onset = int(np.argmax(env >= thresh_frac * peak))
    preroll = int(preroll_ms / 1000.0 * sr)
    cut = max(onset - preroll, 0)
    if cut <= 0:
        return seg, 0.0
    return np.ascontiguousarray(seg[cut:], dtype="float64"), cut / float(sr)


def _onset_offset(seg: np.ndarray, sr: int, thresh_frac: float = 0.15) -> float:
    """Seconds from the start of ``seg`` to its first supra-threshold sample (a crude energy onset)
    — used to MEASURE where a warped syllable actually speaks vs. where it was placed."""
    if seg.size == 0:
        return 0.0
    env = np.abs(seg)
    peak = float(env.max()) or 1.0
    idx = np.argmax(env >= thresh_frac * peak)
    return float(idx) / float(sr)


# ---------------------------------------------------------------------------
# median-pitch helpers — the ONE global register correction (mode='median-pitch').
# ---------------------------------------------------------------------------
def _score_median_note_hz(score: dict):
    """Median of the score's syllable notes, in Hz — the register the take should be moved TO.

    Only notes that resolve to a real Hz value contribute (``None`` notes are rests / no-pitch
    slots and are skipped). Returns ``None`` when the score has no pitched syllable."""
    hzs = []
    for syl in score["syllables"]:
        hz = note_to_hz_any(syl.get("note"))
        if hz is not None and hz > 0:
            hzs.append(float(hz))
    return float(np.median(hzs)) if hzs else None


def _median_shift_semitones(take_hz, target_hz) -> float:
    """Semitones to move ``take_hz`` (the take's measured median f0) onto ``target_hz`` (the score's
    median note): ``12*log2(target/take)``. A CONSTANT shift preserves the take's melodic contour —
    every frame moves by the same interval — while correcting the register. Returns ``0.0`` when
    either value is missing/non-positive (no shift, contour and register both left alone)."""
    if not take_hz or not target_hz or take_hz <= 0 or target_hz <= 0:
        return 0.0
    return float(12.0 * np.log2(float(target_hz) / float(take_hz)))


def _measure_take_median_hz(vocal, sr, target_hint=None):
    """The take's median f0 (Hz) via vox-core's bass-safe guarded ruler when importable,
    else a low-floor pyworld median fallback. ``target_hint`` steers the guarded ruler's bass/high
    handling (sub-90 Hz carriers are forced to the low-floor pass)."""
    try:
        from vox_core import measure_f0_guarded

        return measure_f0_guarded(np.ascontiguousarray(vocal, dtype="float64"), int(sr),
                                  target_hint=target_hint)["f0_hz"]
    except Exception:  # noqa: BLE001 — vox-core absent (unit tests); fall back to low-floor pw
        import pyworld as pw

        x = np.ascontiguousarray(vocal, dtype="float64")
        ceil = 1000.0 if (target_hint is not None and float(target_hint) >= 130.0) else 200.0
        f0, t = pw.harvest(x, int(sr), f0_floor=30.0, f0_ceil=ceil, frame_period=5.0)
        f0 = pw.stonemask(x, f0, t, int(sr))
        voiced = f0[f0 > 0]
        return float(np.median(voiced)) if voiced.size else None


def warp_pieces(vocal, sr, score, bpm=None, mode="full"):
    """The per-syllable warp stage (one whisper pass): align, anchor, and WORLD-render each syllable
    onto its grid slot — WITHOUT the final overlap-add. Returns ``(pieces, align_report)`` where each
    piece is::

        {"start_sample": int,       # grid placement
         "faded": float64,          # warped + equal-power edge fades (assembly-ready)
         "warped": float64,         # warped, pre-fade
         "source": float64,         # the trimmed source segment fed to WORLD (for ESTOI pairing)
         "row": {...}}              # the report row (onset err, ratio, sustain, note, ...)

    A hold that outruns :data:`TIME_RATIO_MAX` is finished by :func:`_sustain_loop` — the vowel's
    steady state is crossfade-looped to fill the slot — under the note-filling modes only. The row's
    ``sustained`` / ``sustain_loops`` say whether that happened and how many loops it took, and
    ``achieved_dur`` reflects the sustained length.

    :func:`warp_to_score` assembles ``faded`` onto the grid; a measurement pass can pair ``source``
    with ``warped`` (same content, same length) for a frame-aligned per-syllable ESTOI — the only
    ESTOI that means anything on a signal whose whole purpose is to move onsets off their source
    positions.
    """
    from vox_larynx import world  # sibling tool on PYTHONPATH

    if mode not in WARP_MODES:
        raise ValueError(f"unknown warp mode {mode!r}; expected one of {WARP_MODES}")

    score = load_score(score)
    bpm = float(bpm if bpm is not None else score["meta"]["bpm"])
    spb = 60.0 / bpm
    vocal = np.ascontiguousarray(vocal, dtype="float64")

    # Mode → per-syllable warp behaviour. 'full' imposes the score note per syllable; every other
    # mode keeps the take's native contour (no per-syllable F0). 'onsets'/'median-pitch' also tighten
    # the time-ratio clamp so the interior is only nudged (the onset lands on grid by placement).
    impose_pitch = mode == "full"
    if mode in ("onsets", "median-pitch"):
        ratio_min, ratio_max = 1.0 / ONSET_RATIO_CAP, ONSET_RATIO_CAP
    else:
        ratio_min, ratio_max = TIME_RATIO_MIN, TIME_RATIO_MAX
    # Vowel-sustain applies ONLY under the note-filling clamp: a hold that outruns TIME_RATIO_MAX is
    # supposed to fill its slot. The onset-oriented modes cap at ONSET_RATIO_CAP on purpose (the
    # onset lands on grid by placement, the interior is only nudged) — sustaining there would
    # re-introduce exactly the note-filling they exist to avoid.
    sustain_allowed = ratio_max >= TIME_RATIO_MAX

    groups = _word_groups(score)
    expected_text = " ".join(w for w, _ in groups)
    words, align_report = align_words(vocal, sr, expected_text)
    anchors = syllable_anchors(words, score)

    xfade = max(int(_XFADE_MS / 1000.0 * sr), 1)
    xfade_s = xfade / float(sr)
    pieces = []
    for syl, anch in zip(score["syllables"], anchors):
        target_start = syl["start_beat"] * spb
        target_dur = syl["dur_beats"] * spb
        note_hz = note_to_hz_any(syl["note"])  # reported regardless; imposed only in 'full' mode
        seg = _extract(vocal, sr, anch["start"], anch["end"])
        seg, _trimmed = _trim_lead(seg, sr)  # voicing on the grid, not the whisper bracket's dead air
        # Render a hair longer than the slot (concat's trick): the segment rings one crossfade tail
        # PAST its grid boundary so its equal-power fade-out overlaps the next syllable's fade-in —
        # a constant-power crossfade at the seam while the onset stays EXACTLY on the grid.
        warped, ratio = _warp_segment(seg, sr, target_dur + xfade_s,
                                      note_hz if impose_pitch else None, world,
                                      ratio_min=ratio_min, ratio_max=ratio_max)
        # Past the 4x cap the stretch alone leaves the syllable short of its note; loop the vowel's
        # steady state to close the rest (onset/offset untouched). Reported per row, honestly.
        if sustain_allowed:
            warped, sustain = _sustain_loop(warped, sr, target_dur + xfade_s)
        else:
            sustain = {"sustained": False, "loops": 0}
        faded = _equal_power_fades(warped, xfade)
        onset_off = _onset_offset(warped, sr)
        pieces.append({
            "start_sample": int(round(target_start * sr)),
            "faded": faded,
            "warped": warped,
            "source": seg,
            "row": {
                "syllable": syl["text"],
                "word": syl["word"],
                "note": syl["note"],
                "hz": (round(float(note_hz), 2) if note_hz is not None else None),
                "pitch_imposed": bool(impose_pitch and note_hz is not None),
                "target_t": round(float(target_start), 4),
                "target_dur": round(float(target_dur), 4),
                "actual_t_after_warp": round(float(target_start + onset_off), 4),
                "onset_err_ms": round(float(onset_off) * 1000.0, 1),
                "ratio": round(float(ratio), 3),
                "sustained": bool(sustain["sustained"]),
                "sustain_loops": int(sustain["loops"]),
                "achieved_dur": round(len(warped) / float(sr), 4),
            },
        })
    return pieces, align_report


def assemble_pieces(pieces, sr):
    """Overlap-add the ``faded`` warped segments onto the grid at their ``start_sample`` and
    peak-normalise to 0.9. The equal-power edge fades sum to a constant-power crossfade wherever two
    segments abut. Returns ``float32`` mono."""
    total = max((p["start_sample"] + len(p["faded"]) for p in pieces), default=0) + 1
    out = np.zeros(total, dtype="float64")
    for p in pieces:
        s, g = p["start_sample"], p["faded"]
        end = min(s + len(g), total)
        if end > s:
            out[s:end] += g[: end - s]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak * 0.9
    return out.astype("float32")


def warp_to_score(vocal, sr, score, bpm=None, mode="full"):
    """Warp ``vocal`` so each syllable's onset lands on its TONGUE-score grid slot, at an aggression
    set by ``mode`` (the diction↔grid dial — see :data:`WARP_MODES`).

    Runs :func:`align_words` (expected text derived FROM the score's word sequence) then
    :func:`syllable_anchors`, then per syllable (:func:`warp_pieces`): extract the actual span,
    WORLD-render it onto the grid at ``start_beat``, and equal-power-crossfade it. The ``mode``
    governs how much is imposed:

    * ``'full'`` — elastic time to ``dur_beats`` AND constant F0 to the ``note`` (max lock).
    * ``'time-only'`` — elastic time to ``dur_beats``, NO pitch (native contour kept).
    * ``'onsets'`` — onsets pinned to grid, interior only nudged (ratio ≤ ``ONSET_RATIO_CAP``), no
      pitch.
    * ``'median-pitch'`` — ``'onsets'`` timing + ONE global pitch shift moving the take's median f0
      onto the score's median note (contour preserved, register corrected).

    Unvoiced gaps between grid slots stay silent (the score's rests). Returns ``(samples float32
    mono, warp_report)`` where ``warp_report`` is ``{"mode", "bpm", "sr", "align": <report>,
    "syllables": [ row, ... ]}`` — each row ``{syllable, word, note, hz, pitch_imposed, target_t,
    target_dur, actual_t_after_warp, onset_err_ms, ratio, sustained, sustain_loops,
    achieved_dur}``. Under the note-filling modes a hold longer than the ``TIME_RATIO_MAX`` stretch
    allows is finished by a vowel steady-state crossfade loop (``sustained``/``sustain_loops``);
    the onset-oriented modes never sustain. In ``'median-pitch'`` the
    report also carries ``"global_shift": {take_median_hz, score_median_hz, shift_semitones}``.
    """
    score = load_score(score)
    bpm = float(bpm if bpm is not None else score["meta"]["bpm"])
    pieces, align_report = warp_pieces(vocal, sr, score, bpm=bpm, mode=mode)
    out = assemble_pieces(pieces, sr)
    report = {"mode": mode, "bpm": bpm, "sr": int(sr), "align": align_report,
              "syllables": [p["row"] for p in pieces]}

    if mode == "median-pitch":
        # ONE constant-interval shift on the assembled take: measure the take's median f0 (bass-safe
        # ruler, hinted by the target register), then transpose the whole clip so its median lands on
        # the score's median note. A single semitone offset ⇒ contour preserved, register corrected.
        from vox_larynx import world  # sibling tool on PYTHONPATH

        score_med = _score_median_note_hz(score)
        take_med = _measure_take_median_hz(vocal, sr, target_hint=score_med)
        shift = _median_shift_semitones(take_med, score_med)
        if abs(shift) > 1e-3:
            shifted, _params = world.render(np.ascontiguousarray(out, dtype="float64"), int(sr),
                                            semitones=shift)
            peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
            out = (shifted / peak * 0.9).astype("float32") if peak > 0 else shifted.astype("float32")
        report["global_shift"] = {
            "take_median_hz": (round(float(take_med), 2) if take_med else None),
            "score_median_hz": (round(float(score_med), 2) if score_med else None),
            "shift_semitones": round(float(shift), 3),
        }

    return out, report
