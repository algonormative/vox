"""`vox-dataset` CLI — the dataset-health doctor.

    vox-dataset health <dir> [--profile diffsinger-acoustic] [--json] [--transcripts x.csv]
    vox-dataset coverage <dir> --phones [--transcripts x.csv] [--whisper]
    vox-dataset coverage --script script.txt [--words "..."] [--gate] [--json]
    vox-dataset profiles

`health` measures every clip under <dir>, aggregates, and (with --profile) scores it against a
rubric — a pretty report by default, or a machine report with --json. `coverage --phones` is a
quick phone-inventory view. Transcripts come from --transcripts (CSV: filename,text) or, with
--whisper, from faster-whisper if installed (otherwise phone coverage is reported as 'unknown').

`coverage --script/--words` is the TEXT-ONLY mode: no audio is read at all — it g2p's a wordlist
(a syllabary, a synthetic prompt set) and reports coverage against the 39-phone ARPABET inventory,
so a recording script can be gated BEFORE anyone records it. The weighted % is the same metric the
diffsinger-acoustic rubric scores, so a script can be pre-validated against its >= 95% target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import health, phones, rubric

# The rubric profile whose phone-coverage target a script is pre-validated against.
_TARGET_PROFILE = "diffsinger-acoustic"


# --------------------------------------------------------------------------- formatting
def _c(s: str, code: str, color: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if color else s


_STATUS_STYLE = {"pass": "32", "warn": "33", "fail": "31", "na": "90"}
_STATUS_MARK = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "na": " NA "}
_GRADE_STYLE = {"READY": "32", "USABLE": "36", "MARGINAL": "33", "UNFIT": "31", "BLOCKED": "31"}


def _fmt_num(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def render_report(report: dict, scored: dict | None, color: bool = True) -> str:
    m = report["metrics"]
    fmt = report["format"]
    dist = report["distributions"]
    lines: list[str] = []
    dirn = report.get("directory", "?")
    lines.append(_c(f"dataset-health  {dirn}", "1", color))
    lines.append("")

    # Summary block.
    lines.append(_c("  corpus", "1", color))
    lines.append(f"    clips              {m['clip_count']}  (usable {m['usable_clip_count']})")
    lines.append(f"    total minutes      {_fmt_num(m['total_minutes'])}"
                 f"   usable {_fmt_num(m['usable_minutes'])}")
    lines.append(f"    sample rates       {fmt['sr_counts']}")
    lines.append(f"    channels           {fmt['channel_counts']}")
    lines.append(f"    bit depth          {fmt['bitdepth_counts']}")
    lines.append("")

    # Audio quality.
    lines.append(_c("  audio quality", "1", color))
    lines.append(f"    clipped clips      {_fmt_num(m['frac_clipped'])}  fraction")
    lines.append(f"    SNR (mean)         {_fmt_num(m['mean_snr_db'])} dB"
                 f"   [{_fmt_num(dist['snr_db']['min'])} .. {_fmt_num(dist['snr_db']['max'])}]")
    lines.append(f"    noise floor (med)  {_fmt_num(m['median_noise_floor_dbfs'])} dBFS")
    lines.append(f"    dryness (med tail) {_fmt_num(m['median_dryness_db'])} dB")
    lines.append(f"    peak dBFS          [{_fmt_num(dist['peak_dbfs']['min'])} .."
                 f" {_fmt_num(dist['peak_dbfs']['max'])}]")
    lines.append("")

    # Duration + pitch.
    d = dist["duration_s"]
    lines.append(_c("  distribution", "1", color))
    lines.append(f"    duration (s)       min {_fmt_num(d['min'])}  med {_fmt_num(d['median'])}"
                 f"  max {_fmt_num(d['max'])}")
    p = report["pitch"]
    if p["have_data"]:
        lines.append(f"    pitch span         {p['span_semitones']} semitones"
                     f"  ({_fmt_num(p['lo_hz'])}-{_fmt_num(p['hi_hz'])} Hz)")
    else:
        lines.append("    pitch span         (no voiced material found)")
    lines.append("")

    # Phone coverage.
    ph = report["phones"]
    lines.append(_c("  phone coverage", "1", color))
    if not ph["have_transcripts"]:
        lines.append(_c("    unknown  (no transcripts — supply --transcripts or --whisper)", "90", color))
    else:
        cov = ph["coverage"]
        lines.append(f"    inventory          {cov['n_present']}/{cov['n_inventory']} phones"
                     f"  ({_fmt_num(m['phone_coverage_raw_pct'])}% raw,"
                     f" {_fmt_num(m['phone_coverage_weighted_pct'])}% weighted)")
        if cov["missing"]:
            lines.append(_c(f"    missing            {', '.join(cov['missing'])}", "31", color))
        if cov["rare"]:
            lines.append(_c(f"    rare (<=2)         {', '.join(cov['rare'])}", "33", color))
    lines.append("")

    # Rubric.
    if scored is not None:
        grade = scored["grade"]
        gstyle = _GRADE_STYLE.get(grade, "0")
        lines.append(_c(f"  rubric: {scored['profile']}", "1", color))
        head = f"    SCORE {scored['score']:.1f}/100   {grade}"
        lines.append(_c(head, gstyle, color))
        if scored["critical_fail"]:
            lines.append(_c("    ** a critical check FAILED — dataset is disqualified as-is **", "31", color))
        lines.append("")
        for c in scored["checks"]:
            mark = _c(_STATUS_MARK[c["status"]], _STATUS_STYLE[c["status"]], color)
            val = _fmt_num(c["value"])
            lines.append(f"    [{mark}] {c['label']:<38} = {val}")
            if c["status"] in ("warn", "fail") and c.get("remediation"):
                lines.append(_c(f"            -> {c['remediation']}", "90", color))
        lines.append("")
    return "\n".join(lines)


def _phone_grid(counts: dict, color: bool = True) -> list[str]:
    """The per-phone ``NAME:count`` grid, 8 to a row (red = absent, yellow = thin)."""
    lines: list[str] = []
    row: list[str] = []
    for p in phones.ARPABET_INVENTORY:
        n = counts.get(p, 0)
        cell = f"{p}:{n}"
        if n == 0:
            cell = _c(cell, "31", color)
        elif n <= 2:
            cell = _c(cell, "33", color)
        row.append(cell)
        if len(row) == 8:
            lines.append("  " + "  ".join(row))
            row = []
    if row:
        lines.append("  " + "  ".join(row))
    return lines


def render_coverage(report: dict, color: bool = True) -> str:
    ph = report["phones"]
    lines = [_c(f"phone coverage  {report.get('directory','?')}", "1", color), ""]
    if not ph["have_transcripts"]:
        lines.append(_c("  unknown — no transcripts. Supply --transcripts CSV or --whisper.", "90", color))
        return "\n".join(lines)
    cov = ph["coverage"]
    lines.append(f"  present  {cov['n_present']}/{cov['n_inventory']}"
                 f"  ({cov['raw_pct']}% raw, {cov['weighted_pct']}% weighted)")
    lines.append("")
    lines.extend(_phone_grid(ph["counts"], color))
    if cov["missing"]:
        lines.append("")
        lines.append(_c(f"  missing: {', '.join(cov['missing'])}", "31", color))
    return "\n".join(lines)


def render_script_coverage(report: dict, target: float | None, color: bool = True) -> str:
    """Render the text-only (pre-recording) script coverage report."""
    cov = report["coverage"]
    lines = [_c("phone coverage  (script — text only, no audio)", "1", color), ""]
    lines.append(f"  words    {report['words']}  ({report['words_in_dict']} in dict,"
                 f" {report['words_out']} out)")
    lines.append(f"  present  {cov['n_present']}/{cov['n_inventory']}"
                 f"  ({cov['raw_pct']}% raw, {cov['weighted_pct']}% weighted)")
    lines.append("")
    lines.extend(_phone_grid(report["counts"], color))
    lines.append("")
    if cov["missing"]:
        lines.append(_c(f"  missing: {', '.join(cov['missing'])}", "31", color))
    if cov["rare"]:
        lines.append(_c(f"  rare (<=2): {', '.join(cov['rare'])}", "33", color))
    if report["oov"]:
        lines.append(_c(f"  OOV: {', '.join(report['oov'])}", "33", color))
    else:
        lines.append("  OOV: none")
    if target is not None:
        ok = cov["weighted_pct"] >= target
        verdict = _c("PASS", "32", color) if ok else _c("FAIL", "31", color)
        lines.append(f"  {_TARGET_PROFILE} target: weighted >= {_fmt_num(target)}%  -> {verdict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- commands
def _load_transcripts(args) -> dict | None:
    if args.transcripts:
        return health.load_transcripts(args.transcripts)
    return None


def _progress(i, total, path):
    print(f"\r  measuring {i}/{total}  {path.name[:40]:<40}", end="", file=sys.stderr, flush=True)


def cmd_health(args) -> int:
    transcripts = _load_transcripts(args)
    use_whisper = args.whisper
    if use_whisper and not health.whisper_available():
        print("note: --whisper set but faster-whisper is not installed; phone coverage will be 'unknown'.",
              file=sys.stderr)
        use_whisper = False
    report = health.measure_dataset(
        args.directory, transcripts=transcripts, use_whisper=use_whisper,
        whisper_size=args.whisper_size,
        progress=None if args.json else _progress,
    )
    if not args.json:
        print(file=sys.stderr)  # end progress line
    scored = rubric.score(report, args.profile) if args.profile else None

    if args.json:
        out = {"report": _strip_clips(report), "rubric": scored}
        text = json.dumps(out, indent=2, default=_json_default)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
    else:
        text = render_report(report, scored, color=_use_color(args))
        print(text)
        if args.out:
            Path(args.out).write_text(
                json.dumps({"report": _strip_clips(report), "rubric": scored},
                           indent=2, default=_json_default),
                encoding="utf-8")
            print(f"\n  json written: {args.out}", file=sys.stderr)
    if scored and scored["critical_fail"]:
        return 2
    return 0


def _script_text(args) -> str | None:
    """The text selecting the text-only mode: --script file + --words, concatenated (None if neither)."""
    parts: list[str] = []
    if args.script:
        parts.append(Path(args.script).read_text(encoding="utf-8"))
    if args.words:
        parts.append(args.words)
    return "\n".join(parts) if parts else None


def _weighted_target(profile: str = _TARGET_PROFILE) -> float | None:
    """The rubric's weighted phone-coverage target (e.g. 95), or None if unreadable."""
    try:
        prof = rubric.load_profile(profile)
    except Exception:  # noqa: BLE001 — a missing/unreadable profile just drops the verdict line
        return None
    for check in prof.get("checks", []):
        if check.get("metric") == "phone_coverage_weighted_pct":
            try:
                return float(check["target"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def cmd_coverage_script(args, text: str) -> int:
    """Text-only phone coverage: g2p a wordlist, no audio touched."""
    if args.transcripts or args.whisper:
        print("note: text-only mode (--script/--words) — --transcripts/--whisper ignored.",
              file=sys.stderr)
    if not phones.have_pronouncing():
        print("error: text-only coverage needs the `pronouncing` (CMUdict) package installed.",
              file=sys.stderr)
        return 1
    report = phones.script_coverage(text)
    target = _weighted_target()
    if args.json:
        out = dict(report)
        out["target_profile"] = _TARGET_PROFILE
        out["target_weighted_pct"] = target
        print(json.dumps(out, indent=2, default=_json_default))
    else:
        print(render_script_coverage(report, target, color=_use_color(args)))
    if args.gate and target is not None and report["coverage"]["weighted_pct"] < target:
        return 1
    return 0


def cmd_coverage(args) -> int:
    text = _script_text(args)
    if text is not None:
        return cmd_coverage_script(args, text)
    transcripts = _load_transcripts(args)
    use_whisper = args.whisper and health.whisper_available()
    report = health.measure_dataset(
        args.directory, transcripts=transcripts, use_whisper=use_whisper,
        whisper_size=args.whisper_size, progress=None,
    )
    if args.json:
        print(json.dumps(report["phones"], indent=2, default=_json_default))
    else:
        print(render_coverage(report, color=_use_color(args)))
    return 0


def cmd_profiles(args) -> int:
    for name in rubric.list_profiles():
        prof = rubric.load_profile(name)
        desc = (prof.get("description") or "").strip().replace("\n", " ")
        print(f"  {name:<22} {desc[:90]}")
    return 0


def _strip_clips(report: dict) -> dict:
    """Drop the per-clip array for the top-level json unless it's wanted (keeps output compact)."""
    out = dict(report)
    out.pop("clips", None)
    return out


def _json_default(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _use_color(args) -> bool:
    if args.no_color:
        return False
    return sys.stdout.isatty()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vox-dataset", description=__doc__)
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    sub = parser.add_subparsers(dest="command", required=True)

    h = sub.add_parser("health", help="measure + score a dataset")
    h.add_argument("directory")
    h.add_argument("--profile", help="rubric profile (see `profiles`); omit for measurements only")
    h.add_argument("--transcripts", help="CSV of filename,text for phone/syllable coverage")
    h.add_argument("--whisper", action="store_true", help="transcribe missing text with faster-whisper")
    h.add_argument("--whisper-size", default="base", help="faster-whisper model size (default base)")
    h.add_argument("--json", action="store_true", help="emit machine JSON instead of a report")
    h.add_argument("--out", help="also write full JSON report to this path")
    h.set_defaults(func=cmd_health)

    c = sub.add_parser("coverage",
                       help="quick phone-inventory view — of a dataset, or (--script/--words) of a "
                            "recording script before it is recorded")
    c.add_argument("directory", nargs="?", help="dataset directory (omit when using --script/--words)")
    c.add_argument("--phones", action="store_true", help="show the ARPABET phone inventory")
    c.add_argument("--script", help="text file of words/lines to g2p (text-only mode, no audio read)")
    c.add_argument("--words", help="inline text to g2p (text-only mode); combines with --script")
    c.add_argument("--gate", action="store_true",
                   help=f"text-only mode: exit 1 when weighted coverage is below the "
                        f"{_TARGET_PROFILE} target")
    c.add_argument("--transcripts", help="CSV of filename,text")
    c.add_argument("--whisper", action="store_true", help="transcribe with faster-whisper")
    c.add_argument("--whisper-size", default="base")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_coverage)

    p = sub.add_parser("profiles", help="list available rubric profiles")
    p.set_defaults(func=cmd_profiles)

    args = parser.parse_args(argv)
    if args.command == "coverage" and not (args.directory or args.script or args.words):
        parser.error("coverage needs a dataset <directory>, or --script FILE / --words TEXT "
                     "for the text-only (pre-recording) mode")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
