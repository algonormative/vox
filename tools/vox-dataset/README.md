# vox-dataset — the dataset-health doctor

Rubric-scored voicebank-fitness for a directory of audio clips. Measures each clip, aggregates the
corpus, and scores it against a per-target-model rubric (0-100 + pass/warn/fail + a remediation
line per check). Model-extensible: one yaml profile per voicebank kind.

```bash
vox-dataset health <dir> [--profile diffsinger-acoustic] [--json] [--transcripts x.csv] [--whisper]
vox-dataset coverage <dir> --phones [--transcripts x.csv]
vox-dataset coverage --script script.txt [--words "..."] [--gate] [--json]   # text only, no audio
vox-dataset profiles
```

## What it measures

**Per clip** (`health.measure_clip`): sample-rate / channels / bit-depth conformance, duration,
peak + clipping check, noise-floor estimate (dBFS of the quietest 200 ms), an SNR proxy, a dryness
proxy (reverb-tail energy after each speech offset), bass-safe **guarded F0** stats (median /
range / voiced fraction — shares smpl-take's `measure_f0_guarded` ruler), and — with a transcript —
syllable rate + an ARPABET phone histogram.

**Aggregate**: total + usable minutes, a semitone pitch-coverage map, phone coverage vs the full
39-phone ARPABET inventory (which phones are missing/rare, raw and English-frequency-weighted),
duration distribution, and dryness / noise / SNR distributions.

## Rubric profiles (`src/vox_dataset/profiles/*.yaml`)

- **diffsinger-acoustic** — singing voicebank: >= 30 usable min, dry, mono 44.1 kHz, phone
  coverage >= 95% (weighted), pitch span, 5-15 s segments, no clipping.
- **vocoder-ft** — phoneme-FREE: audio quality (SNR, noise floor, no clipping), diversity (pitch
  span, clip count), minutes.
- **styletts2-ft** — spoken TTS: 30 min - 1 h, transcript phone coverage, spoken syllable rate.

Add a model by dropping a new yaml in `profiles/`. Each check is
`{id, metric, op, target, weight, severity, warn?, remediation}`; ops are
`ge/le/gt/lt/eq/in_range/frac_in_range/frac_ge/frac_le/dist_frac`. A check whose metric is
unavailable (e.g. phone coverage with no transcripts) scores `na` and is excluded from the total;
a failed `critical` check flags `critical_fail` (grade **BLOCKED**).

## Transcripts

Phone/syllable coverage needs text. Supply `--transcripts` (CSV `filename,text`) or `--whisper`
(faster-whisper, an optional dependency). Without either, phone coverage is reported **unknown**
and the transcript-dependent rubric checks become `na` — everything else still scores.

## Gating a recording script before you record it

`coverage --script <file.txt>` / `--words "..."` is the **text-only** mode: no audio is read
(`--transcripts`/`--whisper` are ignored). It g2p's the wordlist through the same CMUdict path and
scores it with the same `phones.coverage`, so the weighted % *is* the `phone_coverage_weighted_pct`
metric the **diffsinger-acoustic** rubric checks against its >= 95% target — a syllabary or
synthetic prompt set can be pre-validated before anyone sings it. Output: covered `n/39` raw +
weighted, the phone grid, missing / rare phones, out-of-dictionary words, and a PASS/FAIL line
against that target. It is a **report**: exit 0 even when coverage is incomplete. Pass `--gate`
to exit 1 instead when weighted coverage is below the target (for CI / script pre-checks).

## Dev

Isolated 3.11 venv (pyworld/parselmouth have no 3.14 wheels), reached via PYTHONPATH discovery of
the shared guarded-F0 ruler from `vox-core` (a declared dependency).

```bash
cd tools/vox-dataset && uv run pytest -q
```
