---
name: vox-analyze
description: Measure and interpret a voice recording with the local vox ear and vector tools. Use when asked to analyze a vocal take or voice sample, report pitch/formants/HNR/jitter/shimmer/vibrato, place a voice on vox's six axes, compare dry and processed voice renders, or check whether a voice render matches a requested character. Supports local audio files and produces evidence-based reports that separate measurements from listening judgments.
---

# Vox Analyze

Produce a measured ear/vector report for a local recording. Preserve the source
file and keep computed observations separate from interpretation.

## Prepare

Require `uv`, `smpl`, the vox dispatcher, `vox-ear`, and `vox-vector`. If a
command is missing, show these exact installs and stop until they succeed:

```bash
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
uv tool install git+https://github.com/algonormative/vox#subdirectory=packages/vox
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-ear
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-vector
```

Resolve the requested file and analysis intent. Do not overwrite or transform
the original.

## Measure

Choose a report basename derived from the input (for example,
`take-vox-20260807-153000`) and refuse to replace existing report files unless
the user explicitly approves it. Render once and preserve both the
machine-readable frames and the human report:

```bash
smpl read INPUT.wav | vox ear describe | vox vector measure | \
  smpl view > UNIQUE-REPORT.ndjson 2> UNIQUE-REPORT.md
```

Use `vox vector diff --target '{...}'` only when the user supplied a target
coordinate. Do not invent target values.

For A/B work, run the same pipeline independently on both files, using a
different collision-safe basename for each. Keep their reports distinct and
calculate only like-for-like deltas with the same units.

## Report

Return four compact sections:

1. **Source** — filename, duration/sample rate when available, and requested
   scope.
2. **Ear measurements** — F0, F1–F4, HNR, jitter, shimmer, and vibrato values
   that are actually present; include units.
3. **Six-axis vector** — humanness, breathiness, roughness, intelligibility,
   multiplicity, and spatiality, each on its reported scale.
4. **Interpretation and limits** — explain what the numbers support, flag
   tracker uncertainty or silence/noise, and reserve perceptual conclusions for
   listening evidence.

Never infer identity, age, gender, health, ethnicity, or consent from acoustic
measurements. Do not describe a subjective quality as measured unless a frame
contains the supporting value. Cite the report path and preserve it for the
user's own review.
