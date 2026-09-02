# Installing vox

vox is a compilation of small packages, not one monolith. Each tool
installs into its own isolated environment and is found by name, so you
install only what you use and a heavy dependency in one tool can never
break another. This page is the map.

Everything installs with [uv](https://docs.astral.sh/uv/).

## The shape

- **`vox`** (the dispatcher) is a tiny dependency-free command:
  `vox ear …` runs the separately installed `vox-ear`. If a tool is
  missing, the error prints the exact install command for it.
- **Tools** live under `tools/` and install independently. Most speak
  the [smpl](https://github.com/algonormative/smpl) frame protocol on
  stdin/stdout, so they sit inside smpl pipes.
- **`vox-core`** is a shared library some tools depend on (pulled in
  automatically; you never install it directly).

## Install

```bash
# the dispatcher
uv tool install git+https://github.com/algonormative/vox#subdirectory=packages/vox

# then any tools you want (each is standalone):
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-ear
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-larynx
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-vector
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-lyric
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-corpus
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-flow
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-syllabank
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-dataset
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-take
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-bodies
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-tongue
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-carrier
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-cast
```

For pipe examples, install the smpl core (`smpl read`, `smpl write`,
`smpl view`) too:

```bash
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
```

## First-run recipes

Each recipe is complete: install the listed pieces, then run its command.

### Analyze a recording

```bash
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
uv tool install git+https://github.com/algonormative/vox#subdirectory=packages/vox
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-ear
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-vector

curl -LO https://algonormative.github.io/vox/assets/guide-sung.wav
smpl read guide-sung.wav | vox ear describe | vox vector measure | smpl view
```

The shipped demo reports median F0 near `130.83 Hz` and HNR near `24.08 dB`.
Once those rows appear, replace `guide-sung.wav` with your own recording.

### Make macOS `say` sing

Install smpl and the dispatcher as above, then:

```bash
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-lyric
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-tongue
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-larynx
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-ear
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-vector
```

This path also needs a Mac with `say`. The
[six-command guide](https://algonormative.github.io/vox/singing.html) starts at a
spoken line and ends at a measured five-voice choir; it does not need
SuperCollider.

### Convert with an RVC cast

Install smpl and the dispatcher as above, then:

```bash
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-cast
vox cast setup
vox cast import --model ~/Downloads/mycast.pth --name mycast
vox cast list
smpl read take.wav | vox cast convert --model mycast --trust-model | smpl write voiced.wav
```

`setup` needs `uv` and a network connection. It builds an isolated engine
(about 3 GB total) and downloads 732,380,624 bytes (about 732 MB decimal) of
shared HuBERT/RMVPE inference assets. It does not install a person's voice.
Import only a self-trained, explicitly licensed, or synthetic model you are
authorized to use; see [CASTS.md](CASTS.md).

## Dependency matrix

| Tool | Python | Heavy deps | Optional | System binaries | Without the optional pieces |
|---|---|---|---|---|---|
| `vox` (dispatcher) | ≥3.11 | none | — | — | — |
| `vox-ear` | 3.10–3.12 | praat-parselmouth | — | — | — |
| `vox-larynx` | 3.10–3.12 | pyworld | — | — | — |
| `vox-vector` | ≥3.10 | none (numpy/scipy) | — | — | upstream `vox ear` frames enrich it; without them it falls back to its own signal proxies |
| `vox-lyric` | ≥3.10 | none (pure text) | pronouncing (CMUdict) | — | phones are `null`, syllable counts fall back to a heuristic |
| `vox-corpus` | ≥3.10 | none (numpy/soundfile) | — | — | — |
| `vox-flow` | ≥3.10 | none (numpy/soundfile) | — | `say`, `ffmpeg` | score compiles anywhere; render needs `say`, chains need `ffmpeg` |
| `vox-syllabank` | 3.10–3.12 | pyworld | `[fof]` extra (smpl-synth) | `say`; SuperCollider for FOF | seeds from `say` alone; FOF vowels skip without SC |
| `vox-dataset` | 3.10–3.12 | pyworld | `[whisper]` extra | — | phone coverage reads "unknown" without transcripts/whisper |
| `vox-take` | 3.10–3.12 | via siblings (pyworld, parselmouth) | — | `say`/`ffmpeg` for flow cards | larynx cards run without any system binary |
| `vox-bodies` | 3.10–3.12 | via siblings + smpl-synth | — | SuperCollider for SC engines | larynx-recipe bodies render without SC |
| `vox-tongue` | 3.10–3.12 | pyworld (+ siblings) | `[whisper]` extra | `say` for the sing path | compile/emit-ds work anywhere; sing needs `say`; warp needs whisper |
| `vox-carrier` | 3.10–3.12 | via siblings + smpl-analysis (librosa) | — | `say`, `ffmpeg`; SuperCollider for SC bodies | the top of the render graph; needs the full stack |
| `vox-cast` | ≥3.10 | none in the tool venv | — | `uv` (builds the engine venv) | `import`/`list`/`info`/`setup --status` work without loading the ML stack; `convert` degrades to a `vox cast setup` hint until the one-time engine build (a separate Python 3.10 venv with rvc-python + pinned torch, ~3 GB total and ~732 MB shared base assets; see [CASTS.md](CASTS.md)) |
| `vox-core` (library) | 3.10–3.12 | pyworld | praat-parselmouth | — | F0 ruler degrades to pyworld-only; also ships the voxFof/voxGrowl/voxSubSaw/voxThroat synthdefs as package data |

Notes on the pins:

- **Why 3.10–3.12 on some tools**: `pyworld` and `praat-parselmouth`
  are compiled wheels with no CPython 3.13+ builds yet. Isolated
  installs mean this pin never constrains the rest of your system.
- `pyworld` also needs `setuptools<81` (it imports `pkg_resources`);
  the affected tools pin this for you.
- smpl interop needs `smplstream`, which every frame-speaking tool
  declares as a git dependency on the smpl repo; uv resolves it
  automatically.

## Check what you have

```bash
vox --help        # lists known tools, with the install command for each
vox ear --help    # errors with the install command if vox-ear is absent
```

## Development

```bash
git clone https://github.com/algonormative/vox && cd vox
# per tool:
cd tools/vox-ear && uv sync && uv run pytest -q
# the workspace packages (dispatcher + vox-core):
cd ../.. && uv sync && uv run pytest packages -q
```
