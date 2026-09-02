# Voice casts (RVC models)

A **cast** is a trained RVC voice-conversion model. It re-voices a take while
keeping the performed words, timing, and pitch contour. `vox cast` manages a
local cast library and converts audio through it as an smpl pipe stage.

Training is out of scope. vox loads standard RVC exports; it does not train a
voice, supply a person's model, or determine that a model is lawful to use.
Use only a self-trained voice, an explicitly licensed model made with informed
consent, or a synthetic voice. Do not use casts to impersonate someone.

The complete first-run web walkthrough is the
[RVC cast guide](https://algonormative.github.io/vox/casting.html).

## Install

The runnable pipe needs smpl, the light vox dispatcher, and `vox-cast`:

```bash
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
uv tool install git+https://github.com/algonormative/vox#subdirectory=packages/vox
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-cast
```

## Shared engine setup is not cast installation

The ML stack behind RVC has pins that cannot share an environment with the
rest of vox. `vox-cast` therefore builds and invokes a separate engine venv:

```bash
vox cast setup --status   # inspect; changes nothing
vox cast setup            # one-time networked setup
```

Setup creates a Python 3.10 environment under `~/.vox/engines/rvc` (about
3 GB after torch and dependencies) and downloads 732,380,624 bytes (about
732 MB decimal) of shared HuBERT/RMVPE inference assets. Those assets are
pinned to [Hugging Face commit
`bbb6736b97a98df0a87fe3592c0a061c53f0a75f`](https://huggingface.co/daswer123/rvc_base/commit/bbb6736b97a98df0a87fe3592c0a061c53f0a75f)
and are used by every cast. They
are content and pitch models, **not a person's voice**.

After setup, conversion is local and offline. `VOX_RVC_ENGINE` overrides the
engine location. `vox cast setup --fresh` intentionally rebuilds an existing
engine. The Python stack installs from the checked-in, fully resolved
hash-verified lock at `tools/vox-cast/src/vox_cast/engine-requirements.lock`.
Do not upgrade its load-bearing pins by hand; see
`tools/vox-cast/src/vox_cast/engine.py`.

## Model provenance gate

Before importing a model:

1. Confirm who trained it, from what source recordings, and under what consent.
2. Confirm the license explicitly permits your intended use and, if relevant,
   redistribution.
3. Verify the published checksum before loading the `.pth`.
4. Read the model card for architecture, sample rate, limitations, and use
   restrictions.

RVC `.pth` files are pickle-based artifacts. Treat an unknown model as
executable untrusted input, not merely an audio file.

Compare the downloaded files with the publisher's `SHA256SUMS.txt` before
import. Use `shasum -a 256 FILE.pth FILE.index` on macOS or
`sha256sum FILE.pth FILE.index` on Linux. A matching hash identifies the bytes;
it does not establish consent or a lawful license.

A complete local or release bundle looks like:

```text
mycast/
├── mycast.pth          # required weights
├── mycast.index        # optional retrieval bank
├── config.json         # optional trainer sidecar
├── model_info.json     # optional trainer sidecar
├── MODEL_CARD.md       # source, consent, intended use, limitations
├── LICENSE.md          # permission and restrictions
└── SHA256SUMS.txt      # release checksums
```

No hosted first-party model is claimed by this project. Bring a model you are
authorized to use. A standard RVC v2 export works; use `--arch v1` at conversion
time only for a v1 model.

## Import and inventory

Import a directory or one `.pth`. `--index` selects an explicit retrieval
bank; `--name` chooses the stable library name:

```bash
vox cast import --model ~/Downloads/mycast --name mycast

# Or name the files explicitly:
vox cast import --model ~/Downloads/mycast.pth \
  --index ~/Downloads/mycast.index --name mycast
```

Import copies the weights, selected index, and common
JSON/YAML/text/image/Markdown sidecars into
`~/.vox/casts/<name>/` (or `VOX_CASTS_DIR`) and refuses to overwrite an
existing cast. It prints a JSON receipt. Inventory and inspect without loading
the ML stack:

```bash
vox cast list
vox cast info --model mycast
```

The import and info receipts include SHA-256 for the `.pth` and optional
`.index`, plus an inventory of copied provenance sidecars. `list` stays fast by
omitting full-file hashes.

`--model` on `info` and `convert` also accepts a cast directory or a direct
`.pth` path. A bare name resolves under the local cast library.

## Convert

The pipe stage keeps upstream frames and appends the converted audio:

```bash
smpl read take.wav | vox cast convert --model mycast --trust-model | smpl write voiced.wav
```

`--trust-model` is an explicit acknowledgment that RVC `.pth` weights can
execute Python code when loaded. The engine venv isolates dependencies; it is
**not** a security sandbox. Run only a model whose publisher and exact checksum
you trust.

Standalone conversion uses the same implementation:

```bash
vox cast convert --in take.wav --model mycast --trust-model \
  --pitch 0 --index-rate 0.5 | smpl write voiced.wav
```

The output frame uses a `…wet` role and records the model name and every knob in
`params`. Keep the original take with the converted output so the result can be
reviewed against its source.

## The knobs

| Flag | Default | What it does |
|---|---:|---|
| `--pitch` | `0` | Transpose in semitones before conversion; useful when source and cast ranges differ. |
| `--index-rate` | `0.5` | Retrieval blend from 0–1; higher pulls harder toward the cast's timbre bank. |
| `--protect` | `0.33` | Consonant/breath protection from 0–0.5; lower applies more conversion. |
| `--f0-method` | `rmvpe` | Pitch tracker; `rmvpe` is the robust default, `harvest` the classic fallback. |
| `--rms-mix` | `1.0` | Loudness-envelope blend; 1 keeps the source dynamics, 0 takes the cast's. |
| `--arch` | `v2` | RVC model generation (`v1` or `v2`). |
| `--device` | `cpu:0` | Deterministic default; `cuda:0` when available; `mps` is opt-in and known unstable. |

Change one control at a time. Listen for intelligibility, pitch stability,
consonant damage, breath artifacts, and whether the performance survives.
Measurements can support that review, but cannot prove identity, consent, or
authorization.

## Publishing a redistributable first-party cast

If this project later produces a cast with documented redistribution rights,
publish each version through [**GitHub Releases**](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), not as Git objects and not
through Git LFS. Attach:

- the `.pth` weights;
- the optional `.index` retrieval bank;
- a model card naming the dataset source and consent, trainer and RVC
  architecture, intended and prohibited uses, and known limitations;
- the license; and
- checksums for every binary asset.

Release mechanics do not create permission. Verify consent and redistribution
rights first, and do not advertise a model as available until the release and
checksums actually exist.

## Lineage

The techniques under `vox cast`—VITS, HuBERT/ContentVec, RMVPE, faiss
retrieval, and the RVC software family—are cited in
[REFERENCES.md](REFERENCES.md#vox-cast--rvc-voice-conversion).
