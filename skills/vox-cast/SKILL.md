---
name: vox-cast
description: Safely operate local RVC voice conversion with vox cast. Use when asked to check RVC readiness, set up the isolated inference engine, import or inventory an authorized voice model, inspect model provenance, convert a local vocal take, tune conversion controls, or compare dry and converted results. Enforces consent and provenance checks, never supplies a person's model, and keeps large network setup explicit.
---

# Vox Cast

Run a provenance-first, local RVC workflow: check readiness, import an
authorized model, convert, and compare dry against wet.

## Gate the model

Proceed only with a self-trained voice, an explicitly licensed model made with
informed consent, or a synthetic voice. Ask for the model's source, license,
intended use, and checksum when they are not documented. Do not find, download,
or use a model of a public figure or another person without authorization. Do
not assist deceptive impersonation.

Treat `.pth` weights as untrusted executable input. Prefer a bundle containing
the `.pth`, optional `.index`, model card, license, and checksums.
Verify the weights and index against the published receipt with
`shasum -a 256` on macOS or `sha256sum` on Linux before import. A matching hash
identifies the artifact but does not establish consent or permission.

## Check readiness

Require `uv`, `smpl`, the vox dispatcher, and `vox-cast`. If missing, show:

```bash
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
uv tool install git+https://github.com/algonormative/vox#subdirectory=packages/vox
uv tool install git+https://github.com/algonormative/vox#subdirectory=tools/vox-cast
```

Inspect without changing state:

```bash
vox cast setup --status
vox cast list
```

If the engine is absent, explain before running `vox cast setup`: it needs the
network, builds an isolated environment about 3 GB total, and downloads
732,380,624 bytes (about 732 MB) of shared HuBERT/RMVPE inference assets. Those
assets are not a person's voice. Get confirmation before starting this large
setup.

## Import and inspect

After the provenance gate passes, import without overwriting an existing name:

```bash
vox cast import --model MODEL.pth --index MODEL.index --name NAME
vox cast list
vox cast info --model NAME
```

Omit `--index` when none exists. `--model` may instead name a directory; import
copies recognized sidecars. Preserve the JSON import receipt and verify that
`info` matches the expected files and metadata.

## Convert

Keep the dry input and write a new output:

```bash
smpl read INPUT.wav | vox cast convert --model NAME --trust-model | smpl write OUTPUT.wav
```

Use `--trust-model` only after the provenance and checksum gate passes. It
acknowledges that pickle-based weights can execute Python code; the engine venv
is dependency isolation, not a security sandbox. Start with defaults and
`cpu:0`. Change one control at a time: `--pitch` for
range, `--index-rate` for retrieval strength, and `--protect` for consonants
and breaths. Record the command; the output frame also carries model and knob
parameters.

## Compare

Listen dry against wet for intelligibility, pitch stability, consonant damage,
breath artifacts, and preservation of the performance. When `vox-ear` and
`vox-vector` are installed, run `$vox-analyze` independently on input and
output and report like-for-like deltas. Measurements can characterize the
render; they cannot prove identity, consent, or authorization.

Return readiness, provenance evidence, import receipt, conversion settings,
output path, comparison, and unresolved risks. Keep final artistic and release
judgment with the user. Never redistribute a model unless its license and
consent explicitly allow it; first-party redistributable casts belong in
GitHub Releases with weights, optional index, model card, license, and
checksums—not Git or Git LFS.
