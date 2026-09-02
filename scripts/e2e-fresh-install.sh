#!/usr/bin/env bash
# Fresh-install end-to-end: prove every documented command runs as shown from
# an isolated install of this checkout. Set VOX_GIT_REF (for example, `main` or
# a release tag) to exercise the published GitHub source instead. Installs into
# an ISOLATED tool dir (never touches your real `uv tool` installs), then runs
# the singing guide's six commands and the README pipes verbatim.
# Needs: uv, macOS `say`, and network access for dependencies.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E2E_ROOT="${E2E_ROOT:-$(mktemp -d)}"
export UV_TOOL_DIR="$E2E_ROOT/tools"
export UV_TOOL_BIN_DIR="$E2E_ROOT/bin"
export PATH="$UV_TOOL_BIN_DIR:$PATH"
SMPL_CAS_DIR="$E2E_ROOT/cas"
export SMPL_CAS_DIR
WORK="$E2E_ROOT/work"
mkdir -p "$WORK"
cd "$WORK" || exit 1

declare -a STEPS RESULTS
fail=0
step() {  # name, then the command as remaining args
  local name="$1"; shift
  echo
  echo "=== $name"
  if "$@"; then
    STEPS+=("$name"); RESULTS+=("pass")
  else
    STEPS+=("$name"); RESULTS+=("FAIL")
    fail=1
  fi
}

# ---------------------------------------------------------- installs ----
# smpl core, exactly as the smpl README documents it.
step "install smpl core" uv tool install --quiet \
  "git+https://github.com/algonormative/smpl#subdirectory=packages/smpl" \
  --with "git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream" \
  --with "git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis"

# Vox tools use the same package boundaries as INSTALL.md. The checkout is the
# default so CI verifies the code under test rather than whatever is currently
# published on `main`; VOX_GIT_REF enables an explicit post-release check.
for t in packages/vox tools/vox-ear tools/vox-larynx tools/vox-vector tools/vox-lyric tools/vox-tongue tools/vox-cast; do
  if [[ -n "${VOX_GIT_REF:-}" ]]; then
    source_spec="git+https://github.com/algonormative/vox@${VOX_GIT_REF}#subdirectory=$t"
  else
    source_spec="$REPO_ROOT/$t"
  fi
  # Refresh the package itself so a same-version wheel from another source
  # (for example an earlier GitHub install) cannot mask checkout changes.
  package_name="${t##*/}"
  step "install $t" uv tool install --quiet \
    --refresh-package "$package_name" "$source_spec"
done

# Guard against PATH fallthrough: a failed install must not silently borrow a
# system-wide binary and fake a pass. Both entrypoints must live in the isolated bin.
fresh_bin() {
  local resolved
  resolved="$(command -v "$1" || true)"
  case "$resolved" in
    "$UV_TOOL_BIN_DIR"/*) return 0 ;;
    *) echo "  $1 resolved outside the fresh install: ${resolved:-not found}"; return 1 ;;
  esac
}
step "smpl binary is the fresh install" fresh_bin smpl
step "vox binary is the fresh install" fresh_bin vox

step "vox --help lists tools" sh -c 'vox --help | grep -q "ear"'
step "smpl --help runs" sh -c 'smpl --help > /dev/null'

# ----------------------------------------------- analyze-card quick start ----
step "analyze card: download shipped take" \
  curl -fL -o guide-sung.wav https://algonormative.github.io/vox/assets/guide-sung.wav
step "analyze card: reproduce measured values" sh -c \
  'smpl read guide-sung.wav | vox ear describe | vox vector measure | smpl view > /dev/null 2>guide-report.md && grep -q "130.83" guide-report.md && grep -q "24.08" guide-report.md'

# ------------------------------------------- the singing guide, verbatim ----
step "guide 1: say speaks" \
  say -v Fred --file-format=WAVE --data-format=LEI16@44100 -o spoken.wav "slow river carry me home"

step "guide 2: build the reviewed lyric packet" sh -c \
  'vox lyric packet --delivery sustained --lines "slow river carry me home" > river-packet.json && grep -q "\"n_rewrite\": 0" river-packet.json'

step "guide 3: compile the packet into the score" \
  vox tongue compile-packet --packet river-packet.json \
    --melody "A2,C3,E3,D3,C3" --bpm 90 --score river.yaml

step "guide 4: sing it" sh -c 'vox tongue sing --score river.yaml --out sung.wav > /dev/null'

step "guide 5: stack a choir (smpl|vox pipe)" sh -c \
  'smpl read sung.wav | vox larynx harmonize --chord 0,3,7 --drone | smpl write choir.wav > /dev/null'

step "guide 6: measure it" sh -c \
  'smpl read choir.wav | vox ear describe | vox vector measure | smpl view | grep -q "voice.f0_median_hz"'

# --------------------------------------------------- README pipes ----
step "README: ear report" sh -c \
  'smpl read sung.wav | vox ear describe | smpl view | grep -q "voice.hnr_db"'
step "README: vector diff" sh -c \
  'smpl read sung.wav | vox ear describe | vox vector diff --target "{\"breathiness\":0.2,\"roughness\":0.1}" | grep -q "mean_abs_error"'
step "README: larynx retune" sh -c \
  'smpl read sung.wav | vox larynx render --semitones 2 | smpl write up2.wav > /dev/null'
step "README: lyric review json" sh -c \
  'vox lyric review --delivery percussive --lines "spit the code back|cut the deck to black" --json | grep -q "n_lines"'

# ------------------------------------------------- cast degradation ----
# The documented behavior WITHOUT the 3 GB engine build: status says so,
# and convert fails with the setup hint (never a traceback). Point the
# engine path inside E2E_ROOT so a host machine's real engine can't leak in.
export VOX_RVC_ENGINE="$E2E_ROOT/vox-engine"
export VOX_CASTS_DIR="$E2E_ROOT/casts"
step "cast: status reports engine absent" sh -c \
  'vox cast setup --status | grep -q "\"installed\": false"'
mkdir -p "$WORK/fakecast" && : > "$WORK/fakecast/fake.pth" && : > "$WORK/fakecast/fake.index"
step "cast: import local model" sh -c \
  'vox cast import --model fakecast --name fakecast | grep -q "\"name\": \"fakecast\""'
step "cast: list imported model" sh -c \
  'vox cast list | grep -q "\"name\": \"fakecast\""'
step "cast: convert degrades with setup hint" sh -c \
  '! (smpl read spoken.wav | vox cast convert --model fakecast >/dev/null 2>cast-err.txt) && grep -q "vox cast setup" cast-err.txt'

# ------------------------------------------------------------ summary ----
echo
echo "== fresh-install E2E summary  (root: $E2E_ROOT)"
for i in "${!STEPS[@]}"; do
  printf '  %-42s %s\n' "${STEPS[$i]}" "${RESULTS[$i]}"
done
exit "$fail"
