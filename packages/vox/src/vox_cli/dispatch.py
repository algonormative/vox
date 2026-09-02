"""The vox dispatcher: git-style external subcommand discovery.

- ``vox <x>`` execs ``vox-<x>`` on PATH (each heavy voice tool is its own
  isolated install, in the smpl two-tier image).
- A missing first-party tool prints the exact install command, not just
  the name of what is absent.
- Built-ins may arrive later; the seam is the same as smpl's dispatcher,
  minus the multicall shims until something needs them.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys

from . import __version__

REPO = "https://github.com/algonormative/vox"

# First-party tools and the subdirectory that installs each; keep in step
# with tools/ as ports land.
KNOWN_TOOLS: dict[str, str] = {
    "ear": "tools/vox-ear",
    "corpus": "tools/vox-corpus",
    "larynx": "tools/vox-larynx",
    "vector": "tools/vox-vector",
    "lyric": "tools/vox-lyric",
    "flow": "tools/vox-flow",
    "syllabank": "tools/vox-syllabank",
    "dataset": "tools/vox-dataset",
    "take": "tools/vox-take",
    "bodies": "tools/vox-bodies",
    "tongue": "tools/vox-tongue",
    "carrier": "tools/vox-carrier",
    "cast": "tools/vox-cast",
}


def _install_sigpipe() -> None:
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass


def _top_level_help() -> int:
    print("vox — voice synthesis and analysis as pipe tools (smpl frame protocol)\n")
    print("usage: vox <command> [args]\n")
    print("commands are separate installs, found by name (`vox ear` runs `vox-ear`):")
    for name, subdir in KNOWN_TOOLS.items():
        print(f"  {name:<12}uv tool install git+{REPO}#subdirectory={subdir}")
    print("\n  vox --version   show version")
    return 0


def main(argv: list[str] | None = None) -> int:
    _install_sigpipe()
    args = list(sys.argv if argv is None else argv)[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        return _top_level_help()
    if args[0] == "--version":
        print(f"vox {__version__}")
        return 0

    name, rest = args[0], args[1:]
    exe = shutil.which(f"vox-{name}")
    if exe is None:
        hint = (
            f"      uv tool install git+{REPO}#subdirectory={KNOWN_TOOLS[name]}\n"
            if name in KNOWN_TOOLS
            else "      No first-party tool by that name; run `vox --help`.\n"
        )
        sys.stderr.write(
            f"vox: unknown command {name!r}\n"
            f"      no `vox-{name}` found on PATH. To install:\n" + hint
        )
        return 127
    env = dict(os.environ)
    return subprocess.call([exe, *rest], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
