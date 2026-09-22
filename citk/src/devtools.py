"""Developer CLI entry points for packaging workflows."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # src/devtools.py -> src -> citk -> workspace root
    return Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> int:
    try:
        completed = subprocess.run(command, cwd=_repo_root())
    except FileNotFoundError as exc:
        raise SystemExit(f"Required command not found: {command[0]}") from exc
    return int(completed.returncode)


def build(argv: list[str] | None = None) -> int:
    """Build wheel and sdist into citk/dist from repository root."""
    extra = list(sys.argv[1:] if argv is None else argv)
    command = [
        sys.executable,
        "-m",
        "build",
        "--outdir",
        "citk/dist",
        "citk",
        *extra,
    ]
    return _run(command)


def publish(argv: list[str] | None = None) -> int:
    """Upload distribution files from citk/dist using twine."""
    twine = shutil.which("twine")
    if twine is None:
        raise SystemExit("twine is required to publish. Install with: pip install 'citk[dev]'")

    dist_dir = _repo_root() / "citk" / "dist"
    artifacts = sorted(str(path) for path in dist_dir.glob("*"))
    if not artifacts:
        raise SystemExit("No distribution artifacts found in citk/dist. Run citk-build first.")

    extra = list(sys.argv[1:] if argv is None else argv)
    command = [twine, "upload", *artifacts, *extra]
    return _run(command)


if __name__ == "__main__":
    raise SystemExit(build())
