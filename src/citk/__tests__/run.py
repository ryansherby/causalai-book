"""Entry point used by the ``citk-test`` console script."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run the citk ``__tests__`` suite and return pytest's exit code.

    Parameters
    ----------
    argv : list of str, optional
        Extra arguments forwarded to pytest (for example ``-k chapter_2``
        or ``-q``). Defaults to ``sys.argv[1:]`` when called from the
        command line.
    """
    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - environment/setup error
        raise SystemExit(
            "pytest is required to run citk tests. Install with: pip install 'citk[test]'"
        ) from exc

    testdir = Path(__file__).resolve().parent
    extra = list(sys.argv[1:] if argv is None else argv)
    return int(pytest.main([str(testdir), *extra]))


if __name__ == "__main__":
    raise SystemExit(main())
