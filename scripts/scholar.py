#!/usr/bin/env python
"""No-install entry point: `python scripts/scholar.py <command> ...`."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from scholar_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
