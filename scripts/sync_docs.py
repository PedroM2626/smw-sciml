"""scripts/sync_docs.py - keep the docs site's README copy equal to the real one.

docs/full-readme.md is this repository's README rendered through mkdocs-material and
MathJax, where GitHub's roughly-1,368-expression rendering budget does not apply. It
is a generated file, and this script is the only writer: it copies README.md over it
so the site can never disagree with the document the CI gates read. The docs workflow
runs it before every build; locally:

    python scripts/sync_docs.py

Exits non-zero when README.md is missing, so a mislocated invocation fails loudly
instead of silently publishing a stale page.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "README.md"
TARGET = REPO / "docs" / "full-readme.md"


def main() -> int:
    if not SOURCE.is_file():
        print(f"scripts/sync_docs.py: {SOURCE} not found", file=sys.stderr)
        return 1
    text = SOURCE.read_text(encoding="utf-8")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"synced README.md ({len(text.splitlines())} lines) "
        f"-> {TARGET.relative_to(REPO).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
