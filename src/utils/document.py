"""
document.py
Where the research document lives, and how to read it whole.

The README used to be the whole document - 4,000 lines on one page - and GitHub stops rendering
math after roughly the first 1,368 expressions of a page, so its second half was unreadable there.
It is now a short landing page, and the document is split into one page per section under
`docs/paper/` (sections 1-9, one page per study section 10.N with its 10.N.M subsections, 11, 12).

Every gate that quotes the document - the artifact-to-prose checks, the physics-claim audit -
reads it through `full_document()`, the README followed by every page in order, so a number moved
from one page to another is still found and a number deleted from all of them still fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
PAPER_DIR = REPO_ROOT / "docs" / "paper"


def paper_pages() -> List[Path]:
    """The section pages in reading order (their names sort that way: 01-, ..., 10-68-, 11-, 12-)."""
    return sorted(PAPER_DIR.glob("*.md"))


def document_files() -> List[Path]:
    """The README, then every section page: each one is a page GitHub renders on its own."""
    return [README_PATH, *paper_pages()]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rebased(path: Path) -> str:
    """A page's text with its links rebased to the repository root.

    The pages sit two levels down, so they link `../../results/figures/x.png` where the README and
    every generator that renders a block of the document write `results/figures/x.png`. Rebasing on
    read lets a generated block be compared with the page that carries it, whichever page it is.
    """
    text = read(path)
    if path.parent == PAPER_DIR:
        text = text.replace("](../../", "](")
    return text


def full_document() -> str:
    """The README and every section page, concatenated in reading order, links rebased to the root."""
    return "\n".join(rebased(path) for path in document_files())
