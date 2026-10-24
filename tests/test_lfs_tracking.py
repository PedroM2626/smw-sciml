r"""Gate: every large binary must actually live in Git LFS, not merely match a pattern.

The .gitattributes patterns shape the *next* commit, not the ones already made: a file
added before git lfs install (or with the smudge filter off) lands as a regular blob
while the attributes still claim it is routed, and every fresh clone silently regains
the tens of megabytes the migration removed. This gate walks the tracked files and
fails on any path matching an LFS suffix that is not an LFS pointer, and on an
attributes file that stopped declaring one of the suffixes it is supposed to route.

Offline by design: it reads the index and .gitattributes only, and skips when the
checkout is not a git repository or git-lfs is not installed (the CI checkouts run
with lfs: true, so both are present there).

Run:  pytest tests/test_lfs_tracking.py -q
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: the suffixes .gitattributes routes through LFS. A suffix missing from the attributes
#: file is as much a regression as a big file stored as a regular blob.
LFS_SUFFIXES = (
    ".pt",
    ".pth",
    ".onnx",
    ".npz",
    ".state",
    ".bk2",
    ".dll",
    ".zip",
    ".mp4",
    ".gif",
    ".png",
    ".sfc",
    ".smc",
)


def _git(*args: str):
    """stdout of a git command, or None when git is absent / the command fails."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _require_git_checkout() -> None:
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout (installed package or exported tree)")


def _lfs_files():
    out = _git("lfs", "ls-files", "--name-only")
    if out is None:
        pytest.skip("git-lfs is not installed, so LFS pointers cannot be listed")
    return {line.strip() for line in out.splitlines() if line.strip()}


def test_every_pattern_matching_tracked_file_is_an_lfs_pointer():
    _require_git_checkout()
    lfs = _lfs_files()
    tracked = (_git("ls-files") or "").splitlines()
    offenders = sorted(
        path for path in tracked if Path(path).suffix.lower() in LFS_SUFFIXES and path not in lfs
    )
    assert not offenders, (
        "tracked files matching the LFS patterns but stored as regular blobs "
        "(run: git add --renormalize . and re-commit): " + ", ".join(offenders)
    )


def test_gitattributes_still_declares_every_lfs_suffix():
    text = (REPO / ".gitattributes").read_text(encoding="utf-8")
    missing = [suffix for suffix in LFS_SUFFIXES if suffix not in text]
    assert not missing, f".gitattributes no longer routes {missing} through LFS"


def test_no_tracked_binary_over_half_a_megabyte_outside_lfs():
    """The size floor: anything big that is not text must already be a pointer.

    README.md is deliberately exempt - half a megabyte of prose is the document, not
    a binary the attributes were written for.
    """
    _require_git_checkout()
    lfs = _lfs_files()
    tracked = (_git("ls-files") or "").splitlines()
    offenders = []
    for path in tracked:
        local = REPO / path
        if path == "README.md" or not local.is_file():
            continue
        if local.stat().st_size <= 512 * 1024:
            continue
        if path in lfs:
            continue
        offenders.append(f"{path} ({local.stat().st_size / (1024 * 1024):.2f} MB)")
    assert not offenders, (
        "tracked files over 512 KB that are neither LFS pointers nor exempt text: "
        + ", ".join(sorted(offenders))
    )
