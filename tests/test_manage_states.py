"""
test_manage_states.py
Unit and integration tests for savestate management, verification, and packaging.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

from scripts.manage_states import (
    SAVESTATE_CATALOG,
    cmd_check,
    cmd_unpack,
    cmd_verify,
)
from tests.conftest import requires_emulator


def test_catalog_has_all_canonical_savestates() -> None:
    assert len(SAVESTATE_CATALOG) == 15
    for filename, spec in SAVESTATE_CATALOG.items():
        assert filename.endswith(".state")
        assert "translevel" in spec
        assert "mode" in spec
        assert "stage_name" in spec
        assert "role" in spec


def test_check_command_succeeds_on_canonical_states() -> None:
    args = argparse.Namespace()
    ret = cmd_check(args)
    assert ret == 0


def test_bundle_archive_integrity() -> None:
    bundle_path = Path("assets/smw_states.tar.gz")
    assert bundle_path.exists(), "assets/smw_states.tar.gz does not exist"
    assert bundle_path.stat().st_size > 500_000

    with tarfile.open(bundle_path, "r:gz") as tar:
        names = tar.getnames()
        assert "manifest.json" in names
        m_f = tar.extractfile("manifest.json")
        assert m_f is not None
        manifest = json.loads(m_f.read().decode("utf-8"))
        assert "states" in manifest
        assert len(manifest["states"]) == 15

        for filename in SAVESTATE_CATALOG:
            assert filename in names
            assert filename in manifest["states"]


def test_unpack_command_in_isolated_tempdir() -> None:
    import scripts.manage_states as ms

    original_raw = ms.RAW_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            ms.RAW_DIR = temp_dir
            args = argparse.Namespace(bundle="assets/smw_states.tar.gz")
            ret = cmd_unpack(args)
            assert ret == 0
            extracted = list(temp_dir.glob("*.state"))
            assert len(extracted) == 15
            for f in extracted:
                assert f.name in SAVESTATE_CATALOG
    finally:
        ms.RAW_DIR = original_raw


@requires_emulator
def test_verify_command_with_emulator() -> None:
    args = argparse.Namespace()
    ret = cmd_verify(args)
    assert ret == 0
