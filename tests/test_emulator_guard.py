"""Tests for the emulator platform guard (tests/conftest.py).

These are the tests that keep CI green on Linux: emulator-dependent tests
must skip instead of erroring when the Windows Libretro core cannot load.
"""

import ctypes

from tests.conftest import emulator_available, requires_emulator


@requires_emulator
def test_guard_passes_on_capable_platform():
    # Only runs where the native core loads (Windows + ROM + DLL); on Linux
    # CI this skips by design instead of failing.
    assert emulator_available() is True


def test_guard_fails_on_missing_files(tmp_path):
    assert (
        emulator_available(
            core_path=str(tmp_path / "missing.dll"),
            rom_path=str(tmp_path / "missing.sfc"),
            state_path=str(tmp_path / "missing.state"),
        )
        is False
    )


def test_guard_fails_when_cdll_raises(monkeypatch):
    def _boom(path):
        raise OSError("cannot load library")

    monkeypatch.setattr(ctypes, "CDLL", _boom)
    # Paths exist here, so only the CDLL failure drives the result.
    assert emulator_available() is False


def test_a_required_emulator_that_is_missing_stops_the_run() -> None:
    """SMW_REQUIRE_EMULATOR=1 turns the silent skip into an error; unset, nothing changes."""
    from tests.conftest import missing_emulator_is_an_error

    assert missing_emulator_is_an_error({"SMW_REQUIRE_EMULATOR": "1"}, available=False)
    assert missing_emulator_is_an_error({"SMW_REQUIRE_EMULATOR": "1"}, available=True) is None
    assert missing_emulator_is_an_error({}, available=False) is None
    assert missing_emulator_is_an_error({"SMW_REQUIRE_EMULATOR": "0"}, available=False) is None
