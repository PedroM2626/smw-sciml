"""Shared pytest fixtures and platform guards.

The Libretro core shipped in ``src/environment/bin`` is a Windows DLL, so any
test that boots the real emulator can only run on Windows with the ROM +
savestates present (CI on ubuntu-latest must skip them instead of erroring).
"""

from __future__ import annotations

import ctypes
import os

import pytest

from src.utils.paths import CORE_PATH, ROM_PATH
from src.utils.paths import STATE_YOSHI_ISLAND_1 as STATE_PATH


def emulator_available(
    core_path: str = CORE_PATH,
    rom_path: str = ROM_PATH,
    state_path: str = STATE_PATH,
) -> bool:
    """True only if the native core exists *and* loads in this process."""
    if not (os.path.exists(core_path) and os.path.exists(rom_path) and os.path.exists(state_path)):
        return False
    try:
        ctypes.CDLL(core_path)
    except OSError:
        return False
    return True


requires_emulator = pytest.mark.skipif(
    not emulator_available(),
    reason="Real SNES emulator core not loadable on this platform (needs Windows + ROM).",
)


def missing_emulator_is_an_error(environ: dict[str, str], available: bool) -> str | None:
    """The reason to stop the run, when the environment demands the emulator and it is absent.

    The hosted CI runners have no ROM, so every emulator test skips there and a green run says
    nothing about the closed-loop code. ``SMW_REQUIRE_EMULATOR=1`` is set by the hardware
    workflow, which runs on a machine that does hold one: there a skip would hide a broken setup,
    so it becomes an error instead.
    """
    if environ.get("SMW_REQUIRE_EMULATOR") == "1" and not available:
        return (
            "SMW_REQUIRE_EMULATOR=1 but the emulator is not usable here: check SMW_ROM, SMW_CORE "
            "and the savestates (`python -m src.cli install-info`)."
        )
    return None


def pytest_configure(config: pytest.Config) -> None:
    reason = missing_emulator_is_an_error(dict(os.environ), emulator_available())
    if reason:
        raise pytest.UsageError(reason)
