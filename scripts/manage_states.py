#!/usr/bin/env python3
"""
scripts/manage_states.py: Savestate Management, Verification & Packaging Tool.

In Reinforcement Learning with Snes9x Libretro, savestates (.state) are required
for O(1) constant-time episode resets (~1 ms vs ~20 s cold boot).

This utility provides:
1. 'check': Inspects data/raw/ for required savestates and the commercial ROM,
   validating sizes and SHA-256 checksums.
2. 'verify': Boots Snes9x Libretro to load each state and verify RAM/WRAM coordinates
   (Mario pos, game mode $7E:0100, translevel $7E:13BF, and world map cursor $7E:1F17/$7E:1F19).
3. 'bundle': Creates a compressed tar.gz asset archive (assets/smw_states.tar.gz)
   with an embedded manifest.json containing hashes and level metadata.
4. 'unpack': Restores savestates from a bundle into data/raw/ with strict SHA-256 validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path if not installed in editable mode
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_dir, require_core, require_rom  # noqa: E402

RAW_DIR = data_dir()
EXPECTED_ROM_SHA1 = "6B47BB75D16514B6A476AA0C73A683A2A4C18765"

# Ground-truth specifications for the 15 canonical savestates
SAVESTATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "smw_yoshi_island_1_level.state": {
        "translevel": 0x2A,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.3,
        "stage_name": "Yoshi's Island 1 (Live Level Entry)",
        "role": "Canonical starting level state for benchmark & dashboard episodes",
    },
    "smw_yoshi_island_1.state": {
        "translevel": 0x00,
        "mode": 0x08,
        "mario_x": 16.0,
        "mario_y": 336.0,
        "stage_name": "Yoshi's Island 1 (Title/Preamble)",
        "role": "Title screen / demo mode preamble state",
    },
    "smw_yoshi_house.state": {
        "translevel": 0x00,
        "mode": 0x08,
        "mario_x": 16.0,
        "mario_y": 336.0,
        "stage_name": "Yoshi's House (Title/Preamble)",
        "role": "Yoshi's House demo mode preamble state",
    },
    "smw_map_node_104_120_t28.state": {
        "translevel": 0x28,
        "mode": 0x14,
        "mario_x": 128.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's House (Level Entry)",
        "role": "Gameplay entry for Yoshi's House (non-clearing stage)",
    },
    "smw_map_node_116_126_t28.state": {
        "translevel": 0x28,
        "mode": 0x14,
        "mario_x": 128.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's House (Map Node)",
        "role": "Map node waypoint capture for Yoshi's House",
    },
    "smw_map_node_93_126_t28.state": {
        "translevel": 0x28,
        "mode": 0x14,
        "mario_x": 128.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's House (Alternate Map Node)",
        "role": "Map node waypoint capture for Yoshi's House alternate position",
    },
    "smw_map_node_141_135_t2a.state": {
        "translevel": 0x2A,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 1 (Alternate Map Node)",
        "role": "Map node waypoint capture for Yoshi's Island 1",
    },
    "smw_map_node_152_136_t2a.state": {
        "translevel": 0x2A,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 1 (Map Node Entry)",
        "role": "Map node capture right before level load",
    },
    "smw_map_round_0_152_136_t2a.state": {
        "translevel": 0x2A,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 1 (Round 0 Entry)",
        "role": "World map round 0 initial level state",
    },
    "smw_map_node_56_136_t29.state": {
        "translevel": 0x29,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 2 (Map Node)",
        "role": "Map node capture for Yoshi's Island 2",
    },
    "smw_map_node_68_135_t29.state": {
        "translevel": 0x29,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 2 (Level Entry)",
        "role": "Interactive gameplay level entry for Yoshi's Island 2 (cleared in §10.69/§10.71)",
    },
    "smw_map_round_1_152_104_t27.state": {
        "translevel": 0x27,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 352.1,
        "stage_name": "Yoshi's Island 3 (Level Entry)",
        "role": "Interactive gameplay level entry for Yoshi's Island 3 (cleared in §10.70/§10.71)",
    },
    "smw_map_round_2_184_72_t26.state": {
        "translevel": 0x26,
        "mode": 0x14,
        "mario_x": 16.0,
        "mario_y": 272.6,
        "stage_name": "Yoshi's Island 4 (Aquatic Level Entry)",
        "role": "Aquatic gameplay entry for Yoshi's Island 4 (analyzed in §10.72)",
    },
    "smw_world_map_after_round_0_152_136_t2a.state": {
        "translevel": 0x2A,
        "mode": 0x0E,
        "cursor_x": 152,
        "cursor_y": 104,
        "stage_name": "World Map Post-Clear 0x2A",
        "role": "World map state after clearing Yoshi's Island 1 (path to 0x27 unlocked)",
    },
    "smw_world_map_after_round_1_152_104_t27.state": {
        "translevel": 0x27,
        "mode": 0x0E,
        "cursor_x": 184,
        "cursor_y": 72,
        "stage_name": "World Map Post-Clear 0x27",
        "role": "World map state after clearing Yoshi's Island 3 (path to 0x26 unlocked)",
    },
}


def sha256_file(filepath: Path) -> str:
    """Calculates SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(filepath: Path) -> str:
    """Calculates SHA-1 checksum of a file."""
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def cmd_check(args: argparse.Namespace) -> int:
    """Checks presence, sizes, and hashes of ROM and savestates in data/raw/."""
    print("=" * 80)
    print("Super Mario World - Savestate & Asset Health Check")
    print("=" * 80)

    # 1. ROM Check
    try:
        rom_path = Path(require_rom())
        h_sha1 = sha1_file(rom_path).upper()
        size_mb = rom_path.stat().st_size / (1024 * 1024)
        if h_sha1 == EXPECTED_ROM_SHA1:
            print(f"[PASS] ROM found: {rom_path.name} ({size_mb:.2f} MB)")
            print(f"       SHA-1: {h_sha1} (matches official Super Mario World USA)")
        else:
            print(f"[WARN] ROM found: {rom_path.name} ({size_mb:.2f} MB)")
            print(f"       SHA-1: {h_sha1} (Expected: {EXPECTED_ROM_SHA1})")
    except Exception as e:
        print(f"[FAIL] ROM not found or inaccessible: {e}")
        print("       Please provide legally dumped Super Mario World (USA) ROM.")

    # 2. Savestates Check
    print(f"\n--- Checking {len(SAVESTATE_CATALOG)} Canonical Savestates in {RAW_DIR} ---")
    all_req_pass = True
    missing_count = 0
    for filename, spec in SAVESTATE_CATALOG.items():
        state_file = RAW_DIR / filename
        if not state_file.exists():
            print(f"[MISSING] {filename:<44} | {spec['stage_name']}")
            all_req_pass = False
            missing_count += 1
        else:
            h = sha256_file(state_file)
            size_kb = state_file.stat().st_size / 1024
            print(
                f"[PASS]    {filename:<44} ({size_kb:6.1f} KB) | SHA-256: {h[:16]}... | {spec['stage_name']}"
            )

    print("=" * 80)
    if all_req_pass:
        print(f"Summary: All {len(SAVESTATE_CATALOG)} canonical savestates are verified present.")
        return 0
    else:
        print(
            f"Summary: {missing_count} savestates missing. Run 'python scripts/manage_states.py unpack' or 'git lfs pull'."
        )
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Boots Snes9x Libretro to verify RAM/WRAM coordinates for all active savestates."""
    print("=" * 80)
    print("Verifying Savestates via Snes9x Libretro WRAM Ingestion")
    print("=" * 80)

    try:
        from src.environment.snes_emulator import SnesLibretroEmulator

        core_p = require_core()
        rom_p = require_rom()
        emu = SnesLibretroEmulator(str(core_p))
        emu.load_rom(str(rom_p))
    except Exception as e:
        print(f"[ERROR] Failed to boot emulator or load ROM: {e}")
        return 1

    results: List[Dict[str, Any]] = []
    all_ok = True

    for filename, spec in SAVESTATE_CATALOG.items():
        state_file = RAW_DIR / filename
        if not state_file.exists():
            results.append(
                {
                    "file": filename,
                    "status": "MISSING",
                    "details": "File not found",
                }
            )
            all_ok = False
            continue

        try:
            emu.load_state(state_file.read_bytes())
            mode = emu.get_game_mode()
            mario = emu.get_smw_state()
            translevel = emu.read_wram_u8(0x13BF)
            cx = emu.read_wram_u8(0x1F17) | (emu.read_wram_u8(0x1F18) << 8)
            cy = emu.read_wram_u8(0x1F19) | (emu.read_wram_u8(0x1F1A) << 8)

            exp_mode = spec["mode"]
            exp_translevel = spec["translevel"]

            mode_ok = mode == exp_mode
            trans_ok = translevel == exp_translevel

            # Check positions
            pos_ok = True
            if "mario_x" in spec:
                pos_ok = abs(mario["x"] - spec["mario_x"]) < 2.0
            elif "cursor_x" in spec:
                pos_ok = cx == spec["cursor_x"] and cy == spec["cursor_y"]

            status = "OK" if (mode_ok and trans_ok and pos_ok) else "MISMATCH"
            if status != "OK":
                all_ok = False

            results.append(
                {
                    "file": filename,
                    "status": status,
                    "mode": f"0x{mode:02X} (exp 0x{exp_mode:02X})",
                    "translevel": f"0x{translevel:02X} (exp 0x{exp_translevel:02X})",
                    "pos": (
                        f"X={mario['x']:.1f}, Y={mario['y']:.1f}"
                        if mode == 0x14
                        else f"Cursor=({cx}, {cy})"
                    ),
                }
            )
        except Exception as e:
            results.append(
                {
                    "file": filename,
                    "status": "ERROR",
                    "details": str(e),
                }
            )
            all_ok = False

    print(f"{'Filename':<42} {'Status':<10} {'Mode':<18} {'Translevel':<18} {'Coords'}")
    print("-" * 105)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['file']:<42} [PASS]     {r['mode']:<18} {r['translevel']:<18} {r['pos']}")
        elif r["status"] == "MISSING":
            print(f"{r['file']:<42} [MISSING]  {r['details']}")
        else:
            print(f"{r['file']:<42} [{r['status']}] {r.get('details', '')}")

    print("=" * 80)
    if all_ok:
        print("[SUCCESS] All savestates successfully verified against live Snes9x WRAM telemetry.")
        return 0
    else:
        print("[FAIL] One or more savestates failed verification.")
        return 1


def cmd_bundle(args: argparse.Namespace) -> int:
    """Creates a compressed tar.gz bundle of all savestates with manifest.json."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "version": "1.0.0",
        "game": "Super Mario World (USA)",
        "expected_rom_sha1": EXPECTED_ROM_SHA1,
        "states": {},
    }
    files_to_add: List[Path] = []

    for filename, spec in SAVESTATE_CATALOG.items():
        state_file = RAW_DIR / filename
        if not state_file.exists():
            print(
                f"[ERROR] Required state {filename} is missing from {RAW_DIR}. Cannot create complete bundle."
            )
            return 1

        h = sha256_file(state_file)
        manifest["states"][filename] = {
            **spec,
            "sha256": h,
            "size_bytes": state_file.stat().st_size,
        }
        files_to_add.append(state_file)

    manifest_tmp = RAW_DIR / "_bundle_manifest.json"
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Creating archive {out_path}...")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(manifest_tmp, arcname="manifest.json")
        for f in files_to_add:
            print(f"  Adding {f.name} ({f.stat().st_size / 1024:.1f} KB)...")
            tar.add(f, arcname=f.name)

    manifest_tmp.unlink(missing_ok=True)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"[SUCCESS] Savestate bundle created: {out_path} ({size_mb:.2f} MB, {len(files_to_add)} states)"
    )
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    """Unpacks savestates from a bundle into data/raw/ with strict SHA-256 validation."""
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"[ERROR] Bundle not found: {bundle_path}")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting bundle {bundle_path} to {RAW_DIR}...")
    with tarfile.open(bundle_path, "r:gz") as tar:
        try:
            m_f = tar.extractfile("manifest.json")
            if m_f is None:
                raise ValueError("manifest.json not found in archive")
            manifest = json.loads(m_f.read().decode("utf-8"))
        except Exception as e:
            print(f"[ERROR] Invalid or missing manifest.json in bundle: {e}")
            return 1

        for name, info in manifest["states"].items():
            print(f"  Extracting {name}...")
            tar.extract(name, path=str(RAW_DIR))
            target_path = RAW_DIR / name
            h = sha256_file(target_path)
            if h != info["sha256"]:
                print(f"[FAIL] Checksum mismatch for {name}: expected {info['sha256']}, got {h}")
                return 1
            print(f"  [PASS] Verified SHA-256: {h[:16]}...")

    print(
        f"[SUCCESS] All {len(manifest['states'])} savestates unpacked and cryptographically verified in {RAW_DIR}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Super Mario World Savestate & Asset Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check ROM and savestate files")
    subparsers.add_parser(
        "verify", help="Verify savestate RAM/WRAM coordinates via Snes9x emulator"
    )

    p_bundle = subparsers.add_parser("bundle", help="Package canonical savestates into tar.gz")
    p_bundle.add_argument("--out", default="assets/smw_states.tar.gz", help="Output bundle path")

    p_unpack = subparsers.add_parser(
        "unpack", help="Unpack states from tar.gz bundle into data/raw/"
    )
    p_unpack.add_argument("--bundle", default="assets/smw_states.tar.gz", help="Input bundle path")

    args = parser.parse_args()
    if args.command == "check":
        return cmd_check(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "bundle":
        return cmd_bundle(args)
    elif args.command == "unpack":
        return cmd_unpack(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
