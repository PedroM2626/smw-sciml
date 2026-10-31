[← 2](02-introduction-and-scientific-motivation.md) · [Contents](../../README.md#contents) · [4 →](04-mathematical-formulation-of-super-mario-world-physics.md)

## 3. SNES Hardware Architecture & WRAM Reverse Engineering

### 3.1 The Ricoh 5A22 CPU and Execution Cycle
The SNES is powered by the Ricoh 5A22 microprocessor (based on the 16-bit WDC 65C816 core), operating at dynamic clock frequencies between 2.68 MHz and 3.58 MHz. The NTSC video subsystem refreshes the display at a nominal rate of 59.94 Hz (~60 frames per second). On every frame interval ($\Delta t \approx 16.67\text{ ms}$):
1. The Vertical Blanking routine (*V-Blank*) transfers graphical assets to VRAM;
2. CPU routines sample controller input registers via serial I/O latches;
3. The game engine computes player accelerations, updates subpixel accumulators, applies ground friction, resolves collision response against tile matrices, and commits the updated player state to static Working RAM (**WRAM**).

The WRAM comprises 128 Kilobytes of memory, mapped across addresses `$7E:0000` to `$7F:FFFF`.

### 3.2 High-Throughput Headless Interface via Libretro Ctypes
To guarantee deterministic execution and high simulation velocity, we developed a native Python `ctypes` wrapper for the official 64-bit `snes9x_libretro.dll` core:
- Direct memory mapping via `retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM = 2)`;
- Synchronous clock cycling via `retro_run()`, exceeding **2,700 frames per second** in headless execution;
- In-memory atomic state serialization (`retro_serialize` / `retro_unserialize`) for instantaneous episode reset without ROM reloading overhead.

### 3.3 Complete WRAM Register Mapping for Player Physics
Through reverse-engineering and symbol table disassembly of the *Super Mario World* ROM, we identified the following critical state addresses:

| WRAM Address | Format / Type | Symbol Name | Functional Description in Game Engine |
| :--- | :--- | :--- | :--- |
| `$7E:0094` - `$7E:0095` | 16-bit uint (Little-Endian) | `Mario_X_Pos` | Player horizontal integer position in phase coordinates ($X_{\text{pix}}$). |
| `$7E:0096` - `$7E:0097` | 16-bit uint (Little-Endian) | `Mario_Y_Pos` | Player vertical integer position in phase coordinates ($Y_{\text{pix}}$). Increases downward. |
| `$7E:13DA` | 8-bit unsigned char | `Mario_X_Sub` | Horizontal subpixel register ($sx \in [0, 255]$). High nibble represents 16 subpixels per pixel. |
| `$7E:13DC` | 8-bit unsigned char | `Mario_Y_Sub` | Vertical subpixel register ($sy \in [0, 255]$). |
| `$7E:007B` | 8-bit signed (Two's Complement) | `Mario_X_Speed` | Instantaneous horizontal velocity in subpixels per frame ($v_x$). Positive = right. |
| `$7E:007D` | 8-bit signed (Two's Complement) | `Mario_Y_Speed` | Instantaneous vertical velocity in subpixels per frame ($v_y$). Positive = fall. |
| `$7E:0077` | 8-bit bitmask | `Mario_Blocked_Status` | Terrain collision flags: Bit 0 = Right Wall; Bit 1 = Left Wall; Bit 2 = Floor/Ground; Bit 3 = Ceiling. |
| `$7E:0072` | 8-bit enum | `Mario_Air_State` | Airborne state: `0` = Solid Ground; `1` = Falling; `2` = Ascending Jump. |
| `$7E:0071` | 8-bit enum | `Mario_Animation_State` | Player animation state: `0` = Normal, `9` = Death sequence, etc. |
| `$7E:0019` | 8-bit enum | `Mario_Powerup` | Powerup tier: `0` = Small, `1` = Super Mario, `2` = Cape, `3` = Fire Flower. |
| `$7E:0015` | 8-bit bitmask | `Controller_Hold` | Held buttons bitmask: format `BYsSUDLR`. |
| `$7E:0016` | 8-bit bitmask | `Controller_Press` | Newly pressed button triggers in current frame (*trigger edge*). |
| `$7E:0017` | 8-bit bitmask | `Controller_Hold_2` | Secondary buttons: `AXLR----`. |
| `$7E:001A` - `$7E:001B` | 16-bit uint (Little-Endian) | `Layer1_Scroll_X` | Horizontal camera in pixels; Mario's screen position is `x - camera`. Identified by behaviour rather than disassembly: 10.59's scan ranks every 16-bit word in WRAM against Mario's x and keeps the ones satisfying the axioms of a scroll, which leaves this address, its mirror at `$7E:1462`, and `$7E:001E` holding exactly half of it (the parallax layer). See `results/scroll_address_scan_metrics.json`. |
| `$7E:0100` | 8-bit enum | `Game_Mode` | Engine execution mode: `0x07` = Title Demo; `0x14` = Interactive In-Level Gameplay. |

---
