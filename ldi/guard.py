"""Memory and disk checks. A failed check leaves the model unloaded."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ldi.config import (
    LOADABLE_PRESSURE,
    PRESSURE_CRITICAL,
    PRESSURE_NORMAL,
    PRESSURE_URGENT,
    PRESSURE_WARNING,
    RESERVE_BYTES,
    SMALL_MODEL_METAL_CEILING,
    ModelSpec,
    gb,
)


class GuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class Machine:
    pressure: int | None
    reclaimable: int | None
    disk_free: int
    on_battery: bool | None
    swap_used: int | None


def pressure_label(level: int | None) -> str:
    if level == PRESSURE_NORMAL:
        return "green"
    if level == PRESSURE_WARNING:
        return "yellow"
    if level in (PRESSURE_URGENT, PRESSURE_CRITICAL):
        return "red"
    if level is None:
        return "unknown"
    return "red"


def parse_vm_stat(text: str) -> tuple[int, dict[str, int]]:
    page_match = re.search(r"page size of (\d+) bytes", text)
    page = int(page_match.group(1)) if page_match else 16384
    pages: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        digits = re.sub(r"\D", "", raw)
        if digits:
            pages[key.strip()] = int(digits)
    return page, pages


def reclaimable_from(page_size: int, pages: dict[str, int]) -> int:
    # Purgeable pages are already inside the other queues, so they are not added.
    names = ("Pages free", "Pages inactive", "Pages speculative")
    return page_size * sum(pages.get(name, 0) for name in names)


def parse_swap_used(text: str) -> int | None:
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if match is None:
        return None
    value = float(match.group(1))
    scale = 1024**3 if match.group(2) == "G" else 1024**2
    return int(value * scale)


def parse_on_battery(text: str) -> bool | None:
    if "Battery Power" in text:
        return True
    if "AC Power" in text:
        return False
    return None


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_snapshot() -> Machine:
    pressure_text = _command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    pressure = None
    if pressure_text is not None:
        try:
            pressure = int(pressure_text.strip())
        except ValueError:
            pressure = None

    reclaimable = None
    vm_text = _command(["vm_stat"])
    if vm_text is not None:
        page, pages = parse_vm_stat(vm_text)
        reclaimable = reclaimable_from(page, pages)

    swap_text = _command(["sysctl", "vm.swapusage"])
    battery_text = _command(["pmset", "-g", "batt"])
    return Machine(
        pressure=pressure,
        reclaimable=reclaimable,
        disk_free=shutil.disk_usage(Path.home()).free,
        on_battery=parse_on_battery(battery_text or ""),
        swap_used=parse_swap_used(swap_text or ""),
    )


def load_block_reason(
    spec: ModelSpec,
    pressure: int | None,
    reclaimable: int | None,
    swap_used: int | None = None,
) -> str | None:
    if (
        swap_used is not None
        and swap_used >= 1024**3
        and spec.metal_limit_bytes > SMALL_MODEL_METAL_CEILING
    ):
        return (
            f"Swap is using {gb(swap_used)}. "
            "A restart clears a long session before the first model load."
        )
    if pressure not in LOADABLE_PRESSURE:
        if pressure is None:
            return (
                "Memory pressure could not be read, so the model stays unloaded. "
                "Run ldi status in Terminal on this Mac."
            )
        return (
            "Memory pressure is red, so the model stays unloaded. "
            "Quit other apps or restart, then run ldi status."
        )
    if reclaimable is None:
        return "Free memory could not be read, so the model stays unloaded."
    if reclaimable < spec.metal_limit_bytes:
        if spec.metal_limit_bytes <= SMALL_MODEL_METAL_CEILING:
            next_step = "Quit other apps, then run ldi status."
        elif spec.key == "7b":
            next_step = (
                "The 3B model needs less free memory. "
                "Restart and quit other apps before the 7B model."
            )
        else:
            next_step = "Restart and quit other apps, then run ldi status."
        return (
            f"The {spec.key} model caps Metal at {gb(spec.metal_limit_bytes)}. "
            f"About {gb(reclaimable)} is reclaimable right now. {next_step}"
        )
    return None


def assess_load(
    spec: ModelSpec,
    pressure: int | None,
    reclaimable: int | None,
    swap_used: int | None = None,
) -> None:
    reason = load_block_reason(spec, pressure, reclaimable, swap_used)
    if reason is not None:
        raise GuardError(reason)


def download_block_reason(approx_bytes: int, free_bytes: int) -> str | None:
    remaining = free_bytes - approx_bytes
    if remaining >= RESERVE_BYTES:
        return None
    return (
        f"Disk free is {gb(free_bytes)}. This download is about {gb(approx_bytes)}. "
        f"About {gb(max(remaining, 0))} would remain afterward. "
        f"Keep at least {gb(RESERVE_BYTES)} free."
    )


def assess_download(approx_bytes: int, free_bytes: int) -> None:
    reason = download_block_reason(approx_bytes, free_bytes)
    if reason is not None:
        raise GuardError(reason)


def assess_setup(pressure: int | None) -> None:
    if pressure is None or pressure in LOADABLE_PRESSURE:
        return
    raise GuardError(
        "Memory pressure is red. Quit other apps or restart, then run ldi setup."
    )
