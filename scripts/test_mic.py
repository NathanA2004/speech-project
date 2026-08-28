"""Confirm real-time PCM capture with no dropped frames.

Usage (from repo root, after `pip install -r requirements.txt`):

    python scripts/test_mic.py
    python scripts/test_mic.py --seconds 8 --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.table import Table  # noqa: E402

from audio.stream import AudioStreamManager, frame_rms  # noqa: E402
from config import (  # noqa: E402
    CHANNELS,
    DTYPE,
    FRAME_DURATION_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
)


def _rms_bar(rms: float, width: int = 40) -> str:
    filled = min(width, int(rms * width * 8))  # speech RMS is typically << 1
    return "#" * filled + "-" * (width - filled)


def _build_table(
    elapsed: float,
    seconds: float,
    rms: float,
    stats,
    last_peak: float,
) -> Table:
    table = Table(title="Local PCM capture (16 kHz / 16-bit / mono / 30 ms)")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("elapsed", f"{elapsed:.2f} s / {seconds:.1f} s")
    table.add_row("RMS", f"{rms:.4f}  peak={last_peak:.4f}")
    table.add_row("level", _rms_bar(rms))
    table.add_row("frames captured", str(stats.frames_captured))
    table.add_row("expected frames", str(stats.expected_frames))
    table.add_row("PortAudio overflows", str(stats.overflow_count))
    table.add_row("queue drops", str(stats.queue_drop_count))
    table.add_row("timing gap", str(stats.timing_gap))
    return table


def list_devices() -> None:
    console = Console()
    console.print("[bold]Input devices[/bold]")
    console.print(sd.query_devices())


def run(seconds: float, device: int | str | None) -> int:
    console = Console()
    console.print(
        f"Capturing {SAMPLE_RATE} Hz, {CHANNELS} ch, {DTYPE}, "
        f"{FRAME_DURATION_MS} ms frames ({FRAME_SAMPLES} samples) for {seconds:.1f}s"
    )

    last_peak = 0.0
    with AudioStreamManager(device=device) as stream:
        t_end = time.perf_counter() + seconds
        with Live(console=console, refresh_per_second=20) as live:
            for pcm in stream.iter_frames(timeout=0.5):
                rms = frame_rms(pcm)
                last_peak = max(last_peak, rms)
                stats = stream.stats()
                live.update(
                    _build_table(
                        elapsed=stats.elapsed_s,
                        seconds=seconds,
                        rms=rms,
                        stats=stats,
                        last_peak=last_peak,
                    )
                )
                if time.perf_counter() >= t_end:
                    break

        stats = stream.stats()
        ring = stream.snapshot_ring_buffer()

    console.print()
    console.print("[bold]Result[/bold]")
    console.print(f"  frames captured : {stats.frames_captured}")
    console.print(f"  expected frames : {stats.expected_frames}")
    console.print(f"  overflows       : {stats.overflow_count}")
    console.print(f"  queue drops     : {stats.queue_drop_count}")
    console.print(f"  timing gap      : {stats.timing_gap}")
    console.print(f"  ring buffer     : {ring.size} samples ({ring.size / SAMPLE_RATE:.2f}s)")
    console.print(f"  dtype           : {ring.dtype if ring.size else np.dtype(np.int16)}")
    console.print(f"  peak RMS        : {last_peak:.4f}")

    # Allow ±2 frames of scheduling jitter; any PortAudio overflow or queue
    # drop is a hard failure (those are real lost samples).
    jitter_ok = stats.timing_gap <= 2
    no_drops = stats.dropped_frames == 0
    enough = stats.frames_captured >= int(seconds * 1000 / FRAME_DURATION_MS) - 2

    if no_drops and jitter_ok and enough:
        console.print("[green]PASS[/green] - real-time PCM capture with no dropped frames")
        return 0

    console.print("[red]FAIL[/red] - dropped frames or capture shortfall")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Test real-time local mic capture")
    parser.add_argument("--seconds", type=float, default=5.0, help="capture duration")
    parser.add_argument("--device", type=str, default=None, help="input device index or name")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    device: int | str | None = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    try:
        return run(seconds=args.seconds, device=device)
    except Exception as exc:  # noqa: BLE001 — show a clear mic/device error
        Console().print(f"[red]Capture failed:[/red] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
