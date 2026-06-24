"""Background GPU power sampling for profiler DVFS shots."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def profiler_gpu_power_enabled() -> bool:
    raw = os.environ.get("PROFILER_GPU_POWER", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _poll_interval_sec() -> float:
    return max(0.02, float(os.environ.get("PROFILER_GPU_POWER_INTERVAL_MS", "100")) / 1000.0)


def _query_gpus() -> list[dict[str, Any]]:
    allowed: set[int] | None = None
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        picked = {int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()}
        if picked:
            allowed = picked
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,power.draw,clocks.current.graphics,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return [{"error": repr(exc)}]

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            if allowed is not None and idx not in allowed:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "power_w": float(parts[1]),
                    "graphics_mhz": int(float(parts[2])),
                    "util_gpu_pct": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


class GpuPowerSampler:
    """Append JSONL power samples tagged with wall-clock epoch seconds."""

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._loop,
            name="profiler-gpu-power",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _loop(self) -> None:
        with self.out_path.open("a", encoding="utf-8") as fh:
            while not self._stop.is_set():
                rec = {
                    "wall_ts": time.time(),
                    "gpus": _query_gpus(),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                self._stop.wait(_poll_interval_sec())


def load_power_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _total_power_w(gpus: list[dict[str, Any]]) -> float | None:
    vals = [float(g["power_w"]) for g in gpus if "power_w" in g]
    if not vals:
        return None
    return sum(vals)


def _interval_overlap(t0: float, t1: float, exclude: list[tuple[float, float]]) -> float:
    overlap = 0.0
    for start, end in exclude:
        overlap += max(0.0, min(t1, end) - max(t0, start))
    return overlap


def integrate_power_joules(
    samples: list[dict[str, Any]],
    exclude_intervals: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Trapezoidal integration of summed GPU power; optionally exclude intervals."""
    exclude = exclude_intervals or []
    if len(samples) < 2:
        return {
            "energy_j": 0.0,
            "energy_excl_pause_j": 0.0,
            "duration_sec": 0.0,
            "duration_excl_pause_sec": 0.0,
            "sample_count": len(samples),
            "mean_power_w": None,
            "mean_power_excl_pause_w": None,
        }

    energy_j = 0.0
    energy_excl_j = 0.0
    duration_sec = 0.0
    duration_excl_sec = 0.0

    for i in range(1, len(samples)):
        t0 = float(samples[i - 1]["wall_ts"])
        t1 = float(samples[i]["wall_ts"])
        dt = t1 - t0
        if dt <= 0:
            continue
        p0 = _total_power_w(samples[i - 1].get("gpus") or [])
        p1 = _total_power_w(samples[i].get("gpus") or [])
        if p0 is None or p1 is None:
            continue
        p_mid = 0.5 * (p0 + p1)
        dt_excl = dt - _interval_overlap(t0, t1, exclude)
        dt_excl = max(0.0, dt_excl)
        energy_j += p_mid * dt
        energy_excl_j += p_mid * dt_excl
        duration_sec += dt
        duration_excl_sec += dt_excl

    mean_w = energy_j / duration_sec if duration_sec > 0 else None
    mean_excl_w = energy_excl_j / duration_excl_sec if duration_excl_sec > 0 else None
    return {
        "energy_j": round(energy_j, 3),
        "energy_excl_pause_j": round(energy_excl_j, 3),
        "duration_sec": round(duration_sec, 6),
        "duration_excl_pause_sec": round(duration_excl_sec, 6),
        "sample_count": len(samples),
        "mean_power_w": round(mean_w, 3) if mean_w is not None else None,
        "mean_power_excl_pause_w": round(mean_excl_w, 3) if mean_excl_w is not None else None,
    }
