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
    """Energy/power capture is MANDATORY — the profiler must never run without it.

    A latency table with blank ``mean_power_w`` is useless for the DVFS energy
    model (it happened silently for L40S/H100 and forced a fallback to the
    gap-free microbench). So this is now always on: the ``PROFILER_GPU_POWER``
    env var is retained only to *reject* attempts to disable it — setting it to
    0/false/off raises rather than silently turning capture off.
    """
    raw = os.environ.get("PROFILER_GPU_POWER", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        raise RuntimeError(
            "PROFILER_GPU_POWER is disabled but energy/power capture is MANDATORY. "
            "Remove the override — the profiler cannot produce a table without power."
        )
    return True


def _poll_interval_sec() -> float:
    # Default 25 ms (40 Hz). The old 100 ms default was too coarse for fast GPUs
    # (L40S/H100): a sustained-config kernel window there can be a few ms, so a
    # 100 ms poll caught <2 samples and integrate_power_joules returned blank
    # mean_power_w — the silent-blank-power incident. 25 ms is near the 20 ms
    # floor and gives sustained configs enough samples on every device tested.
    # Override with PROFILER_GPU_POWER_INTERVAL_MS for slower/faster hardware.
    return max(0.02, float(os.environ.get("PROFILER_GPU_POWER_INTERVAL_MS", "25")) / 1000.0)


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


def _nvml_poll_interval_sec() -> float:
    # NVML reads are in-process C calls (~50-200us), so we can sample far finer
    # than the ~14 Hz subprocess wall. Default 2 ms (500 Hz) so even a few-ms
    # sustained-kernel exec window on a fast GPU (H100/L40S) contains many
    # samples and integrate_power_joules over that window is well-defined.
    return max(0.0005, float(os.environ.get("PROFILER_GPU_POWER_INTERVAL_MS", "2")) / 1000.0)


class _NvmlSource:
    """In-process NVML power/clock/util sampler — fast enough (~kHz) to resolve
    the millisecond kernel exec windows on fast GPUs, unlike the ~14 Hz
    nvidia-smi subprocess. Falls back to subprocess if NVML is unavailable.

    Mirrors the subprocess row schema exactly:
        {"index", "power_w", "graphics_mhz", "util_gpu_pct"}
    and applies the same CUDA_VISIBLE_DEVICES filtering.
    """

    def __init__(self) -> None:
        import pynvml  # raises if unavailable → caller falls back to subprocess

        self._pynvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()

        allowed: set[int] | None = None
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            picked = {int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()}
            if picked:
                allowed = picked

        self._handles: list[tuple[int, Any]] = []
        for i in range(count):
            if allowed is not None and i not in allowed:
                continue
            self._handles.append((i, pynvml.nvmlDeviceGetHandleByIndex(i)))
        if not self._handles:
            # No visible device matched the filter — treat as unusable so the
            # caller falls back rather than silently producing empty samples.
            raise RuntimeError("NVML: no visible GPU after CUDA_VISIBLE_DEVICES filter")

        # Instantaneous power field is preferable to the (moving-average)
        # power.usage on GPUs/drivers that expose it; probe once.
        self._instant_field = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", None)

    def _power_w(self, handle: Any) -> float:
        pynvml = self._pynvml
        if self._instant_field is not None:
            try:
                vals = pynvml.nvmlDeviceGetFieldValues(handle, [self._instant_field])
                fv = vals[0]
                if getattr(fv, "nvmlReturn", 1) == 0:
                    return float(fv.value.uiVal) / 1000.0  # mW → W
            except Exception:
                pass
        return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0  # mW → W

    def query(self) -> list[dict[str, Any]]:
        pynvml = self._pynvml
        rows: list[dict[str, Any]] = []
        for idx, handle in self._handles:
            try:
                power_w = self._power_w(handle)
                graphics_mhz = int(pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS))
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception as exc:  # pragma: no cover
                return [{"error": repr(exc)}]
            rows.append({
                "index": idx,
                "power_w": power_w,
                "graphics_mhz": graphics_mhz,
                "util_gpu_pct": util,
            })
        return rows

    def close(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


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
        # Prefer the in-process NVML sampler (kHz-capable, resolves ms kernel
        # windows). Fall back to the nvidia-smi subprocess only if NVML can't
        # initialise — but energy capture itself is never optional.
        nvml: _NvmlSource | None = None
        try:
            nvml = _NvmlSource()
        except Exception:
            nvml = None

        if nvml is not None:
            interval = _nvml_poll_interval_sec()
            query = nvml.query
        else:
            interval = _poll_interval_sec()
            query = _query_gpus

        try:
            with self.out_path.open("w", encoding="utf-8") as fh:
                while not self._stop.is_set():
                    rec = {
                        "wall_ts": time.time(),
                        "gpus": query(),
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    self._stop.wait(interval)
        finally:
            if nvml is not None:
                nvml.close()


def compute_achieved_mhz(
    samples: list[dict[str, Any]],
    busy_threshold: float = 10.0,
) -> float | None:
    """Median graphics clock over under-load (busy) samples.

    Filters to samples where any GPU has util_gpu_pct >= busy_threshold,
    then returns the median graphics_mhz across all GPUs in those samples.
    Returns None if no busy samples exist.
    """
    clocks: list[float] = []
    for rec in samples:
        gpus = rec.get("gpus") or []
        row_busy = any(
            float(g.get("util_gpu_pct") or 0.0) >= busy_threshold for g in gpus
        )
        if not row_busy:
            continue
        for g in gpus:
            mhz = g.get("graphics_mhz")
            if mhz is not None:
                clocks.append(float(mhz))
    if not clocks:
        return None
    clocks_sorted = sorted(clocks)
    n = len(clocks_sorted)
    mid = n // 2
    if n % 2 == 1:
        return clocks_sorted[mid]
    return 0.5 * (clocks_sorted[mid - 1] + clocks_sorted[mid])


def compute_idle_power_w(
    samples: list[dict[str, Any]],
    busy_threshold: float = 10.0,
) -> float | None:
    """Mean total GPU power during idle (low-util) samples.

    A sample is idle when ALL GPUs have util_gpu_pct < busy_threshold.
    Returns None if no idle samples exist.
    """
    total_power: list[float] = []
    for rec in samples:
        gpus = rec.get("gpus") or []
        if not gpus:
            continue
        row_busy = any(
            float(g.get("util_gpu_pct") or 0.0) >= busy_threshold for g in gpus
        )
        if row_busy:
            continue
        vals = [float(g["power_w"]) for g in gpus if "power_w" in g]
        if vals:
            total_power.append(sum(vals))
    if not total_power:
        return None
    return round(sum(total_power) / len(total_power), 3)


def compute_power_hz(samples: list[dict[str, Any]]) -> float | None:
    """Effective power sampling rate in Hz."""
    if len(samples) < 2:
        return None
    ts = [float(s["wall_ts"]) for s in samples if "wall_ts" in s]
    if len(ts) < 2:
        return None
    duration = ts[-1] - ts[0]
    if duration <= 0:
        return None
    return round((len(ts) - 1) / duration, 3)


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
        # Return None (not 0.0) so a 1- or 0-sample shot is not mistaken
        # for a legitimate zero-energy measurement.
        return {
            "energy_j": None,
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
