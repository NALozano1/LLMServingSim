"""Bench run metrics: throughput, latency, energy, exec vs wall time."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.core.gpu_power import integrate_power_joules, load_power_samples
from bench.core.latency import extract_from_bench_dir
from profiler.core.exec_metrics import (
    load_markers,
    pause_intervals_from_markers,
    sum_marker_pause_sec,
)


def _parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def _token_totals(requests: list[dict[str, Any]]) -> dict[str, int]:
    inp = sum(int(r.get("input_toks", 0)) for r in requests)
    out = sum(int(r.get("output_toks", 0)) for r in requests)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


def build_bench_run_metrics(
    output_dir: Path,
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    barrier_wait_sec: float = 0.0,
    layer_pause: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate wall/exec timing, throughput, energy, and latency."""
    output_dir = Path(output_dir)
    meta_path = output_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    started_at = started_at or meta.get("started_at")
    finished_at = finished_at or meta.get("finished_at")
    t0 = _parse_iso(started_at)
    t1 = _parse_iso(finished_at)
    wall_sec = max(0.0, (t1 - t0)) if t0 is not None and t1 is not None else None

    markers = load_markers(output_dir / "dvfs_markers.jsonl")
    marker_pause_sec = sum_marker_pause_sec(markers)
    pause_intervals = pause_intervals_from_markers(markers)
    pause_sec = max(marker_pause_sec, float(barrier_wait_sec or 0.0))

    exec_sec = None
    if wall_sec is not None:
        exec_sec = max(0.0, wall_sec - pause_sec)

    requests_path = output_dir / "requests.jsonl"
    requests: list[dict[str, Any]] = []
    if requests_path.is_file():
        requests = [
            json.loads(line)
            for line in requests_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    tokens = _token_totals(requests)

    throughput: dict[str, Any] = {}
    if wall_sec and wall_sec > 0:
        throughput["output_tok_per_sec_wall"] = round(
            tokens["output_tokens"] / wall_sec, 3
        )
        throughput["total_tok_per_sec_wall"] = round(
            tokens["total_tokens"] / wall_sec, 3
        )
    if exec_sec is not None and exec_sec > 0:
        throughput["output_tok_per_sec_exec"] = round(
            tokens["output_tokens"] / exec_sec, 3
        )
        throughput["total_tok_per_sec_exec"] = round(
            tokens["total_tokens"] / exec_sec, 3
        )

    power_path = output_dir / "gpu_power" / "bench.jsonl"
    power_samples = load_power_samples(power_path)
    energy = integrate_power_joules(power_samples, exclude_intervals=pause_intervals)

    latency: dict[str, Any] = {}
    try:
        latency = extract_from_bench_dir(output_dir)
    except (FileNotFoundError, ValueError):
        latency = {}

    dvfs_modes = sorted({m.get("mode") for m in markers if m.get("mode")})
    freqs = [m.get("freq_mhz") for m in markers if m.get("freq_mhz") is not None]

    return {
        "output_dir": str(output_dir),
        "model": meta.get("engine_kwargs", {}).get("model"),
        "num_requests": len(requests),
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_sec": round(wall_sec, 6) if wall_sec is not None else None,
            "pause_sec": round(pause_sec, 6),
            "exec_sec": round(exec_sec, 6) if exec_sec is not None else None,
            "barrier_wait_sec_worker": round(float(barrier_wait_sec or 0.0), 6),
            "marker_pause_sec": marker_pause_sec,
            "barrier_count": len(markers),
        },
        "throughput": throughput,
        "tokens": tokens,
        "energy": energy,
        "latency": latency,
        "layer_pause": layer_pause or {},
        "dvfs": {
            "modes": dvfs_modes,
            "freq_mhz_observed": freqs[:20],
            "marker_count": len(markers),
        },
        "prefill_only": bool(
            (layer_pause or {}).get("prefill_only")
            or (meta.get("engine_kwargs") or {}).get("prefill_only")
        ),
        "decode_max_pauses_per_pass": (
            (meta.get("engine_kwargs") or {}).get("decode_max_pauses_per_pass")
        ),
        "artifacts": {
            "meta": str(meta_path),
            "requests": str(requests_path),
            "markers": str(output_dir / "dvfs_markers.jsonl"),
            "power": str(power_path),
        },
    }


def write_bench_run_metrics(
    output_dir: Path,
    path: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    metrics = build_bench_run_metrics(output_dir, **kwargs)
    out = path or (Path(output_dir) / "run_exec_metrics.json")
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
