"""Execution time and energy metrics excluding DVFS pause windows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from profiler.core.gpu_power import integrate_power_joules, load_power_samples


def load_markers(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def markers_for_shot(
    markers: list[dict[str, Any]],
    shot_key: str,
) -> list[dict[str, Any]]:
    prefix = f"{shot_key}:"
    return [
        m
        for m in markers
        if str(m.get("barrier_id", "")).startswith(prefix)
        or str(m.get("shot_id", "")) == shot_key
    ]


def pause_intervals_from_markers(
    markers: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for rec in markers:
        start = rec.get("pause_start")
        end = rec.get("pause_end")
        if start is None or end is None:
            pause_sec = rec.get("pause_sec")
            if pause_sec is None:
                continue
            start = float(end) - float(pause_sec)
        intervals.append((float(start), float(end)))
    return intervals


def sum_marker_pause_sec(markers: list[dict[str, Any]]) -> float:
    total = 0.0
    for rec in markers:
        if "pause_sec" in rec:
            total += float(rec["pause_sec"])
        elif "pause_start" in rec and "pause_end" in rec:
            total += float(rec["pause_end"]) - float(rec["pause_start"])
    return round(total, 6)


def _clip_samples_to_window(
    samples: list[dict[str, Any]],
    t_start: float,
    t_end: float,
) -> list[dict[str, Any]]:
    return [
        s
        for s in samples
        if t_start <= float(s["wall_ts"]) <= t_end
    ]


def compute_shot_exec_metrics(
    fire_timing: dict[str, Any],
    *,
    shot_markers: list[dict[str, Any]],
    power_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive exec-only runtime/energy from worker timing, markers, and power."""
    measured_sec = float(fire_timing.get("measured_sec") or 0.0)
    warmup_sec = float(fire_timing.get("warmup_sec") or 0.0)
    fire_total_sec = float(fire_timing.get("fire_total_sec") or 0.0)
    barrier_wait_sec = float(fire_timing.get("barrier_wait_sec") or 0.0)

    marker_pause_sec = sum_marker_pause_sec(shot_markers)
    pause_intervals = pause_intervals_from_markers(shot_markers)

    pause_sec = barrier_wait_sec if barrier_wait_sec > 0 else marker_pause_sec
    measured_exec_sec = max(0.0, measured_sec - pause_sec)
    fire_exec_sec = max(0.0, fire_total_sec - pause_sec)

    out: dict[str, Any] = {
        "warmup_sec": round(warmup_sec, 6),
        "measured_sec": round(measured_sec, 6),
        "measured_exec_sec": round(measured_exec_sec, 6),
        "effective_runtime_sec": round(measured_exec_sec, 6),
        "fire_total_sec": round(fire_total_sec, 6),
        "fire_exec_sec": round(fire_exec_sec, 6),
        "barrier_wait_sec": round(barrier_wait_sec, 6),
        "marker_pause_sec": marker_pause_sec,
        "pause_intervals": [
            {"pause_start": s, "pause_end": e} for s, e in pause_intervals
        ],
        "barrier_count": len(shot_markers),
    }

    layer_freqs = [m.get("freq_mhz") for m in shot_markers if m.get("freq_mhz")]
    if layer_freqs:
        out["layer_freq_mhz"] = layer_freqs

    if power_samples:
        m_start = fire_timing.get("measured_wall_start")
        m_end = fire_timing.get("measured_wall_end")
        if m_start is not None and m_end is not None:
            power_samples = _clip_samples_to_window(
                power_samples,
                float(m_start),
                float(m_end),
            )
        power_all = integrate_power_joules(power_samples)
        power_exec = integrate_power_joules(power_samples, pause_intervals)
        out["power"] = {
            "total": power_all,
            "exec_excl_pause": power_exec,
        }
        out["energy_j"] = power_all.get("energy_j")
        out["energy_excl_pause_j"] = power_exec.get("energy_excl_pause_j")
        out["effective_energy_j"] = power_exec.get("energy_excl_pause_j")

    return out


def append_shot_exec_metrics(out_dir: Path, record: dict[str, Any]) -> None:
    path = out_dir / "shot_exec_metrics.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_shot_exec_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _sum_field(records: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(r.get(key) or 0.0) for r in records), 6)


def _shot_freq_normalized_exec_sec(
    rec: dict[str, Any],
    ref_freq_mhz: int,
) -> float:
    """Reference-frequency equivalent exec time (approx equal split per layer)."""
    exec_sec = float(rec.get("measured_exec_sec") or 0.0)
    layer_freqs = rec.get("layer_freq_mhz") or []
    if not layer_freqs:
        return exec_sec
    seg = exec_sec / len(layer_freqs)
    return sum(seg * (float(f) / ref_freq_mhz) for f in layer_freqs if f)


def aggregate_run_exec_metrics(
    records: list[dict[str, Any]],
    *,
    ref_freq_mhz: int | None = None,
) -> dict[str, Any]:
    """Sum per-shot exec metrics into run-level totals."""
    if not records:
        return {"shot_count": 0}

    total_measured_exec = _sum_field(records, "measured_exec_sec")
    total_measured = _sum_field(records, "measured_sec")
    total_pause = _sum_field(records, "barrier_wait_sec")
    total_marker_pause = _sum_field(records, "marker_pause_sec")
    total_energy = _sum_field(records, "energy_j")
    total_energy_exec = _sum_field(records, "energy_excl_pause_j")
    total_warmup = _sum_field(records, "warmup_sec")
    total_fire = _sum_field(records, "fire_total_sec")
    total_fire_exec = _sum_field(records, "fire_exec_sec")
    total_rpc = _sum_field(records, "rpc_wall_sec")

    out: dict[str, Any] = {
        "shot_count": len(records),
        "measured_sec": total_measured,
        "measured_exec_sec": total_measured_exec,
        "barrier_wait_sec": total_pause,
        "marker_pause_sec": total_marker_pause,
        "warmup_sec": total_warmup,
        "fire_total_sec": total_fire,
        "fire_exec_sec": total_fire_exec,
        "rpc_wall_sec": total_rpc,
        "energy_j": round(total_energy, 3),
        "energy_excl_pause_j": round(total_energy_exec, 3),
        "effective_runtime_sec": total_measured_exec,
        "effective_energy_j": round(total_energy_exec, 3),
        "pause_overhead_sec": total_pause,
        "pause_overhead_frac": (
            round(total_pause / total_measured, 6) if total_measured > 0 else 0.0
        ),
    }

    if ref_freq_mhz and ref_freq_mhz > 0:
        norm_total = sum(
            _shot_freq_normalized_exec_sec(rec, ref_freq_mhz) for rec in records
        )
        out["freq_normalized_exec_sec"] = round(norm_total, 6)
        out["ref_freq_mhz"] = ref_freq_mhz

    return out


def write_run_exec_metrics(
    out_dir: Path,
    *,
    ref_freq_mhz: int | None = None,
    extra: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    architecture: dict[str, Any] | None = None,
    tp: int | None = None,
) -> dict[str, Any] | None:
    """Read shot_exec_metrics.jsonl and write run_exec_metrics.json."""
    shot_path = out_dir / "shot_exec_metrics.jsonl"
    records = load_shot_exec_metrics(shot_path)
    if not records:
        return None

    if ref_freq_mhz is None:
        raw = os.environ.get("DVFS_REF_FREQ_MHZ", "").strip()
        if raw.isdigit():
            ref_freq_mhz = int(raw)

    run_rec = aggregate_run_exec_metrics(records, ref_freq_mhz=ref_freq_mhz)
    if extra:
        run_rec.update(extra)

    if model_config is not None and architecture is not None and tp is not None:
        if os.environ.get("PROFILER_SYNTH_SERVING_LATENCY", "0").strip() in (
            "1",
            "true",
            "yes",
        ):
            try:
                from profiler.core.latency_metrics import synthesize_reference_latency

                latency = synthesize_reference_latency(
                    out_dir,
                    architecture,
                    model_config,
                    tp=tp,
                )
                if latency:
                    run_rec["serving_latency"] = latency
                    run_rec["ttft_sec"] = latency["ttft_sec"]
                    run_rec["ttft_ms"] = latency["ttft_ms"]
                    run_rec["tpot_sec"] = latency["tpot_sec"]
                    run_rec["tpot_ms"] = latency["tpot_ms"]
                    run_rec["itl_sec"] = latency["itl_sec"]
                    run_rec["e2e_latency_sec"] = latency["e2e_latency_sec"]
            except Exception as exc:
                from profiler.core import logger as log

                log.warning("serving latency synthesis skipped: %s", exc)

    (out_dir / "run_exec_metrics.json").write_text(
        json.dumps(run_rec, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_rec


def build_shot_exec_record(
    shot_key: str,
    fire_timing: dict[str, Any],
    markers_path: Path,
    power_path: Path | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shot_markers = markers_for_shot(load_markers(markers_path), shot_key)
    power_samples = load_power_samples(power_path) if power_path else None
    metrics = compute_shot_exec_metrics(
        fire_timing,
        shot_markers=shot_markers,
        power_samples=power_samples,
    )
    record = {
        "shot_key": shot_key,
        **metrics,
    }
    if extra:
        record.update(extra)
    return record
