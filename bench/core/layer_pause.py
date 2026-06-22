"""Bench integration for in-place vLLM layer-boundary pause."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from vllm_layer_pause.config import (
    WORKER_EXTENSION_CLS,
    barrier_session_id,
    external_host_poller_enabled,
    host_poller_enabled,
    layer_pause_enabled,
    pause_only_enabled,
)

__all__ = [
    "WORKER_EXTENSION_CLS",
    "barrier_session_id",
    "host_poller_enabled",
    "install_layer_pause",
    "layer_pause_enabled",
    "layer_pause_engine_kwargs",
    "markers_path",
    "start_host_poller",
    "stop_host_poller",
    "uninstall_layer_pause",
    "verify_layer_pause_markers",
]


def layer_pause_engine_kwargs() -> dict[str, Any]:
    if not layer_pause_enabled():
        return {}
    return {
        "worker_extension_cls": WORKER_EXTENSION_CLS,
        "enforce_eager": True,
    }


def markers_path(watch_root: Path) -> Path:
    return watch_root / "dvfs_markers.jsonl"


def start_host_poller(watch_root: Path, repo_root: Path) -> subprocess.Popen[Any]:
    poller = repo_root / "profiler" / "jobs" / "dvfs_barrier_host_poller.sh"
    if not poller.is_file():
        raise FileNotFoundError(f"missing host poller: {poller}")
    freq_meta = watch_root / "gpu_freq"
    freq_meta.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if pause_only_enabled():
        env["DVFS_PAUSE_ONLY"] = "1"
        env["VLLM_PAUSE_ONLY"] = "1"
    proc = subprocess.Popen(
        ["bash", str(poller), str(watch_root), str(freq_meta)],
        env=env,
    )
    return proc


def stop_host_poller(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _first_rank(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        return result[0] if result else {}
    return result or {}


async def install_layer_pause(engine, watch_root: Path) -> dict[str, Any]:
    session = barrier_session_id()
    raw = await engine.collective_rpc(
        "layer_pause_install",
        args=(str(watch_root), session),
    )
    info = _first_rank(raw)
    info["markers_path"] = str(markers_path(watch_root))
    return info


async def uninstall_layer_pause(engine) -> dict[str, Any]:
    raw = await engine.collective_rpc("layer_pause_uninstall")
    return _first_rank(raw)


def install_layer_pause_sync(engine, watch_root: Path) -> dict[str, Any]:
    session = barrier_session_id()
    raw = engine.collective_rpc(
        "layer_pause_install",
        args=(str(watch_root), session),
    )
    info = _first_rank(raw)
    info["markers_path"] = str(markers_path(watch_root))
    return info


def uninstall_layer_pause_sync(engine) -> dict[str, Any]:
    raw = engine.collective_rpc("layer_pause_uninstall")
    return _first_rank(raw)


def verify_layer_pause_markers(
    watch_root: Path,
    *,
    min_markers: int = 1,
    require_pause_only: bool = False,
    require_dvfs: bool = False,
) -> dict[str, Any]:
    path = markers_path(watch_root)
    if not path.is_file():
        raise RuntimeError(f"missing markers file: {path}")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < min_markers:
        raise RuntimeError(
            f"expected >= {min_markers} layer_boundary markers, got {len(records)}"
        )

    modes = {rec.get("mode") for rec in records}
    if require_pause_only and modes - {"pause_only"}:
        raise RuntimeError(f"expected pause_only markers, saw modes={modes}")
    if require_dvfs:
        if "dvfs" not in modes:
            raise RuntimeError(f"expected dvfs markers, saw modes={modes}")
        bad = [
            rec
            for rec in records
            if rec.get("mode") == "dvfs"
            and not rec.get("freq_apply_ok")
        ]
        if bad:
            print(
                f"WARN: {len(bad)} dvfs markers with freq_apply_ok=false "
                "(continuing — check CUDA_VISIBLE_DEVICES / gpu_freq_lock)",
                file=sys.stderr,
            )

    pause_secs = [float(rec.get("pause_sec", 0.0)) for rec in records]
    return {
        "markers_path": str(path),
        "marker_count": len(records),
        "modes": sorted(modes),
        "pause_sec_sum": round(sum(pause_secs), 6),
        "pause_sec_mean": round(sum(pause_secs) / len(pause_secs), 6),
        "sample": records[:3],
    }
