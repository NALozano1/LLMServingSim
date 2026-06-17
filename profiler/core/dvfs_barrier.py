"""In-place layer-boundary DVFS barriers for profiler host ↔ worker sync.

Worker side (inside vLLM TP worker): post-forward hooks block after each
decoder layer, write ``pending.json``, spin until ``ack.json`` appears.

Host side (profiler runner): background poller applies GPU frequency via
``gpu_freq_lock.py``, waits for settle, writes ack + ``dvfs_markers.jsonl``.

Environment (host):
    DVFS_LAYER_PAUSE=1          Enable barriers for each ``fire()`` call
    DVFS_FREQ_SCHEDULE=700,900,1100,1300   MHz cycle at layer boundaries
    GPU_FREQ_SETTLE_SEC=1.5     Seconds to wait after each freq change
    ENGS2950_ROOT=/data/engs-glass/engs2950
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def dvfs_layer_pause_enabled() -> bool:
    return os.environ.get("DVFS_LAYER_PAUSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dvfs_host_poller_enabled() -> bool:
    return os.environ.get("DVFS_HOST_POLLER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dvfs_pause_only_enabled() -> bool:
    """Pause at layer boundaries but do not change GPU frequency."""
    return os.environ.get("DVFS_PAUSE_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pause_only_delay_sec() -> float:
    return float(os.environ.get("PAUSE_ONLY_DELAY_SEC", "0.05"))


def parse_freq_schedule() -> list[int]:
    raw = os.environ.get("DVFS_FREQ_SCHEDULE", "700,900,1100,1300")
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("DVFS_FREQ_SCHEDULE is empty")
    return out


def gpu_freq_settle_sec() -> float:
    return float(os.environ.get("GPU_FREQ_SETTLE_SEC", "1.5"))


def engs2950_root() -> Path:
    return Path(os.environ.get("ENGS2950_ROOT", "/data/engs-glass/engs2950"))


def gpu_freq_lock_script() -> Path:
    return engs2950_root() / "shared/scripts/gpu_freq_lock.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Host: frequency control
# ---------------------------------------------------------------------------

def apply_gpu_freq_mhz(mhz: int, meta_dir: Path) -> dict[str, Any]:
    """Lock all visible GPUs to *mhz* via site helper."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    script = gpu_freq_lock_script()
    if not script.is_file():
        return {"ok": False, "error": f"missing {script}"}
    proc = subprocess.run(
        [sys.executable, str(script), "apply", "--mhz", str(mhz),
         "--out-dir", str(meta_dir)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "mhz": mhz,
    }


def restore_gpu_freq(meta_dir: Path) -> dict[str, Any]:
    meta_dir.mkdir(parents=True, exist_ok=True)
    script = gpu_freq_lock_script()
    if not script.is_file():
        return {"ok": False, "error": f"missing {script}"}
    proc = subprocess.run(
        [sys.executable, str(script), "restore", "--out-dir", str(meta_dir)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


# ---------------------------------------------------------------------------
# Host: poller thread
# ---------------------------------------------------------------------------

@dataclass
class DvfsBarrierPoller:
    """Watch *barrier_dir* and apply DVFS while the worker is blocked."""

    barrier_dir: Path
    markers_path: Path
    freq_meta_dir: Path
    freq_schedule: list[int]
    settle_sec: float
    on_marker: Callable[[dict[str, Any]], None] | None = None

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _freq_idx: int = field(default=0, init=False)
    _errors: list[str] = field(default_factory=list, init=False)

    def start(self) -> None:
        self.barrier_dir.mkdir(parents=True, exist_ok=True)
        self.markers_path.parent.mkdir(parents=True, exist_ok=True)
        self.freq_meta_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="dvfs-barrier-poller",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def _next_freq(self) -> int:
        mhz = self.freq_schedule[self._freq_idx % len(self.freq_schedule)]
        self._freq_idx += 1
        return mhz

    def _append_marker(self, record: dict[str, Any]) -> None:
        with self.markers_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.on_marker is not None:
            self.on_marker(record)

    def _run(self) -> None:
        pending = self.barrier_dir / "pending.json"
        ack = self.barrier_dir / "ack.json"
        while not self._stop.is_set():
            if not pending.is_file():
                time.sleep(0.0005)
                continue
            try:
                payload = json.loads(pending.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._errors.append(f"read pending: {exc}")
                time.sleep(0.01)
                continue

            pause_start = time.time()
            if dvfs_pause_only_enabled():
                time.sleep(pause_only_delay_sec())
                target_mhz = None
                apply_ok = None
            else:
                target_mhz = self._next_freq()
                apply_result = apply_gpu_freq_mhz(
                    target_mhz, self.freq_meta_dir
                )
                apply_ok = apply_result.get("ok")
                if not apply_ok:
                    self._errors.append(
                        f"freq apply {target_mhz} MHz failed: {apply_result}"
                    )
                time.sleep(self.settle_sec)
            pause_end = time.time()

            marker = {
                "event": "layer_boundary",
                "mode": "pause_only" if dvfs_pause_only_enabled() else "dvfs",
                "wall_ts": _utc_now(),
                "pause_start": pause_start,
                "pause_end": pause_end,
                "pause_sec": pause_end - pause_start,
                "freq_mhz": target_mhz,
                "settle_sec": (
                    pause_only_delay_sec()
                    if dvfs_pause_only_enabled()
                    else self.settle_sec
                ),
                "layer_idx": payload.get("layer_idx"),
                "layer_name": payload.get("layer_name"),
                "shot_id": payload.get("shot_id"),
                "barrier_id": payload.get("barrier_id"),
                "freq_apply_ok": apply_ok,
            }
            self._append_marker(marker)

            ack.write_text(
                json.dumps(
                    {
                        "barrier_id": payload.get("barrier_id"),
                        "freq_mhz": target_mhz,
                        "ack_ts": _utc_now(),
                    }
                ),
                encoding="utf-8",
            )
            try:
                pending.unlink()
            except OSError:
                pass

            time.sleep(0.001)


def make_shot_barrier_dir(out_dir: Path, shot_key: str) -> Path:
    safe = shot_key.replace("/", "_").replace(" ", "_")
    return out_dir / "dvfs_barriers" / safe
