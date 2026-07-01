"""Unified power+latency capture record.

One ``CaptureRecord`` is emitted per profiled timing sample (per shot for MoE,
per shot×layer for dense/attention/per_sequence).  It binds latency and power
from the *same* shot window so the two quantities are inseparable.

Canonical output: ``captures.jsonl`` (one JSON line per record) under the
per-category output directory (the same directory that holds moe.csv, etc.).
"""

from __future__ import annotations

import datetime
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from profiler.core.gpu_power import (
    compute_achieved_mhz,
    compute_idle_power_w,
    compute_power_hz,
    load_power_samples,
)

# ---------------------------------------------------------------------------
# Clock-lock tolerance (single principled value used everywhere).
# Tighter than the old audit_gpu_clocks.py default of 100 MHz and the
# dvfs_barrier hold-wait of ±15 MHz.  25 MHz sits between them: tight
# enough to catch a failed lock (V100 throttling typically moves ≥50 MHz),
# loose enough to tolerate normal boost jitter around the locked setpoint.
# ---------------------------------------------------------------------------
CLOCK_TOLERANCE_MHZ: int = 25


# ---------------------------------------------------------------------------
# CaptureRecord — the atomic unit of measurement
# ---------------------------------------------------------------------------

@dataclass
class CaptureRecord:
    """One profiled timing sample with power and clock provenance.

    Latency and power come from the same shot window, so if you have
    one you have the other.
    """
    # --- identity ---
    device: str           # hardware tag, e.g. "V100_1100MHz" or "V100"
    model: str
    tp: int
    dtype: str
    category: str         # "moe" | "dense" | "attention" | "per_sequence"
    layer: str | None     # canonical layer name; None for the moe category
    tokens: int           # total new tokens; for per_sequence = num sequences
    activated_experts: int  # 0 for non-MoE categories

    # --- clock provenance ---
    target_mhz: int | None      # MHz extracted from hw tag; None = no lock
    achieved_mhz: float | None  # median graphics clock over busy power samples
    clock_ok: bool | None       # |achieved − target| ≤ CLOCK_TOLERANCE_MHZ; None if no target

    # --- latency ---
    latency_us: float           # mean CUDA-event time per invocation over iterations
    latency_std_us: float | None  # std over iterations (None: not available yet)
    iterations: int             # measurement_iterations used for this shot

    # --- energy (trapezoidal over the timed measurement window) ---
    energy_j: float | None      # ∫P·dt trapezoidal over the shot's power samples
    mean_power_w: float | None  # energy_j / duration
    power_samples: int          # number of power samples in the measurement window
    power_hz: float | None      # effective sampling rate

    # --- idle power (pre/low-util window at the locked clock) ---
    idle_power_w: float | None

    # --- provenance ---
    job_id: str
    timestamp: str  # ISO 8601 UTC

    # --- self-description ---
    # True when power_samples > 0 AND achieved_mhz is not None AND energy_j is not None.
    # Lets downstream distinguish a fully-instrumented record from a timing-only one
    # without inferring from multiple null fields.  Defaults to False for backward
    # compatibility when loading old JSONL records that predate this field.
    has_power: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def target_mhz_from_hw_tag(hw_tag: str) -> int | None:
    """Extract target MHz from a hardware tag string.

    "V100_1100MHz" → 1100,  "V100" → None.
    """
    m = re.search(r"_(\d+)MHz$", hw_tag)
    return int(m.group(1)) if m else None


def resolve_target_mhz(hw_tag: str) -> int | None:
    """Resolve the profiling target clock from hw_tag, then GPU_FREQ_MHZ env var.

    Priority:
      1. MHz suffix encoded in the hardware tag: "V100_900MHz" → 900.
         This is the normal path when run_arc_v100_profile.sh auto-names
         HARDWARE as "V100_${GPU_FREQ_MHZ}MHz".
      2. GPU_FREQ_MHZ environment variable: set by the launcher when a clock
         lock is requested.  HARDWARE may be pre-exported as a bare tag (e.g.
         ``HARDWARE=V100 GPU_FREQ_MHZ=900``) so the MHz suffix is absent from
         the hw_tag.  This fallback recovers the target in that case.
      3. None — genuinely uncapped / boost run (no lock requested).

    Never returns 0 or a negative value; malformed env strings are ignored.
    """
    from_tag = target_mhz_from_hw_tag(hw_tag)
    if from_tag is not None:
        return from_tag
    env_val = os.environ.get("GPU_FREQ_MHZ", "").strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    return None


def clock_ok_for(
    achieved_mhz: float | None,
    target_mhz: int | None,
    tolerance: int = CLOCK_TOLERANCE_MHZ,
) -> bool | None:
    """True iff |achieved − target| ≤ tolerance.  None when no target."""
    if target_mhz is None or achieved_mhz is None:
        return None
    return abs(achieved_mhz - target_mhz) <= tolerance


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")


# ---------------------------------------------------------------------------
# Shot → token / expert extraction (mirrors categories.py logic)
# ---------------------------------------------------------------------------

def _shot_tokens(category_name: str, shot: Any) -> int:
    """Total new tokens for the shot (or num-sequences for per_sequence)."""
    if category_name == "per_sequence":
        return len(shot.requests)
    # Dense, MoE, attention: sum of new_tokens across requests.
    return sum(new for new, _ in shot.requests)


def _shot_activated_experts(category_name: str, shot: Any) -> int:
    if category_name == "moe" and shot.experts is not None:
        return int(shot.experts["activated"])
    return 0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_capture_records(
    *,
    shot_key: str,
    category_name: str,
    shot: Any,
    timings: list[Any],   # list[TimingSample]
    arch: Any,            # Architecture (unused but kept for future use)
    tp: int,
    args: Any,            # ProfileArgs
    exec_record: dict[str, Any],
    power_path: Path | None,
) -> list[CaptureRecord]:
    """Build one CaptureRecord per timing sample from a single shot.

    For the MoE category there is one timing per shot → one record.
    For dense/attention/per_sequence there is one timing per matched
    layer → one record per layer.

    Latency and power are bound here: both come from the same shot
    window and are written into the same record atomically.
    """
    # ---- power-derived scalars ----------------------------------------
    power_samples_list: list[dict[str, Any]] = []
    if power_path is not None:
        power_samples_list = load_power_samples(power_path)

    achieved_mhz = compute_achieved_mhz(power_samples_list)
    idle_power_w = compute_idle_power_w(power_samples_list)
    p_hz = compute_power_hz(power_samples_list)

    # ---- clock provenance -----------------------------------------------
    # resolve_target_mhz checks the hw_tag first, then falls back to the
    # GPU_FREQ_MHZ env var.  This handles the case where HARDWARE was
    # pre-exported without a MHz suffix (e.g. HARDWARE=V100) while the
    # launcher simultaneously set GPU_FREQ_MHZ=900 to lock the clock.
    target_mhz = resolve_target_mhz(args.hardware)
    c_ok = clock_ok_for(achieved_mhz, target_mhz)

    # ---- energy from exec_record ----------------------------------------
    energy_j: float | None = exec_record.get("energy_j")
    power_nested = exec_record.get("power") or {}
    power_total = power_nested.get("total") or {}
    mean_power_w: float | None = power_total.get("mean_power_w")
    n_power_samples: int = int(power_total.get("sample_count") or 0)

    # ---- identity fields ------------------------------------------------
    tokens = _shot_tokens(category_name, shot)
    activated_experts = _shot_activated_experts(category_name, shot)
    device = args.hardware
    model = args.model
    dtype = args.dtype or (
        (args.model_config or {}).get("torch_dtype") or "unknown"
    )
    iterations = args.measurement_iterations
    job_id = _job_id()
    timestamp = _utcnow_iso()

    # ---- has_power: True only when all three instrumentation signals are present ---
    hp = (
        n_power_samples > 0
        and achieved_mhz is not None
        and energy_j is not None
    )

    # ---- one record per timing sample -----------------------------------
    records: list[CaptureRecord] = []
    for sample in timings:
        layer = None if category_name == "moe" else sample.layer
        records.append(CaptureRecord(
            device=device,
            model=model,
            tp=tp,
            dtype=dtype,
            category=category_name,
            layer=layer,
            tokens=tokens,
            activated_experts=activated_experts,
            target_mhz=target_mhz,
            achieved_mhz=achieved_mhz,
            clock_ok=c_ok,
            latency_us=float(sample.microseconds),
            latency_std_us=None,
            iterations=iterations,
            energy_j=energy_j,
            mean_power_w=mean_power_w,
            power_samples=n_power_samples,
            power_hz=p_hz,
            idle_power_w=idle_power_w,
            job_id=job_id,
            timestamp=timestamp,
            has_power=hp,
        ))
    return records


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def append_capture_records(out_dir: Path, records: list[CaptureRecord]) -> None:
    """Append CaptureRecords to ``out_dir/captures.jsonl`` (one JSON line each)."""
    path = out_dir / "captures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), sort_keys=False) + "\n")


def load_captures(path: Path) -> list[CaptureRecord]:
    """Load all CaptureRecords from a ``captures.jsonl`` file."""
    if not path.is_file():
        return []
    records: list[CaptureRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            records.append(CaptureRecord(**d))
    return records


# ---------------------------------------------------------------------------
# Run-level audit sidecar
# ---------------------------------------------------------------------------

def write_preliminary_audit_json(out_dir: Path) -> None:
    """Write a preliminary audit.json marking the category sweep as in-progress.

    Called at the START of a category's shot loop so that a crash mid-sweep
    leaves ``verdict_ok=False`` rather than a stale "pass" from a prior run.
    ``write_audit_json`` overwrites this with the real verdict on clean completion.
    """
    (out_dir / "audit.json").write_text(
        json.dumps({"verdict_ok": False, "verdict_reason": "in_progress"}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_audit_json(
    out_dir: Path,
    *,
    target_mhz: int | None,
    tolerance: int = CLOCK_TOLERANCE_MHZ,
) -> dict[str, Any]:
    """Read captures.jsonl and write an audit.json summary.

    Downstream logic can read audit.json to refuse to trust or skip
    un-audited / clock-failed rows without re-reading every capture.
    """
    captures_path = out_dir / "captures.jsonl"
    records = load_captures(captures_path)

    # Per achieved_mhz bucket: collect achieved clocks
    achieved_clocks: list[float] = [
        r.achieved_mhz for r in records if r.achieved_mhz is not None
    ]
    clock_ok_count = sum(1 for r in records if r.clock_ok is True)
    clock_fail_count = sum(1 for r in records if r.clock_ok is False)
    clock_unknown_count = sum(1 for r in records if r.clock_ok is None)

    median_achieved: float | None = None
    if achieved_clocks:
        median_achieved = statistics.median(achieved_clocks)

    verdict_ok: bool
    if target_mhz is None:
        # No lock target → no verdict possible; treat as informational pass.
        verdict_ok = True
        verdict_reason = "no target clock (default boost)"
    elif not achieved_clocks:
        verdict_ok = False
        verdict_reason = "no power samples with clock data"
    else:
        off = abs(median_achieved - target_mhz)
        verdict_ok = off <= tolerance
        verdict_reason = (
            f"median_achieved={median_achieved:.0f} target={target_mhz} "
            f"delta={off:+.0f} tolerance={tolerance}"
        )

    audit: dict[str, Any] = {
        "verdict_ok": verdict_ok,
        "verdict_reason": verdict_reason,
        "target_mhz": target_mhz,
        "tolerance_mhz": tolerance,
        "median_achieved_mhz": round(median_achieved, 1) if median_achieved is not None else None,
        "clock_ok_records": clock_ok_count,
        "clock_fail_records": clock_fail_count,
        "clock_unknown_records": clock_unknown_count,
        "total_records": len(records),
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit
