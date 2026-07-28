#!/usr/bin/env python3
"""Compare real-B2B smoke CSVs vs legacy gapfree latency + profiler shot.

Reads newest real_b2b_*.csv under power_calib/, joins against:
  - gapfree_latency.csv (legacy wall/iters latency)
  - profiler tp1 moe.csv (CUDA-event shot)
  - power_calib_<DEV>.csv (legacy gapfree power) when present
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/data/engs-glass/engs2950")
CALIB = ROOT / "DVFS-MoE/LLMServingSim/validation/recon_experiments/power_calib"
GAPFREE = ROOT / ".paper_repo/results/final/power_calib/gapfree_latency/gapfree_latency.csv"
PROF_QWEN = ROOT / "DVFS-MoE/dvfs-policy/profiler/perf"
OUT = CALIB / "real_b2b_vs_legacy_compare.csv"

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
AE = 8


def _load_gapfree() -> dict[tuple[str, int, int], float]:
    out: dict[tuple[str, int, int], float] = {}
    with GAPFREE.open() as f:
        for r in csv.DictReader(f):
            if r["model"] != MODEL:
                continue
            out[(r["device"], int(r["freq_mhz"]), int(r["batch"]))] = float(r["gapfree_latency_ms"])
    return out


def _prof_ms(device: str, freq: int, batch: int) -> float | None:
    p = PROF_QWEN / f"{device}_{freq}MHz" / MODEL / "fp16" / "tp1" / "moe.csv"
    if not p.exists():
        return None
    with p.open() as f:
        for r in csv.DictReader(f):
            if int(r["tokens"]) == batch and int(r["activated_experts"]) == AE:
                return float(r["time_us"]) / 1000.0
    return None


def _legacy_power(device: str, freq: int, batch: int) -> float | None:
    p = CALIB / f"power_calib_{device}.csv"
    if not p.exists():
        return None
    with p.open() as f:
        for r in csv.DictReader(f):
            if (
                MODEL in r.get("model", "")
                and int(float(r["freq_mhz"])) == freq
                and int(r["batch"]) == batch
            ):
                return float(r["P_active_gapfree_w"])
    return None


def main() -> int:
    smokes = sorted(CALIB.glob("real_b2b_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not smokes:
        print(f"No real_b2b_*.csv under {CALIB}", file=sys.stderr)
        return 1

    gap = _load_gapfree()
    rows_out: list[dict] = []
    seen: set[tuple[str, int, int]] = set()

    for smoke in smokes:
        with smoke.open() as f:
            for r in csv.DictReader(f):
                # device column may be full nvidia name
                name = r["device"]
                if "A100" in name:
                    dev = "A100"
                elif "V100" in name:
                    dev = "V100"
                elif "H100" in name:
                    dev = "H100"
                elif "L40" in name:
                    dev = "L40S"
                else:
                    continue
                freq = int(r["freq_mhz"])
                batch = int(r["batch"])
                key = (dev, freq, batch)
                if key in seen:
                    continue
                seen.add(key)

                cuda_ms = float(r["cuda_latency_ms"])
                wall_ms = float(r["wall_latency_ms"])
                legacy_gf = gap.get(key)
                prof = _prof_ms(dev, freq, batch)
                leg_p = _legacy_power(dev, freq, batch)
                real_p = float(r["P_active_gapfree_w"])

                rows_out.append(
                    {
                        "source_csv": smoke.name,
                        "device": dev,
                        "freq_mhz": freq,
                        "batch": batch,
                        "mode": r.get("mode", ""),
                        "real_cuda_ms": round(cuda_ms, 4),
                        "real_wall_ms": round(wall_ms, 4),
                        "legacy_gapfree_ms": round(legacy_gf, 4) if legacy_gf else "",
                        "profiler_shot_ms": round(prof, 4) if prof else "",
                        "real_over_profiler": round(cuda_ms / prof, 3) if prof and prof > 0 else "",
                        "legacy_over_profiler": round(legacy_gf / prof, 3)
                        if legacy_gf and prof and prof > 0
                        else "",
                        "real_P_w": round(real_p, 2),
                        "legacy_P_w": round(leg_p, 2) if leg_p else "",
                        "verdict_ok": r.get("verdict_ok", ""),
                    }
                )

    if not rows_out:
        print("No comparable rows found", file=sys.stderr)
        return 1

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows_out, key=lambda x: (x["device"], x["freq_mhz"], x["batch"])))

    print(f"Wrote {OUT} ({len(rows_out)} rows)")
    print()
    print(f"{'dev':5s} {'freq':4s} {'batch':5s} {'mode':7s} {'cuda_ms':>10s} {'legacy_ms':>10s} {'prof_ms':>10s} {'real/prof':>9s} {'leg/prof':>9s}")
    for r in sorted(rows_out, key=lambda x: (x["device"], x["freq_mhz"], x["batch"])):
        print(
            f"{r['device']:5s} {r['freq_mhz']:4} {r['batch']:5} {str(r['mode']):7s} "
            f"{r['real_cuda_ms']:>10} {str(r['legacy_gapfree_ms']):>10} {str(r['profiler_shot_ms']):>10} "
            f"{str(r['real_over_profiler']):>9} {str(r['legacy_over_profiler']):>9}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
