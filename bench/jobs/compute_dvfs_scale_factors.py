#!/usr/bin/env python3
"""Compute dvfs_scale factors from hardware validation TSV.

Reads bench/results/<campaign>/validation_results.tsv and writes
dvfs_scale_factors.json alongside it. Scale = ttft_clock / ttft_uncapped.

Used by run_sim_validation_sweep.sh to set --dvfs-scale for Qwen tp4 arms
(which lack per-clock profiler traces).

Usage:
    python3 bench/jobs/compute_dvfs_scale_factors.py \\
        bench/results/v100_prefill_valid_20260625/validation_results.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def model_key(row):
    """Return a short key like 'qwen_tp4'."""
    model = row.get("model") or ""
    tp = row.get("tp_size") or "1"
    if "Qwen" in model or "qwen" in model.lower():
        return f"qwen_tp{tp}"
    run_id = row.get("run_id", "")
    parts = run_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tsv", type=Path, help="validation_results.tsv path")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: <tsv_dir>/dvfs_scale_factors.json)")
    args = ap.parse_args()

    if not args.tsv.is_file():
        print(f"ERROR: {args.tsv} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_tsv(args.tsv)

    # Group by model key
    by_model = {}
    for r in rows:
        mk = model_key(r)
        by_model.setdefault(mk, []).append(r)

    out = {}
    for mk, model_rows in sorted(by_model.items()):
        # Find uncapped baseline
        uncapped = None
        for r in model_rows:
            arm = (r.get("arm_label") or "").lower()
            if arm == "uncapped" or r.get("gpu_freq_mhz_target") in (None, "", "None"):
                if r.get("status") == "completed" and r.get("ttft_median_ms") not in (None, "", "None"):
                    uncapped = r
                    break

        if uncapped is None:
            print(f"WARN: {mk} — no completed uncapped arm found, skipping scale factors", file=sys.stderr)
            continue

        ttft_uncapped = float(uncapped["ttft_median_ms"])
        out[mk] = {}

        for r in sorted(model_rows, key=lambda x: (x.get("gpu_freq_mhz_target") or "0")):
            arm = r.get("arm_label") or r.get("run_id") or "?"
            status = r.get("status", "unknown")
            clock_ok = r.get("clock_verdict_ok", "")
            ttft_raw = r.get("ttft_median_ms")

            if status != "completed" or ttft_raw in (None, "", "None"):
                out[mk][arm] = {"status": status, "ttft_median_ms": None, "scale": None}
                continue

            ttft = float(ttft_raw)
            scale = ttft / ttft_uncapped if ttft_uncapped > 0 else None
            target_mhz = r.get("gpu_freq_mhz_target")
            out[mk][arm] = {
                "ttft_median_ms": round(ttft, 3),
                "scale": round(scale, 6) if scale is not None else None,
                "target_mhz": int(float(target_mhz)) if target_mhz not in (None, "", "None") else None,
                "achieved_mhz": r.get("gpu_freq_mhz_achieved"),
                "clock_verdict_ok": clock_ok,
                "status": status,
            }

    # Print table
    print(f"\n{'Model':<12} {'Arm':<14} {'TTFT hw (ms)':>13} {'Scale':>8} {'ClkOK':>6}")
    print("-" * 60)
    for mk, arms in out.items():
        for arm, v in arms.items():
            ttft_s = f"{v['ttft_median_ms']:.1f}" if v["ttft_median_ms"] else "---"
            scale_s = f"{v['scale']:.4f}" if v["scale"] else "---"
            ok_s = str(v.get("clock_verdict_ok", "?"))
            print(f"{mk:<12} {arm:<14} {ttft_s:>13} {scale_s:>8} {ok_s:>6}")

    out_path = args.out or args.tsv.parent / "dvfs_scale_factors.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
