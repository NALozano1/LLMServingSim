#!/usr/bin/env python3
"""Compare LLMServingSim TTFT predictions against real V100 hardware measurements.

Reads:
  - bench/results/<campaign>/validation_results.tsv  — hardware TTFT per arm
  - bench/results/sim_sweep/<run_id>.csv              — sim TTFT per arm (nanoseconds)

Writes bench/results/sim_vs_hardware_comparison.tsv and prints a summary table.

Usage:
    python3 bench/jobs/compare_sim_vs_hardware.py \\
        bench/results/v100_prefill_valid_20260625/validation_results.tsv \\
        bench/results/sim_sweep/

Tier 1: uncapped arms — absolute sim accuracy without any DVFS scaling.
Tier 2: clocked arms  — Phi uses per-clock traces; Qwen uses --dvfs-scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def load_hw_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["run_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def load_sim_csv(path):
    """Return median TTFT in ms from a sim output CSV (times in nanoseconds)."""
    ttfts = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("TTFT") or row.get("ttft")
            if raw and raw.strip():
                ttfts.append(float(raw.strip()))
    if not ttfts:
        return None
    return statistics.median(ttfts) / 1e6  # ns → ms


def model_short(run_id):
    parts = run_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hw_tsv", type=Path, help="validation_results.tsv from hardware campaign")
    ap.add_argument("sim_dir", type=Path, help="directory containing sim_sweep CSVs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output TSV path (default: <sim_dir>/../sim_vs_hardware_comparison.tsv)")
    args = ap.parse_args()

    if not args.hw_tsv.is_file():
        print(f"ERROR: {args.hw_tsv} not found", file=sys.stderr)
        sys.exit(1)
    if not args.sim_dir.is_dir():
        print(f"ERROR: {args.sim_dir} not a directory", file=sys.stderr)
        sys.exit(1)

    hw = load_hw_tsv(args.hw_tsv)

    results = []
    missing_sim = []
    for run_id, hw_row in sorted(hw.items()):
        sim_csv = args.sim_dir / f"{run_id}.csv"
        hw_status = hw_row.get("status", "unknown")
        hw_ttft = hw_row.get("ttft_median_ms")
        hw_ttft = float(hw_ttft) if hw_ttft not in (None, "", "None") else None

        if not sim_csv.is_file():
            missing_sim.append(run_id)
            results.append({
                "run_id": run_id,
                "model_key": model_short(run_id),
                "arm_label": hw_row.get("arm_label", "?"),
                "target_mhz": hw_row.get("gpu_freq_mhz_target"),
                "hw_status": hw_status,
                "hw_ttft_median_ms": hw_ttft,
                "sim_ttft_median_ms": None,
                "abs_err_ms": None,
                "rel_err_pct": None,
                "clock_verdict_ok": hw_row.get("clock_verdict_ok"),
            })
            continue

        sim_ttft = load_sim_csv(sim_csv)

        abs_err = abs(sim_ttft - hw_ttft) if (sim_ttft is not None and hw_ttft is not None) else None
        rel_err = (abs_err / hw_ttft * 100) if (abs_err is not None and hw_ttft and hw_ttft > 0) else None

        results.append({
            "run_id": run_id,
            "model_key": model_short(run_id),
            "arm_label": hw_row.get("arm_label", "?"),
            "target_mhz": hw_row.get("gpu_freq_mhz_target"),
            "hw_status": hw_status,
            "hw_ttft_median_ms": round(hw_ttft, 3) if hw_ttft is not None else None,
            "sim_ttft_median_ms": round(sim_ttft, 3) if sim_ttft is not None else None,
            "abs_err_ms": round(abs_err, 3) if abs_err is not None else None,
            "rel_err_pct": round(rel_err, 2) if rel_err is not None else None,
            "clock_verdict_ok": hw_row.get("clock_verdict_ok"),
        })

    # Write TSV
    out_path = args.out or args.sim_dir.parent / "sim_vs_hardware_comparison.tsv"
    fieldnames = ["run_id", "model_key", "arm_label", "target_mhz",
                  "hw_status", "hw_ttft_median_ms", "sim_ttft_median_ms",
                  "abs_err_ms", "rel_err_pct", "clock_verdict_ok"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_path}")

    # Print table grouped by model
    by_model = {}
    for r in results:
        by_model.setdefault(r["model_key"], []).append(r)

    print(f"\n{'='*80}")
    print(f"Sim vs Hardware TTFT comparison")
    print(f"Hardware: {args.hw_tsv}")
    print(f"Sim:      {args.sim_dir}")
    print(f"{'='*80}")

    completed_errs = []
    for mk, rows in sorted(by_model.items()):
        print(f"\n  {mk}:")
        print(f"    {'arm':<14} {'hw_ttft':>10} {'sim_ttft':>10} {'abs_err':>9} {'rel_err':>8} {'tier':<8}")
        print(f"    {'-'*62}")
        for r in sorted(rows, key=lambda x: (x.get("target_mhz") or "0")):
            arm = r.get("arm_label") or r["run_id"]
            hw_s = f"{r['hw_ttft_median_ms']:.1f}" if r["hw_ttft_median_ms"] else f"({r['hw_status']})"
            sim_s = f"{r['sim_ttft_median_ms']:.1f}" if r["sim_ttft_median_ms"] else "no csv"
            err_s = f"{r['abs_err_ms']:.1f}" if r["abs_err_ms"] is not None else "---"
            rel_s = f"{r['rel_err_pct']:.1f}%" if r["rel_err_pct"] is not None else "---"
            tier = "Tier1" if (r.get("target_mhz") or "").lower() in ("", "none", "null") or not r.get("target_mhz") else "Tier2"
            print(f"    {arm:<14} {hw_s:>10} {sim_s:>10} {err_s:>9} {rel_s:>8} {tier:<8}")
            if r["rel_err_pct"] is not None:
                completed_errs.append(r["rel_err_pct"])

    if completed_errs:
        mean_err = sum(completed_errs) / len(completed_errs)
        print(f"\n  Mean |rel_err| across {len(completed_errs)} matched arms: {mean_err:.1f}%")

    if missing_sim:
        print(f"\n  Missing sim CSVs for {len(missing_sim)} arms: {', '.join(missing_sim)}")
        print(f"  Run bench/jobs/run_sim_validation_sweep.sh on nserver15 to generate them.")


if __name__ == "__main__":
    main()
