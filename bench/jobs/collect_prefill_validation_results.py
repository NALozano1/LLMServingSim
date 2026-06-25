#!/usr/bin/env python3
"""Collect Tier-1 / Tier-2 prefill validation results from a campaign directory.

Walks CAMPAIGN_DIR/runs/*/results/summary.json and writes:
  - validation_results.tsv  — flat table, one row per (model, clock) arm
  - validation_summary.json — structured results + per-model latency vs clock table

Usage:
    python3 bench/jobs/collect_prefill_validation_results.py \\
        bench/campaigns/v100_prefill_valid_20260625/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def collect(campaign_dir: Path) -> list[dict[str, Any]]:
    runs_root = campaign_dir / "runs"
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "results" / "summary.json"
        if not summary_path.is_file():
            rows.append({"run_id": run_dir.name, "status": "pending"})
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = s.get("timing") or {}
        latency = s.get("latency") or {}
        ttft = latency.get("ttft") or {}
        energy = s.get("energy") or {}
        rows.append({
            "run_id": run_dir.name,
            "status": s.get("status", "unknown"),
            "arm_label": s.get("arm_label"),
            "model": s.get("model"),
            "tp_size": s.get("tp_size"),
            "num_reqs": s.get("num_reqs"),
            "fix_input_length": s.get("fix_input_length"),
            "gpu_freq_mhz_target": s.get("gpu_freq_mhz_target"),
            "gpu_freq_mhz_achieved": s.get("gpu_freq_mhz_achieved"),
            "clock_verdict_ok": s.get("clock_verdict_ok"),
            "wall_sec": timing.get("wall_sec"),
            "exec_sec": timing.get("exec_sec"),
            "ttft_median_ms": ttft.get("median_ms"),
            "ttft_mean_ms": ttft.get("mean_ms"),
            "ttft_p90_ms": ttft.get("p90_ms"),
            "energy_j": energy.get("energy_j"),
            "energy_excl_pause_j": energy.get("energy_excl_pause_j"),
            "mean_power_w": energy.get("mean_power_w"),
            "mean_power_excl_pause_w": energy.get("mean_power_excl_pause_w"),
            "slurm_job_id": s.get("slurm_job_id"),
            "hostname": s.get("hostname"),
            "recorded_at": s.get("recorded_at"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--json", action="store_true", help="print JSON to stdout instead of TSV")
    args = ap.parse_args()

    campaign_dir = args.campaign_dir
    if not campaign_dir.is_dir():
        print(f"ERROR: {campaign_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = collect(campaign_dir)
    if not rows:
        print("No runs found.", file=sys.stderr)
        sys.exit(0)

    tsv_path = campaign_dir / "validation_results.tsv"
    fieldnames = list(rows[0].keys())
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {tsv_path}")

    summary_path = campaign_dir / "validation_summary.json"
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        mk = str(r.get("model") or r.get("run_id"))
        by_model.setdefault(mk, []).append(r)

    clk_tables: dict[str, list[dict]] = {}
    for model, model_rows in by_model.items():
        table = []
        for r in sorted(model_rows, key=lambda x: (x.get("gpu_freq_mhz_target") or 9999)):
            if r.get("status") != "completed":
                table.append({"clock": r.get("arm_label", "?"), "status": r["status"]})
                continue
            table.append({
                "clock_label": r.get("arm_label"),
                "target_mhz": r.get("gpu_freq_mhz_target"),
                "achieved_mhz": r.get("gpu_freq_mhz_achieved"),
                "clock_ok": r.get("clock_verdict_ok"),
                "ttft_median_ms": r.get("ttft_median_ms"),
                "ttft_p90_ms": r.get("ttft_p90_ms"),
                "energy_excl_pause_j": r.get("energy_excl_pause_j"),
                "mean_power_excl_pause_w": r.get("mean_power_excl_pause_w"),
            })
        clk_tables[model] = table

    out = {
        "campaign_dir": str(campaign_dir),
        "total_runs": len(rows),
        "completed": sum(1 for r in rows if r.get("status") == "completed"),
        "pending": sum(1 for r in rows if r.get("status") == "pending"),
        "failed": sum(1 for r in rows if r.get("status") not in ("completed", "pending")),
        "by_model": clk_tables,
        "all_rows": rows,
    }
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")

    print(f"\n{'='*70}")
    print(f"Campaign: {campaign_dir.name}")
    print(f"  {out['completed']} completed / {out['total_runs']} total ({out['pending']} pending, {out['failed']} failed)")
    for model, table in clk_tables.items():
        short = model.split("/")[-1]
        print(f"\n  {short}:")
        print(f"    {'clock':<12} {'achieved':>10} {'ttft_med_ms':>12} {'energy_excl_j':>14} {'ok':>4}")
        for t in table:
            if "status" in t and t.get("status") != "completed":
                print(f"    {t.get('clock','?'):<12}  ({t['status']})")
                continue
            ok = "Y" if t.get("clock_ok") else ("?" if t.get("clock_ok") is None else "N")
            print(f"    {t.get('clock_label','?'):<12} {t.get('achieved_mhz') or '---':>10}"
                  f" {t.get('ttft_median_ms') or '---':>12} {t.get('energy_excl_pause_j') or '---':>14} {ok:>4}")

    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
