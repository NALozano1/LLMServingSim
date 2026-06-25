#!/usr/bin/env python3
"""Aggregate transition-effects calibration sweep results.

Walks CALIB_DIR/runs/*/results/summary.json and emits:
  - calib_results.tsv   — flat table, one row per arm run
  - calib_summary.json  — structured stats: mean/std per (n_transitions, apply_mode)

Usage:
    python3 bench/jobs/collect_transition_calib_results.py \\
        bench/campaigns/v100_transition_calib_phi_20260624/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def collect(calib_dir: Path) -> dict[str, Any]:
    runs_root = calib_dir / "runs"
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "results" / "summary.json"
        if not summary_path.is_file():
            rows.append({"run_id": run_dir.name, "status": "pending"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = summary.get("timing") or {}
        energy = summary.get("energy") or {}
        throughput = summary.get("throughput") or {}
        row = {
            "run_id": summary.get("run_id", run_dir.name),
            "status": summary.get("status", "completed"),
            "arm_label": summary.get("arm_label"),
            "n_transitions": summary.get("n_transitions"),
            "dvfs_apply_mode": summary.get("dvfs_apply_mode"),
            "iteration": summary.get("calib_spec", {}).get("iteration"),
            "model": summary.get("model"),
            "slurm_job_id": summary.get("slurm_job_id"),
            "hostname": summary.get("hostname"),
            "wall_sec": timing.get("wall_sec"),
            "exec_sec": timing.get("exec_sec"),
            "pause_sec": timing.get("pause_sec"),
            "marker_count": summary.get("marker_count"),
            "ioctl_wall_sec_mean": summary.get("ioctl_wall_sec_mean"),
            "ioctl_wall_sec_sum": summary.get("ioctl_wall_sec_sum"),
            "energy_j": energy.get("energy_j"),
            "energy_excl_pause_j": energy.get("energy_excl_pause_j"),
            "total_tok_per_sec_exec": throughput.get("total_tok_per_sec_exec"),
            "dvfs_barrier_layers": summary.get("dvfs_barrier_layers"),
            "dvfs_freq_schedule": summary.get("dvfs_freq_schedule"),
            "summary_path": str(summary_path),
            "artifacts_dir": str(run_dir / "artifacts"),
        }
        rows.append(row)

    # Write TSV.
    out_tsv = calib_dir / "calib_results.tsv"
    if rows:
        fieldnames: list[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with out_tsv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote {out_tsv}", file=sys.stderr)

    # Group by (n_transitions, apply_mode) for summary stats.
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "completed":
            continue
        key = (r.get("n_transitions"), r.get("dvfs_apply_mode", "sync"))
        groups.setdefault(key, []).append(r)

    stats: list[dict[str, Any]] = []
    for (n_trans, apply_mode), group in sorted(groups.items(), key=lambda x: (x[0][0] or 0, x[0][1])):
        wall_times = [float(r["wall_sec"]) for r in group if r.get("wall_sec") is not None]
        exec_times = [float(r["exec_sec"]) for r in group if r.get("exec_sec") is not None]
        pause_times = [float(r["pause_sec"]) for r in group if r.get("pause_sec") is not None]
        ioctl_means = [
            float(r["ioctl_wall_sec_mean"])
            for r in group
            if r.get("ioctl_wall_sec_mean") is not None
        ]
        stats.append({
            "n_transitions": n_trans,
            "dvfs_apply_mode": apply_mode,
            "n_completed": len(group),
            "wall_sec_mean": _mean(wall_times),
            "wall_sec_std": _std(wall_times),
            "exec_sec_mean": _mean(exec_times),
            "exec_sec_std": _std(exec_times),
            "pause_sec_mean": _mean(pause_times),
            "pause_sec_std": _std(pause_times),
            "ioctl_wall_sec_mean_of_means": _mean(ioctl_means),
        })

    # Compute marginal per-transition cost relative to n=0 baseline.
    baseline_exec = None
    for s in stats:
        if s["n_transitions"] == 0 and s["exec_sec_mean"] is not None:
            baseline_exec = s["exec_sec_mean"]
            break

    for s in stats:
        if baseline_exec is not None and s["exec_sec_mean"] is not None and s["n_transitions"]:
            excess = s["exec_sec_mean"] - baseline_exec
            s["marginal_exec_overhead_sec"] = round(excess, 6)
            n = int(s["n_transitions"])
            s["per_transition_overhead_sec"] = round(excess / n, 6) if n else None
        else:
            s["marginal_exec_overhead_sec"] = None
            s["per_transition_overhead_sec"] = None

    out_json = calib_dir / "calib_summary.json"
    summary_doc = {
        "calib_dir": str(calib_dir),
        "total_rows": len(rows),
        "completed": sum(1 for r in rows if r.get("status") == "completed"),
        "pending": sum(1 for r in rows if r.get("status") == "pending"),
        "baseline_exec_sec": baseline_exec,
        "stats_by_arm": stats,
    }
    out_json.write_text(json.dumps(summary_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    return summary_doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calib_dir", help="Path to campaign root directory.")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    args = ap.parse_args()

    calib_dir = Path(args.calib_dir).resolve()
    if not calib_dir.is_dir():
        print(f"ERROR: {calib_dir} is not a directory", file=sys.stderr)
        return 1

    doc = collect(calib_dir)

    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        print(f"\n=== Transition-effects calibration results: {calib_dir.name} ===")
        print(f"  completed / total: {doc['completed']} / {doc['total_rows']}")
        if doc.get("baseline_exec_sec"):
            print(f"  baseline exec_sec: {doc['baseline_exec_sec']:.4f}s")
        print()
        for s in doc.get("stats_by_arm", []):
            n = s["n_transitions"]
            mode = s["dvfs_apply_mode"]
            mean_e = s.get("exec_sec_mean")
            std_e = s.get("exec_sec_std")
            per_t = s.get("per_transition_overhead_sec")
            iw = s.get("ioctl_wall_sec_mean_of_means")
            print(
                f"  n={n:3d}  mode={mode:5s}  exec={mean_e:.4f}s±{std_e:.4f}s"
                if mean_e is not None and std_e is not None
                else f"  n={n:3d}  mode={mode:5s}  exec=N/A",
                end="",
            )
            if per_t is not None:
                print(f"  per_transition={per_t*1000:.1f}ms", end="")
            if iw is not None:
                print(f"  ioctl_wall={iw*1000:.1f}ms", end="")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
