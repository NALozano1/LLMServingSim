#!/usr/bin/env python3
"""Aggregate bench layer-boundary DVFS campaign results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def collect(campaign_dir: Path) -> dict[str, Any]:
    runs_root = campaign_dir / "runs"
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "results" / "summary.json"
        spec_path = run_dir / "run_spec.json"
        if not summary_path.is_file():
            rows.append({"run_id": run_dir.name, "status": "pending"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = summary.get("timing") or {}
        energy = summary.get("energy") or {}
        throughput = summary.get("throughput") or {}
        marker = summary.get("marker_summary") or {}
        row = {
            "run_id": summary.get("run_id", run_dir.name),
            "status": summary.get("status", "complete"),
            "campaign_mode": summary.get("campaign_mode"),
            "iteration": summary.get("iteration"),
            "model": summary.get("model"),
            "slurm_job_id": summary.get("slurm_job_id"),
            "hostname": summary.get("hostname"),
            "wall_sec": timing.get("wall_sec"),
            "exec_sec": timing.get("exec_sec"),
            "pause_sec": timing.get("pause_sec"),
            "barrier_count": timing.get("barrier_count"),
            "energy_j": energy.get("energy_j"),
            "energy_excl_pause_j": energy.get("energy_excl_pause_j"),
            "total_tok_per_sec_exec": throughput.get("total_tok_per_sec_exec"),
            "n_dvfs_transitions": summary.get("n_dvfs_transitions"),
            "dvfs_barrier_layers": summary.get("dvfs_barrier_layers"),
            "dvfs_freq_schedule": summary.get("dvfs_freq_schedule"),
            "summary_path": str(summary_path),
            "run_spec": str(spec_path) if spec_path.is_file() else "",
            "sim_replication": str(run_dir / "sim_replication.json"),
            "artifacts_dir": str(run_dir / "artifacts"),
        }
        rows.append(row)

    out_tsv = campaign_dir / "campaign_results.tsv"
    if rows:
        fieldnames: list[str] = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with out_tsv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    manifest = {
        "campaign_dir": str(campaign_dir),
        "total_runs": len(rows),
        "complete": sum(1 for r in rows if r.get("status") == "completed"),
        "pending": sum(1 for r in rows if r.get("status") == "pending"),
        "results_tsv": str(out_tsv),
        "runs": rows,
    }
    (campaign_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign_dir", type=Path)
    args = p.parse_args()
    m = collect(args.campaign_dir)
    print(json.dumps({k: m[k] for k in ("campaign_dir", "total_runs", "complete", "pending", "results_tsv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
