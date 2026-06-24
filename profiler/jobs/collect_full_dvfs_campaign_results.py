#!/usr/bin/env python3
"""Aggregate V100 full-model DVFS campaign results into a summary table + YAML bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _yaml_dump(data: Any) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return json.dumps(data, indent=2) + "\n"


def collect(campaign_dir: Path) -> dict[str, Any]:
    runs_root = campaign_dir / "runs"
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "results" / "summary.json"
        spec_path = run_dir / "run_spec.yaml"
        if not summary_path.is_file():
            rows.append({"run_id": run_dir.name, "status": "pending"})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "run_id": summary.get("run_id", run_dir.name),
            "status": "complete",
            "campaign_mode": summary.get("campaign_mode"),
            "iteration": summary.get("iteration"),
            "model": summary.get("model"),
            "hardware": summary.get("hardware"),
            "job_id": summary.get("job_id"),
            "effective_runtime_sec": summary.get("effective_runtime_sec"),
            "energy_j": summary.get("energy_j"),
            "energy_excl_pause_j": summary.get("energy_excl_pause_j"),
            "barrier_wait_sec": summary.get("barrier_wait_sec"),
            "n_dvfs_transitions": summary.get("n_dvfs_transitions"),
            "gpu_freq_mhz": summary.get("gpu_freq_mhz"),
            "summary_path": str(summary_path),
            "sim_replication_yaml": str(run_dir / "sim_replication.yaml"),
            "run_spec_yaml": str(spec_path) if spec_path.is_file() else "",
            "artifacts_dir": str(run_dir / "artifacts"),
        }
        run_exec = summary.get("run_exec_metrics") or {}
        if run_exec:
            row["run_total_effective_runtime_sec"] = run_exec.get("effective_runtime_sec")
            row["run_total_effective_energy_j"] = run_exec.get("effective_energy_j")
            row["run_shot_count"] = run_exec.get("shot_count")
        rows.append(row)

    out_csv = campaign_dir / "campaign_results.tsv"
    if rows:
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    aggregate = {
        "campaign_dir": str(campaign_dir),
        "total_runs": len(rows),
        "complete": sum(1 for r in rows if r.get("status") == "complete"),
        "pending": sum(1 for r in rows if r.get("status") == "pending"),
        "results_tsv": str(out_csv),
        "runs": rows,
    }
    (campaign_dir / "campaign_results.yaml").write_text(
        _yaml_dump(aggregate), encoding="utf-8"
    )
    return aggregate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path, required=True)
    args = p.parse_args()
    agg = collect(args.campaign_dir)
    print(
        f"complete={agg['complete']}/{agg['total_runs']} "
        f"pending={agg['pending']}"
    )
    print(f"wrote {args.campaign_dir / 'campaign_results.tsv'}")
    print(f"wrote {args.campaign_dir / 'campaign_results.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
