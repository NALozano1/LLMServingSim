#!/usr/bin/env python3
"""Aggregate real bench TTFT/TPOT across a V100 DVFS replication campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign_dir", type=Path)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="summary TSV (default: <campaign>/results_summary.tsv)",
    )
    args = p.parse_args()

    campaign = args.campaign_dir
    out_tsv = args.output or (campaign / "results_summary.tsv")
    rows: list[dict] = []

    for run_dir in sorted((campaign / "runs").glob("r*")):
        summary_path = run_dir / "summary.json"
        real_path = run_dir / "real_latency.json"
        sim_path = run_dir / "sim_replication.json"
        spec_path = run_dir / "run_spec.json"

        if not summary_path.is_file() and not real_path.is_file():
            continue

        spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
        summary = (
            json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        )
        real = json.loads(real_path.read_text()) if real_path.is_file() else {}
        sim = json.loads(sim_path.read_text()) if sim_path.is_file() else {}

        dvfs = spec.get("dvfs") or sim.get("dvfs") or {}
        rows.append(
            {
                "run_id": run_dir.name,
                "model": spec.get("model", ""),
                "mode": spec.get("mode", ""),
                "hardware": spec.get("hardware", dvfs.get("hardware_label", "")),
                "gpu_freq_mhz": dvfs.get("gpu_freq_mhz", ""),
                "freq_schedule": ",".join(str(x) for x in (dvfs.get("freq_schedule") or [])),
                "slurm_job_id": summary.get("slurm_job_id", ""),
                "real_ttft_mean_ms": real.get("ttft", {}).get("mean_ms", ""),
                "real_ttft_p99_ms": real.get("ttft", {}).get("p99_ms", ""),
                "real_tpot_mean_ms": real.get("tpot", {}).get("mean_ms", ""),
                "real_tpot_p99_ms": real.get("tpot", {}).get("p99_ms", ""),
                "real_e2e_mean_ms": real.get("e2e_latency", {}).get("mean_ms", ""),
                "sim_hardware": (sim.get("llmservingsim") or {}).get("hardware", ""),
                "sim_profile_exists": (sim.get("llmservingsim") or {}).get(
                    "profile_exists", ""
                ),
                "bench_dir": str(run_dir / "bench"),
            }
        )

    if not rows:
        print(f"No completed runs under {campaign / 'runs'}")
        return 1

    fieldnames = list(rows[0].keys())
    with out_tsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "campaign_dir": str(campaign),
        "run_count": len(rows),
        "results_tsv": str(out_tsv),
        "runs": rows,
    }
    (campaign / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_tsv} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
