#!/usr/bin/env python3
"""Backfill summary.json for runs where the collect block didn't write one.

Reads meta.json, requests.jsonl, gpu_power/bench.jsonl directly to compute
timing / TTFT / energy — the same math as the inline COLLECT block in
run_arc_v100_prefill_validation.sh.

Usage:
    python3 bench/jobs/backfill_summary_from_raw.py bench/campaigns/CAMPAIGN_DIR/
    python3 bench/jobs/backfill_summary_from_raw.py bench/campaigns/CAMPAIGN_DIR/runs/my_arm/
    python3 bench/jobs/backfill_summary_from_raw.py --dry-run bench/campaigns/CAMPAIGN_DIR/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


def compute_summary(run_dir: Path, gpu_idx: int = 0) -> dict:
    freq_summary = run_dir / "gpu_freq" / "gpu_freq_hold_summary.json"
    meta_path = run_dir / "meta.json"
    reqs_path = run_dir / "requests.jsonl"
    power_path = run_dir / "gpu_power" / "bench.jsonl"

    freq_data = json.loads(freq_summary.read_text()) if freq_summary.exists() else {}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Timing from meta.json
    wall_sec = None
    if meta.get("started_at") and meta.get("finished_at"):
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        t0 = datetime.strptime(meta["started_at"], fmt)
        t1 = datetime.strptime(meta["finished_at"], fmt)
        wall_sec = (t1 - t0).total_seconds()

    timing = {"wall_sec": wall_sec, "exec_sec": wall_sec, "pause_sec": 0.0}

    # TTFT from requests.jsonl
    ttft_ms_list = []
    if reqs_path.exists():
        for line in reqs_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("first_token_ts") is not None and r.get("queued_ts") is not None:
                ttft_ms_list.append((r["first_token_ts"] - r["queued_ts"]) * 1000.0)

    def _pct(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
        return round(s[idx], 3)

    ttft = {
        "count": len(ttft_ms_list),
        "mean_ms": round(statistics.mean(ttft_ms_list), 3) if ttft_ms_list else None,
        "median_ms": round(statistics.median(ttft_ms_list), 3) if ttft_ms_list else None,
        "p90_ms": _pct(ttft_ms_list, 90),
        "p99_ms": _pct(ttft_ms_list, 99),
    }
    latency = {"ttft": ttft}

    # Energy from gpu_power/bench.jsonl (target GPU index only)
    energy_j = None
    mean_power_w = None
    if power_path.exists():
        samples = [json.loads(l) for l in power_path.read_text().splitlines() if l.strip()]
        powers_w, wall_ts = [], []
        for s in samples:
            match = next((g for g in s.get("gpus", []) if g.get("index") == gpu_idx), None)
            if match and match.get("power_w") is not None:
                powers_w.append(match["power_w"])
                wall_ts.append(s["wall_ts"])
        if len(powers_w) > 1:
            energy_j = sum(
                (powers_w[i] + powers_w[i + 1]) / 2.0 * (wall_ts[i + 1] - wall_ts[i])
                for i in range(len(powers_w) - 1)
            )
            mean_power_w = statistics.mean(powers_w)

    energy = {
        "energy_j": round(energy_j, 3) if energy_j is not None else None,
        "energy_excl_pause_j": round(energy_j, 3) if energy_j is not None else None,
        "mean_power_w": round(mean_power_w, 3) if mean_power_w is not None else None,
        "mean_power_excl_pause_w": round(mean_power_w, 3) if mean_power_w is not None else None,
    }

    # Determine arm_label from freq_data or run_dir name
    arm_label = run_dir.name
    gpu_freq_mhz_target = freq_data.get("target_mhz") or freq_data.get("target_freq_mhz")
    if gpu_freq_mhz_target is None:
        # Try to infer from run_dir name: phi_tp1_1300mhz → 1300
        name = run_dir.name
        if "uncapped" in name:
            gpu_freq_mhz_target = None
            arm_label = name.split("_")[-1] if "uncapped" in name else name
        elif name.endswith("mhz"):
            try:
                gpu_freq_mhz_target = int(name.split("_")[-1].rstrip("mhz"))
            except ValueError:
                pass

    return {
        "arm_label": arm_label,
        "model": meta.get("model"),
        "gpu_freq_mhz_target": gpu_freq_mhz_target,
        "gpu_freq_mhz_achieved": freq_data.get("under_load_median_mhz"),
        "clock_verdict_ok": freq_data.get("verdict_ok"),
        "tp_size": meta.get("engine_kwargs", {}).get("tensor_parallel_size", 1),
        "num_reqs": ttft["count"],
        "fix_input_length": None,
        "slurm_job_id": None,
        "hostname": None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "timing": timing,
        "latency": latency,
        "energy": energy,
        "throughput": None,
    }


def backfill_run(run_dir: Path, dry_run: bool = False, force: bool = False) -> bool:
    results_dir = run_dir / "results"
    summary_path = results_dir / "summary.json"

    if summary_path.exists() and not force:
        return False  # already has summary

    if not (run_dir / "requests.jsonl").exists():
        return False  # no request data — run failed or incomplete

    summary = compute_summary(run_dir)
    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

    t = summary.get("timing") or {}
    la = (summary.get("latency") or {}).get("ttft") or {}
    en = summary.get("energy") or {}
    print(
        f"{'[DRY] ' if dry_run else ''}Backfilled {run_dir.name}: "
        f"wall={t.get('wall_sec','?'):.1f}s  "
        f"ttft_median={la.get('median_ms','?')}ms  "
        f"energy={en.get('energy_excl_pause_j','?')}J  "
        f"power={en.get('mean_power_excl_pause_w','?')}W"
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="Campaign dir (walks runs/*) or individual run dirs")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written, don't write")
    ap.add_argument("--force", action="store_true", help="overwrite existing summary.json")
    args = ap.parse_args()

    total, backfilled = 0, 0
    for d in args.dirs:
        if not d.is_dir():
            print(f"WARNING: {d} is not a directory, skipping", file=sys.stderr)
            continue
        # Check if this is a campaign dir (has runs/) or a direct run dir
        runs_root = d / "runs"
        if runs_root.is_dir():
            run_dirs = sorted(r for r in runs_root.iterdir() if r.is_dir())
        else:
            run_dirs = [d]

        for run_dir in run_dirs:
            total += 1
            if backfill_run(run_dir, dry_run=args.dry_run, force=args.force):
                backfilled += 1

    print(f"\n{backfilled}/{total} runs backfilled.")


if __name__ == "__main__":
    main()
