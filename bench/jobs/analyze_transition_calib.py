#!/usr/bin/env python3
"""Analyze transition-calib campaign results: barrier overhead by mode (async vs sync).

Walks one or more tcalib campaign directories, pairs n>0 runs with their n=0
baseline, and reports how much each DVFS barrier blocks the vLLM worker thread.

Usage:
    python3 bench/jobs/analyze_transition_calib.py [campaign_dir ...] [--stage-to DIR]

    If no campaign_dir given, scans bench/campaigns/ for dirs whose name contains
    "tcalib".  --stage-to writes TSV + JSON to RESULTS_DIR/ for git commit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_campaigns(repo_root):
    camp_root = repo_root / "bench" / "campaigns"
    if not camp_root.is_dir():
        return []
    return sorted(d for d in camp_root.iterdir() if d.is_dir() and "tcalib" in d.name)


def _collect_runs(campaign_dir):
    """Return list of dicts with raw per-run data, or [] if campaign not ready."""
    manifest = _load_json(campaign_dir / "manifest.json")
    if not manifest:
        return []

    rows = []
    for entry in manifest.get("runs", []):
        run_id = entry["run_id"]
        run_dir = campaign_dir / "runs" / run_id

        spec = _load_json(run_dir / "calib_spec.json")
        if not spec:
            continue

        metrics_path = run_dir / "bench" / "run_exec_metrics.json"
        metrics = _load_json(metrics_path)
        if not metrics:
            # Try artifacts copy
            metrics = _load_json(run_dir / "artifacts" / "run_exec_metrics.json")
        if not metrics:
            rows.append({
                "run_id": run_id,
                "campaign": campaign_dir.name,
                "status": "missing",
                "model_key": spec.get("model_key", "?"),
                "dvfs_apply_mode": spec.get("dvfs_apply_mode", "?"),
                "n_transitions": spec["dvfs"]["n_transitions"],
                "iteration": spec.get("iteration", 0),
            })
            continue

        timing = metrics.get("timing", {})
        dvfs = metrics.get("dvfs", {})
        latency = metrics.get("latency") or {}
        ttft = latency.get("ttft") or {}

        rows.append({
            "run_id": run_id,
            "campaign": campaign_dir.name,
            "status": "ok",
            "model_key": spec.get("model_key", "?"),
            "dvfs_apply_mode": spec.get("dvfs_apply_mode", "?"),
            "n_transitions": spec["dvfs"]["n_transitions"],
            "barrier_layers": spec["dvfs"].get("barrier_layers", []),
            "freq_schedule_mhz": spec["dvfs"].get("freq_schedule_mhz", []),
            "iteration": spec.get("iteration", 0),
            "wall_sec": timing.get("wall_sec"),
            "exec_sec": timing.get("exec_sec"),
            "pause_sec": timing.get("pause_sec"),
            "barrier_wait_sec_worker": timing.get("barrier_wait_sec_worker"),
            "barrier_count": timing.get("barrier_count"),
            "marker_count": dvfs.get("marker_count"),
            "dvfs_modes": dvfs.get("modes", []),
            "ttft_ms": ttft.get("mean_ms"),
        })

    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(all_rows):
    """Pair each n>0 run with its n=0 baseline; compute overheads."""
    # Build baseline lookup: (model_key, dvfs_apply_mode) -> list of exec_sec
    baselines = {}
    for r in all_rows:
        if r["status"] == "ok" and r["n_transitions"] == 0:
            key = (r["model_key"], r["dvfs_apply_mode"])
            baselines.setdefault(key, []).append(r["exec_sec"])

    baseline_exec = {k: mean(v) for k, v in baselines.items()}

    # Also try cross-mode baselines (n=0 async baseline usable for async mode etc.)
    # For runs where apply_mode baseline is missing, fall back to any n=0 same model
    any_baseline = {}
    for r in all_rows:
        if r["status"] == "ok" and r["n_transitions"] == 0:
            any_baseline.setdefault(r["model_key"], []).append(r["exec_sec"])
    any_baseline_exec = {k: mean(v) for k, v in any_baseline.items()}

    results = []
    for r in all_rows:
        if r["status"] != "ok" or r["n_transitions"] == 0:
            continue

        key = (r["model_key"], r["dvfs_apply_mode"])
        base = baseline_exec.get(key) or any_baseline_exec.get(r["model_key"])

        bw = r.get("barrier_wait_sec_worker") or 0.0
        n = r["n_transitions"] or 1
        per_barrier = bw / n if n > 0 else None

        results.append({
            "campaign": r["campaign"],
            "run_id": r["run_id"],
            "model_key": r["model_key"],
            "dvfs_apply_mode": r["dvfs_apply_mode"],
            "n_transitions": r["n_transitions"],
            "exec_sec": r["exec_sec"],
            "wall_sec": r["wall_sec"],
            "barrier_wait_sec_worker": bw,
            "per_barrier_sec": round(per_barrier, 4) if per_barrier is not None else None,
            "exec_overhead_vs_baseline_sec": round(r["exec_sec"] - base, 4) if base else None,
            "wall_overhead_vs_baseline_sec": None,  # filled below
            "baseline_exec_sec": round(base, 4) if base else None,
            "marker_count": r.get("marker_count"),
            "barrier_count": r.get("barrier_count"),
            "ttft_ms": r.get("ttft_ms"),
            "freq_schedule_mhz": r.get("freq_schedule_mhz", []),
            "barrier_layers": r.get("barrier_layers", []),
        })

    # Wall overhead: need baseline wall_sec per (model, mode)
    wall_baselines = {}
    for r in all_rows:
        if r["status"] == "ok" and r["n_transitions"] == 0 and r.get("wall_sec"):
            key = (r["model_key"], r["dvfs_apply_mode"])
            wall_baselines.setdefault(key, []).append(r["wall_sec"])
    wall_baseline_mean = {k: mean(v) for k, v in wall_baselines.items()}
    any_wall_baseline = {}
    for r in all_rows:
        if r["status"] == "ok" and r["n_transitions"] == 0 and r.get("wall_sec"):
            any_wall_baseline.setdefault(r["model_key"], []).append(r["wall_sec"])
    any_wall_mean = {k: mean(v) for k, v in any_wall_baseline.items()}

    for res in results:
        key = (res["model_key"], res["dvfs_apply_mode"])
        base_wall = wall_baseline_mean.get(key) or any_wall_mean.get(res["model_key"])
        if base_wall and res["wall_sec"]:
            res["wall_overhead_vs_baseline_sec"] = round(res["wall_sec"] - base_wall, 4)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v, precision=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def print_table(results):
    if not results:
        print("No completed n>0 runs found.")
        return

    cols = [
        ("model", 8, lambda r: r["model_key"]),
        ("mode", 7, lambda r: r["dvfs_apply_mode"]),
        ("n_trans", 7, lambda r: str(r["n_transitions"])),
        ("per_barrier_s", 13, lambda r: _fmt(r["per_barrier_sec"])),
        ("barrier_wait_s", 14, lambda r: _fmt(r["barrier_wait_sec_worker"])),
        ("exec_s", 8, lambda r: _fmt(r["exec_sec"])),
        ("exec_ovhd_s", 11, lambda r: _fmt(r["exec_overhead_vs_baseline_sec"])),
        ("wall_ovhd_s", 11, lambda r: _fmt(r["wall_overhead_vs_baseline_sec"])),
        ("markers_ok", 10, lambda r: "Y" if r["marker_count"] and r["marker_count"] >= r["n_transitions"] else f"{r['marker_count']}/{r['n_transitions']}"),
        ("campaign", 35, lambda r: r["campaign"]),
    ]
    header = "  ".join(name.ljust(w) for name, w, _ in cols)
    sep = "  ".join("-" * w for name, w, _ in cols)
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: (x["model_key"], x["dvfs_apply_mode"], x["n_transitions"])):
        row = "  ".join(fn(r).ljust(w) for name, w, fn in cols)
        print(row)

    print()
    print("Key: per_barrier_s = barrier_wait_sec_worker / n_transitions")
    print("     exec_ovhd_s   = exec_sec(n>0) - exec_sec(baseline n=0) [N/A = no baseline yet]")
    print("     wall_ovhd_s   = wall_sec(n>0) - wall_sec(baseline n=0)")
    print("     markers_ok    = dvfs.marker_count >= n_transitions (drain-race check)")


def write_outputs(results, all_rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_dir / "calib_analysis.tsv"
    if results:
        fieldnames = list(results[0].keys())
        with tsv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = dict(r)
                row["freq_schedule_mhz"] = json.dumps(row["freq_schedule_mhz"])
                row["barrier_layers"] = json.dumps(row["barrier_layers"])
                w.writerow(row)
        print(f"Wrote {tsv_path}")

    missing = [r for r in all_rows if r["status"] == "missing"]
    json_path = out_dir / "calib_analysis.json"
    json_path.write_text(json.dumps({
        "results": results,
        "missing_runs": [{"run_id": r["run_id"], "campaign": r["campaign"]} for r in missing],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "campaign_dirs", nargs="*", type=Path,
        help="Campaign dir(s) to analyze. Default: all tcalib dirs in bench/campaigns/",
    )
    ap.add_argument(
        "--stage-to", type=Path, metavar="RESULTS_DIR",
        help="Write TSV + JSON to RESULTS_DIR/ for git commit",
    )
    args = ap.parse_args()

    campaign_dirs = args.campaign_dirs or _find_campaigns(repo_root)
    if not campaign_dirs:
        print("No tcalib campaign directories found.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for d in campaign_dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"WARN: {d} not a directory, skipping", file=sys.stderr)
            continue
        rows = _collect_runs(d)
        if not rows:
            print(f"  {d.name}: no manifest or no runs", file=sys.stderr)
        else:
            ok = sum(1 for r in rows if r["status"] == "ok")
            miss = sum(1 for r in rows if r["status"] == "missing")
            print(f"  {d.name}: {ok} ok, {miss} missing")
        all_rows.extend(rows)

    if not all_rows:
        print("No runs found in any campaign.", file=sys.stderr)
        sys.exit(1)

    results = analyze(all_rows)

    print()
    print_table(results)
    print()

    # Decide where to write output files
    if args.stage_to:
        out_dir = Path(args.stage_to)
        write_outputs(results, all_rows, out_dir)
        print()
        print("To commit:")
        print(f"  cd {repo_root}")
        print(f"  git add {out_dir.relative_to(repo_root) if out_dir.is_relative_to(repo_root) else out_dir}")
        print(f"  git commit -m 'results: calib analysis'")
    elif campaign_dirs:
        first = Path(campaign_dirs[0])
        if first.is_dir():
            write_outputs(results, all_rows, first)


if __name__ == "__main__":
    main()
