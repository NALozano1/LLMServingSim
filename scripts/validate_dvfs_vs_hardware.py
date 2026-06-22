#!/usr/bin/env python3
"""Compare LLMServingSim layer-boundary DVFS runs against real ARC V100 bench runs.

Reads simulator logs (per run_id) and the synced bench ground truth, matches by
run_id, and writes a clearly-labelled comparison into the validation folder:
    validation/layer_boundary_dvfs/comparison.{csv,md}

It records the profiler clock-audit status in the header so a comparison built on
suspect (un-re-profiled) data can never be mistaken for a clean one.

Usage:
    python3 scripts/validate_dvfs_vs_hardware.py \
        --sim-runs   outputs/mini3x3_layer_dvfs_sim/runs \
        --bench-runs <synced ARC bench>/runs \
        --out        validation/layer_boundary_dvfs

Sim side  : <sim-runs>/<run_id>/sim.log   (Total energy / NPU energy / Total clocks / TTFT)
Bench side: <bench-runs>/<run_id>/results/summary.json  (exec_sec, energy_j, energy_excl_pause_j)

If the bench side is absent the run is reported PENDING (sim-only) rather than failing,
so this can be run before the real data lands.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

_RE_TOTAL_ENERGY_KJ = re.compile(r"Total energy consumption \(kJ\):\s+([0-9.]+)")
_RE_NPU_ENERGY_J = re.compile(r"NPU energy consumption \(J\):\s+([0-9.]+)")
_RE_CLOCKS_NS = re.compile(r"Total clocks \(ns\):\s+(\d+)")
_RE_TTFT_MS = re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)")


def parse_sim_log(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    if m := _RE_TOTAL_ENERGY_KJ.search(text):
        out["sim_total_energy_j"] = float(m.group(1)) * 1000.0
    if m := _RE_NPU_ENERGY_J.search(text):
        out["sim_gpu_energy_j"] = float(m.group(1))
    if m := _RE_CLOCKS_NS.search(text):
        out["sim_latency_s"] = int(m.group(1)) / 1e9
    if m := _RE_TTFT_MS.search(text):
        out["sim_ttft_ms"] = float(m.group(1))
    return out or None


def parse_bench_summary(run_dir: Path) -> dict | None:
    """Pull exec_sec / energy from the bench ground truth, tolerating layout drift."""
    for rel in ("results/summary.json", "bench/run_exec_metrics.json", "summary.json"):
        p = run_dir / rel
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out: dict = {}
        for k in ("exec_sec", "measured_exec_sec", "effective_runtime_sec"):
            if isinstance(d.get(k), (int, float)):
                out["real_exec_s"] = float(d[k])
                break
        for k in ("energy_excl_pause_j", "effective_energy_j", "energy_j"):
            if isinstance(d.get(k), (int, float)):
                out["real_energy_j"] = float(d[k])
                out["real_energy_field"] = k
                break
        if out:
            out["real_source"] = str(p)
            return out
    return None


def clock_audit_status(repo: Path, bench_runs: Path | None) -> str:
    """Best-effort note on whether the feeding profiles passed the clock audit."""
    audit = repo / "profiler" / "jobs" / "audit_gpu_clocks.py"
    perf = repo / "profiler" / "perf"
    if not audit.is_file() or not perf.is_dir():
        return "UNKNOWN (audit tool or profiler/perf missing)"
    try:
        r = subprocess.run(
            ["python3", str(audit), str(perf / "V100_900MHz"), str(perf / "V100_1100MHz")],
            capture_output=True, text=True, timeout=120,
        )
        return "CLEAN (clock audit passed)" if r.returncode == 0 else \
               "PROFILES_SUSPECT (clock audit flagged V100 profiles — re-profile before trusting)"
    except Exception:
        return "UNKNOWN (audit did not run)"


def pct_err(sim, real):
    if sim is None or real in (None, 0):
        return None
    return (sim - real) / real * 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-runs", default="outputs/mini3x3_layer_dvfs_sim/runs")
    ap.add_argument("--bench-runs", default=None,
                    help="synced ARC bench runs dir (<run_id>/results/summary.json). Omit for sim-only.")
    ap.add_argument("--out", default="validation/layer_boundary_dvfs")
    ap.add_argument("--energy-basis", choices=["gpu", "total"], default="gpu",
                    help="which sim energy to compare against bench energy_j (default: gpu/NPU-only)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sim_runs = Path(args.sim_runs)
    bench_runs = Path(args.bench_runs) if args.bench_runs else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ids = sorted(p.name for p in sim_runs.iterdir() if p.is_dir()) if sim_runs.is_dir() else []
    if not run_ids:
        print(f"No sim runs under {sim_runs}. Run the mini3x3 sim driver first.")
        return 1

    audit = clock_audit_status(repo, bench_runs)
    rows = []
    for run_id in run_ids:
        sim = parse_sim_log(sim_runs / run_id / "sim.log") or {}
        real = parse_bench_summary(bench_runs / run_id) if bench_runs else None
        sim_energy = sim.get("sim_gpu_energy_j") if args.energy_basis == "gpu" else sim.get("sim_total_energy_j")
        row = {
            "run_id": run_id,
            "sim_latency_s": sim.get("sim_latency_s"),
            "sim_energy_j": sim_energy,
            "sim_energy_basis": args.energy_basis,
            "real_exec_s": (real or {}).get("real_exec_s"),
            "real_energy_j": (real or {}).get("real_energy_j"),
            "real_energy_field": (real or {}).get("real_energy_field"),
            "latency_pct_err": pct_err(sim.get("sim_latency_s"), (real or {}).get("real_exec_s")),
            "energy_pct_err": pct_err(sim_energy, (real or {}).get("real_energy_j")),
            "status": "OK" if real else "PENDING (sim-only; no bench data)",
        }
        rows.append(row)

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v, suf=""):
        return f"{v:.3f}{suf}" if isinstance(v, (int, float)) else "—"

    matched = [r for r in rows if r["status"] == "OK"]
    md = [
        "# Layer-boundary DVFS — sim vs real V100 comparison",
        "",
        f"- Profile clock-audit status: **{audit}**",
        f"- Sim energy basis: **{args.energy_basis}** ("
        + ("NPU/GPU-only" if args.energy_basis == "gpu" else "total system") + ")",
        f"- Bench data: {'present' if bench_runs else 'NOT PROVIDED — sim-only, comparison PENDING'}",
        "",
        "| run_id | sim lat (s) | real exec (s) | lat %err | sim E (J) | real E (J) | E %err | status |",
        "|--------|------------:|--------------:|---------:|----------:|-----------:|-------:|--------|",
    ]
    for r in rows:
        md.append(
            f"| {r['run_id']} | {fmt(r['sim_latency_s'])} | {fmt(r['real_exec_s'])} | "
            f"{fmt(r['latency_pct_err'],'%')} | {fmt(r['sim_energy_j'])} | {fmt(r['real_energy_j'])} | "
            f"{fmt(r['energy_pct_err'],'%')} | {r['status']} |"
        )
    if matched:
        e = [abs(r["energy_pct_err"]) for r in matched if r["energy_pct_err"] is not None]
        l = [abs(r["latency_pct_err"]) for r in matched if r["latency_pct_err"] is not None]
        md += ["", f"**Mean |error|** over {len(matched)} matched runs — "
               f"energy {sum(e)/len(e):.1f}%, latency {sum(l)/len(l):.1f}%" if e and l else ""]
    out_md = out_dir / "comparison.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path} and {out_md}")
    print(f"Clock-audit status: {audit}")
    print(f"Runs: {len(rows)}  matched-with-bench: {len(matched)}  pending: {len(rows)-len(matched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
