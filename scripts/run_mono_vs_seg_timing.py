#!/usr/bin/env python3
"""Compare monolithic vs segmented wall time at multiple request counts."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/cluster/single_node_single_instance.json"
DEFAULT_DATASET = "workloads/example_trace.jsonl"
_RE_WALL = re.compile(
    r"Total simulation time:\s+(\d+)h\s+(\d+)m\s+([\d.]+)s"
)
_RE_CLOCKS = re.compile(r"Total clocks \(ns\):\s+(\d+)")


def _parse_output(text: str) -> dict:
    wall_s = None
    if m := _RE_WALL.search(text):
        h, mi, s = m.groups()
        wall_s = int(h) * 3600 + int(mi) * 60 + float(s)
    clocks = int(_RE_CLOCKS.search(text).group(1)) if _RE_CLOCKS.search(text) else None
    return {"wall_s": wall_s, "sim_clocks_ns": clocks}


def run_one(mode: str, num_reqs: int, out_dir: Path, dataset: str) -> dict:
    out_csv = out_dir / f"{mode}_{num_reqs}req.csv"
    log_path = out_dir / f"{mode}_{num_reqs}req.log"
    cmd = [
        sys.executable,
        "-m",
        "serving",
        "--cluster-config",
        CONFIG,
        "--dtype",
        "bfloat16",
        "--block-size",
        "16",
        "--dataset",
        dataset,
        "--output",
        str(out_csv.relative_to(REPO)),
        "--num-reqs",
        str(num_reqs),
        "--no-enable-prefix-caching",
        "--log-level",
        "WARNING",
    ]
    if mode == "seg":
        cmd.append("--forward-segments")
        cmd.append("per_block")

    print(f"\n>>> {mode} num_reqs={num_reqs} ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    host_wall = time.time() - t0
    log_text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(log_text, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(log_text.strip().splitlines()[-15:])
        raise RuntimeError(f"{mode} n={num_reqs} failed:\n{tail}")

    parsed = _parse_output(log_text)
    row = {
        "mode": mode,
        "dataset": dataset,
        "num_reqs": num_reqs,
        "wall_s": parsed["wall_s"] or host_wall,
        "host_wall_s": round(host_wall, 3),
        "sim_clocks_ns": parsed["sim_clocks_ns"],
        "log": str(log_path.relative_to(REPO)),
        "csv": str(out_csv.relative_to(REPO)),
    }
    print(
        f"    wall={row['wall_s']:.1f}s  clocks={row['sim_clocks_ns']}  host={host_wall:.1f}s",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-reqs", nargs="+", type=int, default=[1, 3, 10])
    parser.add_argument(
        "--out-dir",
        default="outputs/branch_compare/timing_study",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Workload JSONL (example_trace has only 10 entries)",
    )
    parser.add_argument("--modes", nargs="+", choices=["mono", "seg"], default=["mono", "seg"])
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n in args.num_reqs:
        for mode in args.modes:
            rows.append(run_one(mode, n, out_dir, args.dataset))

    summary = out_dir / "summary.csv"
    fields = ["mode", "dataset", "num_reqs", "wall_s", "host_wall_s", "sim_clocks_ns", "log", "csv"]
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    mono = {r["num_reqs"]: r for r in rows if r["mode"] == "mono"}
    seg = {r["num_reqs"]: r for r in rows if r["mode"] == "seg"}

    print("\n=== Summary ===")
    print(f"{'N':>4}  {'mono_wall':>10}  {'seg_wall':>10}  {'ratio':>8}  {'mono_s/req':>12}  {'seg_s/req':>12}")
    for n in sorted(set(mono) & set(seg)):
        mw = mono[n]["wall_s"]
        sw = seg[n]["wall_s"]
        ratio = sw / mw if mw else 0
        print(
            f"{n:4d}  {mw:10.1f}  {sw:10.1f}  {ratio:8.1f}x  "
            f"{mw/n:12.2f}  {sw/n:12.2f}"
        )
    print(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
