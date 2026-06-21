#!/usr/bin/env python3
"""Benchmark segmented runs with --layer-hardware-alternate (RTXPRO6000 <-> A6000)."""

from __future__ import annotations

import argparse
import csv
import json
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
_TRACE_ROOTS = (
    "inputs/trace/RTXPRO6000/meta-llama",
    "inputs/trace/A6000/meta-llama",
    "inputs/workload/RTXPRO6000/meta-llama",
    "inputs/workload/A6000/meta-llama",
)


def _parse_output(text: str) -> dict:
    wall_s = None
    if m := _RE_WALL.search(text):
        h, mi, s = m.groups()
        wall_s = int(h) * 3600 + int(mi) * 60 + float(s)
    clocks = int(_RE_CLOCKS.search(text).group(1)) if _RE_CLOCKS.search(text) else None
    return {"wall_s": wall_s, "sim_clocks_ns": clocks}


def clear_hw_cache() -> None:
    for rel in _TRACE_ROOTS:
        path = REPO / rel
        if path.exists():
            import shutil
            shutil.rmtree(path)


def count_layer_switches(events_path: Path) -> dict:
    if not events_path.is_file():
        return {"layer_switches": 0, "invalid": 0}
    switches = 0
    invalid = 0
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("kind") != "dvfs_switch":
                continue
            if ev.get("trigger") != "layer":
                continue
            switches += 1
            old_hw = ev.get("old_hardware")
            new_hw = ev.get("new_hardware")
            if old_hw == new_hw:
                invalid += 1
    return {"layer_switches": switches, "invalid": invalid}


def run_alternate(num_reqs: int, out_dir: Path, dataset: str, clear_cache: bool) -> dict:
    if clear_cache:
        clear_hw_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seg_alternate_{num_reqs}req.csv"
    log_path = out_dir / f"seg_alternate_{num_reqs}req.log"
    events_path = out_dir / f"seg_alternate_{num_reqs}req_events.jsonl"
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
        "--forward-segments",
        "per_block",
        "--layer-hardware-alternate",
        "--work-events",
        str(events_path.relative_to(REPO)),
        "--log-level",
        "INFO",
    ]
    print(f"\n>>> seg+alternate num_reqs={num_reqs} clear_cache={clear_cache} ...", flush=True)
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
        tail = "\n".join(log_text.strip().splitlines()[-20:])
        raise RuntimeError(f"seg+alternate n={num_reqs} failed:\n{tail}")

    parsed = _parse_output(log_text)
    switches = count_layer_switches(events_path)
    reuse_hits = len(re.findall(r"Reusing cached", log_text))
    row = {
        "mode": "seg_alternate",
        "dataset": dataset,
        "num_reqs": num_reqs,
        "wall_s": parsed["wall_s"] or host_wall,
        "host_wall_s": round(host_wall, 3),
        "sim_clocks_ns": parsed["sim_clocks_ns"],
        "layer_switches": switches["layer_switches"],
        "invalid_switches": switches["invalid"],
        "cache_reuse_log_hits": reuse_hits,
        "log": str(log_path.relative_to(REPO)),
        "csv": str(out_csv.relative_to(REPO)),
        "events": str(events_path.relative_to(REPO)),
    }
    print(
        f"    wall={row['wall_s']:.1f}s  clocks={row['sim_clocks_ns']}  "
        f"switches={row['layer_switches']}  cache_hits={reuse_hits}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-reqs", nargs="+", type=int, default=[1, 10])
    parser.add_argument(
        "--out-dir",
        default="outputs/branch_compare/dvfs_speedup/layer_hw_alternate",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--no-clear-cache",
        action="store_true",
        help="keep existing trace/workload cache (warm rerun)",
    )
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    rows = [
        run_alternate(n, out_dir, args.dataset, clear_cache=not args.no_clear_cache)
        for n in args.num_reqs
    ]

    summary = out_dir / "summary.csv"
    fields = list(rows[0].keys())
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
