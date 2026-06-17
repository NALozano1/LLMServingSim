#!/usr/bin/env python3
"""Run monolithic Qwen3-30B-A3B sims at each bundled V100 clock profile (no DVFS switching).

Writes per-run CSV/logs plus a summary table with simulated runtime and energy.

Example:
  python3 scripts/run_v100_qwen_mhz_matrix.py
  python3 scripts/run_v100_qwen_mhz_matrix.py --num-reqs 1 --hardware V100 V100_700MHz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs/cluster/single_node_v100_qwen_moe_power.json"
DEFAULT_HARDWARE = [
    "V100",
    "V100_700MHz",
    "V100_900MHz",
    "V100_1100MHz",
    "V100_1300MHz",
    "V100_1400MHz",
]
V100_REF_MHZ = 1530  # SXM2 boost; used to scale active NPU power with clock
V100_ACTIVE_W_AT_REF = 300.0
V100_IDLE_W = 28.0
V100_STANDBY_W = 220.0

_RE_CLOCKS = re.compile(r"Total clocks \(ns\):\s+(\d+)")
_RE_ENERGY_KJ = re.compile(r"Total energy consumption \(kJ\):\s+([\d.]+)")
_RE_SIM_S = re.compile(r"Total latency \(s\):\s+([\d.]+)")
_RE_WALL = re.compile(
    r"Total simulation time:\s+(\d+)h\s+(\d+)m\s+([\d.]+)s"
)


def _mhz_from_hardware(hardware: str) -> int:
    if hardware == "V100":
        return 1250
    match = re.match(r"^V100_(\d+)MHz$", hardware)
    if not match:
        raise ValueError(f"unsupported V100 hardware tag: {hardware}")
    return int(match.group(1))


def _npu_power_entry(hardware: str) -> dict:
    mhz = _mhz_from_hardware(hardware)
    scale = mhz / V100_REF_MHZ
    return {
        "idle_power": V100_IDLE_W,
        "standby_power": V100_STANDBY_W,
        "active_power": round(V100_ACTIVE_W_AT_REF * scale, 2),
        "standby_duration": 18,
    }


def _build_cluster_config(hardware: str) -> dict:
    cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    inst = cfg["nodes"][0]["instances"][0]
    inst["hardware"] = hardware
    # MoE weights are ~56GB at tp1; bundled V100 profiles are tp1 timing grids.
    # Relax HBM floor so the sim can run — latencies still come from profiler CSVs.
    inst["npu_mem"]["mem_size"] = 64
    npu = cfg["nodes"][0]["power"]["npu"]
    npu.clear()
    npu[hardware] = _npu_power_entry(hardware)
    return cfg


def _parse_run_log(text: str) -> dict:
    out = {
        "sim_clocks_ns": None,
        "sim_latency_s": None,
        "total_energy_kj": None,
        "wall_time_s": None,
    }
    if m := _RE_CLOCKS.search(text):
        out["sim_clocks_ns"] = int(m.group(1))
    if m := _RE_SIM_S.search(text):
        out["sim_latency_s"] = float(m.group(1))
    if m := _RE_ENERGY_KJ.search(text):
        out["total_energy_kj"] = float(m.group(1))
    if m := _RE_WALL.search(text):
        h, mi, s = m.groups()
        out["wall_time_s"] = int(h) * 3600 + int(mi) * 60 + float(s)
    return out


def _read_request_csv(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[0]
    return {
        "request_latency_ns": int(row["latency"]),
        "ttft_ns": int(row["TTFT"]),
        "tpot_ns": int(row["TPOT"]),
    }


def run_one(
    hardware: str,
    *,
    out_dir: Path,
    num_reqs: int,
    dataset: str,
    log_level: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "cluster.json"
    cfg_path.write_text(
        json.dumps(_build_cluster_config(hardware), indent=4) + "\n",
        encoding="utf-8",
    )
    run_csv = out_dir / "run.csv"
    run_log = out_dir / "run.log"
    mhz = _mhz_from_hardware(hardware)

    cmd = [
        sys.executable,
        "-m",
        "serving",
        "--cluster-config",
        str(cfg_path.relative_to(REPO)),
        "--dtype",
        "float16",
        "--block-size",
        "16",
        "--dataset",
        dataset,
        "--output",
        str(run_csv.relative_to(REPO)),
        "--num-reqs",
        str(num_reqs),
        "--log-interval",
        "1.0",
        "--log-level",
        log_level,
        "--no-enable-prefix-caching",
    ]

    print(f"\n=== {hardware} ({mhz} MHz) ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    wall_s = time.time() - t0
    log_text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    run_log.write_text(log_text, encoding="utf-8")

    if proc.returncode != 0:
        tail = "\n".join(log_text.strip().splitlines()[-20:])
        raise RuntimeError(f"{hardware} failed (exit {proc.returncode}):\n{tail}")

    parsed = _parse_run_log(log_text)
    parsed.update(_read_request_csv(run_csv))
    parsed.update(
        {
            "hardware": hardware,
            "mhz": mhz,
            "active_power_w": _npu_power_entry(hardware)["active_power"],
            "wall_time_s": parsed.get("wall_time_s") or wall_s,
            "run_csv": str(run_csv.relative_to(REPO)),
            "run_log": str(run_log.relative_to(REPO)),
        }
    )
    return parsed


def _write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "hardware",
        "mhz",
        "active_power_w",
        "sim_clocks_ns",
        "sim_latency_s",
        "request_latency_ns",
        "ttft_ns",
        "tpot_ns",
        "total_energy_kj",
        "wall_time_s",
        "run_csv",
        "run_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        nargs="+",
        default=DEFAULT_HARDWARE,
        help="V100 profiler hardware tags to run",
    )
    parser.add_argument("--num-reqs", type=int, default=1)
    parser.add_argument(
        "--dataset",
        default="workloads/example_trace.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/v100_qwen_mhz_matrix",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    out_root = REPO / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for hardware in args.hardware:
        row = run_one(
            hardware,
            out_dir=out_root / hardware,
            num_reqs=args.num_reqs,
            dataset=args.dataset,
            log_level=args.log_level,
        )
        summary_rows.append(row)
        print(
            f"  sim_clocks={row.get('sim_clocks_ns')} ns  "
            f"energy={row.get('total_energy_kj')} kJ  "
            f"wall={row.get('wall_time_s'):.1f}s",
            flush=True,
        )

    summary_path = out_root / "summary.csv"
    _write_summary(summary_rows, summary_path)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"{row['hardware']:14} {row['mhz']:5} MHz  "
            f"clocks={row.get('sim_clocks_ns')} ns  "
            f"energy={row.get('total_energy_kj')} kJ  "
            f"latency={row.get('request_latency_ns')} ns"
        )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
