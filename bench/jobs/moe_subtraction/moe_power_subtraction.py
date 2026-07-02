#!/usr/bin/env python3
"""MoE power-by-subtraction validation.

Backs the MoE-layer GPU energy out of a REAL all-GPU energy measurement using
the simulator's energy_breakdown.json, and checks it against an independent
expectation built from the profiler's measured MoE per-shot GPU power.

Idea
----
The simulator reports GPU-only ("NPU") energy as::

    npu_total_j = idle_energy_j + moe_active_energy_j + nonmoe_active_energy_j

The sim's *non-MoE* part (idle floor + attention/dense/router active) is the
piece we trust structurally, so we subtract the sim MoE slice from the sim
NPU total to get the non-MoE energy, then back the MoE energy out of the REAL
measurement::

    sim_nonMoE      = npu_total_j - moe_active_energy_j        (from sim)
    MoE_backed_out  = measured_all_gpu_j - sim_nonMoE          (real - sim non-MoE)

Independently, from measured hardware power::

    expected_MoE    = (moe_p95_power_w - idle_power_w) * moe_time_s

    agreement_pct   = 100 * MoE_backed_out / expected_MoE

For the DENSE control (no MoE layers, moe_active_energy_j == 0)::

    control_residual_j   = measured_all_gpu_j - npu_total_j     (should be ~0)
    control_residual_pct = 100 * control_residual_j / measured_all_gpu_j

Usage
-----
    python moe_power_subtraction.py --mode moe   --summary <summary.json> \
        --breakdown <energy_breakdown.json> [--moe-p95-power-w W | --moe-power-glob GLOB]
    python moe_power_subtraction.py --mode dense --summary <summary.json> \
        --breakdown <energy_breakdown.json>
    python moe_power_subtraction.py --selftest
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys


# ----------------------------------------------------------------------
# Pure computation (unit-testable)
# ----------------------------------------------------------------------
def moe_subtraction(measured_all_gpu_j, npu_total_j, moe_active_energy_j,
                    moe_time_s, idle_power_w, moe_p95_power_w):
    """Return the MoE power-by-subtraction result dict."""
    sim_nonMoE = npu_total_j - moe_active_energy_j
    moe_backed_out = measured_all_gpu_j - sim_nonMoE
    expected_moe = (moe_p95_power_w - idle_power_w) * moe_time_s
    agreement_pct = (100.0 * moe_backed_out / expected_moe) if expected_moe else float("nan")
    return {
        "mode": "moe",
        "measured_all_gpu_j": measured_all_gpu_j,
        "npu_total_j": npu_total_j,
        "moe_active_energy_j": moe_active_energy_j,
        "moe_time_s": moe_time_s,
        "idle_power_w": idle_power_w,
        "moe_p95_power_w": moe_p95_power_w,
        "sim_nonMoE_j": round(sim_nonMoE, 3),
        "MoE_backed_out_j": round(moe_backed_out, 3),
        "expected_MoE_j": round(expected_moe, 3),
        "agreement_pct": round(agreement_pct, 3),
    }


def dense_control(measured_all_gpu_j, npu_total_j):
    """Return the dense-control residual dict (MoE_backed_out should be ~0)."""
    residual = measured_all_gpu_j - npu_total_j
    residual_pct = (100.0 * residual / measured_all_gpu_j) if measured_all_gpu_j else float("nan")
    return {
        "mode": "dense",
        "measured_all_gpu_j": measured_all_gpu_j,
        "npu_total_j": npu_total_j,
        "control_residual_j": round(residual, 3),
        "control_residual_pct": round(residual_pct, 3),
    }


def compute_moe_p95_power_w(power_glob):
    """p95 of ``power_w`` over profiler MoE per-shot gpu_power JSONL captures."""
    values = []
    for path in _glob.glob(power_glob):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for gpu in json.loads(line).get("gpus", []):
                    values.append(float(gpu["power_w"]))
    if not values:
        raise ValueError(f"no power_w samples matched {power_glob!r}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[idx]


# Default location of the Qwen1.5-MoE profiler MoE power captures (relative to repo root).
DEFAULT_MOE_POWER_GLOB = (
    "profiler/perf/V100/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16/tp1/gpu_power/moe*.jsonl"
)


# ----------------------------------------------------------------------
# Self-test with fabricated inputs
# ----------------------------------------------------------------------
def _selftest():
    # --- MoE arm: fabricated, self-consistent numbers ---
    r = moe_subtraction(
        measured_all_gpu_j=1000.0,
        npu_total_j=800.0,
        moe_active_energy_j=500.0,
        moe_time_s=10.0,
        idle_power_w=50.0,
        moe_p95_power_w=100.0,
    )
    assert r["sim_nonMoE_j"] == 300.0, r
    assert r["MoE_backed_out_j"] == 700.0, r
    assert r["expected_MoE_j"] == 500.0, r
    assert r["agreement_pct"] == 140.0, r

    # --- Dense control: fabricated ---
    d = dense_control(measured_all_gpu_j=1472.808, npu_total_j=1450.0)
    assert d["control_residual_j"] == 22.808, d
    assert d["control_residual_pct"] == 1.549, d

    # --- p95 helper on a fabricated JSONL ---
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "moe_x.jsonl")
        with open(p, "w") as fh:
            for w in (70, 72, 71, 140, 141, 150, 72, 73, 74, 141):
                fh.write(json.dumps({"gpus": [{"power_w": w}]}) + "\n")
        p95 = compute_moe_p95_power_w(os.path.join(td, "moe*.jsonl"))
        assert p95 == 150, p95

    print("SELFTEST OK")
    print("  fabricated MoE   :", json.dumps(r))
    print("  fabricated dense :", json.dumps(d))
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="MoE power-by-subtraction validation")
    ap.add_argument("--mode", choices=["moe", "dense"], help="arm to evaluate")
    ap.add_argument("--summary", help="real run summary.json (measured all-GPU energy)")
    ap.add_argument("--breakdown", help="sim energy_breakdown.json")
    ap.add_argument("--moe-p95-power-w", type=float, default=None,
                    help="override measured MoE p95 GPU power (W)")
    ap.add_argument("--moe-power-glob", default=DEFAULT_MOE_POWER_GLOB,
                    help="glob for profiler MoE per-shot gpu_power JSONL (used when "
                         "--moe-p95-power-w is not given)")
    ap.add_argument("--selftest", action="store_true", help="run fabricated-input unit tests")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not (args.mode and args.summary and args.breakdown):
        ap.error("--mode, --summary and --breakdown are required unless --selftest")

    with open(args.summary) as fh:
        summary = json.load(fh)
    with open(args.breakdown) as fh:
        bd = json.load(fh)

    measured = float(summary["energy"]["energy_j"])

    if args.mode == "dense":
        result = dense_control(measured, float(bd["npu_total_j"]))
    else:
        moe_p95 = args.moe_p95_power_w
        if moe_p95 is None:
            moe_p95 = compute_moe_p95_power_w(args.moe_power_glob)
        result = moe_subtraction(
            measured_all_gpu_j=measured,
            npu_total_j=float(bd["npu_total_j"]),
            moe_active_energy_j=float(bd["moe_active_energy_j"]),
            moe_time_s=float(bd["moe_time_s"]),
            idle_power_w=float(bd["idle_power_w"]),
            moe_p95_power_w=float(moe_p95),
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
