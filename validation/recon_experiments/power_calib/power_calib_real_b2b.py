#!/usr/bin/env python3
"""Real host-free gapfree (B2B) smoke: inplace + CUDA-event latency (± CUDA graph).

Compared to legacy power_calib.measure_active_power_gapfree:
  - inplace=True  (no per-iter empty_like alloc)
  - CUDA-event timing for kernel latency (not wall_s/iter_count alone)
  - optional CUDA Graph replay to remove Python launch gaps
  - concurrent NVML power sampling (same plateau logic)

Outputs a dedicated CSV so legacy power_calib_*.csv is untouched.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import torch

# Reuse helpers from the existing calib harness.
from power_calib import (  # noqa: E402
    PowerSampler,
    _resolve_model_dims,
    build_kernel_inputs,
    measure_idle_power,
    sample_clock_mhz,
    warmup_gpu,
)

DTYPE = torch.float16


def _try_capture_graph(
    hidden_states,
    w1,
    w2,
    topk_ids,
    topk_weights,
    expert_map,
    n_experts_global,
):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Warmup on side stream before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                global_num_experts=n_experts_global,
                expert_map=expert_map,
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            global_num_experts=n_experts_global,
            expert_map=expert_map,
        )
    return g


def measure_real_b2b(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    n_experts_global: int,
    window_s: float = 9.0,
    chunk_s: float = 1.0,
    ramp_s: float = 1.5,
    poll_ms: int = 25,
    use_graph: bool = True,
    batch: int = 0,
    ae: int = 0,
    freq: int = 0,
) -> dict:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    mode = "graph" if use_graph else "inplace"
    print(
        f"\n[real-b2b/{mode}] batch={batch} ae={ae} freq={freq}MHz  "
        f"window={window_s:.1f}s chunk={chunk_s:.1f}s",
        flush=True,
    )

    # Warmup (inplace — no alloc).
    for _ in range(8):
        fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            global_num_experts=n_experts_global,
            expert_map=expert_map,
        )
    torch.cuda.synchronize()

    graph = None
    if use_graph:
        try:
            graph = _try_capture_graph(
                hidden_states, w1, w2, topk_ids, topk_weights, expert_map, n_experts_global
            )
            print("[real-b2b] CUDA Graph capture OK", flush=True)
        except Exception as e:
            print(f"[real-b2b] CUDA Graph capture FAILED ({e}); falling back to inplace loop", flush=True)
            graph = None
            mode = "inplace"

    sampler = PowerSampler(poll_ms=poll_ms)
    sampler.start()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    t_wall0 = time.perf_counter()
    t_chunk = t_wall0
    iter_count = 0

    start_ev.record()
    while True:
        t_now = time.perf_counter()
        if t_now - t_wall0 >= window_s:
            break

        if graph is not None:
            graph.replay()
        else:
            fused_experts(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                global_num_experts=n_experts_global,
                expert_map=expert_map,
            )
        iter_count += 1

        # Bound queue depth only in non-graph mode (graphs are single-kernel replays).
        if graph is None:
            t_now = time.perf_counter()
            if t_now - t_chunk >= chunk_s:
                torch.cuda.synchronize()
                t_chunk = time.perf_counter()

    end_ev.record()
    end_ev.synchronize()
    t_wall1 = time.perf_counter()

    records = sampler.stop()
    total_wall_s = t_wall1 - t_wall0
    cuda_ms_total = float(start_ev.elapsed_time(end_ev))
    cuda_latency_ms = cuda_ms_total / max(iter_count, 1)
    wall_latency_ms = (total_wall_s * 1000.0) / max(iter_count, 1)

    plateau = [(t, p) for t, p in records if t >= ramp_s] or records
    powers = [p for _, p in plateau]
    p_active = median(powers) if powers else float("nan")
    p_mean = sum(powers) / len(powers) if powers else float("nan")
    if len(powers) > 1:
        var = sum((p - p_mean) ** 2 for p in powers) / len(powers)
        cv_pct = 100.0 * (var ** 0.5) / p_mean if p_mean else float("nan")
    else:
        cv_pct = float("nan")

    print(
        f"[real-b2b/{mode}] iters={iter_count} wall={total_wall_s:.2f}s  "
        f"cuda_lat={cuda_latency_ms:.4f}ms  wall_lat={wall_latency_ms:.4f}ms  "
        f"P_active={p_active:.2f}W cv={cv_pct:.2f}%",
        flush=True,
    )

    return {
        "mode": mode,
        "iter_count": iter_count,
        "wall_s": round(total_wall_s, 3),
        "cuda_latency_ms": round(cuda_latency_ms, 6),
        "wall_latency_ms": round(wall_latency_ms, 6),
        "P_active_gapfree_w": round(p_active, 3) if powers else float("nan"),
        "plateau_cv_pct": round(cv_pct, 2) if powers else float("nan"),
        "plateau_n": len(powers),
    }


def run_sweep(
    model_id: str,
    freq: int,
    batches: list[int],
    ae: int,
    hidden_size: int,
    moe_inter: int,
    n_experts_global: int,
    window_s: float,
    chunk_s: float,
    ramp_s: float,
    poll_ms: int,
    use_graph: bool,
    out_csv: Path,
) -> list[dict]:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", flush=True)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"[cuda] Device: {device_name}", flush=True)
    print(f"[clock] target={freq}MHz achieved_before={sample_clock_mhz()}MHz", flush=True)

    warmup_gpu()
    p_idle = measure_idle_power(duration_s=2.0, poll_ms=poll_ms)

    rows: list[dict] = []
    for batch in batches:
        print(f"\n{'='*70}\nREAL-B2B  MODEL={model_id} BATCH={batch} ae={ae} freq={freq}\n{'='*70}", flush=True)
        hs, w1, w2, topk_ids, topk_weights, expert_map = build_kernel_inputs(
            batch=batch,
            ae=ae,
            n_experts_global=n_experts_global,
            hidden_size=hidden_size,
            moe_inter=moe_inter,
        )

        # Short burst to bring clocks up.
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        for _ in range(5):
            fused_experts(
                hidden_states=hs, w1=w1, w2=w2,
                topk_weights=topk_weights, topk_ids=topk_ids,
                inplace=True,
                global_num_experts=n_experts_global,
                expert_map=expert_map,
            )
        torch.cuda.synchronize()

        result = measure_real_b2b(
            hidden_states=hs,
            w1=w1,
            w2=w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            n_experts_global=n_experts_global,
            window_s=window_s,
            chunk_s=chunk_s,
            ramp_s=ramp_s,
            poll_ms=poll_ms,
            use_graph=use_graph,
            batch=batch,
            ae=ae,
            freq=freq,
        )

        achieved = sample_clock_mhz()
        verdict_ok = achieved is not None and abs(achieved - freq) <= 30
        row = {
            "device": device_name,
            "model": model_id,
            "freq_mhz": freq,
            "batch": batch,
            "ae": ae,
            "mode": result["mode"],
            "cuda_latency_ms": result["cuda_latency_ms"],
            "wall_latency_ms": result["wall_latency_ms"],
            "P_active_gapfree_w": result["P_active_gapfree_w"],
            "plateau_cv_pct": result["plateau_cv_pct"],
            "P_idle_w": round(p_idle, 3),
            "P_dynamic_w": round(result["P_active_gapfree_w"] - p_idle, 3)
            if result["P_active_gapfree_w"] == result["P_active_gapfree_w"]
            else "",
            "achieved_mhz": achieved if achieved is not None else "",
            "verdict_ok": verdict_ok,
            "plateau_n": result["plateau_n"],
            "iter_count": result["iter_count"],
            "wall_s": result["wall_s"],
        }
        rows.append(row)
        print(
            f"[result] batch={batch} mode={result['mode']} "
            f"cuda_ms={result['cuda_latency_ms']:.4f} wall_ms={result['wall_latency_ms']:.4f} "
            f"P={result['P_active_gapfree_w']:.2f}W verdict_ok={verdict_ok}",
            flush=True,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\n[output] Wrote {len(rows)} rows → {out_csv}", flush=True)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--freq", type=int, default=900)
    p.add_argument("--batches", type=str, default="8,32,512,8192")
    p.add_argument("--ae", type=int, default=None)
    p.add_argument("--window", type=float, default=9.0)
    p.add_argument("--chunk", type=float, default=1.0)
    p.add_argument("--ramp", type=float, default=1.5)
    p.add_argument("--poll-ms", type=int, default=25)
    p.add_argument("--no-graph", action="store_true", help="Disable CUDA Graph; inplace loop only")
    p.add_argument("--out-csv", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hidden_size, moe_inter, n_experts_global, top_k = _resolve_model_dims(args.model)
    except Exception as e:
        print(f"ERROR resolving model dims: {e}", flush=True)
        return 1

    ae = args.ae if args.ae is not None else top_k
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    print(
        f"[config] model={args.model} hidden={hidden_size} inter={moe_inter} "
        f"experts={n_experts_global} ae={ae} batches={batches} graph={not args.no_graph}",
        flush=True,
    )

    run_sweep(
        model_id=args.model,
        freq=args.freq,
        batches=batches,
        ae=ae,
        hidden_size=hidden_size,
        moe_inter=moe_inter,
        n_experts_global=n_experts_global,
        window_s=args.window,
        chunk_s=args.chunk,
        ramp_s=args.ramp,
        poll_ms=args.poll_ms,
        use_graph=not args.no_graph,
        out_csv=Path(args.out_csv),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
