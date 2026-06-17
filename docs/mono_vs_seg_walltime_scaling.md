# Monolithic vs segmented wall-time scaling

**Date:** 2026-06-16  
**Branch:** `feat/dvfs-scale`  
**Config:** `single_node_single_instance.json`, bf16, `example_trace.jsonl`, `--no-enable-prefix-caching`  
**Command:** `python3 scripts/run_mono_vs_seg_timing.py --num-reqs 1 3 10`  
**Artifacts:** `outputs/branch_compare/timing_study/summary.csv`

## Results

| N req | Mono wall (s) | Seg wall (s) | Ratio (seg/mono) | Mono s/req | Seg s/req |
|------:|--------------:|-------------:|-----------------:|-----------:|----------:|
| 1 | 7.1 | 157.1 | **22.2×** | 7.09 | 157.14 |
| 3 | 14.0 | 306.5 | **21.9×** | 4.67 | 102.15 |
| 10 | 15.3 | 346.7 | **22.7×** | 1.53 | 34.67 |

Sim clocks track as expected (seg ≈ mono +0.001% at each N).

## Interpretation

### The ~22× ratio is **not** a one-time fixed cost

If segmentation were a fixed startup tax, seg time would look like `T_fixed + N × T_seg_per_req` and the **ratio would shrink** as N grows. That is **not** what we see: **ratio stays ~22×** for N = 1, 3, and 10.

The penalty is tied to **work done** (forward passes × segments), not a single lump sum at job start.

### Segmented wall time **does not** scale linearly with N

| N | Seg wall | vs N=1 |
|---|----------|--------|
| 1 | 157 s | 1.0× |
| 3 | 307 s | 2.0× |
| 10 | 347 s | **2.2×** |

Ten requests only ~2.2× the wall time of one request in segmented mode (not 10×). Per-request seg cost drops from ~157 s/req (N=1) to ~35 s/req (N=10).

**Cause:** shape-based workload reuse — segments name traces/graphs  
`instance{id}_t{total_len}p{num_prefill}_s{stage}`  
so forwards with the same token shape reuse Chakra ET files instead of regenerating every time.

### Monolithic also amortizes

Mono goes 7.1 s → 15.3 s for 1→10 reqs. Continuous batching combines requests; you do not pay 10× full trace builds.

## Practical takeaway

| Question | Answer |
|----------|--------|
| Is 22× a fixed overhead? | **No** — ratio vs mono stays ~22× because both modes scale with work. |
| Does seg get cheaper per req on long runs? | **Yes** — N=10 seg wall is only ~2.2× N=1 (not 10×). |
| Expect 22× on long jobs? | **~22× vs mono at the same N**; absolute seg time grows sub-linearly in N due to reuse. |

## Re-run

```bash
docker exec servingsim_docker bash -c \
  'cd /app/LLMServingSim && python3 scripts/run_mono_vs_seg_timing.py --num-reqs 1 3 10'
```
