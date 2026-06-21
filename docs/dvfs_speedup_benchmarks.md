# DVFS speedup benchmarks (P0)

**Branch:** `feat/dvfs-speedup`  
**Dataset:** `workloads/example_trace.jsonl`, N=10, Llama-8B bf16  
**Harness:** `scripts/run_dvfs_speedup_benchmark.sh`

## Results (example_trace, N=10)

| Tier | Commit | Seg wall (s) | vs baseline | Mono wall (s) | Seg/mono |
|------|--------|-------------:|------------:|--------------:|---------:|
| Baseline (pre-P0) | `8243f59` | **346.9** | 1.0× | — | — |
| P0.1 Chakra skip (mtime) | `8e22147` | **344.8** | 1.01× | — | — |
| P0.2 + trace skip | `5c6e282` | **47.1** | **7.4×** | 14.5 | **3.3×** |

Prior study on same workload (pre-speedup branch): mono ~15.3 s, seg ~347 s (~23×).

### ShareGPT N=10 (measured on P0.2)

| Mode | Wall (s) | Sim clocks (ns) | vs original seg |
|------|---------:|----------------:|------------------:|
| Mono | 77.2 | 10,269,946,516 | — |
| Seg (original, pre-P0) | 1774 | 10,270,053,172 | 1.0× |
| Seg (P0.2) | **17.8** | 10,144,740,432 | **99.7×** |

Per-request latencies match mono/original-seg within ~1–2% (sim correctness OK). Seg host wall is now **below mono** because decode-heavy runs reuse cached traces/graphs while mono still pays full trace+Chakra per forward.


## Findings

1. **P0.2 (trace skip) is the big win** — skipping `_synthesize_trace_stage` + bookends stops trace rewrites, which unlocks shape reuse within a run.
2. **P0.1 alone (mtime ET vs trace) barely helped** because `generate_trace` always rewrote the trace file, bumping mtime past `llm.0.et` on every segment hit. Chakra skip only becomes effective once trace synthesis is skipped.
3. **P0.1 fix:** segment Chakra skip now keys off the `.meta` sidecar (same as trace cache), not trace mtime.
4. **Sim clocks:** P0.2 seg `1663438891` ns vs baseline `1665096395` ns (~0.1% lower). Worth monitoring; may be benign ordering noise.

---

## Layer-hardware-alternate (RTXPRO6000 ↔ A6000)

`--forward-segments per_block --layer-hardware-alternate` on `feat/dvfs-speedup` (P0.2).

Harness: `scripts/run_layer_hw_alternate_timing.py`

### example_trace

| N | Seg+alt wall | Sim clocks (ns) | Layer switches | vs pre-P0 alt wall (~153s @ N=1) |
|--:|-------------:|----------------:|---------------:|----------------------------------:|
| 1 | **9.4 s** | 926,109,202 | 2,310 | **16× faster** |
| 10 | **56.7 s** | 1,791,130,717 | 4,587 | — |

**Sim parity (N=1):** latency 879,182,394 ns vs pre-P0 alternate 880,808,800 ns (**−0.18%**). TTFT/TPOT within ~3%.

Cache reuse works across both hardware profiles — traces are keyed by `{hardware}/{model}/{shape}_s{stage}`, so RTX and A6000 each maintain separate segment caches (~4.7k–8.7k reuse log hits per run).


## Reproduce

```bash
# example_trace N=10 (≈13 min total)
bash scripts/run_dvfs_speedup_benchmark.sh

# ShareGPT N=10 (longer)
NUM_REQS=10 DATASET=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  bash scripts/run_dvfs_speedup_benchmark.sh
```

Outputs: `outputs/branch_compare/dvfs_speedup/{baseline,p0_1_chakra,p0_2_full}/`
