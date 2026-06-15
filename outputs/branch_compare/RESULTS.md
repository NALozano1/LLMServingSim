# Branch & layer-hardware comparison results

**Date:** 2026-06-15  
**Branch:** `feat/dvfs-scale` @ `91e19f0` (pushed to `origin`)  
**Config:** `configs/cluster/single_node_single_instance.json`, `--dtype bfloat16`, `workloads/example_trace.jsonl`, `--num-req 1`, `--no-enable-prefix-caching`  
**Docker:** `servingsim_docker`, cwd `/app/LLMServingSim`

---

## 1. Monolithic parity: `main` vs `feat/dvfs-scale`

Default path (`--forward-segments off`). CSVs are **byte-for-byte identical**.

| Run | Branch | Sim clocks (ns) | Wall time | Latency (ns) | TTFT (ns) | TPOT (ns) |
|-----|--------|-----------------|-----------|--------------|-----------|-----------|
| mono 1-req | `main` | 827,547,677 | 6.4s | 780,620,869 | 11,008,270 | 11,153,805 |
| mono 1-req | `feat/dvfs-scale` | 827,547,677 | 6.3s | 780,620,869 | 11,008,270 | 11,153,805 |
| mono 10-req | `main` | 1,665,077,255 | 11.3s | (per-req) | — | — |
| mono 10-req | `feat/dvfs-scale` | 1,665,077,255 | 11.6s | (per-req) | — | — |

**Artifacts:** `main_mono_1req.csv`, `feat_dvfs-scale_mono_1req.csv`, `main_mono_10req.csv`, `feat_dvfs-scale_mono_10req.csv`

**Conclusion:** Monolithic simulation is unchanged on the feature branch.

---

## 2. Segmented forward (`--forward-segments per_block`)

34 ASTRA submissions per forward (prologue + 32 blocks + head). Bookend stubs add ~0.3% sim-time overhead vs monolithic.

| Run | Sim clocks (ns) | Δ vs mono | Wall time | Latency (ns) | TTFT (ns) | TPOT (ns) |
|-----|-----------------|-----------|-----------|--------------|-----------|-----------|
| mono 1-req | 827,547,677 | — | ~6.3s | 780,620,869 | 11,008,270 | 11,153,805 |
| seg 1-req (no alt) | 830,169,131 | **+0.31%** | ~2m 22s | 783,242,323 | 11,339,062 | 11,187,003 |

**Artifacts:** `feat_dvfs-scale_seg_1req.csv`

Segmentation increases wall time ~22× (trace/graph regen per segment) but barely moves simulated latency.

---

## 3. Layer-hardware-alternate: same device vs real swap

### Same device (no alternation)

Hide secondary profiler aliases so `resolve_hardware_pair` returns `(RTXPRO6000, RTXPRO6000)`. Simulator logs:

```
Instance 0: only one hardware profile for meta-llama/Llama-3.1-8B; layer alternation disabled
```

**Command:**
```bash
# profiler/perf/A6000 and v0 seed aliases temporarily moved aside
python -m serving ... --forward-segments per_block --layer-hardware-alternate \
  --work-events outputs/branch_compare/feat_seg_samehw_events.jsonl \
  --output outputs/branch_compare/feat_seg_samehw_1req.csv
```

| Metric | seg (no alt) | seg + `--layer-hardware-alternate` (same device) | Δ |
|--------|--------------|---------------------------------------------------|---|
| Sim clocks (ns) | 830,169,131 | 830,169,131 | **0%** |
| Latency (ns) | 783,242,323 | 783,242,323 | 0% |
| TTFT (ns) | 11,339,062 | 11,339,062 | 0% |
| TPOT (ns) | 11,187,003 | 11,187,003 | 0% |
| Wall time | 2m 22s | 2m 39s | ~12% (I/O noise) |
| `dvfs_switch` events | 0 | **0** | — |
| `segment_complete` events | — | 2,310 | — |

**Artifacts:** `feat_seg_samehw_1req.csv`, `feat_seg_samehw_events.jsonl`, `feat_seg_samehw_run.log`

**Conclusion:** Enabling `--layer-hardware-alternate` with a single hardware profile has **no simulated impact** — the toggle path is correctly gated off when `pair[0] == pair[1]`.

---

### Real alternation (RTXPRO6000 ↔ A6000)

A6000 profiler data is **auto-seeded** from RTXPRO6000 at **1.25× layer times** (demo alias, not measured silicon).

**Command:**
```bash
python -m serving ... --forward-segments per_block --layer-hardware-alternate \
  --work-events outputs/branch_compare/feat_seg_alternate_events.jsonl \
  --output outputs/branch_compare/feat_seg_alternate_1req.csv
```

| Metric | seg (no alt) | seg + real alt | Δ |
|--------|--------------|----------------|---|
| Sim clocks (ns) | 830,169,131 | 927,735,608 | **+11.8%** |
| Latency (ns) | 783,242,323 | 880,808,800 | +12.5% |
| TTFT (ns) | 11,339,062 | 12,805,813 | +12.9% |
| TPOT (ns) | 11,187,003 | 12,579,753 | +12.4% |
| Wall time | 2m 22s | 2m 33s | — |
| `dvfs_switch` events | 0 | **2,310** | — |
| `segment_complete` events | — | 2,310 | — |

**Hardware switch pattern (2,310 transitions, 0 invalid):**
- `RTXPRO6000 → A6000`: 1,155
- `A6000 → RTXPRO6000`: 1,155

Strict alternation every non-final segment boundary.

**Artifacts:** `feat_seg_alternate_1req.csv`, `feat_seg_alternate_events.jsonl`, `feat_seg_alternate_run.log`

**Conclusion:** Observed ~12% slowdown matches the seeded 1.25× A6000 scale averaged across alternating layers (roughly half the layers run on each profile → ~1.125× geometric mean, plus bookend/stub effects).

---

## 4. Summary table (1-req smoke)

| Scenario | Sim clocks (ns) | vs mono | `dvfs_switch` |
|----------|-----------------|---------|-----------------|
| `main` mono | 827,547,677 | — | — |
| `feat` mono | 827,547,677 | 0% | — |
| `feat` seg | 830,169,131 | +0.31% | 0 |
| `feat` seg + alt flag, same device | 830,169,131 | +0.31% | 0 |
| `feat` seg + RTXPRO6000↔A6000 | 927,735,608 | +12.1% | 2,310 |

---

## 5. Reproduce

```bash
# Monolithic (either branch)
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json --dtype bfloat16 \
  --dataset workloads/example_trace.jsonl --output outputs/branch_compare/mono.csv \
  --num-req 1 --no-enable-prefix-caching --forward-segments off --log-level WARNING'

# Segmented, no swap
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json --dtype bfloat16 \
  --dataset workloads/example_trace.jsonl --output outputs/branch_compare/seg.csv \
  --num-req 1 --no-enable-prefix-caching --forward-segments per_block --log-level WARNING'

# Segmented + real hardware alternation
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json --dtype bfloat16 \
  --dataset workloads/example_trace.jsonl --output outputs/branch_compare/seg_alt.csv \
  --num-req 1 --no-enable-prefix-caching --forward-segments per_block \
  --layer-hardware-alternate --work-events outputs/branch_compare/events.jsonl --log-level WARNING'

# Segmented + alt flag but same device: hide profiler/perf/A6000 and v0 seed aliases first
```

---

## 6. Notes

- **A6000 is synthetic:** seeded copy of RTXPRO6000 profiler CSVs with 1.25× `time_us`. Real measured A6000 data would change the alternation delta.
- **Wall time >> sim time for segmented runs** because each of 34 segments regenerates traces/graphs; this is a known dev-mode cost, not simulated serving latency.
- Legacy run `outputs/segment_layer_hw.csv` / `outputs/layer_hw_events.jsonl` (Jun 15 earlier) matches the new `feat_seg_alternate_*` pattern (9,382 event lines, 2,310 switches).
