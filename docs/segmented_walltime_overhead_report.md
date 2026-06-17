# Segmented vs monolithic wall-time overhead (~23×)

**Date:** 2026-06-16  
**Branch:** `feat/dvfs-scale`  
**Workload:** `sharegpt-llama-3.1-8b-300-sps10.jsonl`, Llama-8B, `single_node_single_instance.json`, bf16  

Related: [`mono_vs_seg_walltime_scaling.md`](mono_vs_seg_walltime_scaling.md), [`outputs/branch_compare/timing_study_sharegpt/`](../outputs/branch_compare/timing_study_sharegpt/)

---

## Executive summary

The **~23× wall-time penalty** is real and stable across request counts. It is **not** because ASTRA simulates 23× more GPU work — **sim clocks are essentially identical** between mono and seg at each N.

The penalty is almost entirely **Python orchestration tax**:

1. **~34× more ASTRA submissions per forward pass** (prologue + 32 blocks + head for Llama-8B).
2. **Every submission** runs `generate_trace` → `generate_graph` (Chakra subprocess) → ASTRA IPC round-trip, even when the workload file already exists.
3. **No effective skip/cache** on trace synthesis or Chakra conversion today (despite design notes suggesting otherwise).

Segmentation buys **layer-boundary hooks** (hardware/DVFS swap). You pay for that in **host time**, not modeled GPU time.

---

## Measured data

### ShareGPT (real request counts)

| N | Mono wall | Seg wall | Ratio | Mono s/req | Seg s/req | Sim clocks (mono ≈ seg) |
|--:|----------:|---------:|------:|-----------:|----------:|-------------------------|
| 10 | 77.7 s | 1774 s | **22.8×** | 7.8 | 177 | 10.27B ns |
| 20 | 78.3 s | 1875 s | **23.9×** | 3.9 | 94 | 13.13B ns |
| 100 | 111.3 s | 2551 s | **22.9×** | 1.1 | 26 | 26.30B ns |

### example_trace (N≤10 only; dataset has 10 lines)

| N | Mono | Seg | Ratio |
|--:|-----:|----:|------:|
| 1 | 7.1 s | 157 s | 22.2× |
| 10 | 15.3 s | 347 s | 22.7× |

N=20/100 on `example_trace` **do not add requests** (capped at 10).

### Scaling shape

| Observation | Implication |
|-------------|-------------|
| Ratio ~23× stable at N=10,20,100 | Overhead scales **with simulated work**, not a one-time startup fee |
| Seg wall sub-linear in N (1774→2551 for 10→100 reqs) | **Trace/workload reuse** by shape key helps absolute time, not the mono ratio |
| Mono wall also sub-linear | Continuous batching amortizes forwards |

---

## Where the time goes (code path)

### Monolithic (`--forward-segments off`)

Per scheduler step that launches a new batch workload:

```
generate_trace(batch)          # full forward: prologue + 32 blocks + head in one file
generate_graph(batch)          # one Chakra subprocess
write_flush(astra, workload)   # one ASTRA iteration
read_wait()                    # block until "Waiting"
```

Mono trace synthesis can **copy repeated transformer blocks** inside one file (`can_copy` in `_synthesize_trace`), so one Chakra conversion covers all layers.

### Segmented (`--forward-segments per_block`)

Per **segment** (34 per forward for Llama-8B):

```
generate_trace(batch, stage_idx=k)     # _synthesize_trace_stage: ONE stage only
  → write trace.txt
  → read back, _apply_segment_bookends (stub embed/head on non-final segments)
  → rewrite trace.txt
generate_graph(batch, stage_idx=k)   # Chakra subprocess — ALWAYS runs
write_flush(astra, workload)
read_wait()
on_segment_done() / hardware toggle
schedule same batch with layer_cursor++
```

Relevant files:

| File | Role |
|------|------|
| `serving/__main__.py` ~832–852 | Per-segment trace + graph + ASTRA submit |
| `serving/core/trace_generator.py` | `_synthesize_trace_stage`, bookends |
| `serving/core/graph_generator.py` | `subprocess.run(chakra converter)` — **no skip-if-exists** |
| `serving/core/forward_segments.py` | Slug `instance{id}_t{total_len}p{num_prefill}_s{stage}` |
| `serving/core/controller.py` | `read_wait()` blocks on every ASTRA iteration |

### What reuse actually does today

Workload paths reuse across batches with the **same token shape** (`total_len`, `num_prefill`) and `stage_idx`:

```
inputs/workload/RTXPRO6000/meta-llama/Llama-3.1-8B/instance0_t{batch_shape}_s{stage}/llm.0.et
```

That explains why seg **absolute** wall time grows slowly with N (1774 s → 2551 s for 10→100 reqs).

**But:** `generate_trace` and `generate_graph` still run on every segment submission. There is **no** `if os.path.isfile(llm.0.et): return` in `graph_generator.py`. `WORK_LOG.md` mentions “skip regen if `llm.0.et` exists” — **not implemented** in current code.

So reuse helps only indirectly (overwriting same paths, OS cache, possibly faster Chakra on tiny traces) — not by skipping work.

---

## Why ~23× and not ~34×?

Naïve expectation: 34 segments per forward → 34× wall. Observed ~23× because:

1. **Not every scheduler step runs all 34 segments** — prefill/decode/chunking mix varies; some iterations are `pass`/sync.
2. **Shape reuse** — decode steps with identical `(total_len, num_prefill)` hit the same 34 ET paths; repeated Chakra + trace work on identical inputs is faster than cold builds.
3. **Mono is not free** — monolithic still does trace + Chakra per forward; ratio is `(34 × seg_per_segment) / (1 × mono_per_forward)`, not `34 / 1` in wall seconds.
4. **Segment traces are smaller** — one block per file vs full model; Chakra converts less text per call (but subprocess overhead remains).

---

## Cost breakdown (estimated)

For ShareGPT N=10 seg (~1774 s):

| Component | Per segment (order of magnitude) | × invocations | Notes |
|-----------|----------------------------------|---------------|-------|
| `_synthesize_trace_stage` + bookends | 10–50 ms | thousands | Python + pandas profile lookups |
| Chakra `subprocess.run` | 0.3–2 s | thousands | **Likely dominant**; spawns Python, parses trace, writes ET |
| ASTRA analytical iter | 1–50 ms | thousands | Fast; not the bottleneck |
| `read_wait` + Python loop | small | thousands | Adds up |

**Rule of thumb:** if Chakra averages ~0.5–1 s per segment and you run ~2000–3500 segment submissions for 10 ShareGPT reqs, wall ≈ 1000–3500 s — matches observation.

---

## Ideas to cut wall time (prioritized)

### P0 — High impact, low risk (implement first)

#### 1. Skip Chakra when ET is fresh

In `graph_generator.py`, before `subprocess.run`:

```python
et_path = f"{workload_dir}/llm.0.et"
trace_path = f"../../../inputs/trace/{file_name}.txt"
if os.path.isfile(et_path) and os.path.getmtime(et_path) >= os.path.getmtime(trace_path):
    return  # reuse cached graph
```

`WORK_LOG` already describes this; it was never wired up. **Expected gain:** large on repeated shapes (decode-heavy runs) — potentially **5–15×** seg speedup when reuse is high.

#### 2. Skip trace synthesis when segment file unchanged

Skip `_synthesize_trace_stage` if `hardware`, `dvfs_scale`, and batch shape match a memoized key and trace file exists. Only re-run when hardware alternation changes `hardware` or shape is new.

**Expected gain:** similar to P0 on decode; essential for `--layer-hardware-alternate`.

#### 3. `--forward-segments per_block` only when needed

Default off (already). Document: use mono for throughput/energy sweeps; seg only for layer-wise DVFS experiments.

---

### P1 — Medium effort, strong payoff

#### 4. Pre-build segment library offline

One-shot script: for each `(hardware, model, stage_idx, representative shapes)` generate traces + Chakra ETs. Sim run becomes lookup + ASTRA only.

Fits V100 MHz matrix + layer alternation (small shape grid during decode).

#### 5. Persistent Chakra worker

Replace per-call `subprocess.run` with a long-lived converter process (stdin/stdout protocol). Amortize Python startup/import cost across thousands of calls.

#### 6. In-process trace bookends

Avoid write → read → bookend → rewrite cycle in `generate_trace`. Apply bookends during initial synthesis.

#### 7. Segment-aware power path

`power_model.reset_log()` every `generate_trace` call adds minor overhead; batch power flush per forward instead of per segment.

---

### P2 — Architectural / larger changes

#### 8. ASTRA multi-workload queue

Submit all 34 segment ETs in one ASTRA session without 34 Python round-trips — **only if** layer-boundary hooks move into ASTRA or a shim (hard; defeats current design).

#### 9. Analytical fast path for cached segments

If ET exists and hardware unchanged, advance sim clocks from profile DB without launching ASTRA for that segment. Use ASTRA only for comm/contention validation.

**Risk:** drifts from cycle-level fidelity.

#### 10. Coarser segments

`per_block` uses 34 stages. Option: `--forward-segments per_layer_group` (e.g. 4 blocks) reduces hooks but still allows some DVFS control — **34 → ~9** submissions.

#### 11. Parallel Chakra across stages

Thread pool pre-convert all 34 stages for a shape while ASTRA runs — overlaps CPU with sim.

---

### P3 — Measurement / ops

#### 12. Add `--profile-orchestration` mode

Log timestamps around `generate_trace`, `generate_graph`, `write_flush`, `read_wait` to JSONL. One run quantifies actual breakdown instead of estimates.

#### 13. Warm ET cache in CI / before sweeps

Run one throwaway req to populate `inputs/workload/` before timing study.

---

## Recommended roadmap

| Phase | Work | Target |
|-------|------|--------|
| **1** | Implement ET skip + trace skip (P0 #1–2) | Seg within **2–5×** of mono on repeated-shape workloads |
| **2** | `--profile-orchestration` + one ShareGPT run | Confirm Chakra vs trace split |
| **3** | Offline segment library for V100 Qwen DVFS | Layer-alt experiments without 30 min waits |
| **4** | Persistent Chakra or coarser segments if still too slow | Interactive iteration |

---

## When to use which mode

| Goal | Mode |
|------|------|
| Latency / energy / throughput numbers | **Monolithic** |
| Layer-wise hardware / DVFS swap | **Segmented** (+ accept wall tax or implement P0 cache) |
| V100 MHz comparison (no switching) | **Monolithic** per hardware tag (`run_v100_qwen_mhz_matrix.py`) |
| Layer-alt V100 DVFS | **Segmented** after P0 cache lands |

---

## Quick reference commands

```bash
# Monolithic (fast)
python3 -m serving --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype bfloat16 --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --num-reqs 100 --output outputs/mono.csv

# Segmented (slow; layer hooks)
python3 -m serving ... --forward-segments per_block --layer-hardware-alternate ...

# Timing study
python3 scripts/run_mono_vs_seg_timing.py \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl --num-reqs 10 20 100
```
