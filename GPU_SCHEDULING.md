# GPU scheduling guide (profile → simulate)

What to run on **real GPUs** (profiler) vs what the **simulator** models, and how
TP / PP / EP / DP map to devices. Written for **nserver15** / RTXPRO6000 work.

---

## Two environments

| Phase | Container | Needs real GPU? | Purpose |
|-------|-----------|-----------------|---------|
| **Profile** | `./scripts/docker-vllm.sh` | **Yes** | Measure per-layer latency → `profiler/perf/` |
| **Simulate** | `./scripts/docker-sim.sh` | No (CPU-only sim) | Cycle-level serving sim using profile CSVs |

You only touch real GPUs when **profiling**. Simulation reads pre-recorded CSVs.

---

## Golden rules

1. **`num_npus = tp_size × pp_size`** per instance (always).
2. **Simulator `--dtype` must match the profile folder** (e.g. `bfloat16` → `bf16/`).
3. **Each `tp_size` you simulate needs a `tp<N>/` profile folder** for layers that shard with TP (attention, GEMMs). `tp_stable` layers (layernorm, sampler) can reuse `tp1/`.
4. **TP must be a power of 2** and divide `num_attention_heads` (Llama-3.1-8B: 32 heads → TP ∈ {1, 2, 4, 8, 16, 32}).
5. **PP** splits layers across GPUs in one instance; **DP** is multiple instances with the same `dp_group`; **EP** is MoE expert sharding (shares GPUs with TP).

---

## Bundled profiles on this machine (RTXPRO6000)

All current bundles only ship **`tp1/` and `tp2/`** under `bf16/`:

| Model | Path | TP folders |
|-------|------|------------|
| Llama-3.1-8B | `profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/` | tp1, tp2 |
| Qwen3-32B | `profiler/perf/RTXPRO6000/Qwen/Qwen3-32B/bf16/` | tp1, tp2 |
| Qwen3-30B-A3B (MoE) | `profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507/bf16/` | tp1, tp2 |

**Ready to simulate today (Llama-3.1-8B):** `tp_size ∈ {1, 2}`, `--dtype bfloat16`.

For `tp_size ∈ {4, 8, 16, 32}` you must **re-profile** first (see below).

---

## What each GPU does (by parallelism mode)

### A. Independent replicas — `tp=1, pp=1` (simplest)

**Config pattern:** N instances, each `num_npus: 1`, `tp_size: 1`.

```
GPU 0 → Instance 0 (full model)
GPU 1 → Instance 1 (full model)
...
```

- **Profile needed:** `tp1/` only.
- **Generate:**
  ```bash
  python3 scripts/generate_cluster_config.py \
    --num-nodes 2 --instances-per-node 4 \
    --output configs/cluster/generated_8gpu.json
  ```
- **Simulate:** `--request-routing-policy LOAD` spreads requests across instances.
- **Status:** ✅ 8-GPU run validated (~100s for 10 requests).

---

### B. Tensor parallel — `tp>1, pp=1` (one instance, multiple GPUs)

**Config pattern:** 1 instance, `num_npus: tp_size`, `tp_size: N`.

```
Instance 0:
  GPU 0..N-1 → same batch, heads sharded, ALLREDUCE after o_proj / down_proj
```

| tp_size | GPUs / instance | Profile folder | Llama-3.1-8B valid? |
|---------|-----------------|----------------|---------------------|
| 2 | 2 | `tp2/` | ✅ bundled |
| 4 | 4 | `tp4/` | ❌ profile first |
| 8 | 8 | `tp8/` | ❌ profile first |

- **Generate:**
  ```bash
  python3 scripts/generate_cluster_config.py \
    --num-nodes 1 --instances-per-node 1 --tp-size 2 \
    --output configs/cluster/generated_tp2.json
  ```

---

### C. Pipeline parallel — `pp>1` (layers split across GPUs)

**Config pattern:** 1 instance, `num_npus: tp×pp`, `pp_size: P`.

```
Instance 0 (example tp=2, pp=2 → 4 GPUs):
  GPU 0,1 → Pipeline stage 0 (early layers)   [TP group within stage]
  GPU 2,3 → Pipeline stage 1 (later layers)   [TP group within stage]
  Activations passed stage-to-stage (P2P in trace)
```

| Config | GPUs | Profile | Notes |
|--------|------|---------|-------|
| tp=1, pp=2 | 2 | `tp1/` | 2 stages, 1 GPU each |
| tp=2, pp=2 | 4 | `tp2/` | 2 stages × 2-way TP |

- **Profile:** still driven by **`tp_size`** (not pp). PP changes trace layout, not which `tp<N>/` CSV set you load.
- **Generate:**
  ```bash
  python3 scripts/generate_cluster_config.py \
    --num-nodes 1 --instances-per-node 1 --tp-size 2 --pp-size 2 \
    --output configs/cluster/generated_tp2_pp2.json
  ```
- **Simulate:** expect **long silent startup** (many minutes) while PP traces build; see runtime estimator output.
- **Status:** 🔄 in progress on nserver15.

---

### D. MoE expert parallel — `ep>1` (dense models: N/A)

**Config pattern (single instance):** `tp_size: 2`, `ep_size: 2`, `num_npus: 2`.

```
Instance 0:
  GPU 0,1 → TP + EP: experts sharded, ALLTOALL around MoE block
```

- **Model:** MoE only (e.g. Qwen3-30B-A3B).
- **Profile:** `tp_size` folder + MoE categories in bundle.
- **Generate:**
  ```bash
  python3 scripts/generate_cluster_config.py \
    --num-nodes 1 --instances-per-node 1 \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tp-size 2 --ep-size 2 \
    --output configs/cluster/generated_moe_ep.json
  ```

---

### E. MoE DP + EP — multiple instances, one `dp_group`

**Config pattern:** 2+ instances, same `dp_group`, `ep_size` = total EP across group.

```
Node 0: Instance 0 (tp=2, ep=4) → GPUs 0,1
Node 1: Instance 1 (tp=2, ep=4) → GPUs 2,3
        dp_group "A" → expert shards span instances, wave-synced
```

- **Rules:** all instances in group share `tp_size` and `ep_size`; `ep_size % num_instances == 0`.
- **Generate:**
  ```bash
  python3 scripts/generate_cluster_config.py \
    --num-nodes 2 --instances-per-node 1 \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tp-size 2 --ep-size 4 --dp-group A \
    --link-bw 128 16 --link-latency 500 20000 \
    --output configs/cluster/generated_moe_dp_ep.json
  ```

---

## Scaling to many GPUs (e.g. 1024)

| Goal | Layout | Profile need |
|------|--------|--------------|
| 1024 independent replicas | 128 nodes × 8 inst × tp1 | `tp1/` only |
| 16 models × tp64 | 16 inst × 64 GPUs, tp=64 | `tp64/` per model (re-profile) |
| 32 models × tp32 | 32 inst × 32 GPUs | `tp32/` (re-profile) |

```bash
python3 scripts/generate_cluster_config.py \
  --total-gpus 1024 --gpus-per-node 8 --tp-size 1 \
  --indent 0 --output configs/cluster/generated_1024gpu.json
```

---

## When to profile (real GPU schedule)

Run **inside vLLM container** on a machine with the target GPU:

```bash
./scripts/docker-vllm.sh
cd /workspace
```

Edit `profiler/profile.sh`:

```bash
MODEL="meta-llama/Llama-3.1-8B"
HARDWARE="RTXPRO6000"
TP_DEGREES="1,2,4,8"    # must include 1; add every tp_size you plan to simulate
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=256
```

Then:

```bash
./profiler/profile.sh
```

**What happens on the physical GPU:** profiler runs **one TP degree at a time** (`tensor_parallel_size=N` in vLLM). For `TP_DEGREES="1,2,4"` you get three sequential profiling passes on the same GPU(s), writing:

```
profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp1/
profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp2/
profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16/tp4/
```

**Schedule checklist before simulating a new TP:**

| Step | Action |
|------|--------|
| 1 | Confirm `tp_size` divides attention heads |
| 2 | Profile with `TP_DEGREES` including that `tp_size` |
| 3 | Confirm `profiler/perf/.../tp<N>/dense.csv` exists |
| 4 | Generate cluster JSON with matching `--tp-size` |
| 5 | Simulate with matching `--dtype bfloat16` |

---

## When to simulate (no GPU needed)

```bash
docker start -ai servingsim_docker
cd /app/LLMServingSim

python -m serving \
  --cluster-config 'configs/cluster/<your>.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/<run>.csv' \
  --log-interval 1.0
```

Add `--request-routing-policy LOAD` for multi-instance replica farms.

**Runtime auto-logged** to `calibration/runtimes/runs.jsonl` after each successful run.

---

## Quick reference: config → GPUs → profile

| You want | Instances | GPUs/inst | tp | pp | ep | dp_group | Profile dirs |
|----------|-----------|-----------|----|----|-----|----------|--------------|
| 1 replica | 1 | 1 | 1 | 1 | 1 | — | tp1 |
| 8 replicas | 8 | 1 | 1 | 1 | 1 | — | tp1 |
| TP-2 single instance | 1 | 2 | 2 | 1 | 1 | — | tp2 |
| TP-2 × PP-2 | 1 | 4 | 2 | 2 | 1 | — | tp2 |
| MoE EP | 1 | 2 | 2 | 1 | 2 | — | tp2 (MoE model) |
| MoE DP+EP (2 nodes) | 2 | 2 | 2 | 1 | 4 | A | tp2 (MoE model) |

---

## Related files

| File | What |
|------|------|
| `scripts/generate_cluster_config.py` | Build homogeneous cluster JSON + runtime estimate |
| `configs/cluster/README.md` | Upstream schema reference |
| `profiler/profile.sh` | Real-GPU profiling entry point |
| `WORK_LOG.md` | What's been run / validated |
| `calibration/runtimes/runs.jsonl` | Measured wall-clock times |
