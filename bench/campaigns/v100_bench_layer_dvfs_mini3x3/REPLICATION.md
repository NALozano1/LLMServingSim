# LLMServingSim replication — bench layer-boundary DVFS (prefill)

**Campaign:** `v100_bench_layer_dvfs_mini3x3`  
**RNG seed:** `20260616` (same as full 78-run campaign; permutations p01–p03 match)
**Iterations per permutation:** 3 (`i0`, `i1`, `i2`)

## Bench workload (all runs)

| Parameter | Value |
|-----------|-------|
| Engine | sync `vLLM.LLM` (`VLLM_USE_V1=0`) |
| Phase | prefill only (`VLLM_BENCH_PREFILL_ONLY=1`, `max_tokens=1`) |
| Requests | 1 |
| Input tokens | 64 (fixed) |
| Output tokens | 0 (prefill-only) |
| Dataset seed | 42 |
| dtype | float16 |
| tp_size | 1 |
| GPU | 1× V100 (`CUDA_VISIBLE_DEVICES=0`) |

## Models

| Key | Model | Layers | max_num_batched_tokens | max_num_seqs |
|-----|-------|--------|------------------------|--------------|
| `phi` | `microsoft/Phi-tiny-MoE-instruct` | 32 | 256 | 32 |
| `qwen` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 24 | 256 | 8 |

## Scattered DVFS permutations

At each listed layer boundary the host locks GPU clocks to the paired MHz before the worker continues the forward. Pause time is excluded from `exec_sec`.

### `p01` — `phi` (`microsoft/Phi-tiny-MoE-instruct`)

| Layer | MHz after barrier |
|-------|-------------------|
| 7 | 1400 |
| 9 | 700 |
| 10 | 700 |
| 14 | 900 |
| 16 | 1400 |
| 17 | 1400 |
| 20 | 900 |
| 26 | 700 |

**Profiler profiles for simulation:**
- 700 MHz → `profiler/perf/V100_700MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 900 MHz → `profiler/perf/V100_900MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 1400 MHz → `profiler/perf/V100_1400MHz/microsoft/Phi-tiny-MoE-instruct/fp16`

**Slurm run IDs:**
- `runs/scat_p01_phi_i0/` (iteration 0)
- `runs/scat_p01_phi_i1/` (iteration 1)
- `runs/scat_p01_phi_i2/` (iteration 2)

### `p01` — `qwen` (`Qwen/Qwen1.5-MoE-A2.7B-Chat`)

| Layer | MHz after barrier |
|-------|-------------------|
| 1 | 1100 |
| 19 | 900 |
| 20 | 1400 |
| 22 | 1400 |

**Profiler profiles for simulation:**
- 900 MHz → `profiler/perf/V100_900MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`
- 1100 MHz → `profiler/perf/V100_1100MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`
- 1400 MHz → `profiler/perf/V100_1400MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`

**Slurm run IDs:**
- `runs/scat_p01_qwen_i0/` (iteration 0)
- `runs/scat_p01_qwen_i1/` (iteration 1)
- `runs/scat_p01_qwen_i2/` (iteration 2)

### `p02` — `phi` (`microsoft/Phi-tiny-MoE-instruct`)

| Layer | MHz after barrier |
|-------|-------------------|
| 1 | 700 |
| 18 | 1100 |
| 27 | 900 |

**Profiler profiles for simulation:**
- 700 MHz → `profiler/perf/V100_700MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 900 MHz → `profiler/perf/V100_900MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 1100 MHz → `profiler/perf/V100_1100MHz/microsoft/Phi-tiny-MoE-instruct/fp16`

**Slurm run IDs:**
- `runs/scat_p02_phi_i0/` (iteration 0)
- `runs/scat_p02_phi_i1/` (iteration 1)
- `runs/scat_p02_phi_i2/` (iteration 2)

### `p02` — `qwen` (`Qwen/Qwen1.5-MoE-A2.7B-Chat`)

| Layer | MHz after barrier |
|-------|-------------------|
| 2 | 1100 |
| 9 | 900 |

**Profiler profiles for simulation:**
- 900 MHz → `profiler/perf/V100_900MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`
- 1100 MHz → `profiler/perf/V100_1100MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`

**Slurm run IDs:**
- `runs/scat_p02_qwen_i0/` (iteration 0)
- `runs/scat_p02_qwen_i1/` (iteration 1)
- `runs/scat_p02_qwen_i2/` (iteration 2)

### `p03` — `phi` (`microsoft/Phi-tiny-MoE-instruct`)

| Layer | MHz after barrier |
|-------|-------------------|
| 3 | 700 |
| 12 | 900 |
| 14 | 1400 |
| 15 | 1400 |
| 16 | 1100 |
| 17 | 1400 |

**Profiler profiles for simulation:**
- 700 MHz → `profiler/perf/V100_700MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 900 MHz → `profiler/perf/V100_900MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 1100 MHz → `profiler/perf/V100_1100MHz/microsoft/Phi-tiny-MoE-instruct/fp16`
- 1400 MHz → `profiler/perf/V100_1400MHz/microsoft/Phi-tiny-MoE-instruct/fp16`

**Slurm run IDs:**
- `runs/scat_p03_phi_i0/` (iteration 0)
- `runs/scat_p03_phi_i1/` (iteration 1)
- `runs/scat_p03_phi_i2/` (iteration 2)

### `p03` — `qwen` (`Qwen/Qwen1.5-MoE-A2.7B-Chat`)

| Layer | MHz after barrier |
|-------|-------------------|
| 0 | 900 |
| 1 | 900 |
| 9 | 1100 |
| 18 | 1400 |
| 22 | 900 |

**Profiler profiles for simulation:**
- 900 MHz → `profiler/perf/V100_900MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`
- 1100 MHz → `profiler/perf/V100_1100MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`
- 1400 MHz → `profiler/perf/V100_1400MHz/Qwen/Qwen1.5-MoE-A2.7B-Chat/fp16`

**Slurm run IDs:**
- `runs/scat_p03_qwen_i0/` (iteration 0)
- `runs/scat_p03_qwen_i1/` (iteration 1)
- `runs/scat_p03_qwen_i2/` (iteration 2)

## Run matrix (18 jobs)

| run_id | model | perm | iter |
|--------|-------|------|------|
| `scat_p01_phi_i0` | `phi` | `p01` | 0 |
| `scat_p01_phi_i1` | `phi` | `p01` | 1 |
| `scat_p01_phi_i2` | `phi` | `p01` | 2 |
| `scat_p01_qwen_i0` | `qwen` | `p01` | 0 |
| `scat_p01_qwen_i1` | `qwen` | `p01` | 1 |
| `scat_p01_qwen_i2` | `qwen` | `p01` | 2 |
| `scat_p02_phi_i0` | `phi` | `p02` | 0 |
| `scat_p02_phi_i1` | `phi` | `p02` | 1 |
| `scat_p02_phi_i2` | `phi` | `p02` | 2 |
| `scat_p02_qwen_i0` | `qwen` | `p02` | 0 |
| `scat_p02_qwen_i1` | `qwen` | `p02` | 1 |
| `scat_p02_qwen_i2` | `qwen` | `p02` | 2 |
| `scat_p03_phi_i0` | `phi` | `p03` | 0 |
| `scat_p03_phi_i1` | `phi` | `p03` | 1 |
| `scat_p03_phi_i2` | `phi` | `p03` | 2 |
| `scat_p03_qwen_i0` | `qwen` | `p03` | 0 |
| `scat_p03_qwen_i1` | `qwen` | `p03` | 1 |
| `scat_p03_qwen_i2` | `qwen` | `p03` | 2 |

## Measured outputs (per run)

After completion, compare simulation against:

- `runs/<run_id>/results/summary.json` — `wall_sec`, `exec_sec`, `pause_sec`, `energy_j`
- `runs/<run_id>/bench/run_exec_metrics.json` — full timing + throughput
- `runs/<run_id>/bench/dvfs_markers.jsonl` — per-barrier clock readings

Aggregate TSV:

```bash
python3 bench/jobs/collect_bench_layer_campaign_results.py bench/campaigns/v100_bench_layer_dvfs_mini3x3
```

## Simulation notes

1. Use the per-frequency `profiler/perf/V100_<MHz>MHz/<model>/fp16` tables listed above.
2. Replay a **single 64-token prefill** (no decode) through all decoder layers.
3. At each barrier layer, switch the active hardware profile to the target MHz (LLMServingSim does not model DVFS transition latency — compare against `exec_sec`).
4. `pause_sec` / barrier wait is host clock-settle overhead on real hardware only.

