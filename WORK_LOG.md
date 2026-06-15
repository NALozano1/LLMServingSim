# LLMServingSim work log

Running notes for setup and experiments on **nserver15** (`~/DVFS_MoE/LLMServingSim`).
Update this file as we try new configs/runs.

---

## Environment

| Item | Status |
|------|--------|
| Host | `nserver15`, Ubuntu 22.04 |
| Docker Engine | Installed system-wide (Jun 10, 2026) |
| Docker group | `alexluzano` in `docker` group (use `newgrp docker` or fresh shell if permission denied) |
| Simulator container | `servingsim_docker` (`astrasim/tutorial-micro2024`) |
| ASTRA-Sim build | Done inside container (`./scripts/compile.sh`) |
| Git submodules | Initialized with `git submodule update --init --recursive` |

**Attach to simulator:**
```bash
docker start -ai servingsim_docker
cd /app/LLMServingSim
```

---

## What has worked

### Setup & build
- [x] Install Docker on host (`docker-ce`, system service enabled)
- [x] Launch simulator via `./scripts/docker-sim.sh` (or `sudo docker` before group membership)
- [x] Initialize submodules (Chakra + ASTRA-Sim deps were empty without this)
- [x] `./scripts/compile.sh` **inside container** — ASTRA-Sim analytical backend built successfully

### Simulations (end-to-end)
- [x] **Single node, single instance** — `configs/cluster/single_node_single_instance.json`, `--dtype bfloat16`
  - Output: `outputs/example_single_run.csv`
  - 10 requests, ~13s wall time
- [x] **Generated 8-GPU replica cluster** — `configs/cluster/generated_8gpu.json`, `--dtype bfloat16`, `LOAD` routing
  - Command validated Jun 10, 2026
  - 8 instances across 2 nodes, 10 requests completed cleanly
  - Output: `outputs/generated_8gpu_run.csv`
  - **Wall time:** ~1m 41s (`Total simulation time` in sim output)
  - Per-instance TTFT/TPOT metrics printed for instances 0–7

### Config generator
- [x] **Created `scripts/generate_cluster_config.py`** — upstream only ships small hand-written JSONs in `configs/cluster/`; we added a generator so we can scale out and try **newer parallelism dimensions/combinations** without editing hundreds of instance blocks by hand.
  - **Supported knobs:** `tp_size`, `pp_size`, `ep_size`, `dp_group`, plus layout via `--num-nodes` / `--instances-per-node` or `--total-gpus` / `--gpus-per-node`
  - **Validates** against `config_builder` rules (head-count TP, MoE expert divisibility, DP+EP group sizing, profile warnings)
  - **Homogeneous clusters only** — same TP/PP/EP/DP on every instance; mixed per-instance layouts still need manual JSON
- [x] Generator output passes `config_builder.build_cluster_config()` for:
  - `generated_8gpu.json` (8 × TP=1 replicas)
  - `generated_tp2_pp2.json` (1 × TP=2, PP=2)
  - `generated_moe_dp_ep.json` (Qwen MoE, TP=2 + EP=4 + DP group across 2 nodes)
- [x] **Runtime estimator** in generator (`estimate_runtime()` / `_CALIBRATION` in `scripts/generate_cluster_config.py`) — printed after each generate; recalibrate constants as we log more runs below

---

## What has not worked (or not yet verified)

### Setup / build
- [ ] `./scripts/docker-sim.sh` on host **before Docker install** → `docker: command not found`
- [ ] `docker` without group/sudo → `permission denied` on `/var/run/docker.sock`
- [ ] `./scripts/compile.sh` on **host** with mismatched protobuf:
  - `pip` protoc 4.25.x vs `/usr/local` libprotobuf 5.27 → `PROTOBUF_TSAN_READ` / `_tsan_detect_race` compile errors
  - Regenerating protos on host then linking → undefined references to `absl` / `inflate` (static libprotobuf)
  - **Fix:** build only inside `servingsim_docker`

### Simulations
- [ ] `--dtype float16` with bundled Llama-3.1-8B on RTXPRO6000 → missing `profiler/perf/.../fp16` (only `bf16` shipped)
  - **Fix:** use `--dtype bfloat16`
- [ ] **TP=64** for Llama-3.1-8B — invalid (`num_attention_heads=32`; max power-of-2 TP is 32)
- [ ] **1024-GPU** generated config — not run yet (expect long runtime / memory)
- [ ] `generated_tp2_pp2.json` — **in progress / pending** (Jun 10; ~9m+ silent PP startup observed, not hung — see calibration table)
- [ ] `generated_moe_dp_ep.json` (Qwen3-30B-A3B) — config valid, **sim not run yet**

---

## Runtime calibration

**Canonical data:** `calibration/runtimes/runs.jsonl` (one JSON record per run). See `calibration/runtimes/README.md`.

**Auto-recorded:** every successful `python -m serving` appends to `runs.jsonl` (disable: `LLMSERVINGSIM_RECORD_RUNTIME=0`). Manual backfill: `scripts/record_runtime.py`.

Use `runs.jsonl` to tune `_CALIBRATION` in `scripts/generate_cluster_config.py`.

**Recorded so far:** single instance (~13s), 8-GPU replicas (~100.5s). `generated_tp2_pp2` — pending.

---

## Known constraints (for config design)

- **Profiles:** Bundled RTXPRO6000 + Llama-3.1-8B only has `bf16/tp1` and `bf16/tp2`. Higher TP needs re-profiling.
- **dtype:** Match profile variant (`bfloat16` → `bf16` folder).
- **Parallelism:** `num_npus == tp_size * pp_size`. DP via `--dp-group` across instances; EP for MoE models.
- **Build path:** Simulator compile + run inside Docker; host bare-metal build not recommended.
- **Generator:** Homogeneous clusters only (same TP/PP/EP/DP per instance). Mixed per-instance layouts = hand-edited JSON.

---

## Commands that work today

### Single GPU (smoke test)
```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0
```

### 8-GPU generated cluster
```bash
python3 scripts/generate_cluster_config.py \
  --num-nodes 2 --instances-per-node 4 \
  --output configs/cluster/generated_8gpu.json

python -m serving \
  --cluster-config 'configs/cluster/generated_8gpu.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --request-routing-policy LOAD \
  --output 'outputs/generated_8gpu_run.csv' \
  --log-interval 1.0
```

### Regenerate other configs (validated, not all sim-run)
```bash
python3 scripts/generate_cluster_config.py \
  --num-nodes 1 --instances-per-node 1 --tp-size 2 --pp-size 2 \
  --output configs/cluster/generated_tp2_pp2.json

python3 scripts/generate_cluster_config.py \
  --num-nodes 2 --instances-per-node 1 \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tp-size 2 --ep-size 4 --dp-group A \
  --link-bw 128 16 --link-latency 500 20000 \
  --output configs/cluster/generated_moe_dp_ep.json
```

---

## Changelog

| Date | Note |
|------|------|
| 2026-06-10 | Docker installed; submodules fixed; container build OK |
| 2026-06-10 | First successful sim: single instance, `bfloat16` |
| 2026-06-10 | Created `scripts/generate_cluster_config.py` for TP/PP/EP/DP combinations and large-scale layouts (beyond bundled cluster JSONs) |
| 2026-06-10 | **8-GPU generated config sim passed** — `outputs/generated_8gpu_run.csv` (~1m 41s) |
| 2026-06-10 | Runtime estimator added to config generator; calibration table in this log |
| 2026-06-15 | **dynamic-dvfs branch:** layer-wise work availability — `--work-events` JSONL + `python -m serving.tools.query_work_state --at-ns T` |
| 2026-06-15 | **feat/dvfs-scale:** \`--dvfs-switch-at\` + \`--dvfs-scale\` latency multiplier at iteration boundary (baseline 1.67e9 ns → 2.20e9 ns @ 1.5× after 0.5s) |

---

## Forward segments (layer-boundary control) — Jun 15, 2026

### Goal
Split each logical forward pass into multiple ASTRA submissions (one per transformer block) so DVFS / hardware alias swaps can fire between layers, not only between batches.

### CLI
```bash
python -m serving ... --forward-segments per_block
```
- Requires `tp_size=1` (single NPU per instance); rejects DP groups.
- Default remains monolithic (`--forward-segments off`).

### Files changed (minimal surface)
| File | Change |
|------|--------|
| `serving/core/forward_segments.py` | **NEW** — registry helpers, shape-based workload slug |
| `serving/core/request.py` | `Batch.layer_cursor`, `num_stages`, `segment_end`, `awaiting_segment_submit` |
| `serving/core/scheduler.py` | `_enable_segments`, `_segment_continuation`, `on_segment_done` |
| `serving/core/trace_generator.py` | `_synthesize_trace_stage`, `stage_idx` param, Chakra pad row, trace cache |
| `serving/core/graph_generator.py` | `stage_idx` + skip regen if `llm.0.et` exists |
| `serving/core/utils.py` | `get_workload(..., stage_idx)` |
| `serving/core/work_state.py` | `log_segment_complete` event |
| `serving/__main__.py` | CLI, segment registry, defer `add_done` until final segment |
| `serving/tests/test_forward_segments.py` | **NEW** — unit tests (3) |

### Workload naming (shape reuse)
Segments use `instance{id}_t{total_len}p{num_prefill}_s{stage}` so decode steps with the same token shape reuse traces/graphs across batches.

### Validation
- [x] Unit tests: `python3 -m unittest serving.tests.test_forward_segments`
- [x] Baseline (monolithic) smoke: 1 req, ~7s wall time
- [x] Full `--forward-segments per_block` E2E: 1 req completes (~2m 24s wall; 34 segments × trace/graph gen)
- [x] **ASTRA partial-graph root cause** — segment ET files SIGSEGV without embed+head bookends; see [`docs/forward_segments_debug.md`](docs/forward_segments_debug.md)
- [x] **`add_done` batch_id fix** — final segment passes explicit `batch_id` (iteration≠batch when segmented)
- [x] Full `--forward-segments per_block` E2E after bookend + add_done fixes (~2m 24s wall, 1 req)

### Branch comparison + layer-hardware-alternate (Jun 15, 2026)

Full results: [`outputs/branch_compare/RESULTS.md`](outputs/branch_compare/RESULTS.md)

| Scenario (1-req) | Sim clocks (ns) | vs mono | Notes |
|------------------|-----------------|---------|-------|
| `main` mono | 827,547,677 | — | CSV identical to feat |
| `feat` mono | 827,547,677 | 0% | unchanged default path |
| `feat` seg (`per_block`) | 830,169,131 | +0.31% | bookend stub overhead |
| `feat` seg + `--layer-hardware-alternate` (same device) | 830,169,131 | +0.31% | 0 `dvfs_switch` events; flag gated off |
| `feat` seg + RTXPRO6000↔A6000 | 927,735,608 | +12.1% | 2,310 switches; A6000 seeded @ 1.25× |

Same-device alt run: hide `profiler/perf/A6000` + v0 seed aliases → `resolve_hardware_pair` returns `(RTXPRO6000, RTXPRO6000)` → warning *"layer alternation disabled"*.

### Next steps
1. Wire `--dvfs-schedule` JSON layer triggers (not just blind alternation).
2. Real profiler aliases per GPU/frequency (replace seeded A6000).
3. Extend to `tp>1` / DP with per-segment barriers.

