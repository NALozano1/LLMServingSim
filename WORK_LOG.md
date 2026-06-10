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

## Runtime calibration (wall clock)

Use this table to tune `_CALIBRATION` in `scripts/generate_cluster_config.py`.
Dataset: `workloads/example_trace.jsonl` (10 requests) unless noted.
Host: nserver15, `servingsim_docker`, `--dtype bfloat16`, `--log-interval 1.0`.

| Config | Instances | GPUs | TP | PP | EP | DP | Wall time | First `[1.0s]` log | Notes |
|--------|-----------|------|----|----|----|----|-----------|-------------------|-------|
| `single_node_single_instance.json` | 1 | 1 | 1 | 1 | — | — | **~13s** | ~1s | Sim prints `Total simulation time: 0h 0m 13.160s` |
| `generated_8gpu.json` | 8 | 8 | 1 | 1 | — | — | **~1m 41s** | ~1s | `Total simulation time: 0h 1m 40.547s` |
| `generated_tp2_pp2.json` | 1 | 4 | 2 | 2 | — | — | *TBD* | *~9–10m est.* | Long PP startup; was ~100% CPU at 9m, no heartbeat yet |

**Where runtime appears:**
- **Simulator stdout:** `Total simulation time: 0h Xm Ys` at end of every successful run
- **This file:** calibration table above (manual — update after each run)
- **Generator:** heuristic only until we plug in calibrated rows

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
