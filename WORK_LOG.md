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

---

## ARC / HTC — real vLLM layer-boundary pause + DVFS (bench)

**Branch:** `feat/layer-boundary-dvfs-pause`  
**Host:** ARC HTC V100 (`interactive` partition, 1× GPU per job)

### What works (prefill-only, validated)

- [x] **Pause-only smoke** — `python -m bench run` with sync `vLLM.LLM` (`VLLM_USE_V1=0`), `worker_extension_cls`, external host poller (`dvfs_barrier_host_poller.sh`). Job `8009732` PASS: 32 layer markers, `wall≈17s`, `exec≈8.2s`, `pause≈8.7s`.
- [x] **Layer-boundary DVFS** — host poller applies `nvidia-smi-clocks` at each barrier; `run_exec_metrics.json` records wall/exec/pause/energy/throughput.
- [x] **78-run campaign** — same scattered + fixed permutations as profiler plan (`bench/jobs/generate_bench_layer_dvfs_campaign.py`, seed `20260616`). Per-run `sim_replication.json` + `results/summary.json` for simulator replay.
- [x] **Single-GPU targeting** — `CUDA_VISIBLE_DEVICES=0` / `GPU_FREQ_GPU_INDICES=0` (Slurm `--exclusive` exposed all node GPUs and inflated power).

### Bench entrypoints

| Script | Purpose |
|--------|---------|
| `bench/jobs/run_arc_v100_bench_layer_pause_smoke.sh` | Pause-only prefill (`VLLM_BENCH_PREFILL_ONLY=1`) |
| `bench/jobs/run_arc_v100_bench_layer_pause_decode_smoke.sh` | Prefill + decode; `DVFS_DECODE_MAX_PAUSES_PER_PASS=1` |
| `bench/jobs/run_arc_v100_bench_layer_boundary.sh` | Prefill DVFS smoke |
| `bench/jobs/run_arc_v100_bench_layer_campaign.sh` | Single campaign permutation |
| `bench/jobs/submit_arc_v100_bench_layer_campaign.sh` | Submit chained 78-run campaign |

### Key env vars (bench)

| Variable | Purpose |
|----------|---------|
| `VLLM_LAYER_PAUSE=1` | Enable in-place layer hooks |
| `VLLM_EXTERNAL_HOST_POLLER=1` | Poller outside Apptainer (sudo clocks on ARC) |
| `VLLM_BENCH_PREFILL_ONLY=1` | Prefill-only (no decode steps) |
| `VLLM_BENCH_GPU_POWER=1` | Sample `gpu_power/bench.jsonl` |
| `DVFS_BARRIER_LAYERS` | Scattered mode: which layers pause |
| `DVFS_FREQ_SCHEDULE` | MHz per barrier (or fixed MHz for all layers) |
| `DVFS_DECODE_MAX_PAUSES_PER_PASS=1` | Decode: max one pause per forward pass |

### Architecture notes

- **Sync `LLM` not `AsyncLLM`** — V1 async decode deadlocks when forward hooks block (~`DVFS_BARRIER_TIMEOUT_SEC`).
- **Prefill-only** uses `max_tokens=1` (vLLM rejects 0); V0 hooks still fire on prefill only when `VLLM_BENCH_PREFILL_ONLY=1`.
- **Decode** (new): token-count heuristic in `InPlaceLayerBarrier` — full barriers on prefill, at most one pause per decode forward (first eligible layer).
- **Do not use** profiler `python -m profiler slice` campaign for full-model dense forwards — dummy 1-layer models mismatch `DVFS_BARRIER_LAYERS`.

### Replication artifacts (per campaign run)

- `runs/<run_id>/sim_replication.json` — barrier layers, MHz schedule, profiler profile paths
- `runs/<run_id>/results/summary.json` — wall/exec/pause/energy/throughput/markers
- `runs/<run_id>/artifacts/` — copies of `run_exec_metrics.json`, `dvfs_markers.jsonl`, `gpu_power/`, `node_meta.json`

Collect: `python3 bench/jobs/collect_bench_layer_campaign_results.py bench/campaigns/v100_bench_layer_dvfs_20260622/`

### Changelog (ARC bench)

| Date | Note |
|------|------|
| 2026-06-22 | Prefill-only bench layer pause + DVFS validated on V100; 78-run campaign submitted |
| 2026-06-22 | Decode support: `DVFS_DECODE_MAX_PAUSES_PER_PASS`, decode smoke job |
| 2026-06-24 | Transition calib (`run_arc_v100_bench_transition_calib.sh`): measures DVFS barrier overhead in sync vs async mode. `barrier_wait_sec_worker` is the per-barrier blocking time seen by the vLLM worker thread. Sync n=1: 17.3 s (dominated by clock settle). Async n=1: 0.7 s (dispatch returns before clock settles — worker unblocked immediately). |
| 2026-06-24 | Node heterogeneity confirmed: g049 (12 h interactive) is the only interactive node that honours `sudo nvidia-smi-clocks`. g048 does not bind. H100 on short partition (`htc-g060`) binds; L40S and V100-PCIE on short do not. |
| 2026-06-24 | Async barrier marker bug found + fixed: calib runner was killing the host poller immediately after the container exited, before the Python marker-write block completed. Fixed with a 30 s drain wait (polling `dvfs_markers.jsonl` for expected marker count) before sending SIGTERM. |
| 2026-06-25 | Prefill-only sim-accuracy validation matrix: `run_arc_v100_prefill_validation.sh` + `submit_prefill_validation_matrix.sh`. Phi tp1 + Qwen3-30B tp4 at {uncapped, 700, 900, 1100, 1300, 1400} MHz on g049. No layer-pause; clock-locked + audited; `gpu_power` on. Tier-1 (no-DVFS uncapped) + Tier-2 (per-clock accuracy) for LLMServingSim validation. `collect_prefill_validation_results.py` aggregates results. |
| 2026-06-25 | Fixed `--sps` omission in prefill validation dataset generator call (was always required); resubmitted all 12 validation jobs (8030500–8030511). |
| 2026-06-25 | Fixed `bench_prefill_only_enabled` scope bug in `bench/core/runner.py`: import was placed in `_drive`'s local scope but the call site is inside `_one()`, a closure in `_submit_all()` — a separate top-level function. Moved import to top of `_submit_all`. Commit `f876f194`. Re-queued 3 failed Phi 512-tok runs as 8030549–8030551. |
| 2026-06-25 | Populated `profiler/perf/V100/Qwen/Qwen3-30B/fp16/tp4/moe.csv` from dvfs-policy tp4 uncapped run (tokens 1–4096, ae 2–32). Commit `c6e47d37`. |
| 2026-06-25 | Added Phi 256-tok validation campaign (`v100_prefill_valid_256tok_20260625`, jobs 8030529–8030535): profiler trace max is 256 tokens — this campaign stays within range for a clean attention extrapolation check. |

---

## Future work

- **P/D disaggregation + `--forward-segments per_block`.** Currently mutually
  exclusive: per-block layer-boundary DVFS is gated to colocated, single-NPU
  (`tp_size=1`, no DP) instances, and a `pd_type != null` cluster now raises a
  clear error instead of hanging. Supporting both would let prefill and decode
  workers scale clocks independently. Needs: (1) `Scheduler.on_segment_done` to
  use a prefill instance's true (doubled) NPU range when gating segment
  completion, and (2) coordination of the prefill->decode handoff across
  segmented passes. Guard lives in `serving/__main__.py` (forward_segments setup).
