# Layer-boundary DVFS validation — LLMServingSim vs real V100

**This is the validation comparison folder.** Numbers here are the simulator
checked against **real ARC V100 hardware** for the layer-boundary DVFS workload.
If a `comparison.md` is present, that is the head-to-head result to trust.

> ⚠️ **STATUS: PENDING re-profiling.** As of the last update this comparison is
> **not yet valid**. The V100 profiler tables that feed the simulator were
> captured with the GPU clock lock not holding (700 MHz ran at ~1102, 900 at
> ~1402, the Qwen 1100 MHz MoE shots throttled to 817 — see
> `profiler/jobs/audit_gpu_clocks.py`). A meaningful comparison requires the
> prerequisites below to be done first. Any `comparison.*` generated before then
> is labelled `PROFILES_SUSPECT` and must not be cited.

## What is being validated

The simulator's **per-transformer-block forward segmentation** with
**scattered layer-boundary DVFS** (`--forward-segments per_block
--dvfs-layer-schedule`), against the real vLLM layer-boundary pause/DVFS bench
run on a single V100. We compare **execution latency** and **energy** per
permutation (throughput is derived — the workload is a single prefill).

## Setup (identical on both sides)

| Parameter | Value |
|-----------|-------|
| GPU | 1× Tesla V100-SXM2-32GB, `tp_size=1` |
| dtype | float16 (`fp16` profiler variant) |
| Workload | single 64-token **prefill**, 1 request, output 0 (prefill-only) |
| Dataset | ShareGPT fixed-length, seed 42 (`...-fixlen64o0.jsonl`) |
| `max_num_batched_tokens` | 256 |
| `max_num_seqs` | phi = 32, qwen = 8 |

| Model key | Model | Decoder layers |
|-----------|-------|----------------|
| `phi` | `microsoft/Phi-tiny-MoE-instruct` | 32 |
| `qwen` | `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 24 |

### Real hardware (ground truth) — ARC HTC V100

- Engine: sync `vLLM.LLM`, `VLLM_USE_V1=0`, prefill-only (`max_tokens=1`).
- Layer-boundary pause via `worker_extension_cls` + external host poller
  (`profiler/jobs/dvfs_barrier_host_poller.sh`) that locks GPU clocks with the
  scattered schedule at each barrier.
- Submitted by the bench campaign; per-run ground truth lands in
  `bench/campaigns/v100_bench_layer_dvfs_mini3x3/runs/<run_id>/results/summary.json`
  (`wall_sec`, `exec_sec`, `pause_sec`, `energy_j`). **Compare against
  `exec_sec`** — the simulator does not model DVFS clock-settle latency, which
  is the `pause_sec` component on real hardware.

### Simulator — LLMServingSim

```
python -m serving \
  --cluster-config <cluster_{phi,qwen}_moe_dvfs.json>  \
  --dataset <shared/sharegpt-...-fixlen64o0.jsonl>      \
  --dtype float16 --max-num-seqs <32|8> --max-num-batched-tokens 256 --num-reqs 1 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --forward-segments per_block \
  --dvfs-layer-schedule <perm>.json
```

- Per-frequency profiler tables: `profiler/perf/V100_<MHz>MHz/<model>/fp16/tp1`
  (1400 MHz uses the default `V100` tag). MHz→tag mapping in
  `serving/core/v100_measured_assets.py`.
- Energy from the simulator power model, anchored to the measured idle/active/
  standby power in `outputs/mini3x3_layer_dvfs_sim/measured_assets_manifest.json`.

### The 6 permutations (scattered `layer → MHz`)

| perm | model | barrier layers → MHz |
|------|-------|----------------------|
| p01 | phi  | 7→1400, 9→700, 10→700, 14→900, 16→1400, 17→1400, 20→900, 26→700 |
| p01 | qwen | 1→1100, 19→900, 20→1400, 22→1400 |
| p02 | phi  | 1→700, 18→1100, 27→900 |
| p02 | qwen | 2→1100, 9→900 |
| p03 | phi  | 3→700, 12→900, 14→1400, 15→1400, 16→1100, 17→1400 |
| p03 | qwen | 0→900, 1→900, 9→1100, 18→1400, 22→900 |

3 iterations each (`i0/i1/i2`); the simulator is deterministic so iterations
collapse, but they are kept to match the bench run matrix.

## Prerequisites before this comparison is valid

1. **Re-profile V100** (at minimum Qwen 700/900/1100; ideally all clocks for
   both models) on ARC with the clock-lock verification active:
   `GPU_FREQ_VERIFY_STRICT=1` in `profiler/jobs/run_arc_v100_profile.sh` (aborts
   if the lock drifts >±100 MHz; post-flight `audit_gpu_clocks.py` gate).
2. **Sync the bench ground truth** from ARC into a local directory, one
   `<run_id>/results/summary.json` per permutation×iteration.
3. **Re-run the simulator sweep** on the fresh profiles, then generate the
   comparison (below).

## Generate / refresh the comparison

```
python3 scripts/validate_dvfs_vs_hardware.py \
  --sim-runs   outputs/mini3x3_layer_dvfs_sim/runs \
  --bench-runs <synced ARC bench>/runs \
  --out        validation/layer_boundary_dvfs
```

Writes `comparison.csv` (machine-readable) and `comparison.md` (per-permutation
sim-vs-real latency/energy with % error, plus a provenance header recording the
profiler clock-audit status so a suspect run can never be mistaken for a clean
one).

## Known reference points (PRELIMINARY — profiles suspect)

> These predate re-profiling. The feeding V100 tables failed the clock audit, so
> treat magnitudes as order-of-magnitude only and **do not** trust per-frequency
> deltas. Recorded here as a sanity anchor, not as the validation result.

**Real hardware anchor (user, earlier ARC run):** ~**600 J** for a single Qwen
prefill (prefill-only), **GPU-only** measurement (user recollection — to confirm
against the synced bench data). Compare against the simulator's **GPU/NPU-only**
energy, not the system total. Baseline standing: real ~600 J vs sim 415 J →
simulator is ~30% low on GPU energy (expected to shift once the V100 profiles are
re-measured with the clock lock holding). Exact schedule for this anchor TBD.

**Simulator, Qwen single 64-tok prefill @ default V100 (no DVFS, no segments):**

| metric | value |
|--------|------:|
| GPU/NPU energy | 415 J |
| total-system energy | 1220 J |
| latency | ~3.0 s |

The ~3 s latency is consistent with the real run (~600 J ÷ ~200 W ≈ 3 s), i.e.
both sim and bench measure the same barrier-laden prefill-only mode. The ~600 J
real anchor sits between the sim's GPU-only (415 J) and total-system (1220 J).

**Simulator, DVFS permutations (total-system energy, `PROFILES_SUSPECT`):**

| perm | model | latency (s) | total energy (J) |
|------|-------|------------:|-----------------:|
| p01 | qwen | 3.23 | 1000 |
| p02 | qwen | 3.07 | 960 |
| p03 | qwen | 3.10 | 640 |
| p01 | phi | 0.87 | 340 |
| p02 | phi | 0.95 | 330 |
| p03 | phi | 0.85 | 250 |
