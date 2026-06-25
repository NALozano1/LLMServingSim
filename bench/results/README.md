# Bench Results

Aggregated validation results committed here so they travel with the code.
Raw campaign data (timeseries, per-request JSONL, GPU power traces) lives on the
ARC filesystem under `bench/campaigns/` (gitignored — too large to commit).

## Directory layout

```
bench/results/
  <campaign_name>/
    validation_results.tsv     — flat table, one row per (model, clock) arm
    validation_summary.json    — structured summary + per-model clock-sweep table
```

## How to populate after runs complete

```bash
cd /data/engs-glass/engs2950/DVFS-MoE/LLMServingSim

# Collect and stage to bench/results/
python3 bench/jobs/collect_prefill_validation_results.py \
    bench/campaigns/v100_prefill_valid_20260625/ \
    --stage-to bench/results/

# Commit
git add bench/results/
git commit -m "results: add V100 prefill validation results (Phi tp1, Qwen tp4)"
```

Repeat for the 256-token campaign:
```bash
python3 bench/jobs/collect_prefill_validation_results.py \
    bench/campaigns/v100_prefill_valid_256tok_20260625/ \
    --stage-to bench/results/

git add bench/results/
git commit -m "results: add V100 256-tok prefill validation (Phi tp1)"
```

## Pending campaigns (as of 2026-06-25)

All jobs are queued on htc-g049. Expected completion ~4-5 hours after queueing.

| Campaign | Models | Input len | Clock arms | Status |
|----------|--------|-----------|------------|--------|
| `v100_prefill_valid_20260625` | Phi-tiny tp1, Qwen3-30B tp4 | 512 tok | uncapped + 700/900/1100/1300/1400 MHz | running/queued |
| `v100_prefill_valid_256tok_20260625` | Phi-tiny tp1 | 256 tok | uncapped + 700/900/1100/1300/1400 MHz | queued |

Monitor: `squeue -M htc -u $USER --format='%.10i %.30j %.2t %.10M %R'`

## Using results for LLMServingSim validation

### Tier 1 — uncapped accuracy

Compare `ttft_median_ms` from the uncapped arm against LLMServingSim without any
`--dvfs-scale`:

```bash
# On nserver15 inside the Docker container:
python -m serving \
  --cluster-config configs/cluster/generated_phi_tp1.json \
  --dtype float16 \
  --dataset bench/campaigns/v100_prefill_valid_20260625/shared/sharegpt-*.jsonl \
  --output outputs/phi_tp1_uncapped_sim.csv
```

### Tier 2 — per-clock accuracy

For each locked-clock arm, run the sim with `--dvfs-scale`:
```
dvfs_scale ≈ ttft_hw_at_clock / ttft_hw_uncapped
```

The sim uses a uniform latency multiplier; the validation checks whether that
linear approximation holds across the V100's clock range (700–1400 MHz).

### Key fields in validation_results.tsv

| Column | Meaning |
|--------|---------|
| `arm_label` | `uncapped` or `<N>mhz` |
| `gpu_freq_mhz_target` | Requested clock (null = uncapped) |
| `gpu_freq_mhz_achieved` | Mean achieved clock during inference (from NVML) |
| `clock_verdict_ok` | True if achieved ≈ target (within tolerance) |
| `ttft_median_ms` | Median time-to-first-token across all requests |
| `ttft_p90_ms` | P90 TTFT |
| `energy_excl_pause_j` | GPU energy excluding any barrier pause time |
| `mean_power_excl_pause_w` | Mean GPU power during inference |
