# MoE power-by-subtraction validation — RUNBOOK (run on nserver15)

Validate that the simulator's **MoE-layer** GPU energy is correct by backing it
out of the REAL all-GPU energy measurement and comparing to an independent
expectation built from measured MoE GPU power.

**Why nserver15 only:** these `python -m serving ...` runs need ASTRA-Sim
(compiled) + `msgspec`. Neither is available on the ARC host where this package
was staged, so the two sim runs and the subtraction were NOT executed here — the
patch was `py_compile`d and the subtraction script unit-tested with fabricated
inputs only.

All paths below are relative to the repo root
`/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim`.

---

## 0. Pull / apply the patch

The MoE-energy-split patch is already applied in this working tree. If you pull a
clean checkout on nserver15 instead, apply it with:

```bash
git apply bench/jobs/moe_subtraction/sim_power_split.patch
```

Patch touches only: `serving/core/power_model.py`, `serving/core/trace_generator.py`,
`serving/__main__.py`. It is guarded so power-modeling-OFF behaviour is byte-identical
(the MoE buckets simply stay 0.0 / empty).

Sanity check:
```bash
python -m py_compile serving/core/power_model.py serving/core/trace_generator.py serving/__main__.py
```

---

## Timing-tag vs power-value split (READ THIS)

Qwen1.5-MoE and Llama-8B have profiler **timing** tables ONLY under the default
`V100` tag (= 1400 MHz; there are no 900 MHz `moe.csv`/`attention.csv` for either
model). So both cluster configs use:

- **hardware tag `V100`**  → selects the **1400 MHz** profiler timing tables.
- **`power["npu"]["V100"]` values** → the **measured 900 MHz** numbers
  (idle **50.03 W**, active **154.43 W**).
- **`--dvfs-scale 1.556`** → stretches the 1400 MHz layer latencies to emulate
  **900 MHz timing**. `1.556 = 1400 / 900` (clock-linear / compute-bound
  approximation; the profiler captures show the GPU actually held ~1402 MHz, so
  "achieved 1400" ≈ nominal). This multiplies every profiled layer latency —
  including the MoE `moe_time_s` — so the emulated MoE time matches ~900 MHz.

Net: **timing ≈ 900 MHz** (via `--dvfs-scale`) and **power values = measured 900 MHz**,
so `energy = power × time` is a consistent 900 MHz estimate even though the timing
tables carry the `V100`/1400 tag.

Instance `hardware` is set to `"V100"` (matching the single `power["npu"]` key),
which also sidesteps the `build_dvfs_cluster_config` hardcoded-hardware KeyError at
`serving/core/config_builder.py:387`. The configs are hand-written for this reason
(the helper would key power by `V100_900MHz` and try to load nonexistent 900 MHz
timing tables).

---

## 1. Qwen1.5-MoE arm (MoE)

```bash
mkdir -p bench/jobs/moe_subtraction/out/qwen15moe_900

python -m serving \
  --cluster-config bench/jobs/moe_subtraction/cluster_qwen15moe_900.json \
  --dataset bench/campaigns/v100_prefill_valid_20260702/shared/sharegpt-Qwen_Qwen1.5-MoE-A2.7B-Chat-50-sps100-seed42-ml4096-fixlen512o0.jsonl \
  --num-reqs 50 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --dtype float16 \
  --dvfs-scale 1.556 \
  --output bench/jobs/moe_subtraction/out/qwen15moe_900/requests.csv
```

Produces:
- `bench/jobs/moe_subtraction/out/qwen15moe_900/requests.csv` (per-request metrics)
- `bench/jobs/moe_subtraction/out/qwen15moe_900/energy_breakdown.json`
  (written next to `--output`; keys: `npu_total_j`, `moe_active_energy_j`,
  `moe_time_s`, `nonmoe_active_energy_j`, `idle_energy_j`, `active_power_w`,
  `idle_power_w`).

> Note: `--dataset` points at the **shared** sharegpt file the real run consumed
> (see the run's `meta.json` `dataset_path`), NOT the run's `requests.jsonl`
> (that file is the real run's per-request *output* and is not a sim input format).
> Prefix caching / chunked prefill are left at their defaults; prompts are random
> 512-token ids so radix reuse is negligible. Add `--no-enable-prefix-caching` if
> you want to force it off.

## 2. Llama-3.1-8B arm (DENSE control)

```bash
mkdir -p bench/jobs/moe_subtraction/out/llama8b_900

python -m serving \
  --cluster-config bench/jobs/moe_subtraction/cluster_llama8b_900.json \
  --dataset bench/campaigns/v100_prefill_valid_20260702/shared/sharegpt-meta-llama_Llama-3.1-8B-50-sps100-seed42-ml4096-fixlen512o0.jsonl \
  --num-reqs 50 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --dtype float16 \
  --dvfs-scale 1.556 \
  --output bench/jobs/moe_subtraction/out/llama8b_900/requests.csv
```

Dense model → `moe_active_energy_j == 0`, `moe_time_s == 0` in its breakdown.

---

## 3. Subtraction / agreement

```bash
# Qwen MoE arm (MoE p95 power auto-read from profiler captures = 141.06 W;
# override with --moe-p95-power-w if desired)
python bench/jobs/moe_subtraction/moe_power_subtraction.py --mode moe \
  --summary bench/campaigns/v100_prefill_valid_20260702/runs/v100_qwen15moe_tp1_900mhz/results/summary.json \
  --breakdown bench/jobs/moe_subtraction/out/qwen15moe_900/energy_breakdown.json

# Llama dense control
python bench/jobs/moe_subtraction/moe_power_subtraction.py --mode dense \
  --summary bench/campaigns/v100_prefill_valid_20260702/runs/v100_llama8b_tp1_900mhz/results/summary.json \
  --breakdown bench/jobs/moe_subtraction/out/llama8b_900/energy_breakdown.json
```

Math:
- `sim_nonMoE      = npu_total_j - moe_active_energy_j`
- `MoE_backed_out  = measured_all_gpu_j - sim_nonMoE`
- `expected_MoE    = (moe_p95_power_w - idle_power_w) * moe_time_s`   (idle 50.03 W, MoE p95 141.06 W)
- `agreement_pct   = 100 * MoE_backed_out / expected_MoE`
- dense control: `control_residual_j = measured_all_gpu_j - npu_total_j` (≈ 0)

Real measured all-GPU energies (single V100 @ 900 MHz), already in the summaries:
- qwen1.5-MoE = **42706.482 J** (102.6 W)
- llama8b (dense) = **1472.808 J** (110.2 W)

Self-test the script anytime with:
```bash
python bench/jobs/moe_subtraction/moe_power_subtraction.py --selftest
```

---

## 4. Send back

- `bench/jobs/moe_subtraction/out/qwen15moe_900/energy_breakdown.json`
- `bench/jobs/moe_subtraction/out/qwen15moe_900/requests.csv`
- `bench/jobs/moe_subtraction/out/llama8b_900/energy_breakdown.json`
- `bench/jobs/moe_subtraction/out/llama8b_900/requests.csv`
- the two subtraction-script stdout blocks (JSON) from step 3
