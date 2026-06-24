# LLMServingSim TODO

## Pending investigations

### Bench + profiler campaign failures (Jun 22)
- **Bench campaign** `bench/campaigns/v100_bench_layer_dvfs_20260622/` — 78 runs submitted, only 4 completed (`scat_p01_phi_i0/i1/i2`, `scat_p01_qwen_i0`). Jobs 8010073–8010150 all gone from queue. Need to check failure logs and diagnose.
- **Profiler campaign** `profiler/campaigns/v100_full_dvfs_20260622/` — 78 runs submitted, ~6 partial results, no summaries. Same issue.
- **Action:** check slurm `.out`/`.err` logs for a failed run, find root cause, re-submit.

### Pending H100 profiler v0 jobs (cancelled Jun 23)
- Jobs 8010183–8010236 — `profiler/v0/jobs/rendered/moe_*.sbatch` for H100 TP8 across Phi-tiny, Mistral, DeepSeek, Meta, Qwen MoE models. Cancelled to free queue. Re-submit when ready.

### Other cancelled pending jobs
- `fig_bypass` (8007746, 8007754, 8008556, 8008562) — bypass DVFS fig experiment
- `llmsim_*` devel jobs (7972506, 7972725, 7972811) — stuck on PartitionTimeLimit
- `smi_setc` (8007821) — clock verification job
- `moe_only`, `bash` (7967218) — misc devel leftovers

## Active work

- Extending LLMServingSim to support DVFS — validating recent changes (in progress Jun 23)

### Validation runs in flight (Jun 24)
- **§0 re-profile** (force, fixed clock-verify framing): jobs `8021127`–`8021139`, chained.
  13 model×freq: Phi-tiny-MoE + Qwen1.5-MoE @ {700,900,1100,1300}, Qwen3-30B @ {700,900,1100,1300,1400}. tp1, `GPU_FREQ_VERIFY_STRICT=1`.
  Gate for everything else. When done: `python3 profiler/jobs/audit_gpu_clocks.py profiler/perf/V100_*` must be green; expect monotonic clock→latency (resolves "1100 < 900" artifact).
- **§1 per-frequency homogeneous bench**: jobs `8021235`–`8021242`, chained sequentially (no two locked-clock benches share a GPU).
  {700,900,1100,1300} × {Phi-tiny-MoE, Qwen1.5-MoE}. Unchanged `python -m bench run` (no pause hooks), fixed 64-tok prefill / 64-tok decode, 100 reqs.
  Ported profiler's clock-verify into `run_arc_v100_bench_throughput.sh` (pre-flight read-back + post-flight audit) and added `GPU_FREQ_MHZ`/`HARDWARE`/`DEPEND_JID` passthrough to `submit_arc_v100_bench_throughput.sh`.
  Output: `bench/results/V100_<MHz>/<model>/tp1/<jid>/` — `run_exec_metrics.json` (energy_j; pause=0 so energy_j IS active energy), `gpu_power/bench.jsonl`, `timeseries.csv`.
