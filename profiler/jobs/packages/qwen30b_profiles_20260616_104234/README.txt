Qwen3-30B-A3B-Instruct-2507 LLMServingSim profiler bundle
Created: 2026-06-16T10:42:36Z

Contents:
  perf/          V100 baseline (tp1,2,4), V100_<MHz> DVFS (tp1), optional V100_kv* probe
  perf/RTXPRO6000/  Reference bf16 profile (tp1,tp2)
  checkpoints/   Job manifests and KV probe results TSV
  logs/          Recent llmsim_qwen30b_* Slurm stdout/stderr
  meta/INVENTORY.txt  Per-hardware CSV inventory

Smoke settings (V100*): ATTENTION_MAX_KV=256, MSQ=32, MNBT=256, MEASUREMENT_ITERATIONS=1
Done marker: meta.yaml under each hardware tag's fp16/ directory.
