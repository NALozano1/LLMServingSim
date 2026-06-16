set -euo pipefail
source "/data/engs-glass/engs2950/shared/scripts/load_hf_token.sh"
export MODEL='Qwen/Qwen3-30B-A3B-Instruct-2507'
export DTYPE='float16'
export MAX_NUM_BATCHED_TOKENS='2048'
export MAX_NUM_SEQS='256'
export MEASUREMENT_ITERATIONS='1'
export KV_PROBE_STEPS='256 512 1024 2048 4096 8192'
export KV_PROBE_STEP_TIMEOUT_SEC='7200'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/run_arc_v100_kv_probe.sh"