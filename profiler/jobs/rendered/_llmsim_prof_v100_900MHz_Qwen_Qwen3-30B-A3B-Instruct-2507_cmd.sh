set -euo pipefail
source "/data/engs-glass/engs2950/shared/scripts/load_hf_token.sh"
export MODEL='Qwen/Qwen3-30B-A3B-Instruct-2507'
export TP_DEGREES='1'
export HARDWARE='V100_900MHz'
export GPU_FREQ_MHZ='900'
export FULL_PROFILE='0'
export VERBOSITY='--verbose'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
export DTYPE='float16'
export SKIP_SKEW='1'
export MAX_NUM_BATCHED_TOKENS='256'
export MAX_NUM_SEQS='32'
export ATTENTION_MAX_KV='256'
export MEASUREMENT_ITERATIONS='1'
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/run_arc_v100_profile.sh"