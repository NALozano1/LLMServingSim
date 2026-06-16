set -euo pipefail
export HF_TOKEN=''
export MODEL='meta-llama/Llama-3.1-8B'
export TP_DEGREES='1'
export HARDWARE='V100_900MHz'
export GPU_FREQ_MHZ='900'
export FULL_PROFILE='0'
export VERBOSITY='--verbose'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
export DTYPE='float16'
export SKIP_SKEW=''
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/run_arc_v100_profile.sh"