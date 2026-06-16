set -euo pipefail
export HF_TOKEN=''
export MODEL='mistralai/Mixtral-8x7B-v0.1'
export TP_DEGREES='1,2,4'
export HARDWARE='V100'
export FULL_PROFILE='0'
export VERBOSITY='--verbose'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/run_arc_v100_profile.sh"