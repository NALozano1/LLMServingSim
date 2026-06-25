set -euo pipefail
source "/data/engs-glass/engs2950/shared/scripts/load_hf_token.sh"
export CALIB_DIR='bench/campaigns/v100_tcalib_slim_sync'
export RUN_ID='tcalib_n000_phi_i00'
export CALIB_SPEC_JSON='bench/campaigns/v100_tcalib_slim_sync/runs/tcalib_n000_phi_i00/calib_spec.json'
export DVFS_APPLY_MODE='sync'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/bench/jobs/run_arc_v100_bench_transition_calib.sh"