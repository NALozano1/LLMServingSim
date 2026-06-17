#!/usr/bin/env bash
# Smoke test: in-place layer-boundary DVFS pause during profiler fire().
#
# Runs profiler inside Apptainer; host-side shell poller applies DVFS
# (sudo nvidia-smi-clocks) while the vLLM worker blocks between layers.
#
#   export HF_TOKEN=...
#   bash profiler/jobs/run_arc_v100_layer_dvfs_smoke.sh
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"
HOST_POLLER="${JOBS_ROOT}/dvfs_barrier_host_poller.sh"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
HARDWARE="${HARDWARE:-V100_layer_dvfs_smoke}"
TP_DEGREES="${TP_DEGREES:-1}"
DTYPE="${DTYPE:-float16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
VERBOSITY="${VERBOSITY:-"--verbose"}"

export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export DVFS_LAYER_PAUSE_MAX_SHOTS="${DVFS_LAYER_PAUSE_MAX_SHOTS:-1}"
export DVFS_FREQ_SCHEDULE="${DVFS_FREQ_SCHEDULE:-700,900,1100}"
export GPU_FREQ_SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-1.0}"
export ENGS2950_ROOT="${ENGS_GLASS}"

VARIANT_TAG="fp16"
case "${DTYPE}" in
  float16) VARIANT_TAG="fp16" ;;
  bfloat16) VARIANT_TAG="bf16" ;;
  *) VARIANT_TAG="${DTYPE}" ;;
esac

PERF_VARIANT_ROOT="${REPO_ROOT}/profiler/perf/${HARDWARE}/${MODEL}/${VARIANT_TAG}"
TP_WATCH_ROOT="${PERF_VARIANT_ROOT}/tp1"
FREQ_META_DIR="${TP_WATCH_ROOT}/gpu_freq"

mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${TP_WATCH_ROOT}" "${FREQ_META_DIR}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"
trap 'kill "${POLLER_PID:-}" 2>/dev/null || true; gpu_freq_lock_trap_restore "${FREQ_META_DIR}"' EXIT

chmod +x "${HOST_POLLER}"
bash "${HOST_POLLER}" "${TP_WATCH_ROOT}" "${FREQ_META_DIR}" &
POLLER_PID=$!

echo "=== Layer-boundary DVFS smoke (ARC / engs-glass) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL=$MODEL  HARDWARE=$HARDWARE  TP=$TP_DEGREES"
echo "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}  GPU_FREQ_SETTLE_SEC=${GPU_FREQ_SETTLE_SEC}"
echo "DVFS_LAYER_PAUSE_MAX_SHOTS=${DVFS_LAYER_PAUSE_MAX_SHOTS}"
echo "HOST_POLLER_PID=${POLLER_PID}"
echo "WATCH_ROOT=${TP_WATCH_ROOT}"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset — gated models may fail." >&2
fi

OLD_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
unset SLURM_JOB_ACCOUNT

"$CONTAINER_RUNTIME" exec --cleanenv --nv \
  -B "${REPO_ROOT}:${REPO_ROOT}" \
  -B "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  -B "${SCRATCH}:${SCRATCH}" \
  -B /dev/shm:/dev/shm \
  --pwd "${REPO_ROOT}" \
  --env "HOME=${SCRATCH}/h" \
  --env "TMPDIR=${SCRATCH}/t" \
  --env "VLLM_CACHE_ROOT=${SCRATCH}/v" \
  --env "TRITON_CACHE_DIR=${SCRATCH}/triton" \
  --env "TORCH_HOME=${SCRATCH}/torch" \
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "DVFS_LAYER_PAUSE_MAX_SHOTS=${DVFS_LAYER_PAUSE_MAX_SHOTS}" \
  --env "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}" \
  --env "GPU_FREQ_SETTLE_SEC=${GPU_FREQ_SETTLE_SEC}" \
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec python3 -m profiler slice "'"${MODEL}"'" \
    --hardware "'"${HARDWARE}"'" \
    --tp-refresh "'"${TP_DEGREES}"'" \
    --group dense \
    --dtype "'"${DTYPE}"'" \
    --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
    --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
    --attention-max-kv "'"${ATTENTION_MAX_KV}"'" \
    --measurement-iterations "'"${MEASUREMENT_ITERATIONS}"'" \
    --skip-skew \
    --force \
    '"${VERBOSITY}"''

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

echo "=== Smoke done ==="
echo "Markers: ${TP_WATCH_ROOT}/dvfs_markers.jsonl"
if [[ -f "${TP_WATCH_ROOT}/dvfs_markers.jsonl" ]]; then
  echo "--- dvfs_markers.jsonl ---"
  cat "${TP_WATCH_ROOT}/dvfs_markers.jsonl"
fi
