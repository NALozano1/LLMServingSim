#!/usr/bin/env bash
# Incremental ATTENTION_MAX_KV probe for Qwen3-30B on V100 (tp1, attention only).
#
# Steps through KV caps (default: 256→8192, doubling) until timeout or failure.
# Each step writes to profiler/perf/V100_kv<MHz>/<MODEL>/fp16/tp1/attention.csv
# and appends a row to profiler/jobs/checkpoints/qwen30b_kv_probe_results.tsv.
#
# Usage (on an allocated V100 node or inside Slurm {{COMMAND}}):
#   bash profiler/jobs/run_arc_v100_kv_probe.sh
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DTYPE="${DTYPE:-float16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
KV_PROBE_STEP_TIMEOUT_SEC="${KV_PROBE_STEP_TIMEOUT_SEC:-7200}"
KV_PROBE_STEPS="${KV_PROBE_STEPS:-256 512 1024 2048 4096 8192}"
KV_PROBE_HARDWARE_PREFIX="${KV_PROBE_HARDWARE_PREFIX:-V100_kv}"
PROBE_RESULTS="${PROBE_RESULTS:-${JOBS_ROOT}/checkpoints/qwen30b_kv_probe_results.tsv}"

mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "$(dirname "${PROBE_RESULTS}")"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"
trap 'gpu_freq_lock_trap_restore "${REPO_ROOT}/profiler/perf/_kv_probe_gpu_freq" 2>/dev/null || true' EXIT

echo "=== Qwen3-30B KV probe (ARC / engs-glass) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL=$MODEL  DTYPE=$DTYPE"
echo "MAX_NUM_SEQS=$MAX_NUM_SEQS  MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "KV_PROBE_STEPS=${KV_PROBE_STEPS}"
echo "KV_PROBE_STEP_TIMEOUT_SEC=${KV_PROBE_STEP_TIMEOUT_SEC}"
echo "PROBE_RESULTS=${PROBE_RESULTS}"
echo "SCRATCH=$SCRATCH"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset — gated models may fail." >&2
fi

if [[ ! -f "${PROBE_RESULTS}" ]]; then
  printf 'recorded_at\tjob_id\tattention_max_kv\thardware\tstatus\telapsed_sec\tattention_rows\tnotes\n' \
    > "${PROBE_RESULTS}"
fi

read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
cd "${REPO_ROOT}"
for kv in ${KV_PROBE_STEPS}; do
  hw="${KV_PROBE_HARDWARE_PREFIX}${kv}"
  out="${REPO_ROOT}/profiler/perf/${hw}/${MODEL}/fp16/tp1/attention.csv"
  echo ""
  echo "=== KV probe: ATTENTION_MAX_KV=${kv} HARDWARE=${hw} ==="
  t0=$(date +%s)
  status=ok
  notes=""
  if ! timeout "${KV_PROBE_STEP_TIMEOUT_SEC}" python3 -m profiler slice "${MODEL}" \
      --hardware "${hw}" \
      --tp 1 \
      --tp-refresh 1 \
      --group attention \
      --dtype "${DTYPE}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --attention-max-kv "${kv}" \
      --measurement-iterations "${MEASUREMENT_ITERATIONS}" \
      --skip-skew \
      --force \
      --verbose; then
    ec=$?
    if [[ "${ec}" -eq 124 ]]; then
      status=timeout
      notes="exceeded ${KV_PROBE_STEP_TIMEOUT_SEC}s"
    else
      status=failed
      notes="exit ${ec}"
    fi
  fi
  elapsed=$(( $(date +%s) - t0 ))
  rows=0
  if [[ -f "${out}" ]]; then
    rows=$(( $(wc -l < "${out}") - 1 ))
    [[ "${rows}" -lt 0 ]] && rows=0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-local}" "${kv}" "${hw}" \
    "${status}" "${elapsed}" "${rows}" "${notes}" >> "${PROBE_RESULTS}"
  echo "=== step result: kv=${kv} status=${status} elapsed=${elapsed}s rows=${rows} ==="
  if [[ "${status}" != "ok" ]]; then
    echo "Stopping KV probe after first non-ok step (${status})."
    exit 0
  fi
done
echo "=== KV probe finished all steps ==="
INNER_EOF

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
  --env "HF_CACHE_DIR=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "TRANSFORMERS_CACHE=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "REPO_ROOT=${REPO_ROOT}" \
  --env "MODEL=${MODEL}" \
  --env "DTYPE=${DTYPE}" \
  --env "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
  --env "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  --env "MEASUREMENT_ITERATIONS=${MEASUREMENT_ITERATIONS}" \
  --env "KV_PROBE_STEPS=${KV_PROBE_STEPS}" \
  --env "KV_PROBE_STEP_TIMEOUT_SEC=${KV_PROBE_STEP_TIMEOUT_SEC}" \
  --env "KV_PROBE_HARDWARE_PREFIX=${KV_PROBE_HARDWARE_PREFIX}" \
  --env "PROBE_RESULTS=${PROBE_RESULTS}" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-local}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec bash -s' <<< "${INNER}"

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

echo "=== KV probe done. Results: ${PROBE_RESULTS} ==="
