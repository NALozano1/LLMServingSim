#!/usr/bin/env bash
# Run LLMServingSim profiler on an ARC GPU node (intended for 1× V100).
#
# Writable caches: /data/engs-glass/engs2950/.llmsim/<jobid>/ (short paths —
# vLLM ZMQ IPC sockets max out at 107 chars). Uses --cleanenv so compute-node
# $HOME quota and stray host env vars do not leak into the container.
#
# Optional GPU_FREQ_MHZ locks graphics clocks via the engs2950 DVFS path:
#   shared/scripts/gpu_freq_lock_lib.sh → gpu_freq_lock.py → site nvidia-smi-clocks
# (see .cursor/rules/arc-node-gpu-clocks.mdc). Restores on exit.
# When set, HARDWARE defaults to V100_<MHz> unless HARDWARE is already exported.
#
# Clock-lock verification (guards against importing mislabeled profiles):
#   - Pre-flight read-back: after applying the lock, asserts nvidia-smi reports
#     within GPU_FREQ_TOLERANCE_MHZ of the target (catches a lock that never took).
#   - Post-flight audit: runs audit_gpu_clocks.py over the captured gpu_power
#     samples (catches mid-run throttling, e.g. heavy MoE kernels losing the lock).
#   GPU_FREQ_TOLERANCE_MHZ (default 100) — allowed drift around the target.
#   GPU_FREQ_VERIFY_STRICT  (default 1)  — 1 aborts on mismatch, 0 warns only.
#
# Usage:
#   export HF_TOKEN=...
#   bash profiler/jobs/run_arc_v100_profile.sh
#
#   GPU_FREQ_MHZ=1300 HARDWARE=V100_1300MHz bash profiler/jobs/run_arc_v100_profile.sh

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
if [[ -n "${GPU_FREQ_MHZ:-}" ]]; then
  HARDWARE="${HARDWARE:-V100_${GPU_FREQ_MHZ}MHz}"
else
  HARDWARE="${HARDWARE:-V100}"
fi
TP_DEGREES="${TP_DEGREES:-1,4}"
DTYPE="${DTYPE:-float16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-8192}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
VERBOSITY="${VERBOSITY:-"--verbose"}"

# Clock-lock verification knobs (always in scope; used pre- and post-flight).
GPU_FREQ_TOLERANCE_MHZ="${GPU_FREQ_TOLERANCE_MHZ:-100}"
GPU_FREQ_VERIFY_STRICT="${GPU_FREQ_VERIFY_STRICT:-1}"

VARIANT_TAG="fp16"
case "${DTYPE}" in
  float16) VARIANT_TAG="fp16" ;;
  bfloat16) VARIANT_TAG="bf16" ;;
  float32) VARIANT_TAG="fp32" ;;
  fp8) VARIANT_TAG="fp8" ;;
  *) VARIANT_TAG="${DTYPE}" ;;
esac
PERF_VARIANT_ROOT="${REPO_ROOT}/profiler/perf/${HARDWARE}/${MODEL}/${VARIANT_TAG}"
FREQ_META_DIR="${PERF_VARIANT_ROOT}/gpu_freq"

SKIP_SKEW_FLAG=()
if [[ "${FULL_PROFILE:-0}" != "1" && -z "${SKIP_SKEW:-}" ]]; then
  SKIP_SKEW_FLAG=(--skip-skew)
elif [[ -n "${SKIP_SKEW:-}" && "${SKIP_SKEW}" != "0" ]]; then
  SKIP_SKEW_FLAG=(--skip-skew)
fi

VERBOSITY_FLAG=()
case "${VERBOSITY}" in
  --verbose|--silent) VERBOSITY_FLAG=("${VERBOSITY}") ;;
esac

mkdir -p \
  "${SCRATCH}/t" \
  "${SCRATCH}/h" \
  "${SCRATCH}/v" \
  "${SCRATCH}/triton" \
  "${SCRATCH}/torch" \
  "${SCRATCH}/pip" \
  "${FREQ_META_DIR}" \
  "${HF_CACHE_ROOT}/hub" \
  "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"
trap 'gpu_freq_lock_trap_restore "${FREQ_META_DIR}"' EXIT

echo "=== LLMServingSim profiler (ARC / engs-glass) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL=$MODEL  HARDWARE=$HARDWARE  TP_DEGREES=$TP_DEGREES  DTYPE=$DTYPE"
echo "GPU_FREQ_MHZ=${GPU_FREQ_MHZ:-<default/boost>}"
echo "FULL_PROFILE=${FULL_PROFILE:-0}  SKIP_SKEW=${SKIP_SKEW:-<unset>}  VERBOSITY=${VERBOSITY}"
echo "VLLM_IMAGE=$VLLM_IMAGE"
echo "HF_CACHE_ROOT=$HF_CACHE_ROOT"
echo "SCRATCH=$SCRATCH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

if [[ -n "${GPU_FREQ_MHZ:-}" ]]; then
  gpu_freq_lock_apply "${FREQ_META_DIR}" "${GPU_FREQ_MHZ}"

  # Pre-flight read-back: confirm the lock took before spending an hour
  # profiling at the wrong clock. -lgc pins the clock even at idle, so a
  # large deviation here means the lock never applied.
  _gpu_idx="${GPU_VERIFY_INDEX:-0}"
  _measured_gr="$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits \
      -i "${_gpu_idx}" 2>/dev/null | head -1 | tr -dc '0-9')"
  if [[ -n "${_measured_gr}" ]]; then
    _delta=$(( _measured_gr - GPU_FREQ_MHZ )); _absdelta="${_delta#-}"
    echo "Clock read-back: target=${GPU_FREQ_MHZ}MHz measured=${_measured_gr}MHz" \
         "delta=${_delta}MHz (tol=±${GPU_FREQ_TOLERANCE_MHZ})"
    if (( _absdelta > GPU_FREQ_TOLERANCE_MHZ )); then
      echo "ERROR: GPU clock lock did not hold (measured ${_measured_gr}MHz" \
           "vs target ${GPU_FREQ_MHZ}MHz, ±${GPU_FREQ_TOLERANCE_MHZ} tol)." >&2
      if [[ "${GPU_FREQ_VERIFY_STRICT}" == "1" ]]; then
        echo "Aborting before profiling (set GPU_FREQ_VERIFY_STRICT=0 to override)." >&2
        exit 3
      fi
      echo "WARN: continuing despite clock mismatch (GPU_FREQ_VERIFY_STRICT=0)." >&2
    fi
  else
    echo "WARN: could not read back GPU clock via nvidia-smi;" \
         "skipping pre-flight verification." >&2
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset — gated models may fail tokenizer/config download." >&2
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
  --env "HF_CACHE_DIR=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "TRANSFORMERS_CACHE=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec ./profiler/profile_cli.sh \
    --model "'"${MODEL}"'" \
    --hardware "'"${HARDWARE}"'" \
    --tp "'"${TP_DEGREES}"'" \
    --dtype "'"${DTYPE}"'" \
    --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
    --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
    --attention-max-kv "'"${ATTENTION_MAX_KV}"'" \
    --measurement-iterations "'"${MEASUREMENT_ITERATIONS}"'" \
    '"${SKIP_SKEW_FLAG[*]}"' \
    '"${VERBOSITY_FLAG[*]}"''

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

# Post-flight audit: the pre-flight read-back only sees the idle/start clock.
# This inspects the per-shot gpu_power captures and flags any run whose measured
# clocks drifted out of band mid-profiling (e.g. heavy MoE kernels throttling
# below the lock) so mislabeled latency tables are never silently imported.
AUDIT="${JOBS_ROOT}/audit_gpu_clocks.py"
if [[ -f "${AUDIT}" ]] && command -v python3 >/dev/null 2>&1; then
  echo "=== Post-flight GPU clock audit (${HARDWARE}) ==="
  if python3 "${AUDIT}" --tolerance "${GPU_FREQ_TOLERANCE_MHZ}" "${PERF_VARIANT_ROOT}"; then
    echo "Clock audit passed."
  else
    echo "ERROR: post-flight clock audit FLAGGED captured samples for ${HARDWARE}." >&2
    echo "       Profiles under ${PERF_VARIANT_ROOT} are likely mislabeled — re-profile." >&2
    if [[ "${GPU_FREQ_VERIFY_STRICT}" == "1" && -n "${GPU_FREQ_MHZ:-}" ]]; then
      exit 4
    fi
  fi
else
  echo "WARN: ${AUDIT} or python3 unavailable; skipping post-flight clock audit." >&2
fi

echo "=== Done. Outputs under: ${PERF_VARIANT_ROOT}/ ==="
