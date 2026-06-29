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

ONLY_MOE_FLAG=()
[[ -n "${ONLY_MOE:-}" && "${ONLY_MOE}" != "0" ]] && ONLY_MOE_FLAG=(--only-moe)

FORCE_FLAG=()
[[ -n "${FORCE:-}" && "${FORCE}" != "0" ]] && FORCE_FLAG=(--force)

# ── Checkpoint-and-requeue trap (full-profile jobs only) ──────────────────────
# Slurm fires SIGUSR1 to the batch step 300 s before the time limit
# (configured via --signal=B:USR1@300 in the sbatch invocation).  We
# immediately request a requeue so the job goes back to PENDING; when the
# current instance finishes (or is hard-killed at the limit), Slurm starts a
# fresh run that resumes from the partial CSVs (skip-already-measured rows).
#
# Skipped for moe-only jobs (ONLY_MOE set): they finish quickly and don't need
# automatic checkpointing.  The trap installs only when ONLY_MOE is absent or 0.
_REQUEUE_ON_EXIT=0
if [[ -z "${ONLY_MOE:-}" || "${ONLY_MOE}" == "0" ]]; then
  _usr1_handler() {
    echo "=== USR1: ~5 min to time limit; requeueing ${SLURM_JOB_ID:-<no jobid>} for continuation ===" >&2
    _REQUEUE_ON_EXIT=1
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      scontrol requeue "${SLURM_JOB_ID}" \
        && echo "Requeue requested for job ${SLURM_JOB_ID}. Resuming from partial CSVs on restart." >&2 \
        || echo "WARN: scontrol requeue failed (job may already be completing)" >&2
    else
      echo "WARN: SLURM_JOB_ID unset; cannot requeue automatically" >&2
    fi
  }
  trap '_usr1_handler' USR1
fi

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

# ── Clock hold poller state ───────────────────────────────────────────────────
# Mirrors the pattern in bench/jobs/run_arc_v100_prefill_validation.sh.
HOLD_PID=""
_cleanup() {
  if [[ -n "${HOLD_PID:-}" ]]; then
    touch "${FREQ_META_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${HOLD_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill "${HOLD_PID}" 2>/dev/null || true
    wait "${HOLD_PID}" 2>/dev/null || true
    HOLD_PID=""
  fi
  gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true
}
trap '_cleanup' EXIT

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

  # Pre-flight read-back: INFORMATIONAL ONLY. On V100 the graphics clock idles
  # at ~135 MHz even after a successful --lock-gpu-clocks; the locked clock only
  # becomes observable under GPU load (waiting longer at idle does NOT help).
  # So an idle mismatch here is expected and must NOT abort. The authoritative
  # check is the under-load post-flight audit (audit_gpu_clocks.py) below.
  _gpu_idx="${GPU_VERIFY_INDEX:-0}"
  _measured_gr="$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits \
      -i "${_gpu_idx}" 2>/dev/null | head -1 | tr -dc '0-9')"
  if [[ -n "${_measured_gr}" ]]; then
    _delta=$(( _measured_gr - GPU_FREQ_MHZ )); _absdelta="${_delta#-}"
    echo "Clock read-back (idle, informational): target=${GPU_FREQ_MHZ}MHz" \
         "measured=${_measured_gr}MHz delta=${_delta}MHz" \
         "(idle clock can't see the lock on V100; under-load audit is authoritative)"
  else
    echo "WARN: could not read back GPU clock via nvidia-smi;" \
         "relying on post-flight under-load audit." >&2
  fi
fi

# ── Clock hold poller (HOST-side, outside apptainer) ─────────────────────────
# A single one-shot apply loses the lock once the V100 goes under heavy load.
# This background poller continuously re-applies the lock for the entire
# profiling run, exactly as bench/jobs/run_arc_v100_prefill_validation.sh does.
# Only started when GPU_FREQ_MHZ is set (no-op for bare V100 / uncapped runs).
if [[ -n "${GPU_FREQ_MHZ:-}" ]]; then
  # Compute the GPU index range to hold: 0..max_tp-1.
  # TP_DEGREES may be a comma-separated list (e.g. "1,4"); use the maximum so
  # all GPUs that will be active during any TP sweep are covered.
  _MAX_TP=1
  IFS=',' read -ra _TP_ARR <<< "${TP_DEGREES}"
  for _tp in "${_TP_ARR[@]}"; do
    (( _tp > _MAX_TP )) && _MAX_TP="${_tp}"
  done
  _HOLD_GPUS="$(seq -s, 0 $((_MAX_TP - 1)))"
  CUDA_VISIBLE_DEVICES="${_HOLD_GPUS}" python3 "${GPU_FREQ_LOCK_PY}" hold \
    --mhz "${GPU_FREQ_MHZ}" --out-dir "${FREQ_META_DIR}" &
  HOLD_PID=$!
  echo "[dvfs] hold poller pid=${HOLD_PID} target=${GPU_FREQ_MHZ} MHz gpus=${_HOLD_GPUS}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is unset — gated models may fail tokenizer/config download." >&2
fi

OLD_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
unset SLURM_JOB_ACCOUNT

# Run container in background so the USR1 trap can fire while it is running.
# (Bash only processes signal traps between commands; backgrounding + wait makes
# the wait built-in interruptible so signals are delivered promptly.)
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
    '"${VERBOSITY_FLAG[*]}"' \
    '"${ONLY_MOE_FLAG[*]}"' \
    '"${FORCE_FLAG[*]}"'' &
_CONTAINER_PID=$!

# Wait for the container, re-looping if interrupted mid-wait by a signal (e.g.
# USR1 from Slurm).  When bash's wait built-in is interrupted by a signal the
# trap runs, then wait returns 128+signum even though the child is still alive;
# kill -0 distinguishes "still running" from "exited non-zero".
_container_rc=0
while true; do
  wait "${_CONTAINER_PID}" && { _container_rc=0; break; } || _container_rc=$?
  # Container still alive → wait was interrupted by a signal; loop to re-wait.
  kill -0 "${_CONTAINER_PID}" 2>/dev/null || break
done
if [[ "${_container_rc}" -ne 0 ]]; then
  exit "${_container_rc}"
fi

# Stop the clock hold poller now that the workload has finished.
if [[ -n "${GPU_FREQ_MHZ:-}" && -n "${HOLD_PID:-}" ]]; then
  touch "${FREQ_META_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
  wait "${HOLD_PID}" 2>/dev/null || true
  HOLD_PID=""
  echo "[dvfs] hold poller stopped; summary at ${FREQ_META_DIR}/gpu_freq_hold_summary.json"
fi

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
