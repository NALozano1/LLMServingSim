#!/usr/bin/env bash
# Real vLLM throughput benchmark on ARC V100 (tp1) via python -m bench run.
#
# Default model: microsoft/Phi-mini-MoE-instruct (fits easily on 32GB V100).
# Smaller MoE: microsoft/Phi-tiny-MoE-instruct (3.8B total / 1.1B active).
# Qwen1.5-MoE needs the qwen15-v100-tight preset (auto-applied); see
# profiler/jobs/v100_bench_presets.sh.
# Generates a ShareGPT JSONL workload, then replays it through vLLM.
#
# Outputs: bench/results/V100/<safe_model>/tp1/<jobid>/
#   meta.json, requests.jsonl, timeseries.csv (prompt/gen tok/s per tick)
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

MODEL="${MODEL:-microsoft/Phi-mini-MoE-instruct}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
NUM_REQS="${NUM_REQS:-100}"
SPS="${SPS:-10}"
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-1.0}"
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-}"
# Clock-lock verification (mirrors run_arc_v100_profile.sh): pre-flight read-back
# confirms the lock took before benching, post-flight audit scans the captured
# gpu_power samples for mid-run throttling. STRICT=1 aborts on mismatch, 0 warns.
GPU_FREQ_TOLERANCE_MHZ="${GPU_FREQ_TOLERANCE_MHZ:-100}"
GPU_FREQ_VERIFY_STRICT="${GPU_FREQ_VERIFY_STRICT:-1}"

# shellcheck source=/dev/null
source "${JOBS_ROOT}/v100_bench_presets.sh"
v100_bench_apply_preset "${MODEL}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
SHAREGPT_FIX_LEN="${SHAREGPT_FIX_LEN:-0}"
FIX_INPUT_LENGTH="${FIX_INPUT_LENGTH:-128}"
FIX_OUTPUT_LENGTH="${FIX_OUTPUT_LENGTH:-512}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/:' '__')"
if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  HARDWARE="${HARDWARE:-V100_${GPU_FREQ_MHZ}MHz}"
else
  HARDWARE="${HARDWARE:-V100}"
fi
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/bench/results/${HARDWARE}/${SAFE_MODEL}/tp${TP_SIZE}/${JOB_TAG}}"
# Repo bind mount is writable on ARC compute nodes; .llmsim/shared_workloads is not.
if [[ "${SHAREGPT_FIX_LEN}" == "1" ]]; then
  _ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
else
  _ds_tag="ml${MAX_MODEL_LEN}"
fi
DATASET="${DATASET:-${REPO_ROOT}/workloads/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl}"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"
trap 'gpu_freq_lock_trap_restore "${FREQ_META_DIR}" 2>/dev/null || true' EXIT

NODE="$(hostname -s 2>/dev/null || hostname)"
NODELIST="${SLURM_NODELIST:-}"
if [[ -n "${NODELIST}" ]] && command -v scontrol >/dev/null 2>&1; then
  NODELIST="$(scontrol show hostnames "${NODELIST}" | paste -sd, -)"
fi

echo "=== V100 throughput bench (vLLM) ==="
echo "MODEL=${MODEL}  TP=${TP_SIZE}  DTYPE=${DTYPE}"
echo "HARDWARE=${HARDWARE}  GPU_FREQ_MHZ=${GPU_FREQ_MHZ:-<default>}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}  MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}  SHAREGPT_FIX_LEN=${SHAREGPT_FIX_LEN}"
echo "DATASET=${DATASET}"
echo "OUT_DIR=${OUT_DIR}"
echo "NODE=${NODE}  NODELIST=${NODELIST}"
nvidia-smi -L || true

if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  gpu_freq_lock_apply "${FREQ_META_DIR}" "${GPU_FREQ_MHZ}"

  # Pre-flight read-back: INFORMATIONAL ONLY. On V100 the graphics clock idles
  # at ~135 MHz even after a successful --lock-gpu-clocks; the locked clock only
  # becomes observable under GPU load. So an idle mismatch here is expected and
  # must NOT abort. The authoritative check is the under-load post-flight audit
  # (audit_gpu_clocks.py over captured gpu_power samples) below.
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
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_CACHE_DIR=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "TRANSFORMERS_CACHE=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets 2>/dev/null || true; exec bash -s' <<INNER
set -euo pipefail
cd "${REPO_ROOT}"

if [[ ! -s "${DATASET}" ]]; then
  echo "Generating ShareGPT dataset (${NUM_REQS} reqs @ ${SPS} sps) ..."
  if [[ "${SHAREGPT_FIX_LEN}" == "1" ]]; then
    python3 -m workloads.generators sharegpt \
      --model "${MODEL}" \
      --num-reqs "${NUM_REQS}" \
      --sps "${SPS}" \
      --seed "${SEED}" \
      --output "${DATASET}" \
      --fix-len \
      --fix-input-length "${FIX_INPUT_LENGTH}" \
      --fix-output-length "${FIX_OUTPUT_LENGTH}"
  else
    python3 -m workloads.generators sharegpt \
      --model "${MODEL}" \
      --num-reqs "${NUM_REQS}" \
      --sps "${SPS}" \
      --seed "${SEED}" \
      --output "${DATASET}"
  fi
fi

echo "Running bench (strict replay) ..."
python3 -m bench run \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --output-dir "${OUT_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tick-seconds "${TICK_SECONDS}" \
  --num-reqs "${NUM_REQS}" \
  --log-level INFO
INNER

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

out = Path("${OUT_DIR}")
freq = "${GPU_FREQ_MHZ}"
node_meta = {
    "hostname": "${NODE}",
    "slurm_nodelist": "${NODELIST}",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "hardware": "${HARDWARE}",
    "gpu_freq_mhz": int(freq) if freq.isdigit() else None,
    "model": "${MODEL}",
    "tp_size": int("${TP_SIZE}"),
    "dtype": "${DTYPE}",
    "dataset": "${DATASET}",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
if (out / "meta.json").exists():
    bench_meta = json.loads((out / "meta.json").read_text())
    node_meta["bench"] = bench_meta
(out / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")
print(f"node_meta: {out / 'node_meta.json'}")
PY

# Post-flight audit: the pre-flight read-back only sees the idle/start clock.
# This inspects the captured gpu_power JSONL and flags any run whose measured
# clocks drifted out of band mid-bench (e.g. heavy MoE kernels throttling below
# the lock) so mislabeled latency/energy ground truth is never trusted. The
# audit recovers the V100_<MHz> tag from the OUT_DIR ancestry.
if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  AUDIT="${JOBS_ROOT}/audit_gpu_clocks.py"
  if [[ -f "${AUDIT}" ]] && command -v python3 >/dev/null 2>&1; then
    echo "=== Post-flight GPU clock audit (${HARDWARE}) ==="
    if python3 "${AUDIT}" --tolerance "${GPU_FREQ_TOLERANCE_MHZ}" "${OUT_DIR}"; then
      echo "Clock audit passed."
    else
      echo "ERROR: post-flight clock audit FLAGGED captured samples for ${HARDWARE}." >&2
      echo "       Bench under ${OUT_DIR} is likely mislabeled — re-run." >&2
      if [[ "${GPU_FREQ_VERIFY_STRICT}" == "1" ]]; then
        exit 4
      fi
    fi
  else
    echo "WARN: ${AUDIT} or python3 unavailable; skipping post-flight clock audit." >&2
  fi
fi

echo "=== Done. Throughput outputs: ${OUT_DIR} ==="
echo "  timeseries.csv — prompt_throughput / gen_throughput (tok/s per tick)"
