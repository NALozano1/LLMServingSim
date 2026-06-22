#!/usr/bin/env bash
# Campaign-aware V100 bench runner: real vLLM TTFT/TPOT + optional mid-run DVFS.
#
# Required env (set by submit script):
#   CAMPAIGN_DIR, RUN_ID, RUN_SPEC_JSON
# Optional:
#   BENCH_FREQ_SCHEDULE=700,1300   two freqs, switch halfway through bench
#   BENCH_FREQ_SWITCH_SEC          override auto switch delay
#   GPU_FREQ_MHZ                   single fixed freq (when no schedule)
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
BENCH_JOBS="${REPO_ROOT}/bench/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

RUN_ID="${RUN_ID:?RUN_ID}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:?CAMPAIGN_DIR}"
RUN_SPEC_JSON="${RUN_SPEC_JSON:?RUN_SPEC_JSON}"
RUN_DIR="${CAMPAIGN_DIR}/runs/${RUN_ID}"

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
NUM_REQS="${NUM_REQS:-100}"
SPS="${SPS:-10}"
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-1.0}"
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-}"
BENCH_FREQ_SCHEDULE="${BENCH_FREQ_SCHEDULE:-}"
BENCH_FREQ_SWITCH_SEC="${BENCH_FREQ_SWITCH_SEC:-}"

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
HARDWARE="${HARDWARE:-V100}"
if [[ -n "${BENCH_FREQ_SCHEDULE}" ]]; then
  IFS=',' read -r FREQ_A FREQ_B _ <<< "${BENCH_FREQ_SCHEDULE},,"
  HARDWARE="${HARDWARE:-V100_dvfs_${FREQ_A}_${FREQ_B}}"
  GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-${FREQ_A}}"
elif [[ -n "${GPU_FREQ_MHZ}" ]]; then
  HARDWARE="${HARDWARE:-V100_${GPU_FREQ_MHZ}MHz}"
fi

OUT_DIR="${OUT_DIR:-${RUN_DIR}/bench}"
DATASET="${DATASET:-${CAMPAIGN_DIR}/shared/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}.jsonl}"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${RUN_DIR}" \
  "${CAMPAIGN_DIR}/shared" \
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

if [[ -z "${BENCH_FREQ_SWITCH_SEC}" && -n "${BENCH_FREQ_SCHEDULE}" ]]; then
  # Engine boot ~30s + half the request arrival span + 60s decode cushion.
  BENCH_FREQ_SWITCH_SEC=$((30 + (NUM_REQS / SPS) / 2 + 60))
fi

echo "=== Campaign bench ${RUN_ID} ==="
echo "MODEL=${MODEL}  HARDWARE=${HARDWARE}  GPU_FREQ_MHZ=${GPU_FREQ_MHZ:-default}"
echo "BENCH_FREQ_SCHEDULE=${BENCH_FREQ_SCHEDULE:-<none>}"
echo "BENCH_FREQ_SWITCH_SEC=${BENCH_FREQ_SWITCH_SEC:-<n/a>}"
echo "OUT_DIR=${OUT_DIR}"
echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"

nvidia-smi -L || true

SWITCHER_PID=""
if [[ -n "${BENCH_FREQ_SCHEDULE}" ]]; then
  IFS=',' read -r FREQ_A FREQ_B _ <<< "${BENCH_FREQ_SCHEDULE},,"
  gpu_freq_lock_apply "${FREQ_META_DIR}" "${FREQ_A}"
  chmod +x "${BENCH_JOBS}/bench_freq_switcher.sh"
  "${BENCH_JOBS}/bench_freq_switcher.sh" \
    "${OUT_DIR}" "${FREQ_A}" "${FREQ_B}" "${BENCH_FREQ_SWITCH_SEC}" &
  SWITCHER_PID=$!
elif [[ -n "${GPU_FREQ_MHZ}" ]]; then
  gpu_freq_lock_apply "${FREQ_META_DIR}" "${GPU_FREQ_MHZ}"
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

date +%s > "${OUT_DIR}/bench_running"
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
rm -f "${OUT_DIR}/bench_running"
INNER

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

if [[ -n "${SWITCHER_PID}" ]]; then
  wait "${SWITCHER_PID}" || true
fi

python3 "${BENCH_JOBS}/extract_real_latency.py" "${OUT_DIR}" \
  -o "${RUN_DIR}/real_latency.json"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path("${RUN_DIR}")
out = Path("${OUT_DIR}")
spec = json.loads(Path("${RUN_SPEC_JSON}").read_text())
node_meta = {
    "run_id": "${RUN_ID}",
    "campaign_dir": "${CAMPAIGN_DIR}",
    "slurm_job_id": "${SLURM_JOB_ID:-local}",
    "hostname": "${NODE}",
    "slurm_nodelist": "${NODELIST}",
    "hardware": "${HARDWARE}",
    "gpu_freq_mhz": int("${GPU_FREQ_MHZ}") if "${GPU_FREQ_MHZ}".isdigit() else None,
    "bench_freq_schedule": "${BENCH_FREQ_SCHEDULE}" or None,
    "bench_freq_switch_sec": int("${BENCH_FREQ_SWITCH_SEC}") if "${BENCH_FREQ_SWITCH_SEC}".isdigit() else None,
    "model": "${MODEL}",
    "dataset": "${DATASET}",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
(out / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")
(run_dir / "run_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")

real = json.loads((run_dir / "real_latency.json").read_text())
summary = {
    "run_id": "${RUN_ID}",
    "status": "completed",
    "real_ttft_mean_ms": real["ttft"].get("mean_ms"),
    "real_tpot_mean_ms": real["tpot"].get("mean_ms"),
    "real_e2e_mean_ms": real["e2e_latency"].get("mean_ms"),
    "bench_dir": str(out),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

python3 "${BENCH_JOBS}/build_sim_replication.py" \
  --repo "${REPO_ROOT}" \
  --run-id "${RUN_ID}" \
  --spec "${RUN_SPEC_JSON}" \
  --bench-out "${OUT_DIR}" \
  -o "${RUN_DIR}/sim_replication.json"

# Patch dataset path into sim_replication (shared workload).
python3 - <<PY
import json
from pathlib import Path
p = Path("${RUN_DIR}/sim_replication.json")
b = json.loads(p.read_text())
b["dataset"] = "${DATASET}"
b["workload"]["dataset"] = "${DATASET}"
p.write_text(json.dumps(b, indent=2) + "\n")
PY

echo "=== Campaign run ${RUN_ID} done ==="
echo "  REAL latency: ${RUN_DIR}/real_latency.json"
echo "  Sim match:    ${RUN_DIR}/sim_replication.json"
