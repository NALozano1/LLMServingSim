#!/usr/bin/env bash
# A/B: one dense shot without layer pause vs one with pause-only (no DVFS).
# Single vLLM engine boot; timing starts after in-shot warmup (measured_sec).
#
#   export HF_TOKEN=...
#   bash profiler/jobs/run_arc_v100_layer_pause_ab.sh
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
HOST_POLLER="${JOBS_ROOT}/dvfs_barrier_host_poller.sh"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
HARDWARE="${HARDWARE:-V100_layer_pause_ab}"
TP_DEGREES="${TP_DEGREES:-1}"
DTYPE="${DTYPE:-float16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
VERBOSITY="${VERBOSITY:-"--verbose"}"
PAUSE_ONLY_DELAY_SEC="${PAUSE_ONLY_DELAY_SEC:-0.05}"

export DVFS_LAYER_PAUSE_MAX_SHOTS="${DVFS_LAYER_PAUSE_MAX_SHOTS:-1}"
export PROFILER_MAX_SHOTS="${PROFILER_MAX_SHOTS:-${DVFS_LAYER_PAUSE_MAX_SHOTS}}"
export ENGS2950_ROOT="${ENGS_GLASS}"
export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export DVFS_PAUSE_ONLY=1
export PAUSE_ONLY_DELAY_SEC

VARIANT_TAG="fp16"
case "${DTYPE}" in
  float16) VARIANT_TAG="fp16" ;;
  bfloat16) VARIANT_TAG="bf16" ;;
  *) VARIANT_TAG="${DTYPE}" ;;
esac

TP_ROOT="${REPO_ROOT}/profiler/perf/${HARDWARE}/${MODEL}/${VARIANT_TAG}/tp1"
AB_JSON="${TP_ROOT}/ab_compare.json"
LOG_DIR="${JOBS_ROOT}/logs/ab_${JOB_TAG}"
LOG_FILE="${LOG_DIR}/pause_ab.log"
mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${TP_ROOT}" \
  "${LOG_DIR}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

POLLER_PID=""
trap 'kill "${POLLER_PID:-}" 2>/dev/null || true' EXIT

echo "=== Layer pause A/B (single engine, post-warmup timing) ==="
echo "MODEL=$MODEL  HARDWARE=$HARDWARE  PAUSE_ONLY_DELAY_SEC=${PAUSE_ONLY_DELAY_SEC}"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset" >&2
fi

OLD_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
unset SLURM_JOB_ACCOUNT

rm -f "${TP_ROOT}/dvfs_markers.jsonl" "${TP_ROOT}/shot_timings.jsonl"
chmod +x "${HOST_POLLER}"
bash "${HOST_POLLER}" "${TP_ROOT}" "${TP_ROOT}/gpu_freq" &
POLLER_PID=$!
echo "HOST_POLLER_PID=${POLLER_PID} watch=${TP_ROOT}" >&2

t0="$(date +%s.%N)"

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
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "DVFS_PAUSE_ONLY=1" \
  --env "PAUSE_ONLY_DELAY_SEC=${PAUSE_ONLY_DELAY_SEC}" \
  --env "PROFILER_MAX_SHOTS=${PROFILER_MAX_SHOTS}" \
  --env "SLURM_JOB_ID=${JOB_TAG}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec python3 -m profiler pause-ab "'"${MODEL}"'" \
    --hardware "'"${HARDWARE}"'" \
    --tp "'"${TP_DEGREES}"'" \
    --tp-refresh "'"${TP_DEGREES}"'" \
    --group dense \
    --dtype "'"${DTYPE}"'" \
    --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
    --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
    --attention-max-kv "'"${ATTENTION_MAX_KV}"'" \
    --measurement-iterations "'"${MEASUREMENT_ITERATIONS}"'" \
    --skip-skew \
    --force \
    '"${VERBOSITY}"'' >"${LOG_FILE}" 2>&1

t1="$(date +%s.%N)"
wall="$(python3 -c "print(round(float('${t1}') - float('${t0}'), 3))")"
echo "${wall}" > "${LOG_FILE}.wall_sec"

kill "${POLLER_PID}" 2>/dev/null || true
POLLER_PID=""

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

if [[ ! -f "${AB_JSON}" ]]; then
  echo "ERROR: missing ${AB_JSON}" >&2
  tail -n 80 "${LOG_FILE}" >&2 || true
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path

ab = json.loads(Path("${AB_JSON}").read_text(encoding="utf-8"))
ab["container_wall_sec"] = float("${wall}")
ab["log"] = "${LOG_FILE}"
Path("${AB_JSON}").write_text(json.dumps(ab, indent=2) + "\n", encoding="utf-8")
print(json.dumps(ab, indent=2))
PY

echo ""
echo "=== A/B summary (post-warmup measured_sec) ==="
python3 - <<PY
import json
from pathlib import Path
ab = json.loads(Path("${AB_JSON}").read_text())
np = ab["nopause"]
pa = ab["pause"]
d = ab["delta"]
print(f"engine_boot_sec={ab.get('engine_boot_sec')}")
print(f"nopause  measured_sec={np.get('measured_sec')}  warmup_sec={np.get('warmup_sec')}")
print(f"pause    measured_sec={pa.get('measured_sec')}  warmup_sec={pa.get('warmup_sec')}")
print(f"delta    measured_sec={d.get('measured_sec')}")
print(f"markers  count={pa.get('marker_count')}  summed_pause_sec={pa.get('marker_pause_sum_sec')}")
print(f"container_wall_sec={ab.get('container_wall_sec')}")
PY
echo "wrote ${AB_JSON}"
