#!/usr/bin/env bash
# A/B: one dense shot without layer pause vs one with pause-only (no DVFS).
# Records wall times and writes ab_compare.json under the pause arm tp1 dir.
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
HARDWARE_BASE="${HARDWARE_BASE:-V100_layer_pause_ab}"
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

VARIANT_TAG="fp16"
case "${DTYPE}" in
  float16) VARIANT_TAG="fp16" ;;
  bfloat16) VARIANT_TAG="bf16" ;;
  *) VARIANT_TAG="${DTYPE}" ;;
esac

HARDWARE_NOPAUSE="${HARDWARE_BASE}_nopause"
HARDWARE_PAUSE="${HARDWARE_BASE}_pause"
TP_NOPAUSE="${REPO_ROOT}/profiler/perf/${HARDWARE_NOPAUSE}/${MODEL}/${VARIANT_TAG}/tp1"
TP_PAUSE="${REPO_ROOT}/profiler/perf/${HARDWARE_PAUSE}/${MODEL}/${VARIANT_TAG}/tp1"
AB_JSON="${TP_PAUSE}/ab_compare.json"
LOG_DIR="${JOBS_ROOT}/logs/ab_${JOB_TAG}"
mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${TP_NOPAUSE}" "${TP_PAUSE}" \
  "${LOG_DIR}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

POLLER_PID=""
trap 'kill "${POLLER_PID:-}" 2>/dev/null || true' EXIT

profiler_slice() {
  local arm_label="$1"
  local hardware="$2"
  local log_file="$3"
  shift 3
  # Remaining args: env assignments for this arm (KEY=val ...)

  local -a env_args=()
  local kv key val
  for kv in "$@"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    env_args+=(--env "${key}=${val}")
  done

  echo ""
  echo "=== A/B arm: ${arm_label} (hardware=${hardware}) ==="
  local t0 t1 wall
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
    --env "DVFS_LAYER_PAUSE_MAX_SHOTS=${DVFS_LAYER_PAUSE_MAX_SHOTS}" \
    --env "PROFILER_MAX_SHOTS=${PROFILER_MAX_SHOTS}" \
    "${env_args[@]}" \
    "$VLLM_IMAGE" \
    bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec python3 -m profiler slice "'"${MODEL}"'" \
      --hardware "'"${hardware}"'" \
      --tp-refresh "'"${TP_DEGREES}"'" \
      --group dense \
      --dtype "'"${DTYPE}"'" \
      --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
      --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
      --attention-max-kv "'"${ATTENTION_MAX_KV}"'" \
      --measurement-iterations "'"${MEASUREMENT_ITERATIONS}"'" \
      --skip-skew \
      --force \
      '"${VERBOSITY}"'' 2>&1 | tee "${log_file}"

  t1="$(date +%s.%N)"
  wall="$(python3 -c "print(round(float('${t1}') - float('${t0}'), 3))")"
  echo "ARM ${arm_label} slice_wall_sec=${wall}" >&2
  echo "${wall}"
}

extract_dense_shot_sec() {
  local log_file="$1"
  python3 - <<PY
import re
from pathlib import Path
text = Path("${log_file}").read_text(encoding="utf-8", errors="replace")
# Rich progress: "1/1 0:00:07" on the dense line
m = re.search(r"TP=1\s+dense.*?(\d+)/(\d+)\s+(\d+):(\d{2}):(\d{2})", text)
if not m:
    print("")
else:
    _done, _total, h, mi, s = m.groups()
    print(round(int(h) * 3600 + int(mi) * 60 + int(s), 3))
PY
}

sum_marker_pause_sec() {
  local markers="$1"
  python3 - <<PY
import json
from pathlib import Path
p = Path("${markers}")
if not p.is_file():
    print("0")
else:
    total = 0.0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += float(rec.get("pause_end", 0)) - float(rec.get("pause_start", 0))
    print(round(total, 3))
PY
}

echo "=== Layer pause A/B (nopause vs pause-only) ==="
echo "MODEL=$MODEL  PAUSE_ONLY_DELAY_SEC=${PAUSE_ONLY_DELAY_SEC}"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset" >&2
fi

OLD_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
unset SLURM_JOB_ACCOUNT

# --- Arm A: no pause ---
LOG_NOPAUSE="${LOG_DIR}/nopause.log"
WALL_NOPAUSE="$(profiler_slice nopause "${HARDWARE_NOPAUSE}" "${LOG_NOPAUSE}" \
  DVFS_LAYER_PAUSE=0 \
  DVFS_HOST_POLLER=0 \
  DVFS_PAUSE_ONLY=0)"
SHOT_NOPAUSE="$(extract_dense_shot_sec "${LOG_NOPAUSE}")"

# --- Arm B: pause only ---
rm -f "${TP_PAUSE}/dvfs_markers.jsonl"
chmod +x "${HOST_POLLER}"
export DVFS_PAUSE_ONLY=1
export PAUSE_ONLY_DELAY_SEC
bash "${HOST_POLLER}" "${TP_PAUSE}" "${TP_PAUSE}/gpu_freq" &
POLLER_PID=$!
echo "HOST_POLLER_PID=${POLLER_PID} watch=${TP_PAUSE}"

LOG_PAUSE="${LOG_DIR}/pause.log"
WALL_PAUSE="$(profiler_slice pause "${HARDWARE_PAUSE}" "${LOG_PAUSE}" \
  DVFS_LAYER_PAUSE=1 \
  DVFS_HOST_POLLER=1 \
  DVFS_PAUSE_ONLY=1 \
  PAUSE_ONLY_DELAY_SEC="${PAUSE_ONLY_DELAY_SEC}")"
kill "${POLLER_PID}" 2>/dev/null || true
POLLER_PID=""

SHOT_PAUSE="$(extract_dense_shot_sec "${LOG_PAUSE}")"
MARKERS="${TP_PAUSE}/dvfs_markers.jsonl"
MARKER_PAUSE_SUM="$(sum_marker_pause_sec "${MARKERS}")"
MARKER_COUNT="$(wc -l < "${MARKERS}" 2>/dev/null || echo 0)"

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

DELTA_WALL="$(python3 -c "wn=float('''${WALL_NOPAUSE}'''); wp=float('''${WALL_PAUSE}'''); print(round(wp-wn, 3))")"
DELTA_SHOT=""
if [[ -n "${SHOT_NOPAUSE}" && -n "${SHOT_PAUSE}" ]]; then
  DELTA_SHOT="$(python3 -c "print(round(float('''${SHOT_PAUSE}''') - float('''${SHOT_NOPAUSE}'''), 3))")"
fi

python3 - <<PY
import json
from pathlib import Path

def fnum(s):
    s = (s or "").strip()
    return float(s) if s else None

record = {
    "job_id": "${JOB_TAG}",
    "model": "${MODEL}",
    "pause_only_delay_sec": float("${PAUSE_ONLY_DELAY_SEC}"),
    "measurement_iterations": int("${MEASUREMENT_ITERATIONS}"),
    "nopause": {
        "hardware": "${HARDWARE_NOPAUSE}",
        "slice_wall_sec": fnum("${WALL_NOPAUSE}"),
        "dense_category_sec": fnum("${SHOT_NOPAUSE}"),
        "log": "${LOG_NOPAUSE}",
    },
    "pause": {
        "hardware": "${HARDWARE_PAUSE}",
        "slice_wall_sec": fnum("${WALL_PAUSE}"),
        "dense_category_sec": fnum("${SHOT_PAUSE}"),
        "marker_pause_sum_sec": fnum("${MARKER_PAUSE_SUM}"),
        "marker_count": int("${MARKER_COUNT}"),
        "log": "${LOG_PAUSE}",
        "markers": "${MARKERS}",
    },
    "delta": {
        "slice_wall_sec": fnum("${DELTA_WALL}"),
        "dense_category_sec": fnum("${DELTA_SHOT}") if "${DELTA_SHOT}" else None,
    },
}
out = Path("${AB_JSON}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY

echo ""
echo "=== A/B summary ==="
echo "nopause  slice_wall_sec=${WALL_NOPAUSE}  dense_category_sec=${SHOT_NOPAUSE:-n/a}"
echo "pause    slice_wall_sec=${WALL_PAUSE}  dense_category_sec=${SHOT_PAUSE:-n/a}"
echo "delta    slice_wall_sec=${DELTA_WALL}  dense_category_sec=${DELTA_SHOT:-n/a}"
echo "markers  count=${MARKER_COUNT}  summed_pause_sec=${MARKER_PAUSE_SUM}"
echo "wrote ${AB_JSON}"
