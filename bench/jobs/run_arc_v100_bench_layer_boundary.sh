#!/usr/bin/env bash
# Real vLLM bench with layer-boundary DVFS (host poller outside Apptainer).
#
#   export HF_TOKEN=...
#   bash bench/jobs/run_arc_v100_bench_layer_boundary.sh
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

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
NUM_REQS="${NUM_REQS:-1}"
SPS="${SPS:-100}"
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-0.5}"

export VLLM_LAYER_PAUSE=1
export VLLM_HOST_POLLER=1
export VLLM_EXTERNAL_HOST_POLLER=1
export VLLM_BENCH_PREFILL_ONLY="${VLLM_BENCH_PREFILL_ONLY:-1}"
export VLLM_BENCH_GPU_POWER=1
export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export DVFS_FREQ_SCHEDULE="${DVFS_FREQ_SCHEDULE:-700,1400}"
export GPU_FREQ_SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-0}"
export GPU_FREQ_STABLE_TOL_MHZ="${GPU_FREQ_STABLE_TOL_MHZ:-15}"
export GPU_FREQ_STABLE_READS="${GPU_FREQ_STABLE_READS:-2}"
export GPU_FREQ_STABLE_TIMEOUT_SEC="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}"
export PROFILER_GPU_POWER="${PROFILER_GPU_POWER:-1}"
export PROFILER_GPU_POWER_INTERVAL_MS="${PROFILER_GPU_POWER_INTERVAL_MS:-100}"
export ENGS2950_ROOT="${ENGS_GLASS}"
unset DVFS_PAUSE_ONLY
unset VLLM_PAUSE_ONLY

# shellcheck source=/dev/null
source "${REPO_ROOT}/profiler/jobs/v100_bench_presets.sh"
v100_bench_apply_preset "${MODEL}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
SHAREGPT_FIX_LEN="${SHAREGPT_FIX_LEN:-1}"
FIX_INPUT_LENGTH="${FIX_INPUT_LENGTH:-64}"
FIX_OUTPUT_LENGTH="${FIX_OUTPUT_LENGTH:-0}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/:' '__')"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/bench/results/V100_layer_dvfs/${SAFE_MODEL}/tp${TP_SIZE}/${JOB_TAG}}"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"
MARKERS="${OUT_DIR}/dvfs_markers.jsonl"
METRICS="${OUT_DIR}/run_exec_metrics.json"
_ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl}"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${FREQ_META_DIR}" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"

_dvfs_job_cleanup() {
  kill "${POLLER_PID:-}" 2>/dev/null || true
  POLLER_PID=""
  echo "[dvfs] job cleanup: restoring default GPU clocks on $(hostname -s) ..."
  gpu_freq_lock_force_restore "${FREQ_META_DIR}" || true
}
trap '_dvfs_job_cleanup' EXIT

echo "[dvfs] restore default GPU clocks before bench ..."
FIRST_MHZ="${DVFS_FREQ_SCHEDULE%%,*}"
export FREQ_META_DIR FIRST_MHZ GPU_FREQ_STABLE_TOL_MHZ ENGS2950_ROOT
gpu_freq_lock_force_restore "${FREQ_META_DIR}" || true
python3 - <<'PY'
import json
import os
import subprocess
import sys
import time

repo = os.environ["ENGS2950_ROOT"]
py = f"{repo}/shared/scripts/gpu_freq_lock.py"
out_dir = os.environ["FREQ_META_DIR"]
first = int(os.environ.get("FIRST_MHZ", "700"))
tol = int(os.environ.get("GPU_FREQ_STABLE_TOL_MHZ", "15"))
max_tries = int(os.environ.get("GPU_FREQ_RESTORE_TRIES", "5"))

def smi_mhz():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,clocks.current.graphics", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    return int(float(line[0].split(",")[-1].strip()))

def looks_locked(mhz):
    return mhz is not None and abs(mhz - first) <= tol

for attempt in range(1, max_tries + 1):
    subprocess.run([sys.executable, py, "restore", "--out-dir", out_dir], check=False)
    time.sleep(0.5)
    cur = smi_mhz()
    print(f"[dvfs] post-restore attempt {attempt}/{max_tries}: graphics={cur} MHz")
    if not looks_locked(cur):
        sys.exit(0)
    time.sleep(1.0)

cur = smi_mhz()
print(
    f"[dvfs] WARN: GPU still reads ~{first} MHz after {max_tries} restores "
    f"(graphics={cur}); transitions may fail on this node",
    file=sys.stderr,
)
PY

rm -f "${MARKERS}"
chmod +x "${HOST_POLLER}"
export CUDA_VISIBLE_DEVICES=0
export GPU_FREQ_GPU_INDICES=0
bash "${HOST_POLLER}" "${OUT_DIR}" "${FREQ_META_DIR}" &
POLLER_PID=$!

echo "=== vLLM bench layer-boundary DVFS (prefill-only, real weights) ==="
echo "MODEL=${MODEL}  OUT_DIR=${OUT_DIR}"
echo "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}  VLLM_BENCH_PREFILL_ONLY=${VLLM_BENCH_PREFILL_ONLY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  HOST_POLLER_PID=${POLLER_PID}"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics --format=csv || true

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
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --env "VLLM_BENCH_PREFILL_ONLY=${VLLM_BENCH_PREFILL_ONLY}" \
  --env "VLLM_LAYER_PAUSE=1" \
  --env "VLLM_HOST_POLLER=1" \
  --env "VLLM_EXTERNAL_HOST_POLLER=1" \
  --env "VLLM_BENCH_GPU_POWER=1" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}" \
  --env "GPU_FREQ_SETTLE_SEC=${GPU_FREQ_SETTLE_SEC}" \
  --env "PROFILER_GPU_POWER=${PROFILER_GPU_POWER}" \
  --env "PROFILER_GPU_POWER_INTERVAL_MS=${PROFILER_GPU_POWER_INTERVAL_MS}" \
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets 2>/dev/null || true; exec bash -s' <<INNER
set -euo pipefail
cd "${REPO_ROOT}"

if [[ ! -s "${DATASET}" ]]; then
  python3 -m workloads.generators sharegpt \
    --model "${MODEL}" \
    --num-reqs "${NUM_REQS}" \
    --sps "${SPS}" \
    --seed "${SEED}" \
    --output "${DATASET}" \
    --fix-len \
    --fix-input-length "${FIX_INPUT_LENGTH}" \
    --fix-output-length "${FIX_OUTPUT_LENGTH}"
fi

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

kill "${POLLER_PID}" 2>/dev/null || true
POLLER_PID=""
echo "[dvfs] bench exited — restoring default GPU clocks ..."
gpu_freq_lock_force_restore "${FREQ_META_DIR}" || true

echo "=== Verifying DVFS markers and run metrics ==="
python3 - <<PY
import json
import sys
from pathlib import Path

from bench.core.layer_pause import verify_layer_pause_markers

out_dir = Path("${OUT_DIR}")
cfg_path = Path("${REPO_ROOT}/configs/model/${MODEL}.json")
num_layers = 1
if cfg_path.is_file():
    num_layers = int(json.loads(cfg_path.read_text()).get("num_hidden_layers", 1))

summary = verify_layer_pause_markers(
    out_dir,
    min_markers=max(1, num_layers),
    require_dvfs=True,
)
print(json.dumps(summary, indent=2))

metrics_path = Path("${METRICS}")
if not metrics_path.is_file():
    print("FAIL: missing run_exec_metrics.json", file=sys.stderr)
    sys.exit(1)

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
timing = metrics.get("timing") or {}
throughput = metrics.get("throughput") or {}
energy = metrics.get("energy") or {}

wall = timing.get("wall_sec")
exec_sec = timing.get("exec_sec")
pause = timing.get("pause_sec")
if wall is None or exec_sec is None:
    print("FAIL: metrics missing wall/exec timing", file=sys.stderr)
    sys.exit(1)
if pause is None or pause <= 0:
    print("FAIL: expected positive pause_sec from layer barriers", file=sys.stderr)
    sys.exit(1)
if exec_sec >= wall:
    print(
        f"FAIL: exec_sec ({exec_sec}) should be < wall_sec ({wall})",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"METRICS wall={wall}s exec={exec_sec}s pause={pause}s "
    f"out_tok/s_exec={throughput.get('output_tok_per_sec_exec')} "
    f"energy={energy.get('energy_j')}J energy_exec={energy.get('energy_excl_pause_j')}J"
)

print(
    f"PASS: prefill-only layer-boundary DVFS bench ({summary['marker_count']} markers, "
    f"exec={exec_sec:.2f}s wall={wall:.2f}s)"
)
PY

echo "Markers: ${MARKERS}"
echo "Metrics: ${METRICS}"
echo "Bench output: ${OUT_DIR}"
