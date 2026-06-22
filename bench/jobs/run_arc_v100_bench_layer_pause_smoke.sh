#!/usr/bin/env bash
# Smoke: in-place layer pause/continue during real vLLM bench (any model).
#
# Uses python -m bench run with worker_extension_cls + host poller.
# No profiler slice, no DVFS frequency changes (pause-only ack).
#
#   export HF_TOKEN=...
#   bash bench/jobs/run_arc_v100_bench_layer_pause_smoke.sh
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"

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
export VLLM_PAUSE_ONLY=1
export VLLM_PAUSE_ONLY_DELAY_SEC="${VLLM_PAUSE_ONLY_DELAY_SEC:-0.05}"
export VLLM_BENCH_PREFILL_ONLY="${VLLM_BENCH_PREFILL_ONLY:-1}"
export VLLM_BENCH_GPU_POWER=1
# Legacy aliases (host poller + profiler paths)
export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export DVFS_PAUSE_ONLY=1
export PAUSE_ONLY_DELAY_SEC="${VLLM_PAUSE_ONLY_DELAY_SEC}"
export PROFILER_GPU_POWER="${PROFILER_GPU_POWER:-1}"
export ENGS2950_ROOT="${ENGS_GLASS}"

HOST_POLLER="${JOBS_ROOT}/dvfs_barrier_host_poller.sh"

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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/bench/results/V100_layer_pause_smoke/${SAFE_MODEL}/tp${TP_SIZE}/${JOB_TAG}}"
_ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl}"
MARKERS="${OUT_DIR}/dvfs_markers.jsonl"
METRICS="${OUT_DIR}/run_exec_metrics.json"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

export CUDA_VISIBLE_DEVICES=0
export GPU_FREQ_GPU_INDICES=0

_poll_cleanup() {
  kill "${POLLER_PID:-}" 2>/dev/null || true
  POLLER_PID=""
}
trap '_poll_cleanup' EXIT

rm -f "${MARKERS}"
chmod +x "${HOST_POLLER}"
bash "${HOST_POLLER}" "${OUT_DIR}" "${OUT_DIR}/gpu_freq" &
POLLER_PID=$!

echo "=== vLLM bench layer pause smoke (real weights, any model) ==="
echo "MODEL=${MODEL}  OUT_DIR=${OUT_DIR}"
echo "VLLM_LAYER_PAUSE=1  VLLM_PAUSE_ONLY=1  VLLM_BENCH_PREFILL_ONLY=${VLLM_BENCH_PREFILL_ONLY}  delay=${VLLM_PAUSE_ONLY_DELAY_SEC}s"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HOST_POLLER_PID=${POLLER_PID}"
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
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  --env "VLLM_BENCH_PREFILL_ONLY=${VLLM_BENCH_PREFILL_ONLY}" \
  --env "VLLM_LAYER_PAUSE=1" \
  --env "VLLM_HOST_POLLER=1" \
  --env "VLLM_EXTERNAL_HOST_POLLER=1" \
  --env "VLLM_PAUSE_ONLY=1" \
  --env "VLLM_PAUSE_ONLY_DELAY_SEC=${VLLM_PAUSE_ONLY_DELAY_SEC}" \
  --env "VLLM_BENCH_GPU_POWER=1" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "DVFS_PAUSE_ONLY=1" \
  --env "PAUSE_ONLY_DELAY_SEC=${VLLM_PAUSE_ONLY_DELAY_SEC}" \
  --env "PROFILER_GPU_POWER=${PROFILER_GPU_POWER}" \
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

echo "=== Verifying layer pause markers and run metrics ==="
python3 - <<PY
import json
import sys
from pathlib import Path

from bench.core.layer_pause import verify_layer_pause_markers

cfg_path = Path("${REPO_ROOT}/configs/model/${MODEL}.json")
num_layers = 1
if cfg_path.is_file():
    num_layers = int(json.loads(cfg_path.read_text()).get("num_hidden_layers", 1))

summary = verify_layer_pause_markers(
    Path("${OUT_DIR}"),
    min_markers=max(1, num_layers),
    require_pause_only=True,
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

print(
    f"METRICS wall={wall}s exec={exec_sec}s pause={pause}s "
    f"out_tok/s_exec={throughput.get('output_tok_per_sec_exec')} "
    f"energy={energy.get('energy_j')}J"
)
print(
    f"PASS: prefill-only layer pause smoke ({summary['marker_count']} markers, "
    f"mean pause {summary['pause_sec_mean']:.3f}s)"
)
PY

echo "Markers: ${MARKERS}"
echo "Metrics: ${METRICS}"
echo "Bench output: ${OUT_DIR}"
