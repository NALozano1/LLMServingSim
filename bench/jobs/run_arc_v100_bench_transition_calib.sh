#!/usr/bin/env bash
# Transition-effects calibration run: single A/B arm of the sweep.
#
# Reads CALIB_SPEC_JSON for all parameters.  The spec encodes the arm:
#   n_transitions=0   -> baseline (no DVFS barriers at all)
#   n_transitions=N   -> N scattered barriers, one freq transition each
#
# Required env:
#   CALIB_DIR        — campaign root (bench/campaigns/v100_transition_calib_*)
#   RUN_ID           — unique run ID for this arm
#   CALIB_SPEC_JSON  — path to this arm's spec JSON
#
# Optional:
#   DVFS_APPLY_MODE=sync|async (default: async — reason we exist)
#   MODEL, HF_CACHE_ROOT, VLLM_IMAGE, CONTAINER_RUNTIME, SCRATCH
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/bench/jobs"
PROFILER_JOBS="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"
ASYNC_POLLER="${JOBS_ROOT}/dvfs_barrier_host_poller_async.sh"

CALIB_DIR="${CALIB_DIR:?CALIB_DIR required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
CALIB_SPEC_JSON="${CALIB_SPEC_JSON:?CALIB_SPEC_JSON required}"
RUN_DIR="${CALIB_DIR}/runs/${RUN_ID}"
ARTIFACTS_DIR="${RUN_DIR}/artifacts"
RESULTS_DIR="${RUN_DIR}/results"

# Apply mode: default to async for calib (that is the whole point).
# Set DVFS_APPLY_MODE=sync to run in blocking mode for comparison.
export DVFS_APPLY_MODE="${DVFS_APPLY_MODE:-async}"

# Parse spec.
eval "$(python3 - <<PY
import json
from pathlib import Path
spec = json.loads(Path("${CALIB_SPEC_JSON}").read_text())
bench = spec.get("bench") or {}
dvfs = spec.get("dvfs") or {}
layers = dvfs.get("barrier_layers") or []
freqs = dvfs.get("freq_schedule_mhz") or []
print(f"export N_TRANSITIONS='{dvfs.get('n_transitions', 0)}'")
print(f"export MODEL='{spec['model']}'")
print(f"export DTYPE='{bench.get('dtype', 'float16')}'")
print(f"export TP_SIZE='{bench.get('tp_size', 1)}'")
print(f"export NUM_REQS='{bench.get('num_reqs', 1)}'")
print(f"export SPS='{bench.get('sps', 100)}'")
print(f"export SEED='{bench.get('seed', 42)}'")
print(f"export TICK_SECONDS='{bench.get('tick_seconds', 0.5)}'")
print(f"export FIX_INPUT_LENGTH='{bench.get('fix_input_length', 64)}'")
print(f"export FIX_OUTPUT_LENGTH='{bench.get('fix_output_length', 0)}'")
print(f"export MAX_NUM_SEQS='{spec.get('max_num_seqs', 1)}'")
print(f"export MAX_NUM_BATCHED_TOKENS='{spec.get('max_num_batched_tokens', 512)}'")
print(f"export MAX_MODEL_LEN='{spec.get('max_model_len', 512)}'")
print(f"export GPU_MEMORY_UTILIZATION='{spec.get('gpu_memory_utilization', 0.92)}'")
print(f"export DVFS_BARRIER_LAYERS='{','.join(str(x) for x in layers)}'")
print(f"export DVFS_FREQ_SCHEDULE='{','.join(str(x) for x in freqs)}'")
print(f"export CALIB_ITERATION='{spec.get('iteration', 0)}'")
print(f"export ARM_LABEL='{spec.get('arm_label', '')}'")
PY
)"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/:' '__')"
OUT_DIR="${RUN_DIR}/bench"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"
MARKERS="${OUT_DIR}/dvfs_markers.jsonl"
METRICS="${OUT_DIR}/run_exec_metrics.json"
_ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
DATASET="${CALIB_DIR}/shared/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl"

# Env passthrough for the Apptainer inner shell.
if [[ "${N_TRANSITIONS}" -gt 0 ]]; then
  export VLLM_LAYER_PAUSE=1
  export VLLM_HOST_POLLER=1
  export VLLM_EXTERNAL_HOST_POLLER=1
  export DVFS_LAYER_PAUSE=1
  export DVFS_HOST_POLLER=1
  RUN_WITH_BARRIERS=1
else
  # Baseline: no barriers at all — turn layer pause off.
  unset VLLM_LAYER_PAUSE VLLM_HOST_POLLER VLLM_EXTERNAL_HOST_POLLER
  unset DVFS_LAYER_PAUSE DVFS_HOST_POLLER
  RUN_WITH_BARRIERS=0
fi
export VLLM_BENCH_PREFILL_ONLY=1
export VLLM_BENCH_GPU_POWER=1
export PROFILER_GPU_POWER=1
export PROFILER_GPU_POWER_INTERVAL_MS="${PROFILER_GPU_POWER_INTERVAL_MS:-100}"
export GPU_FREQ_SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-0}"
export GPU_FREQ_STABLE_TOL_MHZ="${GPU_FREQ_STABLE_TOL_MHZ:-15}"
export GPU_FREQ_STABLE_READS="${GPU_FREQ_STABLE_READS:-2}"
export GPU_FREQ_STABLE_TIMEOUT_SEC="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}"
export ENGS2950_ROOT="${ENGS_GLASS}"
unset DVFS_PAUSE_ONLY VLLM_PAUSE_ONLY

# shellcheck source=/dev/null
source "${PROFILER_JOBS}/v100_bench_presets.sh"
v100_bench_apply_preset "${MODEL}"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${FREQ_META_DIR}" "${ARTIFACTS_DIR}" "${RESULTS_DIR}" \
  "${CALIB_DIR}/shared" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"
export CUDA_VISIBLE_DEVICES=0
export GPU_FREQ_GPU_INDICES=0

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"

POLLER_PID=""

_calib_cleanup() {
  if [[ -n "${POLLER_PID:-}" ]]; then
    kill "${POLLER_PID}" 2>/dev/null || true
    POLLER_PID=""
  fi
  gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true
}
trap '_calib_cleanup' EXIT

gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true

rm -f "${MARKERS}"

if [[ "${RUN_WITH_BARRIERS}" -eq 1 ]]; then
  chmod +x "${ASYNC_POLLER}"
  DVFS_APPLY_MODE="${DVFS_APPLY_MODE}" \
  bash "${ASYNC_POLLER}" "${OUT_DIR}" "${FREQ_META_DIR}" &
  POLLER_PID=$!
fi

echo "=== Transition-effects calibration run ==="
echo "RUN_ID=${RUN_ID}  ARM=${ARM_LABEL}  N_TRANSITIONS=${N_TRANSITIONS}  ITER=${CALIB_ITERATION}"
echo "MODEL=${MODEL}  DVFS_APPLY_MODE=${DVFS_APPLY_MODE}"
echo "BARRIER_LAYERS=${DVFS_BARRIER_LAYERS:-none}  FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE:-none}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset" >&2
fi

OLD_ACCOUNT="${SLURM_JOB_ACCOUNT:-}"
unset SLURM_JOB_ACCOUNT

# Build inner env args array; only pass barrier-related vars when needed.
# Initialise as empty array — safe to expand even when empty (no nounset error).
_inner_barrier_args=()
if [[ "${RUN_WITH_BARRIERS}" -eq 1 ]]; then
  _inner_barrier_args+=(
    --env "VLLM_LAYER_PAUSE=1"
    --env "VLLM_HOST_POLLER=1"
    --env "VLLM_EXTERNAL_HOST_POLLER=1"
    --env "DVFS_LAYER_PAUSE=1"
    --env "DVFS_HOST_POLLER=1"
    --env "DVFS_BARRIER_LAYERS=${DVFS_BARRIER_LAYERS}"
    --env "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}"
  )
fi

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
  --env "VLLM_BENCH_PREFILL_ONLY=1" \
  --env "VLLM_BENCH_GPU_POWER=1" \
  --env "PROFILER_GPU_POWER=${PROFILER_GPU_POWER}" \
  --env "PROFILER_GPU_POWER_INTERVAL_MS=${PROFILER_GPU_POWER_INTERVAL_MS}" \
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
  "${_inner_barrier_args[@]}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets 2>/dev/null || true; exec bash -s' <<INNER
set -euo pipefail
cd "${REPO_ROOT}"
if [[ ! -s "${DATASET}" ]]; then
  python3 -m workloads.generators sharegpt \
    --model "${MODEL}" --num-reqs "${NUM_REQS}" --sps "${SPS}" --seed "${SEED}" \
    --output "${DATASET}" --fix-len \
    --fix-input-length "${FIX_INPUT_LENGTH}" --fix-output-length "${FIX_OUTPUT_LENGTH}"
fi
python3 -m bench run \
  --model "${MODEL}" --dataset "${DATASET}" --output-dir "${OUT_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" --dtype "${DTYPE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tick-seconds "${TICK_SECONDS}" --num-reqs "${NUM_REQS}" --log-level INFO
INNER

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

# Give the poller time to finish writing markers before killing it.
# The async path dispatches the ioctl before acking the worker, then continues
# wait-stable + marker-write after the container exits — so the container can
# finish before those steps complete.  Wait up to 30 s for all expected entries.
if [[ -n "${POLLER_PID:-}" && "${N_TRANSITIONS:-0}" -gt 0 ]]; then
  _waited_markers=0
  for _mi in $(seq 1 30); do
    _mc=$(python3 -c "
from pathlib import Path
p = Path('${MARKERS}')
print(sum(1 for l in p.read_text('utf-8').splitlines() if l.strip()) if p.exists() else 0)
" 2>/dev/null || echo 0)
    if [[ "${_mc}" -ge "${N_TRANSITIONS}" ]]; then _waited_markers=1; break; fi
    sleep 1
  done
  [[ "${_waited_markers}" -eq 1 ]] || echo "[calib] WARN: timed out waiting for ${N_TRANSITIONS} markers (got ${_mc:-0})" >&2
fi

if [[ -n "${POLLER_PID:-}" ]]; then
  kill "${POLLER_PID}" 2>/dev/null || true
  wait "${POLLER_PID}" 2>/dev/null || true
  POLLER_PID=""
fi
gpu_freq_lock_force_restore "${FREQ_META_DIR}" || true

python3 - <<COLLECT
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path("${RUN_DIR}")
out_dir = Path("${OUT_DIR}")
artifacts = run_dir / "artifacts"
results = run_dir / "results"
spec = json.loads(Path("${CALIB_SPEC_JSON}").read_text(encoding="utf-8"))
n_transitions = int("${N_TRANSITIONS}")

metrics_path = out_dir / "run_exec_metrics.json"
if not metrics_path.is_file():
    raise SystemExit("FAIL: missing run_exec_metrics.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
timing = metrics.get("timing") or {}

# For baseline (0 transitions), pause_sec should be 0; for armed runs, > 0.
if n_transitions > 0:
    if (timing.get("pause_sec") or 0) <= 0 and "${DVFS_APPLY_MODE}" == "sync":
        raise SystemExit("FAIL: expected positive pause_sec for n_transitions > 0 in sync mode")

# Load markers to count async ioctl wall times.
markers_path = out_dir / "dvfs_markers.jsonl"
markers = []
if markers_path.is_file():
    for line in markers_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            markers.append(json.loads(line))

ioctl_walls = [
    float(m["ioctl_wall_sec"])
    for m in markers
    if m.get("ioctl_wall_sec") is not None
]

for name in (
    "run_exec_metrics.json", "dvfs_markers.jsonl", "meta.json",
    "requests.jsonl", "timeseries.csv",
):
    src = out_dir / name
    if src.is_file():
        shutil.copy2(src, artifacts / name)
for sub in ("gpu_power", "gpu_freq"):
    src = out_dir / sub
    if src.is_dir():
        dst = artifacts / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

shutil.copy2(Path("${CALIB_SPEC_JSON}"), artifacts / "calib_spec.json")

node_meta = {
    "run_id": "${RUN_ID}",
    "calib_dir": "${CALIB_DIR}",
    "arm_label": "${ARM_LABEL}",
    "n_transitions": n_transitions,
    "dvfs_apply_mode": "${DVFS_APPLY_MODE}",
    "slurm_job_id": "${JOB_TAG}",
    "hostname": socket.gethostname(),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "model": "${MODEL}",
    "dvfs_barrier_layers": "${DVFS_BARRIER_LAYERS:-}",
    "dvfs_freq_schedule": "${DVFS_FREQ_SCHEDULE:-}",
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "bench_dir": str(out_dir),
    "calib_spec": spec,
}
(out_dir / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")
(artifacts / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")

result = {
    **node_meta,
    "status": "completed",
    "run_exec_metrics": metrics,
    "timing": timing,
    "throughput": metrics.get("throughput") or {},
    "energy": metrics.get("energy") or {},
    "tokens": metrics.get("tokens") or {},
    "marker_count": len(markers),
    "ioctl_wall_sec_mean": round(sum(ioctl_walls) / len(ioctl_walls), 6) if ioctl_walls else None,
    "ioctl_wall_sec_sum": round(sum(ioctl_walls), 6) if ioctl_walls else None,
}
results.mkdir(parents=True, exist_ok=True)
(results / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "run_id": result["run_id"],
    "arm_label": result["arm_label"],
    "n_transitions": n_transitions,
    "dvfs_apply_mode": "${DVFS_APPLY_MODE}",
    "status": "completed",
    "wall_sec": timing.get("wall_sec"),
    "exec_sec": timing.get("exec_sec"),
    "pause_sec": timing.get("pause_sec"),
    "ioctl_wall_sec_mean": result["ioctl_wall_sec_mean"],
    "marker_count": len(markers),
}, indent=2))
COLLECT

echo "PASS: calib run ${RUN_ID} (arm=${ARM_LABEL} n_transitions=${N_TRANSITIONS})"
echo "  results: ${RESULTS_DIR}/summary.json"
echo "  artifacts: ${ARTIFACTS_DIR}/"
