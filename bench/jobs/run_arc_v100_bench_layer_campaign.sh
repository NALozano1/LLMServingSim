#!/usr/bin/env bash
# Campaign run: real vLLM prefill + layer-boundary DVFS (scattered or fixed perm).
#
# Required: CAMPAIGN_DIR, RUN_ID, RUN_SPEC_JSON
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"
HOST_POLLER="${JOBS_ROOT}/dvfs_barrier_host_poller.sh"

CAMPAIGN_DIR="${CAMPAIGN_DIR:?CAMPAIGN_DIR}"
RUN_ID="${RUN_ID:?RUN_ID}"
RUN_SPEC_JSON="${RUN_SPEC_JSON:?RUN_SPEC_JSON}"
RUN_DIR="${CAMPAIGN_DIR}/runs/${RUN_ID}"
ARTIFACTS_DIR="${RUN_DIR}/artifacts"
RESULTS_DIR="${RUN_DIR}/results"

eval "$(python3 - <<PY
import json
from pathlib import Path
spec = json.loads(Path("${RUN_SPEC_JSON}").read_text())
bench = spec.get("bench") or {}
dvfs = spec.get("dvfs") or {}
layers = dvfs.get("barrier_layers") or []
freqs = dvfs.get("freq_schedule_mhz") or []
mhz = dvfs.get("gpu_freq_mhz")
print(f"export CAMPAIGN_MODE='{spec.get('campaign_mode', '')}'")
print(f"export MODEL='{spec['model']}'")
print(f"export NUM_HIDDEN_LAYERS='{spec.get('num_hidden_layers', '')}'")
print(f"export ITERATION='{spec.get('iteration', 0)}'")
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
print(f"export GPU_FREQ_MHZ='{mhz or ''}'")
PY
)"

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

export VLLM_LAYER_PAUSE=1
export VLLM_HOST_POLLER=1
export VLLM_EXTERNAL_HOST_POLLER=1
export VLLM_BENCH_PREFILL_ONLY=1
export VLLM_BENCH_GPU_POWER=1
export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export GPU_FREQ_SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-0}"
export GPU_FREQ_STABLE_TOL_MHZ="${GPU_FREQ_STABLE_TOL_MHZ:-15}"
export GPU_FREQ_STABLE_READS="${GPU_FREQ_STABLE_READS:-2}"
export GPU_FREQ_STABLE_TIMEOUT_SEC="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}"
export PROFILER_GPU_POWER=1
export PROFILER_GPU_POWER_INTERVAL_MS="${PROFILER_GPU_POWER_INTERVAL_MS:-100}"
export ENGS2950_ROOT="${ENGS_GLASS}"
unset DVFS_PAUSE_ONLY VLLM_PAUSE_ONLY

# shellcheck source=/dev/null
source "${JOBS_ROOT}/v100_bench_presets.sh"
v100_bench_apply_preset "${MODEL}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/:' '__')"
OUT_DIR="${RUN_DIR}/bench"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"
MARKERS="${OUT_DIR}/dvfs_markers.jsonl"
METRICS="${OUT_DIR}/run_exec_metrics.json"
_ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
DATASET="${CAMPAIGN_DIR}/shared/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl"

mkdir -p "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${FREQ_META_DIR}" "${ARTIFACTS_DIR}" "${RESULTS_DIR}" \
  "${CAMPAIGN_DIR}/shared" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"
# Slurm --exclusive exposes every GPU on the node; tp1 bench/DVFS must target one.
export CUDA_VISIBLE_DEVICES=0
export GPU_FREQ_GPU_INDICES=0

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"

_dvfs_job_cleanup() {
  kill "${POLLER_PID:-}" 2>/dev/null || true
  POLLER_PID=""
  gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true
}
trap '_dvfs_job_cleanup' EXIT

gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true

rm -f "${MARKERS}"
chmod +x "${HOST_POLLER}"
bash "${HOST_POLLER}" "${OUT_DIR}" "${FREQ_META_DIR}" &
POLLER_PID=$!

echo "=== Bench layer DVFS campaign run ==="
echo "RUN_ID=${RUN_ID}  MODE=${CAMPAIGN_MODE}  ITER=${ITERATION}"
echo "MODEL=${MODEL}  BARRIER_LAYERS=${DVFS_BARRIER_LAYERS:-all}"
echo "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN unset" >&2
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
  --env "VLLM_BENCH_PREFILL_ONLY=1" \
  --env "VLLM_LAYER_PAUSE=1" \
  --env "VLLM_HOST_POLLER=1" \
  --env "VLLM_EXTERNAL_HOST_POLLER=1" \
  --env "VLLM_BENCH_GPU_POWER=1" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "DVFS_BARRIER_LAYERS=${DVFS_BARRIER_LAYERS}" \
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
    --model "${MODEL}" --num-reqs "${NUM_REQS}" --sps "${SPS}" --seed "${SEED}" \
    --output "${DATASET}" --fix-len \
    --fix-input-length "${FIX_INPUT_LENGTH}" \
    --fix-output-length "${FIX_OUTPUT_LENGTH}"
fi
python3 -m bench run \
  --model "${MODEL}" --dataset "${DATASET}" --output-dir "${OUT_DIR}" \
  --tensor-parallel-size "${TP_SIZE}" --dtype "${DTYPE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tick-seconds "${TICK_SECONDS}" --num-reqs "${NUM_REQS}" --log-level INFO
INNER

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"
kill "${POLLER_PID}" 2>/dev/null || true
POLLER_PID=""
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
spec = json.loads(Path("${RUN_SPEC_JSON}").read_text(encoding="utf-8"))

min_markers = 1
if spec.get("campaign_mode") == "scattered":
    min_markers = len((spec.get("dvfs") or {}).get("barrier_layers") or [1])
else:
    min_markers = int(spec.get("num_hidden_layers") or 1)

from bench.core.layer_pause import verify_layer_pause_markers
summary = verify_layer_pause_markers(
    out_dir, min_markers=min_markers, require_dvfs=True,
)

metrics_path = out_dir / "run_exec_metrics.json"
if not metrics_path.is_file():
    raise SystemExit("FAIL: missing run_exec_metrics.json")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
timing = metrics.get("timing") or {}
if (timing.get("pause_sec") or 0) <= 0:
    raise SystemExit("FAIL: expected positive pause_sec")

for name in (
    "run_exec_metrics.json", "dvfs_markers.jsonl", "meta.json",
    "requests.jsonl", "timeseries.csv",
):
    src = out_dir / name
    if src.is_file():
        shutil.copy2(src, artifacts / name)
gp = out_dir / "gpu_power"
if gp.is_dir():
    dst = artifacts / "gpu_power"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(gp, dst)
gf = out_dir / "gpu_freq"
if gf.is_dir():
    dst = artifacts / "gpu_freq"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(gf, dst)

shutil.copy2(Path("${RUN_SPEC_JSON}"), artifacts / "run_spec.json")
sim_src = run_dir / "sim_replication.json"
if sim_src.is_file():
    shutil.copy2(sim_src, artifacts / "sim_replication.json")

node_meta = {
    "run_id": "${RUN_ID}",
    "campaign_dir": "${CAMPAIGN_DIR}",
    "campaign_mode": "${CAMPAIGN_MODE}",
    "slurm_job_id": "${JOB_TAG}",
    "hostname": socket.gethostname(),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "model": "${MODEL}",
    "dvfs_barrier_layers": "${DVFS_BARRIER_LAYERS}",
    "dvfs_freq_schedule": "${DVFS_FREQ_SCHEDULE}",
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "bench_dir": str(out_dir),
    "run_spec": spec,
}
(out_dir / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")
(artifacts / "node_meta.json").write_text(json.dumps(node_meta, indent=2) + "\n")

result = {
    **node_meta,
    "status": "completed",
    "marker_summary": summary,
    "run_exec_metrics": metrics,
    "timing": timing,
    "throughput": metrics.get("throughput") or {},
    "energy": metrics.get("energy") or {},
    "tokens": metrics.get("tokens") or {},
    "n_dvfs_transitions": summary.get("marker_count"),
}
results.mkdir(parents=True, exist_ok=True)
(results / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "run_id": result["run_id"],
    "status": "completed",
    "wall_sec": timing.get("wall_sec"),
    "exec_sec": timing.get("exec_sec"),
    "pause_sec": timing.get("pause_sec"),
    "energy_j": (metrics.get("energy") or {}).get("energy_j"),
    "markers": summary.get("marker_count"),
}, indent=2))
COLLECT

echo "PASS: campaign run ${RUN_ID}"
echo "  results: ${RESULTS_DIR}/summary.json"
echo "  artifacts: ${ARTIFACTS_DIR}/"
