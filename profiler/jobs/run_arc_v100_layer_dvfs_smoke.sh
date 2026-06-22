#!/usr/bin/env bash
# Smoke test: in-place layer-boundary DVFS pause during profiler fire().
#
# Runs profiler inside Apptainer; host-side shell poller applies DVFS
# (sudo nvidia-smi-clocks), verifies stable clocks, then acks worker.
#
#   export HF_TOKEN=...
#   bash profiler/jobs/run_arc_v100_layer_dvfs_smoke.sh
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

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
HARDWARE="${HARDWARE:-V100_layer_dvfs_smoke}"
TP_DEGREES="${TP_DEGREES:-1}"
DTYPE="${DTYPE:-float16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-2048}"
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-2}"
VERBOSITY="${VERBOSITY:-"--verbose"}"

export DVFS_LAYER_PAUSE=1
export DVFS_HOST_POLLER=1
export DVFS_LAYER_PAUSE_MAX_SHOTS="${DVFS_LAYER_PAUSE_MAX_SHOTS:-1}"
export PROFILER_MAX_SHOTS="${PROFILER_MAX_SHOTS:-${DVFS_LAYER_PAUSE_MAX_SHOTS}}"
export DVFS_FREQ_SCHEDULE="${DVFS_FREQ_SCHEDULE:-700,1400}"
export GPU_FREQ_SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-0}"
export GPU_FREQ_STABLE_TOL_MHZ="${GPU_FREQ_STABLE_TOL_MHZ:-15}"
export GPU_FREQ_STABLE_READS="${GPU_FREQ_STABLE_READS:-2}"
export GPU_FREQ_STABLE_TIMEOUT_SEC="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}"
export ENGS2950_ROOT="${ENGS_GLASS}"
unset DVFS_PAUSE_ONLY
export PROFILER_GPU_POWER="${PROFILER_GPU_POWER:-1}"
export PROFILER_GPU_POWER_INTERVAL_MS="${PROFILER_GPU_POWER_INTERVAL_MS:-100}"

VARIANT_TAG="fp16"
case "${DTYPE}" in
  float16) VARIANT_TAG="fp16" ;;
  bfloat16) VARIANT_TAG="bf16" ;;
  *) VARIANT_TAG="${DTYPE}" ;;
esac

PERF_VARIANT_ROOT="${REPO_ROOT}/profiler/perf/${HARDWARE}/${MODEL}/${VARIANT_TAG}"
TP_WATCH_ROOT="${PERF_VARIANT_ROOT}/tp1"
FREQ_META_DIR="${TP_WATCH_ROOT}/gpu_freq"
MARKERS="${TP_WATCH_ROOT}/dvfs_markers.jsonl"
RUN_TAG="${RUN_TAG:-${HARDWARE}}"

mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${TP_WATCH_ROOT}" "${FREQ_META_DIR}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

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

# Prior failed jobs may leave clocks locked (~705 MHz); 700→1400 transitions
# then fail (ioctl OK, clocks stuck). Always reset before smoke (see 8008603 vs 8008908).
echo "[dvfs] restore default GPU clocks before smoke ..."
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
    f"(graphics={cur}); 700→1400 transitions may fail on this node",
    file=sys.stderr,
)
PY
GPU_FREQ_LOCK_STATE=0

rm -f "${MARKERS}"
chmod +x "${HOST_POLLER}"
bash "${HOST_POLLER}" "${TP_WATCH_ROOT}" "${FREQ_META_DIR}" &
POLLER_PID=$!

echo "=== Layer-boundary DVFS smoke (ARC / engs-glass) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL=$MODEL  HARDWARE=$HARDWARE  TP=$TP_DEGREES  NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS}"
echo "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}  GPU_FREQ_SETTLE_SEC=${GPU_FREQ_SETTLE_SEC}"
echo "GPU_FREQ_STABLE_TOL_MHZ=${GPU_FREQ_STABLE_TOL_MHZ}  STABLE_READS=${GPU_FREQ_STABLE_READS}"
echo "HOST_POLLER_PID=${POLLER_PID}"
echo "WATCH_ROOT=${TP_WATCH_ROOT}"
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
  --env "TORCH_HOME=${SCRATCH}/torch" \
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "DVFS_LAYER_PAUSE=1" \
  --env "DVFS_HOST_POLLER=1" \
  --env "PROFILER_MAX_SHOTS=${PROFILER_MAX_SHOTS}" \
  --env "DVFS_FREQ_SCHEDULE=${DVFS_FREQ_SCHEDULE}" \
  --env "GPU_FREQ_SETTLE_SEC=${GPU_FREQ_SETTLE_SEC}" \
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
  --env "SLURM_JOB_ID=${JOB_TAG}" \
  --env "PROFILER_GPU_POWER=${PROFILER_GPU_POWER}" \
  --env "PROFILER_GPU_POWER_INTERVAL_MS=${PROFILER_GPU_POWER_INTERVAL_MS}" \
  "$VLLM_IMAGE" \
  bash -c 'pip install -q datasets matplotlib 2>/dev/null || true; exec python3 -m profiler slice "'"${MODEL}"'" \
    --hardware "'"${HARDWARE}"'" \
    --tp-refresh "'"${TP_DEGREES}"'" \
    --group dense \
    --dtype "'"${DTYPE}"'" \
    --max-num-batched-tokens "'"${MAX_NUM_BATCHED_TOKENS}"'" \
    --max-num-seqs "'"${MAX_NUM_SEQS}"'" \
    --attention-max-kv "'"${ATTENTION_MAX_KV}"'" \
    --measurement-iterations "'"${MEASUREMENT_ITERATIONS}"'" \
    --num-hidden-layers "'"${NUM_HIDDEN_LAYERS}"'" \
    --skip-skew \
    --force \
    '"${VERBOSITY}"''

export SLURM_JOB_ACCOUNT="${OLD_ACCOUNT}"

kill "${POLLER_PID}" 2>/dev/null || true
POLLER_PID=""
echo "[dvfs] vLLM exited — restoring default GPU clocks before verification ..."
gpu_freq_lock_force_restore "${FREQ_META_DIR}" || true

echo "=== Smoke done ==="
echo "Markers: ${MARKERS}"
if [[ -f "${MARKERS}" ]]; then
  echo "--- dvfs_markers.jsonl ---"
  cat "${MARKERS}"
fi

python3 - <<PY
import json
import sys
from pathlib import Path

markers = Path("${MARKERS}")
if not markers.is_file():
    print("FAIL: no markers file", file=sys.stderr)
    sys.exit(1)

lines = [json.loads(ln) for ln in markers.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not lines:
    print("FAIL: empty markers", file=sys.stderr)
    sys.exit(1)

schedule = [int(x.strip()) for x in "${DVFS_FREQ_SCHEDULE}".split(",") if x.strip()]
expected_layers = int("${NUM_HIDDEN_LAYERS}")
expected_by_layer = {
    f"layers.{i}": schedule[i % len(schedule)]
    for i in range(expected_layers)
}

tol = int("${GPU_FREQ_STABLE_TOL_MHZ}")
fail = False

if len(lines) < expected_layers:
    print(
        f"FAIL: expected {expected_layers} markers, got {len(lines)}",
        file=sys.stderr,
    )
    fail = True

seen_layers: set[str] = set()
for rec in lines:
    mode = rec.get("mode")
    mhz = rec.get("freq_mhz")
    apply_ok = rec.get("freq_apply_ok")
    stable_ok = rec.get("freq_stable_ok")
    clocks = rec.get("clocks_after") or []
    layer_name = rec.get("layer_name")
    print(
        f"marker layer={layer_name} mode={mode} "
        f"freq_mhz={mhz} apply_ok={apply_ok} stable_ok={stable_ok} "
        f"stable_sec={rec.get('freq_stable_sec')} pause_sec={rec.get('pause_sec', 0):.3f}"
    )
    if layer_name:
        seen_layers.add(layer_name)
    if mode != "dvfs":
        print("  FAIL: expected mode=dvfs", file=sys.stderr)
        fail = True
        continue
    if not apply_ok:
        print("  FAIL: freq_apply_ok is false", file=sys.stderr)
        fail = True
    if not stable_ok:
        print("  FAIL: freq_stable_ok is false", file=sys.stderr)
        fail = True
    if mhz is None:
        print("  FAIL: missing freq_mhz", file=sys.stderr)
        fail = True
        continue
    exp = expected_by_layer.get(layer_name)
    if exp is not None and int(mhz) != int(exp):
        print(
            f"  FAIL: layer {layer_name} freq_mhz={mhz}, expected {exp}",
            file=sys.stderr,
        )
        fail = True
    for g in clocks:
        cur = g.get("graphics_mhz")
        if cur is None:
            continue
        if abs(int(cur) - int(mhz)) > tol:
            print(
                f"  FAIL: gpu {g.get('index')} at {cur} MHz, target {mhz} +/- {tol}",
                file=sys.stderr,
            )
            fail = True

for layer_name, exp_mhz in expected_by_layer.items():
    if layer_name not in seen_layers:
        print(f"FAIL: missing marker for {layer_name} (expected {exp_mhz} MHz)", file=sys.stderr)
        fail = True

if fail:
    print("FAIL: DVFS smoke verification failed", file=sys.stderr)
    sys.exit(1)
print(
    f"PASS: layer-boundary DVFS smoke ({expected_layers} layers, "
    f"stable freq verified)"
)
PY

EXEC_METRICS="${TP_WATCH_ROOT}/shot_exec_metrics.jsonl"
RUN_METRICS="${TP_WATCH_ROOT}/run_exec_metrics.json"
if [[ -f "${EXEC_METRICS}" ]]; then
  echo ""
  echo "=== Exec metrics (excluding pause windows) ==="
  python3 - <<PY
import json
from pathlib import Path

path = Path("${EXEC_METRICS}")
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    print(
        f"shot={rec.get('shot_key')} "
        f"arm={rec.get('arm') or '-'} "
        f"measured={rec.get('measured_sec')}s "
        f"effective={rec.get('effective_runtime_sec') or rec.get('measured_exec_sec')}s "
        f"pause={rec.get('barrier_wait_sec') or rec.get('marker_pause_sec')}s "
        f"energy={rec.get('energy_j')}J "
        f"energy_exec={rec.get('energy_excl_pause_j') or rec.get('effective_energy_j')}J"
    )

run_path = Path("${RUN_METRICS}")
summary = {
    "job_id": "${JOB_TAG}",
    "run_tag": "${RUN_TAG}",
    "model": "${MODEL}",
    "hardware": "${HARDWARE}",
    "dvfs_freq_schedule": "${DVFS_FREQ_SCHEDULE}",
    "num_hidden_layers": int("${NUM_HIDDEN_LAYERS}"),
    "markers": "${MARKERS}",
    "exec_metrics": "${EXEC_METRICS}",
    "run_exec_metrics": str(run_path),
}
records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
if records:
    summary["last_shot"] = records[-1]
if run_path.is_file():
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summary["run_totals"] = run
    print("")
    print(
        f"RUN effective_runtime={run.get('effective_runtime_sec')}s "
        f"pause_overhead={run.get('pause_overhead_sec')}s "
        f"effective_energy={run.get('effective_energy_j')}J "
        f"shots={run.get('shot_count')}"
    )
    if run.get("ttft_ms") is not None:
        ref = (run.get("serving_latency") or {}).get("reference_workload") or {}
        print(
            f"RUN TTFT={run.get('ttft_ms')}ms TPOT={run.get('tpot_ms')}ms "
            f"(synthetic profile estimate — ignore; use exec metrics above)"
        )
Path("${TP_WATCH_ROOT}/dvfs_smoke_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
)
print(f"wrote ${TP_WATCH_ROOT}/dvfs_smoke_summary.json")
PY
fi
