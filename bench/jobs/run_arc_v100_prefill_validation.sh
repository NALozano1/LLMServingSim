#!/usr/bin/env bash
# Single-arm prefill validation run: clock-locked, no layer-pause, power-on.
#
# One run = one (model, clock, tp) combination.  Designed to feed
# LLMServingSim sim-accuracy comparison (Tier-1 / Tier-2 validation).
#
# Required env:
#   MODEL          — HF model name
#   OUT_DIR        — output directory for this arm
#   CAMPAIGN_DIR   — campaign root (for shared dataset cache)
#
# Optional env:
#   GPU_FREQ_MHZ   — target clock in MHz; empty / unset = run uncapped
#   TP_SIZE        — tensor-parallel degree (default 1)
#   NUM_REQS       — number of prefill requests (default 50)
#   FIX_INPUT_LENGTH — fixed input token count (default 512)
#   MAX_MODEL_LEN  — vLLM max_model_len (default 4096)
#   MAX_NUM_SEQS   — vLLM max_num_seqs (default 8)
#   MAX_NUM_BATCHED_TOKENS — vLLM batch token budget (default 8192)
#   GPU_MEMORY_UTILIZATION — (default 0.92)
#   ARM_LABEL      — friendly label for summary (auto-derived if empty)
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/bench/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"

MODEL="${MODEL:?MODEL required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:?CAMPAIGN_DIR required}"
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-}"          # empty = uncapped
TP_SIZE="${TP_SIZE:-1}"
NUM_REQS="${NUM_REQS:-50}"
FIX_INPUT_LENGTH="${FIX_INPUT_LENGTH:-512}"
FIX_OUTPUT_LENGTH="${FIX_OUTPUT_LENGTH:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"
SPS="${SPS:-100}"
TICK_SECONDS="${TICK_SECONDS:-0.5}"

if [[ -z "${ARM_LABEL:-}" ]]; then
  if [[ -n "${GPU_FREQ_MHZ}" ]]; then
    ARM_LABEL="${GPU_FREQ_MHZ}mhz"
  else
    ARM_LABEL="uncapped"
  fi
fi

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${ENGS_GLASS}/infra/hf_cache}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

SAFE_MODEL="$(echo "${MODEL}" | tr '/:' '__')"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"
RESULTS_DIR="${OUT_DIR}/results"
_ds_tag="ml${MAX_MODEL_LEN}-fixlen${FIX_INPUT_LENGTH}o${FIX_OUTPUT_LENGTH}"
DATASET="${CAMPAIGN_DIR}/shared/sharegpt-${SAFE_MODEL}-${NUM_REQS}-sps${SPS}-seed${SEED}-${_ds_tag}.jsonl"

# ── Power + prefill-only env (always on, no barriers) ─────────────────────────
export VLLM_BENCH_PREFILL_ONLY=1
export VLLM_BENCH_GPU_POWER=1
export PROFILER_GPU_POWER=1
export PROFILER_GPU_POWER_INTERVAL_MS="${PROFILER_GPU_POWER_INTERVAL_MS:-100}"
# Layer-pause MUST be off — these are clean validation runs.
unset VLLM_LAYER_PAUSE VLLM_HOST_POLLER VLLM_EXTERNAL_HOST_POLLER 2>/dev/null || true
unset DVFS_LAYER_PAUSE DVFS_HOST_POLLER 2>/dev/null || true

mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${FREQ_META_DIR}" "${RESULTS_DIR}" \
  "${CAMPAIGN_DIR}/shared" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"

# ── Clock hold poller ─────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"

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

gpu_freq_lock_force_restore "${FREQ_META_DIR}" 2>/dev/null || true

if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  python3 "${GPU_FREQ_LOCK_PY}" hold --mhz "${GPU_FREQ_MHZ}" \
    --out-dir "${FREQ_META_DIR}" &
  HOLD_PID=$!
  echo "[dvfs] hold poller pid=${HOLD_PID} target=${GPU_FREQ_MHZ} MHz"
fi

echo "=== Prefill validation run ==="
echo "MODEL=${MODEL}  ARM_LABEL=${ARM_LABEL}  GPU_FREQ_MHZ=${GPU_FREQ_MHZ:-uncapped}"
echo "TP=${TP_SIZE}  NUM_REQS=${NUM_REQS}  FIX_INPUT_LENGTH=${FIX_INPUT_LENGTH}"
echo "OUT_DIR=${OUT_DIR}"
nvidia-smi -L || true

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
  --env "VLLM_BENCH_PREFILL_ONLY=1" \
  --env "VLLM_BENCH_GPU_POWER=1" \
  --env "PROFILER_GPU_POWER=1" \
  --env "PROFILER_GPU_POWER_INTERVAL_MS=${PROFILER_GPU_POWER_INTERVAL_MS:-100}" \
  --env "ENGS2950_ROOT=${ENGS_GLASS}" \
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

# ── Clock audit ───────────────────────────────────────────────────────────────
if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  touch "${FREQ_META_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
  wait "${HOLD_PID}" 2>/dev/null || true
  HOLD_PID=""

  HOLD_SUMMARY="${FREQ_META_DIR}/gpu_freq_hold_summary.json"
  if [[ ! -f "${HOLD_SUMMARY}" ]]; then
    echo "[dvfs] AUDIT ERROR: no hold summary at ${HOLD_SUMMARY}" >&2
    echo '{"status":"AUDIT_MISSING","gpu_freq_mhz":'"${GPU_FREQ_MHZ}"'}' \
      > "${RESULTS_DIR}/summary.json"
    exit 3
  fi
  AUDIT=$(python3 - "${HOLD_SUMMARY}" "${GPU_FREQ_MHZ}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); tgt = int(sys.argv[2])
ok = bool(d.get("verdict_ok"))
print(f"target={tgt}MHz ul_median={d.get('under_load_median_mhz')}MHz "
      f"on_target={d.get('on_target_frac')} reapplies={d.get('reapplies')} ok={ok}")
sys.exit(0 if ok else 1)
PY
  ) && AUDIT_RC=0 || AUDIT_RC=$?
  echo "[dvfs] clock audit: ${AUDIT}"
  if [[ "${AUDIT_RC}" != "0" ]]; then
    echo "[dvfs] AUDIT FAILED — run is mislabelled; not writing clean summary." >&2
    echo '{"status":"CLOCK_NOT_HELD","gpu_freq_mhz":'"${GPU_FREQ_MHZ}"'}' \
      > "${RESULTS_DIR}/summary.json"
    exit 3
  fi
fi

# ── Collect summary ───────────────────────────────────────────────────────────
python3 - <<COLLECT
import json, socket
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path("${OUT_DIR}")
results_dir = Path("${RESULTS_DIR}")
metrics_path = out_dir / "run_exec_metrics.json"
freq_summary = out_dir / "gpu_freq" / "gpu_freq_hold_summary.json"

metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
freq_data = json.loads(freq_summary.read_text()) if freq_summary.exists() else {}

summary = {
    "arm_label": "${ARM_LABEL}",
    "model": "${MODEL}",
    "gpu_freq_mhz_target": int("${GPU_FREQ_MHZ}") if "${GPU_FREQ_MHZ}" else None,
    "gpu_freq_mhz_achieved": freq_data.get("under_load_median_mhz"),
    "clock_verdict_ok": freq_data.get("verdict_ok"),
    "tp_size": int("${TP_SIZE}"),
    "num_reqs": int("${NUM_REQS}"),
    "fix_input_length": int("${FIX_INPUT_LENGTH}"),
    "slurm_job_id": "${JOB_TAG}",
    "hostname": socket.gethostname(),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "status": "completed",
    "timing": metrics.get("timing"),
    "latency": metrics.get("latency"),
    "energy": metrics.get("energy"),
    "throughput": metrics.get("throughput"),
}
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[collect] wrote {results_dir}/summary.json")

t = summary.get("timing") or {}
lat = (summary.get("latency") or {}).get("ttft") or {}
en = summary.get("energy") or {}
print(f"  wall={t.get('wall_sec')}s  ttft_median={lat.get('median_ms')}ms  "
      f"energy={en.get('energy_excl_pause_j')}J  power={en.get('mean_power_excl_pause_w')}W")
COLLECT
