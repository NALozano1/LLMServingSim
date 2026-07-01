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
#   GPU_TYPE       — device type tag written to summary.json (e.g. v100, a100, h100)
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
GPU_TYPE="${GPU_TYPE:-}"          # device type tag written to summary.json

if [[ -z "${ARM_LABEL:-}" ]]; then
  if [[ -n "${GPU_FREQ_MHZ}" ]]; then
    ARM_LABEL="${GPU_FREQ_MHZ}mhz"
  else
    ARM_LABEL="uncapped"
  fi
fi

SCRATCH="${SCRATCH:-${ENGS_GLASS}/.llmsim/${JOB_TAG}}"
# Persistent Triton kernel cache shared across all jobs (avoids multi-worker race on first compile)
TRITON_PERSISTENT="${ENGS_GLASS}/.cache/triton"
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
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" \
  "${SCRATCH}/pip" "${OUT_DIR}" "${FREQ_META_DIR}" "${RESULTS_DIR}" \
  "${CAMPAIGN_DIR}/shared" \
  "${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}" \
  "${HF_CACHE_ROOT}/hub" "${HF_CACHE_ROOT}/.cache/huggingface" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${TRITON_PERSISTENT}"

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

# ── Preflight: ensure clock helper is usable if clock-locking is requested ────
# Fail clean here rather than silently running uncapped or accumulating reapply failures.
if [[ -n "${GPU_FREQ_MHZ:-}" ]] && ! sudo -n /usr/local/sbin/nvidia-smi-clocks --help >/dev/null 2>&1; then
  echo "[dvfs] FATAL: nvidia-smi-clocks helper unavailable on $(hostname) — cannot lock ${GPU_FREQ_MHZ}MHz on this device" >&2
  exit 2
fi

if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  # Restrict hold poller to inference GPUs only (0..TP_SIZE-1) so idle GPUs
  # don't trigger false off-target reapplies and distort the under-load median.
  _HOLD_GPUS="$(seq -s, 0 $((TP_SIZE - 1)))"
  CUDA_VISIBLE_DEVICES="${_HOLD_GPUS}" python3 "${GPU_FREQ_LOCK_PY}" hold --mhz "${GPU_FREQ_MHZ}" \
    --out-dir "${FREQ_META_DIR}" &
  HOLD_PID=$!
  echo "[dvfs] hold poller pid=${HOLD_PID} target=${GPU_FREQ_MHZ} MHz gpus=${_HOLD_GPUS}"
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
  -B "${TRITON_PERSISTENT}:/scratch/triton" \
  -B /dev/shm:/dev/shm \
  --pwd "${REPO_ROOT}" \
  --env "HOME=${SCRATCH}/h" \
  --env "TMPDIR=${SCRATCH}/t" \
  --env "VLLM_CACHE_ROOT=${SCRATCH}/v" \
  --env "TRITON_CACHE_DIR=/scratch/triton" \
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
import json, socket, statistics
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path("${OUT_DIR}")
results_dir = Path("${RESULTS_DIR}")
freq_summary = out_dir / "gpu_freq" / "gpu_freq_hold_summary.json"
meta_path    = out_dir / "meta.json"
reqs_path    = out_dir / "requests.jsonl"
power_path   = out_dir / "gpu_power" / "bench.jsonl"

freq_data = json.loads(freq_summary.read_text()) if freq_summary.exists() else {}
meta      = json.loads(meta_path.read_text())    if meta_path.exists()     else {}

# ── Timing from meta.json ──────────────────────────────────────────────────
wall_sec = None
if meta.get("started_at") and meta.get("finished_at"):
    from datetime import datetime as _dt
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    t0 = _dt.strptime(meta["started_at"], fmt)
    t1 = _dt.strptime(meta["finished_at"], fmt)
    wall_sec = (t1 - t0).total_seconds()

timing = {"wall_sec": wall_sec, "exec_sec": wall_sec, "pause_sec": 0.0}

# ── Latency from requests.jsonl ────────────────────────────────────────────
# TTFT = (first_token_ts - queued_ts) * 1000 ms  (timestamps are monotonic seconds)
ttft_ms_list = []
if reqs_path.exists():
    for line in reqs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("first_token_ts") is not None and r.get("queued_ts") is not None:
            ttft_ms_list.append((r["first_token_ts"] - r["queued_ts"]) * 1000.0)

def _pct(xs, p):
    if not xs: return None
    s = sorted(xs)
    idx = max(0, min(len(s)-1, int(len(s)*p/100)))
    return round(s[idx], 3)

ttft = {
    "count": len(ttft_ms_list),
    "mean_ms":   round(statistics.mean(ttft_ms_list),   3) if ttft_ms_list else None,
    "median_ms": round(statistics.median(ttft_ms_list), 3) if ttft_ms_list else None,
    "p90_ms":    _pct(ttft_ms_list, 90),
    "p99_ms":    _pct(ttft_ms_list, 99),
}
latency = {"ttft": ttft}

# ── Energy from gpu_power/bench.jsonl (sum across all TP GPUs) ────────────────
# The sampler (profiler/core/gpu_power.py) writes one JSONL record per poll tick
# with a "gpus" list containing every GPU visible via CUDA_VISIBLE_DEVICES.
# Slurm's gres plugin sets CUDA_VISIBLE_DEVICES to exactly the job's allocated
# GPUs and that env is inherited through Apptainer --nv, so summing all "gpus"
# entries in each sample is safe and correct.
#
# CUDA_VISIBLE_DEVICES remapping note: on a multi-GPU node Slurm may assign
# non-zero-based indices (e.g. GPUs 2-5 on an 8-GPU node).  We therefore sum
# ALL entries from the samples rather than filtering to indices 0..TP_SIZE-1,
# and report per-GPU values keyed by the actual system GPU index.
#
# Backward-compatible: at TP_SIZE=1 energy_j_total == the old energy_j (GPU-0 only).
tp_size = int("${TP_SIZE}")
energy_j_total   = None
energy_j_gpu0    = None   # GPU index 0 specifically — legacy continuity field
energy_j_per_gpu = []     # list ordered by sorted GPU index
mean_power_w     = None

if power_path.exists():
    samples = [json.loads(l) for l in power_path.read_text().splitlines() if l.strip()]

    # Build per-GPU and summed total time-series for trapezoidal integration.
    per_gpu_ts = {}  # {gpu_index: [(wall_ts, power_w), ...]}
    total_ts   = []  # [(wall_ts, sum_of_all_gpu_powers), ...]

    for s in samples:
        wt   = s.get("wall_ts")
        gpus = s.get("gpus", [])
        if wt is None:
            continue
        valid = [(g["index"], float(g["power_w"]))
                 for g in gpus
                 if g.get("index") is not None and g.get("power_w") is not None]
        if not valid:
            continue
        total_ts.append((float(wt), sum(pw for _, pw in valid)))
        for idx, pw in valid:
            per_gpu_ts.setdefault(idx, []).append((float(wt), pw))

    def _trapz(pairs):
        """Trapezoidal integration of [(t, power_w), ...] pairs."""
        if len(pairs) < 2:
            return None
        pairs = sorted(pairs)
        return sum(
            (pairs[i][1] + pairs[i+1][1]) / 2.0 * (pairs[i+1][0] - pairs[i][0])
            for i in range(len(pairs) - 1)
            if pairs[i+1][0] > pairs[i][0]
        )

    # Total energy (sum of all visible GPUs).
    total_e = _trapz(total_ts)
    energy_j_total = round(total_e, 3) if total_e is not None else None

    # Mean total power = total_energy / total_duration.
    if energy_j_total is not None and len(total_ts) >= 2:
        t_sorted = sorted(total_ts)
        dur = t_sorted[-1][0] - t_sorted[0][0]
        if dur > 0:
            mean_power_w = round(energy_j_total / dur, 3)

    # Per-GPU energies (sorted by actual GPU index).
    sorted_idxs = sorted(per_gpu_ts)
    per_gpu_energies = {}
    for _idx in sorted_idxs:
        _e = _trapz(per_gpu_ts[_idx])
        per_gpu_energies[_idx] = round(_e, 3) if _e is not None else None
    energy_j_per_gpu = [per_gpu_energies[i] for i in sorted_idxs]
    energy_j_gpu0    = per_gpu_energies.get(0)  # legacy: GPU index 0 specifically

energy = {
    # energy_j = TOTAL across all TP GPUs (was GPU-index-0-only before this fix).
    # At TP_SIZE=1 the value is identical to the old single-GPU result.
    "energy_j":               energy_j_total,
    "energy_j_total":         energy_j_total,
    "energy_j_per_gpu":       energy_j_per_gpu,
    "energy_j_gpu0":          energy_j_gpu0,    # legacy: GPU index 0 only
    "energy_excl_pause_j":    energy_j_total,   # no pause-exclusion in validation runs
    "mean_power_w":           mean_power_w,
    "mean_power_excl_pause_w": mean_power_w,
}

summary = {
    "arm_label": "${ARM_LABEL}",
    "model": "${MODEL}",
    "gpu_type": "${GPU_TYPE}" if "${GPU_TYPE}" else None,
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
    "timing": timing,
    "latency": latency,
    "energy": energy,
    "throughput": None,
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
