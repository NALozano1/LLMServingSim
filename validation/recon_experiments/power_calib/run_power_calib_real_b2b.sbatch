#!/bin/bash
# run_power_calib_real_b2b.sbatch — Host-free gapfree smoke (inplace + CUDA-event ± graph).
#
# PURPOSE
# -------
# Same clock-hold + apptainer harness as run_power_calib.sbatch, but runs
# power_calib_real_b2b.py: inplace=True, CUDA-event latency, optional CUDA Graph.
# Writes a separate CSV (does not overwrite legacy power_calib_<DEV>.csv).
#
# Usage:
#   sbatch -M htc --clusters=htc --account=engs-glass \
#          --partition=short --gres=gpu:a100:1 \
#          --mail-user=alex.lozano@eng.ox.ac.uk --mail-type=BEGIN,END,FAIL \
#          --nodes=1 --cpus-per-task=4 --mem=32G --time=00:45:00 \
#          validation/recon_experiments/power_calib/run_power_calib_real_b2b.sbatch \
#          <FREQ_MHZ> [BATCHES] [AE] [MODEL_HF_ID]
#
# Args:
#   $1 FREQ_MHZ    — target clock (default 900)
#   $2 BATCHES     — comma-sep (default "8,32,512,8192" smoke)
#   $3 AE          — activated experts (default: model top_k)
#   $4 MODEL_HF_ID — HF model id (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#
#SBATCH --job-name=real_b2b_smoke
#SBATCH --clusters=htc
#SBATCH --account=engs-glass
#SBATCH --mail-user=alex.lozano@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/validation/recon_experiments/power_calib/logs/real_b2b-%j.out
#SBATCH --error=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/validation/recon_experiments/power_calib/logs/real_b2b-%j.err

set -uo pipefail

# ── Parse args ───────────────────────────────────────────────────────────────
FREQ_MHZ="${1:-900}"
BATCHES="${2:-8,32,512,8192}"
AE="${3:-}"
MODEL_HF_ID="${4:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE=/data/engs-glass/engs2950
REPO=$BASE/DVFS-MoE/LLMServingSim
SIF=$BASE/DVFS-MoE/moe.sif
CALIB_DIR=$REPO/validation/recon_experiments/power_calib
SCRIPT=$CALIB_DIR/power_calib_real_b2b.py
# Auto-detect device so V100/L40S/A100/H100 runs write to separate CSVs (no collision).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
case "$GPU_NAME" in
  *V100*) DEV=V100 ;;
  *L40S*|*L40*) DEV=L40S ;;
  *A100*) DEV=A100 ;;
  *H100*) DEV=H100 ;;
  *)      DEV=$(echo "$GPU_NAME" | tr -cd 'A-Za-z0-9' | cut -c1-10) ;;
esac
OUT_CSV=$CALIB_DIR/real_b2b_${DEV}_${FREQ_MHZ}MHz_j${SLURM_JOB_ID:-local}.csv

HF_CACHE_ROOT=$BASE/infra/hf_cache
CLOCKS_BIN=/usr/local/sbin/nvidia-smi-clocks

JOB_TAG="${SLURM_JOB_ID:-local}"
SCRATCH=$BASE/.llmsim/real_b2b_${JOB_TAG}
HOLD_DIR=$CALIB_DIR/logs/hold_real_b2b_${JOB_TAG}

HOLD_INTERVAL_SEC=1
HOLD_TOLERANCE_MHZ=30
HOLD_MAX_SEC=7200
HOLD_POWER_THR_W=55

# Physical GPU index (matches system helper's root-side view)
PHYSICAL_GPU_IDX="${SLURM_JOB_GPUS:-0}"
PHYSICAL_GPU_IDX="${PHYSICAL_GPU_IDX%%,*}"

HF_TOKEN="$(cat "${BASE}/.secrets/hf_token" 2>/dev/null || echo "")"

mkdir -p \
  "${SCRATCH}/h" "${SCRATCH}/t" "${SCRATCH}/v" \
  "${SCRATCH}/triton" "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${HOLD_DIR}" \
  "${CALIB_DIR}/logs" \
  "${BASE}/.apptainer_cache/cache" \
  "${REPO}/profiler/jobs/apptainer_tmp/${JOB_TAG}"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${BASE}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${REPO}/profiler/jobs/apptainer_tmp/${JOB_TAG}"

echo "================================================================"
echo "real_b2b_smoke  Node=$(hostname)  Job=${JOB_TAG}"
echo "  DEV=${DEV}  FREQ=${FREQ_MHZ}MHz  BATCHES=${BATCHES}  AE=${AE:-<model_top_k>}"
echo "  MODEL=${MODEL_HF_ID}"
echo "  SCRIPT=${SCRIPT}"
echo "  OUT_CSV=${OUT_CSV}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "  Physical GPU index for helper: ${PHYSICAL_GPU_IDX}"
echo "  METHOD: REAL gapfree — inplace=True + CUDA-event (± CUDA Graph)"
echo "          NVML sampled at ~25ms in background thread"
echo "================================================================"

nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,clocks.sm,clocks.max.sm \
           --format=csv,noheader || true

# Sanity-check sudo access to system clock helper.
if sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" "${FREQ_MHZ}" "${FREQ_MHZ}" 2>/dev/null; then
    echo "[calib] Initial sudo test OK — resetting before hold poller takes over"
    sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" --reset 2>/dev/null || true
else
    SUDO_RC=$?
    echo "[calib] WARNING: sudo test rc=${SUDO_RC} — hold poller will retry."
fi

HOLD_PID=""
CALIB_PID=""

cleanup() {
    echo "[calib] cleanup: stopping hold poller ..."
    [[ -n "${HOLD_PID}" ]] && {
        touch "${HOLD_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
        sleep 2
        kill "${HOLD_PID}" 2>/dev/null || true
        wait "${HOLD_PID}" 2>/dev/null || true
    }
    echo "[calib] cleanup: resetting GPU clocks (physical idx=${PHYSICAL_GPU_IDX}) ..."
    sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" --reset 2>/dev/null || true
    rb=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null || echo "?")
    echo "[calib]   post-reset max SM clock: ${rb}"
    [[ -n "${CALIB_PID}" ]] && kill "${CALIB_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "[calib] SIGTERM → cleanup"; exit 143' TERM
trap 'echo "[calib] SIGINT  → cleanup"; exit 130' INT
trap 'echo "[calib] SIGHUP  → cleanup"; exit 129' HUP

unset SLURM_JOB_ACCOUNT 2>/dev/null || true

# ── Step 1: Start HOST hold poller ───────────────────────────────────────────
echo ""
echo "=== [1] Starting HOST hold poller (target=${FREQ_MHZ}MHz, power_thr=${HOLD_POWER_THR_W}W) ==="

python3 - <<HOLD_PYEOF > "${HOLD_DIR}/hold.log" 2>&1 &
import json, os, subprocess, time, statistics, pathlib

cap_mhz       = ${FREQ_MHZ}
power_thr_w   = ${HOLD_POWER_THR_W}
interval_sec  = ${HOLD_INTERVAL_SEC}
tolerance_mhz = ${HOLD_TOLERANCE_MHZ}
max_sec       = ${HOLD_MAX_SEC}
clocks_bin    = "${CLOCKS_BIN}"
physical_idx  = "${PHYSICAL_GPU_IDX}"
hold_dir      = pathlib.Path("${HOLD_DIR}")
stop_path     = hold_dir / "gpu_freq_hold.stop"
log_path      = hold_dir / "gpu_freq_hold.jsonl"
summary_path  = hold_dir / "gpu_freq_hold_summary.json"
hold_dir.mkdir(parents=True, exist_ok=True)
if stop_path.exists():
    stop_path.unlink()

def smi_sample():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,clocks.current.graphics,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        out = []
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 4: continue
            try:
                out.append({"index": int(p[0]), "graphics_mhz": float(p[1]),
                            "util_pct": float(p[2]), "power_w": float(p[3])})
            except Exception:
                pass
        return out
    except Exception:
        return []

def apply_lock(idx, mhz):
    try:
        r = subprocess.run(
            ["sudo", "-n", clocks_bin, "--id", str(idx), str(mhz), str(mhz)],
            capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False

t0 = time.time()
under_load_clocks = []
ticks = 0
reapplies = 0

while not stop_path.exists() and (time.time() - t0) < max_sec:
    gpus = smi_sample()
    rec = {"t": round(time.time() - t0, 1), "gpus": gpus}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    for g in gpus:
        if g.get("power_w", 0) >= power_thr_w:
            under_load_clocks.append(int(g["graphics_mhz"]))
            if abs(g["graphics_mhz"] - cap_mhz) > tolerance_mhz:
                if apply_lock(physical_idx, cap_mhz):
                    reapplies += 1
    ticks += 1
    time.sleep(interval_sec)

ul_med = int(statistics.median(under_load_clocks)) if under_load_clocks else None
verdict_ok = (ul_med is not None and abs(ul_med - cap_mhz) <= tolerance_mhz)
summary = {
    "target_mhz": cap_mhz, "tolerance_mhz": tolerance_mhz,
    "under_load_median_mhz": ul_med, "under_load_samples": len(under_load_clocks),
    "verdict_ok": verdict_ok, "ticks": ticks, "reapplies": reapplies,
    "physical_gpu_idx": physical_idx,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"hold poller done: ul_median={ul_med} verdict_ok={verdict_ok} reapplies={reapplies}")
HOLD_PYEOF

HOLD_PID=$!
echo "[calib] Hold poller pid=${HOLD_PID} (HOST — NOT inside container)"
sleep 2

# ── Step 2: Apply initial clock lock and verify ───────────────────────────────
echo ""
echo "=== [2] Applying initial clock lock to ${FREQ_MHZ}MHz ==="
sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" "${FREQ_MHZ}" "${FREQ_MHZ}" || true
sleep 1
nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits || true

# ── Step 3: Run power_calib_real_b2b.py inside container ──────────────────────
echo ""
echo "=== [3] Running power_calib_real_b2b.py inside container ==="
echo "        MODEL=${MODEL_HF_ID}  BATCHES=${BATCHES}  AE=${AE:-<model_top_k>}  FREQ=${FREQ_MHZ}MHz"

# Build optional --ae flag (omit entirely if AE is empty so py defaults to top_k)
AE_ARG=""
if [[ -n "${AE}" ]]; then
    AE_ARG="--ae ${AE}"
fi

apptainer exec --cleanenv --nv \
  -B "${BASE}:${BASE}" \
  -B "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  -B "${SCRATCH}:${SCRATCH}" \
  -B /dev/shm:/dev/shm \
  --pwd "${CALIB_DIR}" \
  --env "HOME=${SCRATCH}/h" \
  --env "TMPDIR=${SCRATCH}/t" \
  --env "VLLM_CACHE_ROOT=${SCRATCH}/v" \
  --env "TRITON_CACHE_DIR=${SCRATCH}/triton" \
  --env "TORCH_HOME=${SCRATCH}/torch" \
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_HUB_OFFLINE=1" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --env "PYTHONUNBUFFERED=1" \
  --env "VLLM_ENABLE_V1_MULTIPROCESSING=0" \
  --env "PYTHONPATH=${CALIB_DIR}:${PYTHONPATH:-}" \
  "${SIF}" \
  python3 -u "${SCRIPT}" \
    --model       "${MODEL_HF_ID}" \
    --freq        "${FREQ_MHZ}" \
    --batches     "${BATCHES}" \
    ${AE_ARG} \
    --window      9 \
    --chunk       1.0 \
    --ramp        1.5 \
    --poll-ms     25 \
    --out-csv     "${OUT_CSV}" \
  &

CALIB_PID=$!
echo "[calib] real_b2b pid=${CALIB_PID}"
wait "${CALIB_PID}"
CALIB_RC=$?
CALIB_PID=""
echo "[calib] real_b2b exit code: ${CALIB_RC}"

# ── Step 4: Stop hold poller ──────────────────────────────────────────────────
echo ""
echo "=== [4] Stopping hold poller ==="
touch "${HOLD_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
sleep 3
wait "${HOLD_PID}" 2>/dev/null || true
HOLD_PID=""

# ── Step 5: Hold poller verdict ───────────────────────────────────────────────
echo ""
echo "=== [5] Hold poller verdict ==="
HOLD_SUMMARY="${HOLD_DIR}/gpu_freq_hold_summary.json"
if [[ -f "${HOLD_SUMMARY}" ]]; then
    cat "${HOLD_SUMMARY}"
    python3 - <<PYEOF
import json, sys
with open("${HOLD_SUMMARY}") as f:
    s = json.load(f)
verdict = s.get("verdict_ok", False)
median  = s.get("under_load_median_mhz", "?")
reapply = s.get("reapplies", 0)
print(f"[verdict] verdict_ok={verdict}  ul_median={median} MHz  reapplies={reapply}")
if not verdict:
    print("WARNING: hold poller did NOT confirm ${FREQ_MHZ}MHz under load. Results INVALID.")
    sys.exit(1)
else:
    print("PASS: hold poller confirms ${FREQ_MHZ}MHz held under load.")
PYEOF
else
    echo "[calib] WARNING: hold summary not found at ${HOLD_SUMMARY}"
fi

# ── Step 6: Output summary ────────────────────────────────────────────────────
echo ""
echo "=== [6] Output ==="
if [[ -f "${OUT_CSV}" ]]; then
    echo "Results CSV: ${OUT_CSV}"
    wc -l "${OUT_CSV}"
    echo "--- header + rows for this freq (${FREQ_MHZ}MHz) / model ---"
    head -1 "${OUT_CSV}"
    grep ",${FREQ_MHZ}," "${OUT_CSV}" | grep "${MODEL_HF_ID}" || true
else
    echo "WARNING: result CSV not found at ${OUT_CSV}"
fi

echo ""
echo "=== real_b2b_smoke DONE ==="
echo "    DEV=${DEV}  MODEL=${MODEL_HF_ID}  FREQ=${FREQ_MHZ}MHz  BATCHES=${BATCHES}  calib rc=${CALIB_RC}"
exit "${CALIB_RC}"
