#!/bin/bash
# run_moe_table_ep_decouple_L40S_smoke.sbatch
#
# PHASE 0 — L40S clock-lock smoke test.
#
# Submits a SINGLE 1-GPU L40S job to verify that clocks lock at a mid-range
# frequency (~1500 MHz) under a real MoE profiler load.  Reports hold-poller
# verdict_ok + achieved_mhz.  Run this FIRST; only proceed to the full sweep
# once lockable freq range is established.
#
# L40S specs:
#   - Architecture: Ada Lovelace (sm89), 48 GB GDDR6X
#   - Base clock: ~1065 MHz   Boost cap: ~2520 MHz
#   - Clock helper: /usr/local/sbin/nvidia-smi-clocks (same as V100/H100)
#   - Hold strategy: unconditional re-apply every tick (mirrors H100 pattern;
#     Ada may not fire the V100 power-gated threshold at single-layer loads)
#
# Usage (NEVER --reservation):
#   sbatch -M htc --clusters=htc --account=engs-glass --qos=priority \
#          --partition=short --gres=gpu:l40s:1 \
#          --nodes=1 --cpus-per-task=8 --mem=48G --time=01:00:00 \
#          profiler/jobs/run_moe_table_ep_decouple_L40S_smoke.sbatch \
#          [MODEL_HF_ID] [FREQ_MHZ] [TP_LIST]
#
# Args:
#   $1 MODEL_HF_ID  — default: "Qwen/Qwen3-30B-A3B-Instruct-2507"
#   $2 FREQ_MHZ     — target frequency, default: 1500
#   $3 TP_LIST      — default: 1,16,32 (smoke: just tp1 + two high-TP)
#
# Output: profiler/perf_ep_decouple/L40S_<FREQ>MHz/...
# Log:    profiler/jobs/logs_table_held/ep_dec_l40s_<JOBID>.log
#
# NEVER use --reservation.
#
#SBATCH --job-name=ep_dec_l40s_smoke
#SBATCH --output=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/logs_table_held/ep_dec_l40s-%j.out
#SBATCH --error=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/logs_table_held/ep_dec_l40s-%j.err

set -uo pipefail

MODEL_HF_ID="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
FREQ_MHZ="${2:-1500}"
TP_LIST="${3:-1,16,32}"
# tp1 is required by replicate_tp_stable (tp-invariant dense/attention layers are copied
# FROM tp1); auto-include it so a caller cannot omit it and hit a late replication failure.
case ",${TP_LIST}," in *",1,"*) ;; *) TP_LIST="1,${TP_LIST}" ;; esac

HW_LABEL="L40S_${FREQ_MHZ}MHz"

ENGS_GLASS="/data/engs-glass/engs2950"
LLMSIM_ROOT="${ENGS_GLASS}/DVFS-MoE/LLMServingSim"
HF_CACHE_ROOT="${ENGS_GLASS}/infra/hf_cache"
CLOCKS_BIN="/usr/local/sbin/nvidia-smi-clocks"
VLLM_IMAGE="${ENGS_GLASS}/DVFS-MoE/moe.sif"

JOB_TAG="${SLURM_JOB_ID:-local}"
SCRATCH="${ENGS_GLASS}/.llmsim/${JOB_TAG}_ep_dec_l40s"

# CRITICAL: never write EP-decouple emulations into profiler/perf/.
# That path is reserved for full-MoE characterization tables.
PERF_OUT="${LLMSIM_ROOT}/profiler/perf_ep_decouple"
if [[ "${PERF_OUT%/}" == */profiler/perf ]] || [[ "${PERF_OUT}" == */profiler/perf_table_held ]] || [[ "${PERF_OUT}" == */profiler/perf_table_held/* ]]; then
  echo "FATAL: PERF_OUT=${PERF_OUT} must not be a full-MoE tree (use perf_ep_decouple)." >&2
  exit 2
fi
HOLD_DIR="${LLMSIM_ROOT}/profiler/jobs/logs_table_held/hold_${JOB_TAG}"

MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=256
MEASUREMENT_ITERS=15
HOLD_INTERVAL_SEC=1
HOLD_TOLERANCE_MHZ=30
HOLD_MAX_SEC=3600   # 1 h ceiling (smoke)

# Physical GPU index from Slurm.
PHYSICAL_GPU_IDX="${SLURM_JOB_GPUS:-0}"
PHYSICAL_GPU_IDX="${PHYSICAL_GPU_IDX%%,*}"

mkdir -p \
  "${SCRATCH}/t" "${SCRATCH}/h" "${SCRATCH}/v" "${SCRATCH}/triton" \
  "${SCRATCH}/torch" "${SCRATCH}/pip" \
  "${PERF_OUT}" "${HOLD_DIR}" \
  "${ENGS_GLASS}/.apptainer_cache/cache" \
  "${LLMSIM_ROOT}/profiler/jobs/apptainer_tmp/${JOB_TAG}" \
  "${LLMSIM_ROOT}/profiler/jobs/logs_table_held"

export HOME="${SCRATCH}/h"
export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${LLMSIM_ROOT}/profiler/jobs/apptainer_tmp/${JOB_TAG}"
export HF_TOKEN="$(cat "${ENGS_GLASS}/.secrets/hf_token" 2>/dev/null || echo "")"

echo "================================================================"
echo "ep_dec_l40s_smoke  Node=$(hostname)  Job=${JOB_TAG}"
echo "Model:    ${MODEL_HF_ID}"
echo "HW:       ${HW_LABEL}"
echo "Target:   ${FREQ_MHZ} MHz  (L40S base~1065, boost~2520)"
echo "TP_LIST:  ${TP_LIST}  (EP_DECOUPLE=1)"
echo "PerfOut:  ${PERF_OUT}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "Physical GPU index: ${PHYSICAL_GPU_IDX}"
echo ""
echo "PHASE 0 SMOKE: verifying L40S clock lock ONLY."
echo "  If verdict_ok=false → STOP, do not launch full sweep."
echo "================================================================"

nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,clocks.sm,clocks.max.sm --format=csv,noheader

# ── L40S clock lock: apply unconditionally at startup ────────────────────────
# Strategy mirrors H100: apply at startup + unconditional re-apply every tick.
# L40S (Ada Lovelace) single-layer MoE draws < power-gate threshold, so
# power-conditioned re-apply (V100 pattern) would not fire reliably.
CLOCKS_USE_ID=0
if sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" "${FREQ_MHZ}" "${FREQ_MHZ}" 2>/dev/null; then
    echo "[L40S] Clock lock APPLIED (--id form) at ${FREQ_MHZ} MHz"
    CLOCKS_USE_ID=1
elif sudo -n "${CLOCKS_BIN}" "${FREQ_MHZ}" "${FREQ_MHZ}" 2>/dev/null; then
    echo "[L40S] Clock lock APPLIED (no-id / batch-all form) at ${FREQ_MHZ} MHz"
    CLOCKS_USE_ID=0
else
    SUDO_RC=$?
    CLOCKS_USE_ID=0
    echo "[L40S] WARNING: sudo test rc=${SUDO_RC} — both lock forms failed; poller will keep retrying."
fi
echo "[L40S] CLOCKS_USE_ID=${CLOCKS_USE_ID}"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || true

HOLD_PID=""
PROFILER_PID=""

cleanup() {
    echo "[L40S] cleanup: stopping hold poller ..."
    [[ -n "${HOLD_PID}" ]] && {
        touch "${HOLD_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
        sleep 2; kill "${HOLD_PID}" 2>/dev/null || true; wait "${HOLD_PID}" 2>/dev/null || true
    }
    echo "[L40S] cleanup: resetting GPU clocks (physical idx=${PHYSICAL_GPU_IDX}, use_id=${CLOCKS_USE_ID:-0}) ..."
    if [[ "${CLOCKS_USE_ID:-0}" == "1" ]]; then
        sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" --reset 2>/dev/null || true
    else
        sudo -n "${CLOCKS_BIN}" --reset 2>/dev/null || true
    fi
    rb=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null || echo "?")
    echo "[L40S]   post-reset max SM clock: ${rb}"
    [[ -n "${PROFILER_PID}" ]] && kill "${PROFILER_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "[L40S] SIGTERM -> cleanup"; exit 143' TERM
trap 'echo "[L40S] SIGINT  -> cleanup"; exit 130' INT
trap 'echo "[L40S] SIGHUP  -> cleanup"; exit 129' HUP

unset SLURM_JOB_ACCOUNT 2>/dev/null || true

# ── Step 1: Start hold poller ─────────────────────────────────────────────────
echo ""
echo "=== [1] Starting hold poller (target=${FREQ_MHZ} MHz, unconditional re-apply) ==="

python3 - <<HOLD_PYEOF > "${HOLD_DIR}/hold.log" 2>&1 &
import json, os, subprocess, time, statistics, pathlib

cap_mhz       = ${FREQ_MHZ}
# L40S idle power varies; use 100W as heavy-load threshold (Ada draws ~15W idle).
# Also track "active" series at 30W as fallback (above GPU idle baseline).
power_thr_w   = 100
active_thr_w  = 30
interval_sec  = ${HOLD_INTERVAL_SEC}
tolerance_mhz = ${HOLD_TOLERANCE_MHZ}
max_sec       = ${HOLD_MAX_SEC}
clocks_bin    = "${CLOCKS_BIN}"
physical_idx  = "${PHYSICAL_GPU_IDX}"
use_id        = ${CLOCKS_USE_ID}
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
        if use_id:
            cmd = ["sudo", "-n", clocks_bin, "--id", str(idx), str(mhz), str(mhz)]
        else:
            cmd = ["sudo", "-n", clocks_bin, str(mhz), str(mhz)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False

# Apply unconditionally at startup.
apply_lock(physical_idx, cap_mhz)

t0 = time.time()
under_load_clocks = []
active_clocks = []
ticks = 0
reapplies = 0

while not stop_path.exists() and (time.time() - t0) < max_sec:
    gpus = smi_sample()
    rec = {"t": round(time.time() - t0, 1), "gpus": gpus}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    # Unconditional re-apply every tick to counteract Ada boost/power-gating.
    if apply_lock(physical_idx, cap_mhz):
        reapplies += 1
    for g in gpus:
        pw = g.get("power_w", 0)
        clk = int(g["graphics_mhz"])
        if pw >= power_thr_w:
            under_load_clocks.append(clk)
        if pw >= active_thr_w:
            active_clocks.append(clk)
    ticks += 1
    time.sleep(interval_sec)

# Verdict: prefer heavy-load samples, fall back to active (non-idle).
ul_series = under_load_clocks if under_load_clocks else active_clocks
ul_med = int(statistics.median(ul_series)) if ul_series else None
verdict_ok = (ul_med is not None and abs(ul_med - cap_mhz) <= tolerance_mhz)
summary = {
    "target_mhz": cap_mhz, "tolerance_mhz": tolerance_mhz,
    "under_load_median_mhz": ul_med,
    "under_load_samples": len(under_load_clocks),
    "active_samples": len(active_clocks),
    "verdict_source": "under_load" if under_load_clocks else "active",
    "verdict_ok": verdict_ok, "ticks": ticks, "reapplies": reapplies,
    "physical_gpu_idx": physical_idx,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"hold poller done: ul_median={ul_med} verdict_ok={verdict_ok} reapplies={reapplies}")
HOLD_PYEOF

HOLD_PID=$!
echo "[L40S] Hold poller pid=${HOLD_PID}"
sleep 2

# ── Step 2: Launch profiler in container ──────────────────────────────────────
echo ""
echo "=== [2] Launching profiler (PROFILER_EP_DECOUPLE=1, TP_LIST=${TP_LIST}) ==="

apptainer exec --cleanenv --nv \
  -B "${LLMSIM_ROOT}:${LLMSIM_ROOT}" \
  -B "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
  -B "${SCRATCH}:${SCRATCH}" \
  -B "${PERF_OUT}:${PERF_OUT}" \
  -B /dev/shm:/dev/shm \
  --pwd "${LLMSIM_ROOT}" \
  --env "HOME=${SCRATCH}/h" \
  --env "TMPDIR=${SCRATCH}/t" \
  --env "VLLM_CACHE_ROOT=${SCRATCH}/v" \
  --env "TRITON_CACHE_DIR=${SCRATCH}/triton" \
  --env "TORCH_HOME=${SCRATCH}/torch" \
  --env "PIP_CACHE_DIR=${SCRATCH}/pip" \
  --env "HF_HOME=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HUGGINGFACE_HUB_CACHE=${HF_CACHE_ROOT}/hub" \
  --env "HF_CACHE_DIR=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "TRANSFORMERS_CACHE=${HF_CACHE_ROOT}/.cache/huggingface" \
  --env "HF_HUB_OFFLINE=1" \
  --env "HF_TOKEN=${HF_TOKEN}" \
  --env "CUDA_VISIBLE_DEVICES=0" \
  --env "PROFILER_GPU_POWER=1" \
  --env "PROFILER_GPU_POWER_INTERVAL_MS=50" \
  --env "PROFILER_MOE_FUSED=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "PROFILER_EP_DECOUPLE=1" \
  "${VLLM_IMAGE}" \
  bash -c "pip install -q datasets matplotlib 2>/dev/null || true; exec python3 -u -m profiler profile --skip-skew \
    '${MODEL_HF_ID}' \
    --hardware '${HW_LABEL}' \
    --tp ${TP_LIST} \
    --dtype float16 \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --measurement-iterations ${MEASUREMENT_ITERS} \
    --num-hidden-layers 1 \
    --only-moe \
    --force \
    --out-root '${PERF_OUT}' \
    --verbose" \
  > "${LLMSIM_ROOT}/profiler/jobs/logs_table_held/ep_dec_l40s_${JOB_TAG}.log" 2>&1 &

PROFILER_PID=$!
echo "[L40S] Profiler pid=${PROFILER_PID}"

# ── Step 3: Wait for profiler ─────────────────────────────────────────────────
echo ""
echo "=== [3] Waiting for profiler ==="
PROFILER_EXIT=0
wait "${PROFILER_PID}" || PROFILER_EXIT=$?
PROFILER_PID=""
echo "[L40S] Profiler exit code: ${PROFILER_EXIT}"

touch "${HOLD_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
sleep 3
wait "${HOLD_PID}" 2>/dev/null || true
HOLD_PID=""

# ── Step 4: Hold poller verdict ───────────────────────────────────────────────
echo ""
echo "=== [4] Hold poller verdict ==="
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
source  = s.get("verdict_source", "?")
print(f"[verdict] verdict_ok={verdict}  ul_median={median} MHz  reapplies={reapply}  source={source}")
if not verdict:
    print("PHASE 0 FAIL: hold poller did NOT confirm ${FREQ_MHZ} MHz under load.")
    print("  If ul_median >> ${FREQ_MHZ}, L40S boost is overriding the lock.")
    print("  Try a lower target freq, or verify sudo helper works on L40S nodes.")
    sys.exit(1)
else:
    print("PHASE 0 PASS: hold poller confirms ${FREQ_MHZ} MHz held under load.")
    print("  => L40S clock lock is viable. Proceed to Phase 1 full sweep.")
PYEOF
fi

# ── Step 5: Output CSVs ───────────────────────────────────────────────────────
echo ""
echo "=== [5] Output CSVs ==="
BASE_PATH="${PERF_OUT}/${HW_LABEL}/${MODEL_HF_ID}/fp16"

for TP in $(echo "${TP_LIST}" | tr ',' ' '); do
    CSV="${BASE_PATH}/tp${TP}/moe.csv"
    if [[ -f "${CSV}" ]]; then
        echo ""
        echo "tp${TP}/moe.csv FOUND: ${CSV}"
        wc -l "${CSV}"
        head -3 "${CSV}"
        python3 - <<PYEOF
import csv, sys
path = "${CSV}"
rows = list(csv.DictReader(open(path)))
if not rows:
    print("  ERROR: empty CSV!")
    sys.exit(1)
ae_vals = sorted(set(int(r['activated_experts']) for r in rows))
tok_vals = sorted(set(int(r['tokens']) for r in rows))
achieved = [r.get('achieved_mhz','') for r in rows if r.get('achieved_mhz','')]
achieved_nums = [float(x) for x in achieved if x]
print(f"  ae values:  {ae_vals}")
print(f"  token grid: {tok_vals[:6]}{'...' if len(tok_vals)>6 else ''}")
if achieved_nums:
    med = sorted(achieved_nums)[len(achieved_nums)//2]
    ok  = abs(med - ${FREQ_MHZ}) <= 30
    print(f"  achieved_mhz (median): {med:.0f}  target: ${FREQ_MHZ}  clock_clean: {ok}")
    if not ok:
        print("  WARNING: achieved_mhz out of tolerance — L40S boost cap may exceed ${FREQ_MHZ}!")
else:
    print("  achieved_mhz: not in table (older schema)")
PYEOF
    else
        echo "tp${TP}/moe.csv MISSING: ${CSV}"
    fi
done

echo ""
echo "All moe.csv under ${PERF_OUT}/${HW_LABEL}:"
find "${PERF_OUT}/${HW_LABEL}" -name "moe.csv" 2>/dev/null | sort

echo ""
echo "=== PHASE 0 smoke L40S @ ${FREQ_MHZ}MHz DONE (profiler rc=${PROFILER_EXIT}) ==="
echo "Review verdict_ok and achieved_mhz above before launching Phase 1."
exit "${PROFILER_EXIT}"
