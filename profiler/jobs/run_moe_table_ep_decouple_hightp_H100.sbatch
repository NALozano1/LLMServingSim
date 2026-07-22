#!/bin/bash
# run_moe_table_ep_decouple_hightp_H100.sbatch
#
# Capture EP-decouple moe.csv tables for high EP degrees (tp16, tp32, tp64, tp128)
# that cannot run real tensor-parallel on ARC (max 8 GPUs), on an H100 PCIe node.
#
# Mechanism: PROFILER_EP_DECOUPLE=1 shards experts on a single GPU without
# sharding attention.  For Qwen3-30B-A3B (128 experts):
#   tp16  → 128/16  = 8 experts/rank, top_k = max(1, 8/16)  = 1
#   tp32  → 128/32  = 4 experts/rank, top_k = max(1, 8/32)  = 1
#   tp64  → 128/64  = 2 experts/rank, top_k = max(1, 8/64)  = 1
#   tp128 → 128/128 = 1 expert/rank,  top_k = 1
#
# tp1 is included in TP_LIST to satisfy the replicate_tp_stable require_tp1
# guard; with EP_DECOUPLE=1 and tp=1 the result is standard 128-expert tp1.
#
# Output lands ONLY under profiler/perf_ep_decouple/ (never profiler/perf/):
#   profiler/perf_ep_decouple/H100_<FREQ>MHz/Qwen/.../fp16/tp{1,16,32,64,128}/moe.csv
# Do NOT treat these as full-MoE characterization tables for reconstruction MAE.
#
# UNVALIDATED NOTE: These are single-GPU EP emulations. ARC cannot run real
# EP16/32/64/128.  Validated method: only ≤EP8 on V100.  Treat tp16/32/64/128
# tables as extrapolation.
#
# H100 clock notes:
#   - Hardware: H100 PCIe 80GB (htc-g053-055, g058); gpu:h100:4 mix/up; g059-060 boot.
#   - SM freq range on ARC PCIe nodes: ~1200-1755 MHz.
#   - Target freqs for Qwen3-30B DVFS: 1500 and 1620 MHz (existing GT points in
#     moe-dvfs-dataset); also 1200 MHz and 1410 MHz for full coverage.
#   - DO NOT target >1620 if boost-cap is 1620 — poller will reject it.
#   - H100 hold strategy: apply lock unconditionally at startup AND every poller
#     tick (single-layer dummy MoE draws <90W → power-gated threshold never fires,
#     so unconditional re-apply is required).
#   - 1755 is the H100 PCIe boost cap; do NOT target it as a "locked" freq.
#
# Usage (NEVER --reservation):
#   sbatch -M htc --clusters=htc --account=engs-glass --qos=priority \
#          --partition=short --gres=gpu:h100:1 \
#          --nodes=1 --cpus-per-task=8 --mem=90G --time=02:00:00 \
#          profiler/jobs/run_moe_table_ep_decouple_hightp_H100.sbatch \
#          <MODEL_HF_ID> <FREQ_MHZ> [TP_LIST]
#
# Args:
#   $1 MODEL_HF_ID  — e.g. "Qwen/Qwen3-30B-A3B-Instruct-2507"
#   $2 FREQ_MHZ     — target frequency, e.g. 1620
#   $3 TP_LIST      — comma-separated tp degrees (default: 1,16,32,64,128)
#
# NEVER use --reservation.
#
#SBATCH --job-name=ep_dec_h100
#SBATCH --output=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/logs_table_held/ep_dec_h100-%j.out
#SBATCH --error=/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/logs_table_held/ep_dec_h100-%j.err

set -uo pipefail

MODEL_HF_ID="${1:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
FREQ_MHZ="${2:-1620}"
TP_LIST="${3:-1,16,32,64,128}"
# tp1 is required by replicate_tp_stable (tp-invariant dense/attention layers are copied
# FROM tp1); auto-include it so a caller cannot omit it and hit a late replication failure.
case ",${TP_LIST}," in *",1,"*) ;; *) TP_LIST="1,${TP_LIST}" ;; esac

HW_LABEL="H100_${FREQ_MHZ}MHz"

ENGS_GLASS="/data/engs-glass/engs2950"
LLMSIM_ROOT="${ENGS_GLASS}/DVFS-MoE/LLMServingSim"
HF_CACHE_ROOT="${ENGS_GLASS}/infra/hf_cache"
CLOCKS_BIN="/usr/local/sbin/nvidia-smi-clocks"
VLLM_IMAGE="${ENGS_GLASS}/DVFS-MoE/moe.sif"

JOB_TAG="${SLURM_JOB_ID:-local}"
SCRATCH="${ENGS_GLASS}/.llmsim/${JOB_TAG}_ep_dec_h100"

# CRITICAL: never write EP-decouple emulations into profiler/perf/.
# That path is reserved for full-MoE characterization tables. Mixing them
# produced a bogus ~43% H100 Qwen reconstruction MAE (2026-07-17).
# _variant_root() builds: out_root / hardware / model_path / variant
# → profiler/perf_ep_decouple / H100_1620MHz / Qwen/... / fp16
PERF_OUT="${LLMSIM_ROOT}/profiler/perf_ep_decouple"
# Hard abort if anyone retargets PERF_OUT at the canonical full-MoE tree.
if [[ "${PERF_OUT%/}" == */profiler/perf ]] || [[ "${PERF_OUT}" == */profiler/perf_table_held ]] || [[ "${PERF_OUT}" == */profiler/perf_table_held/* ]]; then
  echo "FATAL: PERF_OUT=${PERF_OUT} must not be a full-MoE tree (use perf_ep_decouple). Incident 2026-07-17." >&2
  exit 2
fi
HOLD_DIR="${LLMSIM_ROOT}/profiler/jobs/logs_table_held/hold_${JOB_TAG}"

MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=256
MEASUREMENT_ITERS=15
HOLD_INTERVAL_SEC=1
HOLD_TOLERANCE_MHZ=30
HOLD_MAX_SEC=7200   # 2 h ceiling

# Physical GPU index from Slurm (matches system helper's root-side view).
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
echo "ep_decouple_hightp_H100  Node=$(hostname)  Job=${JOB_TAG}"
echo "Model:    ${MODEL_HF_ID}"
echo "HW:       ${HW_LABEL}"
echo "Target:   ${FREQ_MHZ} MHz"
echo "TP_LIST:  ${TP_LIST}  (EP_DECOUPLE=1)"
echo "PerfOut:  ${PERF_OUT}"
echo "Expected: ${PERF_OUT}/${HW_LABEL}/Qwen/Qwen3-30B-A3B-Instruct-2507/fp16/tp{16,32,64,128}/moe.csv"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "Physical GPU index: ${PHYSICAL_GPU_IDX}"
echo ""
echo "EMULATION NOTE: tp16/32/64/128 are SINGLE-GPU EP_DECOUPLE emulations."
echo "  ARC cannot run real 16/32/64/128-way EP/TP. Treat as extrapolation (unvalidated beyond EP8)."
echo "================================================================"

nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,clocks.sm,clocks.max.sm --format=csv,noheader

# ── H100 clock lock: apply unconditionally now AND every poller tick ──────────
# H100 single-layer MoE draws <90W → power-gated threshold rarely fires.
# Strategy (from run_moe_table_H100.sbatch): apply now + re-apply every tick.
# Try --id form first; fall back to no-id batch-all form.
CLOCKS_USE_ID=0
if sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" "${FREQ_MHZ}" "${FREQ_MHZ}" 2>/dev/null; then
    echo "[H100] Clock lock APPLIED (--id form) at ${FREQ_MHZ} MHz"
    CLOCKS_USE_ID=1
elif sudo -n "${CLOCKS_BIN}" "${FREQ_MHZ}" "${FREQ_MHZ}" 2>/dev/null; then
    echo "[H100] Clock lock APPLIED (no-id / batch-all form) at ${FREQ_MHZ} MHz"
    CLOCKS_USE_ID=0
else
    SUDO_RC=$?
    echo "[H100] WARNING: sudo test rc=${SUDO_RC} — both lock forms failed; poller will retry."
fi
echo "[H100] CLOCKS_USE_ID=${CLOCKS_USE_ID}"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null || true

HOLD_PID=""
PROFILER_PID=""

cleanup() {
    echo "[H100] cleanup: stopping hold poller ..."
    [[ -n "${HOLD_PID}" ]] && {
        touch "${HOLD_DIR}/gpu_freq_hold.stop" 2>/dev/null || true
        sleep 2; kill "${HOLD_PID}" 2>/dev/null || true; wait "${HOLD_PID}" 2>/dev/null || true
    }
    echo "[H100] cleanup: resetting GPU clocks (physical idx=${PHYSICAL_GPU_IDX}) ..."
    if [[ "${CLOCKS_USE_ID:-0}" == "1" ]]; then
        sudo -n "${CLOCKS_BIN}" --id "${PHYSICAL_GPU_IDX}" --reset 2>/dev/null || true
    else
        sudo -n "${CLOCKS_BIN}" --reset 2>/dev/null || true
    fi
    rb=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null || echo "?")
    echo "[H100]   post-reset max SM clock: ${rb}"
    [[ -n "${PROFILER_PID}" ]] && kill "${PROFILER_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'echo "[H100] SIGTERM → cleanup"; exit 143' TERM
trap 'echo "[H100] SIGINT  → cleanup"; exit 130' INT
trap 'echo "[H100] SIGHUP  → cleanup"; exit 129' HUP

unset SLURM_JOB_ACCOUNT 2>/dev/null || true

# ── Step 1: Start hold poller ─────────────────────────────────────────────────
echo ""
echo "=== [1] Starting hold poller (target=${FREQ_MHZ} MHz, unconditional re-apply) ==="

python3 - <<HOLD_PYEOF > "${HOLD_DIR}/hold.log" 2>&1 &
import json, os, subprocess, time, statistics, pathlib

cap_mhz       = ${FREQ_MHZ}
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

# Re-apply the lock unconditionally at startup (holds through the whole run).
apply_lock(physical_idx, cap_mhz)

t0 = time.time()
under_load_clocks = []   # power_w >= 150W (heavy compute)
active_clocks = []       # power_w >= 78W (any non-idle window)
ticks = 0
reapplies = 0
power_thr_w = 150
active_thr_w = 78   # just above H100 PCIe idle (~73-77W)

while not stop_path.exists() and (time.time() - t0) < max_sec:
    gpus = smi_sample()
    rec = {"t": round(time.time() - t0, 1), "gpus": gpus}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    # Unconditional re-apply keeps the clock pinned (H100 MoE draw is <90W,
    # below heavy-load threshold, so power-gated re-apply would never fire).
    if apply_lock(physical_idx, cap_mhz):
        reapplies += 1
    for g in gpus:
        p = g.get("power_w", 0)
        clk = int(g["graphics_mhz"])
        if p >= power_thr_w:
            under_load_clocks.append(clk)
        if p >= active_thr_w:
            active_clocks.append(clk)
    ticks += 1
    time.sleep(interval_sec)

# Verdict: prefer heavy-load samples; fall back to active (non-idle) samples.
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
echo "[H100] Hold poller pid=${HOLD_PID}"
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
  > "${LLMSIM_ROOT}/profiler/jobs/logs_table_held/ep_dec_h100_${JOB_TAG}.log" 2>&1 &

PROFILER_PID=$!
echo "[H100] Profiler pid=${PROFILER_PID}"

# ── Step 3: Wait for profiler ─────────────────────────────────────────────────
echo ""
echo "=== [3] Waiting for profiler ==="
PROFILER_EXIT=0
wait "${PROFILER_PID}" || PROFILER_EXIT=$?
PROFILER_PID=""
echo "[H100] Profiler exit code: ${PROFILER_EXIT}"

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
src     = s.get("verdict_source", "?")
print(f"[verdict] verdict_ok={verdict}  ul_median={median} MHz  reapplies={reapply}  source={src}")
if not verdict:
    print("WARNING: hold poller did NOT confirm ${FREQ_MHZ} MHz under load.")
    print("  If ul_median > ${FREQ_MHZ}+30, the H100 boost cap prevented the lock.")
    sys.exit(1)
else:
    print("PASS: hold poller confirms ${FREQ_MHZ} MHz held under load.")
PYEOF
fi

# ── Step 5: Locate output CSVs and validate ───────────────────────────────────
echo ""
echo "=== [5] Output CSVs ==="

BASE_PATH="${PERF_OUT}/${HW_LABEL}/Qwen/Qwen3-30B-A3B-Instruct-2507/fp16"

# Expected ae/rank per TP degree (for validation checks below)
declare -A EXPECTED_AE
EXPECTED_AE[16]=8
EXPECTED_AE[32]=4
EXPECTED_AE[64]=2
EXPECTED_AE[128]=1

for TP in 16 32 64 128; do
    CSV="${BASE_PATH}/tp${TP}/moe.csv"
    EXP_AE="${EXPECTED_AE[$TP]}"
    if [[ -f "${CSV}" ]]; then
        echo ""
        echo "tp${TP}/moe.csv FOUND: ${CSV}"
        wc -l "${CSV}"
        head -4 "${CSV}"
        python3 - <<PYEOF
import csv, sys
path = "${CSV}"
exp_ae = ${EXP_AE}
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
if ae_vals == [exp_ae]:
    print(f"  ae CHECK: PASS (ae={exp_ae}, correct for tp=${TP})")
else:
    print(f"  ae CHECK: FAIL (expected [{exp_ae}], got {ae_vals})")
if achieved_nums:
    med = sorted(achieved_nums)[len(achieved_nums)//2]
    ok  = abs(med - ${FREQ_MHZ}) <= 30
    print(f"  achieved_mhz (median): {med:.0f}  target: ${FREQ_MHZ}  clock_clean: {ok}")
    if not ok:
        print("  WARNING: achieved_mhz out of tolerance — check H100 boost cap!")
else:
    print("  achieved_mhz: not in table (older schema)")
PYEOF
    else
        echo ""
        echo "tp${TP}/moe.csv MISSING: ${CSV}"
    fi
done

echo ""
echo "All moe.csv under ${BASE_PATH}:"
find "${BASE_PATH}" -name "moe.csv" 2>/dev/null | sort

echo ""
echo "=== ep_decouple_hightp_H100 ${MODEL_HF_ID} @ ${FREQ_MHZ}MHz DONE (profiler rc=${PROFILER_EXIT}) ==="
echo "EMULATION REMINDER: tp16/32/64/128 are NOT validated at real EP16/32/64/128."
echo "  Only method validated: PROFILER_EP_DECOUPLE <= EP8 on V100."
exit "${PROFILER_EXIT}"
