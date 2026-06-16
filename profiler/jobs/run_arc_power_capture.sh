#!/usr/bin/env bash
# Idle GPU + server power capture on an ARC GPU node (nvidia-smi + ipmitool dcmi).
#
# Same permission class as fingerprint-fork monitor gpu.sh (nvidia-smi, no sudo).
# Server power uses ipmitool like LLMServingSim profiler/power examples; may be NA on
# nodes without BMC device (logged in metadata + server_power.csv status column).
#
# Optional DVFS via shared gpu_freq_lock_lib.sh (passwordless sudo on compute nodes).
#
# Usage:
#   GPU_TYPE=v100 IDLE_SEC=120 bash profiler/jobs/run_arc_power_capture.sh
#   GPU_FREQ_MHZ=1100 bash profiler/jobs/run_arc_power_capture.sh
#
set -euo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POWER_DIR="${REPO_ROOT}/profiler/power"
JOB_TAG="${SLURM_JOB_ID:-local}"
GPU_FREQ_LIB="${ENGS_GLASS}/shared/scripts/gpu_freq_lock_lib.sh"

GPU_TYPE="${GPU_TYPE:-unknown}"
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-}"
IDLE_SEC="${IDLE_SEC:-120}"
INTERVAL_MS="${INTERVAL_MS:-1000}"
INTERVAL_SEC="${INTERVAL_SEC:-1}"

TAG="${POWER_TAG:-${GPU_TYPE}}"
if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  TAG="${GPU_TYPE}_${GPU_FREQ_MHZ}MHz"
fi

OUT_ROOT="${POWER_OUT_ROOT:-${REPO_ROOT}/profiler/power/data}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${TAG}/${JOB_TAG}}"

mkdir -p "${OUT_DIR}"

# shellcheck source=/dev/null
source "${GPU_FREQ_LIB}"
trap 'gpu_freq_lock_trap_restore "${OUT_DIR}/gpu_freq" 2>/dev/null || true' EXIT

echo "=== ARC power capture ==="
echo "OUT_DIR=${OUT_DIR}"
echo "GPU_TYPE=${GPU_TYPE}  GPU_FREQ_MHZ=${GPU_FREQ_MHZ:-<default>}"
echo "IDLE_SEC=${IDLE_SEC}  INTERVAL_MS=${INTERVAL_MS}"
nvidia-smi -L 2>/dev/null || echo "WARN: no nvidia-smi on this host"

export OUT_DIR NODE NODELIST GPU_FREQ_MHZ IPMI_OK
NODE="$(hostname -s 2>/dev/null || hostname)"
NODELIST="${SLURM_NODELIST:-}"
if [[ -n "${NODELIST}" ]] && command -v scontrol >/dev/null 2>&1; then
  NODELIST="$(scontrol show hostnames "${NODELIST}" | paste -sd, -)"
fi
export NODE NODELIST
IPMI_OK=0
command -v ipmitool >/dev/null 2>&1 && ipmitool dcmi power reading >/dev/null 2>&1 && IPMI_OK=1
export IPMI_OK

bash "${POWER_DIR}/capture_power_metadata.sh" "${OUT_DIR}" "${GPU_FREQ_MHZ}"

if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  gpu_freq_lock_apply "${OUT_DIR}/gpu_freq" "${GPU_FREQ_MHZ}"
fi

GPU_PID=""
SRV_PID=""
cleanup_loggers() {
  for pid in "${GPU_PID}" "${SRV_PID}"; do
    [[ -z "${pid}" ]] && continue
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
}
trap 'cleanup_loggers; gpu_freq_lock_trap_restore "${OUT_DIR}/gpu_freq" 2>/dev/null || true' EXIT

DURATION_SEC="${IDLE_SEC}" \
  bash "${POWER_DIR}/profile_gpu_power.sh" "${OUT_DIR}/gpu_power.csv" &
GPU_PID=$!

DURATION_SEC="${IDLE_SEC}" INTERVAL_SEC="${INTERVAL_SEC}" \
  bash "${POWER_DIR}/profile_server_power.sh" "${OUT_DIR}/server_power.csv" &
SRV_PID=$!

echo "Power loggers started (gpu pid=${GPU_PID}, server pid=${SRV_PID}). Idling ${IDLE_SEC}s ..."
sleep "${IDLE_SEC}"

cleanup_loggers
GPU_PID=""
SRV_PID=""

export OUT_DIR IDLE_SEC
python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import os

out = Path(os.environ["OUT_DIR"])
p = out / "metadata.json"
meta = json.loads(p.read_text())
meta["idle_sec"] = int(os.environ.get("IDLE_SEC", "0"))
meta["finished_at"] = datetime.now(timezone.utc).isoformat()
for name, key in [("gpu_power.csv", "gpu_power_samples"), ("server_power.csv", "server_power_samples")]:
    fp = out / name
    if fp.exists():
        meta[key] = sum(1 for ln in fp.read_text().splitlines() if ln and not ln.startswith("#"))
p.write_text(json.dumps(meta, indent=2) + "\n")
PY

echo "=== Done. Outputs: ${OUT_DIR} ==="
