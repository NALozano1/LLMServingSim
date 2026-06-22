#!/usr/bin/env bash
# Host-side mid-bench GPU frequency switch (runs outside Apptainer).
#
# Usage:
#   bench_freq_switcher.sh <out_dir> <freq_a> <freq_b> <switch_after_sec>
#
# Applies freq_a immediately (caller may have already done this), waits until
# OUT_DIR/bench_running exists for switch_after_sec wall seconds, then applies
# freq_b directly (no restore between transitions — required on V100 mid-run).
#
set -euo pipefail

OUT_DIR="${1:?out_dir}"
FREQ_A="${2:?freq_a}"
FREQ_B="${3:?freq_b}"
SWITCH_AFTER_SEC="${4:?switch_after_sec}"

ENGS2950_ROOT="${ENGS2950_ROOT:-/data/engs-glass/engs2950}"
GPU_FREQ_PY="${ENGS2950_ROOT}/shared/scripts/gpu_freq_lock.py"
FREQ_META_DIR="${OUT_DIR}/gpu_freq"
LOG="${OUT_DIR}/freq_switch.log"

mkdir -p "${FREQ_META_DIR}"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG}"
}

log "switcher start: ${FREQ_A} -> ${FREQ_B} after ${SWITCH_AFTER_SEC}s"

# Wait for bench to mark itself running.
for _ in $(seq 1 600); do
  if [[ -f "${OUT_DIR}/bench_running" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "${OUT_DIR}/bench_running" ]]; then
  log "WARN: bench_running marker never appeared; aborting switch"
  exit 0
fi

start_epoch="$(cat "${OUT_DIR}/bench_running")"
target_epoch=$((start_epoch + SWITCH_AFTER_SEC))

now_epoch="$(date +%s)"
if (( now_epoch < target_epoch )); then
  sleep $((target_epoch - now_epoch))
fi

log "applying ${FREQ_B} MHz (direct apply, no restore)"
if python3 "${GPU_FREQ_PY}" apply --mhz "${FREQ_B}" --out-dir "${FREQ_META_DIR}"; then
  log "apply ok"
else
  log "ERROR: apply ${FREQ_B} failed"
  exit 1
fi

GPU_FREQ_STABLE_TOL_MHZ="${GPU_FREQ_STABLE_TOL_MHZ:-15}" \
GPU_FREQ_STABLE_READS="${GPU_FREQ_STABLE_READS:-2}" \
GPU_FREQ_STABLE_TIMEOUT_SEC="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}" \
  python3 "${GPU_FREQ_PY}" wait-stable --mhz "${FREQ_B}" --out-dir "${FREQ_META_DIR}" \
  >> "${LOG}" 2>&1 || log "WARN: wait-stable failed"

python3 - <<PY >> "${LOG}" 2>&1
import json
from pathlib import Path
rec = {
    "from_mhz": int("${FREQ_A}"),
    "to_mhz": int("${FREQ_B}"),
    "switch_after_sec": int("${SWITCH_AFTER_SEC}"),
    "switched_at_epoch": $(date +%s),
}
Path("${OUT_DIR}/freq_switch.json").write_text(json.dumps(rec, indent=2) + "\n")
PY

log "switch complete"
