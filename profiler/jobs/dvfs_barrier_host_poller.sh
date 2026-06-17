#!/usr/bin/env bash
# Host-side DVFS poller for in-place layer barriers (runs outside Apptainer).
#
# Watches ``<watch_root>/dvfs_barriers/*/pending.json``. When a worker
# blocks at a layer boundary, applies GPU frequency via gpu_freq_lock.py,
# waits for settle, writes ack.json + appends dvfs_markers.jsonl.
#
# Usage:
#   dvfs_barrier_host_poller.sh <watch_root> [freq_meta_dir]
#
# Environment:
#   DVFS_FREQ_SCHEDULE=700,900,1100,1300
#   GPU_FREQ_SETTLE_SEC=1.5
#   ENGS2950_ROOT=/data/engs-glass/engs2950
#
set -euo pipefail

WATCH_ROOT="${1:?watch_root required}"
FREQ_META_DIR="${2:-${WATCH_ROOT}/gpu_freq}"
MARKERS="${WATCH_ROOT}/dvfs_markers.jsonl"
ENGS2950_ROOT="${ENGS2950_ROOT:-/data/engs-glass/engs2950}"
GPU_FREQ_PY="${ENGS2950_ROOT}/shared/scripts/gpu_freq_lock.py"
SCHEDULE="${DVFS_FREQ_SCHEDULE:-700,900,1100,1300}"
SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-1.5}"

IFS=',' read -r -a FREQS <<< "${SCHEDULE}"
if [[ ${#FREQS[@]} -eq 0 ]]; then
  echo "[dvfs-poller] empty DVFS_FREQ_SCHEDULE" >&2
  exit 1
fi

mkdir -p "${FREQ_META_DIR}" "$(dirname "${MARKERS}")"
FREQ_IDX=0

echo "[dvfs-poller] watching ${WATCH_ROOT}/dvfs_barriers schedule=${SCHEDULE} settle=${SETTLE_SEC}s"

while true; do
  shopt -s nullglob
  for pending in "${WATCH_ROOT}"/dvfs_barriers/*/pending.json; do
    [[ -f "${pending}" ]] || continue
    barrier_dir="$(dirname "${pending}")"
    ack="${barrier_dir}/ack.json"
    rm -f "${ack}"

    layer_idx="$(python3 -c "import json; print(json.load(open('${pending}'))['layer_idx'])" 2>/dev/null || echo "?")"
    layer_name="$(python3 -c "import json; print(json.load(open('${pending}'))['layer_name'])" 2>/dev/null || echo "?")"
    barrier_id="$(python3 -c "import json; print(json.load(open('${pending}'))['barrier_id'])" 2>/dev/null || echo "")"

    mhz="${FREQS[$((FREQ_IDX % ${#FREQS[@]}))]}"
    FREQ_IDX=$((FREQ_IDX + 1))

    pause_start="$(date +%s.%N)"
    echo "[dvfs-poller] layer=${layer_name} idx=${layer_idx} -> ${mhz} MHz"
    freq_ok_py=False
    if python3 "${GPU_FREQ_PY}" apply --mhz "${mhz}" --out-dir "${FREQ_META_DIR}"; then
      freq_ok_py=True
    else
      echo "[dvfs-poller] WARN freq apply failed for ${mhz} MHz" >&2
    fi
    sleep "${SETTLE_SEC}"
    pause_end="$(date +%s.%N)"

    python3 - <<PY
import json
from datetime import datetime, timezone
record = {
    "event": "layer_boundary",
    "wall_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    "pause_start": float("${pause_start}"),
    "pause_end": float("${pause_end}"),
    "freq_mhz": int("${mhz}"),
    "settle_sec": float("${SETTLE_SEC}"),
    "layer_idx": "${layer_idx}",
    "layer_name": "${layer_name}",
    "barrier_id": """${barrier_id}""",
    "freq_apply_ok": ${freq_ok_py},
    "poller": "host_shell",
}
with open("""${MARKERS}""", "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
ack = {
    "barrier_id": """${barrier_id}""",
    "freq_mhz": int("${mhz}"),
    "ack_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
}
with open("""${ack}""", "w", encoding="utf-8") as fh:
    json.dump(ack, fh)
PY
    rm -f "${pending}"
  done
  shopt -u nullglob
  sleep 0.001
done
