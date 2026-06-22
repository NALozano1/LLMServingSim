#!/usr/bin/env bash
# Host-side DVFS poller for in-place layer barriers (runs outside Apptainer).
#
# Watches ``<watch_root>/dvfs_barriers/*/pending.json``. When a worker
# blocks at a layer boundary, applies GPU frequency via gpu_freq_lock.py,
# waits for stable clocks (nvidia-smi poll), optional settle, writes
# ack.json + appends dvfs_markers.jsonl.
#
# Usage:
#   dvfs_barrier_host_poller.sh <watch_root> [freq_meta_dir]
#
# Environment:
#   DVFS_PAUSE_ONLY=1           Ack barriers only; no gpu_freq_lock (smoke)
#   PAUSE_ONLY_DELAY_SEC=0.05   Artificial pause length in pause-only mode
#   DVFS_FREQ_SCHEDULE=700,900,1100,1300
#   GPU_FREQ_SETTLE_SEC=0       Optional extra sleep after stable verification (default 0)
#   GPU_FREQ_STABLE_TOL_MHZ=15
#   GPU_FREQ_STABLE_READS=2
#   GPU_FREQ_STABLE_TIMEOUT_SEC=15
#   ENGS2950_ROOT=/data/engs-glass/engs2950
#
set -euo pipefail

WATCH_ROOT="${1:?watch_root required}"
FREQ_META_DIR="${2:-${WATCH_ROOT}/gpu_freq}"
MARKERS="${WATCH_ROOT}/dvfs_markers.jsonl"
ENGS2950_ROOT="${ENGS2950_ROOT:-/data/engs-glass/engs2950}"
GPU_FREQ_PY="${ENGS2950_ROOT}/shared/scripts/gpu_freq_lock.py"
PAUSE_ONLY="${DVFS_PAUSE_ONLY:-0}"
SCHEDULE="${DVFS_FREQ_SCHEDULE:-700,900,1100,1300}"
SETTLE_SEC="${GPU_FREQ_SETTLE_SEC:-0}"
PAUSE_DELAY="${PAUSE_ONLY_DELAY_SEC:-0.05}"
STABLE_TOL="${GPU_FREQ_STABLE_TOL_MHZ:-15}"
STABLE_READS="${GPU_FREQ_STABLE_READS:-2}"
STABLE_TIMEOUT="${GPU_FREQ_STABLE_TIMEOUT_SEC:-15}"

IFS=',' read -r -a FREQS <<< "${SCHEDULE}"
if [[ "${PAUSE_ONLY}" != "1" && ${#FREQS[@]} -eq 0 ]]; then
  echo "[dvfs-poller] empty DVFS_FREQ_SCHEDULE" >&2
  exit 1
fi

mkdir -p "${FREQ_META_DIR}" "$(dirname "${MARKERS}")"
FREQ_IDX=0

if [[ "${PAUSE_ONLY}" == "1" ]]; then
  echo "[dvfs-poller] PAUSE_ONLY watching ${WATCH_ROOT}/dvfs_barriers delay=${PAUSE_DELAY}s"
else
  echo "[dvfs-poller] watching ${WATCH_ROOT}/dvfs_barriers schedule=${SCHEDULE} settle=${SETTLE_SEC}s stable_tol=${STABLE_TOL}MHz"
fi

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

    pause_start="$(date +%s.%N)"
    if [[ "${PAUSE_ONLY}" == "1" ]]; then
      echo "[dvfs-poller] PAUSE_ONLY layer=${layer_name} idx=${layer_idx}"
      sleep "${PAUSE_DELAY}"
      mode_py="pause_only"
      mhz_val=""
      freq_ok_val=""
      stable_ok_val=""
      stable_sec_val=""
      settle_val="${PAUSE_DELAY}"
      clocks_after_json="[]"
    else
      mhz="${FREQS[$((FREQ_IDX % ${#FREQS[@]}))]}"
      FREQ_IDX=$((FREQ_IDX + 1))
      echo "[dvfs-poller] layer=${layer_name} idx=${layer_idx} -> ${mhz} MHz"
      if python3 "${GPU_FREQ_PY}" apply --mhz "${mhz}" --out-dir "${FREQ_META_DIR}"; then
        apply_rc=0
      else
        apply_rc=$?
      fi
      if [[ "${apply_rc}" -eq 0 ]]; then
        freq_ok_val="true"
      else
        freq_ok_val="false"
        echo "[dvfs-poller] WARN freq apply failed for ${mhz} MHz (rc=${apply_rc})" >&2
      fi
      stable_rc=0
      GPU_FREQ_STABLE_TOL_MHZ="${STABLE_TOL}" \
      GPU_FREQ_STABLE_READS="${STABLE_READS}" \
      GPU_FREQ_STABLE_TIMEOUT_SEC="${STABLE_TIMEOUT}" \
        python3 "${GPU_FREQ_PY}" wait-stable --mhz "${mhz}" --out-dir "${FREQ_META_DIR}" \
        --tolerance-mhz "${STABLE_TOL}" --stable-reads "${STABLE_READS}" --timeout "${STABLE_TIMEOUT}" \
        || stable_rc=$?
      if [[ "${stable_rc}" -eq 0 ]]; then
        stable_ok_val="true"
      else
        stable_ok_val="false"
        echo "[dvfs-poller] WARN stable wait failed for ${mhz} MHz (rc=${stable_rc})" >&2
      fi
      stable_sec_val="$(python3 -c "import json; print(json.load(open('${FREQ_META_DIR}/gpu_freq_stable.json')).get('elapsed_sec',''))" 2>/dev/null || echo "")"
      if [[ "${SETTLE_SEC}" != "0" && "${SETTLE_SEC}" != "0.0" ]]; then
        sleep "${SETTLE_SEC}"
      fi
      mode_py="dvfs"
      mhz_val="${mhz}"
      settle_val="${SETTLE_SEC}"
      clocks_after_json="$(python3 -c "import json; print(json.dumps(json.load(open('${FREQ_META_DIR}/gpu_freq_stable.json')).get('clocks',[])))" 2>/dev/null || echo '[]')"
    fi
    pause_end="$(date +%s.%N)"

    MARKERS_PATH="${MARKERS}" \
    ACK_PATH="${ack}" \
    BARRIER_ID="${barrier_id}" \
    LAYER_IDX="${layer_idx}" \
    LAYER_NAME="${layer_name}" \
    PAUSE_START="${pause_start}" \
    PAUSE_END="${pause_end}" \
    MODE="${mode_py}" \
    FREQ_MHZ="${mhz_val}" \
    SETTLE_SEC_VAL="${settle_val}" \
    FREQ_OK="${freq_ok_val}" \
    STABLE_OK="${stable_ok_val:-}" \
    STABLE_SEC="${stable_sec_val:-}" \
    CLOCKS_AFTER="${clocks_after_json}" \
    python3 - <<'PY'
import json
import os
from datetime import datetime, timezone

def _opt_int(s: str):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None

def _opt_bool(s: str):
    s = (s or "").strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None

def _opt_float(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None

try:
    clocks_after = json.loads(os.environ.get("CLOCKS_AFTER", "[]") or "[]")
except json.JSONDecodeError:
    clocks_after = []

record = {
    "event": "layer_boundary",
    "mode": os.environ["MODE"],
    "wall_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    "pause_start": float(os.environ["PAUSE_START"]),
    "pause_end": float(os.environ["PAUSE_END"]),
    "pause_sec": float(os.environ["PAUSE_END"]) - float(os.environ["PAUSE_START"]),
    "freq_mhz": _opt_int(os.environ.get("FREQ_MHZ", "")),
    "settle_sec": float(os.environ["SETTLE_SEC_VAL"]),
    "layer_idx": os.environ["LAYER_IDX"],
    "layer_name": os.environ["LAYER_NAME"],
    "barrier_id": os.environ["BARRIER_ID"],
    "freq_apply_ok": _opt_bool(os.environ.get("FREQ_OK", "")),
    "freq_stable_ok": _opt_bool(os.environ.get("STABLE_OK", "")),
    "freq_stable_sec": _opt_float(os.environ.get("STABLE_SEC", "")),
    "clocks_after": clocks_after,
    "poller": "host_shell",
}
with open(os.environ["MARKERS_PATH"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
ack = {
    "barrier_id": os.environ["BARRIER_ID"],
    "freq_mhz": _opt_int(os.environ.get("FREQ_MHZ", "")),
    "freq_stable_ok": _opt_bool(os.environ.get("STABLE_OK", "")),
    "ack_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
}
with open(os.environ["ACK_PATH"], "w", encoding="utf-8") as fh:
    json.dump(ack, fh)
PY
    rm -f "${pending}"
  done
  shopt -u nullglob
  sleep 0.001
done
