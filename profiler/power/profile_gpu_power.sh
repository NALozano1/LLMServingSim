#!/usr/bin/env bash
# Poll GPU power via nvidia-smi (all visible GPUs). No sudo required.
#
# Usage:
#   profile_gpu_power.sh <output.csv>
# Env: INTERVAL_MS (default 1000), DURATION_SEC (0 = until SIGTERM)
#
set -euo pipefail

OUT="${1:?output csv path}"
INTERVAL_MS="${INTERVAL_MS:-1000}"
DURATION_SEC="${DURATION_SEC:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found (run on a GPU compute node)" >&2
  exit 1
fi

if [[ ! -f "${OUT}" ]]; then
  NODE="$(hostname -s 2>/dev/null || hostname)"
  {
    echo "# node=${NODE} slurm_job_id=${SLURM_JOB_ID:-}"
    echo "timestamp,index,uuid,name,pci.bus_id,utilization.gpu,power.draw,clocks.current.graphics"
  } > "${OUT}"
fi

_query() {
  nvidia-smi --query-gpu=timestamp,index,uuid,name,pci.bus_id,utilization.gpu,power.draw,clocks.current.graphics \
    --format=csv,noheader,nounits 2>/dev/null || true
}

end_ts=0
if [[ "${DURATION_SEC}" -gt 0 ]]; then
  end_ts=$(( $(date +%s) + DURATION_SEC ))
fi

while true; do
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    echo "${line}" >> "${OUT}"
  done < <(_query)
  if [[ "${DURATION_SEC}" -gt 0 ]] && [[ "$(date +%s)" -ge "${end_ts}" ]]; then
    break
  fi
  sleep "$(awk -v ms="${INTERVAL_MS}" 'BEGIN { printf "%.3f", ms/1000 }')"
done
