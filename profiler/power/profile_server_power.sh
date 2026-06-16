#!/usr/bin/env bash
# Poll server (BMC) power via ipmitool dcmi. No sudo on ARC when /dev/ipmi* exists.
#
# Usage:
#   profile_server_power.sh <output.csv>
# Env: INTERVAL_SEC (default 1), DURATION_SEC (0 = until SIGTERM)
#
set -euo pipefail

OUT="${1:?output csv path}"
INTERVAL_SEC="${INTERVAL_SEC:-1}"
DURATION_SEC="${DURATION_SEC:-0}"

if [[ ! -f "${OUT}" ]]; then
  NODE="$(hostname -s 2>/dev/null || hostname)"
  IPMI_OK=0
  command -v ipmitool >/dev/null 2>&1 && ipmitool dcmi power reading >/dev/null 2>&1 && IPMI_OK=1
  {
    echo "# node=${NODE} slurm_job_id=${SLURM_JOB_ID:-} ipmitool_dcmi=${IPMI_OK}"
    echo "timestamp,server_power_w,status"
  } > "${OUT}"
fi

_read_power() {
  if ! command -v ipmitool >/dev/null 2>&1; then
    echo "NA,no_ipmitool"
    return
  fi
  local raw
  if ! raw="$(ipmitool dcmi power reading 2>/dev/null)"; then
    echo "NA,ipmitool_failed"
    return
  fi
  local watts
  watts="$(echo "${raw}" | grep -i "Instantaneous power reading" | awk '{print $(NF-1)}' | head -1)"
  if [[ -z "${watts}" ]]; then
    echo "NA,parse_failed"
  else
    echo "${watts},ok"
  fi
}

end_ts=0
if [[ "${DURATION_SEC}" -gt 0 ]]; then
  end_ts=$(( $(date +%s) + DURATION_SEC ))
fi

while true; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  read -r watts status < <(echo "$(_read_power)" | tr ',' ' ')
  echo "${ts},${watts},${status}" >> "${OUT}"
  if [[ "${DURATION_SEC}" -gt 0 ]] && [[ "$(date +%s)" -ge "${end_ts}" ]]; then
    break
  fi
  sleep "${INTERVAL_SEC}"
done
