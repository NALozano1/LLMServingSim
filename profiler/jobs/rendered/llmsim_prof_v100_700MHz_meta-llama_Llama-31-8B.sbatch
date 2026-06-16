#!/bin/bash
#SBATCH --job-name=llmsim_llama8b_700MHz_meta-llama_Llama-31-8B
#SBATCH --clusters=htc
#SBATCH --account=engs-glass
#SBATCH --partition=interactive
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Source external variables (RENDER_URL and API_SECRET)
source ~/.arc_tracker_secrets

JOB_ID=$SLURM_JOB_ID
JOB_NAME=$SLURM_JOB_NAME
PROJECT="engs2950"
DATA_OUTPUT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/perf"
FILE_LOCATION="$PWD"
mkdir -p logs
LOG_FILE="logs/${JOB_ID}.out"
URL="${RENDER_URL}/update"
SECRET="${API_SECRET}"

# On multi-cluster Slurm, set before `sbatch` or uncomment:
# export TRACKER_SLURM_CLUSTERS=htc
if [[ -n "${TRACKER_SLURM_CLUSTERS:-}" ]]; then
  _TRACK_SQUEUE=(--clusters="${TRACKER_SLURM_CLUSTERS}")
  _TRACK_SACCT=(--clusters="${TRACKER_SLURM_CLUSTERS}")
else
  _TRACK_SQUEUE=()
  _TRACK_SACCT=()
fi
JOB_TRACKER_START="$(squeue "${_TRACK_SQUEUE[@]}" -j "$JOB_ID" -h -o "%S" 2>/dev/null | head -1)"
JOB_TRACKER_DURATION="$(squeue "${_TRACK_SQUEUE[@]}" -j "$JOB_ID" -h -o "%M" 2>/dev/null | head -1)"
if [[ -z "$JOB_TRACKER_START" || "$JOB_TRACKER_START" =~ ^(N/A|Unknown)$ ]]; then
  JOB_TRACKER_START="$(date -Is)"
fi
if [[ -z "$JOB_TRACKER_DURATION" ]]; then
  JOB_TRACKER_DURATION="0:00"
fi

TRACKER_LOG_TAIL_LINES="${TRACKER_LOG_TAIL_LINES:-80}"
TRACKER_PAYLOAD_MAX_CHARS="${TRACKER_PAYLOAD_MAX_CHARS:-28000}"
TRACKER_SLURM_ERR="${TRACKER_SLURM_ERR:-${FILE_LOCATION}/${LOG_FILE%.out}.err}"
TRACKER_LOG="${TRACKER_LOG:-${FILE_LOCATION}/logs/tracker_${JOB_ID}.log}"

# Final HTTP body for /update: jq produces valid JSON (newlines/quotes in logs break hand-built JSON).
# Failed jobs: stderr tail first, then stdout. Successful: stdout; stderr only if the .err file is non-empty.
_tracker_emit_final_payload () {
  local _ec="$1" _stat="$2" _fs="$3" _fd="$4" _opath="$5" _epath="$6"
  local _n="${TRACKER_LOG_TAIL_LINES:-80}"
  local _cap="${TRACKER_PAYLOAD_MAX_CHARS:-28000}"
  local _ot _et _blob

  _ot="$(tail -n "${_n}" "${_opath}" 2>/dev/null || true)"
  _et="$(tail -n "${_n}" "${_epath}" 2>/dev/null || true)"

  if [[ "${_ec}" -ne 0 ]] || [[ "${_stat}" == "failed" ]]; then
    _blob="$(printf 'exit_code=%s\n\n=== stderr (last %s lines): %s ===\n%s\n\n=== stdout (last %s lines): %s ===\n%s\n' \
      "${_ec}" "${_n}" "${_epath}" "${_et:-[empty]}" "${_n}" "${_opath}" "${_ot:-[empty]}")"
  else
    _blob="$(printf 'exit_code=0\n\n=== stdout (last %s lines): %s ===\n%s\n' "${_n}" "${_opath}" "${_ot:-[empty]}")"
    if [[ -s "${_epath}" ]]; then
      _blob="${_blob}"$'\n'"$(printf '=== stderr (last %s lines): %s ===\n%s\n' "${_n}" "${_epath}" "${_et:-[empty]}")"
    fi
  fi

  if [[ "${#_blob}" -gt ${_cap} ]]; then
    _blob="$(printf '[truncated from head; last %s bytes kept]\n' "${_cap}"; printf '%s' "${_blob}" | tail -c "${_cap}")"
  fi

  if command -v jq &>/dev/null; then
    jq -n \
      --arg id "${JOB_ID}" \
      --arg jn "${JOB_NAME}" \
      --arg pj "${PROJECT}" \
      --arg st "${_stat}" \
      --arg fts "${_fs}" \
      --arg fd "${_fd}" \
      --arg lb "${_blob}" \
      --argjson xc "${_ec}" \
      '{job_id:$id,job_name:$jn,project:$pj,status:$st,start_time:$fts,duration:$fd,exit_code:$xc,log_tail:$lb}'
    return 0
  fi

  printf '%s\n' '{"job_id":"'"${JOB_ID}"'","job_name":"","project":"","status":"'"${_stat}"'","start_time":"","duration":"","exit_code":'"${_ec}"',"log_tail":"jq not in PATH on compute node; install jq or read Slurm logs/'"${JOB_ID}"'.err"}'
}

send_update () {
  # Some environments end up with CRLF line endings in ~/.arc_tracker_secrets.
  # Strip carriage returns so curl doesn't see "\r" in URLs/headers.
  local _render_url _secret _url
  _render_url="$(printf "%s" "${RENDER_URL:-}" | tr -d '\r')"
  _secret="$(printf "%s" "${API_SECRET:-}" | tr -d '\r')"
  _url="${_render_url%/}/update"

  local payload="$1"
  local err_file="${FILE_LOCATION}/logs/curl_${JOB_ID}.err"
  mkdir -p "${FILE_LOCATION}/logs"
  : >"${err_file}"

  local http_code="000" rc=0
  http_code="$(curl --max-time 60 \
       --retry 5 \
       --retry-delay 5 \
       --retry-connrefused \
       -sS -o /dev/null -w "%{http_code}" \
       -X POST "${_url}" \
       -H "Authorization: Bearer ${_secret}" \
       -H "Content-Type: application/json" \
       -d "${payload}" 2>>"${err_file}")" || true
  rc=$?
  [[ "${rc}" -ne 0 ]] && http_code="000"

  local _http_ok=0
  if [[ "${http_code}" =~ ^[0-9]+$ ]] && (( http_code >= 200 && http_code < 300 )); then
    _http_ok=1
  fi
  if [[ -n "${TRACKER_LOG:-}" && "${_http_ok}" -eq 0 ]]; then
    {
      printf '%s tracker POST failed http_code=%s curl_rc=%s job_id=%s\n' "$(date -Is)" "${http_code}" "${rc}" "${JOB_ID}"
      printf 'payload_bytes=%s\n' "${#payload}"
      echo "--- last lines Slurm stderr (${TRACKER_SLURM_ERR})"
      tail -n "${TRACKER_LOG_TAIL_LINES:-40}" "${TRACKER_SLURM_ERR:-/dev/null}" 2>/dev/null || echo "(stderr log missing)"
      echo "--- curl stderr"
      tail -n 40 "${err_file}"
    } >>"${TRACKER_LOG}"
  fi

  return 0
}

sync_queue () {
  # Grab all jobs for the current user: JobID, State, Name, StartTime, Duration
  squeue -u $USER -h -o "%A %T %j %S %M" | while read -r q_job_id q_state q_job_name q_start_time q_duration; do
    # Skip our own job since we handle our detailed updates directly
    if [ "$q_job_id" == "$JOB_ID" ]; then
      continue
    fi

    # Map SLURM status to our Notion status
    local mapped_status="running"
    if [ "$q_state" == "PENDING" ]; then
      mapped_status="queued"
    elif [ "$q_state" == "COMPLETED" ] || [ "$q_state" == "CG" ]; then
      mapped_status="completed"
    elif [[ "$q_state" == *"FAIL"* || "$q_state" == *"CANCEL"* || "$q_state" == *"TIMEOUT"* ]]; then
      mapped_status="failed"
    fi

    # Update other jobs with basic info and time info
    send_update "{
      \"job_id\": \"$q_job_id\",
      \"job_name\": \"$q_job_name\",
      \"status\": \"$mapped_status\",
      \"start_time\": \"$q_start_time\",
      \"duration\": \"$q_duration\"
    }"
  done
}

final_update() {
  # Always attempt to post a final state update, even if the job exits early (set -e, signals, etc).
  local ec=$?
  local status="completed"
  if [ "$ec" -ne 0 ]; then
    status="failed"
  fi

  local _final_start="$JOB_TRACKER_START"
  local _final_dur="$JOB_TRACKER_DURATION"
  # Prefer Slurm accounting actuals on exit (more reliable than squeue once the job is finishing).
  _sacc_line="$(sacct "${_TRACK_SACCT[@]}" -j "$JOB_ID" -n -X --format=Start,Elapsed -P --noheader 2>/dev/null | head -1)"
  if [[ -n "$_sacc_line" ]]; then
    IFS='|' read -r _s_start _s_elapsed <<<"$_sacc_line"
    [[ -n "${_s_start:-}" && "${_s_start:-}" != "Unknown" ]] && _final_start="$_s_start"
    [[ -n "${_s_elapsed:-}" && "${_s_elapsed:-}" != "0:00" ]] && _final_dur="$_s_elapsed"
  fi

  local _payload
  _payload="$(_tracker_emit_final_payload "${ec}" "${status}" "${_final_start}" "${_final_dur}" "${FILE_LOCATION}/${LOG_FILE}" "${FILE_LOCATION}/${LOG_FILE%.out}.err")"

  send_update "${_payload}" || true

  # Sync the rest of the queue again on completion
  sync_queue || true
}
trap final_update EXIT

# --- start ---
send_update "{
  \"job_id\": \"$JOB_ID\",
  \"job_name\": \"$JOB_NAME\",
  \"project\": \"$PROJECT\",
  \"data_output\": \"$DATA_OUTPUT\",
  \"file_location\": \"$FILE_LOCATION\",
  \"status\": \"started\",
  \"start_time\": \"$JOB_TRACKER_START\",
  \"duration\": \"$JOB_TRACKER_DURATION\"
}"

# Sync the rest of the queue on startup
sync_queue

# --- run job ---
# === USER COMMAND START ===
set -euo pipefail
export HF_TOKEN=''
export MODEL='meta-llama/Llama-3.1-8B'
export TP_DEGREES='1'
export HARDWARE='V100_700MHz'
export GPU_FREQ_MHZ='700'
export FULL_PROFILE='0'
export VERBOSITY='--verbose'
export HF_CACHE_ROOT='/data/engs-glass/engs2950/infra/hf_cache'
export VLLM_IMAGE='docker://vllm/vllm-openai:v0.19.0'
export CONTAINER_RUNTIME='apptainer'
export DTYPE='float16'
export SKIP_SKEW=''
bash "/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim/profiler/jobs/run_arc_v100_profile.sh"
# === USER COMMAND END ===