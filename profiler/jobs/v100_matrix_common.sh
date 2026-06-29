#!/usr/bin/env bash
# Shared model list, completion checks, freq sweep helpers for V100 profiler jobs.
#
# shellcheck shell=bash
# shellcheck disable=SC2034  # V100_MATRIX_MODELS used by callers

_ENGS2950_ROOT="${ENGS2950_ROOT:-/data/engs-glass/engs2950}"
# shellcheck source=/dev/null
source "${_ENGS2950_ROOT}/shared/scripts/load_hf_token.sh"

V100_MATRIX_MODELS=(
  "meta-llama/Llama-3.1-8B"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "mistralai/Mixtral-8x7B-v0.1"
  "Qwen/Qwen3-32B"
  "meta-llama/Llama-3.1-70B"
)

V100_BASELINE_HARDWARE="${V100_BASELINE_HARDWARE:-V100}"

# Tesla V100-SXM2 typical lock range on ARC (override on submit).
V100_FREQ_MIN_MHZ="${V100_FREQ_MIN_MHZ:-900}"
V100_FREQ_MAX_MHZ="${V100_FREQ_MAX_MHZ:-1530}"
V100_FREQ_STEP_MHZ="${V100_FREQ_STEP_MHZ:-200}"

v100_matrix_model_list() {
  if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    local list=(${MODELS})
    printf '%s\n' "${list[@]}"
    return
  fi
  printf '%s\n' "${V100_MATRIX_MODELS[@]}"
}

v100_matrix_safe_name() {
  echo "$1" | tr '/:' '__' | tr -cd 'A-Za-z0-9_-'
}

v100_matrix_variant_tag() {
  local dtype="${1:-float16}"
  case "$dtype" in
    float16) echo "fp16" ;;
    bfloat16) echo "bf16" ;;
    float32) echo "fp32" ;;
    fp8) echo "fp8" ;;
    *) echo "$dtype" ;;
  esac
}

# HARDWARE label under profiler/perf/ (default clocks = V100).
v100_hardware_label() {
  local mhz="${1:-${GPU_FREQ_MHZ:-}}"
  if [[ -n "$mhz" ]]; then
    echo "V100_${mhz}MHz"
  else
    echo "${V100_BASELINE_HARDWARE}"
  fi
}

# MHz values for locked reruns (excludes default-boost baseline tag V100).
v100_freq_list() {
  local min="${V100_FREQ_MIN_MHZ}"
  local max="${V100_FREQ_MAX_MHZ}"
  local step="${V100_FREQ_STEP_MHZ}"
  local mhz
  for ((mhz = min; mhz <= max; mhz += step)); do
    echo "$mhz"
  done
}

# True when meta.yaml exists (full TP sweep done for that hardware tag).
v100_matrix_model_complete() {
  local root="$1"
  local model="$2"
  local hardware="${3:-${V100_BASELINE_HARDWARE}}"
  local variant
  variant="$(v100_matrix_variant_tag "${DTYPE:-float16}")"
  [[ -f "${root}/profiler/perf/${hardware}/${model}/${variant}/meta.yaml" ]]
}

v100_baseline_matrix_complete() {
  local root="$1"
  local model
  while IFS= read -r model; do
    [[ -z "$model" ]] && continue
    v100_matrix_model_complete "$root" "$model" "${V100_BASELINE_HARDWARE}" || return 1
  done < <(v100_matrix_model_list)
}

# Submit one model profile job; echoes Slurm job id on success.
# Requires caller to set: ROOT RUNNER TEMPLATE PROJECT PARTITION DATA_OUTPUT TIME
# JOB_NAME_PREFIX SUBMIT_LOG MANIFEST CONTINUE_ON_ERROR HF_TOKEN FULL_PROFILE VERBOSITY
# Optional: GPU_FREQ_MHZ (locks clocks + sets HARDWARE if unset)
# $5 replica_idx — when set (e.g. 0..7), appends _g{i} to hardware tag, job name, and
#   rendered paths so per-GPU replica runs land in separate perf directories.
v100_matrix_submit_profile_job() {
  local model="$1"
  local hardware="$2"
  local gpu_freq_mhz="${3:-}"
  local prev_jid="${4:-}"
  local replica_idx="${5:-}"

  # Per-GPU replica suffix: _g0 .. _g7. Applied to hardware tag and file names so
  # each replica's perf tables land in a distinct directory for outlier comparison.
  local rep_suffix=""
  if [[ -n "$replica_idx" ]]; then
    rep_suffix="_g${replica_idx}"
    hardware="${hardware}${rep_suffix}"
  fi

  local safe freq_tag job_name rendered cmd_frag dep_args jid_raw jid
  safe="$(v100_matrix_safe_name "${model}")"
  if [[ -n "$gpu_freq_mhz" ]]; then
    freq_tag="${gpu_freq_mhz}MHz"
    job_name="${JOB_NAME_PREFIX}_${freq_tag}_${safe:0:20}${rep_suffix}"
    rendered="${ROOT}/profiler/jobs/rendered/llmsim_prof_v100_${gpu_freq_mhz}MHz_${safe}${rep_suffix}.sbatch"
    cmd_frag="${ROOT}/profiler/jobs/rendered/_llmsim_prof_v100_${gpu_freq_mhz}MHz_${safe}${rep_suffix}_cmd.sh"
  else
    job_name="${JOB_NAME_PREFIX}_${safe:0:28}${rep_suffix}"
    rendered="${ROOT}/profiler/jobs/rendered/llmsim_prof_v100_${safe}${rep_suffix}.sbatch"
    cmd_frag="${ROOT}/profiler/jobs/rendered/_llmsim_prof_v100_${safe}${rep_suffix}_cmd.sh"
  fi

  read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${model}'
export TP_DEGREES='${TP_DEGREES:-1,2,4}'
export HARDWARE='${hardware}'
export GPU_FREQ_MHZ='${gpu_freq_mhz}'
export FULL_PROFILE='${FULL_PROFILE:-0}'
export VERBOSITY='${VERBOSITY:-"--verbose"}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
export DTYPE='${DTYPE:-float16}'
export SKIP_SKEW='${SKIP_SKEW:-}'
export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS:-2048}'
export MAX_NUM_SEQS='${MAX_NUM_SEQS:-256}'
export ATTENTION_MAX_KV='${ATTENTION_MAX_KV:-8192}'
export MEASUREMENT_ITERATIONS='${MEASUREMENT_ITERATIONS:-3}'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"

  export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT="${PROJECT}" PARTITION="${PARTITION}" DATA_OUTPUT="${DATA_OUTPUT}"
  python3 - <<'PY'
from pathlib import Path
import os

template = Path(os.environ["TEMPLATE"]).read_text()
cmd = Path(os.environ["CMD_FRAG"]).read_text()
repl = {
    "{{JOB_NAME}}": os.environ["JOB_NAME"],
    "{{PROJECT}}": os.environ["PROJECT"],
    "{{PARTITION}}": os.environ["PARTITION"],
    "{{DATA_OUTPUT}}": os.environ["DATA_OUTPUT"],
    "{{COMMAND}}": cmd,
}
for k, v in repl.items():
    if k not in template:
        raise SystemExit(f"missing {k} in template")
    template = template.replace(k, v)
Path(os.environ["RENDERED"]).write_text(template)
PY

  dep_args=()
  if [[ -n "${prev_jid}" ]]; then
    if [[ "${CONTINUE_ON_ERROR:-1}" == "1" ]]; then
      dep_args+=(--dependency="afterany:${prev_jid}")
    else
      dep_args+=(--dependency="afterok:${prev_jid}")
    fi
  fi

  # Compute GPU count from the max tensor-parallel degree so tp4 jobs get 4 GPUs.
  local _max_tp=1
  local _tp_val
  IFS=',' read -ra _tp_vals <<< "${TP_DEGREES:-1}"
  for _tp_val in "${_tp_vals[@]}"; do
    (( _tp_val > _max_tp )) && _max_tp="${_tp_val}"
  done

  jid_raw=$(sbatch --parsable \
    --clusters=htc \
    --account=engs-glass \
    --partition="${PARTITION}" \
    --gres=gpu:v100:${_max_tp} \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --time="${TIME}" \
    --job-name="${job_name}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
    --mail-type=BEGIN,END,FAIL \
    --output="${ROOT}/profiler/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/profiler/jobs/logs/${job_name}_%j.err" \
    "${dep_args[@]}" \
    "${rendered}")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid}" "${model}" "${hardware}" "${gpu_freq_mhz:-default}" \
    >> "${MANIFEST}"

  {
    echo "  job ${jid}  MODEL=${model}  HARDWARE=${hardware}  GPU_FREQ_MHZ=${gpu_freq_mhz:-default}"
    echo "  perf  ${ROOT}/profiler/perf/${hardware}/${model}/"
  } | tee -a "${SUBMIT_LOG}" >&2

  echo "${jid}"
}
