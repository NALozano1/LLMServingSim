#!/usr/bin/env bash
# Submit idle GPU + server power captures for each GPU family + V100 DVFS sweep.
#
# GPU types (1 GPU each, default clocks): a100, h100, v100, l40s
# V100 DVFS chain (same freqs as profiler smoke): default + 700,900,1100,1300 MHz
#
#   ./profiler/jobs/submit_arc_gpu_power_capture.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_power_capture.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered" \
  "${ROOT}/profiler/jobs/checkpoints" "${ROOT}/profiler/power/data"

PROJECT="${PROJECT:-engs2950}"
DATA_OUTPUT="${DATA_OUTPUT:-${ROOT}/profiler/power/data}"
TIME="${TIME:-00:15:00}"
IDLE_SEC="${IDLE_SEC:-120}"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/gpu_power_capture_jobs.tsv"
SUBMIT_LOG="${ROOT}/profiler/jobs/logs/gpu_power_submit_$(date -u +%Y%m%d_%H%M%S).log"

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\tjob_id\tgpu_type\tgpu_freq_mhz\tpartition\tstatus\n' > "${MANIFEST}"
fi

get_partition() {
  case "$1" in
    v100) echo "interactive" ;;
    *) echo "short" ;;
  esac
}

submit_power_job() {
  local gpu_type="$1"
  local gpu_freq_mhz="${2:-}"
  local prev_jid="${3:-}"

  local freq_tag job_name rendered cmd_frag dep_args jid_raw jid
  if [[ -n "${gpu_freq_mhz}" ]]; then
    freq_tag="${gpu_freq_mhz}MHz"
    job_name="llmsim_pwr_${gpu_type}_${freq_tag}"
  else
    freq_tag="default"
    job_name="llmsim_pwr_${gpu_type}_default"
  fi

  safe="${gpu_type}_${freq_tag}"
  rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"
  partition="$(get_partition "${gpu_type}")"

  read -r -d '' CMD <<EOF || true
set -euo pipefail
export GPU_TYPE='${gpu_type}'
export GPU_FREQ_MHZ='${gpu_freq_mhz}'
export IDLE_SEC='${IDLE_SEC}'
export POWER_TAG='${safe}'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"

  export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT="${PROJECT}" PARTITION="${partition}" DATA_OUTPUT="${DATA_OUTPUT}"
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
    dep_args+=(--dependency="afterany:${prev_jid}")
  fi

  jid_raw=$(sbatch --parsable \
    --clusters=htc \
    --account=engs-glass \
    --partition="${partition}" \
    --gres="gpu:${gpu_type}:1" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
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
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid}" "${gpu_type}" "${gpu_freq_mhz:-default}" "${partition}" \
    >> "${MANIFEST}"

  echo "  job ${jid}  ${gpu_type}  freq=${gpu_freq_mhz:-default}  partition=${partition}" | tee -a "${SUBMIT_LOG}" >&2
  echo "${jid}"
}

{
  echo "=== GPU + server power capture matrix ==="
  echo "IDLE_SEC=${IDLE_SEC}  TIME=${TIME}"
  echo "V100 DVFS freqs: default + ${V100_FREQ_MIN_MHZ:-700}-${V100_FREQ_MAX_MHZ:-1300} step ${V100_FREQ_STEP_MHZ:-200}"
} | tee "${SUBMIT_LOG}"

export V100_FREQ_MIN_MHZ=700
export V100_FREQ_MAX_MHZ=1300
export V100_FREQ_STEP_MHZ=200

arc_wait_htc_interactive_slot interactive || true

v100_prev=""
for gpu_type in a100 h100 l40s; do
  echo "SUBMIT ${gpu_type} default (parallel)" | tee -a "${SUBMIT_LOG}"
  submit_power_job "${gpu_type}" "" "" | tee -a "${SUBMIT_LOG}" >&2
done

echo "SUBMIT v100 default + DVFS chain" | tee -a "${SUBMIT_LOG}"
v100_prev="$(submit_power_job v100 "" "")"
for mhz in $(v100_freq_list); do
  v100_prev="$(submit_power_job v100 "${mhz}" "${v100_prev}")"
done
prev_jid="${v100_prev}"

{
  echo "=== Power capture submitted. Last job: ${prev_jid} ==="
  echo "Manifest: ${MANIFEST}"
  echo "Outputs:  ${ROOT}/profiler/power/data/"
} | tee -a "${SUBMIT_LOG}"

echo "${prev_jid}"
