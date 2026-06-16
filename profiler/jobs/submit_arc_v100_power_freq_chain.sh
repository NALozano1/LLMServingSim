#!/usr/bin/env bash
# Chain V100 DVFS power captures after an existing job (700,900,1100,1300 MHz).
# Usage: PREV_JID=7972728 ./profiler/jobs/submit_arc_v100_power_freq_chain.sh
set -euo pipefail

PREV_JID="${PREV_JID:?set PREV_JID to the v100 default power job}"
ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
export V100_FREQ_MIN_MHZ=700 V100_FREQ_MAX_MHZ=1300 V100_FREQ_STEP_MHZ=200
# shellcheck source=/dev/null
source "${ROOT}/profiler/jobs/v100_matrix_common.sh"

RUNNER="${ROOT}/profiler/jobs/run_arc_power_capture.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/gpu_power_capture_jobs.tsv"
TIME="${TIME:-00:15:00}"
IDLE_SEC="${IDLE_SEC:-120}"

prev="${PREV_JID}"
for mhz in $(v100_freq_list); do
  job_name="llmsim_pwr_v100_${mhz}MHz"
  cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"
  rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  read -r -d '' CMD <<EOF || true
set -euo pipefail
export GPU_TYPE='v100'
export GPU_FREQ_MHZ='${mhz}'
export IDLE_SEC='${IDLE_SEC}'
export POWER_TAG='v100_${mhz}MHz'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"
  export TEMPLATE CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT=engs2950 PARTITION=interactive
  export DATA_OUTPUT="${ROOT}/profiler/power/data"
  python3 - <<'PY'
from pathlib import Path
import os
t = Path(os.environ["TEMPLATE"]).read_text()
cmd = Path(os.environ["CMD_FRAG"]).read_text()
for k, v in {
    "{{JOB_NAME}}": os.environ["JOB_NAME"],
    "{{PROJECT}}": os.environ["PROJECT"],
    "{{PARTITION}}": os.environ["PARTITION"],
    "{{DATA_OUTPUT}}": os.environ["DATA_OUTPUT"],
    "{{COMMAND}}": cmd,
}.items():
    t = t.replace(k, v)
Path(os.environ["RENDERED"]).write_text(t)
PY
  jid_raw=$(sbatch --parsable --clusters=htc --account=engs-glass --partition=interactive \
    --gres=gpu:v100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time="${TIME}" \
    --job-name="${job_name}" --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
    --mail-type=BEGIN,END,FAIL \
    --dependency="afterany:${prev}" \
    --output="${ROOT}/profiler/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/profiler/jobs/logs/${job_name}_%j.err" \
    "${rendered}")
  prev="${jid_raw%%;*}"
  printf '%s\t%s\tv100\t%s\tinteractive\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${prev}" "${mhz}" >> "${MANIFEST}"
  echo "job ${prev}  v100  ${mhz}MHz"
done
echo "last=${prev}"
