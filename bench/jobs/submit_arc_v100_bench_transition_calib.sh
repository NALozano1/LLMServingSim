#!/usr/bin/env bash
# Submit V100 transition-effects calibration sweep.
#
# Generates a campaign (if not already present), then submits one chained
# SLURM job per arm.  All arms are chained afterany the previous job so only
# one GPU slot is occupied at a time.
#
# Usage:
#   export HF_TOKEN=$(cat /data/engs-glass/engs2950/.secrets/hf_token)
#   ./bench/jobs/submit_arc_v100_bench_transition_calib.sh
#
# Key env overrides:
#   MODEL=phi|qwen                     (default: phi)
#   TRANSITION_COUNTS=0,1,2,4,8,16,32  (default; must include 0)
#   ITERATIONS=3                        repeats per arm
#   APPLY_MODE=async|sync               DVFS apply mode for armed arms
#   SEED=20260624
#   TIME=00:30:00                       wall time per job
#   CAMPAIGN_DIR=<path>                 override auto-stamped directory
#   DRY_RUN=1                          print sbatch commands without submitting
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/bench/jobs/run_arc_v100_bench_transition_calib.sh"
GENERATOR="${ROOT}/bench/jobs/generate_transition_calib_sweep.py"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"

MODEL="${MODEL:-phi}"
TRANSITION_COUNTS="${TRANSITION_COUNTS:-0,1,2,4,8,16,32}"
ITERATIONS="${ITERATIONS:-3}"
APPLY_MODE="${APPLY_MODE:-async}"
SEED="${SEED:-20260624}"
TIME="${TIME:-00:30:00}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${CAMPAIGN_DIR:-}" ]]; then
  stamp="$(date -u +%Y%m%d)"
  CAMPAIGN_DIR="${ROOT}/bench/campaigns/v100_transition_calib_${MODEL}_${stamp}"
fi

mkdir -p "${ROOT}/bench/jobs/logs" "${ROOT}/profiler/jobs/rendered" "${CAMPAIGN_DIR}"

# Generate sweep specs if not already present.
if [[ ! -f "${CAMPAIGN_DIR}/run_specs.json" ]]; then
  echo "Generating transition-effects calibration sweep in ${CAMPAIGN_DIR} ..."
  python3 "${GENERATOR}" \
    --repo "${ROOT}" \
    --out-dir "${CAMPAIGN_DIR}" \
    --model "${MODEL}" \
    --transition-counts "${TRANSITION_COUNTS}" \
    --iterations "${ITERATIONS}" \
    --apply-mode "${APPLY_MODE}" \
    --seed "${SEED}"
fi

SUBMIT_LOG="${CAMPAIGN_DIR}/submit_$(date -u +%Y%m%d_%H%M%S).log"
MANIFEST="${CAMPAIGN_DIR}/jobs_manifest.tsv"

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\trun_id\tjob_id\tarm_label\tn_transitions\titeration\tstatus\n' > "${MANIFEST}"
fi

submit_one() {
  local run_id="$1"
  local prev_jid="${2:-}"
  local run_dir="${CAMPAIGN_DIR}/runs/${run_id}"
  local spec_path="${run_dir}/calib_spec.json"

  local arm_label n_transitions iteration
  arm_label="$(python3 -c "import json; print(json.load(open('${spec_path}'))['arm_label'])")"
  n_transitions="$(python3 -c "import json; print(json.load(open('${spec_path}'))['dvfs']['n_transitions'])")"
  iteration="$(python3 -c "import json; print(json.load(open('${spec_path}'))['iteration'])")"

  local job_name="llmsim_tcalib_${run_id}"
  local rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  local cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"

  read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export CALIB_DIR='${CAMPAIGN_DIR}'
export RUN_ID='${run_id}'
export CALIB_SPEC_JSON='${spec_path}'
export DVFS_APPLY_MODE='${APPLY_MODE}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"

  export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT=engs2950 PARTITION=interactive
  export DATA_OUTPUT="${CAMPAIGN_DIR}"
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

  local dep_args=()
  if [[ -n "${prev_jid}" ]]; then
    dep_args+=(--dependency="afterany:${prev_jid}")
  fi

  local nodelist_args=()
  if [[ -n "${NODELIST:-}" ]]; then
    nodelist_args+=(--nodelist="${NODELIST}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${job_name} run_id=${run_id} arm=${arm_label} n=${n_transitions} nodelist=${NODELIST:-any}" \
      | tee -a "${SUBMIT_LOG}"
    echo "${prev_jid:-dry}"
    return 0
  fi

  local jid_raw jid
  jid_raw=$(sbatch -M htc --parsable \
    --clusters=htc --account=engs-glass --partition=interactive \
    --gres=gpu:v100:1 --nodes=1 --cpus-per-task=8 --mem=32G \
    --exclusive \
    --time="${TIME}" --job-name="${job_name}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
    --mail-type=BEGIN,END,FAIL \
    --output="${ROOT}/bench/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/bench/jobs/logs/${job_name}_%j.err" \
    "${nodelist_args[@]}" "${dep_args[@]}" "${rendered}")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_id}" "${jid}" "${arm_label}" \
    "${n_transitions}" "${iteration}" >> "${MANIFEST}"

  echo "  job ${jid}  ${run_id}  arm=${arm_label}  n_transitions=${n_transitions}  iter=${iteration}" \
    | tee -a "${SUBMIT_LOG}" >&2
  echo "${jid}"
}

{
  echo "=== V100 transition-effects calibration sweep ==="
  echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
  echo "MODEL=${MODEL}  TRANSITION_COUNTS=${TRANSITION_COUNTS}  ITERATIONS=${ITERATIONS}"
  echo "APPLY_MODE=${APPLY_MODE}  SEED=${SEED}  TIME=${TIME}"
} | tee "${SUBMIT_LOG}"

arc_wait_htc_interactive_slot interactive || true

mapfile -t RUN_LINES < <(python3 - <<PY
import json
from pathlib import Path
specs = json.loads(Path("${CAMPAIGN_DIR}/run_specs.json").read_text())
for s in specs:
    print("\t".join([
        s["run_id"],
        s["arm_label"],
        str(s["dvfs"]["n_transitions"]),
        str(s["iteration"]),
    ]))
PY
)

prev=""
submitted=0
for line in "${RUN_LINES[@]}"; do
  IFS=$'\t' read -r run_id arm_label n_transitions iteration <<< "${line}"
  prev="$(submit_one "${run_id}" "${prev}")"
  submitted=$((submitted + 1))
done

{
  echo "=== Submitted ${submitted} transition-effects calibration jobs ==="
  echo "Last job: ${prev}"
  echo "Manifest: ${MANIFEST}"
  echo ""
  echo "Collect results with:"
  echo "  python3 ${ROOT}/bench/jobs/collect_transition_calib_results.py ${CAMPAIGN_DIR}"
} | tee -a "${SUBMIT_LOG}"

echo "${CAMPAIGN_DIR}"
