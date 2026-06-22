#!/usr/bin/env bash
# Submit V100 bench layer-boundary DVFS campaign (78 runs, same permutations as profiler).
#
#   ./bench/jobs/submit_arc_v100_bench_layer_campaign.sh
#   CAMPAIGN_DIR=bench/campaigns/v100_bench_layer_dvfs_20260622 ./bench/jobs/submit_arc_v100_bench_layer_campaign.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/bench/jobs/run_arc_v100_bench_layer_campaign.sh"
GENERATOR="${ROOT}/bench/jobs/generate_bench_layer_dvfs_campaign.py"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"

TIME="${TIME:-00:45:00}"
SEED="${SEED:-20260616}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${CAMPAIGN_DIR:-}" ]]; then
  stamp="$(date -u +%Y%m%d)"
  CAMPAIGN_DIR="${ROOT}/bench/campaigns/v100_bench_layer_dvfs_${stamp}"
fi

mkdir -p "${ROOT}/bench/jobs/logs" "${ROOT}/profiler/jobs/rendered" "${CAMPAIGN_DIR}"

if [[ ! -f "${CAMPAIGN_DIR}/run_specs.json" ]]; then
  echo "Generating bench campaign specs in ${CAMPAIGN_DIR} ..."
  python3 "${GENERATOR}" --repo "${ROOT}" --out-dir "${CAMPAIGN_DIR}" --seed "${SEED}"
fi

MANIFEST="${CAMPAIGN_DIR}/jobs_manifest.tsv"
SUBMIT_LOG="${CAMPAIGN_DIR}/submit_$(date -u +%Y%m%d_%H%M%S).log"

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\trun_id\tjob_id\tmodel\tmode\titeration\tstatus\n' > "${MANIFEST}"
fi

submit_one() {
  local run_id="$1"
  local prev_jid="${2:-}"
  local run_dir="${CAMPAIGN_DIR}/runs/${run_id}"
  local spec_path="${run_dir}/run_spec.json"

  local model mode iteration
  model="$(python3 -c "import json; print(json.load(open('${spec_path}'))['model'])")"
  mode="$(python3 -c "import json; print(json.load(open('${spec_path}'))['campaign_mode'])")"
  iteration="$(python3 -c "import json; print(json.load(open('${spec_path}'))['iteration'])")"

  local job_name="llmsim_bench_fdvfs_${run_id}"
  local rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  local cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"

  read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export CAMPAIGN_DIR='${CAMPAIGN_DIR}'
export RUN_ID='${run_id}'
export RUN_SPEC_JSON='${spec_path}'
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

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN ${job_name} run_id=${run_id}" | tee -a "${SUBMIT_LOG}"
    echo "${prev_jid:-dry}"
    return 0
  fi

  local jid_raw jid
  jid_raw=$(sbatch -M htc --parsable \
    --clusters=htc --account=engs-glass --partition=interactive \
    --gres=gpu:v100:1 --nodes=1 --cpus-per-task=8 --mem=32G \
    --time="${TIME}" --job-name="${job_name}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
    --mail-type=BEGIN,END,FAIL \
    --output="${ROOT}/bench/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/bench/jobs/logs/${job_name}_%j.err" \
    "${dep_args[@]}" "${rendered}")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_id}" "${jid}" "${model}" \
    "${mode}" "${iteration}" >> "${MANIFEST}"

  echo "  job ${jid}  ${run_id}  ${model}  mode=${mode}  iter=${iteration}" \
    | tee -a "${SUBMIT_LOG}" >&2
  echo "${jid}"
}

{
  echo "=== V100 bench layer-boundary DVFS campaign ==="
  echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
  echo "TIME=${TIME}  SEED=${SEED}"
} | tee "${SUBMIT_LOG}"

arc_wait_htc_interactive_slot interactive || true

mapfile -t RUN_LINES < <(python3 - <<PY
import json
from pathlib import Path
specs = json.loads(Path("${CAMPAIGN_DIR}/run_specs.json").read_text())
for s in specs:
    print("\t".join([
        s["run_id"],
        s["model"],
        s["campaign_mode"],
        str(s["iteration"]),
    ]))
PY
)

prev=""
submitted=0
for line in "${RUN_LINES[@]}"; do
  IFS=$'\t' read -r run_id model mode iteration <<< "${line}"
  prev="$(submit_one "${run_id}" "${prev}")"
  submitted=$((submitted + 1))
done

{
  echo "=== Submitted ${submitted} bench campaign jobs ==="
  echo "Last job: ${prev}"
  echo "Manifest: ${MANIFEST}"
  echo "Collect:  python3 ${ROOT}/bench/jobs/collect_bench_layer_campaign_results.py ${CAMPAIGN_DIR}"
} | tee -a "${SUBMIT_LOG}"

echo "${CAMPAIGN_DIR}"
