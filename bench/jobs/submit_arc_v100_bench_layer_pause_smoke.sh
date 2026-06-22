#!/usr/bin/env bash
# Submit real vLLM bench layer-pause smoke (not profiler slice).
#
#   export HF_TOKEN=...
#   MODEL=microsoft/Phi-tiny-MoE-instruct ./bench/jobs/submit_arc_v100_bench_layer_pause_smoke.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/bench/jobs/run_arc_v100_bench_layer_pause_smoke.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"

mkdir -p "${ROOT}/bench/jobs/logs" "${ROOT}/profiler/jobs/rendered"

MODEL="${MODEL:-microsoft/Phi-tiny-MoE-instruct}"
TIME="${TIME:-01:00:00}"
JOB_NAME="llmsim_bench_layer_pause_smoke"
rendered="${ROOT}/profiler/jobs/rendered/${JOB_NAME}.sbatch"
cmd_frag="${ROOT}/profiler/jobs/rendered/_${JOB_NAME}_cmd.sh"

read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
printf '%s' "${CMD}" > "${cmd_frag}"

export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
export JOB_NAME="${JOB_NAME}" PROJECT=engs2950 PARTITION=interactive
export DATA_OUTPUT="${ROOT}/bench/results"
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
    template = template.replace(k, v)
Path(os.environ["RENDERED"]).write_text(template)
PY

arc_wait_htc_interactive_slot interactive || true

jid_raw=$(sbatch -M htc --parsable \
  --clusters=htc --account=engs-glass --partition=interactive \
  --gres=gpu:v100:1 --nodes=1 --cpus-per-task=8 --mem=32G \
  --time="${TIME}" --job-name="${JOB_NAME}" \
  --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
  --mail-type=BEGIN,END,FAIL \
  --output="${ROOT}/bench/jobs/logs/${JOB_NAME}_%j.out" \
  --error="${ROOT}/bench/jobs/logs/${JOB_NAME}_%j.err" \
  "${rendered}")
jid="${jid_raw%%;*}"

echo "Submitted job ${jid}"
echo "  log: ${ROOT}/bench/jobs/logs/${JOB_NAME}_${jid}.out"
echo "  out: ${ROOT}/bench/results/V100_layer_pause_smoke/<model>/tp1/${jid}/"
