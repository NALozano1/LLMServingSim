#!/usr/bin/env bash
# Submit V100 A/B: nopause vs pause-only layer barriers.
#
#   export HF_TOKEN=...
#   ./profiler/jobs/submit_arc_v100_layer_pause_ab.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_layer_pause_ab.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered"

TIME="${TIME:-00:10:00}"
JOB_NAME="llmsim_layer_pause_ab"
rendered="${ROOT}/profiler/jobs/rendered/${JOB_NAME}.sbatch"
cmd_frag="${ROOT}/profiler/jobs/rendered/_${JOB_NAME}_cmd.sh"

read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL:-meta-llama/Llama-3.1-8B}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
export PAUSE_ONLY_DELAY_SEC='${PAUSE_ONLY_DELAY_SEC:-0.05}'
bash "${RUNNER}"
EOF
printf '%s' "${CMD}" > "${cmd_frag}"

export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
export JOB_NAME="${JOB_NAME}" PROJECT=engs2950 PARTITION=devel
export DATA_OUTPUT="${ROOT}/profiler/perf"
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

arc_wait_htc_devel_slot devel || exit 1

jid_raw="$(sbatch -M htc \
  --clusters=htc \
  --account=engs-glass \
  --partition=devel \
  --gres=gpu:v100:1 \
  --nodes=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time="${TIME}" \
  --job-name="${JOB_NAME}" \
  --mail-user=alex.lozano@eng.ox.ac.uk \
  --mail-type=END,FAIL \
  --output="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.out" \
  --error="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.err" \
  "${rendered}")"
jid="${jid_raw##* }"
echo "Submitted job ${jid}"
echo "  out: ${ROOT}/profiler/jobs/logs/${JOB_NAME}_${jid}.out"
echo "  ab_compare: ${ROOT}/profiler/perf/V100_layer_pause_ab_pause/meta-llama/Llama-3.1-8B/fp16/tp1/ab_compare.json"
