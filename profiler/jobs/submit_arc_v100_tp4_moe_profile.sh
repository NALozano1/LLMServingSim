#!/usr/bin/env bash
# Submit LLMServingSim MoE profiler on HTC interactive (1× V100).
#
# Profiles TP=1,4 on a single V100 via vLLM shape emulation (see run_arc_v100_profile.sh).
# Default model: Qwen3-30B-A3B MoE. EP is simulated later in LLMServingSim, not profiled.
#
# Examples:
#   export HF_TOKEN=hf_...
#   ./profiler/jobs/submit_arc_v100_tp4_moe_profile.sh
#
#   FULL_PROFILE=1 TIME=06:00:00 ./profiler/jobs/submit_arc_v100_tp4_moe_profile.sh
#
#   VERBOSITY="--silent" ./profiler/jobs/submit_arc_v100_tp4_moe_profile.sh
#
#   MODEL=mistralai/Mixtral-8x7B-v0.1 TP_DEGREES=1,4 ./profiler/jobs/submit_arc_v100_tp4_moe_profile.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_profile.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
RENDERED="${ROOT}/profiler/jobs/rendered/llmservingsim_profiler_v100.sbatch"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered"

JOB_NAME="${JOB_NAME:-llmsim_prof_v100}"
PROJECT="${PROJECT:-engs2950}"
PARTITION="interactive"
DATA_OUTPUT="${DATA_OUTPUT:-${ROOT}/profiler/perf}"
TIME="${TIME:-03:00:00}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TP_DEGREES="${TP_DEGREES:-1,4}"
HARDWARE="${HARDWARE:-V100}"
FULL_PROFILE="${FULL_PROFILE:-0}"
VERBOSITY="${VERBOSITY:-"--verbose"}"
HF_TOKEN="${HF_TOKEN:-}"

arc_wait_htc_interactive_slot interactive || exit 1

read -r -d '' CMD <<EOF || true
set -euo pipefail
export HF_TOKEN='${HF_TOKEN}'
export MODEL='${MODEL}'
export TP_DEGREES='${TP_DEGREES}'
export HARDWARE='${HARDWARE}'
export FULL_PROFILE='${FULL_PROFILE}'
export VERBOSITY='${VERBOSITY}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF

CMD_FRAG="${ROOT}/profiler/jobs/rendered/_llmsim_profiler_v100_cmd.sh"
printf '%s' "${CMD}" > "${CMD_FRAG}"

export TEMPLATE="${TEMPLATE}" CMD_FRAG="${CMD_FRAG}" RENDERED="${RENDERED}"
export JOB_NAME="${JOB_NAME}" PROJECT="${PROJECT}" PARTITION="${PARTITION}" DATA_OUTPUT="${DATA_OUTPUT}"
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
print(os.environ["RENDERED"])
PY

echo "Rendered: ${RENDERED}"
jid_raw=$(sbatch --parsable \
  --clusters=htc \
  --account=engs-glass \
  --partition="${PARTITION}" \
  --gres=gpu:v100:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time="${TIME}" \
  --job-name="${JOB_NAME}" \
  --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
  --mail-type=BEGIN,END,FAIL \
  --output="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.out" \
  --error="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.err" \
  "${RENDERED}")
jid="${jid_raw%%;*}"

echo "Submitted job ${jid} (1× V100, MODEL=${MODEL}, TP_DEGREES=${TP_DEGREES}, VERBOSITY=${VERBOSITY})"
echo "  stdout: ${ROOT}/profiler/jobs/logs/${JOB_NAME}_${jid}.out"
echo "  perf:   ${ROOT}/profiler/perf/${HARDWARE}/${MODEL}/"
echo "  watch:  squeue -M htc -j ${jid}"
