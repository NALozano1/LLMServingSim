#!/usr/bin/env bash
# Submit V100 tp1 vLLM throughput benchmark (default: Qwen1.5-MoE).
#
#   ./profiler/jobs/submit_arc_v100_bench_throughput.sh
#   MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat V100_BENCH_PRESET=qwen15-v100-tight ./profiler/jobs/submit_arc_v100_bench_throughput.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_bench_throughput.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered" \
  "${ROOT}/bench/results/V100"

MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
SAFE="$(echo "${MODEL}" | tr '/:' '__' | tr -cd 'A-Za-z0-9_-')"
TIME="${TIME:-02:00:00}"
NUM_REQS="${NUM_REQS:-100}"

# Optional locked-clock bench (§1 per-frequency ground truth). When set, the
# runner locks the GPU clock, tags output V100_<MHz>, and runs the same
# pre-flight read-back + post-flight audit_gpu_clocks.py as the profiler.
GPU_FREQ_MHZ="${GPU_FREQ_MHZ:-}"
GPU_FREQ_VERIFY_STRICT="${GPU_FREQ_VERIFY_STRICT:-1}"
# DEPEND_JID chains jobs sequentially so two locked-clock benches never share a
# GPU and fight over its clock (afterany: run regardless of prior exit status).
DEPEND_JID="${DEPEND_JID:-}"
if [[ -n "${GPU_FREQ_MHZ}" ]]; then
  HARDWARE="${HARDWARE:-V100_${GPU_FREQ_MHZ}MHz}"
  JOB_NAME="${JOB_NAME:-llmsim_bench_v100_${GPU_FREQ_MHZ}MHz_${SAFE:0:16}}"
else
  HARDWARE="${HARDWARE:-V100}"
  JOB_NAME="${JOB_NAME:-llmsim_bench_v100_tp1_${SAFE:0:20}}"
fi

rendered="${ROOT}/profiler/jobs/rendered/${JOB_NAME}.sbatch"
cmd_frag="${ROOT}/profiler/jobs/rendered/_${JOB_NAME}_cmd.sh"

read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL}'
export TP_SIZE='1'
export DTYPE='float16'
export NUM_REQS='${NUM_REQS}'
export SPS='${SPS:-10}'
export SEED='${SEED:-42}'
export V100_BENCH_PRESET='${V100_BENCH_PRESET:-auto}'
export GPU_FREQ_MHZ='${GPU_FREQ_MHZ}'
export HARDWARE='${HARDWARE}'
export GPU_FREQ_VERIFY_STRICT='${GPU_FREQ_VERIFY_STRICT}'
export SHAREGPT_FIX_LEN='${SHAREGPT_FIX_LEN:-}'
export FIX_INPUT_LENGTH='${FIX_INPUT_LENGTH:-128}'
export FIX_OUTPUT_LENGTH='${FIX_OUTPUT_LENGTH:-512}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
printf '%s' "${CMD}" > "${cmd_frag}"

export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
export JOB_NAME="${JOB_NAME}" PROJECT=engs2950 PARTITION=interactive
export DATA_OUTPUT="${ROOT}/bench/results/V100"
python3 - <<'PY'
from pathlib import Path
import os
template = Path(os.environ["TEMPLATE"]).read_text()
cmd = Path(os.environ["CMD_FRAG"]).read_text()
for k, v in {
    "{{JOB_NAME}}": os.environ["JOB_NAME"],
    "{{PROJECT}}": os.environ["PROJECT"],
    "{{PARTITION}}": os.environ["PARTITION"],
    "{{DATA_OUTPUT}}": os.environ["DATA_OUTPUT"],
    "{{COMMAND}}": cmd,
}.items():
    template = template.replace(k, v)
Path(os.environ["RENDERED"]).write_text(template)
PY

# A chained job waits on its dependency, not on a free interactive slot.
if [[ "${SKIP_INTERACTIVE_WAIT:-0}" != "1" && -z "${DEPEND_JID}" ]]; then
  arc_wait_htc_interactive_slot interactive || true
fi

dep_args=()
if [[ -n "${DEPEND_JID}" ]]; then
  dep_args+=(--dependency="afterany:${DEPEND_JID}")
fi

jid_raw=$(sbatch --parsable --clusters=htc --account=engs-glass \
  --partition=interactive --gres=gpu:v100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=64G --time="${TIME}" --job-name="${JOB_NAME}" \
  --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" --mail-type=BEGIN,END,FAIL \
  --output="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.out" \
  --error="${ROOT}/profiler/jobs/logs/${JOB_NAME}_%j.err" \
  "${dep_args[@]}" \
  "${rendered}")
jid="${jid_raw%%;*}"

echo "Submitted ${jid}  MODEL=${MODEL}  HARDWARE=${HARDWARE}  tp=1  NUM_REQS=${NUM_REQS}  dep=${DEPEND_JID:-none}"
echo "Logs: ${ROOT}/profiler/jobs/logs/${JOB_NAME}_${jid}.out"
echo "${jid}"
