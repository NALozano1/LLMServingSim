#!/usr/bin/env bash
# Submit one Slurm job: incremental ATTENTION_MAX_KV probe for Qwen3-30B on V100.
#
# Does NOT cancel other llmsim jobs (unlike submit_arc_v100_qwen30b_dvfs.sh).
#
#   ./profiler/jobs/submit_arc_v100_qwen30b_kv_probe.sh
#
# Override KV ladder or per-step timeout:
#   KV_PROBE_STEPS="512 1024 2048 4096" KV_PROBE_STEP_TIMEOUT_SEC=10800 TIME=24:00:00 \\
#     ./profiler/jobs/submit_arc_v100_qwen30b_kv_probe.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_kv_probe.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered" \
  "${ROOT}/profiler/jobs/checkpoints"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
JOB_NAME="${JOB_NAME_PREFIX:-llmsim_qwen30b_kvprobe}"
PROJECT="${PROJECT:-engs2950}"
PARTITION="interactive"
DATA_OUTPUT="${DATA_OUTPUT:-${ROOT}/profiler/perf}"
TIME="${TIME:-18:00:00}"

export MODEL
export DTYPE="${DTYPE:-float16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
export KV_PROBE_STEPS="${KV_PROBE_STEPS:-256 512 1024 2048 4096 8192}"
export KV_PROBE_STEP_TIMEOUT_SEC="${KV_PROBE_STEP_TIMEOUT_SEC:-7200}"

safe="$(v100_matrix_safe_name "${MODEL}")"
rendered="${ROOT}/profiler/jobs/rendered/${JOB_NAME}_${safe}.sbatch"
cmd_frag="${ROOT}/profiler/jobs/rendered/_${JOB_NAME}_${safe}_cmd.sh"
manifest="${ROOT}/profiler/jobs/checkpoints/qwen30b_kv_probe_jobs.tsv"
submit_log="${ROOT}/profiler/jobs/logs/qwen30b_kv_probe_submit_$(date -u +%Y%m%d_%H%M%S).log"

read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL}'
export DTYPE='${DTYPE}'
export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}'
export MAX_NUM_SEQS='${MAX_NUM_SEQS}'
export MEASUREMENT_ITERATIONS='${MEASUREMENT_ITERATIONS}'
export KV_PROBE_STEPS='${KV_PROBE_STEPS}'
export KV_PROBE_STEP_TIMEOUT_SEC='${KV_PROBE_STEP_TIMEOUT_SEC}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
printf '%s' "${CMD}" > "${cmd_frag}"

export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
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
PY

{
  echo "=== Submit Qwen3-30B KV probe (single job) ==="
  echo "MODEL=${MODEL}"
  echo "KV_PROBE_STEPS=${KV_PROBE_STEPS}"
  echo "KV_PROBE_STEP_TIMEOUT_SEC=${KV_PROBE_STEP_TIMEOUT_SEC} (per step)"
  echo "MSQ=${MAX_NUM_SEQS} MNBT=${MAX_NUM_BATCHED_TOKENS} TIME=${TIME}"
} | tee "${submit_log}"

arc_wait_htc_interactive_slot interactive || exit 1

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
  "${rendered}")
jid="${jid_raw%%;*}"

if [[ ! -f "${manifest}" ]]; then
  printf 'submitted_at\tjob_id\tmodel\tkv_steps\tstep_timeout_sec\tstatus\n' > "${manifest}"
fi
printf '%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid}" "${MODEL}" "${KV_PROBE_STEPS}" "${KV_PROBE_STEP_TIMEOUT_SEC}" \
  >> "${manifest}"

{
  echo "Submitted job ${jid}"
  echo "Watch: squeue -M htc -j ${jid}"
  echo "Results TSV: ${ROOT}/profiler/jobs/checkpoints/qwen30b_kv_probe_results.tsv"
  echo "Logs: ${ROOT}/profiler/jobs/logs/${JOB_NAME}_${jid}.out"
} | tee -a "${submit_log}"

echo "${jid}"
