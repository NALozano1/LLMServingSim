#!/usr/bin/env bash
# Qwen1.5-MoE-A2.7B V100 tp1 throughput bench at default + DVFS freqs (700–1300 MHz).
#
# Uses python -m bench run (real vLLM). Auto-applies qwen15-v100-tight preset
# (max_model_len=512, fix-len workload) so full weights fit with KV headroom.
#
#   ./profiler/jobs/submit_arc_v100_qwen15moe_bench_dvfs.sh
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
  "${ROOT}/profiler/jobs/checkpoints" "${ROOT}/bench/results"

MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
TIME="${TIME:-01:30:00}"
NUM_REQS="${NUM_REQS:-100}"
SPS="${SPS:-10}"
SEED="${SEED:-42}"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/qwen15moe_bench_dvfs_jobs.tsv"
SUBMIT_LOG="${ROOT}/profiler/jobs/logs/qwen15moe_bench_submit_$(date -u +%Y%m%d_%H%M%S).log"

export V100_FREQ_MIN_MHZ=700
export V100_FREQ_MAX_MHZ=1300
export V100_FREQ_STEP_MHZ=200

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\tjob_id\tmodel\thardware\tgpu_freq_mhz\tstatus\n' > "${MANIFEST}"
fi

submit_bench_job() {
  local gpu_freq_mhz="${1:-}"
  local prev_jid="${2:-}"

  local freq_tag job_name hardware rendered cmd_frag dep_args jid_raw jid
  if [[ -n "${gpu_freq_mhz}" ]]; then
    freq_tag="${gpu_freq_mhz}MHz"
    hardware="V100_${freq_tag}"
    job_name="llmsim_bench_qwen15moe_${freq_tag}"
  else
    freq_tag="default"
    hardware="V100"
    job_name="llmsim_bench_qwen15moe_default"
  fi

  rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"

  read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL}'
export TP_SIZE='1'
export DTYPE='float16'
export NUM_REQS='${NUM_REQS}'
export SPS='${SPS}'
export SEED='${SEED}'
export V100_BENCH_PRESET='qwen15-v100-tight'
export HARDWARE='${hardware}'
export GPU_FREQ_MHZ='${gpu_freq_mhz}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"

  export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT=engs2950 PARTITION=interactive
  export DATA_OUTPUT="${ROOT}/bench/results"
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

  dep_args=()
  if [[ -n "${prev_jid}" ]]; then
    dep_args+=(--dependency="afterany:${prev_jid}")
  fi

  jid_raw=$(sbatch --parsable --clusters=htc --account=engs-glass \
    --partition=interactive --gres=gpu:v100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=64G --time="${TIME}" --job-name="${job_name}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" --mail-type=BEGIN,END,FAIL \
    --output="${ROOT}/profiler/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/profiler/jobs/logs/${job_name}_%j.err" \
    "${dep_args[@]}" "${rendered}")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid}" "${MODEL}" "${hardware}" "${gpu_freq_mhz:-default}" \
    >> "${MANIFEST}"

  echo "  job ${jid}  ${hardware}  freq=${gpu_freq_mhz:-default}" | tee -a "${SUBMIT_LOG}" >&2
  echo "${jid}"
}

{
  echo "=== Qwen1.5-MoE V100 tp1 throughput bench (DVFS) ==="
  echo "MODEL=${MODEL}  NUM_REQS=${NUM_REQS}  SPS=${SPS}  TIME=${TIME}"
  echo "FREQS: default + $(v100_freq_list | tr '\n' ' ')"
} | tee "${SUBMIT_LOG}"

arc_wait_htc_interactive_slot interactive || true

prev="$(submit_bench_job "" "")"
for mhz in $(v100_freq_list); do
  prev="$(submit_bench_job "${mhz}" "${prev}")"
done

{
  echo "=== Submitted. Last job: ${prev} ==="
  echo "Manifest: ${MANIFEST}"
  echo "Results:  ${ROOT}/bench/results/V100*/Qwen_Qwen1.5-MoE-A2.7B-Chat/"
} | tee -a "${SUBMIT_LOG}"

echo "${prev}"
