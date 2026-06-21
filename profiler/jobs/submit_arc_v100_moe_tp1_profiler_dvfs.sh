#!/usr/bin/env bash
# V100 tp1 LLMServingSim profiler at default + DVFS freqs (700–1300 MHz, step 200).
#
# Default models: Phi-tiny-MoE + Qwen1.5-MoE (bench-validated on 1× V100 tp1).
# Outputs (importable under profiler/perf/):
#   profiler/perf/V100/<MODEL>/fp16/tp1/...
#   profiler/perf/V100_<MHz>/<MODEL>/fp16/tp1/...
#
#   ./profiler/jobs/submit_arc_v100_moe_tp1_profiler_dvfs.sh
#   MODELS='microsoft/Phi-mini-MoE-instruct' ./profiler/jobs/submit_arc_v100_moe_tp1_profiler_dvfs.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_profile.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

# Bench-aligned DVFS (700–1300 step 200); set before common so 900–1530 defaults don't win.
export V100_FREQ_MIN_MHZ="${V100_FREQ_MIN_MHZ:-700}"
export V100_FREQ_MAX_MHZ="${V100_FREQ_MAX_MHZ:-1300}"
export V100_FREQ_STEP_MHZ="${V100_FREQ_STEP_MHZ:-200}"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered" \
  "${ROOT}/profiler/jobs/checkpoints" "${ROOT}/profiler/perf"

MODELS="${MODELS:-microsoft/Phi-tiny-MoE-instruct Qwen/Qwen1.5-MoE-A2.7B-Chat}"
TIME="${TIME:-03:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-llmsim_prof_moe_tp1}"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/moe_tp1_profiler_dvfs_jobs.tsv"
SUBMIT_LOG="${ROOT}/profiler/jobs/logs/moe_tp1_profiler_submit_$(date -u +%Y%m%d_%H%M%S).log"

export TP_DEGREES="${TP_DEGREES:-1}"
export SKIP_SKEW="${SKIP_SKEW:-1}"
export ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-256}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
export SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export PROJECT=engs2950
export PARTITION=interactive
export DATA_OUTPUT="${ROOT}/profiler/perf"
export JOB_NAME_PREFIX
export RUNNER
export TEMPLATE
export TIME
export MANIFEST
export SUBMIT_LOG

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\tjob_id\tmodel\thardware\tgpu_freq_mhz\tstatus\n' > "${MANIFEST}"
fi

mapfile -t MODEL_LIST < <(printf '%s\n' ${MODELS})
mapfile -t FREQ_LIST < <(v100_freq_list)

{
  echo "=== V100 tp1 MoE profiler (default + DVFS) ==="
  echo "MODELS (${#MODEL_LIST[@]}): ${MODEL_LIST[*]}"
  echo "FREQS: default + ${FREQ_LIST[*]} MHz"
  echo "TP_DEGREES=${TP_DEGREES}  ATTENTION_MAX_KV=${ATTENTION_MAX_KV}"
  echo "TIME=${TIME}"
} | tee "${SUBMIT_LOG}"

prev_jid=""
submitted=0
skipped=0

for MODEL in "${MODEL_LIST[@]}"; do
  cfg="${ROOT}/configs/model/${MODEL}.json"
  if [[ ! -f "${cfg}" ]]; then
    echo "SKIP ${MODEL}: missing ${cfg}" | tee -a "${SUBMIT_LOG}"
    skipped=$((skipped + 1))
    continue
  fi

  for mhz in "" "${FREQ_LIST[@]}"; do
    if [[ -z "${mhz}" ]]; then
      hardware="V100"
      gpu_freq_mhz=""
      freq_tag="default"
    else
      hardware="$(v100_hardware_label "${mhz}")"
      gpu_freq_mhz="${mhz}"
      freq_tag="${mhz}MHz"
    fi

    if [[ "${SKIP_COMPLETE}" == "1" && "${FORCE:-0}" != "1" ]] \
        && v100_matrix_model_complete "${ROOT}" "${MODEL}" "${hardware}"; then
      echo "SKIP ${MODEL} @ ${freq_tag}: already complete" | tee -a "${SUBMIT_LOG}"
      skipped=$((skipped + 1))
      continue
    fi

    safe="$(v100_matrix_safe_name "${MODEL}")"
    job_name="${JOB_NAME_PREFIX}_${freq_tag}_${safe:0:24}"
      rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
      cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"

      read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export MODEL='${MODEL}'
export TP_DEGREES='${TP_DEGREES}'
export HARDWARE='${hardware}'
export GPU_FREQ_MHZ='${gpu_freq_mhz}'
export DTYPE='float16'
export SKIP_SKEW='${SKIP_SKEW}'
export ATTENTION_MAX_KV='${ATTENTION_MAX_KV}'
export MAX_NUM_SEQS='${MAX_NUM_SEQS}'
export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}'
export MEASUREMENT_ITERATIONS='${MEASUREMENT_ITERATIONS}'
export FULL_PROFILE='0'
export VERBOSITY='${VERBOSITY:-"--verbose"}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
      printf '%s' "${CMD}" > "${cmd_frag}"

      export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
      export JOB_NAME="${job_name}"
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

      dep_args=()
      if [[ -n "${prev_jid}" ]]; then
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          dep_args+=(--dependency="afterany:${prev_jid}")
        else
          dep_args+=(--dependency="afterok:${prev_jid}")
        fi
      fi

      if [[ "${SKIP_INTERACTIVE_WAIT:-0}" != "1" && "${submitted}" -eq 0 ]]; then
        arc_wait_htc_interactive_slot interactive || true
      fi

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
        --job-name="${job_name}" \
        --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
        --mail-type=BEGIN,END,FAIL \
        --output="${ROOT}/profiler/jobs/logs/${job_name}_%j.out" \
        --error="${ROOT}/profiler/jobs/logs/${job_name}_%j.err" \
        "${dep_args[@]}" \
        "${rendered}")
      jid="${jid_raw%%;*}"

      printf '%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid}" "${MODEL}" "${hardware}" "${gpu_freq_mhz:-default}" \
        >> "${MANIFEST}"

      echo "  job ${jid}  MODEL=${MODEL}  HARDWARE=${hardware}  GPU_FREQ_MHZ=${gpu_freq_mhz:-default}" \
        | tee -a "${SUBMIT_LOG}"
      prev_jid="${jid}"
      submitted=$((submitted + 1))
  done
done

{
  echo "=== Done: submitted=${submitted} skipped=${skipped} ==="
  echo "Manifest: ${MANIFEST}"
  echo "Import path: ${ROOT}/profiler/perf/V100*/<MODEL>/fp16/"
  if [[ -n "${prev_jid}" ]]; then
    echo "Last job: ${prev_jid}  watch: squeue -M htc -j ${prev_jid}"
  fi
} | tee -a "${SUBMIT_LOG}"
