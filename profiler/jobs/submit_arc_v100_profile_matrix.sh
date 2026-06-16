#!/usr/bin/env bash
# Submit V100 profiler matrix as one Slurm job per model (checkpointed).
#
# Each job profiles a single model at TP 1,2,4 at default GPU clocks, writing:
#   profiler/perf/V100/<MODEL>/fp16/tp{1,2,4}/...
#
# After baseline completes, run the freq sweep:
#   ./profiler/jobs/submit_arc_v100_profile_freq_matrix.sh
#
# Or chain automatically:
#   SCHEDULE_FREQ_MATRIX=1 ./profiler/jobs/submit_arc_v100_profile_matrix.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_profile.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"
FREQ_SUBMIT="${ROOT}/profiler/jobs/submit_arc_v100_profile_freq_matrix.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

mkdir -p "${ROOT}/profiler/jobs/logs" "${ROOT}/profiler/jobs/rendered" \
  "${ROOT}/profiler/jobs/checkpoints"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-llmsim_prof_v100}"
PROJECT="${PROJECT:-engs2950}"
PARTITION="interactive"
DATA_OUTPUT="${DATA_OUTPUT:-${ROOT}/profiler/perf}"
TIME="${TIME:-03:00:00}"

HARDWARE="${HARDWARE:-${V100_BASELINE_HARDWARE}}"
TP_DEGREES="${TP_DEGREES:-1,2,4}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FULL_PROFILE="${FULL_PROFILE:-0}"
VERBOSITY="${VERBOSITY:-"--verbose"}"
HF_TOKEN="${HF_TOKEN:-}"
DRY_RUN="${DRY_RUN:-0}"
SCHEDULE_FREQ_MATRIX="${SCHEDULE_FREQ_MATRIX:-0}"

SUBMIT_LOG="${SUBMIT_LOG:-${ROOT}/profiler/jobs/logs/v100_matrix_submit_$(date -u +%Y%m%d_%H%M%S).log}"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/v100_matrix_jobs.tsv"

mapfile -t MODEL_LIST < <(v100_matrix_model_list)

{
  echo "=== V100 baseline profiler matrix (one job per model, default clocks) ==="
  echo "MODELS (${#MODEL_LIST[@]}): ${MODEL_LIST[*]}"
  echo "HARDWARE=${HARDWARE}  TP_DEGREES=${TP_DEGREES}  SKIP_COMPLETE=${SKIP_COMPLETE}"
  echo "TIME=${TIME}  SCHEDULE_FREQ_MATRIX=${SCHEDULE_FREQ_MATRIX}"
} | tee "${SUBMIT_LOG}"

if [[ ! -f "${MANIFEST}" ]] || ! head -1 "${MANIFEST}" | grep -q gpu_freq_mhz; then
  printf 'submitted_at\tjob_id\tmodel\thardware\tgpu_freq_mhz\tstatus\n' > "${MANIFEST}"
fi

prev_jid=""
submitted=0
skipped=0

for MODEL in "${MODEL_LIST[@]}"; do
  cfg="${ROOT}/configs/model/${MODEL}.json"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP ${MODEL}: missing ${cfg}" | tee -a "${SUBMIT_LOG}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "${SKIP_COMPLETE}" == "1" && "${FORCE:-0}" != "1" ]] \
      && v100_matrix_model_complete "${ROOT}" "${MODEL}" "${HARDWARE}"; then
    echo "SKIP ${MODEL}: already complete (${HARDWARE}/meta.yaml)" | tee -a "${SUBMIT_LOG}"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "${submitted}" -eq 0 && "${DRY_RUN}" != "1" ]]; then
    arc_wait_htc_interactive_slot interactive || exit 1
  fi

  echo "SUBMIT ${MODEL} (default clocks -> ${HARDWARE})" | tee -a "${SUBMIT_LOG}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    prev_jid="dry_$(v100_matrix_safe_name "${MODEL}")"
    submitted=$((submitted + 1))
    continue
  fi

  prev_jid="$(v100_matrix_submit_profile_job "${MODEL}" "${HARDWARE}" "" "${prev_jid}")"
  submitted=$((submitted + 1))
done

{
  echo "=== Baseline done: submitted=${submitted} skipped=${skipped} ==="
  echo "Manifest: ${MANIFEST}"
  echo "Submit log: ${SUBMIT_LOG}"
  if [[ -n "${prev_jid}" && "${DRY_RUN}" != "1" ]]; then
    echo "Last job: ${prev_jid}  watch: squeue -M htc -j ${prev_jid}"
  fi
} | tee -a "${SUBMIT_LOG}"

if [[ "${SCHEDULE_FREQ_MATRIX}" == "1" && "${DRY_RUN}" != "1" && -n "${prev_jid}" ]]; then
  echo "Scheduling freq matrix after baseline job ${prev_jid} ..." | tee -a "${SUBMIT_LOG}"
  BASELINE_DEPEND_JID="${prev_jid}" \
    REQUIRE_BASELINE_COMPLETE=0 \
    TP_DEGREES="${FREQ_TP_DEGREES:-1}" \
    bash "${FREQ_SUBMIT}"
elif [[ "${SCHEDULE_FREQ_MATRIX}" == "1" && "${submitted}" -eq 0 ]]; then
  echo "Baseline already complete — run ./profiler/jobs/submit_arc_v100_profile_freq_matrix.sh" | tee -a "${SUBMIT_LOG}"
fi
