#!/usr/bin/env bash
# Submit V100 profiler reruns at locked GPU clocks (200 MHz steps by default).
#
# DVFS: ARC site helper via shared/scripts/gpu_freq_lock_lib.sh (same path as
# run_moe_ep8_inference_power.sh and vLLM DVFS serving scripts). Do not use raw
# nvidia-smi --lock-gpu-clocks on HTC compute nodes.
#
# Run after baseline (default clocks) matrix completes:
#   profiler/perf/V100/<MODEL>/fp16/tp{1,2,4}/...
#
# Each (model, freq) is one Slurm job. Outputs land under a distinct hardware tag:
#   profiler/perf/V100_<MHz>/<MODEL>/fp16/tp{1,2,4}/...
#   profiler/perf/V100_<MHz>/<MODEL>/fp16/gpu_freq/gpu_freq_apply.json
#
# Examples:
#   ./profiler/jobs/submit_arc_v100_profile_freq_matrix.sh
#
#   # Wait for a specific baseline job before starting:
#   BASELINE_DEPEND_JID=7966819 ./profiler/jobs/submit_arc_v100_profile_freq_matrix.sh
#
#   # Auto-chain from baseline submit:
#   SCHEDULE_FREQ_MATRIX=1 ./profiler/jobs/submit_arc_v100_profile_matrix.sh
#
#   V100_FREQ_MIN_MHZ=900 V100_FREQ_MAX_MHZ=1530 V100_FREQ_STEP_MHZ=200 \\
#     ./profiler/jobs/submit_arc_v100_profile_freq_matrix.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_v100_profile.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

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

TP_DEGREES="${TP_DEGREES:-1,2,4}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
REQUIRE_BASELINE_COMPLETE="${REQUIRE_BASELINE_COMPLETE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FULL_PROFILE="${FULL_PROFILE:-0}"
VERBOSITY="${VERBOSITY:-"--verbose"}"
HF_TOKEN="${HF_TOKEN:-}"
DRY_RUN="${DRY_RUN:-0}"
BASELINE_DEPEND_JID="${BASELINE_DEPEND_JID:-}"

SUBMIT_LOG="${SUBMIT_LOG:-${ROOT}/profiler/jobs/logs/v100_freq_matrix_submit_$(date -u +%Y%m%d_%H%M%S).log}"
MANIFEST="${ROOT}/profiler/jobs/checkpoints/v100_freq_matrix_jobs.tsv"

mapfile -t MODEL_LIST < <(v100_matrix_model_list)
mapfile -t FREQ_LIST < <(v100_freq_list)

{
  echo "=== V100 freq-sweep profiler matrix (one job per model × MHz) ==="
  echo "FREQS (${#FREQ_LIST[@]}): ${FREQ_LIST[*]} MHz  (step=${V100_FREQ_STEP_MHZ})"
  echo "MODELS (${#MODEL_LIST[@]}): ${MODEL_LIST[*]}"
  echo "TP_DEGREES=${TP_DEGREES}  SKIP_COMPLETE=${SKIP_COMPLETE}"
  echo "REQUIRE_BASELINE_COMPLETE=${REQUIRE_BASELINE_COMPLETE}  BASELINE_DEPEND_JID=${BASELINE_DEPEND_JID:-<none>}"
} | tee "${SUBMIT_LOG}"

if [[ "${REQUIRE_BASELINE_COMPLETE}" == "1" && -z "${BASELINE_DEPEND_JID}" ]]; then
  if ! v100_baseline_matrix_complete "${ROOT}"; then
    echo "error: baseline V100 profiles incomplete — finish default-freq matrix first" >&2 | tee -a "${SUBMIT_LOG}"
    echo "  or set REQUIRE_BASELINE_COMPLETE=0 / BASELINE_DEPEND_JID=<jid>" >&2 | tee -a "${SUBMIT_LOG}"
    exit 1
  fi
  echo "Baseline V100 matrix complete for all models." | tee -a "${SUBMIT_LOG}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\tjob_id\tmodel\thardware\tgpu_freq_mhz\tstatus\n' > "${MANIFEST}"
fi

prev_jid="${BASELINE_DEPEND_JID}"
submitted=0
skipped=0

for MODEL in "${MODEL_LIST[@]}"; do
  cfg="${ROOT}/configs/model/${MODEL}.json"
  if [[ ! -f "$cfg" ]]; then
    echo "SKIP ${MODEL}: missing ${cfg}" | tee -a "${SUBMIT_LOG}"
    skipped=$((skipped + 1))
    continue
  fi

  for mhz in "${FREQ_LIST[@]}"; do
    hardware="$(v100_hardware_label "${mhz}")"

    if [[ "${SKIP_COMPLETE}" == "1" && "${FORCE:-0}" != "1" ]] \
        && v100_matrix_model_complete "${ROOT}" "${MODEL}" "${hardware}"; then
      echo "SKIP ${MODEL} @ ${mhz}MHz: already complete" | tee -a "${SUBMIT_LOG}"
      skipped=$((skipped + 1))
      continue
    fi

    if [[ "${submitted}" -eq 0 && "${DRY_RUN}" != "1" ]]; then
      arc_wait_htc_interactive_slot interactive || exit 1
    fi

    echo "SUBMIT ${MODEL} @ ${mhz}MHz -> ${hardware}" | tee -a "${SUBMIT_LOG}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      prev_jid="dry_${mhz}_$(v100_matrix_safe_name "${MODEL}")"
      submitted=$((submitted + 1))
      continue
    fi

    prev_jid="$(v100_matrix_submit_profile_job "${MODEL}" "${hardware}" "${mhz}" "${prev_jid}")"
    submitted=$((submitted + 1))
  done
done

{
  echo "=== Freq matrix done: submitted=${submitted} skipped=${skipped} ==="
  echo "Manifest: ${MANIFEST}"
  echo "Submit log: ${SUBMIT_LOG}"
  if [[ -n "${prev_jid}" && "${DRY_RUN}" != "1" ]]; then
    echo "Last job: ${prev_jid}  watch: squeue -M htc -j ${prev_jid}"
  fi
} | tee -a "${SUBMIT_LOG}"
