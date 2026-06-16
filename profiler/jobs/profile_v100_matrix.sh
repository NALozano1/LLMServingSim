#!/usr/bin/env bash
# DEPRECATED: use submit_arc_v100_profile_matrix.sh (one Slurm job per model).
#
# Local single-model run inside the vLLM container:
#   MODEL=meta-llama/Llama-3.1-8B ./profiler/jobs/profile_v100_matrix.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "error: set MODEL=<hf/id> for a single local profile run." >&2
  echo "For the full matrix, use: ./profiler/jobs/submit_arc_v100_profile_matrix.sh" >&2
  exit 1
fi

HARDWARE="${HARDWARE:-V100}"
TP_DEGREES="${TP_DEGREES:-1,2,4}"
VERBOSITY="${VERBOSITY:-"--verbose"}"

SKIP_SKEW_FLAG=()
if [[ "${FULL_PROFILE:-0}" != "1" && -z "${SKIP_SKEW:-}" ]]; then
  SKIP_SKEW_FLAG=(--skip-skew)
elif [[ -n "${SKIP_SKEW:-}" && "${SKIP_SKEW}" != "0" ]]; then
  SKIP_SKEW_FLAG=(--skip-skew)
fi

FORCE_FLAG=()
[[ -n "${FORCE:-}" && "${FORCE}" != "0" ]] && FORCE_FLAG=(--force)

VERBOSITY_FLAG=()
case "${VERBOSITY}" in
  --verbose|--silent) VERBOSITY_FLAG=("${VERBOSITY}") ;;
esac

exec ./profiler/profile_cli.sh \
  --model "${MODEL}" \
  --hardware "${HARDWARE}" \
  --tp "${TP_DEGREES}" \
  --dtype "${DTYPE:-float16}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --attention-max-kv "${ATTENTION_MAX_KV:-8192}" \
  --measurement-iterations "${MEASUREMENT_ITERATIONS:-3}" \
  "${SKIP_SKEW_FLAG[@]}" \
  "${FORCE_FLAG[@]}" \
  "${VERBOSITY_FLAG[@]}"
