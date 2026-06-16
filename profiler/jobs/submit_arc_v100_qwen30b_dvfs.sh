#!/usr/bin/env bash
# Qwen3-30B-A3B MoE V100 profiler: baseline (tp1,2,4) then DVFS sweep (tp1 every 200 MHz).
#
# Default: SMOKE grid (tiny attention/KV) to validate end-to-end on V100.
# Scale up later:
#   PROFILE_SCALE=1 ATTENTION_MAX_KV=4096 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=2048 \\
#     FORCE=1 ./profiler/jobs/submit_arc_v100_qwen30b_dvfs.sh
#
# Resumes partial CSVs under profiler/perf/V100/Qwen/Qwen3-30B-A3B-Instruct-2507/.
#
#   ./profiler/jobs/submit_arc_v100_qwen30b_dvfs.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
MATRIX="${ROOT}/profiler/jobs/submit_arc_v100_profile_matrix.sh"

echo "=== Cancelling running llmsim profiler jobs ==="
while read -r jid _; do
  [[ -z "${jid}" ]] && continue
  echo "  scancel ${jid}"
  scancel -M htc "${jid}" 2>/dev/null || true
done < <(squeue -M htc -u "${USER}" -h -o '%i %j' 2>/dev/null | grep -E 'llmsim_prof|llmsim_llama|llmsim_qwen' || true)
pkill -f 'submit_arc_v100_profile_freq_matrix' 2>/dev/null || true
sleep 2

export MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"
export JOB_NAME_PREFIX="llmsim_qwen30b"
export TP_DEGREES="1,2,4"
export FREQ_TP_DEGREES="1"
export DTYPE="float16"
export TIME="${TIME:-06:00:00}"
export SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
export SCHEDULE_FREQ_MATRIX=1
export SKIP_SKEW=1

if [[ "${PROFILE_SCALE:-0}" == "1" ]]; then
  export ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-4096}"
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
  export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
  export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
  echo "=== Profile mode: SCALE (feasible attention grid) ==="
else
  # Tiny smoke grid — ~800 attention shots vs ~6k+ at 4096 KV.
  export ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-256}"
  export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
  export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
  export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-1}"
  echo "=== Profile mode: SMOKE (ATTENTION_MAX_KV=${ATTENTION_MAX_KV} MSQ=${MAX_NUM_SEQS} MNBT=${MAX_NUM_BATCHED_TOKENS}) ==="
fi

export V100_FREQ_MIN_MHZ="${V100_FREQ_MIN_MHZ:-700}"
export V100_FREQ_MAX_MHZ="${V100_FREQ_MAX_MHZ:-1400}"
export V100_FREQ_STEP_MHZ="${V100_FREQ_STEP_MHZ:-200}"

echo "=== Submit Qwen3-30B-A3B V100 baseline tp1,2,4 + freq tp1 (${V100_FREQ_MIN_MHZ}-${V100_FREQ_MAX_MHZ} step ${V100_FREQ_STEP_MHZ}) ==="
bash "${MATRIX}"
