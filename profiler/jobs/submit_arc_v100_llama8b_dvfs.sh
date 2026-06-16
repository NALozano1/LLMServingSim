#!/usr/bin/env bash
# Llama-3.1-8B V100 profiler: baseline (tp1+tp2) then DVFS sweep (tp1 every 200 MHz).
#
# Outputs:
#   profiler/perf/V100/meta-llama/Llama-3.1-8B/fp16/tp{1,2}/...
#   profiler/perf/V100_<MHz>/meta-llama/Llama-3.1-8B/fp16/tp1/...
#
# Kills any running llmsim_prof* jobs first.
#
#   export HF_TOKEN=...   # if needed
#   ./profiler/jobs/submit_arc_v100_llama8b_dvfs.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
MATRIX="${ROOT}/profiler/jobs/submit_arc_v100_profile_matrix.sh"

echo "=== Cancelling stuck/running llmsim profiler jobs ==="
while read -r jid _; do
  [[ -z "${jid}" ]] && continue
  echo "  scancel ${jid}"
  scancel -M htc "${jid}" 2>/dev/null || true
done < <(squeue -M htc -u "${USER}" -h -o '%i %j' 2>/dev/null | grep -E 'llmsim_prof|llmsim_llama|llmsim_v100' || true)
sleep 2

export MODELS="meta-llama/Llama-3.1-8B"
export JOB_NAME_PREFIX="llmsim_llama8b"
export TP_DEGREES="1,2"
export FREQ_TP_DEGREES="1"
export DTYPE="float16"
export TIME="${TIME:-02:00:00}"
export SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
export SCHEDULE_FREQ_MATRIX=1

# Match dvfs_bypass_test_700_1400 sweep on ARC V100.
export V100_FREQ_MIN_MHZ="${V100_FREQ_MIN_MHZ:-700}"
export V100_FREQ_MAX_MHZ="${V100_FREQ_MAX_MHZ:-1400}"
export V100_FREQ_STEP_MHZ="${V100_FREQ_STEP_MHZ:-200}"

echo "=== Submit Llama-3.1-8B V100 baseline tp1,2 + freq tp1 (${V100_FREQ_MIN_MHZ}-${V100_FREQ_MAX_MHZ} step ${V100_FREQ_STEP_MHZ}) ==="
bash "${MATRIX}"
