#!/usr/bin/env bash
# Submit mini prefill DVFS campaign: 3 scattered permutations × 3 iterations × 2 models = 18 runs.
#
#   ./bench/jobs/submit_arc_v100_bench_layer_mini3x3.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${ROOT}/bench/campaigns/v100_bench_layer_dvfs_mini3x3}"
SEED="${SEED:-20260616}"

python3 "${ROOT}/bench/jobs/generate_bench_layer_dvfs_campaign.py" \
  --repo "${ROOT}" \
  --out-dir "${CAMPAIGN_DIR}" \
  --seed "${SEED}" \
  --num-scattered-permutations 3 \
  --iterations 3 \
  --scattered-only

# Cancel any stale full-campaign jobs blocking the queue
squeue -M htc -u engs2950 -h -o "%i %j" 2>/dev/null \
  | rg "llmsim_bench_fdvfs" \
  | awk '{print $1}' \
  | xargs -r -n25 scancel -M htc 2>/dev/null || true

export CAMPAIGN_DIR
SKIP_INTERACTIVE_WAIT="${SKIP_INTERACTIVE_WAIT:-1}" \
  bash "${ROOT}/bench/jobs/submit_arc_v100_bench_layer_campaign.sh"

echo ""
echo "Replication guide: ${CAMPAIGN_DIR}/REPLICATION.md"
echo "Collect: python3 ${ROOT}/bench/jobs/collect_bench_layer_campaign_results.py ${CAMPAIGN_DIR}"
