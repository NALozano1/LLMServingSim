#!/usr/bin/env bash
# =============================================================================
# RUN ON NSERVER15 INSIDE THE DOCKER CONTAINER:
#   docker start -ai servingsim_docker
#   cd /app/LLMServingSim
#   bash bench/jobs/run_sim_validation_sweep.sh
# =============================================================================
# Runs LLMServingSim for each (model, clock) validation arm and saves CSVs to
# bench/results/sim_sweep/<run_id>.csv for comparison against hardware TTFT.
#
# Qwen tp4: only uncapped traces exist at tp4; clocked arms use --dvfs-scale
#            read from bench/results/<CAMPAIGN>/dvfs_scale_factors.json
#
# Prerequisites (all inside Docker container):
#   - python -m serving works (msgspec etc. installed)
#   - profiler/perf/V100/ and V100_700MHz/…/V100_1400MHz/ present
#   - bench/results/<CAMPAIGN>/dvfs_scale_factors.json exists
#     (generate with: python3 bench/jobs/compute_dvfs_scale_factors.py <tsv>)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via env vars if needed
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-v100_prefill_valid_20260625}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/bench/results/${CAMPAIGN_NAME}}"
SWEEP_DIR="${SWEEP_DIR:-${REPO_ROOT}/bench/results/sim_sweep}"
TMP_CFG_DIR="${TMP_CFG_DIR:-/tmp/llmsim_sim_sweep_cfgs}"
SCALE_JSON="${RESULTS_DIR}/dvfs_scale_factors.json"

QWEN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
DTYPE="float16"
BLOCK_SIZE=16
N_REQS=100
SPS=100
SEED=42
FIX_INPUT=512
FIX_OUTPUT=1

mkdir -p "${SWEEP_DIR}" "${TMP_CFG_DIR}"

# ---------------------------------------------------------------------------
# Generate datasets (fixed-length, matching the hardware bench workloads)
# ---------------------------------------------------------------------------
QWEN_DATASET="${SWEEP_DIR}/sharegpt_qwen_512tok.jsonl"

if [[ ! -f "${QWEN_DATASET}" ]]; then
  echo "[sweep] Generating Qwen 512-tok dataset…"
  python3 -m workloads.generators sharegpt \
    --model "${QWEN_MODEL}" \
    --num-reqs "${N_REQS}" --sps "${SPS}" --seed "${SEED}" \
    --fix-len --fix-input-length "${FIX_INPUT}" --fix-output-length "${FIX_OUTPUT}" \
    --output "${QWEN_DATASET}"
fi

# ---------------------------------------------------------------------------
# Helper: read dvfs_scale for a given model key + arm from the JSON
# ---------------------------------------------------------------------------
get_scale() {
  local model_key="$1" arm="$2"
  python3 - <<EOF
import json, sys
data = json.load(open('${SCALE_JSON}'))
mk = data.get('${model_key}', {})
arm = mk.get('${arm}', {})
scale = arm.get('scale')
if scale is None:
    print('MISSING', file=sys.stderr)
    sys.exit(1)
print(scale)
EOF
}

# ---------------------------------------------------------------------------
# Helper: run sim for one arm
# ---------------------------------------------------------------------------
run_one() {
  local run_id="$1" hardware="$2" tp_size="$3" dataset="$4" extra_args="${5:-}"
  local cfg="${TMP_CFG_DIR}/cfg_${run_id}.json"
  local out="${SWEEP_DIR}/${run_id}.csv"

  if [[ -f "${out}" ]]; then
    echo "[sweep] SKIP ${run_id} (already exists: ${out})"
    return 0
  fi

  echo "[sweep] ${run_id}  hardware=${hardware}  tp=${tp_size}${extra_args:+  ${extra_args}}"

  # Determine model from run_id prefix
  local model="${QWEN_MODEL}"

  python3 "${REPO_ROOT}/scripts/generate_cluster_config.py" \
    --hardware "${hardware}" \
    --model-name "${model}" \
    --variant "fp16" \
    --tp-size "${tp_size}" \
    --num-nodes 1 --instances-per-node 1 \
    --output "${cfg}" 2>/dev/null

  python -m serving \
    --cluster-config "${cfg}" \
    --dtype "${DTYPE}" \
    --block-size "${BLOCK_SIZE}" \
    --dataset "${dataset}" \
    --output "${out}" \
    --log-interval 1.0 \
    ${extra_args}

  echo "[sweep] DONE ${run_id} -> ${out}"
}

# ---------------------------------------------------------------------------
# Qwen tp4 — uncapped uses traces; clocked arms use --dvfs-scale
# ---------------------------------------------------------------------------
echo ""
echo "=== Qwen tp4 (uncapped traces; clocked arms use --dvfs-scale) ==="

if [[ ! -f "${SCALE_JSON}" ]]; then
  echo "ERROR: ${SCALE_JSON} not found." >&2
  echo "  Generate it first with:" >&2
  echo "    python3 bench/jobs/compute_dvfs_scale_factors.py \\" >&2
  echo "        bench/results/${CAMPAIGN_NAME}/validation_results.tsv" >&2
  exit 1
fi

run_one "qwen_tp4_uncapped" "V100" 4 "${QWEN_DATASET}"

for arm in 700mhz 900mhz 1100mhz 1300mhz 1400mhz; do
  run_id="qwen_tp4_${arm}"
  scale=$(get_scale "qwen_tp4" "${arm}") || {
    echo "WARN: scale for qwen_tp4/${arm} not in ${SCALE_JSON}, skipping" >&2
    continue
  }
  run_one "${run_id}" "V100" 4 "${QWEN_DATASET}" "--dvfs-scale ${scale}"
done

echo ""
echo "=== Sweep complete ==="
echo "Results in: ${SWEEP_DIR}"
echo ""
echo "Next: compare against hardware:"
echo "  python3 bench/jobs/compare_sim_vs_hardware.py \\"
echo "      bench/results/${CAMPAIGN_NAME}/validation_results.tsv \\"
echo "      bench/results/sim_sweep/"
