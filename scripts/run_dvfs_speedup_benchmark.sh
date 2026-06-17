#!/usr/bin/env bash
# Compare mono vs segmented wall time across P0 optimization tiers.
# Git checkouts run on the host; simulations run inside servingsim_docker.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

NUM_REQS="${NUM_REQS:-10}"
DATASET="${DATASET:-workloads/example_trace.jsonl}"
OUT_ROOT="${OUT_ROOT:-outputs/branch_compare/dvfs_speedup}"
MODEL_TRACE_GLOB="inputs/trace/RTXPRO6000/meta-llama"
MODEL_WORKLOAD_GLOB="inputs/workload/RTXPRO6000/meta-llama"
DOCKER="${DOCKER:-docker exec servingsim_docker}"

clear_segment_cache() {
  rm -rf "$REPO/$MODEL_TRACE_GLOB" "$REPO/$MODEL_WORKLOAD_GLOB"
}

run_timing() {
  local label="$1"
  shift
  local modes=("$@")
  clear_segment_cache
  mkdir -p "$REPO/$OUT_ROOT/$label"
  $DOCKER bash -lc "cd /app/LLMServingSim && python3 scripts/run_mono_vs_seg_timing.py \
    --num-reqs $NUM_REQS \
    --dataset $DATASET \
    --out-dir $OUT_ROOT/$label \
    --modes ${modes[*]}"
}

echo "=== DVFS speedup benchmark (N=$NUM_REQS, dataset=$DATASET) ==="

CURRENT_REF="$(git rev-parse --abbrev-ref HEAD)"
restore_ref() {
  git checkout "$CURRENT_REF" >/dev/null 2>&1 || git checkout feat/dvfs-speedup >/dev/null
}
trap restore_ref EXIT

echo "--- baseline (pre-P0) @ 8243f59 ---"
git checkout 8243f59 >/dev/null
run_timing baseline seg

echo "--- P0.1 Chakra skip @ 8e22147 ---"
git checkout 8e22147 >/dev/null
run_timing p0_1_chakra seg

echo "--- P0.2 trace skip @ feat/dvfs-speedup ---"
git checkout feat/dvfs-speedup >/dev/null
run_timing p0_2_full mono seg

echo "Done. Results under $OUT_ROOT/"
