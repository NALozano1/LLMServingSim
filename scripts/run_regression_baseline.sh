#!/usr/bin/env bash
# Regression guard for the NON-DVFS simulator path.
#
# Compares the current working tree against the HEAD-original "golden" outputs
# captured in serving/tests/regression_baseline/ (per-request CSV + total clocks).
# This lets us confirm the DVFS/segment work has not perturbed the original
# functionality WITHOUT having to git-stash back to HEAD every time.
#
# Run from the repo root inside the simulator container:
#   docker exec servingsim_docker bash -lc 'cd /app/LLMServingSim && ./scripts/run_regression_baseline.sh'
#
# Exit code is non-zero if any scenario drifts from golden.
#
# Re-baseline (only after an INTENTIONAL change to original behavior):
#   ./scripts/run_regression_baseline.sh --update
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
GOLDEN="serving/tests/regression_baseline"
DATASET="workloads/example_trace.jsonl"
UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1
# Repo-relative tmp dir: the simulator prepends "../" to --output (its cwd is
# astra-sim/), so an absolute path would become "..//tmp/..." and break.
TMP="outputs/.regression_tmp"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT
fail=0

run_case() {
  local name="$1"; shift
  local golden_csv="$GOLDEN/${name}.csv"
  local golden_clk="$GOLDEN/${name}.clocks"
  local out_csv="$TMP/${name}.csv"
  local log="$TMP/${name}.log"

  # Clear cached inputs so trace/graph generation is exercised fresh each run.
  rm -rf astra-sim/inputs/trace/RTXPRO6000 astra-sim/inputs/workload/RTXPRO6000 2>/dev/null

  python3 -m serving "$@" --output "$out_csv" --log-interval 1.0 > "$log" 2>&1
  local rc=$?
  local clocks
  clocks="$(grep -oE 'Total clocks \(ns\):[[:space:]]+[0-9]+' "$log" | grep -oE '[0-9]+$' | tail -1)"

  if [[ $rc -ne 0 ]]; then
    echo "FAIL  ${name}: sim exited ${rc}"
    tail -8 "$log" | sed 's/^/      /'
    fail=1
    return
  fi

  if [[ $UPDATE -eq 1 ]]; then
    cp "$out_csv" "$golden_csv"
    printf '%s\n' "$clocks" > "$golden_clk"
    echo "UPDATED ${name}: clocks=${clocks}"
    return
  fi

  local cfail=0 msg=""
  if [[ -f "$golden_csv" ]] && diff -q "$golden_csv" "$out_csv" >/dev/null 2>&1; then
    msg="csv=OK"
  else
    msg="csv=MISMATCH"; cfail=1
  fi
  local exp; exp="$(cat "$golden_clk" 2>/dev/null || echo '')"
  if [[ "$clocks" == "$exp" ]]; then
    msg="${msg}  clocks=OK(${clocks})"
  else
    msg="${msg}  clocks=MISMATCH(got=${clocks} want=${exp})"; cfail=1
  fi

  if [[ $cfail -eq 0 ]]; then
    echo "PASS  ${name}: ${msg}"
  else
    echo "FAIL  ${name}: ${msg}"
    [[ "$msg" == *csv=MISMATCH* ]] && diff "$golden_csv" "$out_csv" | head -20 | sed 's/^/      /'
    fail=1
  fi
}

echo "=== Non-DVFS regression vs HEAD-original golden (${GOLDEN}) ==="

# Dense: Llama-3.1-8B on the bundled RTXPRO6000 bf16 profile (single instance).
run_case dense_llama_rtxpro6000 \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype bfloat16 --block-size 16 --dataset "$DATASET"

# MoE: Qwen3-30B-A3B on RTXPRO6000 bf16 (multi-instance, LOAD routing) — exercises
# the expert/MoE emission path that the DVFS work also touches.
run_case moe_qwen3_rtxpro6000 \
  --cluster-config configs/cluster/single_node_moe_multi_instance.json \
  --dtype bfloat16 --block-size 16 --dataset "$DATASET" --request-routing-policy LOAD

if [[ $UPDATE -eq 1 ]]; then
  echo "Golden baselines updated."
  exit 0
fi
if [[ $fail -eq 0 ]]; then
  echo "REGRESSION OK: original (non-DVFS) functionality retained."
else
  echo "REGRESSION FAILED: output drifted from HEAD-original golden."
  exit 1
fi
