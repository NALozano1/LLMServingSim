#!/usr/bin/env bash
# Headless ShareGPT segmented timing study (mono already done for N=10,20,100).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/outputs/branch_compare/timing_study_sharegpt"
LOG_DIR="${OUT_DIR}/logs"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/seg_timing_${STAMP}.log"
PID_FILE="${LOG_DIR}/seg_timing.pid"

mkdir -p "${LOG_DIR}"

{
  echo "=== seg timing study started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "pid=$$"
  echo "root=${ROOT}"
  echo "cmd: python3 scripts/run_mono_vs_seg_timing.py --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl --num-reqs 10 20 100 --modes seg --out-dir outputs/branch_compare/timing_study_sharegpt"
  echo
} >> "${LOG}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}"

python3 scripts/run_mono_vs_seg_timing.py \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --num-reqs 10 20 100 \
  --modes seg \
  --out-dir outputs/branch_compare/timing_study_sharegpt \
  >> "${LOG}" 2>&1

echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$? ===" >> "${LOG}"
