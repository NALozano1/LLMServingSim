#!/usr/bin/env bash
# Submit 2 real-B2B smoke jobs: A100 (contaminated regime) + V100 (control).
set -euo pipefail

ROOT=/data/engs-glass/engs2950
CALIB=$ROOT/DVFS-MoE/LLMServingSim/validation/recon_experiments/power_calib
SBATCH=$CALIB/run_power_calib_real_b2b.sbatch
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
BATCHES="${BATCHES:-8,32,512,8192}"
MAIL_USER="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}"

mkdir -p "$CALIB/logs"
cd "$ROOT/DVFS-MoE/LLMServingSim"

echo "Submitting real-B2B smokes (model=$MODEL batches=$BATCHES)"

# A100 @1410 — where legacy gapfree small-batch latency looked ~3× profiler
A100_JOB=$(sbatch -M htc --clusters=htc --account=engs-glass \
  --partition=short --gres=gpu:a100:1 \
  --nodes=1 --cpus-per-task=4 --mem=32G --time=00:45:00 \
  --mail-user="${MAIL_USER}" --mail-type=BEGIN,END,FAIL \
  --parsable \
  "$SBATCH" 1410 "$BATCHES" "" "$MODEL")
echo "A100 @1410MHz → job ${A100_JOB}"

# V100 @900 — where legacy gapfree ≈ profiler (control)
V100_JOB=$(sbatch -M htc --clusters=htc --account=engs-glass \
  --partition=interactive --gres=gpu:v100:1 \
  --nodes=1 --cpus-per-task=4 --mem=32G --time=00:45:00 \
  --mail-user="${MAIL_USER}" --mail-type=BEGIN,END,FAIL \
  --parsable \
  "$SBATCH" 900 "$BATCHES" "" "$MODEL")
echo "V100 @900MHz  → job ${V100_JOB}"

cat > "$CALIB/logs/real_b2b_smoke_jobs.txt" <<EOF
A100_JOB=${A100_JOB}
V100_JOB=${V100_JOB}
MODEL=${MODEL}
BATCHES=${BATCHES}
SUBMITTED=$(date -Is)
EOF

echo "Job IDs written to $CALIB/logs/real_b2b_smoke_jobs.txt"
echo "When done, compare with:"
echo "  python3 $CALIB/compare_real_b2b_vs_legacy.py"
