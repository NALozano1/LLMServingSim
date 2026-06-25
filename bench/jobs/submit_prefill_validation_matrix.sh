#!/usr/bin/env bash
# Submit the Tier-1 / Tier-2 sim-accuracy validation prefill matrix.
#
# Runs: Phi (tp1) + Qwen3-30B (tp4) + Qwen1.5-MoE (tp1) + Llama-3.1-8B (tp1) +
#       Qwen3-30B-tp2 at {uncapped, 700, 900, 1100, 1300, 1400} MHz.
# All pinned to htc-g049 (verified locking node).
# No layer-pause; prefill-only; power-on; clock audited.
#
# Usage (from LLMServingSim/):
#   bash bench/jobs/submit_prefill_validation_matrix.sh
#   DRY_RUN=1 bash bench/jobs/submit_prefill_validation_matrix.sh
#   MODELS=phi bash bench/jobs/submit_prefill_validation_matrix.sh           # phi only
#   MODELS=qwen bash bench/jobs/submit_prefill_validation_matrix.sh          # qwen tp4 only
#   MODELS="qwen15moe llama8b qwen30btp2" bash bench/jobs/...               # new models
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGS_GLASS="/data/engs-glass/engs2950"
JOBS_ROOT="${REPO_ROOT}/bench/jobs"
DRY_RUN="${DRY_RUN:-0}"
NODELIST="${NODELIST:-htc-g049}"
MODELS="${MODELS:-phi qwen}"  # space-separated subset; default: both

# Campaign dir — unique per submission date
CAMPAIGN_TAG="$(date +%Y%m%d)"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${REPO_ROOT}/bench/campaigns/v100_prefill_valid_${CAMPAIGN_TAG}}"

# Clock sweep (MHz; empty string = uncapped)
FREQ_LIST="${FREQ_LIST:- 700 900 1100 1300 1400}"  # leading space = uncapped slot
FREQ_ARRAY=("" ${FREQ_LIST})                        # first element = uncapped

# Walltime per run (boot + inference).
# Phi-tiny-MoE tp1: ~7 min boot + ~3 min inference.
# Qwen3-30B tp4: ~30 min boot + ~10 min inference (OBSERVED timeout at 00:30:00).
# Qwen3-30B tp2: similar boot to tp4 (2 GPUs but 30GB each vs 15GB at tp4).
# Qwen1.5-MoE tp1: ~8 min boot + ~3 min inference.
# Llama-3.1-8B tp1: ~5 min boot + ~2 min inference.
PHI_TIME="${PHI_TIME:-00:20:00}"
QWEN_TIME="${QWEN_TIME:-01:00:00}"
QWEN15_TIME="${QWEN15_TIME:-00:25:00}"
LLAMA8B_TIME="${LLAMA8B_TIME:-00:20:00}"
QWEN30BTP2_TIME="${QWEN30BTP2_TIME:-01:00:00}"

mkdir -p "${CAMPAIGN_DIR}/shared" "${JOBS_ROOT}/logs"

MANIFEST="${CAMPAIGN_DIR}/manifest.tsv"
[[ -f "${MANIFEST}" ]] || printf 'submitted_at\trun_id\tjob_id\tmodel_key\tclock_mhz\tstatus\n' > "${MANIFEST}"

submit_one() {
  local model_key="$1" model="$2" tp="$3" clk="$4" walltime="$5" gpus="$6"
  local extra_env="${7:-}"   # optional extra "export VAR=VAL" lines for the wrap
  local clk_label
  if [[ -z "${clk}" ]]; then
    clk_label="uncapped"
  else
    clk_label="${clk}mhz"
  fi

  local run_id="${model_key}_tp${tp}_${clk_label}"
  local out_dir="${CAMPAIGN_DIR}/runs/${run_id}"
  local log_prefix="${JOBS_ROOT}/logs/prefval_${run_id}"

  local gres="gpu:v100:${gpus}"

  echo "  ${run_id}: nodelist=${NODELIST} gres=${gres} time=${walltime}" >&2

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_id}" "dry" "${model_key}" "${clk:-uncapped}" "dry_run" \
      >> "${MANIFEST}"
    echo "    [DRY_RUN] skipping submission" >&2
    return 0
  fi

  local freq_env=""
  if [[ -n "${clk}" ]]; then
    freq_env="GPU_FREQ_MHZ=${clk}"
  fi

  local fix_input_env=""
  if [[ -n "${FIX_INPUT_LENGTH:-}" ]]; then
    fix_input_env="FIX_INPUT_LENGTH=${FIX_INPUT_LENGTH}"
  fi

  local jid_raw jid
  jid_raw=$(sbatch -M htc --parsable \
    --clusters=htc \
    --account=engs-glass \
    --partition=interactive \
    --nodelist="${NODELIST}" \
    --gres="${gres}" \
    --nodes=1 \
    --cpus-per-task=16 \
    --mem=64G \
    --exclusive \
    --time="${walltime}" \
    --job-name="prefval_${run_id}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" \
    --mail-type=END,FAIL \
    --output="${log_prefix}_%j.out" \
    --error="${log_prefix}_%j.err" \
    --wrap="set -euo pipefail
export HF_TOKEN=\"\$(cat '${ENGS_GLASS}/.secrets/hf_token')\"
export MODEL='${model}'
export TP_SIZE='${tp}'
export OUT_DIR='${out_dir}'
export CAMPAIGN_DIR='${CAMPAIGN_DIR}'
export ARM_LABEL='${clk_label}'
${freq_env:+export ${freq_env}}
${fix_input_env:+export ${fix_input_env}}
${extra_env:+${extra_env}}
bash '${JOBS_ROOT}/run_arc_v100_prefill_validation.sh'")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_id}" "${jid}" "${model_key}" "${clk:-uncapped}" "submitted" \
    >> "${MANIFEST}"
  echo "    submitted job ${jid}" >&2
}

echo "=== Prefill validation matrix ===" >&2
echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}" >&2
echo "NODELIST=${NODELIST}  DRY_RUN=${DRY_RUN}" >&2
echo "FREQ_ARRAY=(uncapped ${FREQ_LIST})" >&2
echo "" >&2

for model_key in ${MODELS}; do
  case "${model_key}" in
    phi)
      model="microsoft/Phi-tiny-MoE-instruct"
      tp=1; gpus=1; walltime="${PHI_TIME}"
      extra_env=""
      MAX_MODEL_LEN=2048 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=4096 \
      GPU_MEMORY_UTILIZATION=0.92
      ;;
    qwen)
      model="Qwen/Qwen3-30B-A3B-Instruct-2507"
      tp=4; gpus=4; walltime="${QWEN_TIME}"
      extra_env=""
      MAX_MODEL_LEN=4096 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=8192 \
      GPU_MEMORY_UTILIZATION=0.92
      ;;
    qwen15moe)
      # Qwen1.5-MoE-A2.7B-Chat: 14.3B params, ~29GB fp16 — tight, use 0.95
      # Profiler tables: V100 + V100_700/900/1100/1300MHz (moe.csv present)
      model="Qwen/Qwen1.5-MoE-A2.7B-Chat"
      tp=1; gpus=1; walltime="${QWEN15_TIME:-00:25:00}"
      extra_env="export GPU_MEMORY_UTILIZATION=0.95
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS=8192"
      ;;
    llama8b)
      # Llama-3.1-8B: 8B dense, ~16GB fp16 — fits comfortably on 1 V100
      # Profiler tables: V100/fp16/tp1 (attention+dense); no per-clock profiles
      model="meta-llama/Llama-3.1-8B"
      tp=1; gpus=1; walltime="${LLAMA8B_TIME:-00:20:00}"
      extra_env="export GPU_MEMORY_UTILIZATION=0.92
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS=8192"
      ;;
    qwen30btp2)
      # Qwen3-30B-A3B tp2: 60GB total / 2 GPUs = 30GB each — tight, use 0.95
      # Profiler tables: V100/fp16/tp2 (attention+dense); no moe.csv, no per-clock
      model="Qwen/Qwen3-30B-A3B-Instruct-2507"
      tp=2; gpus=2; walltime="${QWEN30BTP2_TIME:-00:35:00}"
      extra_env="export GPU_MEMORY_UTILIZATION=0.95
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=8
export MAX_NUM_BATCHED_TOKENS=8192"
      ;;
    *)
      echo "Unknown model_key: ${model_key}" >&2
      continue
      ;;
  esac
  echo "  model=${model_key} (${model})  tp=${tp}  gpus=${gpus}" >&2

  for clk in "${FREQ_ARRAY[@]}"; do
    submit_one "${model_key}" "${model}" "${tp}" "${clk}" "${walltime}" "${gpus}" "${extra_env}"
    sleep 0.3
  done
done

echo "" >&2
echo "Campaign: ${CAMPAIGN_DIR}" >&2
echo "Manifest: ${MANIFEST}" >&2
echo "Monitor:  squeue -M htc -u \$USER --format='%.10i %.30j %.2t %.10M %b %R'" >&2
echo "" >&2
echo "Collect:  python3 bench/jobs/collect_prefill_validation_results.py ${CAMPAIGN_DIR}" >&2
