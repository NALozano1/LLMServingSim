#!/usr/bin/env bash
# Submit Qwen3-235B-A22B profiler on ARC H100 (4× H100 per node for tp4).
# Sweeps multiple GPU clock frequencies plus one uncapped baseline.
#
# Freq sweep: 1470, 1620, 1755 MHz (supported H100 graphics clocks; mem fixed 1593)
#             plus uncapped baseline (GPU_FREQ_MHZ unset, tag H100).
# Total: 4 jobs.
#
# Outputs land under:
#   profiler/perf/H100/Qwen/Qwen3-235B-A22B-Instruct-2507/bf16/         (uncapped)
#   profiler/perf/H100_<MHz>MHz/Qwen/Qwen3-235B-A22B-Instruct-2507/bf16/ (locked)
#
#   bash profiler/jobs/submit_arc_h100_qwen235b.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/profiler/jobs/run_arc_h100_profile.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
LOGS="${ROOT}/profiler/jobs/logs"
RENDERED="${ROOT}/profiler/jobs/rendered"
export ROOT RUNNER TEMPLATE LOGS RENDERED
mkdir -p "${LOGS}" "${RENDERED}"

MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
JOB_NAME_BASE="llmsim_prof_h100_qwen235b"
export PARTITION="${PARTITION:-short}"
export TIME="${TIME:-12:00:00}"

export MODEL
export TP_DEGREES="${TP_DEGREES:-1,4}"
export DTYPE="${DTYPE:-bfloat16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1048576}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-8192}"
export MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"
export SKIP_SKEW="${SKIP_SKEW:-1}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}"
export VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"

# Freq sweep: explicit MHz values + empty string for uncapped baseline.
FREQ_LIST=(1470 1620 1755 "")

submit_one() {
  local freq_mhz="$1"  # empty string = uncapped baseline

  if [[ -n "${freq_mhz}" ]]; then
    local hw="H100_${freq_mhz}MHz"
    local job_name="${JOB_NAME_BASE}_${freq_mhz}MHz"
  else
    local hw="H100"
    local job_name="${JOB_NAME_BASE}"
  fi

  local rendered_sbatch="${RENDERED}/${job_name}.sbatch"

  python3 - <<PY
from pathlib import Path
import os

template = Path(os.environ["TEMPLATE"]).read_text()
runner   = os.environ["RUNNER"]
freq_mhz = os.environ.get("_FREQ_MHZ", "")
hw       = os.environ["_HW"]

# Replace standard template tokens.
body = template
body = body.replace("{{JOB_NAME}}",    os.environ["_JOB_NAME"])
body = body.replace("{{PROJECT}}",     "engs2950")
body = body.replace("{{PARTITION}}",   os.environ["PARTITION"])
body = body.replace("{{DATA_OUTPUT}}", os.environ["ROOT"] + "/profiler/perf")

# Inject H100-specific SBATCH directives after the partition line.
h100_directives = """#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time={time}
""".format(time=os.environ["TIME"])

body = body.replace(
    "#SBATCH --partition={{PARTITION}}\n".replace("{{PARTITION}}", os.environ["PARTITION"]),
    "#SBATCH --partition={part}\n{extra}".format(
        part=os.environ["PARTITION"], extra=h100_directives
    ),
    1,
)

# Append the actual command.
freq_export = ""
if freq_mhz:
    freq_export = "export GPU_FREQ_MHZ='{freq}'\n".format(freq=freq_mhz)

cmd = """
source /data/engs-glass/engs2950/shared/scripts/load_hf_token.sh
export MODEL='{model}'
export HARDWARE='{hw}'
export TP_DEGREES='{tp}'
export DTYPE='{dtype}'
export MAX_NUM_BATCHED_TOKENS='{mnbt}'
export MAX_NUM_SEQS='{msq}'
export ATTENTION_MAX_KV='{kv}'
export MEASUREMENT_ITERATIONS='{iters}'
export SKIP_SKEW='{skew}'
export HF_CACHE_ROOT='{hf}'
export VLLM_IMAGE='{img}'
{freq_export}bash '{runner}'
""".format(
    model=os.environ["MODEL"],
    hw=hw,
    tp=os.environ["TP_DEGREES"],
    dtype=os.environ["DTYPE"],
    mnbt=os.environ["MAX_NUM_BATCHED_TOKENS"],
    msq=os.environ["MAX_NUM_SEQS"],
    kv=os.environ["ATTENTION_MAX_KV"],
    iters=os.environ["MEASUREMENT_ITERATIONS"],
    skew=os.environ.get("SKIP_SKEW", "1"),
    hf=os.environ["HF_CACHE_ROOT"],
    img=os.environ["VLLM_IMAGE"],
    freq_export=freq_export,
    runner=runner,
)
body += cmd

Path(os.environ["_RENDERED_SBATCH"]).write_text(body)
print(f"Rendered: {os.environ['_RENDERED_SBATCH']}")
PY

  echo "Submitting H100 Qwen3-235B job: HARDWARE=${hw} GPU_FREQ_MHZ=${freq_mhz:-<uncapped>} ..."
  sbatch_out=$(sbatch --clusters=htc "${rendered_sbatch}" 2>&1)
  echo "  sbatch: ${sbatch_out}"
  jid=$(echo "${sbatch_out}" | grep -oP '(?<=batch job )\d+' || true)
  echo "  job ${jid}  MODEL=${MODEL}  HARDWARE=${hw}  GPU_FREQ_MHZ=${freq_mhz:-<uncapped>}  TP=${TP_DEGREES}"
  echo "  perf ${ROOT}/profiler/perf/${hw}/${MODEL}/"
  echo "  log  ${LOGS}/${jid}.out"
}

for freq in "${FREQ_LIST[@]}"; do
  export _FREQ_MHZ="${freq}"
  if [[ -n "${freq}" ]]; then
    export _HW="H100_${freq}MHz"
    export _JOB_NAME="${JOB_NAME_BASE}_${freq}MHz"
  else
    export _HW="H100"
    export _JOB_NAME="${JOB_NAME_BASE}"
  fi
  export _RENDERED_SBATCH="${RENDERED}/${_JOB_NAME}.sbatch"
  submit_one "${freq}"
done
