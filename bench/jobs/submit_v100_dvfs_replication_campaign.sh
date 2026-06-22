#!/usr/bin/env bash
# Submit 10 real-vLLM bench runs (Phi-tiny + Qwen1.5-MoE) with DVFS permutations.
#
# TTFT/TPOT come from live bench requests.jsonl — NOT from LLMServingSim synthesis.
# Each run writes sim_replication.json with the matching simulator command + profile path.
#
#   ./bench/jobs/submit_v100_dvfs_replication_campaign.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
RUNNER="${ROOT}/bench/jobs/run_arc_v100_bench_campaign.sh"
TEMPLATE="${HOME}/arc_slurm_submission_template.sh"
ARC_COMMON="/data/engs-glass/engs2950/shared/gpu_address_tracing/jobs/launchers/arc_slurm_common.sh"
COMMON="${ROOT}/profiler/jobs/v100_matrix_common.sh"

# shellcheck source=/dev/null
source "${ARC_COMMON}"
# shellcheck source=/dev/null
source "${COMMON}"

CAMPAIGN_ID="${CAMPAIGN_ID:-v100_dvfs_replication_$(date -u +%Y%m%d)}"
CAMPAIGN_DIR="${ROOT}/bench/campaigns/${CAMPAIGN_ID}"
TIME="${TIME:-01:30:00}"
NUM_REQS="${NUM_REQS:-100}"
SPS="${SPS:-10}"
SEED="${SEED:-42}"
MANIFEST="${CAMPAIGN_DIR}/jobs_manifest.tsv"
SUBMIT_LOG="${CAMPAIGN_DIR}/submit.log"

mkdir -p "${CAMPAIGN_DIR}/runs" "${CAMPAIGN_DIR}/shared" \
  "${ROOT}/profiler/jobs/rendered" "${ROOT}/profiler/jobs/logs"

python3 - <<PY
import json
from pathlib import Path

campaign = Path("${CAMPAIGN_DIR}")
runs = [
    ("r01_phi_default", "microsoft/Phi-tiny-MoE-instruct", "default", None, None, "phi-tiny-fixlen"),
    ("r02_phi_700", "microsoft/Phi-tiny-MoE-instruct", "fixed", 700, None, "phi-tiny-fixlen"),
    ("r03_phi_1100", "microsoft/Phi-tiny-MoE-instruct", "fixed", 1100, None, "phi-tiny-fixlen"),
    ("r04_phi_700_1300", "microsoft/Phi-tiny-MoE-instruct", "mid_switch", None, [700, 1300], "phi-tiny-fixlen"),
    ("r05_phi_1300_700", "microsoft/Phi-tiny-MoE-instruct", "mid_switch", None, [1300, 700], "phi-tiny-fixlen"),
    ("r06_qwen_default", "Qwen/Qwen1.5-MoE-A2.7B-Chat", "default", None, None, "qwen15-v100-tight"),
    ("r07_qwen_700", "Qwen/Qwen1.5-MoE-A2.7B-Chat", "fixed", 700, None, "qwen15-v100-tight"),
    ("r08_qwen_1100", "Qwen/Qwen1.5-MoE-A2.7B-Chat", "fixed", 1100, None, "qwen15-v100-tight"),
    ("r09_qwen_700_1100", "Qwen/Qwen1.5-MoE-A2.7B-Chat", "mid_switch", None, [700, 1100], "qwen15-v100-tight"),
    ("r10_qwen_900_1300", "Qwen/Qwen1.5-MoE-A2.7B-Chat", "mid_switch", None, [900, 1300], "qwen15-v100-tight"),
]

phi = dict(
    max_model_len=4096, max_num_seqs=64, max_num_batched_tokens=2048,
    sharegpt_fix_len=1, fix_input_length=128, fix_output_length=128,
    gpu_memory_utilization=0.92,
)
qwen = dict(
    max_model_len=512, max_num_seqs=8, max_num_batched_tokens=1024,
    sharegpt_fix_len=1, fix_input_length=192, fix_output_length=64,
    gpu_memory_utilization=0.98,
)

specs = []
for run_id, model, mode, mhz, sched, preset in runs:
    p = qwen if "Qwen" in model else phi
    if mode == "default":
        hardware = "V100"
    elif mode == "fixed":
        hardware = f"V100_{mhz}MHz"
    else:
        hardware = f"V100_dvfs_{sched[0]}_{sched[1]}"
    spec = {
        "run_id": run_id,
        "model": model,
        "mode": mode,
        "hardware": hardware,
        "bench_preset": preset,
        "dvfs": {
            "gpu_freq_mhz": mhz,
            "freq_schedule": sched,
            "hardware_label": hardware,
        },
        "num_reqs": int("${NUM_REQS}"),
        "sps": int("${SPS}"),
        "seed": int("${SEED}"),
        "dtype": "float16",
        "tp_size": 1,
        **p,
    }
    run_dir = campaign / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    specs.append(spec)

(campaign / "campaign_spec.json").write_text(json.dumps(specs, indent=2) + "\n")
print(f"wrote {len(specs)} run specs under {campaign / 'runs'}")
PY

cat > "${CAMPAIGN_DIR}/README.md" <<'EOF'
# V100 DVFS replication campaign

Real vLLM throughput benches (`python -m bench run`) for Phi-tiny-MoE and
Qwen1.5-MoE-A2.7B at default boost and locked DVFS frequencies (700–1300 MHz),
including mid-run frequency switches.

## Metrics source

**TTFT / TPOT / e2e latency** are extracted from `requests.jsonl` (live AsyncLLM
timestamps) into `runs/<id>/real_latency.json`. They are **not** from LLMServingSim
profile synthesis or the discrete-event simulator.

## Per-run outputs

```
runs/<run_id>/
  run_spec.json          — run configuration (tracked)
  run_meta.json          — Slurm node / freq metadata
  real_latency.json      — REAL TTFT/TPOT summary + per-request rows
  sim_replication.json   — LLMServingSim cluster config + serving command
  summary.json           — quick status + mean latencies
  bench/                 — meta.json, requests.jsonl, timeseries.csv
```

## After jobs finish

```bash
python3 bench/jobs/collect_campaign_results.py bench/campaigns/<campaign_id>/
```

## LLMServingSim replication

For each run, `sim_replication.json` contains:

- `llmservingsim.hardware` — profile tag (`V100`, `V100_700MHz`, …)
- `llmservingsim.profile_path` — expected profiler data directory
- `llmservingsim.serving_command` — ready-to-run simulator command
- `llmservingsim.validate_command` — compare sim vs real bench

Mid-switch runs document both phase profile paths; the simulator currently
uses one hardware tag per run — see `notes` in the JSON.
EOF

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'submitted_at\trun_id\tjob_id\tmodel\tmode\thardware\tgpu_freq_mhz\tfreq_schedule\tstatus\n' > "${MANIFEST}"
fi

submit_run() {
  local run_id="$1"
  local run_dir="${CAMPAIGN_DIR}/runs/${run_id}"
  local spec_path="${run_dir}/run_spec.json"
  local prev_jid="${2:-}"

  local spec model mode hardware mhz sched preset
  spec="$(python3 -c "import json; print(json.dumps(json.load(open('${spec_path}'))))")"
  model="$(python3 -c "import json; print(json.load(open('${spec_path}'))['model'])")"
  mode="$(python3 -c "import json; print(json.load(open('${spec_path}'))['mode'])")"
  hardware="$(python3 -c "import json; print(json.load(open('${spec_path}'))['hardware'])")"

  local gpu_freq_mhz="" bench_freq_schedule=""
  gpu_freq_mhz="$(python3 -c "import json; s=json.load(open('${spec_path}')); print(s['dvfs'].get('gpu_freq_mhz') or '')")"
  bench_freq_schedule="$(python3 -c "
import json
s=json.load(open('${spec_path}'))
fs=s['dvfs'].get('freq_schedule')
print(','.join(str(x) for x in fs) if fs else '')
")"

  local preset max_ml max_seqs max_mbt fix fix_in fix_out gmu
  preset="$(python3 -c "import json; print(json.load(open('${spec_path}'))['bench_preset'])")"
  max_ml="$(python3 -c "import json; print(json.load(open('${spec_path}'))['max_model_len'])")"
  max_seqs="$(python3 -c "import json; print(json.load(open('${spec_path}'))['max_num_seqs'])")"
  max_mbt="$(python3 -c "import json; print(json.load(open('${spec_path}'))['max_num_batched_tokens'])")"
  fix="$(python3 -c "import json; print(json.load(open('${spec_path}'))['sharegpt_fix_len'])")"
  fix_in="$(python3 -c "import json; print(json.load(open('${spec_path}'))['fix_input_length'])")"
  fix_out="$(python3 -c "import json; print(json.load(open('${spec_path}'))['fix_output_length'])")"
  gmu="$(python3 -c "import json; print(json.load(open('${spec_path}'))['gpu_memory_utilization'])")"

  local job_name="${run_id}"
  local rendered="${ROOT}/profiler/jobs/rendered/${job_name}.sbatch"
  local cmd_frag="${ROOT}/profiler/jobs/rendered/_${job_name}_cmd.sh"

  read -r -d '' CMD <<EOF || true
set -euo pipefail
source "${ENGS2950_ROOT:-/data/engs-glass/engs2950}/shared/scripts/load_hf_token.sh"
export CAMPAIGN_DIR='${CAMPAIGN_DIR}'
export RUN_ID='${run_id}'
export RUN_SPEC_JSON='${spec_path}'
export MODEL='${model}'
export HARDWARE='${hardware}'
export GPU_FREQ_MHZ='${gpu_freq_mhz}'
export BENCH_FREQ_SCHEDULE='${bench_freq_schedule}'
export V100_BENCH_PRESET='${preset}'
export NUM_REQS='${NUM_REQS}'
export SPS='${SPS}'
export SEED='${SEED}'
export MAX_MODEL_LEN='${max_ml}'
export MAX_NUM_SEQS='${max_seqs}'
export MAX_NUM_BATCHED_TOKENS='${max_mbt}'
export SHAREGPT_FIX_LEN='${fix}'
export FIX_INPUT_LENGTH='${fix_in}'
export FIX_OUTPUT_LENGTH='${fix_out}'
export GPU_MEMORY_UTILIZATION='${gmu}'
export HF_CACHE_ROOT='${HF_CACHE_ROOT:-/data/engs-glass/engs2950/infra/hf_cache}'
export VLLM_IMAGE='${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}'
export CONTAINER_RUNTIME='${CONTAINER_RUNTIME:-apptainer}'
bash "${RUNNER}"
EOF
  printf '%s' "${CMD}" > "${cmd_frag}"

  export TEMPLATE="${TEMPLATE}" CMD_FRAG="${cmd_frag}" RENDERED="${rendered}"
  export JOB_NAME="${job_name}" PROJECT=engs2950 PARTITION=interactive
  export DATA_OUTPUT="${CAMPAIGN_DIR}"
  python3 - <<'PY'
from pathlib import Path
import os
t = Path(os.environ["TEMPLATE"]).read_text()
cmd = Path(os.environ["CMD_FRAG"]).read_text()
for k, v in {
    "{{JOB_NAME}}": os.environ["JOB_NAME"],
    "{{PROJECT}}": os.environ["PROJECT"],
    "{{PARTITION}}": os.environ["PARTITION"],
    "{{DATA_OUTPUT}}": os.environ["DATA_OUTPUT"],
    "{{COMMAND}}": cmd,
}.items():
    t = t.replace(k, v)
Path(os.environ["RENDERED"]).write_text(t)
PY

  local dep_args=()
  if [[ -n "${prev_jid}" ]]; then
    dep_args+=(--dependency="afterany:${prev_jid}")
  fi

  local jid_raw jid
  jid_raw=$(sbatch --parsable --clusters=htc --account=engs-glass \
    --partition=interactive --gres=gpu:v100:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=64G --time="${TIME}" --job-name="${job_name}" \
    --mail-user="${MAIL_USER:-alex.lozano@eng.ox.ac.uk}" --mail-type=BEGIN,END,FAIL \
    --output="${ROOT}/profiler/jobs/logs/${job_name}_%j.out" \
    --error="${ROOT}/profiler/jobs/logs/${job_name}_%j.err" \
    "${dep_args[@]}" "${rendered}")
  jid="${jid_raw%%;*}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${run_id}" "${jid}" "${model}" "${mode}" \
    "${hardware}" "${gpu_freq_mhz:-default}" "${bench_freq_schedule:-}" \
    >> "${MANIFEST}"

  echo "  job ${jid}  ${run_id}  ${hardware}  freq=${gpu_freq_mhz:-default} sched=${bench_freq_schedule:-}" \
    | tee -a "${SUBMIT_LOG}" >&2
  echo "${jid}"
}

{
  echo "=== V100 DVFS replication campaign ==="
  echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
  echo "Runs: 10 (Phi-tiny x5 + Qwen1.5 x5)"
} | tee "${SUBMIT_LOG}"

arc_wait_htc_interactive_slot interactive || true

prev=""
for run_id in r01_phi_default r02_phi_700 r03_phi_1100 r04_phi_700_1300 r05_phi_1300_700 \
                r06_qwen_default r07_qwen_700 r08_qwen_1100 r09_qwen_700_1100 r10_qwen_900_1300; do
  prev="$(submit_run "${run_id}" "${prev}")"
done

{
  echo "=== Submitted campaign ${CAMPAIGN_ID} ==="
  echo "Last job: ${prev}"
  echo "Manifest: ${MANIFEST}"
  echo "Collect:  python3 bench/jobs/collect_campaign_results.py ${CAMPAIGN_DIR}"
} | tee -a "${SUBMIT_LOG}"

echo "${CAMPAIGN_DIR}"
