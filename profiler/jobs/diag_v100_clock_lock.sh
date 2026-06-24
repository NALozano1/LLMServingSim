#!/usr/bin/env bash
# v3: confirm the APPLY-TIMING root cause and validate the fix, back to back.
#   Test X (reproduce bug): apply lock with NO CUDA context, THEN start the burn.
#                           Expect ~1440 under load (lock did not stick).
#   Test Y (proposed fix):  start the burn (context live), THEN apply lock.
#                           Expect ~1100 under load (lock sticks).
# If X~1440 and Y~1100 -> fix = apply the clock lock only after a CUDA context exists.
set -uo pipefail

ENGS_GLASS="/data/engs-glass/engs2950"
REPO_ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
JOBS_ROOT="${REPO_ROOT}/profiler/jobs"
JOB_TAG="${SLURM_JOB_ID:-local}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.19.0}"
CLOCKS_BIN="/usr/local/sbin/nvidia-smi-clocks"
TARGET=1100

export APPTAINER_CACHEDIR="${ENGS_GLASS}/.apptainer_cache/cache"
export APPTAINER_TMPDIR="${JOBS_ROOT}/apptainer_tmp/${JOB_TAG}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

hlock()  { sudo -n "${CLOCKS_BIN}" --id 0 "${TARGET}" "${TARGET}" 2>&1 | sed 's/^/    helper> /' | head -3; }
hreset() { sudo -n "${CLOCKS_BIN}" --id 0 --reset 2>&1 | sed 's/^/    helper> /' | head -2 || true; }
read_clk() {
  local mhz app util pw
  IFS=',' read -r mhz app pw util < <(nvidia-smi --query-gpu=clocks.gr,clocks.applications.gr,power.draw,utilization.gpu \
      --format=csv,noheader,nounits -i 0 | head -1)
  printf "  [%-18s] gr=%4s app=%4s util=%3s%% pw=%sW\n" "$1" "${mhz// /}" "${app// /}" "${util// /}" "${pw// /}"
}
start_burn() {  # launches background burn, returns pid; secs=$1
  apptainer exec --cleanenv --nv -B "${REPO_ROOT}:${REPO_ROOT}" -B /dev/shm:/dev/shm \
    --env "HOME=${APPTAINER_TMPDIR}" "${VLLM_IMAGE}" python3 - "$1" <<'PY' &
import sys, time, torch
secs=float(sys.argv[1]); d=torch.device('cuda')
a=torch.randn(8192,8192,device=d); b=torch.randn(8192,8192,device=d)
t0=time.time()
while time.time()-t0 < secs:
    for _ in range(40): a=(a@b).relu()*1.0001
    torch.cuda.synchronize()
PY
  echo $!
}
wait_warm() {  # wait until util>=80 (torch import on first run takes ~20-40s)
  for _ in $(seq 1 50); do
    u="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -dc '0-9')"
    [ "${u:-0}" -ge 80 ] 2>/dev/null && { echo "  (warm: util=${u}%)"; return; }
    sleep 2
  done
  echo "  (WARN: burn never reached 80% util)"
}
sample() { echo "--- $1 (target ${TARGET}) ---"; for _ in $(seq 1 6); do read_clk "$1"; sleep 1; done; }

echo "================ ENV ================"
echo "node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,uuid,persistence_mode,clocks.max.gr --format=csv,noheader
hreset  # clean slate

echo ""; echo "################ TEST X: lock-then-load (reproduce bug) ################"
echo "[X] applying lock with NO CUDA context held:"
hlock
echo "[X] now starting burn..."
BPID=$(start_burn 70); wait_warm
sample "X:lock-then-load"
kill "${BPID}" 2>/dev/null; wait "${BPID}" 2>/dev/null || true
hreset
sleep 3

echo ""; echo "################ TEST Y: load-then-lock (proposed fix) ################"
echo "[Y] starting burn FIRST (context live)..."
BPID=$(start_burn 70); wait_warm
echo "[Y] now applying lock (context already held):"
hlock
sample "Y:load-then-lock"
kill "${BPID}" 2>/dev/null; wait "${BPID}" 2>/dev/null || true
hreset

echo ""; echo "================ VERDICT ================"
echo "X~1440 & Y~1100  => root cause = apply-timing; fix = lock after CUDA context exists."
