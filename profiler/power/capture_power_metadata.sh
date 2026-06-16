#!/usr/bin/env bash
# Emit metadata.json + CSV comment headers for GPU/server power captures.
#
# Usage:
#   capture_power_metadata.sh <out_dir> [gpu_freq_mhz]
#
set -euo pipefail

OUT_DIR="${1:?out_dir required}"
GPU_FREQ_MHZ="${2:-}"

mkdir -p "${OUT_DIR}"

NODE="$(hostname -s 2>/dev/null || hostname)"
NODELIST="${SLURM_NODELIST:-}"
if [[ -n "${NODELIST}" ]] && command -v scontrol >/dev/null 2>&1; then
  NODELIST="$(scontrol show hostnames "${NODELIST}" | paste -sd, -)"
fi

IPMI_OK=0
if ipmitool dcmi power reading >/dev/null 2>&1; then
  IPMI_OK=1
fi

GPU_INVENTORY="${OUT_DIR}/gpu_inventory.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
  {
    echo "# node=${NODE} nodelist=${NODELIST:-} slurm_job_id=${SLURM_JOB_ID:-} gpu_freq_mhz=${GPU_FREQ_MHZ:-default}"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,serial,clocks.current.graphics,clocks.max.graphics,power.limit \
      --format=csv
  } > "${GPU_INVENTORY}"
else
  echo "index,uuid,name,pci.bus_id,driver_version,serial,clocks.current.graphics,clocks.max.graphics,power.limit" > "${GPU_INVENTORY}"
fi

export OUT_DIR NODE NODELIST GPU_FREQ_MHZ IPMI_OK

python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
node = os.environ.get("NODE", "")
nodelist = os.environ.get("NODELIST", "")
freq = os.environ.get("GPU_FREQ_MHZ", "")
ipmi_ok = os.environ.get("IPMI_OK", "0") == "1"

gpus = []
inv = out_dir / "gpu_inventory.csv"
if inv.exists():
    lines = inv.read_text().splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line.startswith("index,"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        gpus.append({
            "index": parts[0],
            "uuid": parts[1] if len(parts) > 1 else "",
            "name": parts[2] if len(parts) > 2 else "",
            "pci_bus_id": parts[3] if len(parts) > 3 else "",
            "driver_version": parts[4] if len(parts) > 4 else "",
            "serial": parts[5] if len(parts) > 5 else "",
            "clocks_current_graphics_mhz": parts[6] if len(parts) > 6 else "",
            "clocks_max_graphics_mhz": parts[7] if len(parts) > 7 else "",
            "power_limit_w": parts[8] if len(parts) > 8 else "",
        })

meta = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "hostname": node,
    "slurm_nodelist": nodelist,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
    "slurm_gres": os.environ.get("SLURM_JOB_GRES", ""),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "gpu_freq_mhz": int(freq) if freq.isdigit() else None,
    "gpu_freq_label": f"{freq}MHz" if freq.isdigit() else "default",
    "ipmitool_dcmi_available": ipmi_ok,
    "gpus": gpus,
}
(out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

# CSV headers (comment lines preserved for traceability).
{
  echo "# node=${NODE} nodelist=${NODELIST:-} slurm_job_id=${SLURM_JOB_ID:-} gpu_freq_mhz=${GPU_FREQ_MHZ:-default}"
  echo "# inventory=${GPU_INVENTORY}"
  echo "timestamp,index,uuid,name,pci.bus_id,utilization.gpu,power.draw,clocks.current.graphics"
} > "${OUT_DIR}/gpu_power.csv"

{
  echo "# node=${NODE} nodelist=${NODELIST:-} slurm_job_id=${SLURM_JOB_ID:-} ipmitool_dcmi=${IPMI_OK}"
  echo "timestamp,server_power_w,status"
} > "${OUT_DIR}/server_power.csv"

echo "metadata: ${OUT_DIR}/metadata.json"
