#!/usr/bin/env bash
# Bundle Qwen3-30B-A3B profiler outputs (V100 smoke + DVFS + RTX reference) into one tarball.
#
#   ./profiler/jobs/package_qwen30b_profiles.sh
#
set -euo pipefail

ROOT="/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"
MODEL_DIR="Qwen/Qwen3-30B-A3B-Instruct-2507"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
PKG_DIR="${ROOT}/profiler/jobs/packages/qwen30b_profiles_${STAMP}"
ARCHIVE="${ROOT}/profiler/jobs/packages/qwen30b_profiles_${STAMP}.tar.gz"
MODEL_GLOB="Qwen3-30B-A3B-Instruct-2507"

mkdir -p "${PKG_DIR}/perf" "${PKG_DIR}/checkpoints" "${PKG_DIR}/logs" "${PKG_DIR}/meta"

# V100 baseline + DVFS tags + kv probe trees (if any).
while IFS= read -r hw; do
  src="${ROOT}/profiler/perf/${hw}/Qwen/${MODEL_GLOB}"
  [[ -d "${src}" ]] || continue
  dest="${PKG_DIR}/perf/${hw}/Qwen/${MODEL_GLOB}"
  mkdir -p "$(dirname "${dest}")"
  cp -a "${src}" "${dest}"
done < <(find "${ROOT}/profiler/perf" -maxdepth 1 -mindepth 1 -type d \
  \( -name 'V100' -o -name 'V100_*' -o -name 'V100_kv*' \) -printf '%f\n' | sort -u)

# RTX reference (complete bf16 profile from earlier work).
if [[ -d "${ROOT}/profiler/perf/RTXPRO6000/${MODEL_DIR}" ]]; then
  mkdir -p "${PKG_DIR}/perf/RTXPRO6000"
  cp -a "${ROOT}/profiler/perf/RTXPRO6000/${MODEL_DIR}" \
    "${PKG_DIR}/perf/RTXPRO6000/"
fi

for f in \
  checkpoints/v100_matrix_jobs.tsv \
  checkpoints/v100_freq_matrix_jobs.tsv \
  checkpoints/qwen30b_kv_probe_jobs.tsv \
  checkpoints/qwen30b_kv_probe_results.tsv; do
  if [[ -f "${ROOT}/profiler/jobs/${f}" ]]; then
    cp -a "${ROOT}/profiler/jobs/${f}" "${PKG_DIR}/checkpoints/"
  fi
done

# Recent qwen30b Slurm logs.
find "${ROOT}/profiler/jobs/logs" -maxdepth 1 -type f -name 'llmsim_qwen30b*' -mtime -14 \
  -exec cp -a {} "${PKG_DIR}/logs/" \; 2>/dev/null || true

export PKG_DIR
python3 - <<'PY' > "${PKG_DIR}/meta/INVENTORY.txt"
from pathlib import Path
import os

pkg = Path(os.environ["PKG_DIR"])
print("Qwen3-30B-A3B-Instruct-2507 profiler bundle")
print("=" * 60)
for meta in sorted(pkg.glob("perf/**/meta.yaml")):
    hw = meta.parts[meta.parts.index("perf") + 1]
  # model path between hw and variant
    rel = meta.relative_to(pkg / "perf")
    rows = {}
    tp_dir = meta.parent
    for csv in sorted(tp_dir.glob("tp*/**/*.csv")):
        tp = csv.parent.name
        rows.setdefault(tp, []).append(csv.name)
    print(f"\n{hw}  ({rel.parent})")
    print(f"  meta.yaml: yes")
    for tp in sorted(rows):
        print(f"  {tp}: {', '.join(sorted(set(rows[tp])))}")
PY

cat > "${PKG_DIR}/README.txt" <<EOF
Qwen3-30B-A3B-Instruct-2507 LLMServingSim profiler bundle
Created: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Contents:
  perf/          V100 baseline (tp1,2,4), V100_<MHz> DVFS (tp1), optional V100_kv* probe
  perf/RTXPRO6000/  Reference bf16 profile (tp1,tp2)
  checkpoints/   Job manifests and KV probe results TSV
  logs/          Recent llmsim_qwen30b_* Slurm stdout/stderr
  meta/INVENTORY.txt  Per-hardware CSV inventory

Smoke settings (V100*): ATTENTION_MAX_KV=256, MSQ=32, MNBT=256, MEASUREMENT_ITERATIONS=1
Done marker: meta.yaml under each hardware tag's fp16/ directory.
EOF

tar -czf "${ARCHIVE}" -C "${PKG_DIR}/.." "$(basename "${PKG_DIR}")"

echo "Package dir:  ${PKG_DIR}"
echo "Archive:      ${ARCHIVE}"
echo "Size:         $(du -h "${ARCHIVE}" | cut -f1)"
echo ""
cat "${PKG_DIR}/meta/INVENTORY.txt"
