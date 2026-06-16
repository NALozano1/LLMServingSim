#!/usr/bin/env bash
# CLI wrapper for ``python -m profiler profile`` (same knobs as profile.sh).
#
# Usage:
#   ./profiler/profile_cli.sh \
#       --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#       --hardware V100 \
#       --tp 1,4 \
#       --dtype float16 \
#       --skip-skew \
#       --verbose
#
# For edit-in-place defaults, use ./profiler/profile.sh instead.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: profile_cli.sh --model <HF id> --hardware <label> [options]

Required:
  --model <id>              HF-style model path under configs/model/
  --hardware <label>        Output folder tag under profiler/perf/ (e.g. V100)

Options:
  --tp <degrees>            Comma-separated TP sweep (default: 1)
  --dtype <dtype>           bfloat16 | float16 | float32 | fp8
  --kv-cache-dtype <dtype>  auto | fp8 | fp16 | bf16
  --max-num-batched-tokens <n>
  --max-num-seqs <n>
  --attention-max-kv <n>
  --attention-chunk-factor <f>
  --attention-kv-factor <f>
  --measurement-iterations <n>
  --skew-n-factor <f>  --skew-pc-factor <f>  --skew-kp-factor <f>  --skew-kvs-factor <f>
  --variant <name>
  --skip-skew               Skip heterogeneous-decode skew sweep
  --only-skew               Run skew sweep only
  --force                   Wipe CSVs and re-profile
  --verbose                 DEBUG + vLLM stdout
  --silent                  Warnings only
  -h, --help
EOF
}

MODEL=""
HARDWARE=""
TP_DEGREES="1"
VERBOSITY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --hardware) HARDWARE="$2"; shift 2 ;;
    --tp) TP_DEGREES="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --attention-max-kv) ATTENTION_MAX_KV="$2"; shift 2 ;;
    --attention-chunk-factor) ATTENTION_CHUNK_FACTOR="$2"; shift 2 ;;
    --attention-kv-factor) ATTENTION_KV_FACTOR="$2"; shift 2 ;;
    --measurement-iterations) MEASUREMENT_ITERATIONS="$2"; shift 2 ;;
    --skew-n-factor) SKEW_N_FACTOR="$2"; shift 2 ;;
    --skew-pc-factor) SKEW_PC_FACTOR="$2"; shift 2 ;;
    --skew-kp-factor) SKEW_KP_FACTOR="$2"; shift 2 ;;
    --skew-kvs-factor) SKEW_KVS_FACTOR="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --skip-skew) SKIP_SKEW=1; shift ;;
    --only-skew) ONLY_SKEW=1; shift ;;
    --force) FORCE=1; shift ;;
    --verbose) VERBOSITY="--verbose"; shift ;;
    --silent) VERBOSITY="--silent"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$HARDWARE" ]]; then
  echo "error: --model and --hardware are required" >&2
  usage >&2
  exit 1
fi

# shellcheck source=profiler/_profile_run.sh
source "$(dirname "${BASH_SOURCE[0]}")/_profile_run.sh"
