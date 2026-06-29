#!/usr/bin/env bash
# V100 tp1 vLLM bench presets for MoE models that barely fit 32GB HBM.
#
# Usage (sourced by run_arc_v100_bench_throughput.sh):
#   v100_bench_apply_preset "${MODEL}"
#
# Override any value after sourcing, or set V100_BENCH_PRESET=default to skip.
#
# Models:
#   Qwen/Qwen1.5-MoE-A2.7B-Chat       — needs qwen15-v100-tight (weights ~31 GiB fp16)
#
# shellcheck shell=bash

v100_bench_apply_preset() {
  local model="${1:-}"
  local preset="${V100_BENCH_PRESET:-auto}"

  if [[ "${preset}" == "default" ]]; then
    return 0
  fi

  if [[ "${preset}" == "auto" ]]; then
    case "${model}" in
      Qwen/Qwen1.5-MoE*|Qwen/Qwen1.5-MoE-A2.7B*)
        preset="qwen15-v100-tight"
        ;;
      *)
        preset="default"
        ;;
    esac
  fi

  case "${preset}" in
    default) ;;
    qwen15-v100-tight|tight)
      MAX_MODEL_LEN=512
      MAX_NUM_SEQS=8
      MAX_NUM_BATCHED_TOKENS=1024
      GPU_MEMORY_UTILIZATION=0.98
      SHAREGPT_FIX_LEN=1
      FIX_INPUT_LENGTH="${FIX_INPUT_LENGTH:-192}"
      FIX_OUTPUT_LENGTH="${FIX_OUTPUT_LENGTH:-64}"
      ;;
    *)
      echo "WARN: unknown V100_BENCH_PRESET=${preset}; using env defaults" >&2
      ;;
  esac
}
