#!/usr/bin/env bash
# Verify configs/model/<HF id>.json exists and profiler/models/<model_type>.yaml resolves.
#
#   ./profiler/scripts/verify_model_arch.sh
#   ./profiler/scripts/verify_model_arch.sh Qwen/Qwen1.5-MoE-A2.7B-Chat
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ $# -eq 0 ]]; then
  set -- \
    Qwen/Qwen1.5-MoE-A2.7B-Chat
fi

fail=0
for model in "$@"; do
  cfg="configs/model/${model}.json"
  if [[ ! -f "${cfg}" ]]; then
    echo "FAIL ${model}: missing ${cfg}"
    fail=1
    continue
  fi
  mt="$(jq -r '.model_type // empty' "${cfg}")"
  if [[ -z "${mt}" ]]; then
    echo "FAIL ${model}: no model_type in ${cfg}"
    fail=1
    continue
  fi
  yaml="profiler/models/${mt}.yaml"
  if [[ ! -f "${yaml}" ]]; then
    echo "FAIL ${model}: model_type=${mt} but missing ${yaml}"
    echo "      Available: $(ls profiler/models/*.yaml | xargs -n1 basename | sed 's/.yaml//' | paste -sd, -)"
    fail=1
    continue
  fi
  echo "OK   ${model}  model_type=${mt}  ->  ${yaml}"
done

exit "${fail}"
