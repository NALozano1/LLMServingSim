# Shared profiler launcher — expects MODEL, HARDWARE, and optional knobs set.
# Sourced by profiler/profile.sh and profiler/profile_cli.sh.

: "${MODEL:?MODEL is required}"
: "${HARDWARE:?HARDWARE is required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
cd "$REPO_ROOT"

cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")

[[ -n "${TP_DEGREES:-}" ]]             && cmd+=(--tp "$TP_DEGREES")
[[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "${MAX_NUM_SEQS:-}" ]]           && cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -n "${ATTENTION_MAX_KV:-}" ]]       && cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
[[ -n "${ATTENTION_CHUNK_FACTOR:-}" ]] && cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
[[ -n "${ATTENTION_KV_FACTOR:-}" ]]    && cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
[[ -n "${MEASUREMENT_ITERATIONS:-}" ]] && cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
[[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
[[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
[[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
[[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
[[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
[[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)

"${cmd[@]}"
