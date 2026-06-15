# Forward segments — ASTRA debug log

Running log for `--forward-segments per_block` E2E failures. Update as we reproduce and fix.

---

## 2026-06-15 — Root cause: partial Chakra graphs crash ASTRA

### Symptom
- Monolithic sim (`--forward-segments off`) completes (~7s, 1 req).
- Segmented sim hangs in Python at `Controller.read_wait` after first segment submit.

### Initial hypothesis (wrong)
`read_wait` infinite loop — trace/graph generation actually finishes in &lt;1ms. The ASTRA **subprocess** dies or stalls; Python blocks waiting for `"Waiting"` on stdout.

### Isolated ASTRA repro (inside `servingsim_docker`)

```bash
cd /app/LLMServingSim/astra-sim
# Full monolithic graph — OK
(echo ./inputs/workload/event_handler/llm; sleep 1; \
 echo ./inputs/workload/RTXPRO6000/meta-llama/Llama-3.1-8B/instance0_batch0/llm; \
 sleep 1; echo pass; echo exit) | \
 ./build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra \
   --workload-configuration=./inputs/workload/event_handler/llm \
   --system-configuration=./inputs/system/system.json \
   --network-configuration=./inputs/network/network.yml \
   --memory-configuration=./inputs/memory/memory_expansion.json \
   --start-npu-ids=0, --end-npu-ids=0,
# → All Request Has Been Exited

# Partial segment s0 (noop+embedding) — SIGSEGV after iter 1
(echo .../instance0_t10p1_s0/llm; echo pass; echo exit) | AnalyticalAstra ...
# → exit 139
```

### Matrix (hot-switch: event_handler → workload → pass → exit)

| Workload shape | ET size | Result |
|----------------|---------|--------|
| Full `instance0_batch0` (292 layers) | 64 KB | **OK** |
| Segment s0 (noop + embedding) | 854 B | **SIGSEGV** on pass |
| Segment s1 (one transformer block) | 2.4 KB | **SIGSEGV** on pass |
| Embed + block0 + real head tail | ~2.4 KB | **OK** |
| Zero-time embed stub + block0 + head tail | ~2.4 KB | **OK** |
| Zero embed stub + real head only | small | **OK** |
| Block0 only (no embed/head bookends) | ~2.4 KB | **SIGSEGV** |

### Conclusion
Chakra/ASTRA requires a **complete forward skeleton**: trace must start with **embedding** and end with **head** (`final_layernorm`, `lm_head`, `sampler`). Per-block segments need **1 ns bookend stubs** (Chakra skips `comp_time==0` nodes) so only the active stage contributes meaningful latency.

### Secondary bug: `noop_layer` pad
When a segment had one layer, `trace_generator` inserted `noop_layer`, which became the Chakra **INPUT** `MEM_LOAD` node (wrong). Replaced by bookends; kv_load/kv_evict placeholders only if still needed.

### Fix (Python-only)
In `trace_generator.generate_trace` for `stage_idx is not None`:
- `stage_idx > 0`: prepend 1 ns embedding stub (`REMOTE` input).
- `stage_idx < num_stages - 1`: append 1 ns head stub layers.
- Remove `noop_layer` pad.
- Disable segment trace/graph file cache so bookends regenerate.

In `scheduler.add_done` / `__main__`: final segment calls `add_done(..., batch_id=seg_batch_id)` because iteration−1 no longer equals batch_id when a forward uses 34 ASTRA iterations.

### Validation commands

```bash
# Unit tests
cd LLMServingSim && PYTHONPATH=. python3 -m unittest serving.tests.test_forward_segments -v

# Clear stale segment artifacts (docker)
rm -rf astra-sim/inputs/trace/RTXPRO6000/meta-llama/Llama-3.1-8B/instance0_t10p1_s*
rm -rf astra-sim/inputs/workload/RTXPRO6000/meta-llama/Llama-3.1-8B/instance0_t10p1_s*

# E2E segmented smoke
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dtype bfloat16 --dataset workloads/example_trace.jsonl \
  --output outputs/segment_per_block.csv --num-req 1 \
  --forward-segments per_block --log-level WARNING'
```

---

## Status

| Item | Status |
|------|--------|
| Bookend stubs in `trace_generator.py` | implemented |
| `noop_layer` removed | implemented |
| `Controller.read_wait` EOF guard | implemented |
| Hot-switch s0/s1/s33 | done |
| E2E `--forward-segments per_block` (1 req) | done (~2m 24s wall) |
