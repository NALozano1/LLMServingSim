# Forward segments implementation (layer-wise control)

## Summary
Adds `--forward-segments per_block` to split each forward pass into **34 ASTRA submissions** for Llama-8B (prologue + 32 blocks + head), enabling layer-boundary DVFS hooks.

## New flag
```
--forward-segments {off,per_block}   default: off
```

## Data flow
1. `Scheduler` creates batch → `_enable_segments()` sets `num_stages = num_hidden_layers + 2`.
2. `__main__` calls `generate_trace(..., stage_idx=batch.layer_cursor)` and registers `(npu, astra_iter) → (batch_id, stage, is_final)`.
3. On ASTRA completion: intermediate segments call `on_segment_done()`; final segment calls `add_done()`.
4. `awaiting_segment_submit` causes `schedule()` to return the same batch for the next segment.

## Key design choices (minimal diff)
- **Shape-based paths** (`instance0_t10p1_s3`) reuse traces/graphs when batch token shape matches.
- **Trace/graph cache** skips regeneration when files already exist.
- **ASTRA bookends**: non-final segments append 1 ns head stubs; non-prologue segments prepend 1 ns embedding stub (see `docs/forward_segments_debug.md`).
- **Layer hardware alternation**: `--layer-hardware-alternate` toggles `instances[i].hardware` after each segment (requires `--forward-segments per_block`). Auto-picks a second GPU from `profiler/perf` or seeds one from v0 profiler data.
- **add_done guard**: mid-forward-pass iterations do not complete the batch if registry misses.

## Tests
```bash
cd LLMServingSim && PYTHONPATH=. python3 -m unittest serving.tests.test_forward_segments -v
```

## Known issue (resolved 2026-06-15)
Partial segment Chakra graphs caused ASTRA SIGSEGV; Python appeared hung in `read_wait`. Fixed with embedding/head bookend stubs. Details: [`docs/forward_segments_debug.md`](docs/forward_segments_debug.md).
