# EP-rank MoE profiling

## Problem: TP=1 profiles the wrong kernel

The profiler historically skipped `ExpertCategory` for TP > 1 and profiled the full MoE
kernel only at TP=1 (all `num_experts` on one GPU, full batch). The simulator was expected
to scale this analytically by EP size. This is incorrect for three reasons:

1. **Wrong GEMM shape.** TP=1 runs a 128-expert grouped GEMM; an EP=8 rank runs a
   16-expert grouped GEMM. Different weight matrix sizes → different memory bandwidth
   pressure and tensor-core utilization.

2. **Wrong roofline position.** With 128 experts and a batch of B tokens, each expert gets
   ~B tokens. With 16 experts and ~B/8 tokens routed to this rank, each expert gets ~B/8
   tokens — often in the memory-bandwidth-bound regime (small per-expert batches) rather
   than compute-bound.

3. **Load imbalance is ignored.** Assuming N/8 tokens per rank is wrong. With batch size 1
   and top_k=8 across 128 experts (16 per rank), there is ~35% probability a given rank
   receives 0 tokens, ~41% probability it receives 1, ~21% probability it receives 2.
   Any rank can receive 0..`min(top_k, num_local_experts)` tokens at batch size 1.

## EP dispatch mechanics

In vLLM with `--tensor-parallel-size N` for MoE models, EP=TP: each rank holds
`num_experts / N` local experts.

For a batch step:

1. Each token routes to `top_k` global experts (gate).
2. **All-to-all dispatch**: each `(token, expert)` pair is sent to the rank owning that
   expert.
3. Each rank runs its local `FusedMoE` kernel on the received pairs.
4. **All-to-all collect**: results are gathered back and combined.

After dispatch, rank `i` holds a set of `(token, local_expert)` pairs `P_i`. The key
quantities for rank `i`'s compute time are:

| Symbol | Meaning |
|--------|---------|
| `T_i = \|P_i\|` | total local pairs (= effective token-expert workload) |
| `K_i` | distinct local experts that received ≥1 pair (1 .. `num_local_experts`) |

`T_i` can be 0 (rank receives nothing), which is common at small batch sizes.

### Per-token local_top_k

A single arriving token at rank `i` may have contributed 1 or more pairs:

- `local_top_k(t) = 1` — common case: exactly 1 of the token's `top_k` global choices
  landed on rank `i`.
- `local_top_k(t) = 2` — occasional: 2 of the token's choices are on rank `i` (e.g.,
  with top_k=8 and 16 experts per rank out of 128 total, this happens ~21% of the time
  for a single token).
- `local_top_k(t) ≥ 3` — rare.

The `FusedMoE` grouped GEMM does not care how pairs were assigned to tokens — it only
cares about the total pair count and how they are distributed across local experts. So
`T_i` (total pairs) is the correct workload axis, not the number of arriving tokens.

## Fix: EP-sharded per-rank profile

Profile the per-rank kernel on a single GPU with:

- `num_local_experts = num_experts / EP_SIZE` (e.g., 16 for EP=8 on Qwen3-30B)
- `top_k = 1` — each synthetic token contributes exactly 1 pair

With `top_k=1`, the profiler's `tokens` axis equals `T_i` (total local pairs). The
`activated_experts` axis equals `K_i` (distinct active local experts).

The resulting `moe.csv` at TP=8 answers directly:

> Given `T` token-expert pairs arriving at this rank with `K` distinct local experts
> active, at clock `F`, the rank takes `Y` µs.

This is exactly the input the DVFS policy needs to compute per-rank latency under real
routing distributions.

### Why top_k=1 is the correct framing

Setting `top_k=1` is not a simplifying assumption — it reflects the post-dispatch
reality. The `FusedMoE.forward_native` on rank `i` receives pairs that were already
partitioned by the all-to-all. Whether those pairs came from tokens with
`local_top_k=1` or `local_top_k=2` does not affect the grouped GEMM; only `T_i` and
the per-expert batch sizes matter. Using `top_k=1` in the profiler makes the synthetic
routing match this model exactly.

## Implementation

### `engine.py: fuse_engine_kwargs`

When `tp > 1`, after the `SHARD_FIELDS` loop:

1. Find the expert count field (`num_experts`, `num_local_experts`, or
   `n_routed_experts`) and divide by `tp`.
2. Find the top_k field (`num_experts_per_tok`, `num_experts_per_token`, or `moe_k`)
   and set it to `1`.

Both changes go into `sharded_overrides` which is merged into `hf_overrides` before
vLLM boots.

### `categories.py: categories_for`

`ExpertCategory` now runs at all TP degrees:

- `tp=1` — 128-expert full-batch profile (backwards-compatible baseline).
- `tp>1` — EP-sharded profile: `num_experts/tp` local experts, `top_k=1`.

### `run_moe_slice.sh`

`--tp` is passed as `1,${TP}` (not just `${TP}`) because `_parse_tp` requires 1 to be
present for `replicate_tp_stable`.

## Simulation usage

Look up `moe.csv` at the TP level matching the deployment EP size. For a step where
rank `i` receives `T_i` total pairs with `K_i` distinct active experts:

```
latency_i = moe_csv[T_i][K_i]
```

The DVFS opportunity for rank `i` is the gap between `latency_i` at full clock and the
critical-path rank's latency. A rank with `T_i = 0` takes 0 µs for MoE compute and can
idle at the lowest available clock.

## Limitations and future work

- **All-to-all latency** is not measured here; it is handled analytically by ASTRA-Sim.
- **Non-uniform EP** (num_experts not divisible by EP_SIZE) is not supported; a warning
  is emitted and MoE EP sharding is skipped.
- **Grid sparsity**: the `(T, K)` space may have gaps (e.g., `T=3, K=2` is not a
  power-of-two grid point). If the simulation needs off-grid points, linear or log-linear
  interpolation over the `T` axis at fixed `K` is likely valid in the memory-bandwidth-
  bound regime, but should be validated empirically before use.
- **top_k=2 verification**: the `top_k=1` profile may slightly underestimate latency when
  `local_top_k=2` pairs are common (pairs for the same token going to two local experts
  may share cache lines). Profiling a second pass at `top_k=2` and comparing would
  quantify the error; initial expectation is it is small.
