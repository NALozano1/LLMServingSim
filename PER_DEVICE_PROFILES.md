# Per-device heterogeneous profiles (`tp_hardware`)

Branch: `feat/per-device-profiles`. **Python-only — no ASTRA-Sim / Chakra
changes.** ASTRA still simulates one homogeneous TP group from a single trace;
only the per-layer `comp_time` numbers and the energy accounting change.

## What it does

Within **one instance** running TP=N (and EP for MoE), each device (rank) can use
a **different hardware profile** — e.g. NPU 0 at `V100_1100MHz`, NPU 1 at
`V100_700MHz` (clocks are already distinct profiler tags), or even different GPUs.

- **Latency:** per layer, `comp_time = max` over the ranks' profiles (the
  ALLREDUCE/ALLTOALL barrier waits for the slowest rank). For clock-scaled
  profiles this is *exact* (uniform ordering ⇒ latency == slowest rank).
- **Energy:** summed per device — each rank uses **its own** profile's active and
  idle power for **its own** busy time:
  `Σ_i idle(hw_i)·(T − lat_i) + active(hw_i)·lat_i`. So a fast rank that finishes
  early **idles at its own idle power** while the slow rank gates the barrier.
- **MoE:** EP rank `i` also uses its own **load** (`local_tokens_i`), so expert
  imbalance drives per-device latency+energy instead of just stretching the max.

## Config

```json
{
  "model_name": "...",
  "hardware": "V100",                          // primary / rank-0 profile (unchanged, required)
  "tp_size": 2,
  "tp_hardware": ["V100", "V100_700MHz"],      // optional: one profile per TP rank (len == tp_size; [0] == hardware)
  "tp_hardware_scope": "all",                   // "all" (default) | "moe"
  "npu_mem": {...}, "pd_type": null
}
```
Plus a `power.npu.<profile>` block (idle/active/standby/standby_duration) for
**every** profile named in `tp_hardware`.

- **`tp_hardware_scope: "all"`** — per-device heterogeneity on every layer.
- **`tp_hardware_scope: "moe"`** — non-MoE layers (dense/attention/head/prologue)
  run **homogeneous on the primary profile**; only **MoE layers** diverge
  per-device. Models devices that share a clock for dense/attention work and
  DVFS-diverge per-device only for the expert computation.

Omitting `tp_hardware` (or a uniform list) = the existing homogeneous behaviour,
**byte-identical** to before.

## Expert-load imbalance

Per-device MoE load comes from the gate routing (`route_ep`). `BALANCED` (default)
is symmetric (no imbalance); use `--expert-routing-policy RAND` for data-driven
imbalance. A **trace-driven** expert selector (real selections from traces, RAND
fallback) is specced but **not yet implemented** (v1).

## Status / limitations (v1)

- Verified end-to-end with `[RTXPRO6000, A6000]` Llama-3.1-8B bf16 **tp2**
  (A6000 is a synthetic 1.25× copy — mechanism check, not physical numbers):
  292/292 layers `comp_time == max`; uniform `tp_hardware == pure homogeneous`
  (byte-identical); `scope="moe"` on the dense model collapses to homogeneous.
- **Real DVFS-clock validation pending two-clock MoE tp2 profiles** (Phi-tiny-MoE
  tp2 at e.g. 700 + 1100 MHz — see the ARC spec in `TRACKER.md`). To get those,
  profile each clock homogeneously at tp2; the sim mix-and-matches per device.
- Out of v1: per-profile **standby** power (kept primary-keyed); **EP spanning DP**
  (`local_ep > tp_size`); combining with `--dvfs-layer-schedule` /
  `--layer-hardware-alternate` (guarded as mutually exclusive); the trace-driven
  expert selector; per-NPU-distinct `.et` (rejected — would touch the C++ backend).

## Code

`config_builder.py` (`_resolve_tp_hardware`, per-profile `num_npus`),
`power_model.py` (`add_npu_active_energy_per_rank`),
`trace_generator.py` (`TraceCtx.rank_*`, `_max_rank_latency`, per-rank
`_emit_layer`/`_emit_moe_block`), `__main__.py` (threading + guard).
Tests: `serving/tests/test_{tp_hardware,power_per_rank,per_device_trace}.py`.
Verification configs: `configs/cluster/single_node_tp2_hetero_llama.json`
(runnable now) and `single_node_tp2_hetero.json` (V100, pending tp2 data).
