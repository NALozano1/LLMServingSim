# V100 bench layer-boundary DVFS campaign (prefill-only)

Real vLLM `python -m bench run` with in-place layer-boundary pause + DVFS.

See **`REPLICATION.md`** for exact permutations and LLMServingSim replay steps.

## Per-run outputs (`runs/<run_id>/`)

- `run_spec.json` / `sim_replication.json` — configuration for simulation replay
- `bench/` — `run_exec_metrics.json`, `dvfs_markers.jsonl`, `gpu_power/`
- `results/summary.json` — wall/exec/pause/energy/throughput

## Collect

```bash
python3 bench/jobs/collect_bench_layer_campaign_results.py <campaign_dir>
```
