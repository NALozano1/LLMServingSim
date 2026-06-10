# Simulation runtime calibration

Structured wall-clock records for `python -m serving` runs on **nserver15**.
Used to refine the heuristic in `scripts/generate_cluster_config.py` (`_CALIBRATION`).

## Files

| File | Purpose |
|------|---------|
| `runs.jsonl` | One JSON object per line — append-only run log |
| `schema.json` | Field definitions for each record |

## Automatic recording

Every successful `python -m serving` run appends one line to `runs.jsonl` at the end
(see `serving/core/runtime_calibration.py`). You should see:

```text
Recorded runtime calibration to .../calibration/runtimes/runs.jsonl
```

Disable with:

```bash
export LLMSERVINGSIM_RECORD_RUNTIME=0
```

## Manual recording (optional)

For backfilling old runs or fixing a failed auto-record:

```bash
cd /app/LLMServingSim   # repo root
python3 scripts/record_runtime.py \
  --cluster-config 'configs/cluster/generated_8gpu.json' \
  --wall-time '1m 40.547s' \
  --first-log-s 1.0 \
  --instances 8 --gpus 8 --tp-size 1 --pp-size 1 \
  --notes 'backfill'
```

`--wall-time` accepts seconds (`100.5`) or simulator format (`0h 1m 40.547s`).

## Using for heuristics

1. Add every completed run to `runs.jsonl` (manual or `record_runtime.py`).
2. Compare estimate vs actual when regenerating configs.
3. Tune `_CALIBRATION` in `scripts/generate_cluster_config.py` to fit this data.

Optional later: auto-load nearest-neighbor rows from `runs.jsonl` in the generator.
