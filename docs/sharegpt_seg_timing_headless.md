# Headless ShareGPT seg timing study

**Started:** 2026-06-16 (running in `servingsim_docker`)

## What it runs

```bash
python3 scripts/run_mono_vs_seg_timing.py \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --num-reqs 10 20 100 \
  --modes seg \
  --out-dir outputs/branch_compare/timing_study_sharegpt
```

Mono for N=10/20/100 is already done. This job fills in **segmented** wall times.

## Monitor

```bash
# Is it still running?
docker exec servingsim_docker pgrep -af run_sharegpt_seg_timing

# Live log
docker exec servingsim_docker tail -f \
  outputs/branch_compare/timing_study_sharegpt/logs/seg_timing_*.log

# When done: summary table
docker exec servingsim_docker cat \
  outputs/branch_compare/timing_study_sharegpt/summary.csv
```

## Expected runtime

Rough order of magnitude: **tens of minutes to a few hours** (seg N=10 alone may be ~25–30 min at ~22× mono; N=100 may plateau or grow depending on trace reuse).

## Artifacts

| Path | Contents |
|------|----------|
| `logs/seg_timing_*.log` | Full study stdout |
| `logs/seg_timing.pid` | Background PID inside container |
| `seg_{10,20,100}req.log` | Per-run simulator output |
| `summary.csv` | Final wall-time table (overwritten each script run; merge mono rows manually if needed) |

Launcher: `scripts/run_sharegpt_seg_timing_headless.sh`
