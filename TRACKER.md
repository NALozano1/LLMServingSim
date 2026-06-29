# Work tracker

Lightweight "where we left off" across workstreams. **Update the relevant
section whenever you switch context** (status, last-touched date, next step).
Keep entries brief — link out to the real context (README/doc/code) instead of
duplicating it.

- **Branch:** `feat/layer-boundary-dvfs-pause`
- **Sim runs in:** Docker `servingsim_docker` (`/app/LLMServingSim`); host repo at `~/DVFS_MoE/LLMServingSim`
- **Quick checks:** `python3 -m pytest serving/tests/ -q` · non-DVFS regression: `docker exec servingsim_docker bash -lc 'cd /app/LLMServingSim && ./scripts/run_regression_baseline.sh'`

Status key: ✅ done · 🟡 in progress · ⛔ blocked · 💤 not started

---

## 1. DVFS simulator correctness fixes — ✅ (minors open)
_Last touched: 2026-06-22_

Audited the uncommitted forward-segment DVFS work and fixed the correctness
findings. Feature + fixes committed & pushed.

- Done: #1 test fixture, #2 P/D+segments hang→guard, #3 stale `.et` cache,
  #4 coarse segment cache key, #6 synthetic-profile refusal (+ `get_config`
  shadowing bug that broke `--layer-hardware-alternate` and the DP path).
- Commits: `cb1f5cb3` `65d9c099` `10db9084` `309613aa` `6fcfce7b` `14a71927` `a65cdc2a`.
- **Open minors (not done):** #5 work-summary logs wrong `batch_id` under
  segments (observability only); #7 document that segmented mode serializes
  batches (no continuous-batch overlap); #8 `serving/__pycache__/` is
  root-owned (blocks host pytest — `chown`).
- Context: `serving/core/{forward_segments,segment_trace_cache,work_state,layer_dvfs_schedule,hardware_aliases}.py`, `serving/__main__.py`. Future work note (P/D + segments) in `WORK_LOG.md`.

**Next:** decide whether to do minors #5/#7/#8 or leave documented.

---

## 2. V100 re-profiling (clock lock) — ⛔ blocked on ARC
_Last touched: 2026-06-22_

The existing `profiler/perf/V100*` tables are **corrupted**: clock lock didn't
hold (700→~1102, 900→~1402, Qwen 1100 MoE throttled to 817).
Qwen 700/900/1100 are mislabeled. Blocks any trustworthy energy/throughput
validation.

- Fix shipped: pre/post-flight clock verification in `profiler/jobs/run_arc_v100_profile.sh`
  (`GPU_FREQ_VERIFY_STRICT`, ±100MHz) + `profiler/jobs/audit_gpu_clocks.py` gate.
- **Needs (on ARC):** re-profile V100 (≥ Qwen 700/900/1100, ideally all clocks)
  with the gate active. Submit via `profiler/jobs/submit_arc_v100_moe_tp1_profiler_dvfs.sh`.
- Audit existing/new captures: `python3 profiler/jobs/audit_gpu_clocks.py profiler/perf/V100_*`.

**Next:** ARC re-profile run; then re-import and re-run sim sweep.

---

## 3. Sim-vs-hardware validation — 🟡 pending data
_Last touched: 2026-06-22_

Scaffolding built; waiting on real ARC bench results (user getting ARC agent to
sync). Comparison is stamped `PROFILES_SUSPECT` until #2 is done.

- **Folder:** `validation/layer_boundary_dvfs/` (README = full setup; `comparison.{md,csv}` = output).
- **Tool:** `python3 scripts/validate_dvfs_vs_hardware.py --bench-runs <synced ARC>/runs`.
- Anchor: real ~600 J (GPU-only, user recollection) vs sim 415 J GPU / 1220 J system / ~3 s, single Qwen prefill.
- **Need from ARC:** per `run_id` (`scat_p0X_qwen_iN`) a `summary.json`/`run_exec_metrics.json` with `exec_sec` + GPU energy (`energy_excl_pause_j` pref).

**Next:** on data arrival — confirm energy basis (GPU-only; exec vs wall), run the tool, review per-permutation error. Real validation only after #2.

---

## Open questions
- Confirm the ~600 J anchor: GPU-only? measured over `exec` or wall (incl. pause)? which schedule?
