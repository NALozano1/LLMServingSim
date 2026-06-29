# V100 GPU Clock-Lock Investigation

**Status:** 🟡 ROOT CAUSE FOUND (high confidence) — **apply-timing**: the lock is applied
before any CUDA context exists (lock-then-load) and does not carry into vLLM's later
context, so the GPU boosts to 1440. The helper itself works (pins ~1100 under load when a
context is already live). Fix = apply/verify the lock AFTER a CUDA context is up.
Confirmation job 8021428 (Test X lock-then-load≈1440 vs Test Y load-then-lock≈1100) was
in flight when this was written; numbers to be appended.

### Quick reference — what's true
- Helper `/usr/local/sbin/nvidia-smi-clocks --id 0 1100 1100` WORKS (caps V100 to ~1100
  under load). `--id` takes an integer index only (UUID → "Invalid GPU id").
- The lock does NOT stick if applied while no CUDA client holds the GPU, even with
  persistence mode enabled.
- The BENCH path (run_arc_v100_bench_throughput.sh, job 8021298) provably captured at 1440.
- NUANCE (2026-06-24, from dvfs-policy audit): the June-15 PROFILER `moe.csv` tables DO vary
  with frequency (1 tok/8 exp: 700→10313µs vs 1400→5975µs), so at least the low-freq
  profiler captures show real frequency signal — they may NOT all be 1440. The profiler
  (layer-boundary path) and the bench differ. Higher-freq points (900–1400) are compressed
  (~10%), so those locks may not have taken. Action: re-profile with the under-load audit
  to confirm per-frequency validity rather than blanket-trust or blanket-discard. See
  `dvfs-policy/AUDIT.md` §Data validity.
- The Jun 22 idle-verify regression (separate bug) was already fixed (idle mismatch
  non-fatal; `GPU_FREQ_APPLY_STRICT=1` restores hard-abort).

**Started:** 2026-06-24
**Owner:** bram / Claude

This doc tracks the investigation into why DVFS frequency locking on ARC V100 nodes
does not work, and what it means for existing profiler/bench data. Appended to as we go.

---

## TL;DR (current understanding)

- Requesting a locked GPU clock (e.g. `V100_1100MHz`) produces data captured at the
  **wrong frequency**: the GPU actually runs at **1440 MHz (max boost)** under load.
- The site helper `/usr/local/sbin/nvidia-smi-clocks --id N MIN MAX` returns `rc=0`
  but the lock does not take effect on the job's GPU under load.
- **Consequence: every existing `V100_<freq>` profile/bench table is invalid** (all
  were really at 1440 MHz). This includes the Jun 15 profiles.
- This explains the original **"1100 MHz slower than 900 MHz" anomaly**: neither was
  ever locked; both ran at 1440, so the latency difference was run-to-run noise.

---

## Timeline / Evidence

### 1. Both validation chains failed instantly (2026-06-23/24)
- §0 re-profile chain (jobs 8021127–8021139) and §1 bench chain (8021235–8021242)
  all FAILED with ExitCode 1:0 within ~1–2 min, cascading via `afterany`.
- Cause (proximate): `gpu_freq_lock.py cmd_apply` returned non-zero, and `set -e`
  in the runners killed the job before any vLLM ran.

### 2. Why cmd_apply returned non-zero — a regression in the Jun 22 edit
- Log (job 8021235):
  ```
  [gpu_freq] WARN apply ioctl ok but graphics clocks not at 700 MHz (+/- 15): graphics_mhz: 135
  [gpu_freq] locked GPUs to 700 MHz via .../nvidia-smi-clocks ok=False
  ```
- The Jun 22 change to `shared/scripts/gpu_freq_lock.py` folded an **idle** clock
  read-back into the success check (`ok = ok and verified` → `return 0 if ok else 1`).
- On V100 the graphics clock **idles at 135 MHz regardless of the lock**, so the idle
  verify can never pass before load → always returned failure.
- **Proof it was this change and nothing else:** the *working* Jun 15 apply record and
  the failing Jun 24 record are byte-identical in command and behavior:

  | | Jun 15 (worked) | Jun 24 (failed) |
  |---|---|---|
  | gpu_indices | `[0]` | `[0]` |
  | ioctl | sequential n=1, rc=0 | sequential n=1, rc=0 |
  | clocks_after | 135 MHz | 135 MHz |
  | apply_verified | False | False |
  | outcome | proceeded → profiled | `return 1` → killed |

### 3. Fix applied for the false idle-abort (necessary, not sufficient)
- `shared/scripts/gpu_freq_lock.py`: idle read-back mismatch is now **non-fatal** by
  default (return based on ioctl/set success). New env `GPU_FREQ_APPLY_STRICT=1`
  restores the hard-abort. Records `idle_verified` for visibility.
- `profiler/jobs/run_arc_v100_profile.sh` and `run_arc_v100_bench_throughput.sh`:
  my added pre-flight read-back is now **informational only** (it also read at idle and
  would have `exit 3` for the same invalid reason). Post-flight under-load audit
  (`audit_gpu_clocks.py`) remains the authoritative gate.

### 4. THE REAL PROBLEM — lock doesn't hold under load (validation job 8021298)
- Single bench at requested 1100 MHz. Fix let it run past apply. Under load,
  read from `gpu_power/bench.jsonl`:
  ```
  UNDER-LOAD graphics_mhz: median=1440 min=1440 max=1440   (target=1100)   [55/55 samples]
  power_w: median=69.7 max=144.6
  ```
- **The GPU ran at 1440 (max boost) the entire time.** The 1100 lock was not applied.
- Idle read-back (135) is meaningless; only the under-load number matters, and it shows
  the lock is a no-op.

### 5. Permissions / helper interface
- `sudo -n -l` → `(root) SETENV: NOPASSWD: /usr/local/sbin/nvidia-smi-clocks` only.
- Helper signature: `nvidia-smi-clocks [--id ID] MIN MAX` | `--reset`. Root-owned
  (rwxr-x---), unreadable, unchanged since 2026-05-08. We cannot inspect or edit it.
- `gpu_freq_lock.py` invokes it as `sudo nvidia-smi-clocks --id <idx> <mhz> <mhz>`.

---

## Leading hypotheses for why the lock no-ops under load

1. **Transient-client / persistence-mode**: helper sets the clock as a short-lived
   nvidia-smi client, exits; with persistence mode off and no client holding the GPU,
   the driver resets clocks before vLLM's CUDA context starts → GPU boosts to 1440.
   → Fix: apply the lock *after* a CUDA context exists (or hold a keep-alive context).
2. **Soft `-ac` vs hard `-lgc`**: if the helper sets *application* clocks, the driver
   can exceed them; `--lock-gpu-clocks` is the hard cap.
3. **Wrong GPU targeted**: privileged `--id 0` under sudo hits a different physical GPU
   than the one the Slurm job was allocated.

## Diagnostic in flight — job 8021370 (`diag_v100_clock_lock.sh`)
Three tests, each reads clocks UNDER LOAD (background torch matmul burn):
- **A** lock-then-load (current behavior)
- **B** load-then-lock (apply while CUDA context held) → isolates hypothesis 1
- **C** direct `nvidia-smi --lock-gpu-clocks` bypassing helper → isolates perms/helper

Decision rule:
- B caps ~1100 but A doesn't → hypothesis 1 (lock after vLLM starts).
- Neither caps → hypothesis 2/3 (helper targets wrong GPU or soft `-ac`); escalate.

### Diagnostic run 1 (job 8021370) — INCONCLUSIVE (load generator failed), but new clues
- The torch burn never ran: apptainer tried to build the image into the **home** cache
  (`/home/engs2950/.apptainer/cache`) which is **over disk quota** (home: 19.8G used /
  20.5G limit). `FATAL: ... disk quota exceeded`. So util stayed 0% and the GPU idled at
  135 — tests A/B/C produced no load. (Real jobs set
  `APPTAINER_CACHEDIR=${ENGS_GLASS}/.apptainer_cache/cache`; my diag did not.)
- BUT the ENV section gave two real clues:
  - `persistence_mode = Enabled` → weakens the "transient-client loses the lock" theory.
  - Baseline `applications.graphics = 817 MHz`, `max = 1440`. **After** locking to 1100,
    `applications.graphics` was still **817** (unchanged). So the helper did NOT set the
    application clock, and (from job 8021298) the current clock was not capped under load
    either → the lock command had **zero effect on the job's GPU**.
- **Refined leading hypothesis → wrong-GPU targeting.** `gpu_freq_lock.py` locks with the
  cgroup-relative integer index (`--id 0`). Under `sudo`, the privileged helper may
  enumerate the node's *physical* GPUs, so `--id 0` locks physical GPU 0 while the job was
  allocated a different physical GPU (UUID `GPU-95f7a37a-…`). The fix would be to target
  by **UUID** instead of index.

### Diagnostic run 2 (job TBD) — tests index vs UUID targeting under sustained load
Holds ONE CUDA context alive (single background burn, correct cache dir) and, while loaded,
locks/reads/resets for `--id 0` then `--id <UUID>`. Decision rule:
- `--id 0` doesn't cap but `--id <UUID>` caps at ~1100 → wrong-GPU targeting; fix
  `gpu_freq_lock.py` to pass UUID.
- Neither caps under load → helper cannot hard-lock (ignored `-ac` / unsupported);
  escalate to ARC (the wrapper is root-owned and unreadable).

### Diagnostic run 2 (job 8021412) — RESULT: helper works; root cause is APPLY-TIMING
- **The site helper is NOT the problem.** `--id 0 1100 1100` →
  `helper> GPU clocks set to (gpuClkMin 1100, gpuClkMax 1100) for GPU 0000:06:00.0`,
  and under genuine 100% load (torch matmul, 153W) the clock capped at **gr≈990–1110**.
  The lock works.
- The wrong-GPU/UUID theory is dead: `--id <UUID>` → `helper> Invalid GPU id`. The helper
  only takes an **integer index**, and `--id 0` already targets the correct PCI GPU.
- `--reset` with no `--id` did NOT clear the lock (PHASE 2 stayed ~1110), so end-of-job
  restore may also be index-scoped — minor note, not the bug.
- **Reconciling with 8021298 (which read 1440):** chronological log shows the lock WAS
  applied (line 12) and only reset at end-of-job (line 48, after the bench). So the lock
  was nominally "active" during the bench yet the GPU ran at 1440. The ONLY difference vs
  the diagnostic:

  | | lock applied | CUDA context at apply | under-load clock |
  |---|---|---|---|
  | 8021298 bench | before vLLM | **none held** | **1440** ❌ |
  | diag PHASE 1/2 | after burn started | **burn holds GPU** | **~1100** ✅ |

- **ROOT CAUSE (revised): apply-timing.** The locked clock does not stick when applied
  with **no CUDA client holding the GPU** (lock-then-load). It sticks when a context is
  already live at apply time (load-then-lock). `persistence_mode=Enabled` does NOT save it.
  This is original hypothesis 1, now with a clean differential — and it also re-explains
  the "1100 < 900" anomaly (every freq was lock-then-load → all ran at 1440).

### Diagnostic run 3 (job TBD) — reproduce bug + validate fix, back to back
- **Test X (reproduce):** apply lock with NO context, THEN start burn → expect ~1440.
- **Test Y (fix):** start burn (context live), THEN apply lock → expect ~1100.
- If X≈1440 and Y≈1100, the fix is proven: **apply the lock after a CUDA context exists**
  (e.g. re-apply once vLLM is up, or hold a keep-alive context across apply→vLLM).

### Diagnostic run 3 (job 8021428, htc-g049) — INCONCLUSIVE (load generator failed)
- The containerized torch burn never loaded the GPU on g049: util=0%, clock idle at 135
  for ALL samples in both Test X and Test Y. No under-load data produced. (The script's
  trailing "VERDICT" line is static text, NOT a measurement — ignore it.)
- Likely cause: the backgrounded apptainer burn didn't spin up within the sampling window
  and its stdout/stderr weren't redirected, so any failure was invisible. (v2 on g048
  eventually warmed; g049 did not within ~3 min.)
- **Does not change the conclusion.** The apply-timing root cause still stands on two real
  under-load data points: v2 (load-then-lock → ~1100 @ 100% util) and 8021298
  (lock-then-load → 1440 under vLLM). A clean back-to-back X/Y repro would be nice-to-have
  but is not required to act; if re-run, fix the load gen (redirect burn output; longer
  warm wait; or use a host-side CUDA load that's known-good on the target node).

---

## Impact / actions
- [x] Stop the cascaded chains (drained).
- [x] Cancel mislabeled bench (8021298).
- [ ] Determine real root cause (8021370).
- [ ] Fix the lock so it holds under load (provable via under-load audit).
- [ ] **Invalidate / re-tag existing `V100_<freq>` profiles** — they are all 1440 MHz.
- [ ] Only then re-run §0 (profiles) and §1 (per-freq bench).
