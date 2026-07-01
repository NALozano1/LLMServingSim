"""Top-level orchestration.

Entry points:
    run_full(arch_path, args, out_root)
    run_slice(arch_path, args, tp, group, out_root)

Both are called from ``__main__.py``. They differ only in which
categories and TPs are iterated; everything else — engine spin-up,
catalog slicing, shot firing, sink coalescing, tp_stable
replication — is shared.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from profiler.core import logger as log
from profiler.core.categories import (
    CATEGORY_BY_NAME,
    Category,
    categories_for,
)
from profiler.core.config import Architecture, ProfileArgs, load_architecture
from profiler.core.dvfs_barrier import (
    DvfsBarrierPoller,
    dvfs_host_poller_enabled,
    dvfs_layer_pause_enabled,
    gpu_freq_settle_sec,
    make_shot_barrier_dir,
    parse_freq_schedule,
    restore_gpu_freq,
)
from profiler.core.engine import probe_limits, spin_down, spin_up
from profiler.core.exec_metrics import (
    append_shot_exec_metrics,
    build_shot_exec_record,
    write_run_exec_metrics,
)
from profiler.core.capture import (
    append_capture_records,
    build_capture_records,
    target_mhz_from_hw_tag,
    write_audit_json,
    write_preliminary_audit_json,
)
from profiler.core.gpu_power import (
    GpuPowerSampler,
    compute_achieved_mhz,
    load_power_samples,
    profiler_gpu_power_enabled,
)
from profiler.core.hooks.timings import TimingSample
from profiler.core.writer import (
    persist_meta,
    replicate_tp_stable,
    sink_for,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _variant_root(out_root: Path, args: ProfileArgs) -> Path:
    """Build ``<out_root>/<hardware>/<model_path>/<variant>/``.

    Model path preserves the HuggingFace ``org/model`` layout so the
    simulator's loader (which already expects this shape) doesn't need
    to change. Local paths are normalized to their directory name.
    """
    # If `args.model` is a local path (contains "/" and exists on disk),
    # use its final component as the output subfolder; otherwise treat
    # as HF id verbatim.
    model_as_path = Path(args.model)
    if model_as_path.exists() and model_as_path.is_dir():
        model_subpath = model_as_path.name
    else:
        model_subpath = args.model
    return out_root / args.hardware / model_subpath / args.effective_variant


# ---------------------------------------------------------------------------
# DVFS layer pause helpers
# ---------------------------------------------------------------------------

def _shot_key_str(category: Category, shot) -> str:
    return f"{category.name}_{'_'.join(str(x) for x in category.shot_key(shot))}"


def _profiler_max_shots() -> int | None:
    raw = os.environ.get("PROFILER_MAX_SHOTS", "").strip()
    if not raw:
        raw = os.environ.get("DVFS_LAYER_PAUSE_MAX_SHOTS", "").strip()
    if not raw:
        return None
    return max(1, int(raw))


def _append_shot_timing(out_dir: Path, record: dict[str, Any]) -> None:
    path = out_dir / "shot_timings.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _read_fire_timing(timing_path: Path) -> dict[str, Any]:
    if not timing_path.is_file():
        return {}
    return json.loads(timing_path.read_text(encoding="utf-8"))


def _sum_marker_pause_sec(markers_path: Path) -> tuple[float, int]:
    if not markers_path.is_file():
        return 0.0, 0
    total = 0.0
    count = 0
    for line in markers_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += float(rec.get("pause_end", 0)) - float(rec.get("pause_start", 0))
        count += 1
    return round(total, 6), count


def _fire_single_shot(
    llm,
    category: Category,
    arch: Architecture,
    args: ProfileArgs,
    limits,
    tp: int,
    out_dir: Path,
    shot,
    *,
    dvfs_pause: bool,
    arm_label: str | None = None,
    sink=None,
) -> dict[str, Any]:
    """Fire one shot after engine boot; return timing record."""
    catalog_slice = category.catalog_slice(arch)
    shot_key = _shot_key_str(category, shot)
    timing_dir = out_dir / "shot_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{arm_label}" if arm_label else ""
    timing_path = timing_dir / f"{shot_key}{suffix}.json"

    barrier_dir: Path | None = None
    poller: DvfsBarrierPoller | None = None
    freq_meta_dir = out_dir / "gpu_freq"
    markers_path = out_dir / "dvfs_markers.jsonl"

    fire_args: tuple[Any, ...] = (
        shot.as_dict(),
        catalog_slice,
        category.name,
        args.measurement_iterations,
        None,
        str(timing_path),
    )

    if dvfs_pause:
        barrier_dir = make_shot_barrier_dir(out_dir, shot_key)
        if not dvfs_host_poller_enabled():
            poller = DvfsBarrierPoller(
                barrier_dir=barrier_dir,
                markers_path=markers_path,
                freq_meta_dir=freq_meta_dir,
                freq_schedule=parse_freq_schedule(),
                settle_sec=gpu_freq_settle_sec(),
            )
            poller.start()
        fire_args = (
            shot.as_dict(),
            catalog_slice,
            category.name,
            args.measurement_iterations,
            str(barrier_dir),
            str(timing_path),
        )

    power_path = out_dir / "gpu_power" / f"{shot_key}.jsonl"
    power_sampler: GpuPowerSampler | None = None
    if profiler_gpu_power_enabled():
        power_path.parent.mkdir(parents=True, exist_ok=True)
        power_path.unlink(missing_ok=True)
        power_sampler = GpuPowerSampler(power_path)
        power_sampler.start()

    rpc_t0 = time.perf_counter()
    try:
        raw = llm.collective_rpc("fire", args=fire_args)
    finally:
        if power_sampler is not None:
            power_sampler.stop()
        if poller is not None:
            poller.stop()
            for err in poller.errors:
                log.warning("DVFS poller: %s", err)

    rpc_wall_sec = time.perf_counter() - rpc_t0
    fire_timing = _read_fire_timing(timing_path)

    exec_record = build_shot_exec_record(
        shot_key,
        fire_timing,
        markers_path,
        power_path if profiler_gpu_power_enabled() else None,
        extra={
            "arm": arm_label,
            "dvfs_pause": dvfs_pause,
            "rpc_wall_sec": round(rpc_wall_sec, 6),
        },
    )
    append_shot_exec_metrics(out_dir, exec_record)
    if dvfs_pause:
        log.info(
            "%s: exec measured=%.3fs (excl pause %.3fs) energy=%.1fJ (excl pause %.1fJ)",
            shot_key,
            exec_record.get("measured_exec_sec", 0),
            exec_record.get("barrier_wait_sec") or exec_record.get("marker_pause_sec", 0),
            exec_record.get("energy_j") or 0,
            exec_record.get("energy_excl_pause_j") or 0,
        )

    # Compute achieved_mhz from the shot's power samples so it can be
    # coalesced into the per-category CSV and embedded in the CaptureRecord.
    achieved_mhz_val: float | None = None
    if profiler_gpu_power_enabled() and power_path.is_file():
        _ps = load_power_samples(power_path)
        achieved_mhz_val = compute_achieved_mhz(_ps)
    exec_record["achieved_mhz"] = achieved_mhz_val

    timings_dicts = raw[0]
    timings = [
        TimingSample(
            layer=d["layer"],
            microseconds=float(d["microseconds"]),
            gating_ms=d.get("gating_ms"),
            expert_ms=d.get("expert_ms"),
        )
        for d in timings_dicts
    ]
    achieved_extra = {"achieved_mhz": achieved_mhz_val} if achieved_mhz_val is not None else {}
    if sink is not None:
        for point in category.extract_points(shot, timings, arch, tp):
            sink.coalesce(point, extra_values=achieved_extra or None)

    # Emit one CaptureRecord per timing sample — latency and power bound together.
    capture_recs = build_capture_records(
        shot_key=shot_key,
        category_name=category.name,
        shot=shot,
        timings=timings,
        arch=arch,
        tp=tp,
        args=args,
        exec_record=exec_record,
        power_path=power_path if profiler_gpu_power_enabled() else None,
    )
    if capture_recs:
        append_capture_records(out_dir, capture_recs)

    record: dict[str, Any] = {
        "shot_key": shot_key,
        "arm": arm_label,
        "dvfs_pause": dvfs_pause,
        "rpc_wall_sec": round(rpc_wall_sec, 6),
        **fire_timing,
        "measured_exec_sec": exec_record.get("measured_exec_sec"),
        "energy_j": exec_record.get("energy_j"),
        "energy_excl_pause_j": exec_record.get("energy_excl_pause_j"),
        "achieved_mhz": achieved_mhz_val,
    }
    _append_shot_timing(out_dir, record)
    return record


# ---------------------------------------------------------------------------
# Shot firing + point ingestion (shared between run_full and run_slice)
# ---------------------------------------------------------------------------

def _fire_one_category(
    llm,
    category: Category,
    arch: Architecture,
    args: ProfileArgs,
    limits,
    tp: int,
    out_dir: Path,
) -> None:
    """Sweep all of this category's shots, write the resulting CSV.

    Resume behaviour: unless ``args.force`` is set, an existing CSV is
    preloaded into the sink and shots whose key is already covered are
    skipped. The sink's flush at the end writes both preserved and
    newly-measured rows. ``--force`` restores wipe-and-rewrite.
    """
    sink = sink_for(category, out_dir)

    prior_keys: set[tuple] = set()
    if not args.force:
        preloaded = sink.preload()
        if preloaded:
            prior_keys = sink.prior_shot_keys()
            log.info(
                "%s resume: %d prior rows preloaded, "
                "%d prior shot keys recognized",
                category.label, preloaded, len(prior_keys),
            )

    # Materialize shots up-front so the progress bar has a total.
    # Grids are small enough (<a few thousand shots) that holding them
    # in memory is fine.
    all_shots = list(category.compose_shots(arch, args, limits, tp))
    if not all_shots:
        log.warning(
            "category %s produced no shots for tp=%d; skipping",
            category.label, tp,
        )
        return

    if prior_keys:
        shots = [s for s in all_shots if category.shot_key(s) not in prior_keys]
        skipped = len(all_shots) - len(shots)
        if skipped:
            log.info(
                "%s: skipping %d already-measured shots, firing %d new",
                category.label, skipped, len(shots),
            )
    else:
        shots = all_shots

    if not shots:
        # Nothing to fire; still flush so the CSV is rewritten with
        # preloaded rows (a no-op schema repair if anything changed).
        sink.flush()
        log.info("%s: nothing to do (all shots already measured)", category.label)
        return

    max_shots = _profiler_max_shots()
    if max_shots is not None:
        shots = shots[:max_shots]
        log.info(
            "%s: PROFILER_MAX_SHOTS=%d (firing %d shot(s))",
            category.label,
            max_shots,
            len(shots),
        )

    # Write a preliminary audit.json BEFORE the shot loop so that a crash
    # mid-category leaves verdict_ok=False rather than a stale prior pass.
    # write_audit_json at the end of _fire_one_category overwrites this on
    # clean completion.
    write_preliminary_audit_json(out_dir)

    dvfs_pause = dvfs_layer_pause_enabled()
    freq_meta_dir = out_dir / "gpu_freq"
    markers_path = out_dir / "dvfs_markers.jsonl"
    if dvfs_pause:
        log.info(
            "%s: DVFS layer pause enabled (schedule=%s settle=%.2fs host_poller=%s)",
            category.label,
            parse_freq_schedule(),
            gpu_freq_settle_sec(),
            dvfs_host_poller_enabled(),
        )

    label = f"TP={tp}  {category.label}"
    with log.progress(label, total=len(shots)) as bar:
        for shot in shots:
            _fire_single_shot(
                llm,
                category,
                arch,
                args,
                limits,
                tp,
                out_dir,
                shot,
                dvfs_pause=dvfs_pause,
                sink=sink,
            )
            if sink is not None:
                sink.flush()
            bar.advance(1)

    if dvfs_pause and not dvfs_host_poller_enabled():
        restore_result = restore_gpu_freq(freq_meta_dir)
        if not restore_result.get("ok"):
            log.warning("DVFS restore failed: %s", restore_result)

    run_metrics = write_run_exec_metrics(
        out_dir,
        extra={"category": category.name, "tp": tp},
        model_config=args.model_config,
        architecture=arch.model_dump(),
        tp=tp,
    )
    if run_metrics:
        log.success(
            "%s run totals: effective_runtime=%.3fs pause_overhead=%.3fs "
            "effective_energy=%.1fJ (%d shots) → %s",
            category.label,
            run_metrics.get("effective_runtime_sec", 0),
            run_metrics.get("pause_overhead_sec", 0),
            run_metrics.get("effective_energy_j", 0),
            run_metrics.get("shot_count", 0),
            out_dir / "run_exec_metrics.json",
        )
        if run_metrics.get("ttft_ms") is not None:
            log.success(
                "%s reference latency (effective/kernel): TTFT=%.1fms TPOT=%.1fms "
                "e2e=%.1fms (in=%s out=%s)",
                category.label,
                run_metrics.get("ttft_ms", 0),
                run_metrics.get("tpot_ms", 0),
                run_metrics.get("e2e_latency_ms", 0),
                (run_metrics.get("serving_latency") or {})
                .get("reference_workload", {})
                .get("input_tokens"),
                (run_metrics.get("serving_latency") or {})
                .get("reference_workload", {})
                .get("output_tokens"),
            )

    # Write run-level clock audit sidecar from captures accumulated above.
    _target_mhz = target_mhz_from_hw_tag(args.hardware)
    audit_result = write_audit_json(out_dir, target_mhz=_target_mhz)
    if not audit_result.get("verdict_ok"):
        log.warning(
            "%s audit FAILED: %s",
            category.label,
            audit_result.get("verdict_reason"),
        )
    else:
        log.info(
            "%s audit ok: %s",
            category.label,
            audit_result.get("verdict_reason"),
        )

    sink.flush()
    log.success("%s → %s", category.label, sink.path)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_full(
    arch_path: Path,
    args: ProfileArgs,
    out_root: Path,
) -> None:
    """Profile every (tp, category) pair for this architecture × model."""
    arch = load_architecture(arch_path)
    variant_root = _variant_root(out_root, args)

    log.banner(args, variant_root)

    last_engine_kwargs: dict[str, Any] | None = None

    for tp in args.tp_degrees:
        # Skip TPs with nothing non-tp_stable to do. The post-pass
        # replicate_tp_stable will populate their CSVs from tp1.
        if not arch.has_tp_dependent_work(tp):
            log.info("TP=%d has only tp_stable work; deferring to replication",
                     tp)
            continue

        with log.stage(f"TP={tp}  booting vLLM engine"):
            llm, engine_kwargs, tmpdir = spin_up(args, tp)
            last_engine_kwargs = engine_kwargs
            limits = probe_limits(llm)

        # Visibility: what the live engine actually allocated for
        # this (tp, 1-layer-shrunk) configuration. Drives every
        # feasibility filter downstream.
        log.info(
            "TP=%d limits: num_cache_tokens=%d max_model_len=%d "
            "max_num_batched_tokens=%d max_num_seqs=%d%s",
            tp,
            limits.num_cache_tokens,
            limits.max_model_len,
            limits.max_num_batched_tokens,
            limits.max_num_seqs,
            (f" num_experts={limits.num_experts} top_k={limits.top_k}"
             if limits.num_experts else ""),
        )

        tp_root = variant_root / f"tp{tp}"
        tp_root.mkdir(parents=True, exist_ok=True)

        try:
            if not args.only_skew:
                for category in categories_for(arch, tp):
                    if args.only_moe and category.name != "moe":
                        log.info(
                            "only_moe mode: skipping %s category",
                            category.name,
                        )
                        continue
                    _fire_one_category(
                        llm, category, arch, args, limits, tp, tp_root,
                    )
            else:
                log.info("only_skew mode: skipping dense / per_seq / "
                         "attention / moe categories")
            # Skew measurement after all categories — uses the same
            # attention kernel slice but fires shots with non-uniform
            # decode kv distributions. Writes tp_root/skew.csv.
            if not args.skip_skew:
                from profiler.core.skew import sample_skew
                sample_skew(llm, arch, args, limits, tp, tp_root)
        finally:
            spin_down(llm, tmpdir)

    # After every tp has run, copy tp_stable rows from tp1 into the rest.
    # Skip when only_skew=True (nothing new to replicate).
    # When tp1 was NOT profiled in this session, reuse the on-disk tp1/
    # produced by a prior/sibling run and raise loudly if it is absent
    # or incomplete (anti-taint invariant).
    if not args.only_skew:
        with log.stage("replicating tp_stable layers across TP folders"):
            replicate_tp_stable(
                variant_root, arch, args.tp_degrees,
                require_tp1=1 not in args.tp_degrees,
            )

    if last_engine_kwargs is None:
        last_engine_kwargs = {}
    persist_meta(args, arch_path, last_engine_kwargs, variant_root)

    log.done(variant_root)


# ---------------------------------------------------------------------------
# Slice refresh
# ---------------------------------------------------------------------------

def run_slice(
    arch_path: Path,
    args: ProfileArgs,
    tp: int,
    group: str,
    out_root: Path,
) -> None:
    """Re-profile one (tp, category) pair without redoing everything."""
    arch = load_architecture(arch_path)
    variant_root = _variant_root(out_root, args)

    if group not in CATEGORY_BY_NAME:
        raise ValueError(
            f"unknown group {group!r}; must be one of "
            f"{sorted(CATEGORY_BY_NAME)}"
        )
    if tp not in args.tp_degrees:
        raise ValueError(
            f"tp={tp} is not in the session's tp_degrees ({args.tp_degrees})"
        )

    category_cls = CATEGORY_BY_NAME[group]
    category = category_cls()

    if not category.catalog_slice(arch):
        raise ValueError(
            f"architecture has no entries in catalog.{group}; "
            f"nothing to profile"
        )

    log.banner(args, variant_root)
    log.info("Slice refresh: tp=%d group=%s", tp, group)

    with log.stage(f"TP={tp}  booting vLLM engine"):
        llm, engine_kwargs, tmpdir = spin_up(args, tp)
        limits = probe_limits(llm)

    tp_root = variant_root / f"tp{tp}"
    tp_root.mkdir(parents=True, exist_ok=True)

    try:
        _fire_one_category(
            llm, category, arch, args, limits, tp, tp_root,
        )
    finally:
        spin_down(llm, tmpdir)

    # A slice refresh at tp=1 may invalidate prior replication; redo it.
    if tp == 1:
        with log.stage("replicating tp_stable layers"):
            replicate_tp_stable(variant_root, arch, args.tp_degrees)

    persist_meta(args, arch_path, engine_kwargs, variant_root)

    log.done(variant_root)


# ---------------------------------------------------------------------------
# Pause A/B (single engine boot, post-warmup shot timing)
# ---------------------------------------------------------------------------

def run_pause_ab(
    arch_path: Path,
    args: ProfileArgs,
    tp: int,
    group: str,
    out_root: Path,
) -> None:
    """One engine boot, then nopause vs pause shots on the same warm GPU."""
    arch = load_architecture(arch_path)
    variant_root = _variant_root(out_root, args)

    if group not in CATEGORY_BY_NAME:
        raise ValueError(
            f"unknown group {group!r}; must be one of "
            f"{sorted(CATEGORY_BY_NAME)}"
        )
    if tp not in args.tp_degrees:
        raise ValueError(
            f"tp={tp} is not in the session's tp_degrees ({args.tp_degrees})"
        )

    category_cls = CATEGORY_BY_NAME[group]
    category = category_cls()

    if not category.catalog_slice(arch):
        raise ValueError(
            f"architecture has no entries in catalog.{group}; "
            f"nothing to profile"
        )

    log.banner(args, variant_root)
    log.info("Pause A/B: tp=%d group=%s (single engine)", tp, group)

    boot_t0 = time.perf_counter()
    with log.stage(f"TP={tp}  booting vLLM engine"):
        llm, engine_kwargs, tmpdir = spin_up(args, tp)
        limits = probe_limits(llm)
    engine_boot_sec = round(time.perf_counter() - boot_t0, 3)

    tp_root = variant_root / f"tp{tp}"
    tp_root.mkdir(parents=True, exist_ok=True)

    all_shots = list(category.compose_shots(arch, args, limits, tp))
    if not all_shots:
        raise RuntimeError(f"category {category.label} produced no shots")

    max_shots = _profiler_max_shots()
    shot = all_shots[0] if max_shots is None else all_shots[:max_shots][0]
    shot_key = _shot_key_str(category, shot)
    log.info(
        "Pause A/B shot: %s (engine_boot_sec=%.3f)",
        shot_key,
        engine_boot_sec,
    )

    markers_path = tp_root / "dvfs_markers.jsonl"
    markers_path.unlink(missing_ok=True)
    shot_timings_path = tp_root / "shot_timings.jsonl"
    shot_timings_path.unlink(missing_ok=True)

    sink = sink_for(category, tp_root)

    try:
        nopause_rec = _fire_single_shot(
            llm,
            category,
            arch,
            args,
            limits,
            tp,
            tp_root,
            shot,
            dvfs_pause=False,
            arm_label="nopause",
            sink=sink,
        )
        pause_rec = _fire_single_shot(
            llm,
            category,
            arch,
            args,
            limits,
            tp,
            tp_root,
            shot,
            dvfs_pause=True,
            arm_label="pause",
            sink=sink,
        )
    finally:
        if not dvfs_host_poller_enabled():
            restore_result = restore_gpu_freq(tp_root / "gpu_freq")
            if not restore_result.get("ok"):
                log.warning("DVFS restore failed: %s", restore_result)
        spin_down(llm, tmpdir)

    sink.flush()
    marker_pause_sum, marker_count = _sum_marker_pause_sec(markers_path)

    def _delta(key: str) -> float | None:
        a = nopause_rec.get(key)
        b = pause_rec.get(key)
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 6)

    ab_record = {
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "model": args.model,
        "hardware": args.hardware,
        "shot_key": shot_key,
        "engine_boot_sec": engine_boot_sec,
        "measurement_iterations": args.measurement_iterations,
        "pause_only_delay_sec": float(
            os.environ.get("PAUSE_ONLY_DELAY_SEC", "0.05")
        ),
        "single_engine": True,
        "nopause": nopause_rec,
        "pause": {
            **pause_rec,
            "marker_pause_sum_sec": marker_pause_sum,
            "marker_count": marker_count,
            "markers": str(markers_path),
        },
        "delta": {
            "measured_sec": _delta("measured_sec"),
            "rpc_wall_sec": _delta("rpc_wall_sec"),
            "fire_total_sec": _delta("fire_total_sec"),
        },
    }
    ab_path = tp_root / "ab_compare.json"
    ab_path.write_text(json.dumps(ab_record, indent=2) + "\n", encoding="utf-8")

    run_metrics = write_run_exec_metrics(
        tp_root,
        extra={"mode": "pause_ab", "category": category.name, "tp": tp},
        model_config=args.model_config,
        architecture=arch.model_dump(),
        tp=tp,
    )
    if run_metrics:
        log.success(
            "Pause A/B run totals: effective_runtime=%.3fs pause_overhead=%.3fs "
            "effective_energy=%.1fJ (%d shots) → %s",
            run_metrics.get("effective_runtime_sec", 0),
            run_metrics.get("pause_overhead_sec", 0),
            run_metrics.get("effective_energy_j", 0),
            run_metrics.get("shot_count", 0),
            tp_root / "run_exec_metrics.json",
        )

    log.success(
        "Pause A/B: measured_sec nopause=%.3fs pause=%.3fs delta=%.3fs "
        "(markers=%d sum=%.3fs) → %s",
        float(nopause_rec.get("measured_sec", 0)),
        float(pause_rec.get("measured_sec", 0)),
        float(_delta("measured_sec") or 0),
        marker_count,
        marker_pause_sum,
        ab_path,
    )

    if tp == 1:
        with log.stage("replicating tp_stable layers"):
            replicate_tp_stable(variant_root, arch, args.tp_degrees)

    persist_meta(args, arch_path, engine_kwargs, variant_root)
    log.done(variant_root)
