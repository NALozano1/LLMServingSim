"""vLLM benchmark runner — strict replay of an existing dataset.

The runner reads a LLMServingSim-format JSONL workload (the same format
``python -m workloads.generators sharegpt`` produces and ``python -m serving
--dataset`` consumes) and replays every request through vLLM with its
``input_tok_ids`` and ``output_toks`` pinned, so the run is bit-for-bit
comparable to the simulator's view of the same workload.

White-room implementation against ``../vllm``:
  * ``vllm.v1.engine.async_llm.AsyncLLM`` — async engine, ``generate()``
    yields ``RequestOutput`` per chunk, ``RequestOutput.metrics`` carries
    per-request ``RequestStateStats`` (arrival_time / queued_ts /
    scheduled_ts / first_token_ts / last_token_ts).
  * ``vllm.v1.metrics.loggers.StatLoggerBase`` — pluggable per-engine stat
    logger; we hook it via ``BenchStatLogger`` to capture per-iteration
    scheduler/iteration stats for ``timeseries.csv``.

Output: ``<output-dir>/{meta.json, requests.jsonl, timeseries.csv}``.
The dataset itself is not modified — generation lives in
``workloads/generators``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from bench.core import logger as log
from bench.core import recorder


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id passed verbatim to vllm.AsyncLLM.")
    p.add_argument("--dataset", required=True,
                   help="Path to a LLMServingSim-format JSONL workload "
                        "(produced by `python -m workloads.generators`).")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory for this run "
                        "(meta.json/requests.jsonl/timeseries.csv).")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   dest="tensor_parallel_size",
                   help="vLLM tensor_parallel_size.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   dest="data_parallel_size",
                   help="vLLM data_parallel_size (DP across engines).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   dest="enable_expert_parallel", default=False,
                   help="vLLM enable_expert_parallel for MoE models.")
    p.add_argument("--max-num-seqs", type=int, default=128,
                   dest="max_num_seqs",
                   help="vLLM scheduler max_num_seqs (per-engine running cap).")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   dest="max_num_batched_tokens",
                   help="vLLM scheduler max_num_batched_tokens.")
    p.add_argument("--max-model-len", type=int, default=None,
                   dest="max_model_len",
                   help="vLLM max_model_len (None = model's max).")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   dest="gpu_memory_utilization",
                   help="vLLM gpu_memory_utilization (default: vLLM default).")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   dest="kv_cache_dtype",
                   help="vLLM kv_cache_dtype.")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for vLLM.")
    p.add_argument("--tick-seconds", type=float, default=1.0,
                   dest="tick_seconds",
                   help="Stat logger downsample interval (timeseries.csv row spacing).")
    p.add_argument("--num-reqs", type=int, default=0,
                   dest="num_reqs",
                   help="Cap on number of requests from the dataset (0 = all).")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")


def run(args: argparse.Namespace) -> int:
    from bench.core.stat_logger import BenchStatLogger

    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Bench",
        f"vLLM end-to-end run -> {args.output_dir}",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests:
        raise ValueError(f"No requests loaded from {args.dataset}")
    log.info("Loaded %d requests from %s", len(requests), args.dataset)

    from bench.core.layer_pause import layer_pause_enabled

    BenchStatLogger.reset()
    if layer_pause_enabled():
        _drive_sync_llm(args, requests, output_dir)
    else:
        asyncio.run(_drive(args, requests, output_dir))
    return 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> list[dict]:
    """Read a LLMServingSim-format JSONL workload.

    Skips agentic-session rows (with ``sub_requests``) — bench currently
    handles only flat requests. Each row must carry ``input_tok_ids``;
    bench cannot tokenize on the fly because the dataset's tokenizer may
    differ from ``args.model``.
    """
    requests: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sub_requests" in row:
                continue  # agentic sessions: not supported in bench yet
            if "input_tok_ids" not in row or not row["input_tok_ids"]:
                raise ValueError(
                    f"Row missing input_tok_ids in {path}; regenerate the "
                    f"dataset with `python -m workloads.generators`."
                )
            requests.append(row)
            if cap and len(requests) >= cap:
                break
    return requests


# ---------------------------------------------------------------------------
# Sync driver (layer-pause bench — blocking hooks deadlock V1 AsyncLLM decode)
# ---------------------------------------------------------------------------

def _drive_sync_llm(
    args: argparse.Namespace,
    requests: list[dict],
    output_dir: Path,
) -> None:
    import os
    import time

    # V1 async scheduling deadlocks when forward hooks block during decode.
    os.environ.setdefault("VLLM_USE_V1", "0")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from bench.core.exec_metrics import write_bench_run_metrics
    from bench.core.gpu_power import GpuPowerSampler, bench_gpu_power_enabled
    from bench.core.layer_pause import (
        external_host_poller_enabled,
        host_poller_enabled,
        install_layer_pause_sync,
        layer_pause_enabled,
        layer_pause_engine_kwargs,
        start_host_poller,
        stop_host_poller,
        uninstall_layer_pause_sync,
    )
    from bench.core.stat_logger import BenchStatLogger
    from vllm_layer_pause.config import (
        bench_prefill_only_enabled,
        decode_max_pauses_per_pass,
    )

    repo_root = Path(__file__).resolve().parents[2]
    poller_proc = None
    power_sampler: GpuPowerSampler | None = None
    pause_totals: dict[str, Any] = {}
    pause_info: dict[str, Any] = {}
    records: list[dict] = []
    started_at = ""
    finished_at = ""

    if layer_pause_enabled() and host_poller_enabled() and not external_host_poller_enabled():
        poller_proc = start_host_poller(output_dir, repo_root)
        log.info("Layer pause enabled (in-process host poller on %s)", output_dir)
    elif layer_pause_enabled() and external_host_poller_enabled():
        log.info("Layer pause enabled (external host poller on %s)", output_dir)

    engine_kwargs = layer_pause_engine_kwargs()
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "seed": args.seed,
        "disable_log_stats": False,
        **engine_kwargs,
    }
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    engine_kwargs_for_meta = {
        k: llm_kwargs.get(k)
        for k in (
            "model",
            "tensor_parallel_size",
            "data_parallel_size",
            "enable_expert_parallel",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "dtype",
            "kv_cache_dtype",
            "seed",
        )
    }
    if layer_pause_enabled():
        engine_kwargs_for_meta["layer_pause"] = True
        engine_kwargs_for_meta["worker_extension_cls"] = engine_kwargs.get(
            "worker_extension_cls"
        )
        engine_kwargs_for_meta["engine_api"] = "LLM"
    if bench_prefill_only_enabled():
        engine_kwargs_for_meta["prefill_only"] = True
    elif layer_pause_enabled():
        engine_kwargs_for_meta["prefill_only"] = False
        engine_kwargs_for_meta["decode_max_pauses_per_pass"] = (
            decode_max_pauses_per_pass()
        )

    try:
        with log.stage("Booting LLM"):
            with log.capture_stdio():
                llm = LLM(**llm_kwargs)
        started_at = datetime.datetime.utcnow().isoformat() + "Z"

        if layer_pause_enabled():
            with log.stage("Installing layer pause hooks"):
                pause_info = install_layer_pause_sync(llm, output_dir)
                log.info(
                    "Layer pause installed on %d/%d layers -> %s",
                    pause_info.get("layers_installed", 0),
                    pause_info.get("num_model_layers", 0),
                    pause_info.get("barrier_dir", "?"),
                )

        if bench_gpu_power_enabled():
            power_path = output_dir / "gpu_power" / "bench.jsonl"
            power_sampler = GpuPowerSampler(power_path)
            power_sampler.start()
            log.info("GPU power sampling -> %s", power_path)

        try:
            with log.stage(f"Submitting {len(requests)} requests"):
                records = _submit_all_sync(llm, requests, SamplingParams, TokensPrompt)
        finally:
            if power_sampler is not None:
                power_sampler.stop()
            if layer_pause_enabled():
                with log.stage("Removing layer pause hooks"):
                    pause_totals = uninstall_layer_pause_sync(llm)
                    log.info(
                        "Layer pause barrier wait total: %.3fs",
                        pause_totals.get("barrier_wait_sec", 0.0),
                    )
    finally:
        stop_host_poller(poller_proc)
        if "llm" in locals():
            del llm
            time.sleep(0.5)

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version=_vllm_version(),
        engine_kwargs=engine_kwargs_for_meta,
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
    )
    recorder.write_requests(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)

    if layer_pause_enabled():
        metrics = write_bench_run_metrics(
            output_dir,
            started_at=started_at,
            finished_at=finished_at,
            barrier_wait_sec=float(pause_totals.get("barrier_wait_sec", 0.0)),
            layer_pause=pause_info,
        )
        log.info(
            "Run metrics wall=%.2fs exec=%.2fs pause=%.2fs out_tok/s=%s energy=%.1fJ",
            metrics["timing"].get("wall_sec") or 0.0,
            metrics["timing"].get("exec_sec") or 0.0,
            metrics["timing"].get("pause_sec") or 0.0,
            metrics["throughput"].get("output_tok_per_sec_exec"),
            metrics["energy"].get("energy_j") or 0.0,
        )

    log.success(
        "%d requests, %d timeseries rows -> %s",
        len(records),
        len(rows),
        output_dir,
    )


def _submit_all_sync(llm, requests: list[dict], SamplingParams, TokensPrompt) -> list[dict]:
    import time

    from vllm_layer_pause.config import bench_prefill_only_enabled

    t0 = time.perf_counter()
    records: list[dict] = []

    with log.progress("Requests", total=len(requests)) as bar:
        for idx, req in enumerate(requests):
            target = t0 + req["arrival_time_ns"] / 1e9
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            n_out = int(req["output_toks"])
            if bench_prefill_only_enabled() or n_out <= 0:
                # vLLM requires max_tokens>=1; with V0 engine hooks fire on prefill only.
                sp = SamplingParams(
                    max_tokens=1,
                    min_tokens=1,
                    ignore_eos=True,
                    temperature=0.0,
                )
            else:
                sp = SamplingParams(
                    min_tokens=n_out,
                    max_tokens=n_out,
                    ignore_eos=True,
                    temperature=0.0,
                )
            prompt = TokensPrompt(prompt_token_ids=list(req["input_tok_ids"]))
            outputs = llm.generate([prompt], sp)
            last_metrics = None
            if outputs:
                last_metrics = outputs[0].metrics
            records.append(_record_from_metrics(idx, req, last_metrics))
            bar.advance()

    return records


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def _drive(args: argparse.Namespace, requests: list[dict], output_dir: Path) -> None:
    # Imports deferred so `validate` / `--help` works without vLLM installed.
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    from bench.core.exec_metrics import write_bench_run_metrics
    from bench.core.gpu_power import GpuPowerSampler, bench_gpu_power_enabled
    from bench.core.layer_pause import (
        external_host_poller_enabled,
        host_poller_enabled,
        install_layer_pause,
        layer_pause_enabled,
        layer_pause_engine_kwargs,
        start_host_poller,
        stop_host_poller,
        uninstall_layer_pause,
    )
    from bench.core.stat_logger import BenchStatLogger
    from vllm_layer_pause.config import bench_prefill_only_enabled

    repo_root = Path(__file__).resolve().parents[2]
    poller_proc = None
    power_sampler: GpuPowerSampler | None = None
    pause_totals: dict[str, Any] = {}
    pause_info: dict[str, Any] = {}
    records: list[dict] = []
    started_at = ""
    finished_at = ""

    if layer_pause_enabled() and host_poller_enabled() and not external_host_poller_enabled():
        poller_proc = start_host_poller(output_dir, repo_root)
        log.info("Layer pause enabled (in-process host poller on %s)", output_dir)
    elif layer_pause_enabled() and external_host_poller_enabled():
        log.info("Layer pause enabled (external host poller on %s)", output_dir)

    engine_kwargs = layer_pause_engine_kwargs()
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        seed=args.seed,
        disable_log_stats=False,
        **({"gpu_memory_utilization": args.gpu_memory_utilization}
           if args.gpu_memory_utilization is not None else {}),
        **engine_kwargs,
    )
    engine_kwargs_for_meta = _engine_kwargs_for_meta(engine_args)
    if layer_pause_enabled():
        engine_kwargs_for_meta["layer_pause"] = True
        engine_kwargs_for_meta["worker_extension_cls"] = engine_kwargs.get(
            "worker_extension_cls"
        )

    try:
        with log.stage("Booting AsyncLLM"):
            with log.capture_stdio():
                engine = AsyncLLM.from_engine_args(
                    engine_args, stat_loggers=[BenchStatLogger]
                )
        started_at = datetime.datetime.utcnow().isoformat() + "Z"

        if layer_pause_enabled():
            with log.stage("Installing layer pause hooks"):
                pause_info = await install_layer_pause(engine, output_dir)
                log.info(
                    "Layer pause installed on %d/%d layers -> %s",
                    pause_info.get("layers_installed", 0),
                    pause_info.get("num_model_layers", 0),
                    pause_info.get("barrier_dir", "?"),
                )

        if bench_gpu_power_enabled():
            power_path = output_dir / "gpu_power" / "bench.jsonl"
            power_sampler = GpuPowerSampler(power_path)
            power_sampler.start()
            log.info("GPU power sampling -> %s", power_path)

        try:
            with log.stage(f"Submitting {len(requests)} requests"):
                records = await _submit_all(
                    engine, requests, SamplingParams, TokensPrompt
                )
        finally:
            if power_sampler is not None:
                power_sampler.stop()
            if layer_pause_enabled():
                with log.stage("Removing layer pause hooks"):
                    pause_totals = await uninstall_layer_pause(engine)
                    log.info(
                        "Layer pause barrier wait total: %.3fs",
                        pause_totals.get("barrier_wait_sec", 0.0),
                    )
    finally:
        if "engine" in locals():
            with log.stage("Shutting AsyncLLM down"):
                engine.shutdown()
        stop_host_poller(poller_proc)

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Persist outputs.
    # ------------------------------------------------------------------
    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version=_vllm_version(),
        engine_kwargs=engine_kwargs_for_meta,
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
    )
    recorder.write_requests(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)

    if layer_pause_enabled():
        metrics = write_bench_run_metrics(
            output_dir,
            barrier_wait_sec=float(pause_totals.get("barrier_wait_sec", 0.0)),
            layer_pause=pause_info,
        )
        log.info(
            "Run metrics wall=%.2fs exec=%.2fs pause=%.2fs out_tok/s=%s energy=%.1fJ",
            metrics["timing"].get("wall_sec") or 0.0,
            metrics["timing"].get("exec_sec") or 0.0,
            metrics["timing"].get("pause_sec") or 0.0,
            metrics["throughput"].get("output_tok_per_sec_exec"),
            metrics["energy"].get("energy_j") or 0.0,
        )

    log.success(
        "%d requests, %d timeseries rows -> %s",
        len(records), len(rows), output_dir,
    )


async def _submit_all(engine, requests: list[dict], SamplingParams, TokensPrompt) -> list[dict]:
    """Schedule each request at its arrival offset, gather metrics."""
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    completed = [0]  # boxed so the inner closure can mutate

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            target = t0_loop + req["arrival_time_ns"] / 1e9
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            # Strict replay: pin output length to whatever the dataset
            # recorded. ``ignore_eos`` blocks early termination; ``min_tokens``
            # blocks vLLM's async-scheduling early-exit (see vllm/v1/engine/
            # async_llm.py:async-scheduling block) so n_out is exactly fixed.
            n_out = int(req["output_toks"])
            if bench_prefill_only_enabled() or n_out <= 0:
                # vLLM requires max_tokens>=1; prefill_only stops after 1 token.
                sp = SamplingParams(
                    max_tokens=1,
                    min_tokens=1,
                    ignore_eos=True,
                    temperature=0.0,
                )
            else:
                sp = SamplingParams(
                    min_tokens=n_out,
                    max_tokens=n_out,
                    ignore_eos=True,
                    temperature=0.0,
                )
            prompt = TokensPrompt(prompt_token_ids=list(req["input_tok_ids"]))
            request_id = f"bench-{idx}"

            last_metrics = None
            async for output in engine.generate(prompt, sp, request_id):
                if output.metrics is not None:
                    last_metrics = output.metrics

            completed[0] += 1
            bar.advance()
            return _record_from_metrics(idx, req, last_metrics)

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        return await asyncio.gather(*tasks)


def _record_from_metrics(idx: int, req: dict, metrics) -> dict:
    """Project ``RequestStateStats`` onto our flat per-request schema."""
    if metrics is None:
        return {
            "request_id": f"bench-{idx}",
            "input_toks": int(req["input_toks"]),
            "output_toks": int(req["output_toks"]),
            "arrival_time": None,
            "queued_ts": None,
            "scheduled_ts": None,
            "first_token_ts": None,
            "last_token_ts": None,
        }
    return {
        "request_id": f"bench-{idx}",
        "input_toks": int(req["input_toks"]),
        "output_toks": int(req["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "queued_ts": getattr(metrics, "queued_ts", None),
        "scheduled_ts": getattr(metrics, "scheduled_ts", None),
        "first_token_ts": getattr(metrics, "first_token_ts", None),
        "last_token_ts": getattr(metrics, "last_token_ts", None),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_kwargs_for_meta(engine_args) -> dict:
    fields = (
        "model", "tensor_parallel_size", "data_parallel_size",
        "enable_expert_parallel", "max_num_seqs", "max_num_batched_tokens",
        "max_model_len", "dtype", "kv_cache_dtype", "seed",
    )
    return {k: getattr(engine_args, k, None) for k in fields}


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
