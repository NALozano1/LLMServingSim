"""vLLM worker extension.

Registered via ``worker_extension_cls="profiler.core.hooks.extension.Extension"``
when constructing the ``vllm.LLM``. vLLM instantiates one Extension per
TP-rank worker process and exposes its methods through
``llm.collective_rpc(method_name, args=...)``.

The sole public method here is ``fire()``: it takes a serialized Shot
plus a catalog slice (the subset of the layer map relevant to the
category being profiled), runs the synthetic batch through
``model_runner.execute_model`` under ``layerwise_profile``, and
returns per-layer CUDA timings.

Measurement protocol per shot:
    1 warmup forward (discarded) — amortises JIT / paged-buffer setup
    N timed forwards inside ``layerwise_profile`` — the hook aggregates
        ``cuda_time_us`` across invocations; ``extract_samples``
        divides by ``invocations`` to return the per-call mean.

N defaults to ``ProfileArgs.measurement_iterations`` (3). A single
timed sample can swing 15-25%% on large GEMMs due to DVFS / boost
jitter; averaging cuts that noise floor dramatically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from profiler.core.hooks.batch import Shot, assemble_scheduler_output
from profiler.core.hooks.moe_hook import (
    ExpertRoute,
    force_moe_routing,
    single_moe_layer,
)
from profiler.core.hooks.layer_barrier import (
    InPlaceLayerBarrier,
    discover_transformer_layers,
)
from profiler.core.hooks.timings import extract_samples


class Extension:
    """Worker-side profiling entry point.

    vLLM mixes this class into ``Worker`` via dynamic inheritance; ``Worker``
    does not call ``Extension.__init__``, so layer-pause state is lazy-init.
    """

    def _persist_barrier_state(self) -> InPlaceLayerBarrier | None:
        return getattr(self, "_persist_barrier", None)

    def _set_persist_barrier(self, barrier: InPlaceLayerBarrier | None) -> None:
        self._persist_barrier = barrier

    def layer_pause_install(
        self,
        watch_root: str,
        session_id: str = "bench",
    ) -> dict[str, Any]:
        """Install post-forward hooks on every decoder layer for live forwards."""
        if self._persist_barrier_state() is not None:
            self.layer_pause_uninstall()

        root = Path(watch_root)
        barrier_dir = root / "dvfs_barriers" / session_id
        self._set_persist_barrier(
            InPlaceLayerBarrier(
                barrier_dir,
                shot_id=session_id,
            )
        )
        model = self.model_runner.get_model()
        layers = discover_transformer_layers(model)
        installed = self._persist_barrier_state().install(model)
        return {
            "session_id": session_id,
            "watch_root": str(root),
            "barrier_dir": str(barrier_dir),
            "layers_installed": installed,
            "num_model_layers": len(layers),
            "layer_names": [name for _, name, _ in layers],
        }

    def layer_pause_uninstall(self) -> dict[str, Any]:
        """Remove persistent layer pause hooks."""
        barrier = self._persist_barrier_state()
        if barrier is None:
            return {"layers_installed": 0, "barrier_wait_sec": 0.0}
        wait_sec = barrier.barrier_wait_sec
        barrier_dir = str(barrier.barrier_dir)
        barrier.remove()
        self._set_persist_barrier(None)
        return {
            "barrier_dir": barrier_dir,
            "barrier_wait_sec": round(wait_sec, 6),
        }

    def layer_pause_status(self) -> dict[str, Any]:
        """Return whether persistent hooks are installed."""
        barrier = self._persist_barrier_state()
        if barrier is None:
            return {"installed": False, "layers_installed": 0}
        return {
            "installed": True,
            "barrier_dir": str(barrier.barrier_dir),
            "barrier_wait_sec": round(barrier.barrier_wait_sec, 6),
        }

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
        barrier_dir: str | None = None,
        timing_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run one profiling shot and return per-layer timings.

        Args:
            shot_dict: Serialized ``Shot``; rehydrated inside the worker.
            slice_: Serialized catalog slice
                ``{canonical_name: {"vllm": cls, "within": parent, ...}}``
                scoped to the category we're profiling (so timings for
                unrelated layers aren't returned).
            kind: One of ``"dense"``, ``"per_sequence"``, ``"attention"``,
                ``"moe"``. Used to decide whether to forge MoE routing.
            iterations: Number of timed forward passes (averaged via
                the hook's invocation count). Default 3.
            barrier_dir: When set, install in-place layer-boundary
                barriers; worker blocks after each decoder layer until
                the host writes ``ack.json`` (DVFS applied on host).
            timing_path: When set, write per-phase wall times (warmup vs
                measured forwards only) as JSON on the host.

        Returns:
            List of ``TimingSample`` as plain dicts (pickled back to host).
        """
        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))
        fire_t0 = time.perf_counter()

        def _fresh_batch():
            # Rebuild the synthetic SchedulerOutput on every forward so
            # prior-iteration KV writes / request state don't bleed into
            # the next measurement.
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            return batch

        # -- warm-up run, result discarded -----------------------------
        # The first forward pays for JIT compilation, CUDA context
        # setup, paged-attention buffer allocation. We also call
        # sample_tokens to exercise the sampler path (if execute_model
        # returns None it means the scheduler consumed everything and
        # sample_tokens finalizes the step).
        warmup_t0 = time.perf_counter()
        warmup_out = self.model_runner.execute_model(_fresh_batch())
        if warmup_out is None:
            self.model_runner.sample_tokens(None)
        warmup_sec = time.perf_counter() - warmup_t0

        # -- optional MoE routing forge --------------------------------
        route: ExpertRoute | None = None
        if kind == "moe":
            if shot.experts is None or "activated" not in shot.experts:
                raise ValueError(
                    "moe shot missing experts.activated payload"
                )
            moe_layer = single_moe_layer(self.model_runner)
            num_tokens = sum(new for new, _ in shot.requests)
            route = ExpertRoute.forge(
                moe_layer,
                num_tokens=num_tokens,
                activated_experts=int(shot.experts["activated"]),
            )

        # -- measured runs (N iterations, averaged) -------------------
        # Local import so that profiler/__init__.py doesn't require
        # vllm.profiler to be importable at package-import time.
        #
        # vLLM's layerwise_profile hook accumulates ``cuda_time_us``
        # and ``invocations`` across every forward inside its context.
        # ``extract_samples`` divides one by the other, so running
        # execute_model N times here yields the per-call mean — the
        # cheap statistical fix for DVFS / boost-clock jitter that
        # single-sample measurements don't mitigate.
        from vllm.profiler.layerwise_profile import layerwise_profile

        layer_barrier: InPlaceLayerBarrier | None = None
        if barrier_dir:
            shot_id = Path(barrier_dir).name
            layer_barrier = InPlaceLayerBarrier(
                barrier_dir,
                shot_id=shot_id,
            )
            n_layers = layer_barrier.install(self.model_runner.get_model())
            if n_layers == 0:
                layer_barrier.remove()
                layer_barrier = None

        measured_t0 = time.perf_counter()
        measured_wall_start = time.time()
        barrier_wait_sec = 0.0
        try:
            with force_moe_routing(route):
                with layerwise_profile() as hook:
                    for _ in range(iterations):
                        measured_out = self.model_runner.execute_model(
                            _fresh_batch()
                        )
                        if measured_out is None:
                            self.model_runner.sample_tokens(None)
        finally:
            if layer_barrier is not None:
                barrier_wait_sec = layer_barrier.barrier_wait_sec
                layer_barrier.remove()
        measured_sec = time.perf_counter() - measured_t0
        measured_wall_end = time.time()
        fire_total_sec = time.perf_counter() - fire_t0
        measured_exec_sec = max(0.0, measured_sec - barrier_wait_sec)

        if timing_path:
            timing_rec = {
                "warmup_sec": round(warmup_sec, 6),
                "measured_sec": round(measured_sec, 6),
                "measured_exec_sec": round(measured_exec_sec, 6),
                "barrier_wait_sec": round(barrier_wait_sec, 6),
                "fire_total_sec": round(fire_total_sec, 6),
                "fire_exec_sec": round(max(0.0, fire_total_sec - barrier_wait_sec), 6),
                "measured_wall_start": measured_wall_start,
                "measured_wall_end": measured_wall_end,
                "measurement_iterations": iterations,
                "barrier_enabled": bool(barrier_dir),
            }
            out = Path(timing_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(timing_rec, indent=2) + "\n",
                encoding="utf-8",
            )

        stats = hook.results.convert_stats_to_dict()
        summary = stats["summary_stats"]

        samples = extract_samples(summary, slice_)
        return [s.as_dict() for s in samples]
