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
    get_phase_timings_us,
    single_moe_layer,
    _phase_reset,
)
from profiler.core.hooks.layer_barrier import (
    InPlaceLayerBarrier,
    discover_transformer_layers,
)
from profiler.core.hooks.timings import extract_samples
from profiler.core import logger as log

# Module-level flag so the layerwise_profile monkeypatch (applied inside
# fire() after the local vllm import) is applied exactly once per process.
_LAYERWISE_PROFILE_PATCHED: bool = False


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
        from vllm.profiler.layerwise_profile import (
            layerwise_profile,
            LayerwiseProfileResults,
            _ModuleTreeNode,
        )
        from vllm.profiler.utils import event_has_module, event_torch_op_stack_trace

        # ---- one-time monkeypatch ----------------------------------------
        # vLLM's LayerwiseProfileResults._build_module_tree filters events by
        # ``event.start_tid != 1`` (original comment: "For the tensor parallel
        # case for now only look at task 1").  In vLLM v1's EngineCore
        # architecture the model-runner forward is dispatched to a background
        # executor thread whose start_tid is NOT 1, so _module_tree ends up
        # empty for every profiling shot except whichever one happens to run on
        # the main thread during JIT initialisation (typically the max-token
        # compiled size).  An empty _module_tree causes _total_cuda_time() to
        # return 0 and _build_stats_trees() to raise ZeroDivisionError in
        # __exit__, which the caller (below) catches and returns [] — silently
        # dropping all per-layer data for those shots.
        #
        # Fix: replace _build_module_tree with a version that collects events
        # from ALL threads.  This is safe because the profiler always boots
        # vLLM with tensor_parallel_size=1 (see engine.py fuse_engine_kwargs),
        # so there is exactly one forward thread and all nn.Module events
        # belong to the model forward we want to measure.  The timing values
        # are unchanged: we still read cuda_time_us / invocations from the
        # same Kineto event fields.
        global _LAYERWISE_PROFILE_PATCHED
        if not _LAYERWISE_PROFILE_PATCHED:
            def _build_module_tree_all_tids(self) -> None:
                """Patched _build_module_tree: no start_tid filter."""
                self._module_tree = []
                event_tree = self._kineto_results.experimental_event_tree()

                def _df_traversal(event, curr_node=None):
                    # Removed: ``if event.start_tid != 1: return``
                    if event_has_module(event):
                        node = _ModuleTreeNode(event=event, parent=curr_node)
                        if curr_node:
                            curr_node.children.append(node)
                        else:
                            self._module_tree.append(node)
                        curr_node = node

                    is_leaf = event.children is None or len(event.children) == 0
                    if is_leaf and curr_node:
                        node = _ModuleTreeNode(
                            event=event,
                            parent=curr_node,
                            trace=event_torch_op_stack_trace(
                                event,
                                until=lambda x: event_has_module(x),
                            ),
                        )
                        curr_node.children.append(node)
                        curr_node = node

                    for child in event.children:
                        _df_traversal(child, curr_node)

                for root in event_tree:
                    _df_traversal(root)

            LayerwiseProfileResults._build_module_tree = _build_module_tree_all_tids
            log.debug(
                "Patched LayerwiseProfileResults._build_module_tree"
                " (removed start_tid==1 filter)"
            )
            _LAYERWISE_PROFILE_PATCHED = True
        # ---- end monkeypatch ------------------------------------------------

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

        # Reset the phase-split accumulator immediately before the measured
        # loops.  The warmup run above should NOT contribute.
        _phase_reset()

        measured_t0 = time.perf_counter()
        measured_wall_start = time.time()
        # measured_wall_end is set inside the with-layerwise_profile block (see
        # below) immediately after the last iteration, BEFORE the context-manager
        # __exit__ post-processes Kineto data.  Initialise here so it is always
        # defined even if an exception escapes the block.
        measured_wall_end: float = measured_wall_start
        barrier_wait_sec = 0.0
        try:
            try:
                with force_moe_routing(route):
                    with layerwise_profile() as hook:
                        for _ in range(iterations):
                            measured_out = self.model_runner.execute_model(
                                _fresh_batch()
                            )
                            if measured_out is None:
                                self.model_runner.sample_tokens(None)
                        # Capture the compute-window end BEFORE layerwise_profile
                        # __exit__ runs.  The context-manager exit post-processes
                        # the Kineto event tree (and, to a lesser extent, Python
                        # overhead for batch reconstruction accumulates between
                        # iterations), together adding ~0.3–0.7 s that is NOT GPU
                        # compute time.  Placing this timestamp here clips power
                        # integration to the actual measurement window so energy_j
                        # is accurate to ~1-2% for the paper.
                        measured_wall_end = time.time()
            finally:
                if layer_barrier is not None:
                    barrier_wait_sec = layer_barrier.barrier_wait_sec
                    layer_barrier.remove()
        except ZeroDivisionError:
            # After the _build_module_tree monkeypatch above, an empty
            # module_tree (and thus ZeroDivisionError in __exit__) should no
            # longer occur for normal shots.  Keep this handler as a safety
            # net: if it fires, something unexpected emptied the tree (e.g. a
            # vLLM version whose _build_module_tree logic differs from what the
            # patch replaced).
            log.warning(
                "layerwise_profile captured no CUDA events (empty module_tree"
                " despite start_tid patch). Returning empty samples for"
                " kind=%s shot. Check vLLM version compatibility.",
                kind,
            )
            return []

        # Read per-phase CUDA-event timings collected inside hooked_forward_native.
        # For non-MoE kinds these will be zeros (accumulator never incremented).
        phase_gating_us_total, phase_expert_us_total, phase_calls = get_phase_timings_us()
        # Per-call averages (matching how layerwise_profile divides by invocations).
        if phase_calls > 0:
            gating_ms_per_call = (phase_gating_us_total / phase_calls) / 1e3
            expert_ms_per_call = (phase_expert_us_total / phase_calls) / 1e3
        else:
            gating_ms_per_call = None
            expert_ms_per_call = None
        measured_sec = time.perf_counter() - measured_t0
        # NOTE: measured_wall_end was set inside the with-layerwise_profile block
        # (above).  measured_sec still includes profiler post-processing time
        # (intentional: it measures the full worker-side latency); only
        # measured_wall_end is advanced to the tight compute boundary.
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
                # MoE phase-split timings (None for non-moe kinds).
                "gating_ms": round(gating_ms_per_call, 4) if gating_ms_per_call is not None else None,
                "expert_ms": round(expert_ms_per_call, 4) if expert_ms_per_call is not None else None,
                "phase_calls": phase_calls,
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
        # Attach phase-split fields to every sample dict so the host can
        # read them without a separate RPC call.  For non-MoE kinds these
        # are None and the host ignores them.
        sample_dicts = [s.as_dict() for s in samples]
        for d in sample_dicts:
            d["gating_ms"] = round(gating_ms_per_call, 4) if gating_ms_per_call is not None else None
            d["expert_ms"] = round(expert_ms_per_call, 4) if expert_ms_per_call is not None else None
        return sample_dicts
