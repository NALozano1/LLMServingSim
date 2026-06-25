"""MoE forced-routing hook.

To profile the MoE block cleanly across the (tokens, activated_experts)
grid we need to control which experts receive tokens — relying on the
(dummy-weighted) learned gate would give unpredictable activation
patterns and poor grid coverage.

This module provides:

* ``ExpertRoute.forge``: build the ``(topk_weights, topk_ids)`` tensors
  that would force a given number of experts to be activated over a
  given number of tokens.

* ``force_moe_routing``: a context manager that monkey-patches
  ``FusedMoE.forward_native`` for the duration of the block so that
  ``self.router.select_experts`` returns our forged tensors instead
  of whatever the actual learned gate produces. The patch is
  reverted on exit.

The patch targets (``FusedMoE.forward_native`` and
``self.router.select_experts`` / ``self.router._compute_routing``)
are internal vLLM APIs. A vLLM version bump may require updating the
monkey-patch to match renamed or restructured symbols.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Iterator

import torch

# ---------------------------------------------------------------------------
# Thread-local accumulator for per-phase CUDA event timings
# ---------------------------------------------------------------------------

_phase_local = threading.local()


def _phase_reset() -> None:
    """Reset the per-shot phase accumulator.

    Call before each timed forward to start fresh.
    """
    _phase_local.gating_us_sum = 0.0
    _phase_local.expert_us_sum = 0.0
    _phase_local.calls = 0


def _phase_record(gating_us: float, expert_us: float) -> None:
    """Accumulate one MoE call's phase timings (microseconds)."""
    if not hasattr(_phase_local, "calls"):
        _phase_reset()
    _phase_local.gating_us_sum += gating_us
    _phase_local.expert_us_sum += expert_us
    _phase_local.calls += 1


def get_phase_timings_us() -> tuple[float, float, int]:
    """Return (gating_us_total, expert_us_total, n_calls) since last reset.

    Safe to call even if _phase_reset() was never called — returns zeros.
    """
    if not hasattr(_phase_local, "calls"):
        return 0.0, 0.0, 0
    return (
        _phase_local.gating_us_sum,
        _phase_local.expert_us_sum,
        _phase_local.calls,
    )


@dataclass
class ExpertRoute:
    """Precomputed tensors that pin routing to a specific expert set.

    Attributes:
        layer_name: Identifier of the FusedMoE layer this route targets.
            The runtime patch only applies when the currently-forwarding
            layer's ``layer_name`` matches.
        weights: Tensor of shape (num_tokens, top_k); each row is a
            uniform ``1/top_k`` distribution over the chosen experts.
        ids: Integer tensor of shape (num_tokens, top_k) naming the
            experts each token is routed to.
    """

    layer_name: str
    weights: torch.Tensor
    ids: torch.Tensor

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def forge(
        cls,
        layer,
        num_tokens: int,
        activated_experts: int,
    ) -> "ExpertRoute":
        """Allocate ``weights`` and ``ids`` for a single FusedMoE layer.

        Args:
            layer: The ``FusedMoE`` instance we'll patch. Used to read
                ``top_k`` and the expected indices dtype.
            num_tokens: Number of tokens being routed this call.
            activated_experts: How many distinct experts should receive
                at least one token. Must satisfy
                ``top_k <= activated_experts <= num_tokens * top_k``.
        """
        top_k = layer.top_k
        if activated_experts < top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) must be >= "
                f"top_k ({top_k})"
            )
        if activated_experts > num_tokens * top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) cannot exceed "
                f"num_tokens*top_k ({num_tokens * top_k})"
            )

        ids_rows = _cycle_expert_ids(num_tokens, top_k, activated_experts)

        # vLLM's router cares about the dtype of topk_ids: some kernels
        # expect int32, others a specific dtype reported by the router.
        indices_dtype = layer.router._get_indices_type()
        device = next(layer.parameters()).device

        ids = torch.tensor(
            ids_rows,
            device=device,
            dtype=torch.int32 if indices_dtype is None else indices_dtype,
        )
        # Shape sanity check — the kernel will complain later with a
        # cryptic message if this is wrong, so we prefer to fail early.
        expected_shape = (num_tokens, top_k)
        if tuple(ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch: expected {expected_shape}, "
                f"got {tuple(ids.shape)}"
            )

        weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            layer_name=layer.layer_name,
            weights=weights,
            ids=ids,
        )


def _cycle_expert_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    """Assign expert ids deterministically so exactly ``activated_experts``
    distinct ids appear, cycled across the token dimension.

    The specific assignment doesn't matter for latency — only the count
    of distinct activations does. We use the simplest pattern:
    ``id = (token_idx * top_k + offset) % activated_experts``.
    """
    return [
        [
            (token_idx * top_k + offset) % activated_experts
            for offset in range(top_k)
        ]
        for token_idx in range(num_tokens)
    ]


# ---------------------------------------------------------------------------
# Context manager: live FusedMoE patch
# ---------------------------------------------------------------------------

@contextmanager
def force_moe_routing(route: ExpertRoute | None) -> Iterator[None]:
    """Patch ``FusedMoE.forward_native`` to use ``route`` when called.

    If ``route`` is None the function is a no-op (useful for dense
    profile categories where we still pass through the MoE-aware
    execute path).

    The patch is layer-scoped: only the specific FusedMoE whose
    ``layer_name`` matches ``route.layer_name`` is affected; other MoE
    layers (if any) fall through to their normal forward. This matters
    if the model has multiple MoE layers and we're profiling only one
    at a time. For the single-layer test model we override to 1
    decoder layer, this is moot.
    """
    if route is None:
        yield
        return

    # Local import so that host-side code doesn't pay the vLLM import
    # cost just to read this module.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    original_forward_native = FusedMoE.forward_native

    @wraps(original_forward_native)
    def hooked_forward_native(self, hidden_states, router_logits):
        # Only patch the specific layer we care about. Any other
        # FusedMoE encountered during this forward pass uses its
        # normal routing.
        if self.layer_name != route.layer_name:
            return original_forward_native(self, hidden_states, router_logits)

        # We also sanity-check that our forged topk_ids matches the
        # actual per-call token count. If hidden_states is padded
        # differently than expected, bail.
        expected_shape = (hidden_states.shape[0], self.top_k)
        if tuple(route.ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch for {self.layer_name}: "
                f"expected {expected_shape}, got {tuple(route.ids.shape)}"
            )

        # The router's public API is select_experts(...). We override
        # the underlying _compute_routing to return our forged pair,
        # restore it after the single call completes. This is cleaner
        # than swapping select_experts wholesale because
        # select_experts does normalization / validation around
        # _compute_routing that we still want to run.
        #
        # PHASE ISOLATION: We additionally instrument hooked_select_experts
        # with CUDA Event timers to isolate gating time from expert-compute
        # time. The timings are stashed in a thread-local accumulator so
        # that fire() can read them after the layerwise_profile context
        # exits (the profiler's own hook records wall-clock totals for the
        # whole FusedMoE module; we record per-phase splits here in parallel).
        original_select_experts = self.router.select_experts
        original_compute_routing = self.router._compute_routing

        @wraps(original_select_experts)
        def hooked_select_experts(*args, **kwargs):
            @wraps(original_compute_routing)
            def forced_compute_routing(
                _hidden_states: torch.Tensor,
                _router_logits: torch.Tensor,
                _indices_type: torch.dtype | None,
            ):
                # Args are deliberately ignored — the whole point of
                # forced routing is that we return pre-forged values
                # regardless of the learned gate's logits.
                return route.weights, route.ids

            self.router._compute_routing = forced_compute_routing
            try:
                return original_select_experts(*args, **kwargs)
            finally:
                # Always restore _compute_routing, even if select_experts
                # raises (otherwise subsequent MoE calls in the same
                # profile session would keep returning our forged values).
                self.router._compute_routing = original_compute_routing

        self.router.select_experts = hooked_select_experts
        try:
            # ---------- PHASE-ISOLATED CUDA EVENT TIMING ----------
            # We run two pairs of CUDA Events:
            #   (ev_gate_start, ev_gate_end)   — gating phase
            #   (ev_exp_start,  ev_exp_end)    — expert compute phase
            #
            # The trick: we intercept forward_native by splitting the
            # call into the two logical phases ourselves.  forward_native
            # internally does:
            #   1. topk_weights, topk_ids = self.router.select_experts(...)  [GATING]
            #   2. hidden = fused_experts(hidden_states, ...)                [EXPERT COMPUTE]
            # rather than replicate that logic we wrap select_experts again
            # at the instance level to capture the gating boundary.
            #
            # Strategy: use torch.cuda.Event pairs. Record gate_start before
            # select_experts, gate_end after; exp_start immediately after,
            # exp_end after the full forward_native returns.  The difference
            # (forward_native total − gating) gives expert_us; directly
            # measuring gating gives gating_us.
            #
            # NOTE: torch.cuda.Event.elapsed_time() returns milliseconds.
            # We convert to microseconds for consistency with the rest of
            # the profiler (TimingSample.microseconds, writer time_us).
            ev_gate_start = torch.cuda.Event(enable_timing=True)
            ev_gate_end   = torch.cuda.Event(enable_timing=True)
            ev_exp_end    = torch.cuda.Event(enable_timing=True)

            # Wrap select_experts one more time to bookend gating.
            inner_select = self.router.select_experts

            def timed_select_experts(*a, **kw):
                ev_gate_start.record()
                result = inner_select(*a, **kw)
                ev_gate_end.record()
                return result

            self.router.select_experts = timed_select_experts
            try:
                result = original_forward_native(self, hidden_states, router_logits)
            finally:
                self.router.select_experts = inner_select

            ev_exp_end.record()
            torch.cuda.synchronize()

            # elapsed_time returns ms → convert to us
            gating_us = ev_gate_start.elapsed_time(ev_gate_end) * 1e3
            total_us  = ev_gate_start.elapsed_time(ev_exp_end)  * 1e3
            expert_us = max(0.0, total_us - gating_us)

            _phase_record(gating_us, expert_us)
            return result
        finally:
            self.router.select_experts = original_select_experts

    FusedMoE.forward_native = hooked_forward_native
    try:
        yield
    finally:
        # Restore the original class method so future calls (in tests,
        # or after this profile session ends) behave normally.
        FusedMoE.forward_native = original_forward_native


# ---------------------------------------------------------------------------
# Helpers that run worker-side
# ---------------------------------------------------------------------------

def single_moe_layer(model_runner):
    """Return the model's lone ``FusedMoE`` layer.

    We run profiling with ``hf_overrides.num_hidden_layers=1`` so
    there's exactly one of every decoder sub-module — including MoE.
    If for some reason there are zero or more-than-one, raise so the
    caller can investigate rather than forge the wrong route.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    model = model_runner.get_model()
    moe_layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if len(moe_layers) != 1:
        raise RuntimeError(
            f"Expected exactly one FusedMoE layer in the test model, "
            f"got {len(moe_layers)}"
        )
    return moe_layers[0]
