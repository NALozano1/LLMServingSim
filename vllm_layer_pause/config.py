"""Environment configuration for vLLM layer-boundary pause."""

from __future__ import annotations

import os

WORKER_EXTENSION_CLS = "profiler.core.hooks.extension.Extension"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def layer_pause_enabled() -> bool:
    """True when layer-boundary pause hooks should be active."""
    return _truthy("VLLM_LAYER_PAUSE") or _truthy("DVFS_LAYER_PAUSE")


def host_poller_enabled() -> bool:
    return _truthy("VLLM_HOST_POLLER") or _truthy("DVFS_HOST_POLLER")


def pause_only_enabled() -> bool:
    """Pause and ack at each layer; do not change GPU frequency."""
    return _truthy("VLLM_PAUSE_ONLY") or _truthy("DVFS_PAUSE_ONLY")


def pause_only_delay_sec() -> float:
    raw = os.environ.get("VLLM_PAUSE_ONLY_DELAY_SEC") or os.environ.get(
        "PAUSE_ONLY_DELAY_SEC", "0.05"
    )
    return float(raw)


def barrier_session_id() -> str:
    return os.environ.get("VLLM_LAYER_PAUSE_SESSION", "bench").strip() or "bench"


def external_host_poller_enabled() -> bool:
    """Host shell poller runs outside Python (ARC compute node)."""
    return _truthy("VLLM_EXTERNAL_HOST_POLLER")


def dvfs_freq_schedule() -> str | None:
    raw = os.environ.get("DVFS_FREQ_SCHEDULE", "").strip()
    return raw or None


def bench_gpu_power_enabled() -> bool:
    raw = os.environ.get("VLLM_BENCH_GPU_POWER", os.environ.get("PROFILER_GPU_POWER", "1"))
    return raw.strip().lower() in ("1", "true", "yes", "on")


def bench_prefill_only_enabled() -> bool:
    """Prefill-only bench: layer barriers on prompt forward, no decode steps."""
    return _truthy("VLLM_BENCH_PREFILL_ONLY") or _truthy("BENCH_PREFILL_ONLY")


def decode_max_pauses_per_pass() -> int:
    """Max layer-boundary pauses per decode forward (prefill is unrestricted)."""
    raw = os.environ.get("DVFS_DECODE_MAX_PAUSES_PER_PASS", "1")
    return max(0, int(raw))


def decode_token_threshold() -> int:
    """Token count at or below which a forward is treated as decode."""
    return max(1, int(os.environ.get("DVFS_DECODE_TOKEN_THRESHOLD", "4")))


def prefill_min_tokens() -> int:
    """Token count at or above which a forward is always treated as prefill."""
    return max(1, int(os.environ.get("DVFS_PREFILL_MIN_TOKENS", "8")))
