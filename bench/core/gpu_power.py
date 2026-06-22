"""GPU power sampling for bench runs (reuses profiler sampler)."""

from profiler.core.gpu_power import (
    GpuPowerSampler,
    integrate_power_joules,
    load_power_samples,
    profiler_gpu_power_enabled,
)

__all__ = [
    "GpuPowerSampler",
    "bench_gpu_power_enabled",
    "integrate_power_joules",
    "load_power_samples",
]


def bench_gpu_power_enabled() -> bool:
    import os

    raw = os.environ.get("VLLM_BENCH_GPU_POWER", os.environ.get("PROFILER_GPU_POWER", "1"))
    return raw.strip().lower() in ("1", "true", "yes", "on")
