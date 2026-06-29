"""Measured V100 DVFS assets: profiler latency tables + captured GPU power.

For each clock tag (``V100``, ``V100_700MHz``, …) this module resolves:

* **Latency** — ``profiler/perf/<tag>/<model>/fp16/tp1/{dense,attention,moe}.csv``
* **Idle power** — median ``power.draw`` from ``profiler/power/data/v100_<MHz>MHz/*/gpu_power.csv``
* **Active / standby power** — p95 / p99 of ``power_w`` from profiler-shot ``gpu_power/*.jsonl``
  at the same hardware tag (measured while layer profiling ran at that frequency).

No synthetic MHz scaling is applied when measured files are present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]
PERF_ROOT = _REPO / "profiler" / "perf"
POWER_ROOT = _REPO / "profiler" / "power" / "data"

REFERENCE_MODELS = (
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
)

# V100 SXM2 max graphics clock; stock ``V100`` profiler/power bundles are at this frequency.
V100_MAX_MHZ = 1400
V100_MAX_BOOST_MHZ = V100_MAX_MHZ
V100_DEFAULT_HARDWARE = "V100"

# Distinct profiler/power tags for the mini3x3 campaign (1400 MHz → default ``V100``).
CAMPAIGN_HARDWARE_MHZ: tuple[int | None, ...] = (None, 700, 900, 1100)


def is_default_v100_mhz(mhz: int) -> bool:
    return mhz >= V100_MAX_MHZ


def mhz_to_hardware(mhz: int, *, default_tag: str = V100_DEFAULT_HARDWARE) -> str:
    """Map a target MHz to the profiler hardware tag used by LLMServingSim."""
    if is_default_v100_mhz(mhz):
        return default_tag
    return f"V100_{mhz}MHz"


# Alias used by layer-boundary DVFS schedule + replication scripts.
hardware_for_dvfs_mhz = mhz_to_hardware


@dataclass(frozen=True)
class V100HardwareTag:
    hardware: str
    mhz: int | None
    power_data_tag: str


def hardware_tag_for_mhz(mhz: int | None) -> V100HardwareTag:
    if mhz is None or is_default_v100_mhz(mhz):
        return V100HardwareTag(V100_DEFAULT_HARDWARE, mhz, "v100_default")
    return V100HardwareTag(f"V100_{mhz}MHz", mhz, f"v100_{mhz}MHz")


def profiler_variant_root(hardware: str, model: str, variant: str = "fp16") -> Path:
    return PERF_ROOT / hardware / model / variant


def profiler_dense_csv(hardware: str, model: str, variant: str = "fp16") -> Path:
    return profiler_variant_root(hardware, model, variant) / "tp1" / "dense.csv"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_power_capture(tag: str) -> Path | None:
    root = POWER_ROOT / tag
    if not root.is_dir():
        return None
    runs = sorted(root.glob("*/gpu_power.csv"))
    return runs[-1] if runs else None


def _read_capture_idle_w(capture_csv: Path) -> float:
    values: list[float] = []
    with capture_csv.open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "timestamp":
                continue
            try:
                values.append(float(row[6]))
            except (ValueError, IndexError):
                continue
    if not values:
        raise ValueError(f"no power.draw samples in {capture_csv}")
    return float(statistics.median(values))


def _profiler_gpu_power_samples(hardware: str, models: Iterable[str]) -> list[float]:
    values: list[float] = []
    for model in models:
        gpdir = profiler_variant_root(hardware, model) / "tp1" / "gpu_power"
        if not gpdir.is_dir():
            continue
        for jf in gpdir.glob("*.jsonl"):
            for line in jf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                for gpu in json.loads(line).get("gpus", []):
                    values.append(float(gpu["power_w"]))
    return values


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty power sample set")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


@dataclass(frozen=True)
class MeasuredNpuPower:
    idle_power: float
    standby_power: float
    active_power: float
    standby_duration: int = 18
    idle_source: str = ""
    active_source: str = ""
    standby_source: str = ""


def measured_npu_power(
    tag: V100HardwareTag,
    *,
    models: Iterable[str] = REFERENCE_MODELS,
) -> MeasuredNpuPower:
    """Resolve simulator NPU power from measured captures + profiler shot power."""
    capture = _latest_power_capture(tag.power_data_tag)
    shot_powers = _profiler_gpu_power_samples(tag.hardware, models)

    if capture is not None:
        idle_w = _read_capture_idle_w(capture)
        idle_source = str(capture.relative_to(_REPO))
    elif shot_powers:
        idle_w = _percentile(shot_powers, 10)
        idle_source = f"profiler/perf/{tag.hardware}/*/fp16/tp1/gpu_power (p10)"
    else:
        raise FileNotFoundError(
            f"no measured idle power for {tag.hardware}: "
            f"missing {POWER_ROOT / tag.power_data_tag} and profiler gpu_power"
        )

    if not shot_powers:
        raise FileNotFoundError(
            f"no profiler gpu_power samples for {tag.hardware} "
            f"(expected under profiler/perf/{tag.hardware}/<model>/fp16/tp1/gpu_power/)"
        )

    active_w = _percentile(shot_powers, 95)
    standby_w = max(idle_w * 1.5, _percentile(shot_powers, 99))
    active_source = f"profiler/perf/{tag.hardware}/*/fp16/tp1/gpu_power (p95)"
    standby_source = f"profiler/perf/{tag.hardware}/*/fp16/tp1/gpu_power (p99)"

    return MeasuredNpuPower(
        idle_power=round(idle_w, 2),
        standby_power=round(standby_w, 2),
        active_power=round(active_w, 2),
        idle_source=idle_source,
        active_source=active_source,
        standby_source=standby_source,
    )


@dataclass(frozen=True)
class MeasuredAssetReport:
    hardware: str
    mhz: int | None
    profiler_models: dict[str, str]
    power: MeasuredNpuPower


def validate_measured_assets(
    *,
    model: str,
    mhz_values: Iterable[int | None],
    reference_models: Iterable[str] = REFERENCE_MODELS,
) -> list[MeasuredAssetReport]:
    """Ensure profiler + power exist for each MHz; return provenance report."""
    reports: list[MeasuredAssetReport] = []
    seen_hardware: set[str] = set()
    for mhz in mhz_values:
        tag = hardware_tag_for_mhz(mhz)
        if tag.hardware in seen_hardware:
            continue
        seen_hardware.add(tag.hardware)
        dense_path = profiler_dense_csv(tag.hardware, model)
        if not dense_path.is_file():
            raise FileNotFoundError(
                f"missing measured profiler for {tag.hardware}/{model}: {dense_path}"
            )
        model_hashes = {
            m: _sha256_file(profiler_dense_csv(tag.hardware, m))
            for m in reference_models
            if profiler_dense_csv(tag.hardware, m).is_file()
        }
        if not model_hashes:
            raise FileNotFoundError(
                f"no reference profiler bundles under {tag.hardware} "
                f"(expected one of {reference_models})"
            )
        power = measured_npu_power(tag, models=reference_models)
        reports.append(
            MeasuredAssetReport(
                hardware=tag.hardware,
                mhz=tag.mhz,
                profiler_models=model_hashes,
                power=power,
            )
        )
    return reports


def build_dvfs_cluster_config(
    *,
    model_name: str,
    mhz_values: Iterable[int | None],
    npu_mem_gb: int = 32,
) -> dict:
    """Cluster JSON with per-hardware measured NPU power entries."""
    power_entries: dict[str, dict] = {}
    seen_hardware: set[str] = set()
    for mhz in mhz_values:
        tag = hardware_tag_for_mhz(mhz)
        if tag.hardware in seen_hardware:
            continue
        seen_hardware.add(tag.hardware)
        p = measured_npu_power(tag)
        power_entries[tag.hardware] = {
            "idle_power": p.idle_power,
            "standby_power": p.standby_power,
            "active_power": p.active_power,
            "standby_duration": p.standby_duration,
            "num_npus": 0,
        }

    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 20000,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": 512,
                    "mem_bw": 256,
                    "mem_latency": 0,
                },
                "instances": [
                    {
                        "model_name": model_name,
                        "hardware": "V100",
                        "npu_mem": {
                            "mem_size": npu_mem_gb,
                            "mem_bw": 900,
                            "mem_latency": 0,
                        },
                        "num_npus": 1,
                        "tp_size": 1,
                        "pd_type": None,
                    }
                ],
                "power": {
                    "base_node_power": 60,
                    "npu": power_entries,
                    "cpu": {
                        "idle_power": 10,
                        "active_power": 200,
                        "util": 0.15,
                    },
                    "dram": {
                        "dimm_size": 32,
                        "idle_power": 2.0,
                        "energy_per_bit": 6.0,
                        "mem_size": 512,
                    },
                    "link": {
                        "num_links": 1,
                        "idle_power": 5,
                        "energy_per_bit": 4.0,
                    },
                    "nic": {
                        "num_nics": 1,
                        "idle_power": 20,
                    },
                    "storage": {
                        "num_devices": 2,
                        "idle_power": 5,
                    },
                },
            }
        ],
    }
