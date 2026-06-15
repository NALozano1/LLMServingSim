"""Discover GPU hardware profiler aliases and optional scaled seed for DVFS demos."""

from __future__ import annotations

import csv
import logging
import os
import shutil
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_TIME_US_COLS = {"time_us", "latency_us"}


def _profiler_perf_root() -> str:
    """Resolve profiler/perf from repo root or astra-sim cwd."""
    candidates = (
        os.path.join(os.path.dirname(__file__), "..", "..", "profiler", "perf"),
        "profiler/perf",
        "../profiler/perf",
    )
    for path in candidates:
        abspath = os.path.abspath(path)
        if os.path.isdir(abspath):
            return abspath
    return os.path.abspath(candidates[0])


def _v0_perf_root() -> str:
    candidates = (
        os.path.join(os.path.dirname(__file__), "..", "..", "profiler", "v0", "perf_models"),
        "profiler/v0/perf_models",
        "../profiler/v0/perf_models",
    )
    for path in candidates:
        abspath = os.path.abspath(path)
        if os.path.isdir(abspath):
            return abspath
    return os.path.abspath(candidates[0])


def discover_hardware_for_model(model: str, variant: str = "bf16") -> List[str]:
    """Hardware names with profiler/perf/{hw}/{model}/{variant}/."""
    root = _profiler_perf_root()
    if not os.path.isdir(root):
        return []
    found = []
    for hw in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, hw, model, variant)):
            found.append(hw)
    return found


def _discover_v0_hardware_for_model(model: str) -> List[str]:
    root = _v0_perf_root()
    if not os.path.isdir(root):
        return []
    found = []
    for hw in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, hw, model, "tp1", "layers.csv")):
            found.append(hw)
    return found


def _scale_csv_times(path: str, scale: float) -> None:
    if not os.path.isfile(path):
        return
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        fieldnames = list(reader.fieldnames)
        time_col = next((c for c in fieldnames if c in _TIME_US_COLS or c.endswith("_us")), None)
        if time_col is None:
            return
        for row in reader:
            try:
                row[time_col] = str(float(row[time_col]) * scale)
            except (TypeError, ValueError, KeyError):
                pass
            rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_scaled_hardware_alias(
    source_hw: str,
    target_hw: str,
    model: str,
    variant: str = "bf16",
    scale: float = 1.25,
) -> bool:
    """Copy profiler/perf source_hw -> target_hw with scaled layer timings."""
    src = os.path.join(_profiler_perf_root(), source_hw, model, variant)
    dst = os.path.join(_profiler_perf_root(), target_hw, model, variant)
    if os.path.isdir(dst):
        return True
    if not os.path.isdir(src):
        return False
    shutil.copytree(src, dst)
    meta_path = os.path.join(dst, "meta.yaml")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            text = f.read()
        text = text.replace(f"hardware: {source_hw}", f"hardware: {target_hw}")
        with open(meta_path, "w") as f:
            f.write(text)
    for root, _, files in os.walk(dst):
        for name in files:
            if name.endswith(".csv"):
                _scale_csv_times(os.path.join(root, name), scale)
    logger.info(
        "Seeded hardware alias %s from %s (scale=%.2f) for %s/%s",
        target_hw, source_hw, scale, model, variant,
    )
    return True


def resolve_hardware_pair(
    model: str,
    variant: str,
    primary: str,
    alt: Optional[str] = None,
    seed_if_missing: bool = True,
) -> Tuple[str, str]:
    """Return (hw_a, hw_b) for layer-boundary hardware alternation."""
    if alt is not None and alt != primary:
        if seed_if_missing and alt not in discover_hardware_for_model(model, variant):
            seed_scaled_hardware_alias(primary, alt, model, variant)
        if os.path.isdir(os.path.join(_profiler_perf_root(), alt, model, variant)):
            return primary, alt

    discovered = discover_hardware_for_model(model, variant)
    if primary in discovered:
        others = [h for h in discovered if h != primary]
        if others:
            return primary, others[0]
    if len(discovered) >= 2:
        return discovered[0], discovered[1]

    if seed_if_missing:
        for candidate in _discover_v0_hardware_for_model(model):
            if candidate != primary:
                if seed_scaled_hardware_alias(primary, candidate, model, variant):
                    return primary, candidate

    return primary, primary


def toggle_hardware(pair: Tuple[str, str], current: str) -> Tuple[str, str]:
    """Swap to the other hardware in a 2-GPU pair. Returns (old, new)."""
    other = pair[1] if current == pair[0] else pair[0]
    return current, other
