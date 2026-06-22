"""Scattered layer-boundary DVFS schedule for segmented forward simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serving.core.v100_measured_assets import mhz_to_hardware


def layer_index_for_completed_stage(stage_idx: int, num_hidden_layers: int) -> int | None:
    """Return decoder layer index completed by *stage_idx*, or None for bookends."""
    if stage_idx <= 0 or stage_idx > num_hidden_layers:
        return None
    return stage_idx - 1


@dataclass(frozen=True)
class LayerDvfsSchedule:
    barrier_to_mhz: dict[int, int]
    default_hardware: str = "V100"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerDvfsSchedule:
        dvfs = data.get("dvfs") or data
        layers = dvfs.get("barrier_layers") or []
        freqs = dvfs.get("freq_schedule_mhz") or []
        if len(layers) != len(freqs):
            raise ValueError(
                f"barrier_layers ({len(layers)}) and freq_schedule_mhz ({len(freqs)}) length mismatch"
            )
        mapping = {int(layer): int(mhz) for layer, mhz in zip(layers, freqs)}
        default_hw = data.get("default_hardware") or dvfs.get("default_hardware") or "V100"
        return cls(barrier_to_mhz=mapping, default_hardware=default_hw)

    @classmethod
    def from_path(cls, path: str | Path) -> LayerDvfsSchedule:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def hardware_for_layer_barrier(self, layer_idx: int) -> str | None:
        mhz = self.barrier_to_mhz.get(layer_idx)
        if mhz is None:
            return None
        return mhz_to_hardware(mhz, default_tag=self.default_hardware)

    def hardware_after_stage(self, stage_idx: int, num_hidden_layers: int) -> str | None:
        layer_idx = layer_index_for_completed_stage(stage_idx, num_hidden_layers)
        if layer_idx is None:
            return None
        return self.hardware_for_layer_barrier(layer_idx)
