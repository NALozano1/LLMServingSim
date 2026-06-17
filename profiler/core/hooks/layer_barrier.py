"""Worker-side in-place layer barriers (runs inside vLLM TP worker process)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

_DEFAULT_SPIN_TIMEOUT_SEC = float(
    os.environ.get("DVFS_BARRIER_TIMEOUT_SEC", "600")
)


def discover_transformer_layers(
    model: nn.Module,
) -> list[tuple[int, str, nn.Module]]:
    """Return ``(idx, name, module)`` for each decoder block."""
    root = model
    if hasattr(model, "model"):
        root = model.model  # type: ignore[union-attr]

    layers = getattr(root, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
        return [(i, f"layers.{i}", m) for i, m in enumerate(layers)]

    decoder = getattr(root, "decoder", None)
    if decoder is not None:
        dec_layers = getattr(decoder, "layers", None)
        if isinstance(dec_layers, nn.ModuleList) and len(dec_layers) > 0:
            return [
                (i, f"decoder.layers.{i}", m)
                for i, m in enumerate(dec_layers)
            ]

    return []


class InPlaceLayerBarrier:
    """Post-forward hooks: sync GPU, signal host, block until DVFS ack."""

    def __init__(
        self,
        barrier_dir: str | Path,
        *,
        shot_id: str = "",
        spin_timeout_sec: float = _DEFAULT_SPIN_TIMEOUT_SEC,
    ) -> None:
        self.barrier_dir = Path(barrier_dir)
        self.barrier_dir.mkdir(parents=True, exist_ok=True)
        self.shot_id = shot_id
        self.spin_timeout_sec = spin_timeout_sec
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._barrier_seq = 0

    def install(self, model: nn.Module) -> int:
        """Register hooks; return number of layers instrumented."""
        self.remove()
        layers = discover_transformer_layers(model)
        if not layers:
            raise RuntimeError(
                "DVFS layer pause: no transformer layers found on model"
            )

        for layer_idx, layer_name, layer_mod in layers:
            handle = layer_mod.register_forward_hook(
                self._make_hook(layer_idx, layer_name)
            )
            self._handles.append(handle)
        return len(layers)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int, layer_name: str):
        def _hook(
            _module: nn.Module,
            _inputs: Any,
            _output: Any,
            *,
            _layer_idx: int = layer_idx,
            _layer_name: str = layer_name,
        ) -> None:
            torch.cuda.synchronize()
            self._wait_at_barrier(_layer_idx, _layer_name)

        return _hook

    def _wait_at_barrier(self, layer_idx: int, layer_name: str) -> None:
        pending = self.barrier_dir / "pending.json"
        ack = self.barrier_dir / "ack.json"

        if ack.exists():
            try:
                ack.unlink()
            except OSError:
                pass

        barrier_id = f"{self.shot_id}:{layer_idx}:{self._barrier_seq}"
        self._barrier_seq += 1

        payload = {
            "barrier_id": barrier_id,
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "shot_id": self.shot_id,
            "worker_ts": time.time(),
        }
        pending.write_text(json.dumps(payload), encoding="utf-8")

        deadline = time.monotonic() + self.spin_timeout_sec
        while not ack.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"DVFS barrier timeout after layer {layer_name} "
                    f"({self.spin_timeout_sec}s)"
                )
            time.sleep(0.0005)

        try:
            ack_data = json.loads(ack.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ack_data = {}

        if ack_data.get("barrier_id") not in (barrier_id, None):
            for _ in range(100):
                if ack.is_file():
                    try:
                        ack_data = json.loads(ack.read_text(encoding="utf-8"))
                        if ack_data.get("barrier_id") == barrier_id:
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(0.001)

        if pending.exists():
            pending.unlink()
        if ack.exists():
            ack.unlink()

        torch.cuda.synchronize()
