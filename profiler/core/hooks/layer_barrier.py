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


def _decode_token_threshold() -> int:
    """Forwards with at most this many tokens are treated as decode passes."""
    return max(1, int(os.environ.get("DVFS_DECODE_TOKEN_THRESHOLD", "4")))


def _prefill_min_tokens() -> int:
    """Forwards with at least this many tokens are always treated as prefill."""
    return max(1, int(os.environ.get("DVFS_PREFILL_MIN_TOKENS", "8")))


def _decode_max_pauses_per_pass() -> int:
    return max(0, int(os.environ.get("DVFS_DECODE_MAX_PAUSES_PER_PASS", "1")))


def _forward_num_tokens(inputs: Any) -> int:
    if not inputs:
        return 0
    hs = inputs[0]
    if hasattr(hs, "shape") and len(hs.shape) >= 1:
        return int(hs.shape[0])
    return 0


def _skip_cuda_sync() -> bool:
    return os.environ.get("DVFS_BARRIER_SKIP_CUDA_SYNC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _cuda_barrier_sync() -> None:
    if _skip_cuda_sync():
        return
    stream = torch.cuda.current_stream()
    stream.synchronize()


def _barrier_layer_allowlist() -> set[int] | None:
    """When set, only these layer indices trigger DVFS barriers."""
    raw = os.environ.get("DVFS_BARRIER_LAYERS", "").strip()
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out or None


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
        self._barrier_wait_sec = 0.0
        self._pass_decode_pauses = 0

    @property
    def barrier_wait_sec(self) -> float:
        return self._barrier_wait_sec

    def install(self, model: nn.Module) -> int:
        """Register hooks; return number of layers instrumented."""
        self.remove()
        layers = discover_transformer_layers(model)
        if not layers:
            raise RuntimeError(
                "DVFS layer pause: no transformer layers found on model"
            )

        allow = _barrier_layer_allowlist()
        installed = 0
        for layer_idx, layer_name, layer_mod in layers:
            if allow is not None and layer_idx not in allow:
                continue
            handle = layer_mod.register_forward_hook(
                self._make_hook(layer_idx, layer_name)
            )
            self._handles.append(handle)
            installed += 1
        if allow is not None and installed == 0:
            raise RuntimeError(
                "DVFS layer pause: DVFS_BARRIER_LAYERS matched no model layers"
            )
        return installed

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset_barrier_stats(self) -> None:
        self._barrier_wait_sec = 0.0
        self._barrier_seq = 0
        self._pass_decode_pauses = 0

    def _is_decode_pass(self, inputs: Any) -> bool:
        num_tokens = _forward_num_tokens(inputs)
        if num_tokens >= _prefill_min_tokens():
            return False
        return num_tokens > 0 and num_tokens <= _decode_token_threshold()

    def _should_barrier(
        self,
        layer_idx: int,
        inputs: Any,
        *,
        allow: set[int] | None,
    ) -> bool:
        if layer_idx == 0:
            self._pass_decode_pauses = 0

        if not self._is_decode_pass(inputs):
            if allow is not None and layer_idx not in allow:
                return False
            return True

        max_pauses = _decode_max_pauses_per_pass()
        if max_pauses <= 0:
            return False
        if self._pass_decode_pauses >= max_pauses:
            return False
        if allow is not None and layer_idx not in allow:
            return False
        if allow is None and layer_idx != 0:
            return False
        self._pass_decode_pauses += 1
        return True

    def _make_hook(self, layer_idx: int, layer_name: str):
        allow = _barrier_layer_allowlist()

        def _hook(
            _module: nn.Module,
            inputs: Any,
            _output: Any,
            *,
            _layer_idx: int = layer_idx,
            _layer_name: str = layer_name,
        ) -> None:
            if not self._should_barrier(_layer_idx, inputs, allow=allow):
                return
            _cuda_barrier_sync()
            self._wait_at_barrier(_layer_idx, _layer_name)

        return _hook

    def _wait_at_barrier(self, layer_idx: int, layer_name: str) -> None:
        wait_t0 = time.perf_counter()
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

        self._barrier_wait_sec += time.perf_counter() - wait_t0
        _cuda_barrier_sync()
