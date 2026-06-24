"""Reference-workload TTFT / TPOT from profile CSVs (compute-only, pause-free).

Profile ``time_us`` values are CUDA kernel time and already exclude DVFS
barrier idle — they are the effective compute basis for serving estimates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _probe_moe_top_k(model_config: dict[str, Any]) -> int | None:
    for key in (
        "num_experts_per_tok",
        "num_experts_per_token",
        "moe_k",
    ):
        if key in model_config:
            return int(model_config[key])
    return None


def _ref_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else default


def reference_workload_tokens() -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for synthetic latency."""
    return (
        _ref_int("PROFILER_REF_INPUT_TOKENS", 128),
        _ref_int("PROFILER_REF_OUTPUT_TOKENS", 128),
    )


@dataclass
class _MiniBatch:
    """Minimal batch view for trace_generator's ``_build_batch_ctx``."""

    prefill_q_list: list[int]
    prefill_k_list: list[int]
    decode_k_list: list[int]
    num_prefill: int
    num_decode: int
    requests: list[object] = field(default_factory=list)

    @property
    def total_len(self) -> int:
        return sum(self.prefill_q_list) + len(self.decode_k_list)


def _load_meta(variant_root: Path) -> dict[str, Any]:
    meta_path = variant_root / "meta.yaml"
    if not meta_path.is_file():
        return {}
    return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}


def _load_perf_db_tp_dir(
    tp_dir: Path,
    architecture: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a single-TP perf_db from on-disk CSVs under ``tp_dir``."""
    from serving.core.trace_generator import (
        _build_1d_table,
        _build_attention_table,
        _build_moe_table,
        _read_category_csv,
    )

    if not tp_dir.is_dir():
        return None

    tables: dict[str, Any] = {}
    dense_df = _read_category_csv(str(tp_dir / "dense.csv"), None)
    if dense_df is not None:
        tables["dense"] = _build_1d_table(dense_df, "layer", "tokens")

    per_seq_df = _read_category_csv(str(tp_dir / "per_sequence.csv"), None)
    if per_seq_df is not None:
        tables["per_sequence"] = _build_1d_table(per_seq_df, "layer", "sequences")

    attn_df = _read_category_csv(str(tp_dir / "attention.csv"), None)
    if attn_df is not None:
        tables["attention"] = _build_attention_table(attn_df)

    moe_df = _read_category_csv(str(tp_dir / "moe.csv"), None)
    if moe_df is not None:
        tables["moe"] = _build_moe_table(moe_df)

    if not tables:
        return None

    variant_root = tp_dir.parent
    meta = meta if meta is not None else _load_meta(variant_root)
    tp = int(tp_dir.name[2:]) if tp_dir.name.startswith("tp") else 1

    return {
        "meta": meta,
        "architecture": architecture,
        "variant": meta.get("variant", variant_root.name),
        "hardware": meta.get("hardware", variant_root.parent.parent.name),
        "model": meta.get("model", variant_root.parent.name),
        "available_tps": [tp],
        "tables": {tp: tables},
    }


def _make_ctx(
    perf_db: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tp: int,
    moe_top_k: int | None,
):
    from serving.core.trace_generator import TraceCtx

    n_embd = int(model_config["hidden_size"])
    n_head = int(model_config["num_attention_heads"])
    kv_head = int(model_config.get("num_key_value_heads", n_head))
    head_dim = int(model_config.get("head_dim", n_embd // n_head))
    is_moe = bool(moe_top_k and moe_top_k > 0)

    return TraceCtx(
        hardware=perf_db["hardware"],
        model=perf_db["model"],
        config=model_config,
        perf_db=perf_db,
        node_id=0,
        fp=2,
        placement={},
        gate=None,
        enable_attn_offloading=False,
        power_model=None,
        pim_model=None,
        pim_channels=0,
        n_head=n_head,
        kv_head=kv_head,
        head_dim=head_dim,
        is_moe=is_moe,
        pd_type=None,
        tp_size=tp,
        pp_size=1,
        local_ep=1,
        ep_total=1,
        tp_dim=None,
        ep_dim=None,
        dp_sum_total_len=0,
    )


def _batch_prefill(total_tokens: int, kv_prefill: int = 0) -> _MiniBatch:
    return _MiniBatch(
        prefill_q_list=[total_tokens],
        prefill_k_list=[kv_prefill],
        decode_k_list=[],
        num_prefill=1,
        num_decode=0,
        requests=[object()],
    )


def _batch_decode(kv_len: int) -> _MiniBatch:
    return _MiniBatch(
        prefill_q_list=[],
        prefill_k_list=[],
        decode_k_list=[kv_len],
        num_prefill=0,
        num_decode=1,
        requests=[object()],
    )


def _section_latency_ns(ctx, bctx, section: str) -> int:
    from serving.core.trace_generator import (
        _layer_category,
        _layer_latency_for_power,
        _lookup_moe,
        _sequence,
    )

    total = 0
    for layer_name in _sequence(ctx.perf_db, section):
        if layer_name in ("rotary_emb", "moe"):
            continue
        if _layer_category(ctx.perf_db, layer_name) is None:
            continue
        total += int(_layer_latency_for_power(ctx, bctx, layer_name))

    if section == "mlp_moe" and ctx.is_moe:
        top_k = max(int(ctx.config.get("_moe_top_k") or 1), 1)
        total += int(_lookup_moe(ctx.perf_db, bctx.total_len, top_k))
    return total


def _block_latency_ns(ctx, bctx) -> int:
    total = 0
    for section in ("pre_attn", "post_attn", "mlp_dense", "mlp_moe"):
        total += _section_latency_ns(ctx, bctx, section)
    return max(1, total)


def _forward_latency_ns(ctx, batch: _MiniBatch, num_layers: int) -> int:
    from serving.core.trace_generator import _build_batch_ctx

    bctx = _build_batch_ctx(batch, ctx)
    prologue = _section_latency_ns(ctx, bctx, "prologue")
    head = _section_latency_ns(ctx, bctx, "head")
    block = _block_latency_ns(ctx, bctx)
    return prologue + block * num_layers + head


def _prefill_chunks(
    input_tokens: int,
    max_chunk: int | None,
) -> list[tuple[int, int]]:
    if input_tokens <= 0:
        return []
    chunk = max_chunk if max_chunk and max_chunk > 0 else input_tokens
    chunks: list[tuple[int, int]] = []
    remaining = input_tokens
    kv = 0
    while remaining > 0:
        take = min(chunk, remaining)
        chunks.append((take, kv))
        kv += take
        remaining -= take
    return chunks


def _has_required_csvs(tp_dir: Path, is_moe: bool) -> bool:
    needed = ["dense.csv", "attention.csv", "per_sequence.csv"]
    if is_moe:
        needed.append("moe.csv")
    return all((tp_dir / name).is_file() for name in needed)


def synthesize_reference_latency(
    tp_dir: Path,
    architecture: dict[str, Any],
    model_config: dict[str, Any],
    *,
    tp: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, Any] | None:
    """Estimate TTFT / TPOT / e2e for an isolated single-request workload."""
    if input_tokens is None or output_tokens is None:
        ref_in, ref_out = reference_workload_tokens()
        input_tokens = ref_in if input_tokens is None else input_tokens
        output_tokens = ref_out if output_tokens is None else output_tokens

    input_tokens = max(int(input_tokens), 0)
    output_tokens = max(int(output_tokens), 0)

    moe_top_k = _probe_moe_top_k(model_config)
    if not _has_required_csvs(tp_dir, bool(moe_top_k)):
        return None

    perf_db = _load_perf_db_tp_dir(tp_dir, architecture)
    if perf_db is None:
        return None

    cfg = dict(model_config)
    if moe_top_k:
        cfg["_moe_top_k"] = moe_top_k

    ctx = _make_ctx(perf_db, cfg, tp=tp, moe_top_k=moe_top_k)
    num_layers = int(model_config.get("num_hidden_layers") or 1)

    meta = perf_db.get("meta") or {}
    eff = meta.get("engine_effective") or {}
    max_chunk = eff.get("max_num_batched_tokens")

    ttft_ns = 0
    for chunk, kv in _prefill_chunks(input_tokens, max_chunk):
        ttft_ns += _forward_latency_ns(ctx, _batch_prefill(chunk, kv), num_layers)

    decode_step_ns: list[int] = []
    if output_tokens > 1:
        from serving.core.trace_generator import _build_batch_ctx

        prologue_ref = _section_latency_ns(
            ctx,
            _build_batch_ctx(_batch_decode(max(input_tokens, 1)), ctx),
            "prologue",
        )
        for i in range(1, output_tokens):
            kv = input_tokens + i - 1
            full = _forward_latency_ns(ctx, _batch_decode(kv), num_layers)
            decode_step_ns.append(max(1, full - prologue_ref))

    decode_total_ns = sum(decode_step_ns)
    e2e_ns = ttft_ns + decode_total_ns
    tpot_ns = (
        decode_total_ns // max(len(decode_step_ns), 1)
        if decode_step_ns
        else 0
    )

    def _sec(ns: int) -> float:
        return round(ns / 1e9, 6)

    def _ms(ns: int) -> float:
        return round(ns / 1e6, 3)

    return {
        "reference_workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_hidden_layers": num_layers,
            "tp": tp,
            "isolated_single_request": True,
            "chunked_prefill": bool(max_chunk and input_tokens > max_chunk),
            "max_num_batched_tokens": max_chunk,
        },
        "ttft_sec": _sec(ttft_ns),
        "ttft_ms": _ms(ttft_ns),
        "tpot_sec": _sec(tpot_ns),
        "tpot_ms": _ms(tpot_ns),
        "itl_sec": _sec(tpot_ns),
        "itl_ms": _ms(tpot_ns),
        "e2e_latency_sec": _sec(e2e_ns),
        "e2e_latency_ms": _ms(e2e_ns),
        "decode_steps": len(decode_step_ns),
        "effective_basis": "profile_csv_kernel_time_us",
    }
