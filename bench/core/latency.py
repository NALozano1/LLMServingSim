"""Latency extraction from real vLLM bench requests.jsonl."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _same_time_domain(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) < 100_000


def _arrival_ts(req: dict[str, Any]) -> float | None:
    arr = req.get("arrival_time")
    queued = req.get("queued_ts")
    first = req.get("first_token_ts")
    last = req.get("last_token_ts")

    if queued is None:
        return arr
    if arr is None:
        return queued
    if _same_time_domain(arr, first) and _same_time_domain(arr, last):
        return arr
    if _same_time_domain(queued, first) or _same_time_domain(queued, last):
        return queued
    return arr


def per_request_latencies_ms(req: dict[str, Any]) -> dict[str, Any] | None:
    arr = _arrival_ts(req)
    first = req.get("first_token_ts")
    last = req.get("last_token_ts")
    if arr is None or first is None or last is None:
        return None

    out_toks = max(1, int(req.get("output_toks", 1)))
    ttft_ms = (float(first) - float(arr)) * 1000.0
    e2e_ms = (float(last) - float(arr)) * 1000.0
    tpot_ms = (
        (float(last) - float(first)) / (out_toks - 1) * 1000.0
        if out_toks > 1
        else 0.0
    )
    return {
        "request_id": req.get("request_id"),
        "input_toks": int(req.get("input_toks", 0)),
        "output_toks": out_toks,
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p99_ms": None,
        }
    vals = sorted(values)
    n = len(vals)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[idx]

    return {
        "count": n,
        "mean_ms": round(statistics.mean(vals), 3),
        "median_ms": round(statistics.median(vals), 3),
        "p90_ms": round(_pct(0.90), 3),
        "p99_ms": round(_pct(0.99), 3),
    }


def extract_from_bench_dir(bench_dir: Path) -> dict[str, Any]:
    req_path = bench_dir / "requests.jsonl"
    if not req_path.is_file():
        raise FileNotFoundError(f"missing {req_path}")

    per_req: list[dict[str, Any]] = []
    ttft_vals: list[float] = []
    tpot_vals: list[float] = []
    e2e_vals: list[float] = []

    for line in req_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        lat = per_request_latencies_ms(rec)
        if lat is None:
            continue
        per_req.append(lat)
        ttft_vals.append(lat["ttft_ms"])
        if lat["output_toks"] > 1:
            tpot_vals.append(lat["tpot_ms"])
        e2e_vals.append(lat["e2e_ms"])

    meta: dict[str, Any] = {}
    if (bench_dir / "meta.json").is_file():
        meta = json.loads((bench_dir / "meta.json").read_text(encoding="utf-8"))
    node_meta: dict[str, Any] = {}
    if (bench_dir / "node_meta.json").is_file():
        node_meta = json.loads((bench_dir / "node_meta.json").read_text(encoding="utf-8"))

    return {
        "source": "vllm_bench_requests_jsonl",
        "bench_dir": str(bench_dir),
        "meta": meta,
        "node_meta": node_meta,
        "ttft": _stats(ttft_vals),
        "tpot": _stats(tpot_vals),
        "e2e_latency": _stats(e2e_vals),
        "per_request": per_req,
    }
