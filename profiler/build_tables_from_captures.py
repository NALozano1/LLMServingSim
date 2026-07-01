#!/usr/bin/env python3
"""Derive timing and power tables from captures.jsonl.

Reads the canonical ``captures.jsonl`` written by the profiler and produces:

  moe.csv           tokens, activated_experts, time_us, gating_ms, expert_ms, achieved_mhz
  dense.csv         layer, tokens, time_us, achieved_mhz
  attention.csv     prefill_chunk, kv_prefill, n_decode, kv_decode, time_us, achieved_mhz
  per_sequence.csv  sequences, time_us, achieved_mhz  (tokens field = num sequences)
  power_by_mhz.csv  mhz, watts, idle_watts, n_samples

This script is the *source of truth* for derived tables: power and latency are
always co-located in the same capture, so every row in every output table is
anchored to a real (latency, power, clock) triple.

Usage:
    python profiler/build_tables_from_captures.py captures.jsonl
    python profiler/build_tables_from_captures.py captures.jsonl --out-dir ./tables
    python profiler/build_tables_from_captures.py --scan ./perf/V100_1100MHz  # finds all captures.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# JSONL loading (no profiler import needed — reads plain dicts)
# ---------------------------------------------------------------------------

def _load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _avg(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})


# ---------------------------------------------------------------------------
# Per-category table builders
# ---------------------------------------------------------------------------

def _build_moe_csv(records: list[dict], out_dir: Path) -> Path:
    """tokens, activated_experts, time_us, gating_ms, expert_ms, achieved_mhz"""
    moe_recs = [r for r in records if r.get("category") == "moe"]
    # Bucket by (tokens, activated_experts)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in moe_recs:
        key = (r.get("tokens", 0), r.get("activated_experts", 0))
        buckets[key].append(r)

    rows = []
    for key in sorted(buckets):
        grp = buckets[key]
        rows.append({
            "tokens": key[0],
            "activated_experts": key[1],
            "time_us": _avg([r.get("latency_us") for r in grp]),
            "gating_ms": _avg([r.get("gating_ms") for r in grp]),
            "expert_ms": _avg([r.get("expert_ms") for r in grp]),
            "achieved_mhz": _avg([r.get("achieved_mhz") for r in grp]),
        })

    path = out_dir / "moe.csv"
    _write_csv(path, ["tokens", "activated_experts", "time_us", "gating_ms", "expert_ms", "achieved_mhz"], rows)
    return path


def _build_dense_csv(records: list[dict], out_dir: Path) -> Path:
    """layer, tokens, time_us, achieved_mhz"""
    dense_recs = [r for r in records if r.get("category") == "dense"]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in dense_recs:
        key = (r.get("layer", ""), r.get("tokens", 0))
        buckets[key].append(r)

    rows = []
    for key in sorted(buckets):
        grp = buckets[key]
        rows.append({
            "layer": key[0],
            "tokens": key[1],
            "time_us": _avg([r.get("latency_us") for r in grp]),
            "achieved_mhz": _avg([r.get("achieved_mhz") for r in grp]),
        })

    path = out_dir / "dense.csv"
    _write_csv(path, ["layer", "tokens", "time_us", "achieved_mhz"], rows)
    return path


def _build_attention_csv(records: list[dict], out_dir: Path) -> Path:
    """prefill_chunk, kv_prefill, n_decode, kv_decode, time_us, achieved_mhz.

    Attention CaptureRecords store the attention key as part of their layer
    field (formatted by categories.py).  Since rebuild from captures.jsonl
    doesn't have the 4-tuple directly in the identity, we store tokens as the
    sum of new tokens and layer = the timing-sample layer name.  Grouped by
    (layer, tokens).
    """
    attn_recs = [r for r in records if r.get("category") == "attention"]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in attn_recs:
        key = (r.get("layer", ""), r.get("tokens", 0))
        buckets[key].append(r)

    rows = []
    for key in sorted(buckets):
        grp = buckets[key]
        rows.append({
            "layer": key[0],
            "tokens": key[1],
            "time_us": _avg([r.get("latency_us") for r in grp]),
            "achieved_mhz": _avg([r.get("achieved_mhz") for r in grp]),
        })

    path = out_dir / "attention.csv"
    _write_csv(path, ["layer", "tokens", "time_us", "achieved_mhz"], rows)
    return path


def _build_per_sequence_csv(records: list[dict], out_dir: Path) -> Path:
    """sequences (stored in tokens field), time_us, achieved_mhz"""
    seq_recs = [r for r in records if r.get("category") == "per_sequence"]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in seq_recs:
        key = (r.get("layer", ""), r.get("tokens", 0))
        buckets[key].append(r)

    rows = []
    for key in sorted(buckets):
        grp = buckets[key]
        rows.append({
            "layer": key[0],
            "sequences": key[1],
            "time_us": _avg([r.get("latency_us") for r in grp]),
            "achieved_mhz": _avg([r.get("achieved_mhz") for r in grp]),
        })

    path = out_dir / "per_sequence.csv"
    _write_csv(path, ["layer", "sequences", "time_us", "achieved_mhz"], rows)
    return path


def _build_power_by_mhz_csv(records: list[dict], out_dir: Path) -> Path:
    """mhz (rounded to nearest 10 MHz), watts, idle_watts, n_samples.

    Groups by rounded achieved_mhz so minor boost jitter doesn't produce
    a separate row per shot.  Uses mean_power_w (active load power) from
    the record.  idle_watts is the pre-shot idle power at the locked clock.

    Column names match the policy consumer PowerTable in dvfs-policy/throughput.py
    which reads ``mhz``, ``watts``, and ``idle_watts``.
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        mhz = r.get("achieved_mhz")
        if mhz is None:
            continue
        bucket_mhz = int(round(float(mhz) / 10.0)) * 10
        buckets[bucket_mhz].append(r)

    rows = []
    for mhz_bucket in sorted(buckets):
        grp = buckets[mhz_bucket]
        rows.append({
            "mhz": mhz_bucket,
            "watts": _avg([r.get("mean_power_w") for r in grp]),
            "idle_watts": _avg([r.get("idle_power_w") for r in grp]),
            "n_samples": len(grp),
        })

    path = out_dir / "power_by_mhz.csv"
    _write_csv(path, ["mhz", "watts", "idle_watts", "n_samples"], rows)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_tables(captures_path: Path, out_dir: Path) -> dict[str, Path]:
    """Build all derived tables from one captures.jsonl.  Returns {name: path}."""
    records = _load_records(captures_path)
    if not records:
        print(f"WARNING: {captures_path} is empty or missing", file=sys.stderr)
        return {}

    categories = {r.get("category") for r in records}
    produced: dict[str, Path] = {}

    if "moe" in categories:
        produced["moe.csv"] = _build_moe_csv(records, out_dir)
    if "dense" in categories:
        produced["dense.csv"] = _build_dense_csv(records, out_dir)
    if "attention" in categories:
        produced["attention.csv"] = _build_attention_csv(records, out_dir)
    if "per_sequence" in categories:
        produced["per_sequence.csv"] = _build_per_sequence_csv(records, out_dir)

    # Power table is always built when any record has achieved_mhz.
    if any(r.get("achieved_mhz") is not None for r in records):
        produced["power_by_mhz.csv"] = _build_power_by_mhz_csv(records, out_dir)

    return produced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "captures", nargs="?",
        help="Path to captures.jsonl (required unless --scan is used)",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: same directory as captures.jsonl)")
    ap.add_argument("--scan", default=None, metavar="ROOT",
                    help="Recursively find all captures.jsonl under ROOT and build tables beside them")
    args = ap.parse_args()

    if args.scan:
        root = Path(args.scan)
        found = list(root.rglob("captures.jsonl"))
        if not found:
            print(f"No captures.jsonl found under {root}", file=sys.stderr)
            return 1
        for cp in found:
            out = Path(args.out_dir) if args.out_dir else cp.parent
            produced = build_tables(cp, out)
            for name, path in produced.items():
                print(f"  {path}")
        return 0

    if not args.captures:
        ap.print_help()
        return 1

    cp = Path(args.captures)
    if not cp.is_file():
        print(f"ERROR: {cp} not found", file=sys.stderr)
        return 1
    out = Path(args.out_dir) if args.out_dir else cp.parent
    produced = build_tables(cp, out)
    for name, path in produced.items():
        print(f"  {name} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
