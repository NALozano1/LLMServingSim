#!/usr/bin/env python3
"""Extract TTFT / TPOT / e2e latency from real vLLM bench ``requests.jsonl``.

These metrics come from ``python -m bench run`` (live AsyncLLM), not from
LLMServingSim profile synthesis or the discrete-event simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.core.latency import extract_from_bench_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench_dir", type=Path, help="bench run output directory")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write JSON summary (default: <bench_dir>/real_latency.json)",
    )
    args = p.parse_args()

    summary = extract_from_bench_dir(args.bench_dir)
    out = args.output or (args.bench_dir / "real_latency.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ttft = summary["ttft"]
    tpot = summary["tpot"]
    print(f"wrote {out}")
    print(
        f"REAL TTFT mean={ttft.get('mean_ms')}ms median={ttft.get('median_ms')}ms "
        f"(n={ttft.get('count')})"
    )
    print(
        f"REAL TPOT mean={tpot.get('mean_ms')}ms median={tpot.get('median_ms')}ms "
        f"(n={tpot.get('count')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
