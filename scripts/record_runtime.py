#!/usr/bin/env python3
"""Append a wall-clock runtime record to calibration/runtimes/runs.jsonl."""

from __future__ import annotations

import argparse
import sys

from serving.core.runtime_calibration import (
    append_runtime_record,
    build_runtime_record,
    parse_wall_time,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--wall-time", required=True, help="seconds or 0h 1m 40.547s")
    parser.add_argument("--first-log-s", type=float, default=None)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dp-group", default=None)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hardware", default="RTXPRO6000")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset", default="workloads/example_trace.jsonl")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--host", default=None)
    parser.add_argument("--status", default="completed", choices=["completed", "failed", "partial"])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    gpus_per_instance = args.gpus // args.instances
    instances = [
        {
            "num_npus": gpus_per_instance,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "ep_size": args.ep_size,
            "dp_group": args.dp_group,
        }
        for _ in range(args.instances)
    ]

    record = build_runtime_record(
        cluster_config=args.cluster_config,
        instances=instances,
        wall_time_s=parse_wall_time(args.wall_time),
        first_log_wall_s=args.first_log_s,
        model=args.model,
        hardware=args.hardware,
        dtype=args.dtype,
        dataset=args.dataset,
        num_requests=args.num_requests,
        status=args.status,
        notes=args.notes,
        host=args.host,
    )

    path = append_runtime_record(record)
    print(f"Appended to {path}")
    print(f"  wall_time_s={record['wall_time_s']:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
