"""Query per-GPU layer-wise work state at a simulated timestamp."""

import argparse
import csv
import json
import sys

from serving.core.work_state import format_state_report, load_events, reconstruct_at


def main():
    parser = argparse.ArgumentParser(
        prog="python -m serving.tools.query_work_state",
        description="Reconstruct layer-wise work availability at simulated time T",
    )
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--at-ns", type=int, required=True)
    parser.add_argument("--npu", type=int, default=None)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    parser.add_argument("--timeseries", action="store_true")
    parser.add_argument("--step-ns", type=int, default=1_000_000_000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    events = load_events(args.events)
    if not events:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    if args.timeseries:
        max_ts = max(e.get("ts_ns", 0) for e in events)
        rows = []
        ts = 0
        while ts <= max_ts:
            state = reconstruct_at(events, ts)
            for npu_key, npu in state.get("npus", {}).items():
                rows.append({
                    "ts_ns": ts, "npu_id": int(npu_key),
                    "instance_id": npu.get("instance_id"), "status": npu.get("status"),
                    "batch_id": npu.get("batch_id"), "layers_done": npu.get("layers_done", 0),
                    "layers_total": npu.get("layers_total", 0),
                    "layers_remaining": npu.get("layers_remaining", 0),
                })
            ts += args.step_ns
        out = open(args.output, "w", newline="") if args.output else sys.stdout
        if rows:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if args.output:
            out.close()
            print(f"Wrote {len(rows)} rows to {args.output}")
        return

    state = reconstruct_at(events, args.at_ns)
    if args.format == "json":
        print(json.dumps(state, indent=2))
    else:
        print(format_state_report(state, npu_id=args.npu))


if __name__ == "__main__":
    main()
