#!/usr/bin/env python3
"""Audit profiler GPU-clock captures vs the frequency they were *supposed* to be locked at.

Each profiler run writes per-shot power/clock samples to
``profiler/perf/<hw>/<model>/<variant>/tp<N>/gpu_power/*.jsonl`` with lines like::

    {"wall_ts": ..., "gpus": [{"index": 0, "power_w": 147.0, "graphics_mhz": 1440, "util_gpu_pct": 7.0}]}

The target clock is encoded in the hardware tag (``V100_1100MHz`` -> 1100). The bare
``V100`` tag means default boost (no fixed target) and is reported but not flagged.

A run is FLAGGED when the lock did not hold: either the median busy-sample clock is
more than ``--tolerance`` MHz off target, or fewer than ``--min-in-range`` percent of
busy samples sit within tolerance.

Usage:
    python3 profiler/jobs/audit_gpu_clocks.py                 # scan profiler/perf
    python3 profiler/jobs/audit_gpu_clocks.py profiler/perf/V100_1100MHz
    python3 profiler/jobs/audit_gpu_clocks.py --by-shot       # per-shot breakdown
    python3 profiler/jobs/audit_gpu_clocks.py --busy-util 20  # only count util>=20% samples

Exit code is non-zero if any run is flagged, so it works as a pre-import gate.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_HW_MHZ_RE = re.compile(r"_(\d+)MHz$")


def target_mhz_from_tag(hw_tag: str) -> int | None:
    """1100 for 'V100_1100MHz'; None for bare 'V100' (default boost / no fixed target)."""
    m = _HW_MHZ_RE.search(hw_tag)
    return int(m.group(1)) if m else None


def _iter_samples(jsonl_path: Path, gpu_index: int | None):
    """Yield (graphics_mhz, util_pct) from one capture file, tolerating partial lines."""
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line, etc.
        for gpu in rec.get("gpus", []):
            if gpu_index is not None and gpu.get("index") != gpu_index:
                continue
            mhz = gpu.get("graphics_mhz")
            if mhz is None:
                continue
            yield float(mhz), float(gpu.get("util_gpu_pct", 0.0) or 0.0)


def _pct(values, q):
    """Simple percentile (q in [0,100]) on a sorted-able list; stdlib only."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _summarize(clocks, target, tol):
    n = len(clocks)
    if n == 0:
        return None
    in_range = (
        sum(1 for c in clocks if abs(c - target) <= tol) / n * 100.0
        if target is not None
        else None
    )
    return {
        "n": n,
        "median": statistics.median(clocks),
        "p05": _pct(clocks, 5),
        "p95": _pct(clocks, 95),
        "min": min(clocks),
        "max": max(clocks),
        "in_range_pct": in_range,
    }


def find_capture_dirs(roots):
    """Yield every gpu_power/ dir under the given roots (or matching a run dir directly)."""
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"WARN: path not found: {root}", file=sys.stderr)
            continue
        candidates = [root] if root.name == "gpu_power" else sorted(root.rglob("gpu_power"))
        for d in candidates:
            if d.is_dir() and d not in seen:
                seen.add(d)
                yield d


def hw_tag_for(capture_dir: Path) -> str:
    """Recover the profiler hardware tag (e.g. V100_1100MHz) from a gpu_power/ path.

    Layout: .../perf/<hw>/<org>/<model>/<variant>/tp<N>/gpu_power
    The hardware tag is the path component directly under a 'perf' directory.
    """
    parts = capture_dir.parts
    if "perf" in parts:
        i = parts.index("perf")
        if i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: first ancestor that looks like a hardware tag.
    for p in capture_dir.parents:
        if _HW_MHZ_RE.search(p.name) or p.name.startswith("V100") or p.name.startswith("A"):
            return p.name
    return capture_dir.parent.name


def audit(args) -> int:
    flagged = 0
    runs = 0
    rows = []
    for cap_dir in find_capture_dirs(args.paths):
        hw = hw_tag_for(cap_dir)
        target = target_mhz_from_tag(hw)
        run_label = "/".join(cap_dir.parts[max(0, len(cap_dir.parts) - 6):-1])
        runs += 1

        all_clocks, busy_clocks = [], []
        per_shot = []
        for jf in sorted(cap_dir.glob("*.jsonl")):
            shot_clocks = []
            for mhz, util in _iter_samples(jf, args.gpu_index):
                all_clocks.append(mhz)
                shot_clocks.append(mhz)
                if util >= args.busy_util:
                    busy_clocks.append(mhz)
            if args.by_shot and shot_clocks:
                per_shot.append((jf.stem, shot_clocks))

        basis = busy_clocks if busy_clocks else all_clocks
        summ = _summarize(basis, target, args.tolerance)
        if summ is None:
            rows.append((run_label, hw, target, None, False, "no samples"))
            continue

        is_flagged = False
        reasons = []
        if target is not None:
            if abs(summ["median"] - target) > args.tolerance:
                is_flagged = True
                reasons.append(f"median {summ['median']:.0f} off target by {summ['median']-target:+.0f}")
            if summ["in_range_pct"] is not None and summ["in_range_pct"] < args.min_in_range:
                is_flagged = True
                reasons.append(f"only {summ['in_range_pct']:.0f}% in +/-{args.tolerance}")
        else:
            reasons.append("default boost (no target)")
        if is_flagged:
            flagged += 1

        rows.append((run_label, hw, target, summ, is_flagged, "; ".join(reasons)))

        if args.by_shot:
            for shot_name, shot_clocks in per_shot:
                s = _summarize(shot_clocks, target, args.tolerance)
                if s is None:
                    continue
                off = (
                    target is not None
                    and (abs(s["median"] - target) > args.tolerance)
                )
                mark = "  !!" if off else "    "
                rows.append(
                    (f"{mark}  {shot_name}", "", target, s, off, "" if not off else "shot off-target")
                )

    _print_table(rows, args)
    print(f"\nRuns audited: {runs}    FLAGGED: {flagged}", file=sys.stderr)
    if args.json:
        _dump_json(rows)
    return 1 if flagged else 0


def _fmt(v, width=6):
    return f"{v:>{width}.0f}" if isinstance(v, (int, float)) else f"{'':>{width}}"


def _print_table(rows, args):
    hdr = (
        f"{'RUN':<46} {'TARGET':>6} {'MEDIAN':>6} {'P05':>6} {'P95':>6} "
        f"{'MIN':>6} {'MAX':>6} {'IN±'+str(args.tolerance):>7} {'':>4} REASON"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, hw, target, summ, flagged, reason in rows:
        if summ is None:
            print(f"{label[:46]:<46} {_fmt(target)}    (no samples)")
            continue
        flag = "FLAG" if flagged else "ok"
        inr = f"{summ['in_range_pct']:.0f}%" if summ.get("in_range_pct") is not None else "-"
        print(
            f"{label[:46]:<46} {_fmt(target)} {_fmt(summ['median'])} "
            f"{_fmt(summ['p05'])} {_fmt(summ['p95'])} {_fmt(summ['min'])} {_fmt(summ['max'])} "
            f"{inr:>7} {flag:>4} {reason}"
        )


def _dump_json(rows):
    out = []
    for label, hw, target, summ, flagged, reason in rows:
        out.append(
            {"run": label, "hardware": hw, "target_mhz": target,
             "summary": summ, "flagged": flagged, "reason": reason}
        )
    Path("gpu_clock_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Wrote gpu_clock_audit.json", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["profiler/perf"],
                    help="profile roots or gpu_power dirs to scan (default: profiler/perf)")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="MHz tolerance around the target lock (default: 100)")
    ap.add_argument("--busy-util", type=float, default=10.0,
                    help="only count samples with util_gpu_pct >= this when busy samples exist (default: 10)")
    ap.add_argument("--min-in-range", type=float, default=80.0,
                    help="flag if fewer than this %% of busy samples are within tolerance (default: 80)")
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="GPU index to audit; use -1 for all GPUs (default: 0)")
    ap.add_argument("--by-shot", action="store_true",
                    help="also print a per-shot breakdown (reveals per-kernel throttling)")
    ap.add_argument("--json", action="store_true", help="also write gpu_clock_audit.json")
    args = ap.parse_args()
    if not args.paths:
        args.paths = ["profiler/perf"]
    if args.gpu_index < 0:
        args.gpu_index = None
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
