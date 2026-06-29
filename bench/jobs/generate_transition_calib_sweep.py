#!/usr/bin/env python3
"""Generate transition-effects calibration sweep specs.

Produces a campaign directory with one spec JSON per arm:
  - arm n_transitions=0        (baseline: no barriers, no clock changes)
  - arm n_transitions=N        for each N in TRANSITION_COUNTS

Each arm is repeated ITERATIONS times (independent runs with the same
workload so timing variance can be characterised).  The model and workload
settings mirror the existing 78-run campaign defaults so results are
comparable.

Usage:
    python3 bench/jobs/generate_transition_calib_sweep.py \\
        --repo /data/engs-glass/engs2950/DVFS-MoE/LLMServingSim \\
        --out-dir bench/campaigns/v100_transition_calib_<stamp> \\
        [--model qwen|<hf-id>] \\
        [--transition-counts 0,1,2,4,8,16,32] \\
        [--iterations 3] \\
        [--seed 20260624] \\
        [--apply-mode async|sync]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---- sweep defaults ----
_REPO = Path(__file__).resolve().parents[2]

# Model catalogue (same subset as 78-run campaign).
_MODELS: dict[str, dict] = {
    "qwen": {
        "hf_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "num_hidden_layers": 24,
        "max_num_batched_tokens": 1024,
        "max_model_len": 512,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.98,
    },
}

_DEFAULT_TRANSITION_COUNTS = [0, 1, 2, 4, 8, 16, 32]
_DEFAULT_ITERATIONS = 3
_DEFAULT_APPLY_MODE = "async"

# Fixed DVFS schedule used for every armed arm (matches existing campaign).
_FIXED_FREQS = [700, 900, 1100, 1300]


def _run_id(prefix: str, n: int, model_key: str, iteration: int) -> str:
    return f"{prefix}_n{n:03d}_{model_key}_i{iteration:02d}"


def _arm_label(n: int, apply_mode: str) -> str:
    if n == 0:
        return f"baseline_{apply_mode}"
    return f"n{n:03d}_{apply_mode}"


def _scatter_layers(
    rng: random.Random,
    n: int,
    num_layers: int,
) -> list[int]:
    """Pick n distinct layer indices uniformly at random without replacement."""
    pool = list(range(num_layers))
    return sorted(rng.sample(pool, min(n, len(pool))))


def generate_calib_sweep(
    *,
    repo: Path,
    out_dir: Path,
    model_key: str,
    transition_counts: list[int],
    iterations: int,
    apply_mode: str,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shared").mkdir(parents=True, exist_ok=True)

    cfg = _MODELS[model_key]
    model = cfg["hf_id"]
    num_layers = cfg["num_hidden_layers"]

    bench_common = {
        "prefill_only": True,
        "num_reqs": 1,
        "sps": 100,
        "seed": 42,
        "fix_input_length": 64,
        "fix_output_length": 0,
        "tick_seconds": 0.5,
        "dtype": "float16",
        "tp_size": 1,
    }

    run_specs: list[dict] = []

    for n in transition_counts:
        for iteration in range(iterations):
            run_id = _run_id("tcalib", n, model_key, iteration)
            arm = _arm_label(n, apply_mode)

            if n == 0:
                barrier_layers: list[int] = []
                freq_schedule: list[int] = []
            else:
                barrier_layers = _scatter_layers(rng, n, num_layers)
                # Cycle through fixed freqs for each barrier.
                freq_schedule = [
                    _FIXED_FREQS[i % len(_FIXED_FREQS)] for i in range(len(barrier_layers))
                ]

            spec: dict = {
                "run_id": run_id,
                "calib_sweep": "transition_effects",
                "arm_label": arm,
                "iteration": iteration,
                "model": model,
                "model_key": model_key,
                "num_hidden_layers": num_layers,
                "max_num_batched_tokens": cfg["max_num_batched_tokens"],
                "max_model_len": cfg["max_model_len"],
                "max_num_seqs": cfg["max_num_seqs"],
                "gpu_memory_utilization": cfg["gpu_memory_utilization"],
                "dvfs_apply_mode": apply_mode,
                "bench": bench_common,
                "dvfs": {
                    "n_transitions": n,
                    "barrier_layers": barrier_layers,
                    "freq_schedule_mhz": freq_schedule,
                },
            }
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            spec_path = run_dir / "calib_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
            run_specs.append(spec)

    # Manifest.
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "model_key": model_key,
        "model": model,
        "transition_counts": transition_counts,
        "iterations": iterations,
        "apply_mode": apply_mode,
        "seed": seed,
        "total_runs": len(run_specs),
        "runs": [
            {
                "run_id": s["run_id"],
                "arm_label": s["arm_label"],
                "n_transitions": s["dvfs"]["n_transitions"],
                "iteration": s["iteration"],
                "spec_path": str(runs_dir / s["run_id"] / "calib_spec.json"),
            }
            for s in run_specs
        ],
    }
    run_specs_path = out_dir / "run_specs.json"
    run_specs_path.write_text(
        json.dumps(run_specs, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Generated {len(run_specs)} calib runs → {out_dir}",
        file=sys.stderr,
    )
    print(f"  model={model}  transitions={transition_counts}  iters={iterations}  apply_mode={apply_mode}",
          file=sys.stderr)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default=str(_REPO),
        help="Repo root (default: auto-detected from script location).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Campaign output directory (created if absent).",
    )
    ap.add_argument(
        "--model",
        default="qwen",
        help="Model key (qwen) or a full HF model ID. "
             "Custom IDs use qwen defaults for engine settings. (default: qwen)",
    )
    ap.add_argument(
        "--transition-counts",
        default=",".join(str(x) for x in _DEFAULT_TRANSITION_COUNTS),
        help="Comma-separated list of N-transition arm sizes, must include 0 "
             "for the baseline. (default: 0,1,2,4,8,16,32)",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=_DEFAULT_ITERATIONS,
        help="Repeat each arm this many times for variance estimation. (default: 3)",
    )
    ap.add_argument(
        "--apply-mode",
        default=_DEFAULT_APPLY_MODE,
        choices=["async", "sync"],
        help="DVFS apply mode for armed arms (baseline always runs with no poller). "
             "(default: async)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260624,
        help="RNG seed for layer sampling. (default: 20260624)",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model key.
    model_key = args.model
    if model_key not in _MODELS:
        if "/" in model_key or model_key.startswith("microsoft") or model_key.startswith("Qwen"):
            # Treat as full HF id; map to qwen defaults.
            custom_id = model_key
            _MODELS["custom"] = {**_MODELS["qwen"], "hf_id": custom_id}
            model_key = "custom"
        else:
            ap.error(f"Unknown model key '{args.model}'. Use qwen, or a full HF model id.")

    transition_counts = [int(x.strip()) for x in args.transition_counts.split(",")]
    if 0 not in transition_counts:
        print("WARN: 0 not in --transition-counts; adding baseline arm.", file=sys.stderr)
        transition_counts = [0] + transition_counts

    generate_calib_sweep(
        repo=repo,
        out_dir=out_dir,
        model_key=model_key,
        transition_counts=sorted(set(transition_counts)),
        iterations=args.iterations,
        apply_mode=args.apply_mode,
        seed=args.seed,
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
