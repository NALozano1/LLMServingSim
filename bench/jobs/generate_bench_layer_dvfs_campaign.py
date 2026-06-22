#!/usr/bin/env python3
"""Generate bench layer-boundary DVFS campaign (same permutations as profiler campaign)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse permutation definitions from the profiler campaign generator.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "profiler" / "jobs"))
from generate_full_dvfs_campaign import (  # noqa: E402
    ITERATIONS,
    MODELS,
    NUM_SCATTERED_PERMUTATIONS,
    VALID_FREQS_MHZ,
    FIXED_FREQS_MHZ,
    _make_fixed_spec,
    _make_scattered_perm,
    _run_id,
    _yaml_dump,
)


def generate_bench_campaign(*, repo: Path, out_dir: Path, seed: int) -> dict:
    import random

    rng = random.Random(seed)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    permutations: list[dict] = []
    for perm_idx in range(1, NUM_SCATTERED_PERMUTATIONS + 1):
        for model_key in ("phi", "qwen"):
            permutations.append(_make_scattered_perm(rng, model_key, perm_idx))

    run_specs: list[dict] = []

    bench_common = {
        "bench": {
            "prefill_only": True,
            "num_reqs": 1,
            "sps": 100,
            "seed": 42,
            "fix_input_length": 64,
            "fix_output_length": 0,
            "tick_seconds": 0.5,
            "dtype": "float16",
            "tp_size": 1,
        },
        "engine": {
            "api": "LLM",
            "vllm_use_v1": False,
            "worker_extension_cls": "profiler.core.hooks.extension.Extension",
        },
    }

    for perm in permutations:
        for iteration in range(ITERATIONS):
            run_id = _run_id("scat", perm["perm_id"], perm["model_key"], iteration)
            cfg = MODELS[perm["model_key"]]
            spec: dict = {
                "run_id": run_id,
                "campaign_mode": "scattered",
                "iteration": iteration,
                "model": perm["model"],
                "model_key": perm["model_key"],
                "num_hidden_layers": perm["num_hidden_layers"],
                "max_num_batched_tokens": cfg["max_num_batched_tokens"],
                "max_num_seqs": cfg["max_num_seqs"],
                "max_model_len": 512,
                "gpu_memory_utilization": 0.92,
                "dvfs": {
                    "barrier_layers": perm["barrier_layers"],
                    "freq_schedule_mhz": perm["freq_schedule_mhz"],
                    "freq_at_layer": perm["freq_at_layer"],
                },
                "llmservingsim": perm["llmservingsim"],
                **bench_common,
            }
            run_specs.append(spec)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_spec.json").write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
            sim = {
                "run_id": run_id,
                "model": perm["model"],
                "mode": "scattered_dvfs_prefill",
                "iteration": iteration,
                "num_hidden_layers": perm["num_hidden_layers"],
                "dvfs": spec["dvfs"],
                "llmservingsim": {
                    **perm["llmservingsim"],
                    "hardware_profiles": [
                        {
                            "mhz": hp["mhz"],
                            "profile_path": str(repo / hp["profile_path"]),
                        }
                        for hp in perm["llmservingsim"]["hardware_profiles"]
                    ],
                },
                "bench": spec["bench"],
            }
            (run_dir / "sim_replication.json").write_text(
                json.dumps(sim, indent=2) + "\n", encoding="utf-8"
            )

    for model_key in ("phi", "qwen"):
        for mhz in FIXED_FREQS_MHZ:
            for iteration in range(ITERATIONS):
                fixed = _make_fixed_spec(model_key, mhz, iteration)
                run_id = _run_id("fixed", f"{mhz}", model_key, iteration)
                cfg = MODELS[model_key]
                spec = {
                    "run_id": run_id,
                    "campaign_mode": "fixed",
                    "iteration": iteration,
                    "model": fixed["model"],
                    "model_key": model_key,
                    "num_hidden_layers": fixed["num_hidden_layers"],
                    "max_num_batched_tokens": cfg["max_num_batched_tokens"],
                    "max_num_seqs": cfg["max_num_seqs"],
                    "max_model_len": 512,
                    "gpu_memory_utilization": 0.92,
                    "dvfs": {
                        "gpu_freq_mhz": mhz,
                        "freq_schedule_mhz": [mhz],
                    },
                    "llmservingsim": {
                        **fixed["llmservingsim"],
                        "profile_path": str(
                            repo / fixed["llmservingsim"]["profile_path"]
                        ),
                    },
                    **bench_common,
                }
                run_specs.append(spec)
                run_dir = runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_spec.json").write_text(
                    json.dumps(spec, indent=2) + "\n", encoding="utf-8"
                )
                (run_dir / "sim_replication.json").write_text(
                    json.dumps(spec, indent=2) + "\n", encoding="utf-8"
                )

    campaign = {
        "campaign_id": out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "seed": seed,
        "path": "bench_layer_boundary_prefill",
        "models": MODELS,
        "valid_freqs_mhz": VALID_FREQS_MHZ,
        "fixed_freqs_mhz": FIXED_FREQS_MHZ,
        "num_scattered_permutations": NUM_SCATTERED_PERMUTATIONS,
        "iterations_per_run": ITERATIONS,
        "total_runs": len(run_specs),
        "scattered_runs": sum(
            1 for r in run_specs if r["campaign_mode"] == "scattered"
        ),
        "fixed_runs": sum(1 for r in run_specs if r["campaign_mode"] == "fixed"),
    }
    (out_dir / "campaign_spec.json").write_text(
        json.dumps(campaign, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "permutations.json").write_text(
        json.dumps({"permutations": permutations}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_specs.json").write_text(
        json.dumps(run_specs, indent=2) + "\n", encoding="utf-8"
    )
    readme = out_dir / "README.md"
    readme.write_text(
        """# V100 bench layer-boundary DVFS campaign (prefill-only)

Real vLLM `python -m bench run` with in-place layer-boundary pause + DVFS.
Same scattered/fixed permutations as `profiler/jobs/generate_full_dvfs_campaign.py`.

## Per-run outputs (`runs/<run_id>/`)

- `run_spec.json` — full run configuration (tracked for replication)
- `sim_replication.json` — LLMServingSim profile paths + DVFS permutation
- `bench/` — `meta.json`, `requests.jsonl`, `run_exec_metrics.json`, `dvfs_markers.jsonl`, `gpu_power/`
- `results/summary.json` — aggregated readings (wall/exec/pause/energy/throughput)
- `artifacts/` — copy of key bench outputs

## Collect

```bash
python3 bench/jobs/collect_bench_layer_campaign_results.py <campaign_dir>
```
""",
        encoding="utf-8",
    )
    return campaign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260616)
    args = p.parse_args()
    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.out_dir = args.repo / "bench" / "campaigns" / f"v100_bench_layer_dvfs_{stamp}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign = generate_bench_campaign(
        repo=args.repo, out_dir=args.out_dir, seed=args.seed
    )
    print(f"Wrote bench campaign to {args.out_dir}")
    print(
        f"  total_runs={campaign['total_runs']} "
        f"(scattered={campaign['scattered_runs']} fixed={campaign['fixed_runs']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
