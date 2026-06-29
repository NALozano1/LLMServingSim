#!/usr/bin/env python3
"""Generate V100 full-model DVFS campaign specs (scattered + fixed permutations)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

MODELS: dict[str, dict[str, Any]] = {
    "qwen": {
        "model": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "num_hidden_layers": 24,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 8,
        "attention_max_kv": 256,
    },
}

VALID_FREQS_MHZ = [700, 900, 1100, 1400]
FIXED_FREQS_MHZ = [700, 1100, 1400]
NUM_SCATTERED_PERMUTATIONS = 10
ITERATIONS = 3


def _yaml_dump(data: Any) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    return json.dumps(data, indent=2) + "\n"


def _sim_profile_path(model: str, mhz: int, repo: Path) -> str:
    return str(
        repo / "profiler" / "perf" / f"V100_{mhz}MHz" / model / "fp16"
    )


def _make_scattered_perm(
    rng: random.Random,
    model_key: str,
    perm_idx: int,
) -> dict[str, Any]:
    cfg = MODELS[model_key]
    n_layers = int(cfg["num_hidden_layers"])
    n_barriers = rng.randint(2, min(8, n_layers))
    barrier_layers = sorted(rng.sample(range(n_layers), n_barriers))
    freqs = [rng.choice(VALID_FREQS_MHZ) for _ in barrier_layers]
    freq_at_layer = [
        {"layer": layer, "mhz": mhz}
        for layer, mhz in zip(barrier_layers, freqs)
    ]
    unique_freqs = sorted(set(freqs))
    return {
        "perm_id": f"p{perm_idx:02d}",
        "model_key": model_key,
        "model": cfg["model"],
        "num_hidden_layers": n_layers,
        "mode": "scattered_dvfs",
        "barrier_layers": barrier_layers,
        "freq_schedule_mhz": freqs,
        "freq_at_layer": freq_at_layer,
        "n_transitions": n_barriers,
        "llmservingsim": {
            "hardware_profiles": [
                {
                    "mhz": mhz,
                    "profile_path": f"profiler/perf/V100_{mhz}MHz/{cfg['model']}/fp16",
                }
                for mhz in unique_freqs
            ],
            "notes": (
                "Scattered layer-boundary DVFS during a full-model dense forward. "
                "For simulation, use per-frequency profiler profiles and compare "
                "against measured exec metrics (effective_runtime_sec, energy_j)."
            ),
        },
    }


def _make_fixed_spec(model_key: str, mhz: int, iteration: int) -> dict[str, Any]:
    cfg = MODELS[model_key]
    return {
        "perm_id": f"fixed_{mhz}",
        "model_key": model_key,
        "model": cfg["model"],
        "num_hidden_layers": cfg["num_hidden_layers"],
        "mode": "fixed_dvfs",
        "iteration": iteration,
        "gpu_freq_mhz": mhz,
        "llmservingsim": {
            "hardware": f"V100_{mhz}MHz",
            "profile_path": f"profiler/perf/V100_{mhz}MHz/{cfg['model']}/fp16",
            "notes": (
                "Fixed-frequency full profiler sweep (dense, attention, moe). "
                f"Use hardware=V100_{mhz}MHz in cluster config."
            ),
        },
    }


def _run_id(mode: str, perm_id: str, model_key: str, iteration: int) -> str:
    return f"{mode}_{perm_id}_{model_key}_i{iteration}"


def generate_campaign(
    *,
    repo: Path,
    out_dir: Path,
    seed: int = 20260616,
) -> dict[str, Any]:
    rng = random.Random(seed)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    permutations: list[dict[str, Any]] = []
    for perm_idx in range(1, NUM_SCATTERED_PERMUTATIONS + 1):
        for model_key in ("qwen",):
            permutations.append(_make_scattered_perm(rng, model_key, perm_idx))

    run_specs: list[dict[str, Any]] = []

    for perm in permutations:
        for iteration in range(ITERATIONS):
            run_id = _run_id("scat", perm["perm_id"], perm["model_key"], iteration)
            cfg = MODELS[perm["model_key"]]
            spec: dict[str, Any] = {
                "run_id": run_id,
                "campaign_mode": "scattered",
                "iteration": iteration,
                "model": perm["model"],
                "model_key": perm["model_key"],
                "num_hidden_layers": perm["num_hidden_layers"],
                "hardware": f"V100_full_dvfs_{run_id}",
                "dtype": "float16",
                "tp_size": 1,
                "measurement_iterations": 1,
                "max_num_batched_tokens": cfg["max_num_batched_tokens"],
                "max_num_seqs": cfg["max_num_seqs"],
                "attention_max_kv": cfg["attention_max_kv"],
                "dvfs": {
                    "barrier_layers": perm["barrier_layers"],
                    "freq_schedule_mhz": perm["freq_schedule_mhz"],
                    "freq_at_layer": perm["freq_at_layer"],
                },
                "llmservingsim": perm["llmservingsim"],
                "profiler": {
                    "group": "dense",
                    "max_shots": 1,
                    "categories": ["dense"],
                },
            }
            run_specs.append(spec)
            run_path = runs_dir / run_id / "run_spec.yaml"
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(_yaml_dump(spec), encoding="utf-8")
            sim_path = runs_dir / run_id / "sim_replication.yaml"
            sim_body = {
                "run_id": run_id,
                "model": perm["model"],
                "mode": "scattered_dvfs",
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
            }
            sim_path.write_text(_yaml_dump(sim_body), encoding="utf-8")

    for model_key in ("qwen",):
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
                    "hardware": f"V100_{mhz}MHz",
                    "dtype": "float16",
                    "tp_size": 1,
                    "measurement_iterations": 3,
                    "gpu_freq_mhz": mhz,
                    "max_num_batched_tokens": cfg["max_num_batched_tokens"],
                    "max_num_seqs": cfg["max_num_seqs"],
                    "attention_max_kv": cfg["attention_max_kv"],
                    "skip_skew": True,
                    "dvfs": {"gpu_freq_mhz": mhz},
                    "llmservingsim": {
                        **fixed["llmservingsim"],
                        "profile_path": str(
                            repo / fixed["llmservingsim"]["profile_path"]
                        ),
                    },
                    "profiler": {
                        "full_profile": False,
                        "categories": ["dense", "attention", "moe"],
                    },
                }
                run_specs.append(spec)
                run_path = runs_dir / run_id / "run_spec.yaml"
                run_path.parent.mkdir(parents=True, exist_ok=True)
                run_path.write_text(_yaml_dump(spec), encoding="utf-8")
                sim_path = runs_dir / run_id / "sim_replication.yaml"
                sim_path.write_text(_yaml_dump(spec), encoding="utf-8")

    campaign = {
        "campaign_id": out_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "seed": seed,
        "models": MODELS,
        "valid_freqs_mhz": VALID_FREQS_MHZ,
        "fixed_freqs_mhz": FIXED_FREQS_MHZ,
        "num_scattered_permutations": NUM_SCATTERED_PERMUTATIONS,
        "iterations_per_run": ITERATIONS,
        "total_runs": len(run_specs),
        "scattered_runs": sum(1 for r in run_specs if r["campaign_mode"] == "scattered"),
        "fixed_runs": sum(1 for r in run_specs if r["campaign_mode"] == "fixed"),
    }

    (out_dir / "campaign_spec.yaml").write_text(_yaml_dump(campaign), encoding="utf-8")
    (out_dir / "permutations.yaml").write_text(
        _yaml_dump({"permutations": permutations}), encoding="utf-8"
    )
    (out_dir / "run_specs.json").write_text(
        json.dumps(run_specs, indent=2) + "\n", encoding="utf-8"
    )
    return campaign


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        type=Path,
        default=Path("/data/engs-glass/engs2950/DVFS-MoE/LLMServingSim"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Campaign output directory (default: profiler/campaigns/v100_full_dvfs_<date>)",
    )
    p.add_argument("--seed", type=int, default=20260616)
    args = p.parse_args()

    if args.out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.out_dir = args.repo / "profiler" / "campaigns" / f"v100_full_dvfs_{stamp}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign = generate_campaign(repo=args.repo, out_dir=args.out_dir, seed=args.seed)
    print(f"Wrote campaign to {args.out_dir}")
    print(
        f"  total_runs={campaign['total_runs']} "
        f"(scattered={campaign['scattered_runs']} fixed={campaign['fixed_runs']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
