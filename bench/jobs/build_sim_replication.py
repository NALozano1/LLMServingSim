#!/usr/bin/env python3
"""Build LLMServingSim replication bundle for a real vLLM bench run."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def _profile_root(repo: Path, hardware: str, model: str, variant: str = "fp16") -> Path:
    return repo / "profiler" / "perf" / hardware / model / variant


def _cluster_config(model: str, hardware: str, tp: int = 1) -> dict[str, Any]:
    return {
        "num_nodes": 1,
        "link_bw": 16,
        "link_latency": 20000,
        "nodes": [
            {
                "num_instances": 1,
                "cpu_mem": {
                    "mem_size": 512,
                    "mem_bw": 256,
                    "mem_latency": 0,
                },
                "instances": [
                    {
                        "model_name": model,
                        "hardware": hardware,
                        "npu_mem": {
                            "mem_size": 32,
                            "mem_bw": 900,
                            "mem_latency": 0,
                        },
                        "num_npus": tp,
                        "tp_size": tp,
                        "pd_type": None,
                    }
                ],
            }
        ],
    }


def build_replication(
    *,
    repo: Path,
    run_id: str,
    model: str,
    dataset: str,
    bench_out_dir: Path,
    hardware: str,
    gpu_freq_mhz: int | None,
    freq_schedule: list[int] | None,
    bench_preset: str,
    num_reqs: int,
    sps: int,
    seed: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    dtype: str,
    tp: int,
) -> dict[str, Any]:
    variant = "fp16" if dtype in ("float16", "half", "fp16") else dtype
    profile_path = _profile_root(repo, hardware, model, variant)

    sim_hardware = hardware
    sim_notes: list[str] = []

    if freq_schedule and len(freq_schedule) >= 2:
        sim_notes.append(
            "Mid-run DVFS bench: simulator uses one hardware profile tag per run. "
            "For dual-freq runs, compare against a time-weighted mix or run two "
            f"sim passes with {freq_schedule[0]} MHz and {freq_schedule[1]} MHz profiles."
        )
        sim_notes.append(
            f"Phase profiles: "
            + ", ".join(
                f"{mhz} MHz -> profiler/perf/V100_{mhz}MHz/{model}/{variant}/"
                for mhz in freq_schedule
            )
        )
    elif gpu_freq_mhz:
        sim_hardware = f"V100_{gpu_freq_mhz}MHz"
        profile_path = _profile_root(repo, sim_hardware, model, variant)
        sim_notes.append(f"Fixed DVFS: use hardware={sim_hardware} in cluster config.")
    else:
        sim_hardware = "V100"
        profile_path = _profile_root(repo, sim_hardware, model, variant)
        sim_notes.append("Default boost clocks: use hardware=V100 profile.")

    cluster = _cluster_config(model, sim_hardware, tp)
    cluster_path = bench_out_dir / "sim_cluster.json"
    cluster_path.write_text(json.dumps(cluster, indent=2) + "\n", encoding="utf-8")

    serving_cmd = textwrap.dedent(
        f"""\
        cd {repo}
        python -m serving \\
          --cluster-config {cluster_path} \\
          --dataset {dataset} \\
          --output {bench_out_dir / 'sim.csv'} \\
          --dtype {dtype} \\
          --max-num-seqs {max_num_seqs} \\
          --max-num-batched-tokens {max_num_batched_tokens} \\
          --num-reqs {num_reqs} \\
          --log-interval 1.0 \\
          --log-level INFO
        """
    ).strip()

    validate_cmd = textwrap.dedent(
        f"""\
        cd {repo}
        python -m bench validate \\
          --bench-dir {bench_out_dir} \\
          --sim-csv {bench_out_dir / 'sim.csv'} \\
          --sim-log {bench_out_dir / 'sim.log'} \\
          --output-subdir validation
        """
    ).strip()

    bundle = {
        "run_id": run_id,
        "model": model,
        "bench_preset": bench_preset,
        "real_bench_output": str(bench_out_dir),
        "real_latency_json": str(bench_out_dir / "real_latency.json"),
        "dataset": dataset,
        "workload": {
            "num_reqs": num_reqs,
            "sps": sps,
            "seed": seed,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
        "dvfs": {
            "gpu_freq_mhz": gpu_freq_mhz,
            "freq_schedule": freq_schedule,
            "hardware_label": hardware,
        },
        "llmservingsim": {
            "hardware": sim_hardware,
            "profile_path": str(profile_path),
            "profile_exists": profile_path.is_dir() and (profile_path / "tp1" / "dense.csv").is_file(),
            "cluster_config": str(cluster_path),
            "serving_command": serving_cmd,
            "validate_command": validate_cmd,
            "notes": sim_notes,
        },
    }
    return bundle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--spec", type=Path, required=True, help="run spec JSON")
    p.add_argument("--bench-out", type=Path, required=True)
    p.add_argument("-o", type=Path, required=True)
    args = p.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    dvfs = spec.get("dvfs") or {}
    freq_sched = spec.get("freq_schedule") or dvfs.get("freq_schedule")
    model = spec["model"]
    safe_model = model.replace("/", "__").replace(":", "__")
    # Campaign shared workload path (written by run_arc_v100_bench_campaign.sh).
    campaign_dir = args.spec.parent.parent.parent
    dataset = spec.get("dataset") or str(
        campaign_dir
        / "shared"
        / f"sharegpt-{safe_model}-{spec['num_reqs']}-sps{spec['sps']}-seed{spec['seed']}.jsonl"
    )

    bundle = build_replication(
        repo=args.repo,
        run_id=args.run_id,
        model=model,
        dataset=dataset,
        bench_out_dir=args.bench_out,
        hardware=spec["hardware"],
        gpu_freq_mhz=spec.get("gpu_freq_mhz") or dvfs.get("gpu_freq_mhz"),
        freq_schedule=freq_sched,
        bench_preset=spec.get("bench_preset", "auto"),
        num_reqs=int(spec["num_reqs"]),
        sps=int(spec["sps"]),
        seed=int(spec["seed"]),
        max_model_len=int(spec["max_model_len"]),
        max_num_seqs=int(spec["max_num_seqs"]),
        max_num_batched_tokens=int(spec["max_num_batched_tokens"]),
        dtype=spec.get("dtype", "float16"),
        tp=int(spec.get("tp_size", 1)),
    )
    args.o.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
