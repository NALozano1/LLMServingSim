#!/usr/bin/env python3
"""Generate LLMServingSim cluster JSON configs.

Homogeneous clusters (every instance shares the same tp/pp/ep/dp settings):

  # 8 independent TP=1 replicas
  python3 scripts/generate_cluster_config.py \\
    --num-nodes 2 --instances-per-node 4 \\
    --output configs/cluster/generated_8gpu.json

  # 1024 GPUs from total count
  python3 scripts/generate_cluster_config.py \\
    --total-gpus 1024 --gpus-per-node 8 --tp-size 1 \\
    --link-bw 900 100 --link-latency 500 20000 --indent 0 \\
    --output configs/cluster/generated_1024gpu.json

  # TP x PP on one instance per node (4 GPUs/instance, 16 nodes -> 64 GPUs)
  python3 scripts/generate_cluster_config.py \\
    --num-nodes 16 --instances-per-node 1 --tp-size 2 --pp-size 2 \\
    --output configs/cluster/generated_tp2_pp2.json

  # MoE EP only (single instance, tp=2 ep=2) — Qwen3-30B-A3B
  python3 scripts/generate_cluster_config.py \\
    --num-nodes 1 --instances-per-node 1 \\
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    --tp-size 2 --ep-size 2 \\
    --output configs/cluster/generated_moe_ep.json

  # MoE DP+EP across nodes (matches dual_node_moe_dp_ep_intra_inter_instance)
  python3 scripts/generate_cluster_config.py \\
    --num-nodes 2 --instances-per-node 1 \\
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
    --tp-size 2 --ep-size 4 --dp-group A \\
    --link-bw 128 16 --link-latency 500 20000 \\
    --output configs/cluster/generated_moe_dp_ep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


# Wall-clock heuristics calibrated on nserver15 / example_trace.jsonl (10 requests,
# Llama-3.1-8B bf16, analytical backend). Order-of-magnitude only — not a guarantee.
_CALIBRATION = {
    "base_startup_s": 3.0,
    "per_instance_startup_s": 1.0,
    "per_npu_startup_s": 0.5,
    "pp_npu_stage_startup_s": 40.0,  # PP>1: heavy Chakra/trace work per GPU-stage
    "moe_instance_startup_s": 20.0,
    "dp_group_instance_startup_s": 15.0,
    "sec_per_request_1gpu": 1.3,  # ~13s for 10 requests, 1×TP1 instance
    "replica_scale_per_instance": 0.95,
    "pp_body_multiplier_per_stage": 0.75,
    "moe_body_multiplier": 1.5,
}


@dataclass(frozen=True)
class RuntimeEstimate:
    startup_s: float
    body_s: float
    total_s: float
    first_log_s: float
    notes: tuple[str, ...]

    def format_summary(self) -> str:
        def _fmt(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            if seconds < 3600:
                return f"{seconds / 60:.1f}m"
            return f"{seconds / 3600:.1f}h"

        lines = [
            "Runtime estimate (heuristic, example_trace-scale workload):",
            f"  startup (before first [log-interval] line): ~{_fmt(self.startup_s)}",
            f"  simulation body:                           ~{_fmt(self.body_s)}",
            f"  total wall time:                           ~{_fmt(self.total_s)}",
            f"  first heartbeat about:                     ~{_fmt(self.first_log_s)}",
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


DEFAULT_INSTANCE: dict[str, Any] = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    "npu_mem": {
        "mem_size": 96,
        "mem_bw": 1597,
        "mem_latency": 0,
    },
    "pd_type": None,
}

DEFAULT_CPU_MEM: dict[str, Any] = {
    "mem_size": 512,
    "mem_bw": 256,
    "mem_latency": 0,
}


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_link_values(raw: list[str]) -> float | list[float]:
    values = [float(v) for v in raw]
    return values[0] if len(values) == 1 else values


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _load_model_config(model_name: str) -> dict[str, Any]:
    path = os.path.join(_repo_root(), "configs", "model", f"{model_name}.json")
    if not os.path.isfile(path):
        raise ValueError(
            f"model config not found: configs/model/{model_name}.json"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_moe_model(model_cfg: dict[str, Any]) -> bool:
    return "num_local_experts" in model_cfg or "num_experts" in model_cfg


def _num_experts(model_cfg: dict[str, Any]) -> int:
    return model_cfg.get("num_local_experts", model_cfg.get("num_experts", 1))


def _default_ep_size(model_cfg: dict[str, Any], tp_size: int) -> int:
    return tp_size if _is_moe_model(model_cfg) else 1


def _validate_tp_size(tp_size: int, model_cfg: dict[str, Any], model_name: str) -> None:
    if not _is_power_of_two(tp_size):
        raise ValueError(f"tp_size={tp_size} must be a power of 2")

    num_heads = model_cfg.get("num_attention_heads")
    if num_heads is not None and num_heads % tp_size != 0:
        max_tp = num_heads
        while max_tp > 1 and not _is_power_of_two(max_tp):
            max_tp -= 1
        raise ValueError(
            f"tp_size={tp_size} does not divide num_attention_heads={num_heads} "
            f"for {model_name} (largest valid power-of-2 TP is {max_tp})"
        )


def _validate_parallelism(
    *,
    model_cfg: dict[str, Any],
    model_name: str,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_group: str | None,
    dp_group_size: int,
) -> None:
    gpus_per_instance = tp_size * pp_size

    if tp_size < 1 or pp_size < 1 or ep_size < 1:
        raise ValueError("tp_size, pp_size, and ep_size must be >= 1")

    if dp_group is None and ep_size > tp_size:
        raise ValueError(
            f"ep_size ({ep_size}) > tp_size ({tp_size}) requires --dp-group"
        )

    if _is_moe_model(model_cfg):
        experts = _num_experts(model_cfg)
        if experts % ep_size != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must divide model expert count ({experts})"
            )
    elif ep_size != 1:
        raise ValueError(
            f"dense model {model_name} only supports ep_size=1 (got {ep_size})"
        )

    if dp_group is not None:
        if dp_group_size < 2:
            raise ValueError("--dp-group requires at least 2 instances in the cluster")
        if ep_size % dp_group_size != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must be divisible by dp group size ({dp_group_size})"
            )
        local_ep = ep_size // dp_group_size
        if local_ep > tp_size:
            raise ValueError(
                f"local_ep ({local_ep}) = ep_size/dp_group_size cannot exceed tp_size ({tp_size})"
            )

    if gpus_per_instance < 1:
        raise ValueError("tp_size * pp_size must be >= 1")


def _profile_tp_warning(
    hardware: str, model_name: str, variant: str, tp_size: int
) -> str | None:
    org, _, name = model_name.partition("/")
    if not name:
        return None
    meta_path = os.path.join(
        _repo_root(),
        "profiler",
        "perf",
        hardware,
        org,
        name,
        variant,
        "meta.yaml",
    )
    if not os.path.isfile(meta_path):
        return (
            f"no profile bundle at profiler/perf/{hardware}/{org}/{name}/{variant}/"
        )

    try:
        import yaml
    except ImportError:
        return None

    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    tp_degrees = set(meta.get("tp_degrees") or [])
    if tp_size not in tp_degrees:
        listed = ", ".join(f"tp{d}" for d in sorted(tp_degrees))
        return (
            f"bundled profiles only cover [{listed}], not tp{tp_size}; re-profile or pick another TP"
        )
    return None


def _build_instance(
    template: dict[str, Any],
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_group: str | None,
) -> dict[str, Any]:
    inst = deepcopy(template)
    inst["tp_size"] = tp_size
    inst["num_npus"] = tp_size * pp_size
    if pp_size > 1:
        inst["pp_size"] = pp_size
    if ep_size != 1 or _is_moe_model(_load_model_config(inst["model_name"])):
        inst["ep_size"] = ep_size
    if dp_group is not None:
        inst["dp_group"] = dp_group
    return inst


def resolve_layout(
    *,
    num_nodes: int | None,
    instances_per_node: int | None,
    total_gpus: int | None,
    gpus_per_node: int | None,
    tp_size: int,
    pp_size: int,
) -> tuple[int, int, int]:
    gpus_per_instance = tp_size * pp_size

    explicit = num_nodes is not None or instances_per_node is not None
    auto = total_gpus is not None or gpus_per_node is not None
    if explicit and auto:
        raise ValueError(
            "use either (--num-nodes + --instances-per-node) "
            "or (--total-gpus + --gpus-per-node), not both"
        )
    if not explicit and not auto:
        raise ValueError(
            "provide (--num-nodes + --instances-per-node) "
            "or (--total-gpus + --gpus-per-node)"
        )

    if explicit:
        if num_nodes is None or instances_per_node is None:
            raise ValueError("--num-nodes and --instances-per-node must be given together")
        total = num_nodes * instances_per_node * gpus_per_instance
        return num_nodes, instances_per_node, total

    if total_gpus is None or gpus_per_node is None:
        raise ValueError("--total-gpus and --gpus-per-node must be given together")
    if total_gpus % gpus_per_instance != 0:
        raise ValueError(
            f"total_gpus ({total_gpus}) must be divisible by tp_size*pp_size ({gpus_per_instance})"
        )
    if gpus_per_node % gpus_per_instance != 0:
        raise ValueError(
            f"gpus_per_node ({gpus_per_node}) must be divisible by tp_size*pp_size ({gpus_per_instance})"
        )

    total_instances = total_gpus // gpus_per_instance
    inst_per_node = gpus_per_node // gpus_per_instance
    if total_instances % inst_per_node != 0:
        raise ValueError(
            f"total instances ({total_instances}) must be divisible by instances_per_node ({inst_per_node})"
        )
    nodes = total_instances // inst_per_node
    return nodes, inst_per_node, total_gpus


def build_cluster_config(
    *,
    num_nodes: int,
    instances_per_node: int,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_group: str | None,
    model_name: str,
    hardware: str,
    npu_mem: dict[str, Any] | None,
    cpu_mem: dict[str, Any] | None,
    link_bw: float | list[float],
    link_latency: float | list[float],
    pd_type: str | None,
) -> dict[str, Any]:
    instance_template = deepcopy(DEFAULT_INSTANCE)
    instance_template["model_name"] = model_name
    instance_template["hardware"] = hardware
    instance_template["pd_type"] = pd_type
    if npu_mem is not None:
        instance_template["npu_mem"] = deepcopy(npu_mem)

    instance = _build_instance(
        instance_template,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        dp_group=dp_group,
    )

    node_cpu_mem = deepcopy(cpu_mem or DEFAULT_CPU_MEM)
    nodes = []
    for _ in range(num_nodes):
        nodes.append(
            {
                "num_instances": instances_per_node,
                "cpu_mem": deepcopy(node_cpu_mem),
                "instances": [deepcopy(instance) for _ in range(instances_per_node)],
            }
        )

    return {
        "num_nodes": num_nodes,
        "link_bw": link_bw,
        "link_latency": link_latency,
        "nodes": nodes,
    }


def estimate_runtime(
    *,
    total_instances: int,
    total_gpus: int,
    gpus_per_instance: int,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    dp_group: str | None,
    is_moe: bool,
    num_requests: int,
) -> RuntimeEstimate:
    """Rough wall-clock estimate for `python -m serving` on this cluster."""
    c = _CALIBRATION
    notes: list[str] = []

    startup = (
        c["base_startup_s"]
        + c["per_instance_startup_s"] * total_instances
        + c["per_npu_startup_s"] * total_gpus
    )

    if pp_size > 1:
        startup += (
            c["pp_npu_stage_startup_s"]
            * pp_size
            * gpus_per_instance
            * max(1, tp_size // 1)
        )
        notes.append(
            "PP>1 has a large silent startup while pipeline traces are built "
            "(often many minutes before the first throughput log)."
        )

    if is_moe:
        startup += c["moe_instance_startup_s"] * total_instances * max(1, ep_size // max(1, tp_size))
        notes.append("MoE adds trace/routing overhead; EP and DP+EP run slower than dense TP replicas.")

    if dp_group is not None:
        startup += c["dp_group_instance_startup_s"] * total_instances
        notes.append("DP groups wave-synchronize across instances.")

    # Observed: independent TP=1 replicas scale ~linearly with instance count.
    if pp_size > 1:
        body_factor = 1.0 + c["pp_body_multiplier_per_stage"] * (pp_size - 1)
    elif total_instances > 1 and dp_group is None:
        body_factor = max(1.0, total_instances * c["replica_scale_per_instance"])
    else:
        body_factor = 1.0

    if is_moe:
        body_factor *= c["moe_body_multiplier"] * max(1.0, ep_size / max(1, tp_size))
    if dp_group is not None:
        body_factor *= max(1.0, total_instances * 0.8)

    body = c["sec_per_request_1gpu"] * num_requests * body_factor

    if total_gpus >= 256:
        notes.append(f"Large clusters ({total_gpus} GPUs) can be much slower than this estimate.")
    if num_requests != 10:
        notes.append(f"Scaled for --num-requests={num_requests} (calibration used 10).")

    notes.append("Calibrated on nserver15; container CPU load and dataset size dominate variance.")

    total = startup + body
    first_log = min(startup, max(1.0, startup * 0.85))
    return RuntimeEstimate(
        startup_s=startup,
        body_s=body,
        total_s=total,
        first_log_s=first_log,
        notes=tuple(notes),
    )


def validate_cluster_config(config: dict[str, Any]) -> tuple[int, int]:
    if config["num_nodes"] != len(config["nodes"]):
        raise ValueError("num_nodes must equal len(nodes)")

    total_gpus = 0
    total_instances = 0
    for node_idx, node in enumerate(config["nodes"]):
        if node["num_instances"] != len(node["instances"]):
            raise ValueError(f"nodes[{node_idx}]: num_instances mismatch")
        total_instances += len(node["instances"])
        for inst in node["instances"]:
            tp = inst["tp_size"]
            pp = inst.get("pp_size", 1)
            if inst.get("num_npus", tp * pp) != tp * pp:
                raise ValueError("num_npus must equal tp_size * pp_size")
            total_gpus += tp * pp

    return total_instances, total_gpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LLMServingSim cluster config JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    layout = parser.add_argument_group("layout (pick one mode)")
    layout.add_argument("--num-nodes", type=int)
    layout.add_argument("--instances-per-node", type=int)
    layout.add_argument("--total-gpus", type=int)
    layout.add_argument("--gpus-per-node", type=int)

    par = parser.add_argument_group("parallelism")
    par.add_argument("--tp-size", type=int, default=1)
    par.add_argument("--pp-size", type=int, default=1)
    par.add_argument(
        "--ep-size",
        type=int,
        help="Expert-parallel degree (default: tp_size for MoE, 1 for dense)",
    )
    par.add_argument(
        "--dp-group",
        metavar="NAME",
        help="DP group id assigned to every instance (requires >= 2 instances)",
    )

    model = parser.add_argument_group("model / hardware")
    model.add_argument("--model-name", default=DEFAULT_INSTANCE["model_name"])
    model.add_argument("--hardware", default=DEFAULT_INSTANCE["hardware"])
    model.add_argument("--variant", default="bf16", help="Profile variant for warnings")

    net = parser.add_argument_group("network")
    net.add_argument("--link-bw", nargs="+", default=["16"])
    net.add_argument("--link-latency", nargs="+", default=["20000"])

    est = parser.add_argument_group("runtime estimate")
    est.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Expected request count for wall-time estimate (default: 10, matches example_trace.jsonl)",
    )
    est.add_argument(
        "--no-estimate",
        action="store_true",
        help="Skip printing the heuristic runtime estimate",
    )

    parser.add_argument("--output", required=True)
    parser.add_argument("--indent", type=int, default=4)
    args = parser.parse_args()

    model_cfg = _load_model_config(args.model_name)
    ep_size = (
        args.ep_size
        if args.ep_size is not None
        else _default_ep_size(model_cfg, args.tp_size)
    )

    num_nodes, instances_per_node, total_gpus = resolve_layout(
        num_nodes=args.num_nodes,
        instances_per_node=args.instances_per_node,
        total_gpus=args.total_gpus,
        gpus_per_node=args.gpus_per_node,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )
    total_instances = num_nodes * instances_per_node

    _validate_parallelism(
        model_cfg=model_cfg,
        model_name=args.model_name,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        ep_size=ep_size,
        dp_group=args.dp_group,
        dp_group_size=total_instances,
    )

    _validate_tp_size(args.tp_size, model_cfg, args.model_name)

    profile_warning = _profile_tp_warning(
        args.hardware, args.model_name, args.variant, args.tp_size
    )
    if profile_warning:
        print(f"warning: {profile_warning}", file=sys.stderr)

    if args.dp_group and total_instances < 2:
        raise ValueError("--dp-group requires at least 2 instances")

    config = build_cluster_config(
        num_nodes=num_nodes,
        instances_per_node=instances_per_node,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        ep_size=ep_size,
        dp_group=args.dp_group,
        model_name=args.model_name,
        hardware=args.hardware,
        npu_mem=None,
        cpu_mem=None,
        link_bw=_parse_link_values(args.link_bw),
        link_latency=_parse_link_values(args.link_latency),
        pd_type=None,
    )

    total_instances, total_gpus = validate_cluster_config(config)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if args.indent:
            json.dump(config, f, indent=args.indent)
            f.write("\n")
        else:
            json.dump(config, f, separators=(",", ":"))
            f.write("\n")

    gpus_per_instance = args.tp_size * args.pp_size
    mode = []
    if args.tp_size > 1:
        mode.append(f"TP={args.tp_size}")
    if args.pp_size > 1:
        mode.append(f"PP={args.pp_size}")
    if ep_size > 1:
        mode.append(f"EP={ep_size}")
    if args.dp_group:
        mode.append(f"DP group '{args.dp_group}' (size={total_instances})")
    mode_str = ", ".join(mode) if mode else "independent replicas (TP=1)"

    print(f"Wrote {out_path}")
    print(
        f"  {num_nodes} nodes x {instances_per_node} instances/node "
        f"= {total_instances} instances, {total_gpus} GPUs "
        f"({gpus_per_instance} GPUs/instance; {mode_str})"
    )

    if not args.no_estimate:
        runtime = estimate_runtime(
            total_instances=total_instances,
            total_gpus=total_gpus,
            gpus_per_instance=gpus_per_instance,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            ep_size=ep_size,
            dp_group=args.dp_group,
            is_moe=_is_moe_model(model_cfg),
            num_requests=args.num_requests,
        )
        print()
        print(runtime.format_summary())

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
