"""Append simulation wall-clock records to calibration/runtimes/runs.jsonl."""

from __future__ import annotations

import json
import os
import re
import socket
from datetime import date
from typing import Any


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def runs_jsonl_path() -> str:
    return os.path.join(repo_root(), "calibration", "runtimes", "runs.jsonl")


def parse_wall_time(value: str) -> float:
    """Parse seconds float or simulator format like 0h 1m 40.547s."""
    value = value.strip()
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    compact = value.replace(" ", "")
    m = re.fullmatch(
        r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?",
        compact,
    )
    if not m or compact == "":
        raise ValueError(
            f"cannot parse wall time {value!r}; use seconds or e.g. 0h 1m 40.547s"
        )
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = float(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _layout_from_instances(instances: list[dict[str, Any]]) -> dict[str, Any]:
    if not instances:
        return {"instances": 0, "gpus": 0, "tp": 1, "pp": 1, "ep": 1, "dp_group": None}

    def _get(inst: dict, key: str, default: int = 1) -> int:
        return int(inst.get(key, default))

    first = instances[0]
    layout = {
        "instances": len(instances),
        "gpus": sum(_get(i, "num_npus") for i in instances),
        "tp": _get(first, "tp_size"),
        "pp": _get(first, "pp_size"),
        "ep": _get(first, "ep_size"),
        "dp_group": first.get("dp_group"),
    }

    keys = ("tp_size", "pp_size", "ep_size", "dp_group")
    if any(inst.get(k) != first.get(k) for inst in instances[1:] for k in keys):
        layout["heterogeneous"] = True
    return layout


def build_runtime_record(
    *,
    cluster_config: str,
    instances: list[dict[str, Any]],
    wall_time_s: float,
    first_log_wall_s: float | None,
    model: str,
    hardware: str,
    dtype: str,
    dataset: str | None,
    num_requests: int,
    request_routing_policy: str | None = None,
    status: str = "completed",
    notes: str = "",
    host: str | None = None,
) -> dict[str, Any]:
    record = {
        "recorded_at": date.today().isoformat(),
        "host": host or socket.gethostname(),
        "cluster_config": cluster_config,
        "layout": _layout_from_instances(instances),
        "model": model,
        "hardware": hardware,
        "dtype": dtype,
        "dataset": dataset,
        "num_requests": num_requests,
        "wall_time_s": round(wall_time_s, 3),
        "first_log_wall_s": round(first_log_wall_s, 3) if first_log_wall_s is not None else None,
        "status": status,
        "notes": notes,
    }
    if request_routing_policy:
        record["request_routing_policy"] = request_routing_policy
    return record


def append_runtime_record(record: dict[str, Any]) -> str:
    path = runs_jsonl_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path


def record_simulation_runtime(
    *,
    cluster_config: str,
    instances: list[dict[str, Any]],
    wall_time_s: float,
    first_log_wall_s: float | None,
    model: str,
    hardware: str,
    dtype: str,
    dataset: str | None,
    num_requests: int,
    request_routing_policy: str | None = None,
    status: str = "completed",
    notes: str = "",
) -> str | None:
    """Append a run record unless disabled via LLMSERVINGSIM_RECORD_RUNTIME=0."""
    if os.environ.get("LLMSERVINGSIM_RECORD_RUNTIME", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None

    record = build_runtime_record(
        cluster_config=cluster_config,
        instances=instances,
        wall_time_s=wall_time_s,
        first_log_wall_s=first_log_wall_s,
        model=model,
        hardware=hardware,
        dtype=dtype,
        dataset=dataset,
        num_requests=num_requests,
        request_routing_policy=request_routing_policy,
        status=status,
        notes=notes,
    )
    return append_runtime_record(record)
