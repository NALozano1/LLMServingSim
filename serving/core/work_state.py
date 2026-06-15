"""Layer-wise work availability: event logging and state reconstruction."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class LayerRecord:
    idx: int
    name: str
    comp_ns: int
    comm_type: str = "NONE"
    comm_size: int = 0
    is_expert: bool = False
    is_pim: bool = False


@dataclass
class BatchWorkRecord:
    instance_id: int
    batch_id: int
    npu_ids: List[int]
    trace_path: str
    layers: List[LayerRecord]
    is_dummy: bool = False
    total_len: int = 0
    num_prefill: int = 0
    num_decode: int = 0
    request_ids: List[int] = field(default_factory=list)


def trace_path_for(hardware: str, model: str, instance_id: int, batch_id: int) -> str:
    rel = f"inputs/trace/{hardware}/{model}/instance{instance_id}_batch{batch_id}.txt"
    return os.path.join(os.getcwd(), rel)


def parse_trace_manifest(path: str) -> List[LayerRecord]:
    layers: List[LayerRecord] = []
    if not os.path.isfile(path):
        return layers
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Layername"):
                continue
            cols = line.split()
            if not cols:
                continue
            if cols[0] == "EXPERT":
                layers.append(LayerRecord(
                    idx=len(layers), name="EXPERT", comp_ns=0,
                    comm_type=cols[2] if len(cols) > 2 else "NONE",
                    comm_size=int(cols[3]) if len(cols) > 3 else 0, is_expert=True))
                continue
            if cols[0] == "PIM":
                layers.append(LayerRecord(idx=len(layers), name="PIM", comp_ns=0, is_pim=True))
                continue
            if cols[0] in ("COLOCATED", "DISTRIBUTED") or cols[0].startswith("model_parallel"):
                continue
            try:
                comp_ns = int(cols[1])
            except (ValueError, IndexError):
                continue
            layers.append(LayerRecord(
                idx=len(layers), name=cols[0], comp_ns=comp_ns,
                comm_type=cols[8] if len(cols) > 8 else "NONE",
                comm_size=int(cols[9]) if len(cols) > 9 and cols[9].isdigit() else 0))
    return layers


def batch_to_work_record(batch, instance_id, npu_ids, trace_path, is_dummy=False):
    layers = parse_trace_manifest(trace_path)
    return BatchWorkRecord(
        instance_id=instance_id, batch_id=batch.batch_id, npu_ids=list(npu_ids),
        trace_path=trace_path, layers=layers, is_dummy=is_dummy,
        total_len=getattr(batch, "total_len", 0), num_prefill=getattr(batch, "num_prefill", 0),
        num_decode=getattr(batch, "num_decode", 0),
        request_ids=[r.id for r in getattr(batch, "requests", [])])


class WorkStateLogger:
    CSV_FIELDS = [
        "ts_ns", "npu_id", "instance_id", "status", "batch_id",
        "layers_done", "layers_total", "layers_remaining",
        "waiting_reqs", "running_reqs", "pending_router",
    ]

    def __init__(self, events_path=None, summary_path=None, summary_interval_ns=1_000_000_000):
        self.summary_interval_ns = summary_interval_ns
        self._events = []
        self._events_fp = None
        self._summary_fp = None
        self._summary_writer = None
        self._last_summary_ts = -1
        if events_path:
            parent = os.path.dirname(os.path.abspath(events_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._events_fp = open(events_path, "w", encoding="utf-8")
        if summary_path:
            parent = os.path.dirname(os.path.abspath(summary_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._summary_fp = open(summary_path, "w", encoding="utf-8", newline="")
            self._summary_writer = csv.DictWriter(self._summary_fp, fieldnames=self.CSV_FIELDS)
            self._summary_writer.writeheader()

    def close(self):
        if self._events_fp:
            self._events_fp.close()
            self._events_fp = None
        if self._summary_fp:
            self._summary_fp.close()
            self._summary_fp = None

    def _emit(self, event):
        self._events.append(event)
        if self._events_fp:
            self._events_fp.write(json.dumps(event, separators=(",", ":")) + "\n")
            self._events_fp.flush()

    def log_sim_start(self, ts_ns, npu2inst, inst2npus):
        self._emit({"ts_ns": ts_ns, "kind": "sim_start",
                    "npu2inst": {str(k): v for k, v in npu2inst.items()},
                    "inst2npus": {str(k): v for k, v in inst2npus.items()}})

    def log_batch_scheduled(self, ts_ns, record):
        data = asdict(record)
        data["ts_ns"] = ts_ns
        data["kind"] = "batch_scheduled"
        self._emit(data)


    def log_dvfs_switch(self, ts_ns, old_scale, new_scale, instance_ids=None):
        self._emit({
            "ts_ns": ts_ns,
            "kind": "dvfs_switch",
            "old_scale": old_scale,
            "new_scale": new_scale,
            "instance_ids": instance_ids if instance_ids is not None else [],
        })

    def log_iteration_complete(self, ts_ns, npu_id, batch_id, instance_id):
        self._emit({"ts_ns": ts_ns, "kind": "iteration_complete",
                    "npu_id": npu_id, "instance_id": instance_id, "batch_id": batch_id})

    def log_scheduler_snapshot(self, ts_ns, schedulers, router, current, inst2npu_mapping, instances):
        inst_snapshots = []
        for inst_id, sched in enumerate(schedulers):
            start_npu = inst2npu_mapping[inst_id]
            npu_ids = list(range(start_npu, start_npu + instances[inst_id]["num_npus"]))
            inflight = [{"batch_id": b.batch_id, "request_ids": [r.id for r in b.requests],
                         "fired": list(b.fired), "end": list(b.end), "total_len": b.total_len}
                        for b in sched.inflight]
            inst_snapshots.append({
                "instance_id": inst_id,
                "waiting_reqs": len([r for r in sched.request if r.arrival <= current]),
                "queued_reqs": len(sched.request), "inflight": inflight, "npu_ids": npu_ids})
        pending = getattr(router, "_pending_requests", [])
        pending_idx = getattr(router, "_pending_idx", 0)
        pending_router = max(0, len(pending) - pending_idx)
        self._emit({"ts_ns": ts_ns, "kind": "scheduler_snapshot",
                    "pending_router": pending_router, "instances": inst_snapshots})
        if self._summary_writer and ts_ns - self._last_summary_ts >= self.summary_interval_ns:
            state = reconstruct_at(self._events, ts_ns)
            for snap in inst_snapshots:
                waiting = snap["waiting_reqs"]
                running = sum(len(b["request_ids"]) for b in snap["inflight"])
                for npu_id in snap["npu_ids"]:
                    npu_state = state.get("npus", {}).get(str(npu_id), {})
                    self._summary_writer.writerow({
                        "ts_ns": ts_ns, "npu_id": npu_id, "instance_id": snap["instance_id"],
                        "status": npu_state.get("status", "unknown"),
                        "batch_id": npu_state.get("batch_id", ""),
                        "layers_done": npu_state.get("layers_done", 0),
                        "layers_total": npu_state.get("layers_total", 0),
                        "layers_remaining": npu_state.get("layers_remaining", 0),
                        "waiting_reqs": waiting, "running_reqs": running,
                        "pending_router": pending_router})
            self._summary_fp.flush()
            self._last_summary_ts = ts_ns


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def reconstruct_at(events_or_path, ts_ns):
    events = load_events(events_or_path) if isinstance(events_or_path, str) else list(events_or_path)
    window = [e for e in events if e.get("ts_ns", 0) <= ts_ns]
    npu2inst = {}
    for e in window:
        if e.get("kind") == "sim_start":
            npu2inst = {int(k): int(v) for k, v in e.get("npu2inst", {}).items()}
    batches = {(e["instance_id"], e["batch_id"]): e for e in window if e.get("kind") == "batch_scheduled"}
    completed = {(e["npu_id"], e["batch_id"]) for e in window if e.get("kind") == "iteration_complete"}
    latest_snap = next((e for e in reversed(window) if e.get("kind") == "scheduler_snapshot"), None)
    pending_router = latest_snap.get("pending_router", 0) if latest_snap else 0
    inst_waiting, inst_running = {}, {}
    if latest_snap:
        for snap in latest_snap.get("instances", []):
            iid = snap["instance_id"]
            inst_waiting[iid] = snap.get("waiting_reqs", 0)
            inst_running[iid] = sum(len(b.get("request_ids", [])) for b in snap.get("inflight", []))
    all_npu_ids = sorted(npu2inst.keys()) if npu2inst else sorted({e["npu_id"] for e in window if e.get("kind") == "iteration_complete"})
    npus = {}
    for npu_id in all_npu_ids:
        inst_id = npu2inst.get(npu_id)
        active_batch, active_record = None, None
        for (iid, bid), rec in batches.items():
            if npu_id not in rec.get("npu_ids", []):
                continue
            sched_ts = next((e["ts_ns"] for e in window if e.get("kind") == "batch_scheduled" and e.get("instance_id") == iid and e.get("batch_id") == bid), 0)
            if sched_ts > ts_ns or (npu_id, bid) in completed:
                continue
            active_batch, active_record = bid, rec
        if active_record is None:
            last_done, last_layers = None, []
            for (npu, bid) in completed:
                if npu != npu_id:
                    continue
                rec = batches.get((npu2inst.get(npu_id, -1), bid))
                if rec:
                    names = [layer["name"] for layer in rec.get("layers", [])]
                    if len(names) >= len(last_layers):
                        last_done, last_layers = bid, names
            npus[npu_id] = {"instance_id": inst_id, "status": "idle", "batch_id": None,
                            "layers_done": len(last_layers), "layers_total": len(last_layers),
                            "layers_remaining": 0, "completed_batch_id": last_done,
                            "layer_names_done": last_layers, "layer_names_remaining": []}
            continue
        layer_names = [layer["name"] for layer in active_record.get("layers", [])]
        npus[npu_id] = {"instance_id": inst_id, "status": "in_flight", "batch_id": active_batch,
                        "layers_done": 0, "layers_total": len(layer_names), "layers_remaining": len(layer_names),
                        "completed_batch_id": None, "layer_names_done": [], "layer_names_remaining": layer_names,
                        "is_dummy": active_record.get("is_dummy", False),
                        "request_ids": active_record.get("request_ids", []),
                        "total_len": active_record.get("total_len", 0)}
    for (npu_id, bid) in completed:
        inst_id = npu2inst.get(npu_id)
        rec = batches.get((inst_id, bid)) if inst_id is not None else None
        if rec and npus.get(npu_id, {}).get("status") == "idle":
            layer_names = [layer["name"] for layer in rec.get("layers", [])]
            npus[npu_id].update({"layers_done": len(layer_names), "layers_total": len(layer_names),
                                 "completed_batch_id": bid, "layer_names_done": layer_names})
    return {"ts_ns": ts_ns, "ts_ms": ts_ns / 1e6, "pending_router": pending_router,
            "instances": {str(iid): {"waiting_reqs": inst_waiting.get(iid, 0), "running_reqs": inst_running.get(iid, 0)}
                          for iid in set(inst_waiting) | set(inst_running)},
            "npus": {str(k): v for k, v in npus.items()}}


def format_state_report(state, npu_id=None):
    lines = [f"Work state @ {state['ts_ms']:.3f} ms (ts_ns={state['ts_ns']})"]
    lines.append(f"  Router pending: {state.get('pending_router', 0)}")
    for iid, inst in sorted(state.get("instances", {}).items(), key=lambda x: int(x[0])):
        lines.append(f"  Instance {iid}: waiting={inst['waiting_reqs']} running={inst['running_reqs']}")
    npus = state.get("npus", {})
    keys = [str(npu_id)] if npu_id is not None else sorted(npus.keys(), key=int)
    for key in keys:
        npu = npus.get(key)
        if not npu:
            continue
        lines.append(f"  NPU {key} (instance {npu.get('instance_id')}):")
        lines.append(f"    status: {npu['status']}")
        if npu.get("batch_id") is not None:
            lines.append(f"    batch: {npu['batch_id']}  requests: {npu.get('request_ids', [])}")
        lines.append(f"    layers: {npu.get('layers_done', 0)}/{npu.get('layers_total', 0)} done, {npu.get('layers_remaining', 0)} remaining")
        remaining = npu.get("layer_names_remaining") or []
        if remaining:
            preview = ", ".join(remaining[:8])
            if len(remaining) > 8:
                preview += f", ... (+{len(remaining) - 8} more)"
            lines.append(f"    remaining layers: [{preview}]")
        done = npu.get("layer_names_done") or []
        if done and npu.get("completed_batch_id") is not None:
            preview = ", ".join(done[:5])
            if len(done) > 5:
                preview += f", ... (+{len(done) - 5} more)"
            lines.append(f"    last completed batch {npu['completed_batch_id']}: [{preview}]")
    return "\n".join(lines)
