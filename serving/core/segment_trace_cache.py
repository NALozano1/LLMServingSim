import json
import os


def segment_trace_meta_path(trace_path):
    return f"{trace_path}.meta"


def segment_trace_is_fresh(trace_path, dvfs_scale, variant=None):
    """True when a segment trace and sidecar meta match the current DVFS scale
    and dtype/kv variant. Variant is checked here (rather than in the path) so a
    rerun at a different dtype regenerates the trace instead of reusing it."""
    meta_path = segment_trace_meta_path(trace_path)
    if not os.path.isfile(trace_path) or not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("dvfs_scale") == dvfs_scale and meta.get("variant") == variant
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def write_segment_trace_meta(trace_path, dvfs_scale, variant=None):
    meta_path = segment_trace_meta_path(trace_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"dvfs_scale": dvfs_scale, "variant": variant}, f)
