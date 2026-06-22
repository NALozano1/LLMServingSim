"""Split a logical forward pass into per-layer ASTRA submissions."""


def num_stages_for_model(model):
    from .utils import get_config
    return get_config(model)["num_hidden_layers"] + 2


def trace_file_suffix(stage_idx):
    return f"_s{stage_idx}" if stage_idx is not None else ""


def register_astra_segment(registry, sys, next_astra_id, batch_id, stage_idx, num_stages):
    is_final = stage_idx >= num_stages - 1
    registry[(sys, next_astra_id)] = (batch_id, stage_idx, is_final)


def pop_astra_segment(registry, sys, astra_id):
    return registry.pop((sys, astra_id), None)


def segment_workload_slug(instance_id, batch, stage_idx):
    """Reuse trace/graph across batches with identical token shape."""
    return f"instance{instance_id}_t{batch.total_len}p{batch.num_prefill}_s{stage_idx}"
