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
    """Cache key for a segment's trace/graph.

    Reuse is only safe across batches whose emitted latencies are identical, so
    the signature captures everything that changes a segment's trace: token
    counts plus the per-request prefill/decode KV shape (attention is a 4D
    lookup over prefill_q/prefill_k/decode_k). This stops two decode batches
    with the same total_len but different KV state from colliding. The dtype/kv
    variant is handled by the trace .meta sidecar, and the active clock is
    encoded in the trace path's hardware tag.
    """
    import hashlib
    shape = "|".join((
        str(batch.total_len), str(batch.num_prefill), str(batch.num_decode),
        ",".join(map(str, batch.prefill_q_list)),
        ",".join(map(str, batch.prefill_k_list)),
        ",".join(map(str, batch.decode_k_list)),
    ))
    sig = hashlib.sha1(shape.encode()).hexdigest()[:12]
    return f"instance{instance_id}_t{batch.total_len}p{batch.num_prefill}_s{stage_idx}_{sig}"
