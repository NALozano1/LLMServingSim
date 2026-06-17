import os
import subprocess
from time import time
from .request import *
from .logger import get_logger

logger = get_logger("GraphGenerator")


def _cached_graph_is_fresh(repo_root, file_name, output_name, stage_idx=None):
    """True when a cached Chakra graph can be reused for this workload."""
    trace_path = os.path.join(repo_root, "inputs", "trace", f"{file_name}.txt")
    et_path = os.path.join(repo_root, "inputs", "workload", output_name, "llm.0.et")
    if not os.path.isfile(et_path) or not os.path.isfile(trace_path):
        return False
    if stage_idx is not None:
        from .segment_trace_cache import segment_trace_meta_path
        return os.path.isfile(segment_trace_meta_path(trace_path))
    return os.path.getmtime(et_path) >= os.path.getmtime(trace_path)


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, stage_idx=None):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    os.chdir(chakra)

    if event:
        file_name = 'event_handler'
    else:
        if stage_idx is not None:
            from .forward_segments import segment_workload_slug
            file_name = f'{hardware}/{batch.model}/{segment_workload_slug(instance_id, batch, stage_idx)}'
        else:
            file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    if not event and _cached_graph_is_fresh(cwd, file_name, output_name, stage_idx=stage_idx):
        logger.info("Reusing cached graph inputs/workload/%s/llm.0.et", output_name)
        os.chdir(cwd)
        return

    workload_dir = f'../../../inputs/workload/{output_name}'
    os.makedirs(workload_dir, exist_ok=True)
    cmd = f'python -m chakra.src.converter.converter LLM ' \
            f'--input ../../../inputs/trace/{file_name}.txt ' \
            f'--output ../../../inputs/workload/{output_name}/llm ' \
            f'--num-npus {num_npus} ' \
            f'--npu-offset {npu_offset}'

    if enable_local_offloading:
        cmd += ' --local-offloading'

    logger.debug("Generating graph with command: %s", cmd, extra={"node_id": node_id, "instance_id": instance_id})

    cmd = cmd.split()
    subprocess.run(cmd, text=True)    
    os.chdir(cwd)
    return