import os
import subprocess
from time import time
from .request import *
from .logger import get_logger

logger = get_logger("GraphGenerator")

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