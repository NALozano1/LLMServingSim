import os
import unittest

from serving.core.work_state import load_events, parse_trace_manifest, reconstruct_at


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "trace_mini.txt")


class TestWorkState(unittest.TestCase):
    def test_parse_trace_manifest(self):
        layers = parse_trace_manifest(FIXTURE)
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].name, "embedding")
        self.assertEqual(layers[2].comm_type, "ALLREDUCE")

    def test_reconstruct_at(self):
        events = [
            {"ts_ns": 0, "kind": "sim_start", "npu2inst": {"0": "0"}, "inst2npus": {"0": [0]}},
            {"ts_ns": 100, "kind": "batch_scheduled", "instance_id": 0, "batch_id": 0,
             "npu_ids": [0], "trace_path": FIXTURE,
             "layers": [{"name": "embedding"}, {"name": "attention"}, {"name": "o_proj"}],
             "request_ids": [1], "is_dummy": False},
            {"ts_ns": 200, "kind": "scheduler_snapshot", "pending_router": 2,
             "instances": [{"instance_id": 0, "waiting_reqs": 3, "inflight": [{"batch_id": 0, "request_ids": [1]}], "npu_ids": [0]}]},
        ]
        mid = reconstruct_at(events, 150)
        self.assertEqual(mid["npus"]["0"]["status"], "in_flight")
        self.assertEqual(mid["npus"]["0"]["layers_remaining"], 3)
        events.append({"ts_ns": 500, "kind": "iteration_complete", "npu_id": 0, "instance_id": 0, "batch_id": 0})
        done = reconstruct_at(events, 500)
        self.assertEqual(done["npus"]["0"]["status"], "idle")
        self.assertEqual(done["npus"]["0"]["layers_done"], 3)


if __name__ == "__main__":
    unittest.main()
