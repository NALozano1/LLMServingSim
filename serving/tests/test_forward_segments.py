import unittest

from serving.core.forward_segments import (
    num_stages_for_model,
    register_astra_segment,
    pop_astra_segment,
    segment_workload_slug,
)
from serving.core.request import Batch
from serving.core.controller import Controller


class ForwardSegmentsTest(unittest.TestCase):
    def test_num_stages_llama(self):
        n = num_stages_for_model("meta-llama/Llama-3.1-8B")
        self.assertEqual(n, 34)

    def test_segment_slug_reuses_shape(self):
        b1 = Batch(0, "meta-llama/Llama-3.1-8B", 10, 0, [10], [], 1, 0, [10], [], [], 0, 0)
        b2 = Batch(99, "meta-llama/Llama-3.1-8B", 10, 0, [10], [], 1, 0, [10], [], [], 0, 0)
        self.assertEqual(segment_workload_slug(0, b1, 3), segment_workload_slug(0, b2, 3))

    def test_astra_registry(self):
        ctrl = Controller(1)
        reg = {}
        register_astra_segment(reg, 0, 5, 7, 2, 34)
        self.assertEqual(pop_astra_segment(reg, 0, 5), (7, 2, False))
        register_astra_segment(reg, 0, 10, 7, 33, 34)
        self.assertEqual(pop_astra_segment(reg, 0, 10)[2], True)


if __name__ == "__main__":
    unittest.main()
