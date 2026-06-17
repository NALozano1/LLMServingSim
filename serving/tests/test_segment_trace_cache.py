import json
import os
import tempfile
import unittest

from serving.core.segment_trace_cache import (
    segment_trace_is_fresh,
    write_segment_trace_meta,
)


class TestSegmentTraceCache(unittest.TestCase):
    def test_fresh_when_meta_matches_dvfs_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "slug.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("trace")
            write_segment_trace_meta(trace_path, 1.0)
            self.assertTrue(segment_trace_is_fresh(trace_path, 1.0))

    def test_stale_when_dvfs_scale_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "slug.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("trace")
            write_segment_trace_meta(trace_path, 1.0)
            self.assertFalse(segment_trace_is_fresh(trace_path, 1.25))

    def test_missing_meta_not_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = os.path.join(tmp, "slug.txt")
            with open(trace_path, "w", encoding="utf-8") as f:
                f.write("trace")
            self.assertFalse(segment_trace_is_fresh(trace_path, 1.0))


if __name__ == "__main__":
    unittest.main()
