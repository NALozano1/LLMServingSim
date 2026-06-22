import unittest
from unittest.mock import MagicMock, patch

from serving.core.graph_generator import _cached_graph_is_fresh, generate_graph


class TestGraphGeneratorCache(unittest.TestCase):
    def test_fresh_when_et_newer_than_trace(self):
        with unittest.mock.patch("os.path.isfile", return_value=True), unittest.mock.patch(
            "os.path.getmtime", side_effect=[200.0, 100.0]
        ):
            self.assertTrue(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug")
            )

    def test_stale_when_trace_newer_than_et(self):
        with unittest.mock.patch("os.path.isfile", return_value=True), unittest.mock.patch(
            "os.path.getmtime", side_effect=[100.0, 200.0]
        ):
            self.assertFalse(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug")
            )

    def test_missing_files_not_fresh(self):
        with unittest.mock.patch("os.path.isfile", return_value=False):
            self.assertFalse(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug")
            )

    def test_segment_fresh_requires_meta_and_newer_et(self):
        # meta present AND et (200) newer than trace (100) -> fresh
        with unittest.mock.patch("os.path.isfile") as isfile, unittest.mock.patch(
            "os.path.getmtime", side_effect=[200.0, 100.0]
        ):
            isfile.side_effect = lambda p: p.endswith("llm.0.et") or p.endswith(".txt") or p.endswith(".meta")
            self.assertTrue(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug", stage_idx=0)
            )

    def test_segment_stale_when_trace_regenerated(self):
        # Regression for the silent stale-.et bug: meta present but trace (200)
        # newer than et (100) -> must NOT reuse (dvfs_scale change rewrote trace).
        with unittest.mock.patch("os.path.isfile") as isfile, unittest.mock.patch(
            "os.path.getmtime", side_effect=[100.0, 200.0]
        ):
            isfile.side_effect = lambda p: p.endswith("llm.0.et") or p.endswith(".txt") or p.endswith(".meta")
            self.assertFalse(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug", stage_idx=0)
            )

    def test_segment_stale_when_meta_missing(self):
        with unittest.mock.patch("os.path.isfile") as isfile:
            isfile.side_effect = lambda p: p.endswith("llm.0.et") or p.endswith(".txt")  # no .meta
            self.assertFalse(
                _cached_graph_is_fresh("/repo", "HW/model/slug", "HW/model/slug", stage_idx=0)
            )

    def test_generate_graph_skips_chakra_when_cache_fresh(self):
        batch = MagicMock()
        batch.model = "meta-llama/Llama-3.1-8B"
        batch.total_len = 128
        batch.num_prefill = 64
        with patch("serving.core.graph_generator._cached_graph_is_fresh", return_value=True), patch(
            "subprocess.run"
        ) as run_mock, patch("os.chdir"):
            generate_graph(batch, "RTXPRO6000", 1, stage_idx=0)
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
