import os
import tempfile
import unittest
from unittest import mock

from serving.core import hardware_aliases
from serving.core.hardware_aliases import (
    discover_hardware_for_model,
    is_synthetic_profile,
    resolve_hardware_pair,
    toggle_hardware,
)


def _make_source_profile(root, hw, model, variant):
    """Create a minimal measured-looking profile under a temp perf root."""
    vdir = os.path.join(root, hw, model, variant)
    os.makedirs(os.path.join(vdir, "tp1"))
    with open(os.path.join(vdir, "meta.yaml"), "w") as f:
        f.write(f"hardware: {hw}\ngpu: Test GPU\n")
    with open(os.path.join(vdir, "tp1", "dense.csv"), "w") as f:
        f.write("layer,tokens,time_us\nembedding,1,10.0\n")


class HardwareAliasesTest(unittest.TestCase):
    def test_toggle_hardware(self):
        self.assertEqual(toggle_hardware(("A", "B"), "A"), ("A", "B"))
        self.assertEqual(toggle_hardware(("A", "B"), "B"), ("B", "A"))

    def test_qwen_v100_fp16_profiles_discovered(self):
        repo = os.path.join(os.path.dirname(__file__), "..", "..")
        cwd = os.getcwd()
        try:
            os.chdir(repo)
            found = discover_hardware_for_model(
                "Qwen/Qwen3-30B-A3B-Instruct-2507", "fp16",
            )
            self.assertIn("V100", found)
            self.assertIn("V100_700MHz", found)
            self.assertIn("V100_1400MHz", found)
        finally:
            os.chdir(cwd)

    def test_resolve_v100_dvfs_pair_prefers_farthest_clock(self):
        repo = os.path.join(os.path.dirname(__file__), "..", "..")
        cwd = os.getcwd()
        try:
            os.chdir(repo)
            pair = resolve_hardware_pair(
                "Qwen/Qwen3-30B-A3B-Instruct-2507", "fp16", "V100",
                allow_synthetic=False,
            )
            self.assertEqual(pair[0], "V100")
            self.assertEqual(pair[1], "V100_1400MHz")
            pair_low = resolve_hardware_pair(
                "Qwen/Qwen3-30B-A3B-Instruct-2507", "fp16", "V100_700MHz",
                allow_synthetic=False,
            )
            self.assertEqual(pair_low, ("V100_700MHz", "V100_1400MHz"))
        finally:
            os.chdir(cwd)

    def test_explicit_alt_missing_raises_without_opt_in(self):
        # The user asked for a specific pairing; with no measured profile and no
        # opt-in, fabricating synthetic data must be refused with a clear error.
        with tempfile.TemporaryDirectory() as tmp:
            _make_source_profile(tmp, "GPU_A", "org/model", "bf16")
            with mock.patch.object(hardware_aliases, "_profiler_perf_root", return_value=tmp):
                with self.assertRaises(FileNotFoundError):
                    resolve_hardware_pair("org/model", "bf16", "GPU_A", alt="GPU_B",
                                          allow_synthetic=False)

    def test_explicit_alt_seeds_marked_synthetic_with_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_source_profile(tmp, "GPU_A", "org/model", "bf16")
            with mock.patch.object(hardware_aliases, "_profiler_perf_root", return_value=tmp):
                pair = resolve_hardware_pair("org/model", "bf16", "GPU_A", alt="GPU_B",
                                             allow_synthetic=True)
                self.assertEqual(pair, ("GPU_A", "GPU_B"))
                # Fabricated profile must be unmistakably marked synthetic ...
                self.assertTrue(is_synthetic_profile("GPU_B", "org/model", "bf16"))
                # ... and a later run without opt-in must refuse to use it.
                with self.assertRaises(FileNotFoundError):
                    resolve_hardware_pair("org/model", "bf16", "GPU_A", alt="GPU_B",
                                          allow_synthetic=False)


if __name__ == "__main__":
    unittest.main()
