import os
import unittest

from serving.core.hardware_aliases import (
    discover_hardware_for_model,
    resolve_hardware_pair,
    toggle_hardware,
)


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
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "fp16",
                "V100",
                seed_if_missing=False,
            )
            self.assertEqual(pair[0], "V100")
            self.assertEqual(pair[1], "V100_1400MHz")
            pair_low = resolve_hardware_pair(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "fp16",
                "V100_700MHz",
                seed_if_missing=False,
            )
            self.assertEqual(pair_low, ("V100_700MHz", "V100_1400MHz"))
        finally:
            os.chdir(cwd)

    def test_resolve_pair_seeds_from_v0(self):
        repo = os.path.join(os.path.dirname(__file__), "..", "..")
        cwd = os.getcwd()
        try:
            os.chdir(repo)
            pair = resolve_hardware_pair(
                "meta-llama/Llama-3.1-8B", "bf16", "RTXPRO6000", seed_if_missing=True,
            )
            self.assertEqual(pair[0], "RTXPRO6000")
            self.assertNotEqual(pair[0], pair[1])
            self.assertIn(
                pair[1],
                discover_hardware_for_model("meta-llama/Llama-3.1-8B", "bf16"),
            )
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
