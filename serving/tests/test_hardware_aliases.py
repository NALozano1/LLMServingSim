import os
import tempfile
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
