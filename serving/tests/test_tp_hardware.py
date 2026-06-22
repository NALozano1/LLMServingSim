import unittest

from serving.core.config_builder import _resolve_tp_hardware


class TpHardwareConfigTest(unittest.TestCase):
    def test_homogeneous_default(self):
        inst = {"hardware": "V100", "tp_size": 2}
        _resolve_tp_hardware(inst)
        self.assertEqual(inst["tp_hardware"], ["V100", "V100"])

    def test_homogeneous_default_tp1(self):
        inst = {"hardware": "V100", "tp_size": 1}
        _resolve_tp_hardware(inst)
        self.assertEqual(inst["tp_hardware"], ["V100"])

    def test_explicit_heterogeneous_list(self):
        inst = {"hardware": "V100", "tp_size": 2, "tp_hardware": ["V100", "V100_700MHz"]}
        _resolve_tp_hardware(inst)
        self.assertEqual(inst["tp_hardware"], ["V100", "V100_700MHz"])

    def test_length_must_match_tp_size(self):
        inst = {"hardware": "V100", "tp_size": 2, "tp_hardware": ["V100"]}
        with self.assertRaises(ValueError):
            _resolve_tp_hardware(inst)

    def test_first_entry_must_equal_primary(self):
        inst = {"hardware": "V100", "tp_size": 2, "tp_hardware": ["V100_700MHz", "V100"]}
        with self.assertRaises(ValueError):
            _resolve_tp_hardware(inst)

    def test_not_a_list_raises(self):
        inst = {"hardware": "V100", "tp_size": 2, "tp_hardware": "V100"}
        with self.assertRaises(ValueError):
            _resolve_tp_hardware(inst)

    def test_heterogeneous_with_dp_group_raises(self):
        inst = {"hardware": "V100", "tp_size": 2, "dp_group": "A",
                "tp_hardware": ["V100", "V100_700MHz"]}
        with self.assertRaises(ValueError):
            _resolve_tp_hardware(inst)

    def test_uniform_list_with_dp_group_ok(self):
        # uniform (homogeneous) tp_hardware is fine even with a dp_group
        inst = {"hardware": "V100", "tp_size": 2, "dp_group": "A",
                "tp_hardware": ["V100", "V100"]}
        _resolve_tp_hardware(inst)
        self.assertEqual(inst["tp_hardware"], ["V100", "V100"])


if __name__ == "__main__":
    unittest.main()
