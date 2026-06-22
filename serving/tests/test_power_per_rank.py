import unittest

from serving.core.power_model import PowerModel


def _power_config():
    return {
        "base_node_power": 0,
        "npu": {
            "V100":        {"idle_power": 40, "active_power": 140, "standby_power": 150, "standby_duration": 0.1, "num_npus": 2},
            "V100_700MHz": {"idle_power": 30, "active_power": 90,  "standby_power": 100, "standby_duration": 0.1, "num_npus": 0},
        },
        "cpu": {"idle_power": 10, "active_power": 50, "util": 0.1},
        "dram": {"mem_size": 16, "dimm_size": 8, "idle_power": 5},
        "link": {"idle_power": 1, "num_links": 1},
        "nic": {"idle_power": 1, "num_nics": 1},
        "storage": {"idle_power": 1, "num_devices": 1},
    }


class PowerPerRankTest(unittest.TestCase):
    def test_homogeneous_per_rank_equals_old_method(self):
        # Old path: one profile, num_npus=2 NPUs at the same latency.
        old = PowerModel([_power_config()])
        old.add_npu_active_energy_consumption("V100", 0, 1_000_000_000, num_npus=2)
        # New per-rank path with a uniform list of length tp_size must match exactly.
        new = PowerModel([_power_config()])
        new.add_npu_active_energy_per_rank(0, ["V100", "V100"], [1_000_000_000, 1_000_000_000])
        self.assertAlmostEqual(new.net_energies[0]["npu"], old.net_energies[0]["npu"])
        self.assertAlmostEqual(new.net_energies[0]["cpu"], old.net_energies[0]["cpu"])
        # sanity: (140-40)*1.0*2 = 200 J of NPU active-delta
        self.assertAlmostEqual(old.net_energies[0]["npu"], 200.0)

    def test_heterogeneous_sums_per_rank(self):
        pm = PowerModel([_power_config()])
        # rank0 V100 busy 1s, rank1 V100_700MHz busy 2s
        pm.add_npu_active_energy_per_rank(0, ["V100", "V100_700MHz"], [1_000_000_000, 2_000_000_000])
        # NPU = (140-40)*1 + (90-30)*2 = 100 + 120 = 220
        self.assertAlmostEqual(pm.net_energies[0]["npu"], 220.0)
        # CPU charged once on max latency (2s): (50-10)*max(0.7-0.1,0)*2 = 40*0.6*2 = 48
        self.assertAlmostEqual(pm.net_energies[0]["cpu"], 48.0)

    def test_idle_rank_zero_latency_costs_nothing_active(self):
        pm = PowerModel([_power_config()])
        # rank1 idle this layer (local_tokens==0 => lat 0)
        pm.add_npu_active_energy_per_rank(0, ["V100", "V100_700MHz"], [1_000_000_000, 0])
        # only rank0 active: (140-40)*1 = 100
        self.assertAlmostEqual(pm.net_energies[0]["npu"], 100.0)

    def test_base_power_uses_per_profile_idle(self):
        # base_powers idle baseline sums each profile's own idle_power * its num_npus.
        pm = PowerModel([_power_config()])  # V100 num_npus=2, V100_700MHz num_npus=0
        self.assertAlmostEqual(pm.base_powers[0]["npu"], 40 * 2 + 30 * 0)
        # heterogeneous registration (1 each) would give 40 + 30:
        cfg = _power_config()
        cfg["npu"]["V100"]["num_npus"] = 1
        cfg["npu"]["V100_700MHz"]["num_npus"] = 1
        pm2 = PowerModel([cfg])
        self.assertAlmostEqual(pm2.base_powers[0]["npu"], 40 + 30)


if __name__ == "__main__":
    unittest.main()
