"""Unit tests for the unified power+latency capture framework.

Tests run without vLLM, torch, or a GPU.  All external dependencies
are stubbed with simple dataclasses / dicts.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest.mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# -----------------------------------------------------------------------
# Ensure profiler package is importable (editable install or PYTHONPATH).
# -----------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[2]))

from profiler.core.gpu_power import (
    compute_achieved_mhz,
    compute_idle_power_w,
    compute_power_hz,
    integrate_power_joules,
)
from profiler.core.capture import (
    CLOCK_TOLERANCE_MHZ,
    CaptureRecord,
    append_capture_records,
    build_capture_records,
    clock_ok_for,
    load_captures,
    resolve_target_mhz,
    target_mhz_from_hw_tag,
    write_audit_json,
)
from profiler.core.exec_metrics import compute_shot_exec_metrics


# -----------------------------------------------------------------------
# Synthetic helpers
# -----------------------------------------------------------------------

def _make_gpu_sample(wall_ts: float, power_w: float, mhz: int, util: float) -> dict:
    return {
        "wall_ts": wall_ts,
        "gpus": [{"index": 0, "power_w": power_w, "graphics_mhz": mhz, "util_gpu_pct": util}],
    }


def _write_power_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


@dataclass
class _FakeShot:
    requests: list
    experts: dict | None = None


@dataclass
class _FakeTimingSample:
    layer: str
    microseconds: float
    gating_ms: float | None = None
    expert_ms: float | None = None


@dataclass
class _FakeArgs:
    hardware: str = "V100_1100MHz"
    model: str = "Qwen/Qwen3-30B-A3B"
    dtype: str | None = "bfloat16"
    model_config: dict | None = None
    measurement_iterations: int = 3


# -----------------------------------------------------------------------
# gpu_power helpers
# -----------------------------------------------------------------------

class TestComputeAchievedMhz:
    def test_returns_median_of_busy_samples(self):
        samples = [
            _make_gpu_sample(0.0,  200.0, 1100, 80.0),  # busy
            _make_gpu_sample(0.1,  210.0, 1090, 75.0),  # busy
            _make_gpu_sample(0.2,   50.0,  900,  2.0),  # idle → excluded
            _make_gpu_sample(0.3,  195.0, 1110, 90.0),  # busy
        ]
        result = compute_achieved_mhz(samples)
        # Busy clocks: 1100, 1090, 1110 → sorted [1090, 1100, 1110] → median 1100
        assert result == 1100.0

    def test_returns_none_when_no_busy_samples(self):
        samples = [
            _make_gpu_sample(0.0, 50.0, 900, 2.0),
            _make_gpu_sample(0.1, 48.0, 900, 1.0),
        ]
        assert compute_achieved_mhz(samples) is None

    def test_returns_none_for_empty(self):
        assert compute_achieved_mhz([]) is None

    def test_even_count_median(self):
        samples = [
            _make_gpu_sample(0.0, 200.0, 1100, 80.0),
            _make_gpu_sample(0.1, 200.0, 1200, 80.0),
        ]
        result = compute_achieved_mhz(samples)
        assert result == 1150.0  # average of 1100 and 1200

    def test_custom_busy_threshold(self):
        samples = [
            _make_gpu_sample(0.0, 200.0, 1100, 5.0),   # below default 10 but above 4
            _make_gpu_sample(0.1,  50.0,  900, 2.0),   # idle even at 4
        ]
        assert compute_achieved_mhz(samples, busy_threshold=10.0) is None
        assert compute_achieved_mhz(samples, busy_threshold=4.0) == 1100.0


class TestComputeIdlePowerW:
    def test_mean_of_idle_samples(self):
        samples = [
            _make_gpu_sample(0.0, 50.0,  900,  2.0),   # idle
            _make_gpu_sample(0.1, 200.0, 1100, 80.0),  # busy → excluded
            _make_gpu_sample(0.2, 52.0,  900,  3.0),   # idle
        ]
        result = compute_idle_power_w(samples)
        assert result == pytest.approx(51.0, abs=0.01)

    def test_none_when_all_busy(self):
        samples = [_make_gpu_sample(0.0, 200.0, 1100, 80.0)]
        assert compute_idle_power_w(samples) is None

    def test_none_for_empty(self):
        assert compute_idle_power_w([]) is None


class TestComputePowerHz:
    def test_rate_from_timestamps(self):
        samples = [
            _make_gpu_sample(0.0,  50.0, 900, 0.0),
            _make_gpu_sample(0.1,  50.0, 900, 0.0),
            _make_gpu_sample(0.2,  50.0, 900, 0.0),
        ]
        hz = compute_power_hz(samples)
        assert hz == pytest.approx(10.0, rel=0.01)

    def test_none_for_single_sample(self):
        assert compute_power_hz([_make_gpu_sample(0.0, 50.0, 900, 0.0)]) is None

    def test_none_for_empty(self):
        assert compute_power_hz([]) is None


class TestIntegratePowerJoules:
    def test_trapezoidal(self):
        # Two samples 1 second apart at 100 W → 100 J
        samples = [
            _make_gpu_sample(0.0, 100.0, 900, 80.0),
            _make_gpu_sample(1.0, 100.0, 900, 80.0),
        ]
        result = integrate_power_joules(samples)
        assert result["energy_j"] == pytest.approx(100.0, rel=0.001)
        assert result["mean_power_w"] == pytest.approx(100.0, rel=0.001)
        assert result["sample_count"] == 2

    def test_trapezoidal_varying_power(self):
        # 0→1 s: avg = (100+200)/2 = 150 W → 150 J
        samples = [
            _make_gpu_sample(0.0, 100.0, 900, 80.0),
            _make_gpu_sample(1.0, 200.0, 900, 80.0),
        ]
        result = integrate_power_joules(samples)
        assert result["energy_j"] == pytest.approx(150.0, rel=0.001)

    def test_empty_returns_zero(self):
        # F2: 0-sample shot must return energy_j=None (not 0.0) so it is not
        # mistaken for a legitimate zero-energy measurement.
        result = integrate_power_joules([])
        assert result["energy_j"] is None
        assert result["sample_count"] == 0


# -----------------------------------------------------------------------
# capture helpers
# -----------------------------------------------------------------------

class TestTargetMhzFromHwTag:
    def test_parses_mhz_tag(self):
        assert target_mhz_from_hw_tag("V100_1100MHz") == 1100
        assert target_mhz_from_hw_tag("A100_900MHz") == 900
        assert target_mhz_from_hw_tag("H100_1410MHz") == 1410

    def test_returns_none_for_bare_tag(self):
        assert target_mhz_from_hw_tag("V100") is None
        assert target_mhz_from_hw_tag("H100") is None


class TestClockOkFor:
    def test_within_tolerance(self):
        assert clock_ok_for(1100.0, 1100, tolerance=25) is True
        assert clock_ok_for(1115.0, 1100, tolerance=25) is True
        assert clock_ok_for(1075.0, 1100, tolerance=25) is True

    def test_outside_tolerance(self):
        assert clock_ok_for(1130.0, 1100, tolerance=25) is False
        assert clock_ok_for(1050.0, 1100, tolerance=25) is False

    def test_none_when_no_target(self):
        assert clock_ok_for(1100.0, None) is None

    def test_none_when_no_achieved(self):
        assert clock_ok_for(None, 1100) is None


# -----------------------------------------------------------------------
# build_capture_records
# -----------------------------------------------------------------------

class TestBuildCaptureRecords:
    def _make_moe_shot(self, tokens=8, activated=4):
        return _FakeShot(
            requests=[(tokens, 0)],
            experts={"activated": activated},
        )

    def _make_dense_shot(self, tokens=32):
        return _FakeShot(requests=[(tokens, 0)], experts=None)

    def _make_exec_record(self, energy_j=10.5, mean_power_w=150.0, n_samples=20):
        return {
            "energy_j": energy_j,
            "power": {
                "total": {
                    "mean_power_w": mean_power_w,
                    "sample_count": n_samples,
                }
            },
        }

    def test_moe_record_fields(self, tmp_path):
        power_samples = [
            _make_gpu_sample(0.0, 200.0, 1100, 80.0),
            _make_gpu_sample(0.1, 210.0, 1095, 85.0),
            _make_gpu_sample(0.2,  50.0,  900,  2.0),  # idle
        ]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)

        timings = [_FakeTimingSample(layer="moe_block", microseconds=2500.0)]
        args = _FakeArgs(hardware="V100_1100MHz")
        exec_rec = self._make_exec_record()

        recs = build_capture_records(
            shot_key="moe_8_4",
            category_name="moe",
            shot=self._make_moe_shot(8, 4),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record=exec_rec,
            power_path=pp,
        )

        assert len(recs) == 1
        r = recs[0]
        assert r.category == "moe"
        assert r.tokens == 8
        assert r.activated_experts == 4
        assert r.layer is None  # MoE has no layer field
        assert r.latency_us == pytest.approx(2500.0)
        assert r.latency_std_us is None  # not available
        assert r.iterations == 3
        assert r.energy_j == pytest.approx(10.5)
        assert r.mean_power_w == pytest.approx(150.0)
        assert r.power_samples == 20
        assert r.target_mhz == 1100
        assert r.idle_power_w == pytest.approx(50.0, abs=0.01)
        # achieved_mhz: busy clocks are 1100 and 1095 → median 1097.5
        assert r.achieved_mhz == pytest.approx(1097.5, abs=1.0)
        # clock_ok: |1097.5 - 1100| = 2.5 ≤ 25 → True
        assert r.clock_ok is True

    def test_clock_fail_outside_tolerance(self, tmp_path):
        # Busy samples at 1200 MHz when target is 1100 → off by 100, fails
        power_samples = [_make_gpu_sample(0.0, 200.0, 1200, 80.0)]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="x", microseconds=100.0)]
        args = _FakeArgs(hardware="V100_1100MHz")
        recs = build_capture_records(
            shot_key="moe_1_1",
            category_name="moe",
            shot=self._make_moe_shot(1, 1),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record=self._make_exec_record(),
            power_path=pp,
        )
        assert recs[0].clock_ok is False
        assert abs(recs[0].achieved_mhz - 1100) > CLOCK_TOLERANCE_MHZ

    def test_dense_produces_one_record_per_layer(self, tmp_path):
        timings = [
            _FakeTimingSample(layer="qkv_proj", microseconds=100.0),
            _FakeTimingSample(layer="mlp", microseconds=200.0),
        ]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, [_make_gpu_sample(0.0, 150.0, 1100, 70.0)])
        args = _FakeArgs(hardware="V100_1100MHz")
        recs = build_capture_records(
            shot_key="dense_32",
            category_name="dense",
            shot=self._make_dense_shot(32),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record=self._make_exec_record(),
            power_path=pp,
        )
        assert len(recs) == 2
        layers = {r.layer for r in recs}
        assert layers == {"qkv_proj", "mlp"}
        for r in recs:
            assert r.activated_experts == 0

    def test_per_sequence_tokens_is_num_sequences(self, tmp_path):
        shot = _FakeShot(requests=[(1, 0)] * 16, experts=None)  # 16 sequences
        timings = [_FakeTimingSample(layer="lm_head", microseconds=50.0)]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, [_make_gpu_sample(0.0, 100.0, 1100, 70.0)])
        args = _FakeArgs(hardware="V100")
        recs = build_capture_records(
            shot_key="seq_16",
            category_name="per_sequence",
            shot=shot,
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record=self._make_exec_record(),
            power_path=pp,
        )
        assert len(recs) == 1
        assert recs[0].tokens == 16
        # V100 (no MHz tag) → target_mhz=None → clock_ok=None
        assert recs[0].target_mhz is None
        assert recs[0].clock_ok is None

    def test_no_power_path(self, tmp_path):
        timings = [_FakeTimingSample(layer="moe_block", microseconds=300.0)]
        shot = _FakeShot(requests=[(4, 0)], experts={"activated": 2})
        args = _FakeArgs()
        recs = build_capture_records(
            shot_key="moe_4_2",
            category_name="moe",
            shot=shot,
            timings=timings,
            arch=None,
            tp=2,
            args=args,
            exec_record={"energy_j": None, "power": {}},
            power_path=None,
        )
        assert len(recs) == 1
        r = recs[0]
        assert r.achieved_mhz is None
        assert r.clock_ok is None  # no achieved → None
        assert r.idle_power_w is None
        assert r.power_hz is None


# -----------------------------------------------------------------------
# append_capture_records / load_captures roundtrip
# -----------------------------------------------------------------------

class TestCaptureRecordIO:
    def _make_record(self, tokens=8, activated=4, latency_us=2500.0) -> CaptureRecord:
        return CaptureRecord(
            device="V100_1100MHz",
            model="test/model",
            tp=1,
            dtype="bfloat16",
            category="moe",
            layer=None,
            tokens=tokens,
            activated_experts=activated,
            target_mhz=1100,
            achieved_mhz=1098.5,
            clock_ok=True,
            latency_us=latency_us,
            latency_std_us=None,
            iterations=3,
            energy_j=10.5,
            mean_power_w=150.0,
            power_samples=20,
            power_hz=10.0,
            idle_power_w=52.3,
            job_id="8080289",
            timestamp="2026-07-01T10:00:00+00:00",
        )

    def test_roundtrip(self, tmp_path):
        r1 = self._make_record(8, 4, 2500.0)
        r2 = self._make_record(16, 8, 4200.0)
        append_capture_records(tmp_path, [r1, r2])
        loaded = load_captures(tmp_path / "captures.jsonl")
        assert len(loaded) == 2
        assert loaded[0].tokens == 8
        assert loaded[0].activated_experts == 4
        assert loaded[0].latency_us == pytest.approx(2500.0)
        assert loaded[0].achieved_mhz == pytest.approx(1098.5)
        assert loaded[0].clock_ok is True
        assert loaded[1].tokens == 16

    def test_append_is_additive(self, tmp_path):
        r1 = self._make_record(8, 4, 2500.0)
        r2 = self._make_record(16, 8, 4200.0)
        append_capture_records(tmp_path, [r1])
        append_capture_records(tmp_path, [r2])  # second append
        loaded = load_captures(tmp_path / "captures.jsonl")
        assert len(loaded) == 2

    def test_load_empty_file(self, tmp_path):
        p = tmp_path / "captures.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_captures(p) == []

    def test_load_missing_file(self, tmp_path):
        assert load_captures(tmp_path / "nonexistent.jsonl") == []


# -----------------------------------------------------------------------
# audit.json
# -----------------------------------------------------------------------

class TestWriteAuditJson:
    def _make_base_record(self, achieved_mhz=1098.0, clock_ok=True) -> CaptureRecord:
        return CaptureRecord(
            device="V100_1100MHz", model="x", tp=1, dtype="bf16",
            category="moe", layer=None, tokens=8, activated_experts=4,
            target_mhz=1100, achieved_mhz=achieved_mhz, clock_ok=clock_ok,
            latency_us=2500.0, latency_std_us=None, iterations=3,
            energy_j=10.0, mean_power_w=150.0, power_samples=20,
            power_hz=10.0, idle_power_w=52.0,
            job_id="local", timestamp="2026-07-01T00:00:00+00:00",
        )

    def test_verdict_ok(self, tmp_path):
        r = self._make_base_record(1098.0, True)
        append_capture_records(tmp_path, [r])
        audit = write_audit_json(tmp_path, target_mhz=1100)
        assert audit["verdict_ok"] is True
        assert audit["target_mhz"] == 1100
        assert audit["clock_ok_records"] == 1
        assert audit["clock_fail_records"] == 0

    def test_verdict_fail_when_clock_off(self, tmp_path):
        r = self._make_base_record(1200.0, False)
        append_capture_records(tmp_path, [r])
        audit = write_audit_json(tmp_path, target_mhz=1100)
        assert audit["verdict_ok"] is False

    def test_no_target_always_passes(self, tmp_path):
        r = self._make_base_record()
        r = CaptureRecord(**{**vars(r), "target_mhz": None, "clock_ok": None})
        append_capture_records(tmp_path, [r])
        audit = write_audit_json(tmp_path, target_mhz=None)
        assert audit["verdict_ok"] is True

    def test_audit_json_written_to_disk(self, tmp_path):
        r = self._make_base_record()
        append_capture_records(tmp_path, [r])
        write_audit_json(tmp_path, target_mhz=1100)
        assert (tmp_path / "audit.json").is_file()
        data = json.loads((tmp_path / "audit.json").read_text())
        assert "verdict_ok" in data


# -----------------------------------------------------------------------
# build_tables_from_captures.py builder round-trip
# -----------------------------------------------------------------------

class TestBuildTablesFromCaptures:
    def _make_moe_record(self, tokens, activated, latency_us, achieved_mhz,
                          energy_j=10.0, mean_power_w=150.0, idle_power_w=52.0) -> CaptureRecord:
        return CaptureRecord(
            device="V100_1100MHz", model="test/model", tp=1, dtype="bf16",
            category="moe", layer=None,
            tokens=tokens, activated_experts=activated,
            target_mhz=1100, achieved_mhz=achieved_mhz, clock_ok=True,
            latency_us=latency_us, latency_std_us=None, iterations=3,
            energy_j=energy_j, mean_power_w=mean_power_w, power_samples=20,
            power_hz=10.0, idle_power_w=idle_power_w,
            job_id="local", timestamp="2026-07-01T00:00:00+00:00",
        )

    def _make_dense_record(self, layer, tokens, latency_us, achieved_mhz) -> CaptureRecord:
        return CaptureRecord(
            device="V100_1100MHz", model="test/model", tp=1, dtype="bf16",
            category="dense", layer=layer,
            tokens=tokens, activated_experts=0,
            target_mhz=1100, achieved_mhz=achieved_mhz, clock_ok=True,
            latency_us=latency_us, latency_std_us=None, iterations=3,
            energy_j=5.0, mean_power_w=100.0, power_samples=10,
            power_hz=10.0, idle_power_w=50.0,
            job_id="local", timestamp="2026-07-01T00:00:00+00:00",
        )

    def test_moe_csv_row(self, tmp_path):
        """Builder produces correct moe.csv row from MoE captures."""
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from build_tables_from_captures import build_tables

        r = self._make_moe_record(8, 4, 2500.0, 1100.0,
                                   energy_j=10.0, mean_power_w=150.0, idle_power_w=52.0)
        captures_path = tmp_path / "captures.jsonl"
        append_capture_records(tmp_path, [r])

        produced = build_tables(captures_path, tmp_path)
        assert "moe.csv" in produced

        rows = list(_read_csv(produced["moe.csv"]))
        assert len(rows) == 1
        row = rows[0]
        assert int(row["tokens"]) == 8
        assert int(row["activated_experts"]) == 4
        assert float(row["time_us"]) == pytest.approx(2500.0)
        assert float(row["achieved_mhz"]) == pytest.approx(1100.0)

    def test_power_by_mhz_row(self, tmp_path):
        """Builder produces power_by_mhz.csv with correct mean watts."""
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from build_tables_from_captures import build_tables

        records = [
            self._make_moe_record(8, 4, 2500.0, 1100.0, mean_power_w=150.0, idle_power_w=52.0),
            self._make_moe_record(16, 8, 4000.0, 1100.0, mean_power_w=160.0, idle_power_w=51.0),
        ]
        captures_path = tmp_path / "captures.jsonl"
        append_capture_records(tmp_path, records)

        produced = build_tables(captures_path, tmp_path)
        assert "power_by_mhz.csv" in produced

        rows = list(_read_csv(produced["power_by_mhz.csv"]))
        # Both records round to 1100 MHz bucket
        assert len(rows) == 1
        row = rows[0]
        # F1: columns renamed to match dvfs-policy PowerTable consumer
        assert int(row["mhz"]) == 1100
        assert float(row["watts"]) == pytest.approx(155.0, rel=0.01)
        assert float(row["idle_watts"]) == pytest.approx(51.5, rel=0.01)
        assert int(row["n_samples"]) == 2

    def test_dense_csv_row(self, tmp_path):
        """Builder produces dense.csv with layer, tokens, time_us, achieved_mhz."""
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from build_tables_from_captures import build_tables

        records = [
            self._make_dense_record("qkv_proj", 32, 120.0, 1100.0),
            self._make_dense_record("mlp", 32, 200.0, 1100.0),
        ]
        captures_path = tmp_path / "captures.jsonl"
        append_capture_records(tmp_path, records)

        produced = build_tables(captures_path, tmp_path)
        assert "dense.csv" in produced

        rows = {r["layer"]: r for r in _read_csv(produced["dense.csv"])}
        assert "qkv_proj" in rows
        assert float(rows["qkv_proj"]["time_us"]) == pytest.approx(120.0)
        assert float(rows["mlp"]["time_us"]) == pytest.approx(200.0)

    def test_empty_captures_returns_empty(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from build_tables_from_captures import build_tables

        captures_path = tmp_path / "captures.jsonl"
        captures_path.write_text("", encoding="utf-8")
        produced = build_tables(captures_path, tmp_path)
        assert produced == {}

    def test_builder_averages_duplicate_keys(self, tmp_path):
        """Two records with same (tokens, activated_experts) → averaged time_us."""
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from build_tables_from_captures import build_tables

        records = [
            self._make_moe_record(8, 4, 2400.0, 1100.0),
            self._make_moe_record(8, 4, 2600.0, 1100.0),
        ]
        captures_path = tmp_path / "captures.jsonl"
        append_capture_records(tmp_path, records)

        produced = build_tables(captures_path, tmp_path)
        rows = list(_read_csv(produced["moe.csv"]))
        assert len(rows) == 1
        assert float(rows[0]["time_us"]) == pytest.approx(2500.0, rel=0.001)


def _read_csv(path: Path):
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------
# F8 — integration tests (prevents F1/F2/F4 regressions)
# -----------------------------------------------------------------------

class TestF8Integration:
    """Integration tests added by F8 to lock in F1, F2, and F4 fixes."""

    # ---- F1 regression: power_by_mhz.csv columns match the policy consumer ----

    def test_power_csv_policy_consumer(self, tmp_path):
        """power_by_mhz.csv must have mhz/watts/idle_watts so PowerTable parses it."""
        # Add dvfs-policy to path so we can import the real policy consumer.
        dvfs_policy_dir = "/data/engs-glass/engs2950/DVFS-MoE/dvfs-policy"
        sys.path.insert(0, dvfs_policy_dir)
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from throughput import PowerTable  # noqa: PLC0415
        from build_tables_from_captures import build_tables  # noqa: PLC0415

        # Two records at distinct clocks with known power.
        records = [
            CaptureRecord(
                device="V100_1100MHz", model="test/model", tp=1, dtype="bf16",
                category="moe", layer=None, tokens=8, activated_experts=4,
                target_mhz=1100, achieved_mhz=1100.0, clock_ok=True,
                latency_us=2500.0, latency_std_us=None, iterations=3,
                energy_j=10.0, mean_power_w=150.0, power_samples=20,
                power_hz=10.0, idle_power_w=52.0,
                job_id="local", timestamp="2026-07-01T00:00:00+00:00",
            ),
            CaptureRecord(
                device="V100_900MHz", model="test/model", tp=1, dtype="bf16",
                category="moe", layer=None, tokens=8, activated_experts=4,
                target_mhz=900, achieved_mhz=900.0, clock_ok=True,
                latency_us=3000.0, latency_std_us=None, iterations=3,
                energy_j=8.0, mean_power_w=120.0, power_samples=20,
                power_hz=10.0, idle_power_w=45.0,
                job_id="local", timestamp="2026-07-01T00:00:00+00:00",
            ),
        ]
        append_capture_records(tmp_path, records)

        produced = build_tables(tmp_path / "captures.jsonl", tmp_path)
        assert "power_by_mhz.csv" in produced

        # PowerTable must parse without KeyError — this is the F1 regression check.
        pt = PowerTable(produced["power_by_mhz.csv"])
        assert pt.has_measured_idle

        # Exact lookup at the two profiled clocks.
        w_1100 = pt.watts(1100)
        iw_1100 = pt.idle_watts(1100)
        assert w_1100 == pytest.approx(150.0, rel=0.01)
        assert iw_1100 == pytest.approx(52.0, rel=0.01)

        w_900 = pt.watts(900)
        iw_900 = pt.idle_watts(900)
        assert w_900 == pytest.approx(120.0, rel=0.01)
        assert iw_900 == pytest.approx(45.0, rel=0.01)

    # ---- F2 regression: single-sample shot → energy_j is None ----

    def test_single_sample_energy_is_none(self):
        """1-sample (or 0-sample) shot must yield energy_j=None, not 0.0."""
        single = integrate_power_joules([_make_gpu_sample(0.0, 100.0, 1100, 80.0)])
        assert single["energy_j"] is None
        assert single["sample_count"] == 1
        assert single["mean_power_w"] is None  # no interval, so no mean

        empty = integrate_power_joules([])
        assert empty["energy_j"] is None
        assert empty["sample_count"] == 0

    # ---- F4: has_power True / False cases ----

    def test_has_power_true_when_fully_instrumented(self, tmp_path):
        """has_power=True when power_samples>0, achieved_mhz not None, energy_j not None."""
        power_samples = [
            _make_gpu_sample(0.0, 200.0, 1100, 80.0),
            _make_gpu_sample(0.1, 210.0, 1095, 85.0),
        ]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="moe_block", microseconds=2500.0)]
        args = _FakeArgs(hardware="V100_1100MHz")
        recs = build_capture_records(
            shot_key="moe_8_4",
            category_name="moe",
            shot=_FakeShot(requests=[(8, 0)], experts={"activated": 4}),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record={
                "energy_j": 10.5,
                "power": {"total": {"mean_power_w": 150.0, "sample_count": 2}},
            },
            power_path=pp,
        )
        assert recs[0].has_power is True

    def test_has_power_false_when_no_power_path(self, tmp_path):
        """has_power=False when power_path is None (timing-only record)."""
        timings = [_FakeTimingSample(layer="moe_block", microseconds=2500.0)]
        args = _FakeArgs(hardware="V100")
        recs = build_capture_records(
            shot_key="moe_4_2",
            category_name="moe",
            shot=_FakeShot(requests=[(4, 0)], experts={"activated": 2}),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record={"energy_j": None, "power": {}},
            power_path=None,
        )
        assert recs[0].has_power is False

    def test_has_power_false_when_energy_j_none(self, tmp_path):
        """has_power=False when energy_j is None (e.g. single-sample power window)."""
        # One busy power sample → achieved_mhz is set, but only 1 sample →
        # the exec_record energy_j comes back None (F2 behaviour).
        power_samples = [_make_gpu_sample(0.0, 200.0, 1100, 80.0)]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="moe_block", microseconds=500.0)]
        args = _FakeArgs(hardware="V100_1100MHz")
        recs = build_capture_records(
            shot_key="moe_4_2",
            category_name="moe",
            shot=_FakeShot(requests=[(4, 0)], experts={"activated": 2}),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            # energy_j=None simulates a single-sample power window (F2).
            exec_record={"energy_j": None, "power": {}},
            power_path=pp,
        )
        assert recs[0].has_power is False

    def test_has_power_serializes_to_jsonl(self, tmp_path):
        """has_power round-trips through JSONL serialization."""
        power_samples = [
            _make_gpu_sample(0.0, 200.0, 1100, 80.0),
            _make_gpu_sample(0.1, 210.0, 1095, 85.0),
        ]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="moe_block", microseconds=2500.0)]
        args = _FakeArgs(hardware="V100_1100MHz")
        recs = build_capture_records(
            shot_key="moe_8_4",
            category_name="moe",
            shot=_FakeShot(requests=[(8, 0)], experts={"activated": 4}),
            timings=timings,
            arch=None,
            tp=1,
            args=args,
            exec_record={
                "energy_j": 10.5,
                "power": {"total": {"mean_power_w": 150.0, "sample_count": 2}},
            },
            power_path=pp,
        )
        append_capture_records(tmp_path, recs)
        loaded = load_captures(tmp_path / "captures.jsonl")
        assert len(loaded) == 1
        assert loaded[0].has_power is True


# -----------------------------------------------------------------------
# FIX #1 — resolve_target_mhz: GPU_FREQ_MHZ env-var fallback
# -----------------------------------------------------------------------

class TestResolveTargetMhz:
    """resolve_target_mhz threads the target clock even when the hw tag is bare.

    Root cause of the bug: run_arc_v100_profile.sh sets
    ``HARDWARE="${HARDWARE:-V100_${GPU_FREQ_MHZ}MHz}"`` only when HARDWARE is
    NOT already exported.  When HARDWARE=V100 is pre-set and GPU_FREQ_MHZ=900,
    the MHz suffix is absent from the tag and target_mhz_from_hw_tag returns
    None.  resolve_target_mhz falls back to the env var to recover the target.
    """

    def test_hw_tag_with_mhz_suffix_parsed_directly(self):
        assert resolve_target_mhz("V100_900MHz") == 900
        assert resolve_target_mhz("V100_1100MHz") == 1100
        assert resolve_target_mhz("H100_1410MHz") == 1410

    def test_env_var_fallback_when_hw_tag_is_bare(self):
        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "900"}):
            assert resolve_target_mhz("V100") == 900

    def test_env_var_fallback_with_other_bare_tags(self):
        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "1100"}):
            assert resolve_target_mhz("A100") == 1100

    def test_hw_tag_wins_over_gpu_freq_mhz_env(self):
        """When hw_tag carries MHz, it takes precedence over the env var."""
        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "900"}):
            assert resolve_target_mhz("V100_1100MHz") == 1100

    def test_returns_none_when_no_tag_and_no_env(self):
        env_without_freq = {k: v for k, v in os.environ.items() if k != "GPU_FREQ_MHZ"}
        with unittest.mock.patch.dict(os.environ, env_without_freq, clear=True):
            assert resolve_target_mhz("V100") is None
            assert resolve_target_mhz("H100") is None

    def test_ignores_invalid_env_value(self):
        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "notanumber"}):
            assert resolve_target_mhz("V100") is None

    def test_ignores_zero_env_value(self):
        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "0"}):
            assert resolve_target_mhz("V100") is None


class TestBuildCaptureRecordsEnvTarget:
    """build_capture_records sets target_mhz + clock_ok from GPU_FREQ_MHZ env."""

    def test_target_mhz_and_clock_ok_from_env_when_hw_tag_bare(self, tmp_path):
        """When hardware="V100" (bare) and GPU_FREQ_MHZ=900, target_mhz=900, clock_ok set."""
        power_samples = [
            _make_gpu_sample(0.0, 200.0, 900, 80.0),
            _make_gpu_sample(0.1, 210.0, 895, 85.0),
        ]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="moe_block", microseconds=617000.0)]
        args = _FakeArgs(hardware="V100")   # bare tag — no MHz suffix
        exec_rec = {
            "energy_j": 5.0,
            "power": {"total": {"mean_power_w": 100.0, "sample_count": 2}},
        }

        with unittest.mock.patch.dict(os.environ, {"GPU_FREQ_MHZ": "900"}):
            recs = build_capture_records(
                shot_key="moe_8_4",
                category_name="moe",
                shot=_FakeShot(requests=[(8, 0)], experts={"activated": 4}),
                timings=timings,
                arch=None,
                tp=1,
                args=args,
                exec_record=exec_rec,
                power_path=pp,
            )

        r = recs[0]
        # target_mhz must be read from env, NOT left as None
        assert r.target_mhz == 900
        # achieved clocks: 900, 895 → median 897.5; |897.5 - 900| = 2.5 ≤ 25
        assert r.clock_ok is True

    def test_target_mhz_none_when_no_tag_and_no_env(self, tmp_path):
        """Bare tag + no GPU_FREQ_MHZ → target_mhz=None (genuinely uncapped run)."""
        power_samples = [_make_gpu_sample(0.0, 200.0, 1100, 80.0)]
        pp = tmp_path / "shot.jsonl"
        _write_power_jsonl(pp, power_samples)
        timings = [_FakeTimingSample(layer="moe_block", microseconds=500.0)]
        args = _FakeArgs(hardware="V100")
        env_without_freq = {k: v for k, v in os.environ.items() if k != "GPU_FREQ_MHZ"}

        with unittest.mock.patch.dict(os.environ, env_without_freq, clear=True):
            recs = build_capture_records(
                shot_key="moe_4_2",
                category_name="moe",
                shot=_FakeShot(requests=[(4, 0)], experts={"activated": 2}),
                timings=timings,
                arch=None,
                tp=1,
                args=args,
                exec_record={"energy_j": None, "power": {}},
                power_path=pp,
            )

        r = recs[0]
        assert r.target_mhz is None
        assert r.clock_ok is None   # no target → verdict deferred

    def test_audit_json_reflects_env_target(self, tmp_path):
        """write_audit_json uses the env-sourced target so verdict is meaningful."""
        # Simulate a run where GPU_FREQ_MHZ=900, HARDWARE="V100" (bare tag),
        # and the GPU actually held 900 MHz.
        rec = CaptureRecord(
            device="V100", model="test/m", tp=1, dtype="fp16",
            category="moe", layer=None, tokens=8, activated_experts=4,
            target_mhz=900, achieved_mhz=900.0, clock_ok=True,
            latency_us=617000.0, latency_std_us=None, iterations=3,
            energy_j=5.0, mean_power_w=100.0, power_samples=20,
            power_hz=23.0, idle_power_w=45.0,
            job_id="local", timestamp="2026-07-01T00:00:00+00:00",
        )
        append_capture_records(tmp_path, [rec])
        audit = write_audit_json(tmp_path, target_mhz=900)
        assert audit["verdict_ok"] is True
        assert audit["target_mhz"] == 900
        assert "no target clock" not in audit["verdict_reason"]


# -----------------------------------------------------------------------
# FIX #2 — energy window: power clipped to compute window, not full span
# -----------------------------------------------------------------------

class TestEnergyComputeWindowBoundary:
    """energy_j must be integrated over [measured_wall_start, measured_wall_end].

    Root cause: in extension.py the worker previously set measured_wall_end
    AFTER layerwise_profile.__exit__() post-processed Kineto data, adding
    ~0.3–0.7 s of non-compute time.  The fix moves measured_wall_end to
    immediately after the measurement loop (inside the with-block), so the
    window covers only the timed compute iterations.

    These tests verify that exec_metrics._clip_samples_to_window correctly
    restricts integration to the declared [start, end] window, so that once
    extension.py emits tight timestamps the energy figure is accurate.
    """

    def test_energy_clips_to_measured_wall_window(self):
        """Samples outside [measured_wall_start, measured_wall_end] are excluded."""
        # Timeline:
        #   t=0.00: power sampler starts (idle — GPU not yet computing)
        #   t=0.50: measured_wall_start (compute begins)
        #   t=2.35: measured_wall_end   (compute ends, 1.85 s window)
        #   t=2.55: power sampler stops (profiler post-processing / idle)
        idle_w = 50.0
        compute_w = 200.0
        pre_idle = [_make_gpu_sample(t, idle_w, 900, 2.0) for t in [0.0, 0.25]]
        compute_samples = [
            _make_gpu_sample(t, compute_w, 900, 80.0)
            for t in [0.50, 1.00, 1.50, 2.00, 2.35]
        ]
        post_samples = [_make_gpu_sample(2.55, idle_w, 900, 2.0)]
        all_samples = pre_idle + compute_samples + post_samples

        fire_timing = {
            "measured_wall_start": 0.50,
            "measured_wall_end": 2.35,     # tight window (post-fix value)
            "measured_sec": 2.55,           # inflated (includes profiler overhead)
            "warmup_sec": 0.30,
            "fire_total_sec": 3.0,
            "barrier_wait_sec": 0.0,
        }

        result = compute_shot_exec_metrics(
            fire_timing,
            shot_markers=[],
            power_samples=all_samples,
        )

        # Full-window integration (pre-bug): would include idle power outside compute.
        full_energy = integrate_power_joules(all_samples)["energy_j"]
        # Clipped-window integration (post-fix): only compute samples.
        assert result["energy_j"] is not None
        assert result["energy_j"] < full_energy, (
            "energy_j should be lower when clipped to the compute window; "
            "pre-idle and post-sampling samples add ~50 W × 0.7 s ≈ 35 J"
        )

        # The clipped duration must be close to 1.85 s (0.50 → 2.35).
        dur = result["power"]["total"]["duration_sec"]
        assert dur == pytest.approx(1.85, abs=0.05), (
            f"clipped duration {dur:.3f}s should be ~1.85s (compute window)"
        )

    def test_clipped_energy_approximately_correct(self):
        """Energy over the compute window is close to P × dt at constant power."""
        # Constant 200 W over 1.85 s → expect ~370 J.
        compute_w = 200.0
        ts = [0.50 + i * (1.85 / 4) for i in range(5)]  # 5 samples spanning 1.85 s
        compute_samples = [_make_gpu_sample(t, compute_w, 900, 80.0) for t in ts]
        # Add idle bookends outside the window.
        samples = (
            [_make_gpu_sample(0.0, 50.0, 900, 2.0)]
            + compute_samples
            + [_make_gpu_sample(3.0, 50.0, 900, 2.0)]
        )

        fire_timing = {
            "measured_wall_start": 0.50,
            "measured_wall_end": ts[-1],   # = 2.35
            "measured_sec": 3.0,
            "warmup_sec": 0.30,
            "fire_total_sec": 3.5,
            "barrier_wait_sec": 0.0,
        }

        result = compute_shot_exec_metrics(
            fire_timing,
            shot_markers=[],
            power_samples=samples,
        )

        expected_j = compute_w * 1.85   # 370 J
        assert result["energy_j"] == pytest.approx(expected_j, rel=0.02), (
            f"expected ~{expected_j:.0f} J, got {result['energy_j']:.1f} J"
        )

    def test_no_clip_when_wall_timestamps_absent(self):
        """When measured_wall_start/end are absent, all power samples are integrated."""
        samples = [
            _make_gpu_sample(0.0, 200.0, 900, 80.0),
            _make_gpu_sample(1.0, 200.0, 900, 80.0),
        ]
        fire_timing = {
            # deliberately omit measured_wall_start / measured_wall_end
            "measured_sec": 1.0,
            "warmup_sec": 0.0,
            "fire_total_sec": 1.0,
            "barrier_wait_sec": 0.0,
        }
        result = compute_shot_exec_metrics(
            fire_timing,
            shot_markers=[],
            power_samples=samples,
        )
        # Without clipping, the full 2-sample window (1 s at 200 W = 200 J) is used.
        assert result["energy_j"] == pytest.approx(200.0, rel=0.001)
