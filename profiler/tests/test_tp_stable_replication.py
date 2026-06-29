"""Regression tests for tp_stable replication in a tp1-free session.

These tests exercise ``replicate_tp_stable`` directly (no vLLM/GPU
required) using a synthetic architecture and temporary CSV fixtures.

Scenarios covered
-----------------
(a) tp4-only replication SUCCEEDS when a complete on-disk tp1/ exists:
    dense and per_sequence tp4 CSVs get populated from tp1 stable rows
    while non-stable rows already present in tp4 are preserved.

(b) tp4-only RAISES RuntimeError when tp1/ is absent.

(c) persist_meta TP-union: when args.tp_degrees==[4] but tp1/ already
    exists on disk, meta.yaml records tp_degrees==[1,4] and
    skew_fit.per_tp contains entries for both 1 and 4.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from profiler.core.config import Architecture, Catalog, LayerEntry
from profiler.core.writer import (
    _on_disk_tp_dirs,
    _skew_fit_block,
    replicate_tp_stable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch() -> Architecture:
    """Synthetic arch with:
    - dense: 'layernorm' (tp_stable=True), 'gate_up' (tp_stable=False)
    - per_sequence: 'embed' (tp_stable=True)
    - attention: 'attn' (tp_stable=False) — required by validator
    - moe: (empty)
    """
    return Architecture(
        catalog=Catalog(
            dense={
                "layernorm": LayerEntry(vllm="RMSNorm", tp_stable=True),
                "gate_up":   LayerEntry(vllm="MergedColumnParallelLinear"),
            },
            per_sequence={
                "embed": LayerEntry(vllm="VocabParallelEmbedding", tp_stable=True),
            },
            attention={
                "attn": LayerEntry(vllm="Attention"),
            },
            moe={},
        )
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TpStableReplicationTest(unittest.TestCase):

    def test_tp4_only_succeeds_with_complete_tp1(self):
        """(a) Complete tp1/ on disk → tp4 gets tp_stable rows; non-stable
        tp4 rows are preserved unchanged."""
        arch = _make_arch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tp1 = root / "tp1"
            tp4 = root / "tp4"

            # tp1 source: both tp_stable and non-stable layers.
            _write_csv(tp1 / "dense.csv", [
                {"layer": "layernorm", "tokens": "1", "microseconds": "10.0"},
                {"layer": "gate_up",   "tokens": "1", "microseconds": "20.0"},
            ])
            _write_csv(tp1 / "per_sequence.csv", [
                {"layer": "embed", "sequences": "1", "microseconds": "5.0"},
            ])

            # tp4 already has its own gate_up row (non-stable, measured at tp4).
            _write_csv(tp4 / "dense.csv", [
                {"layer": "gate_up", "tokens": "1", "microseconds": "80.0"},
            ])
            # tp4/per_sequence.csv does not exist yet.

            # Should succeed without raising.
            replicate_tp_stable(root, arch, [4], require_tp1=True)

            # tp4/dense.csv must now have BOTH layernorm (replicated) and
            # gate_up (original tp4 value preserved).
            dense4 = {r["layer"]: r for r in _read_csv(tp4 / "dense.csv")}
            self.assertIn("layernorm", dense4,
                          "tp_stable layernorm must be replicated into tp4/dense.csv")
            self.assertEqual(dense4["layernorm"]["microseconds"], "10.0",
                             "replicated value must come from tp1, not tp4")
            self.assertIn("gate_up", dense4,
                          "non-stable gate_up row must be preserved in tp4/dense.csv")
            self.assertEqual(dense4["gate_up"]["microseconds"], "80.0",
                             "non-stable gate_up value must not be overwritten")

            # tp4/per_sequence.csv must now contain the embed row.
            self.assertTrue((tp4 / "per_sequence.csv").exists(),
                            "tp4/per_sequence.csv must be created by replication")
            seq4 = {r["layer"]: r for r in _read_csv(tp4 / "per_sequence.csv")}
            self.assertIn("embed", seq4,
                          "tp_stable embed must be replicated into tp4/per_sequence.csv")
            self.assertEqual(seq4["embed"]["microseconds"], "5.0")

    def test_tp4_only_raises_when_tp1_absent(self):
        """(b) Missing tp1/ directory → RuntimeError naming the path."""
        arch = _make_arch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Deliberately do NOT create tp1/.
            with self.assertRaises(RuntimeError) as ctx:
                replicate_tp_stable(root, arch, [4], require_tp1=True)
            self.assertIn("tp1", str(ctx.exception),
                          "error message must mention 'tp1'")
            self.assertIn(str(root), str(ctx.exception),
                          "error message must name the variant_root path")

    def test_tp4_only_raises_when_dense_csv_missing_stable_rows(self):
        """require_tp1=True raises when tp1/dense.csv exists but omits
        a tp_stable layer (layernorm missing from tp1 dense.csv)."""
        arch = _make_arch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tp1 = root / "tp1"
            # tp1/dense.csv only has gate_up; layernorm is absent.
            _write_csv(tp1 / "dense.csv", [
                {"layer": "gate_up", "tokens": "1", "microseconds": "20.0"},
            ])
            _write_csv(tp1 / "per_sequence.csv", [
                {"layer": "embed", "sequences": "1", "microseconds": "5.0"},
            ])
            with self.assertRaises(RuntimeError) as ctx:
                replicate_tp_stable(root, arch, [4], require_tp1=True)
            self.assertIn("layernorm", str(ctx.exception))

    def test_default_require_tp1_false_warns_and_returns(self):
        """Existing callers without require_tp1 keep the old warn-and-return
        behaviour when tp1/ is absent."""
        arch = _make_arch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # No tp1/ — with default require_tp1=False this must not raise.
            try:
                replicate_tp_stable(root, arch, [4])
            except RuntimeError as exc:
                self.fail(
                    f"replicate_tp_stable raised RuntimeError with "
                    f"require_tp1=False (default): {exc}"
                )


# ---------------------------------------------------------------------------
# Minimal valid skew.csv row (all columns required by fit_alpha.fit_alpha).
# kv_mean is used as kv_min when 'kvs' column is absent:
#   skew_rate = (kv_mean - kv_mean) / (kv_big - kv_mean) = 0
# t_max > t_mean > 0 so _fit_constant_wls gets a non-degenerate signal.
# ---------------------------------------------------------------------------
_SKEW_CSV_HEADER = [
    "alpha", "t_max_us", "t_mean_us", "t_skew_us",
    "pc", "n", "kv_big", "kp", "kv_mean",
]
_SKEW_CSV_ROW = {
    "alpha": "0.5",
    "t_max_us": "200.0",
    "t_mean_us": "100.0",
    "t_skew_us": "150.0",
    "pc": "16",
    "n": "8",
    "kv_big": "1024",
    "kp": "512",
    "kv_mean": "512",
}


def _write_skew_csv(path: Path) -> None:
    """Write a single-row skew.csv fixture sufficient to produce an
    ``enabled=True`` fit result from ``fit_alpha.fit_alpha``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SKEW_CSV_HEADER)
        writer.writeheader()
        writer.writerow(_SKEW_CSV_ROW)


class PersistMetaTpUnionTest(unittest.TestCase):
    """Tests for the ``_on_disk_tp_dirs`` helper and the tp-union logic
    in ``persist_meta`` (exercised via ``_skew_fit_block`` directly to
    avoid needing a GPU / vLLM engine).
    """

    def test_on_disk_tp_dirs_empty_when_no_tp_subdirs(self):
        """_on_disk_tp_dirs returns an empty set for a fresh directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(_on_disk_tp_dirs(root), set())

    def test_on_disk_tp_dirs_absent_root(self):
        """_on_disk_tp_dirs returns an empty set when variant_root does
        not exist yet (the normal state before the first session)."""
        with tempfile.TemporaryDirectory() as tmp:
            nonexistent = Path(tmp) / "does_not_exist"
            self.assertEqual(_on_disk_tp_dirs(nonexistent), set())

    def test_on_disk_tp_dirs_detects_existing_dirs(self):
        """_on_disk_tp_dirs finds tp1/ and tp4/ but ignores 'tpX/'."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tp1").mkdir()
            (root / "tp4").mkdir()
            (root / "tpX").mkdir()    # must be ignored (not integer)
            (root / "other").mkdir()  # must be ignored (wrong prefix)
            self.assertEqual(_on_disk_tp_dirs(root), {1, 4})

    def test_skew_fit_block_union_includes_both_tps(self):
        """When tp1/ and tp4/ both have valid skew.csv files, calling
        _skew_fit_block with effective_tps=[1,4] (the union) yields
        enabled=True and per_tp entries for BOTH 1 and 4 — even though
        the current session only profiled tp4 (args.tp_degrees==[4]).
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skew_csv(root / "tp1" / "skew.csv")
            _write_skew_csv(root / "tp4" / "skew.csv")

            # Simulate what persist_meta now does:
            #   effective_tps = sorted(set(args.tp_degrees) | on_disk_tps)
            # where args.tp_degrees == [4] and on_disk_tps == {1, 4}.
            on_disk = _on_disk_tp_dirs(root)
            effective_tps = sorted({4} | on_disk)

            self.assertEqual(effective_tps, [1, 4],
                             "union of {4} with {1,4} must be [1,4]")

            fit = _skew_fit_block(root, effective_tps)

            self.assertTrue(fit.get("enabled"),
                            "skew_fit block must be enabled when both skew.csv files exist")
            per_tp = fit.get("per_tp", {})
            self.assertIn(1, per_tp,
                          "per_tp must contain an entry for tp=1")
            self.assertIn(4, per_tp,
                          "per_tp must contain an entry for tp=4")

    def test_skew_fit_block_single_session_unchanged(self):
        """In a normal single-session run (only tp1/ on disk, args.tp_degrees==[1])
        the union is a no-op: effective_tps == [1] and per_tp only has tp=1.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skew_csv(root / "tp1" / "skew.csv")

            on_disk = _on_disk_tp_dirs(root)
            effective_tps = sorted({1} | on_disk)

            self.assertEqual(effective_tps, [1],
                             "single-session union must leave tp_degrees unchanged")

            fit = _skew_fit_block(root, effective_tps)

            self.assertTrue(fit.get("enabled"))
            per_tp = fit.get("per_tp", {})
            self.assertIn(1, per_tp)
            self.assertNotIn(4, per_tp)


if __name__ == "__main__":
    unittest.main()
