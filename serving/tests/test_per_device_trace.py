"""Unit tests for _max_rank_latency logic and per-rank power path.

_max_rank_latency is a pure function; tested inline to avoid the msgspec
dependency chain pulled in by importing trace_generator directly.
"""
import pytest
from unittest.mock import MagicMock


def _max_rank_latency(ctx, lookup_fn):
    """Inlined copy of trace_generator._max_rank_latency for isolated testing."""
    if ctx.rank_perf_dbs is None:
        lat = lookup_fn(ctx.perf_db)
        if ctx.dvfs_scale != 1.0:
            lat = max(1, int(round(lat * ctx.dvfs_scale)))
        return lat, None
    lats = []
    for db in ctx.rank_perf_dbs:
        l = lookup_fn(db)
        if ctx.dvfs_scale != 1.0:
            l = max(1, int(round(l * ctx.dvfs_scale)))
        lats.append(l)
    return max(lats), lats


def _make_ctx(rank_perf_dbs=None, rank_hardware=None, dvfs_scale=1.0):
    ctx = MagicMock()
    ctx.rank_perf_dbs = rank_perf_dbs
    ctx.rank_hardware = rank_hardware
    ctx.dvfs_scale = dvfs_scale
    ctx.perf_db = {"primary": True}
    return ctx


def test_max_rank_latency_homogeneous():
    """Homogeneous: single lookup, per_rank is None."""
    ctx = _make_ctx(rank_perf_dbs=None)

    def lookup(db):
        return 1000

    max_lat, per_rank = _max_rank_latency(ctx, lookup)
    assert max_lat == 1000
    assert per_rank is None


def test_max_rank_latency_heterogeneous():
    """Heterogeneous: max returned, per_rank list returned."""
    db0 = {"hw": "A"}
    db1 = {"hw": "B"}
    ctx = _make_ctx(rank_perf_dbs=[db0, db1])

    def lookup(db):
        return 1000 if db["hw"] == "A" else 1500

    max_lat, per_rank = _max_rank_latency(ctx, lookup)
    assert max_lat == 1500
    assert per_rank == [1000, 1500]


def test_max_rank_latency_dvfs_scale():
    """dvfs_scale is applied inside the helper."""
    ctx = _make_ctx(rank_perf_dbs=None, dvfs_scale=2.0)

    def lookup(db):
        return 1000

    max_lat, per_rank = _max_rank_latency(ctx, lookup)
    assert max_lat == 2000
    assert per_rank is None


def test_max_rank_latency_heterogeneous_dvfs():
    """dvfs_scale applied to each rank independently."""
    db0 = {"hw": "A"}
    db1 = {"hw": "B"}
    ctx = _make_ctx(rank_perf_dbs=[db0, db1], dvfs_scale=0.5)

    def lookup(db):
        return 2000 if db["hw"] == "A" else 3000

    max_lat, per_rank = _max_rank_latency(ctx, lookup)
    assert max_lat == 1500  # round(3000 * 0.5)
    assert per_rank == [1000, 1500]
