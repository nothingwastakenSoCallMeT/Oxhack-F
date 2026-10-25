import numpy as np
import pytest

from src.policies import (
    applied_size, cusum_stat, policy_returns, size_from_cusum, stoploss_size,
)


def test_policies_are_causal():
    """A shock from day 800 must not change the size held on or before day 800."""
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0004, 0.008, 1500)
    pos = np.ones(1500)
    ret2 = ret.copy()
    ret2[800:] -= 0.05
    for fn in (
        lambda r: size_from_cusum(cusum_stat(r, pos, 0.0004, 0.008), 10),
        lambda r: stoploss_size(r),
    ):
        a1, a2 = applied_size(fn(ret)), applied_size(fn(ret2))
        assert np.array_equal(a1[:801], a2[:801])
        assert not np.array_equal(a1, a2)


def test_flat_days_do_not_move_cusum():
    ret = np.array([0.01, -0.02, 0.0, 0.0, -0.02])
    pos = np.array([1, 1, 0, 0, 1])
    s = cusum_stat(ret, pos, 0.001, 0.01)
    assert s[2] == s[1] and s[3] == s[1]


def test_size_recovers_when_statistic_falls():
    s = np.array([0, 5, 12, 12, 4, 0])
    assert list(size_from_cusum(s, h=10)) == [1, 0.5, 0, 0, 1, 1]


def test_size_change_costs_only_when_exposed():
    ret = np.zeros(6)
    decided = np.array([1, 0, 0, 0, 0, 0.0])   # cut decided at close of day 1, held from day 2
    pos = np.array([0, 0, 1, 1, 1, 1])          # exposed on day 2 -> pays
    r = policy_returns(ret, pos, decided, cost_bps=10)
    assert r[2] == pytest.approx(-10 / 1e4)
    assert (r[[0, 1, 3, 4, 5]] == 0).all()
    pos = np.array([0, 0, 0, 1, 1, 1])          # flat on day 2 -> free
    assert (policy_returns(ret, pos, decided, cost_bps=10) == 0).all()
