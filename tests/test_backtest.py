import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest


def fake_close(n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx)


def test_no_lookahead():
    """A shock on day d must not change the position held on or before day d."""
    close = fake_close()
    base = run_backtest(close)
    flat_days = base.index[base["position"] == 0]
    assert len(flat_days) > 0
    d = flat_days[len(flat_days) // 2]
    shocked = close.copy()
    shocked.loc[d:] *= 1.5          # +50% jump on day d that persists
    alt = run_backtest(shocked)
    pd.testing.assert_series_equal(
        base.loc[:d, "position"], alt.loc[:d, "position"]
    )


def test_always_long_equals_buy_and_hold():
    close = fake_close()
    sig = pd.Series(1.0, index=close.index)
    res = run_backtest(close, signal=sig, cost_bps=0)
    assert res["equity"].iloc[-1] == pytest.approx(res["bh_equity"].iloc[-1])


def test_single_switch_costs_one_trade():
    close = fake_close()
    sig = pd.Series(np.where(np.arange(len(close)) < 700, 0.0, 1.0), index=close.index)
    res = run_backtest(close, signal=sig, cost_bps=5)
    assert (res["turnover"] > 0).sum() == 1
    assert res["cost"].sum() == pytest.approx(5e-4)