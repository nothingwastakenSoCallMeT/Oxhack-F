import numpy as np
import pytest

from src.backtest import run_backtest
from src.scenarios import make_scenario
from tests.test_backtest import fake_close


def test_no_decay_is_pure_bootstrap():
    res = run_backtest(fake_close(n=3000))
    sc = make_scenario(res, n_days=1000, break_idx=None, seed=3)
    assert np.all(sc.edge_weight == 1)
    # every strategy return must come straight from the real history
    assert set(np.round(sc.strat_ret, 12)) <= set(np.round(res["strat_ret"], 12))


def test_decay_removes_exposed_day_drift():
    res = run_backtest(fake_close(n=4000, seed=1))
    m_on = res.loc[res["position"] > 0, "asset_ret"].mean()
    pre, post = [], []
    for s in range(100):
        sc = make_scenario(res, n_days=1000, break_idx=500, depth=1.0, seed=s)
        on = sc.position > 0
        pre.append(sc.strat_ret[:500][on[:500]].mean())
        post.append(sc.strat_ret[500:][on[500:]].mean())
    assert np.mean(pre) == pytest.approx(m_on, abs=3e-4)
    assert abs(np.mean(post)) < 3e-4