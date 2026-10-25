import numpy as np

from src.backtest import run_backtest
from src.scenarios import make_scenario
from tests.test_backtest import fake_close


def test_depth_zero_is_the_same_as_no_decay():
    res = run_backtest(fake_close(n=3000))
    a = make_scenario(res, n_days=1000, break_idx=500, depth=0.0, ramp_days=250, seed=4)
    b = make_scenario(res, n_days=1000, break_idx=None, seed=4)
    assert np.array_equal(a.strat_ret, b.strat_ret)
