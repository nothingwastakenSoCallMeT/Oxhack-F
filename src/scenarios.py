from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    asset_ret: np.ndarray    # bootstrapped asset returns (benchmark line)
    position: np.ndarray     # paper-strategy exposure, 0/1
    strat_ret: np.ndarray    # paper-strategy returns AFTER edge decay
    edge_weight: np.ndarray  # ground truth: 1 = full edge, 0 = none, <0 = reversed
    break_idx: int | None    # first day of decay, None = no decay
    dates: pd.DatetimeIndex


def block_bootstrap_idx(n_rows: int, n_out: int, block: int, rng) -> np.ndarray:
    n_blocks = int(np.ceil(n_out / block))
    starts = rng.integers(0, n_rows - block, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n_out]


def edge_weights(n: int, break_idx: int | None, depth: float, ramp_days: int) -> np.ndarray:
    """Multiplier on the original edge: 1 before the break, falling to (1 - depth).
    depth=1 -> edge disappears; depth=2 -> edge fully reverses."""
    w = np.ones(n)
    if break_idx is None:
        return w
    t = np.arange(n) - break_idx
    frac = np.clip((t + 1) / max(ramp_days, 1), 0, 1)
    return 1 - depth * frac


def make_scenario(
    res: pd.DataFrame,          # output of run_backtest on the real data
    n_days: int = 2520,         # 10 years
    break_idx: int | None = 1260,
    depth: float = 1.0,
    ramp_days: int = 1,         # 1 = sudden, 250 = gradual over a year
    edge_scale: float = 1.0,    # >1 strengthens the pre-decay edge
    block: int = 40,
    seed: int = 0,
) -> Scenario:
    rng = np.random.default_rng(seed)
    m_on = res.loc[res["position"] > 0, "asset_ret"].mean()   # avg return on exposed days
    idx = block_bootstrap_idx(len(res), n_days, block, rng)

    asset_ret = res["asset_ret"].to_numpy()[idx]
    position = res["position"].to_numpy()[idx]
    strat = res["strat_ret"].to_numpy()[idx]

    w = edge_weights(n_days, break_idx, depth, ramp_days)
    strat = strat + position * m_on * (w * edge_scale - 1)

    dates = pd.bdate_range("2000-01-03", periods=n_days)
    return Scenario(asset_ret, position, strat, w * edge_scale, break_idx, dates)