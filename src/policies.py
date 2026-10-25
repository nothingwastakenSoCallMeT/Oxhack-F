from __future__ import annotations

import numpy as np


def cusum_stat(strat_ret, position, mu0: float, sigma: float) -> np.ndarray:
    """One-sided CUSUM for a fall in mean exposed-day return from mu0 toward 0.
    Flat days carry no information and leave the statistic unchanged.
    Units: multiples of sigma (per-exposed-day return std)."""
    k = mu0 / 2.0
    s = np.zeros(len(strat_ret))
    prev = 0.0
    for t in range(len(strat_ret)):
        if position[t] > 0:
            prev = max(0.0, prev + (k - strat_ret[t]) / sigma)
        s[t] = prev
    return s


def size_from_cusum(s: np.ndarray, h: float) -> np.ndarray:
    """Size decided at the close of day t: 100% -> 50% at h/2 -> 0% at h."""
    size = np.ones(len(s))
    size[s >= h / 2] = 0.5
    size[s >= h] = 0.0
    return size


def stoploss_size(strat_ret, dd_limit: float = 0.15, cooldown: int = 126) -> np.ndarray:
    """Drawdown stop on the paper strategy's equity. Out for `cooldown` days,
    then back in with the peak reset."""
    size = np.ones(len(strat_ret))
    eq = peak = 1.0
    out_until = -1
    for t in range(len(strat_ret)):
        eq *= 1.0 + strat_ret[t]
        if t <= out_until:
            size[t] = 0.0
            peak = eq
            continue
        peak = max(peak, eq)
        if eq / peak - 1.0 <= -dd_limit:
            size[t] = 0.0
            out_until = t + cooldown
    return size


def applied_size(size_decided: np.ndarray) -> np.ndarray:
    """Size decided at the close of t-1 is what is held on day t."""
    return np.concatenate([[1.0], size_decided[:-1]])


def policy_returns(strat_ret, position, size_decided, cost_bps: float = 5.0) -> np.ndarray:
    """Scaled strategy returns, plus a cost when size changes while the strategy is exposed.
    (Approximation: ignores the small interaction between size changes and strategy trades.)"""
    a = applied_size(size_decided)
    extra_cost = np.abs(np.diff(a, prepend=1.0)) * position * cost_bps / 1e4
    return a * strat_ret - extra_cost
