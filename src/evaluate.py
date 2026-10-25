from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import summarize
from src.policies import cusum_stat, size_from_cusum, stoploss_size, policy_returns
from src.scenarios import make_scenario


def reference_stats(res: pd.DataFrame) -> tuple[float, float]:
    """Research-period edge and noise: mean and std of strategy return on exposed days."""
    r = res.loc[res["position"] > 0, "strat_ret"]
    return float(r.mean()), float(r.std())


def detection_sweep(res, h_grid, n_seeds: int = 200, seed0: int = 0, **scenario_kw) -> pd.DataFrame:
    """false_alarm_rate: share of runs with any alarm before the break (whole path if no decay).
    detect_rate / median_delay_days: among runs NOT already in alarm at the break,
    the share that alarm afterwards, and the median days to the first alarm."""
    mu0, sigma = reference_stats(res)
    stats = {h: {"fa": 0, "clean": 0, "delays": []} for h in h_grid}
    for seed in range(seed0, seed0 + n_seeds):
        sc = make_scenario(res, seed=seed, **scenario_kw)
        b = sc.break_idx
        s = cusum_stat(sc.strat_ret, sc.position, mu0, sigma)
        for h in h_grid:
            st = stats[h]
            if b is None:
                st["fa"] += bool((s >= h).any())
                continue
            st["fa"] += bool((s[:b] >= h).any())
            if s[b - 1] < h:
                st["clean"] += 1
                hits = np.flatnonzero(s[b:] >= h)
                if len(hits):
                    st["delays"].append(int(hits[0]))
    rows = []
    for h in h_grid:
        st = stats[h]
        d = st["delays"]
        rows.append({
            "h": h,
            "false_alarm_rate": st["fa"] / n_seeds,
            "detect_rate": len(d) / st["clean"] if st["clean"] else np.nan,
            "median_delay_days": float(np.median(d)) if d else np.nan,
        })
    return pd.DataFrame(rows)


def compare_policies(res, h: float, n_seeds: int = 200, seed0: int = 0,
                     dd_limit: float = 0.15, cooldown: int = 126, **scenario_kw) -> pd.DataFrame:
    mu0, sigma = reference_stats(res)
    rows = []
    for seed in range(seed0, seed0 + n_seeds):
        sc = make_scenario(res, seed=seed, **scenario_kw)
        s = cusum_stat(sc.strat_ret, sc.position, mu0, sigma)
        sizes = {
            "unmonitored": np.ones(len(s)),
            "stop-loss": stoploss_size(sc.strat_ret, dd_limit, cooldown),
            "detector": size_from_cusum(s, h),
        }
        for name, sz in sizes.items():
            m = summarize(pd.Series(policy_returns(sc.strat_ret, sc.position, sz)))
            rows.append({"policy": name, "seed": seed, **m})
    return pd.DataFrame(rows)


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("policy")
    out = pd.DataFrame({
        "median_total_ret": g["total_return"].median(),
        "p5_total_ret": g["total_return"].quantile(0.05),
        "median_sharpe": g["sharpe"].median(),
        "median_max_dd": g["max_drawdown"].median(),
        "p5_max_dd": g["max_drawdown"].quantile(0.05),   # the worst 5% of runs
    })
    wide = df.pivot(index="seed", columns="policy", values="total_return")
    out["beats_unmonitored"] = [
        np.nan if p == "unmonitored" else (wide[p] > wide["unmonitored"]).mean()
        for p in out.index
    ]
    return out.round(3)
