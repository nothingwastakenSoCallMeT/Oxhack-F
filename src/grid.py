from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import compare_policies

DEPTHS = [0.0, 0.5, 1.0, 1.5, 2.0]   # 0 = no decay, 1 = edge to zero, 2 = edge fully reversed
RAMPS = [1, 250]                     # 1 = sudden, 250 = gradual over about a year


def monitoring_grid(res, h: float, depths=DEPTHS, ramps=RAMPS, n_seeds: int = 200,
                    seed0: int = 0, break_idx: int = 1260) -> pd.DataFrame:
    """One row per (depth, ramp_days). For each policy, versus doing nothing:
      d_median_ret : median over seeds of (policy total return - unmonitored total return)
      d_p5_dd      : policy's worst-5% max drawdown minus the unmonitored one
                     (positive = shallower drawdown than doing nothing)
      win_rate     : share of seeds where the policy ends above unmonitored
    Positive d_median_ret and d_p5_dd always mean the policy helped."""
    rows, cache = [], {}
    for depth in depths:
        for ramp in ramps:
            key = (0.0, 1) if depth == 0 else (depth, ramp)   # ramp is irrelevant with no decay
            if key not in cache:
                cache[key] = compare_policies(res, h, n_seeds=n_seeds, seed0=seed0,
                                              break_idx=break_idx, depth=key[0], ramp_days=key[1])
            df = cache[key]
            ret = df.pivot(index="seed", columns="policy", values="total_return")
            dd = df.pivot(index="seed", columns="policy", values="max_drawdown")
            row = {"depth": depth, "ramp_days": ramp}
            for p in ("detector", "stop-loss"):
                row[f"{p}: d_median_ret"] = float((ret[p] - ret["unmonitored"]).median())
                row[f"{p}: d_p5_dd"] = float(dd[p].quantile(0.05) - dd["unmonitored"].quantile(0.05))
                row[f"{p}: win_rate"] = float((ret[p] > ret["unmonitored"]).mean())
            rows.append(row)
    return pd.DataFrame(rows).round(3)


def plot_grid(grid: pd.DataFrame, metric: str = "detector: d_median_ret", ax=None):
    """Heatmap of one metric. Green = the policy helped, red = it hurt (win_rate is centred on 0.5)."""
    import matplotlib.pyplot as plt

    pv = grid.pivot(index="depth", columns="ramp_days", values=metric)
    centre = 0.5 if metric.endswith("win_rate") else 0.0
    span = max(abs(pv.values - centre).max(), 1e-9)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=centre - span, vmax=centre + span, aspect="auto")
    ax.set_xticks(range(len(pv.columns)), [f"{c} d" for c in pv.columns])
    ax.set_yticks(range(len(pv.index)), [f"{d:g}" for d in pv.index])
    ax.set_xlabel("decay ramp")
    ax.set_ylabel("decay depth (1 = edge to 0, 2 = reversed)")
    ax.set_title(metric, fontsize=9)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    return ax
