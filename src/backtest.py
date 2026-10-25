from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma_signal(close: pd.Series, window: int = 200) -> pd.Series:
    """1.0 if close > SMA(window), else 0.0. NaN during warm-up.
    Uses data up to and including day t only."""
    sma = close.rolling(window).mean()
    sig = (close > sma).astype(float)
    return sig.where(sma.notna())


def run_backtest(
    close: pd.Series,
    signal: pd.Series | None = None,
    size: pd.Series | None = None,
    window: int = 200,
    cost_bps: float = 5.0,
    capital: float = 100_000.0,
) -> pd.DataFrame:
    """Long/flat backtest on one asset.

    signal: target position (0/1) decided at the close of day t.
    size:   optional multiplier (0..1) from a monitoring policy, also decided at close of t.
    Position held on day t is last close's target, hence the shift(1).
    Costs are charged on the day a new position starts to earn.
    """
    asset_ret = close.pct_change()
    if signal is None:
        signal = sma_signal(close, window)
    target = signal if size is None else signal * size
    position = target.shift(1)

    df = pd.DataFrame({"asset_ret": asset_ret, "position": position}).dropna()
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["cost"] = df["turnover"] * cost_bps / 1e4
    df["strat_ret"] = df["position"] * df["asset_ret"] - df["cost"]
    df["equity"] = capital * (1 + df["strat_ret"]).cumprod()
    df["bh_equity"] = capital * (1 + df["asset_ret"]).cumprod()
    return df


def summarize(returns: pd.Series) -> dict:
    """Metrics from a daily return series. Works for any policy later."""
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    total = equity.iloc[-1] - 1
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1,
        "ann_vol": returns.std() * np.sqrt(TRADING_DAYS),
        "sharpe": returns.mean() / returns.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }