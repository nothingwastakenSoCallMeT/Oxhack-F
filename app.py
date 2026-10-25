
"""Streamlit demo for the Oxhack-F strategy deterioration experiment.

Run from the repository root with:
    python -m streamlit run app.py

All synthetic scenarios are built using the existing src/ functions. The
benchmark asset return path is *not* altered by the artificial deterioration;
the deterioration is injected into the paper strategy's returns only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import run_backtest
from src.evaluate import reference_stats
from src.grid import monitoring_grid
from src.policies import (
    applied_size,
    cusum_stat,
    policy_returns,
    size_from_cusum,
    stoploss_size,
)
from src.scenarios import Scenario, make_scenario


# --------------------------------
# Parameters fixed before testing
# --------------------------------
CAPITAL = 100_000
H = 40.0
STOP_DRAWDOWN = 0.15
STOP_COOLDOWN = 126
COST_BPS = 5.0
N_DAYS = 2520              # Approximately 10 trading years
BREAK_INDEX = 1260         # Artificial deterioration begins halfway through
ROOT = Path(__file__).resolve().parent
ASSETS = {
    "SPY": "US equities (S&P 500 ETF)",
    "TLT": "Long-term US Treasury bonds (20+ year ETF)",
    "GLD": "Gold (gold-backed ETF)",
}


@dataclass
class Replay:
    """Everything needed to draw one paired simulated experiment."""

    scenario: Scenario
    cusum: np.ndarray
    held_detector_size: np.ndarray
    held_stop_size: np.ndarray
    equity: pd.DataFrame
    mu0: float
    sigma: float


@st.cache_data(show_spinner=False)
def load_backtest(ticker: str) -> pd.DataFrame:
    """Run the same SMA strategy on one cached ETF. No download is needed."""
    if ticker not in ASSETS:
        raise ValueError(f"Unknown ticker: {ticker!r}")
    csv_file = ROOT / "data" / "raw" / f"{ticker}.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(
            f"Cannot find data/raw/{ticker}.csv. Make sure your repository "
            "contains the cached CSV files in data/raw/."
        )
    prices = pd.read_csv(csv_file, index_col="date", parse_dates=True)
    return run_backtest(prices["Close"], cost_bps=COST_BPS, capital=CAPITAL)


def build_replay(
    res: pd.DataFrame,
    seed: int,
    break_idx: int | None,
    depth: float,
    ramp_days: int,
) -> Replay:
    """Connect the existing scenario, detector, and policy functions.

    An important timing rule: policies make their decisions after each day's
    return. applied_size() shifts those decisions by one day before we plot
    actual exposure or calculate policy returns.
    """
    scenario = make_scenario(
        res,
        n_days=N_DAYS,
        break_idx=break_idx,
        depth=depth,
        ramp_days=ramp_days,
        seed=seed,
    )
    mu0, sigma = reference_stats(res)

    cusum = cusum_stat(scenario.strat_ret, scenario.position, mu0, sigma)
    detector_decisions = size_from_cusum(cusum, h=H)
    stop_decisions = stoploss_size(
        scenario.strat_ret,
        dd_limit=STOP_DRAWDOWN,
        cooldown=STOP_COOLDOWN,
    )

    policies = {
        "Asset benchmark": scenario.asset_ret,
        "Unmonitored": policy_returns(
            scenario.strat_ret,
            scenario.position,
            np.ones(N_DAYS),
            cost_bps=COST_BPS,
        ),
        "Stop-loss": policy_returns(
            scenario.strat_ret,
            scenario.position,
            stop_decisions,
            cost_bps=COST_BPS,
        ),
        "CUSUM detector": policy_returns(
            scenario.strat_ret,
            scenario.position,
            detector_decisions,
            cost_bps=COST_BPS,
        ),
    }

    # Trading day 0 is the initial $100,000, before any return is applied.
    equity = pd.DataFrame(
        {
            name: CAPITAL * np.r_[1.0, np.cumprod(1.0 + np.asarray(r))]
            for name, r in policies.items()
        },
        index=pd.Index(np.arange(N_DAYS + 1), name="Trading day"),
    )

    return Replay(
        scenario=scenario,
        cusum=cusum,
        held_detector_size=applied_size(detector_decisions),
        held_stop_size=applied_size(stop_decisions),
        equity=equity,
        mu0=mu0,
        sigma=sigma,
    )


def metrics_until(equity: pd.DataFrame, end_day: int) -> pd.DataFrame:
    """Total return and worst peak-to-trough loss through the displayed day."""
    rows = []
    for name in equity.columns:
        path = equity[name].iloc[: end_day + 1]
        total = path.iloc[-1] / path.iloc[0] - 1
        drawdown = (path / path.cummax() - 1).min()
        rows.append(
            {
                "Policy": name,
                "Final account ($)": f"${path.iloc[-1]:,.0f}",
                "Total return": f"{total:+.1%}",
                "Max drawdown": f"{drawdown:.1%}",
            }
        )
    return pd.DataFrame(rows).set_index("Policy")


def plot_equity(replay: Replay, end_day: int) -> go.Figure:
    """Interactive account-value replay; values are dollars, NOT ETF share prices."""
    fig = go.Figure()
    x = replay.equity.index[: end_day + 1]
    for name in replay.equity.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=replay.equity[name].iloc[: end_day + 1],
                mode="lines",
                name=name,
                hovertemplate=f"{name}: $%{{y:,.2f}}<extra></extra>",
            )
        )

    if replay.scenario.break_idx is not None:
        first_decay_day = replay.scenario.break_idx + 1
        if first_decay_day <= end_day:
            fig.add_vline(
                x=first_decay_day,
                line_dash="dash",
                line_color="gray",
                annotation_text="Decay begins",
                annotation_position="top left",
            )

    fig.update_layout(
        title="Account value by policy",
        xaxis_title="Simulated trading day",
        yaxis_title="Account value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",        # One cursor reports all four policy values
        dragmode="zoom",              # Drag a box to zoom in
        height=520,
        margin=dict(l=20, r=20, t=65, b=20),
        legend=dict(orientation="h", y=1.13, x=0),
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        rangeslider=dict(visible=True, thickness=0.08),
    )
    return fig


def plot_detector(replay: Replay, end_day: int) -> go.Figure:
    """Interactive CUSUM & exposure, aligned by simulated trading day."""
    day = np.arange(end_day + 1)
    # CUSUM becomes known only after each simulated trading day closes.
    s = np.r_[0.0, replay.cusum[:end_day]]
    # Exposure shown was actually held on that day (one-day execution delay).
    held = np.r_[1.0, replay.held_detector_size[:end_day]]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.67, 0.33],
        vertical_spacing=0.11,
        subplot_titles=("Evidence of deterioration (CUSUM)", "Strategy size held"),
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s,
            mode="lines",
            name="CUSUM statistic",
            hovertemplate="CUSUM: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for threshold, label, colour in [
        (H / 2, "Half-size trigger (h/2)", "darkorange"),
        (H, "Full pause trigger (h)", "firebrick"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[0, end_day],
                y=[threshold, threshold],
                mode="lines",
                line=dict(color=colour, dash="dash"),
                name=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=held * 100,
            mode="lines",
            line_shape="hv",
            name="Actual strategy size",
            hovertemplate="Held size: %{y:.0f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if replay.scenario.break_idx is not None:
        first_decay_day = replay.scenario.break_idx + 1
        if first_decay_day <= end_day:
            for row in (1, 2):
                fig.add_vline(
                    x=first_decay_day,
                    line_color="gray",
                    line_dash="dot",
                    row=row,
                    col=1,
                )

    fig.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", y=1.10, x=0),
    )
    fig.update_yaxes(title_text="CUSUM", row=1, col=1)
    fig.update_yaxes(title_text="Exposure (%)", range=[-5, 110],
                     tickvals=[0, 50, 100], row=2, col=1)
    fig.update_xaxes(title_text="Simulated trading day", row=2, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    return fig


def plot_grid_interactive(grid: pd.DataFrame, metric: str) -> go.Figure:
    """Interactive version of the existing fixed-threshold experiment heatmap."""
    pv = grid.pivot(index="depth", columns="ramp_days", values=metric)
    values = pv.to_numpy(dtype=float)
    center = 0.5 if metric.endswith("win_rate") else 0.0
    span = max(float(np.nanmax(np.abs(values - center))), 1e-9)
    fig = go.Figure(
        go.Heatmap(
            x=[f"{int(r)} days" for r in pv.columns],
            y=[f"{d:g}" for d in pv.index],
            z=values,
            zmin=center - span,
            zmax=center + span,
            colorscale="RdYlGn",
            text=values,
            texttemplate="%{text:.3f}",
            hovertemplate=(
                "Decay depth: %{y}<br>Ramp: %{x}<br>Value: %{z:.3f}"
                "<extra></extra>"
            ),
            colorbar=dict(thickness=10),
        )
    )
    fig.update_layout(
        title=metric,
        xaxis_title="Decay ramp",
        yaxis_title="Decay depth",
        height=430,
        margin=dict(l=20, r=10, t=60, b=40),
    )
    return fig


@st.cache_data(show_spinner=False)
def calculate_grid(res: pd.DataFrame) -> pd.DataFrame:
    """Keep the same fixed settings and 200 test seeds used in the notebook."""
    return monitoring_grid(
        res,
        h=H,
        n_seeds=200,
        seed0=0,
        break_idx=BREAK_INDEX,
    )


def show_app() -> None:
    st.set_page_config(page_title="Oxhack-F | Strategy Monitor", layout="wide")
    st.title("Oxhack-F: When should a trading strategy be paused?")
    st.write(
        "A controlled experiment comparing an unmonitored strategy, a drawdown "
        "stop-loss, and a CUSUM-based monitoring rule."
    )
    st.sidebar.header("Market data")
    ticker = st.sidebar.selectbox(
        "Underlying ETF",
        options=list(ASSETS),
        index=0,
        format_func=lambda symbol: f"{symbol} — {ASSETS[symbol]}",
    )

    try:
        res = load_backtest(ticker)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Could not load local {ticker} data: {exc}")
        st.stop()

    mu0, sigma = reference_stats(res)
    st.sidebar.caption(
        f"Historical strategy reference: {mu0 * 252:.2%} annualized "
        f"exposed-day mean; {sigma:.4f} daily volatility."
    )
    st.warning(
        f"**Simulation, not a live trading result.** Data comes from cached "
        f"**{ticker}** history ({ASSETS[ticker]}). Scenarios resample historical "
        "returns and inject artificial changes into the **strategy** returns. "
        "The asset line is a benchmark, not a price path containing the "
        "injected strategy deterioration."
    )
    if ticker != "SPY":
        st.info(
            "**Cross-asset limitation:** h = 40 was selected using SPY "
            "calibration data and has not been independently calibrated for "
            f"{ticker}. Comparisons across assets are exploratory, not "
            "controlled evidence that one detector works equally well on all ETFs."
        )
    if mu0 <= 0.0001:
        st.warning(
            "**Weak historical edge:** this ETF's strategy reference return is "
            "very small relative to its day-to-day noise. Even a 'full reversal' "
            "injects only a small change, so meaningful detection may be difficult."
        )

    st.sidebar.header("Scenario settings")
    choices = {
        "No decay": (None, 0.0),
        "Half the edge disappears": (BREAK_INDEX, 0.5),
        "Edge falls to zero": (BREAK_INDEX, 1.0),
        "Partial reversal": (BREAK_INDEX, 1.5),
        "Full reversal": (BREAK_INDEX, 2.0),
    }
    selected = st.sidebar.selectbox(
        "What happens to the strategy?",
        list(choices),
        index=4,
    )
    break_idx, depth = choices[selected]
    ramp_label = st.sidebar.selectbox(
        "How quickly?",
        ["Sudden (1 trading day)", "Gradual (250 trading days)"],
    )
    ramp_days = 1 if ramp_label.startswith("Sudden") else 250
    seed = st.sidebar.slider("Random simulation seed", min_value=0, max_value=199, value=7)
    end_day = st.sidebar.slider(
        "Replay through trading day", min_value=252, max_value=N_DAYS,
        value=N_DAYS, step=1,
    )
    st.sidebar.caption(
        f"Fixed detector threshold h={H:g}; stop-loss {STOP_DRAWDOWN:.0%} "
        f"with {STOP_COOLDOWN} trading-day cooldown; assumed additional turnover cost {COST_BPS:g} bps."
    )

    replay = build_replay(res, seed, break_idx, depth, ramp_days)
    tab1, tab2, tab3 = st.tabs(
        ["1 · Strategy replay", "2 · Detector decisions", "3 · When monitoring pays"]
    )

    with tab1:
        st.subheader(f"What happens to a $100,000 account using {ticker} data?")
        st.caption(
            "Hover over a trading day to inspect all four account values. "
            "Drag across the chart to zoom, use the range slider beneath it "
            "to focus on a period, and double-click to reset. "
            "These are account values, not ETF share prices."
        )
        fig = plot_equity(replay, end_day)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )
        st.dataframe(metrics_until(replay.equity, end_day), use_container_width=True)

        if break_idx is None:
            st.info("No deterioration has been injected. Any detector cuts here are false alarms.")
        else:
            st.write(
                f"**Ground truth:** deterioration begins on simulated trading day "
                f"{break_idx + 1}. Decay depth = {depth:g}; ramp = {ramp_days} day(s)."
            )
        st.caption(
            "Returns and costs use the repository's existing policy_returns() "
            "approximation. The synthetic strategy is not recalculated from the "
            "resampled asset prices."
        )

    with tab2:
        st.subheader("Why does the detector reduce exposure?")
        st.write(
            "CUSUM accumulates evidence that the strategy is earning less than "
            "its reference return. At h/2 it cuts to 50%; at h it pauses. "
            "The decision made after a day closes is applied **the next day**."
        )
        fig = plot_detector(replay, end_day)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
        )

        held = replay.held_detector_size[:end_day]
        c1, c2, c3 = st.columns(3)
        c1.metric("Days at full size", f"{np.mean(held == 1.0):.1%}")
        c2.metric("Days at half size", f"{np.mean(held == 0.5):.1%}")
        c3.metric("Days fully paused", f"{np.mean(held == 0.0):.1%}")
        st.caption(
            "These percentages describe the strategy-size multiplier. The "
            "underlying moving-average strategy can also be flat, so the "
            "fraction of actual invested days can be lower."
        )

    with tab3:
        st.subheader(f"At what decay severity does monitoring help on {ticker}?")
        st.write(
            "This repeats the notebook's 200-seed experiment: five decay depths "
            "by two deterioration speeds. The detector threshold and stop-loss "
            "settings stay fixed; they are **not** re-optimised here."
        )
        if st.button(f"Calculate the 200-seed comparison for {ticker}", type="primary"):
            st.session_state["show_grid_for"] = ticker

        if st.session_state.get("show_grid_for") == ticker:
            with st.spinner("Calculating the scenario grid (may take a while on a laptop)..."):
                grid = calculate_grid(res)
            policy = st.radio(
                "Compare with unmonitored strategy",
                ["detector", "stop-loss"],
                horizontal=True,
            )
            metric_names = [
                f"{policy}: d_median_ret",
                f"{policy}: d_p5_dd",
                f"{policy}: win_rate",
            ]
            col1, col2, col3 = st.columns(3)
            for container, metric in zip((col1, col2, col3), metric_names):
                with container:
                    fig = plot_grid_interactive(grid, metric)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"scrollZoom": False, "displaylogo": False},
                    )

            st.dataframe(grid, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scenario-grid CSV",
                grid.to_csv(index=False).encode("utf-8"),
                file_name=f"scenario_grid_{ticker}.csv",
                mime="text/csv",
            )
            st.caption(
                "d_median_ret: median of paired total-return differences. "
                "d_p5_dd: difference in worst-5% maximum drawdown (positive means "
                "less severe drawdowns). win_rate: fraction of simulations in "
                "which the monitored policy finishes above unmonitored."
            )
        else:
            st.info(
                "Click the button to compute the full grid. It is intentionally "
                "not recomputed every time you adjust the replay sliders."
            )

    st.divider()
    st.caption(
        "Research/demo prototype only. Synthetic block-bootstrap scenarios, "
        "stylised injected strategy edge, and approximate transaction costs. "
        "No demonstrated out-of-sample alpha or live trading performance. "
        "Fixed h=40 was calibrated on SPY, not the other assets."
    )


if __name__ == "__main__":
    show_app()
