# Oxhack-F

Primary asset: SPY, with TLT and GLD cached as spares for a second real-data example.
Strategy: long when price is above its 200-day moving average, flat otherwise. It’s simple, trades rarely and is well known. Check its baseline Sharpe in hour 1. The decay you inject needs some positive edge to remove, but you’re not claiming this strategy has alpha.

Costs: 5 bps per unit of position change (an assumption for liquid ETFs, and you should say so).
Starting capital: $100,000.


Presentation points : 

5bps per unit of position change is an assumption being made for liquid ETFs

finance points: 
Adjusted close: price corrected for splits and dividends, so returns reflect what an investor earned.
Simple return: Close.pct_change(). For accounting, new value = old value × (1 + position × return).
Trading days: about 252 per year, so annualized volatility = daily std × √252.
Sharpe ratio: annualized mean return divided by annualized volatility (ignoring the risk-free rate). Around 0.3 to 0.5 is normal for buy-and-hold equities.
Drawdown: equity / equity.cummax() - 1. Max drawdown is the worst value of that.
Lookahead: a signal computed from today’s close can only change tomorrow’s position, so in code use position.shift(1). Getting this wrong is the most common way backtests lie.

in the backtest file design choices to understand: 

run_backtest takes a price path, not returns. In hour 1.5-3 the scenario generator will produce synthetic price paths, and the signal has to be recomputed from them.

size is an input series. Your detector will watch the paper strategy (size = 1), whose returns don’t depend on position sizing. So the detector produces size from those returns, and there’s no feedback loop and no need for a slow day-by-day simulation.

for the test_backtest.py file
Run python -m pytest -q from the repo root. You want 3 passed. To prove the lookahead test works, temporarily change position = target.shift(1) to position = target, rerun and confirm test_no_lookahead fails, then change it back. Spend two minutes on this, because it’s the bug that makes a backtest look great and mean nothing.

The detector:
The detector watches the paper strategy’s returns on exposed days only, because flat days say nothing about the edge. It runs a one-sided CUSUM looking for the mean to fall from the research-period edge toward zero.
The reference edge and volatility come from the research backtest res. This is what a practitioner would have, but it’s an assumption you should state: in real life the in-sample edge is overstated, so real detectors would false-alarm more often than your simulation shows.
Position size is 100%, 50% (at h/2) or 0% (at h). The statistic keeps updating on paper returns while paused, so size recovers if the edge comes back.
The stop-loss baseline is a 15% drawdown stop on the paper strategy’s equity, with a 126-day cooldown and a peak reset on re-entry. 15% is about 70% of your backtest’s -21.5% max drawdown, which is a defensible, non-arbitrary choice.

NOTE: on the "h" test(hours 3 to 5) 
