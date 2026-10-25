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