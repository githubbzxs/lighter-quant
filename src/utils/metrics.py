import numpy as np


def total_return(equity: np.ndarray) -> float:
    return equity[-1] if len(equity) > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    high_water = np.maximum.accumulate(equity)
    drawdowns = equity - high_water
    return drawdowns.min()

def annualized_return(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity[-1]
    per_period = (1 + total) ** (periods_per_year / len(equity)) - 1
    return per_period


def sharpe_ratio(equity: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    returns = np.diff(equity)
    if returns.std() == 0:
        return 0.0
    daily = returns.mean() / returns.std()
    return daily * np.sqrt(periods_per_year)
