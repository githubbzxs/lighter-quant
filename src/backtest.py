import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model import load_model
from src.config import Config
from src.utils.metrics import total_return, sharpe_ratio, max_drawdown, annualized_return


def _run_strategy(prob, price, cfg: Config):
    cash = 0.0
    position = 0.0
    entry_price = 0.0
    equity_curve = []
    trades = []
    fee = cfg.backtest.fee_rate
    slippage = cfg.backtest.slippage
    hold = cfg.backtest.hold_ticks
    stop_loss = cfg.backtest.stop_loss
    take_profit = cfg.backtest.take_profit
    for i, p in enumerate(price):
        if position == 0:
            if prob[i] > cfg.backtest.p_buy:
                position = 1
                entry_price = p * (1 + slippage)
                cash -= fee
                trades.append({"i": i, "side": "long", "entry": entry_price})
            elif prob[i] < 1 - cfg.backtest.p_sell:
                position = -1
                entry_price = p * (1 - slippage)
                cash -= fee
                trades.append({"i": i, "side": "short", "entry": entry_price})
        else:
            pnl = (p - entry_price) / entry_price * position
            exit_cond = abs(i - trades[-1]["i"]) >= hold or pnl <= stop_loss or pnl >= take_profit
            if exit_cond:
                cash += pnl - fee
                trades[-1]["exit"] = p
                trades[-1]["pnl"] = pnl
                position = 0
        equity_curve.append(cash)
    return np.array(equity_curve), trades


def _load_dataset(cfg: Config):
    bundle = pd.read_pickle(cfg.backtest.dataset_path)
    X = bundle["X"]
    y = bundle["y"]
    return X, y, bundle.get("features", [])


def run_backtest(cfg: Config, logger) -> None:
    X, y, _ = _load_dataset(cfg)
    model = load_model(cfg.backtest.model_path)
    proba = model.predict_proba(X)[:, -1] if hasattr(model, "predict_proba") else model.predict(X)
    price = np.cumprod(1 + np.random.normal(0, 0.0005, size=len(proba)))
    eq, trades = _run_strategy(proba, price, cfg)

    stats = {
        "total_return": total_return(eq),
        "sharpe": sharpe_ratio(eq),
        "max_drawdown": max_drawdown(eq),
        "annualized": annualized_return(eq, periods_per_year=365 * 24 * 60),
    }
    logger.info("Backtest stats: %s", stats)

    plot_dir = Path(cfg.backtest.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(eq)
    plt.title("Equity Curve")
    plt.xlabel("Tick")
    plt.ylabel("P&L")
    plot_path = plot_dir / "equity_curve.png"
    plt.savefig(plot_path)
    logger.info("Equity curve saved to %s", plot_path)
    trades_path = plot_dir / "trades.csv"
    pd.DataFrame(trades).to_csv(trades_path, index=False)
    logger.info("Trades saved to %s", trades_path)

    if cfg.backtest.grid:
        grid_params = list(itertools.product(*cfg.backtest.grid.values()))
        logger.info("Running grid search with %s combinations", len(grid_params))
        results = []
        keys = list(cfg.backtest.grid.keys())
        for combo in grid_params:
            for k, v in zip(keys, combo):
                setattr(cfg.backtest, k, v)
            eq, _ = _run_strategy(proba, price, cfg)
            results.append(dict(zip(keys, combo)) | {"ret": total_return(eq), "sharpe": sharpe_ratio(eq)})
        grid_path = plot_dir / "grid_search.csv"
        pd.DataFrame(results).to_csv(grid_path, index=False)
        logger.info("Grid search results saved to %s", grid_path)
