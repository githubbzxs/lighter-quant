import argparse
import asyncio
from src.config import load_config
from src.data_collector import BinanceOrderBookCollector
from src.features import build_dataset
from src.model import train_model
from src.backtest import run_backtest
from src.live_trading import run_live_trading
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant trading pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-data", help="Collect Binance orderbook data")
    collect.add_argument("--config", required=True, help="Config path for dataset collection")

    build = sub.add_parser("build-dataset", help="Build features and labels")
    build.add_argument("--config", required=True, help="Dataset config path")

    train = sub.add_parser("train-model", help="Train ML model")
    train.add_argument("--config", required=True, help="Training config path")

    backtest = sub.add_parser("backtest", help="Run backtest and grid search")
    backtest.add_argument("--config", required=True, help="Backtest config path")

    live = sub.add_parser("live-trade", help="Run live trading loop")
    live.add_argument("--config", required=True, help="Live trading config path")

    args = parser.parse_args()

    if args.command == "collect-data":
        cfg = load_config(args.config)
        logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
        collector = BinanceOrderBookCollector(cfg, logger)
        asyncio.run(collector.run())
    elif args.command == "build-dataset":
        cfg = load_config(args.config)
        logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
        build_dataset(cfg, logger)
    elif args.command == "train-model":
        cfg = load_config(args.config)
        logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
        train_model(cfg, logger)
    elif args.command == "backtest":
        cfg = load_config(args.config)
        logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
        run_backtest(cfg, logger)
    elif args.command == "live-trade":
        cfg = load_config(args.config)
        logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
        asyncio.run(run_live_trading(cfg, logger))


if __name__ == "__main__":
    main()
