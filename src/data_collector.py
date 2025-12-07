import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from src.exchange.binance_client import BinanceClient
from src.config import Config
from src.utils.retry import async_retry


class BinanceOrderBookCollector:
    def __init__(self, config: Config, logger) -> None:
        self.config = config
        self.logger = logger
        self.client = BinanceClient(
            symbol=config.binance.symbol,
            depth=config.binance.depth_limit,
            rest_base=config.binance.rest_base,
            ws_base=config.binance.ws_base,
            stream_interval=config.binance.stream_interval,
            logger=logger,
        )
    async def _write_csv(self, rows: List[Dict[str, Any]], path: Path) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        header = list(rows[0].keys())
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)

    async def run(self) -> None:
        filename = f"{self.config.binance.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = Path(self.config.paths.data_dir) / filename
        interval = self.config.dataset.sample_interval_ms / 1000.0
        buffer: List[Dict[str, Any]] = []
        self.logger.info("Starting collector for %s writing to %s", self.config.binance.symbol, out_path)

        async for ob in self.client.depth_stream():
            now = datetime.now(timezone.utc).timestamp()
            row = self._format_row(ob, now)
            buffer.append(row)
            if len(buffer) >= 50:
                await self._write_csv(buffer, out_path)
                buffer.clear()
            await asyncio.sleep(interval)
    def _format_row(self, ob: Dict[str, Any], ts: float) -> Dict[str, Any]:
        bids = sorted(ob["bids"].items(), key=lambda x: -x[0])[: self.config.dataset.top_levels]
        asks = sorted(ob["asks"].items(), key=lambda x: x[0])[: self.config.dataset.top_levels]
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        row: Dict[str, Any] = {
            "exchange_time": ob.get("event_time", 0),
            "local_time": ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
        }
        for i, (p, v) in enumerate(bids, 1):
            row[f"bid_{i}_price"] = p
            row[f"bid_{i}_vol"] = v
        for i, (p, v) in enumerate(asks, 1):
            row[f"ask_{i}_price"] = p
            row[f"ask_{i}_vol"] = v
        for depth in self.config.dataset.agg_depths:
            row[f"bid_vol_top_{depth}"] = sum(v for _, v in bids[:depth])
            row[f"ask_vol_top_{depth}"] = sum(v for _, v in asks[:depth])
        return row


if __name__ == "__main__":
    import yaml
    from src.utils.logging import setup_logging
    from src.config import load_config

    cfg = load_config("configs/dataset_btcusdt.yml")
    logger = setup_logging(cfg.paths.log_dir, cfg.app.log_level)
    collector = BinanceOrderBookCollector(cfg, logger)
    asyncio.run(collector.run())
