import asyncio
import numpy as np
from src.model import load_model
from src.config import Config
from src.exchange.binance_client import BinanceClient
from src.exchange.lighter_client import LighterClient
from src.utils.retry import async_retry


async def run_live_trading(config: Config, logger) -> None:
    model = load_model(config.live.model_path)
    binance = BinanceClient(
        symbol=config.binance.symbol,
        depth=config.binance.depth_limit,
        rest_base=config.binance.rest_base,
        ws_base=config.binance.ws_base,
        stream_interval=config.binance.stream_interval,
        logger=logger,
    )
    lighter = LighterClient(config.lighter, logger)
    position = 0.0
    entry_price = 0.0
    daily_pnl = 0.0
    async for ob in binance.depth_stream():
        best_bid = max(ob["bids"].keys())
        best_ask = min(ob["asks"].keys())
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        feat = np.array([[spread, spread / mid, ob.get("event_time", 0) % 86400]]).astype(float)
        prob = model.predict_proba(feat)[0, -1] if hasattr(model, "predict_proba") else model.predict(feat)[0]

        if daily_pnl <= config.live.max_daily_loss:
            logger.warning("Daily loss limit reached, only flattening positions")
            if position != 0:
                await lighter.close_position(config.binance.symbol, position)
                position = 0
            await asyncio.sleep(0.1)
            continue

        if position == 0:
            if prob > config.live.p_buy:
                res = await lighter.place_order(config.binance.symbol, "BUY", config.live.max_position, "MARKET")
                logger.info("Open long: %s", res)
                position = config.live.max_position
                entry_price = mid
            elif prob < 1 - config.live.p_sell:
                res = await lighter.place_order(config.binance.symbol, "SELL", config.live.max_position, "MARKET")
                logger.info("Open short: %s", res)
                position = -config.live.max_position
                entry_price = mid
        else:
            pnl = (mid - entry_price) / entry_price * position
            if pnl <= config.live.max_single_loss or pnl >= config.live.take_profit:
                side = "SELL" if position > 0 else "BUY"
                res = await lighter.place_order(config.binance.symbol, side, abs(position), "MARKET")
                logger.info("Close position: %s", res)
                daily_pnl += pnl
                position = 0

        await asyncio.sleep(0.05)
