import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Tuple
import httpx
import websockets
from src.utils.retry import async_retry


class BinanceClient:
    def __init__(self, symbol: str, depth: int, rest_base: str, ws_base: str, stream_interval: str, logger) -> None:
        self.symbol = symbol.upper()
        self.depth = depth
        self.rest_base = rest_base
        self.ws_base = ws_base
        self.stream_interval = stream_interval
        self.logger = logger
        # trust_env=True 允许使用系统代理；http2=False 兼容部分环境；超时 10s
        self.session = httpx.AsyncClient(timeout=10, trust_env=True, http2=False)
    @async_retry(retries=3, delay=1.0, backoff=2.0)
    async def get_orderbook_snapshot(self) -> Dict[str, Any]:
        params = {"symbol": self.symbol, "limit": self.depth}
        url = f"{self.rest_base}/fapi/v1/depth"  # TODO: 具体参数与字段请参考 Binance 官方文档（通过 MCP contxt7 查询）
        try:
            resp = await self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            ob = {
                "lastUpdateId": data["lastUpdateId"],
                "bids": {float(p): float(q) for p, q in data.get("bids", [])},
                "asks": {float(p): float(q) for p, q in data.get("asks", [])},
            }
            return ob
        except httpx.ConnectError as exc:
            self.logger.warning("Binance snapshot connect error, retrying: %s", exc)
            raise
        except Exception as exc:
            self.logger.error("Binance snapshot failed: %s", exc)
            raise
    def _apply_diff(self, ob: Dict[str, Dict[float, float]], updates: Dict[str, Any]) -> None:
        for side in ["b", "a"]:
            key = "bids" if side == "b" else "asks"
            for price_str, qty_str in updates.get(side, []):
                price = float(price_str)
                qty = float(qty_str)
                book = ob[key]
                if qty == 0:
                    book.pop(price, None)
                else:
                    book[price] = qty
    async def depth_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        stream = f"{self.symbol.lower()}@depth@{self.stream_interval}"
        url = f"{self.ws_base}/ws/{stream}"  # TODO: WebSocket endpoint与stream格式参考 Binance 官方文档（通过 MCP contxt7 查询）
        while True:
            try:
                snapshot = await self.get_orderbook_snapshot()
                last_update_id = snapshot["lastUpdateId"]
                local_ob = {"bids": snapshot["bids"].copy(), "asks": snapshot["asks"].copy()}
                async with websockets.connect(url, ping_interval=180) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        first_id = data.get("U")
                        final_id = data.get("u")
                        if final_id is None or first_id is None:
                            continue
                        if final_id <= last_update_id:
                            continue
                        if first_id <= last_update_id + 1 <= final_id:
                            self._apply_diff(local_ob, data)
                            last_update_id = final_id
                            yield {
                                "event_time": data.get("E"),
                                "bids": local_ob["bids"],
                                "asks": local_ob["asks"],
                            }
            except Exception as exc:
                self.logger.error("Depth stream error: %s", exc, exc_info=True)
                await asyncio.sleep(1)
