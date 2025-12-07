import time
import hmac
import hashlib
from typing import Any, Dict, Optional
import httpx
from src.config import LighterConfig


class LighterClient:
    def __init__(self, config: LighterConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.session = httpx.AsyncClient(base_url=config.base_url, timeout=10)
    def _sign(self, payload: str) -> str:
        # TODO: 签名规则请参考 Lighter 官方文档（通过 MCP contxt7 查询）
        if not self.config.api_secret:
            return ""
        return hmac.new(self.config.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _headers(self, payload: str = "") -> Dict[str, str]:
        signature = self._sign(payload)
        return {
            "X-API-KEY": self.config.api_key or "",
            "X-SIGNATURE": signature,
            "Content-Type": "application/json",
        }
    async def get_balance(self) -> Dict[str, Any]:
        url = "/api/v1/balance"  # TODO: 参考：Lighter 官方文档 balance 接口（通过 MCP contxt7 查询）
        try:
            resp = await self.session.get(url, headers=self._headers(""))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error("get_balance failed: %s", exc)
            return {"error": str(exc)}

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        url = f"/api/v1/position?symbol={symbol}"  # TODO: 参考：Lighter 官方文档 position 接口（通过 MCP contxt7 查询）
        try:
            resp = await self.session.get(url, headers=self._headers(""))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error("get_position failed: %s", exc)
            return {"error": str(exc)}
    async def place_order(
        self, symbol: str, side: str, size: float, order_type: str, price: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        url = "/api/v1/order"  # TODO: 参考：Lighter 官方文档 下单接口（通过 MCP contxt7 查询）
        payload = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "type": order_type,
            "price": price,
            "timestamp": int(time.time() * 1000),
        }
        try:
            resp = await self.session.post(url, json=payload, headers=self._headers(str(payload)))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error("place_order failed: %s", exc)
            return {"error": str(exc)}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        url = f"/api/v1/order/{order_id}"  # TODO: 参考：Lighter 官方文档 取消订单接口（通过 MCP contxt7 查询）
        try:
            resp = await self.session.delete(url, headers=self._headers(order_id))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self.logger.error("cancel_order failed: %s", exc)
            return {"error": str(exc)}

    async def close_position(self, symbol: str, size: float) -> Dict[str, Any]:
        side = "SELL" if size > 0 else "BUY"
        return await self.place_order(symbol, side, abs(size), "MARKET")
