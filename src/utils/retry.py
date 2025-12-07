import asyncio
import functools
import time
from typing import Callable, TypeVar, Any

T = TypeVar("T")


def retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            d = delay
            for i in range(retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if i == retries - 1:
                        raise
                    time.sleep(d)
                    d *= backoff
        return wrapper
    return decorator


def async_retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> Any:
            d = delay
            for i in range(retries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(d)
                    d *= backoff
        return wrapper
    return decorator
