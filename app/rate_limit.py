"""
Simple in-memory sliding-window rate limiter.

Note: works correctly with a single Uvicorn worker. For multi-worker deployments,
replace with a Redis-backed solution (e.g. fastapi-limiter).
"""
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import HTTPException, Request

from app.config import settings

_WINDOW_SECONDS = 60
_request_times: dict[str, list[datetime]] = defaultdict(list)
_lock = asyncio.Lock()


async def rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 if client exceeds RATE_LIMIT_PER_MINUTE."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    client_ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=_WINDOW_SECONDS)

    async with _lock:
        times = _request_times[client_ip]
        times[:] = [t for t in times if t > cutoff]
        if len(times) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {limit} requests per minute",
                headers={"Retry-After": "60"},
            )
        times.append(now)
