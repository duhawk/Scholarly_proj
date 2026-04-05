from fastapi import Header, HTTPException
from app.config import settings


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Verify X-API-Key header. No-op if API_KEY is not configured (dev mode)."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
