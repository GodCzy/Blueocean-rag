from fastapi import Header, HTTPException, Depends

from src.config.settings import get_settings

async def api_key_auth(x_api_key: str = Header(None, alias="X-API-Key")):
    """Simple API key authentication dependency."""
    settings = get_settings()
    expected_key = getattr(settings, "api_key", None)
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

