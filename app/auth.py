from fastapi import HTTPException, Header
from typing import Optional
from config import API_KEY


# Function to validate API key (moved from config.py)
def validate_api_key(api_key_to_validate: str) -> bool:
    """
    Validate the provided API key against the configured key.
    """
    if not API_KEY: # API_KEY is imported from config
        # If no API key is configured, authentication is disabled (or treat as invalid)
        return False
    return api_key_to_validate == API_KEY


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract raw token from Authorization header if it is a Bearer token."""
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer "):].strip() or None
    return None


# Dependency for API key validation
async def get_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    # Prefer Authorization: Bearer when present (OpenAI clients + ANTHROPIC_AUTH_TOKEN).
    # Fall back to x-api-key (Anthropic / Claude Code ANTHROPIC_API_KEY).
    api_key = _extract_bearer_token(authorization)
    if api_key is None and x_api_key:
        api_key = x_api_key.strip() or None

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing API key. Provide 'Authorization: Bearer YOUR_API_KEY' "
                "or 'x-api-key: YOUR_API_KEY'."
            ),
        )

    # If Authorization was present but not Bearer, and no usable x-api-key, reject format
    if (
        authorization is not None
        and not authorization.startswith("Bearer ")
        and not x_api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. Use 'Authorization: Bearer YOUR_API_KEY' or 'x-api-key: YOUR_API_KEY'.",
        )

    # Validate the API key
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key
