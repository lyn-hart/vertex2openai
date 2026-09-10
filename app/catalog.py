"""
Model catalog: fetches the Express model list from a remote JSON config
(MODELS_CONFIG_URL) with a built-in fallback list.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

import httpx

import config as app_config

logger = logging.getLogger(__name__)

# Fallback list used when the remote config is unreachable or malformed.
BUILTIN_EXPRESS_MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-image",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]

_model_cache: Optional[List[str]] = None
_cache_lock = asyncio.Lock()


async def fetch_and_parse_models_config() -> Optional[List[str]]:
    """
    Fetches the model configuration JSON from MODELS_CONFIG_URL and returns
    the vertex_express_models list. Any other keys (e.g. a legacy
    vertex_models list) are ignored. Returns None on fetch/parse failure.
    """
    if not app_config.MODELS_CONFIG_URL:
        logger.error("MODELS_CONFIG_URL is not set in the environment/config.")
        return None

    logger.info("Fetching model configuration from: %s", app_config.MODELS_CONFIG_URL)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(app_config.MODELS_CONFIG_URL)
            response.raise_for_status()
            data = response.json()

            if (
                isinstance(data, dict)
                and isinstance(data.get("vertex_express_models"), list)
                and data["vertex_express_models"]
            ):
                logger.info("Successfully fetched and parsed model configuration.")
                return [m for m in data["vertex_express_models"] if isinstance(m, str)]

            logger.error("Fetched model configuration has an invalid structure: %s", data)
            return None
    except httpx.RequestError as e:
        logger.error("HTTP request failed while fetching model configuration: %s", e)
        return None
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from model configuration: %s", e)
        return None
    except Exception as e:
        logger.error("An unexpected error occurred while fetching/parsing model configuration: %s", e)
        return None


async def get_express_models() -> List[str]:
    """
    Returns the cached Express model list. Fetches on first use; falls back
    to the built-in list when fetching fails.
    """
    global _model_cache
    async with _cache_lock:
        if _model_cache is None:
            logger.info("Model cache is empty. Fetching configuration...")
            _model_cache = await fetch_and_parse_models_config()
            if _model_cache is None:
                logger.warning(
                    "Using the built-in default model list due to fetch/parse failure."
                )
                _model_cache = list(BUILTIN_EXPRESS_MODELS)
    return _model_cache


async def refresh_models_config_cache() -> bool:
    """
    Forces a refresh of the model configuration cache.
    Returns True if a remote list was fetched, False otherwise (cache then
    falls back to the built-in list).
    """
    global _model_cache
    logger.info("Attempting to refresh model configuration cache...")
    async with _cache_lock:
        new_models = await fetch_and_parse_models_config()
        if new_models is not None:
            _model_cache = new_models
            logger.info("Model configuration cache refreshed successfully.")
            return True
        else:
            logger.error("Failed to refresh model configuration cache.")
            if _model_cache is None:
                _model_cache = list(BUILTIN_EXPRESS_MODELS)
            return False
