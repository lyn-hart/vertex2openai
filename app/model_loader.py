import httpx
import asyncio
import json
from typing import List, Dict, Optional, Any

# Assuming config.py is in the same directory level for Docker execution
import config as app_config 

import logging
logger = logging.getLogger(__name__)

_model_cache: Optional[Dict[str, List[str]]] = None
_cache_lock = asyncio.Lock()

async def fetch_and_parse_models_config() -> Optional[Dict[str, List[str]]]:
    """
    Fetches the model configuration JSON from the URL specified in app_config.
    Parses it and returns a dictionary with 'vertex_models' and 'vertex_express_models'.
    Returns None if fetching or parsing fails.
    """
    if not app_config.MODELS_CONFIG_URL:
        logger.error("MODELS_CONFIG_URL is not set in the environment/config.")
        return None

    logger.info(f"Fetching model configuration from: {app_config.MODELS_CONFIG_URL}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(app_config.MODELS_CONFIG_URL)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            
            # Basic validation of the fetched data structure
            if isinstance(data, dict) and \
               "vertex_models" in data and isinstance(data["vertex_models"], list) and \
               "vertex_express_models" in data and isinstance(data["vertex_express_models"], list):
                logger.info("Successfully fetched and parsed model configuration.")
                
                return {
                    "vertex_models": data["vertex_models"],
                    "vertex_express_models": data["vertex_express_models"]
                }
            else:
                logger.error(f"Fetched model configuration has an invalid structure: {data}")
                return None
    except httpx.RequestError as e:
        logger.error(f"HTTP request failed while fetching model configuration: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model configuration: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching/parsing model configuration: {e}")
        return None

async def get_models_config() -> Dict[str, List[str]]:
    """
    Returns the cached model configuration.
    If not cached, fetches and caches it.
    Returns a default empty structure if fetching fails.
    """
    global _model_cache
    async with _cache_lock:
        if _model_cache is None:
            logger.info("Model cache is empty. Fetching configuration...")
            _model_cache = await fetch_and_parse_models_config()
            if _model_cache is None: # If fetching failed, use a default empty structure
                logger.warning("Using default empty model configuration due to fetch/parse failure.")
                _model_cache = {"vertex_models": [], "vertex_express_models": []}
    return _model_cache

async def get_vertex_models() -> List[str]:
    config = await get_models_config()
    return config.get("vertex_models", [])

async def get_vertex_express_models() -> List[str]:
    config = await get_models_config()
    return config.get("vertex_express_models", [])

async def refresh_models_config_cache() -> bool:
    """
    Forces a refresh of the model configuration cache.
    Returns True if successful, False otherwise.
    """
    global _model_cache
    logger.info("Attempting to refresh model configuration cache...")
    async with _cache_lock:
        new_config = await fetch_and_parse_models_config()
        if new_config is not None:
            _model_cache = new_config
            logger.info("Model configuration cache refreshed successfully.")
            return True
        else:
            logger.error("Failed to refresh model configuration cache.")
            # Optionally, decide if we want to clear the old cache or keep it
            # _model_cache = {"vertex_models": [], "vertex_express_models": []} # To clear
            return False