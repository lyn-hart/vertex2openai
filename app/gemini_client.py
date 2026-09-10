"""
Shared Gemini client construction and model-feature parsing.

Used by both OpenAI chat completions and Anthropic Messages routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

from project_id_discovery import discover_project_id

import logging
logger = logging.getLogger(__name__)




@dataclass
class ModelFeatures:
    """Parsed model string features after prefix/suffix stripping."""

    original_model: str
    base_model_name: str
    is_grounded_search: bool = False
    is_nothinking_model: bool = False
    is_max_thinking_model: bool = False
    is_2k_image_model: bool = False
    is_4k_image_model: bool = False


def parse_model_features(model: str) -> ModelFeatures:
    """Parse model name suffixes used by this adapter."""
    is_grounded_search = model.endswith("-search")
    is_nothinking_model = model.endswith("-nothinking")
    is_max_thinking_model = model.endswith("-max")
    is_2k_image_model = model.endswith("-2k")
    is_4k_image_model = model.endswith("-4k")

    base_model_name = model

    if is_grounded_search:
        base_model_name = base_model_name[: -len("-search")]
    elif is_nothinking_model:
        base_model_name = base_model_name[: -len("-nothinking")]
    elif is_max_thinking_model:
        base_model_name = base_model_name[: -len("-max")]
    elif is_2k_image_model:
        base_model_name = base_model_name[: -len("-2k")]
    elif is_4k_image_model:
        base_model_name = base_model_name[: -len("-4k")]

    return ModelFeatures(
        original_model=model,
        base_model_name=base_model_name,
        is_grounded_search=is_grounded_search,
        is_nothinking_model=is_nothinking_model,
        is_max_thinking_model=is_max_thinking_model,
        is_2k_image_model=is_2k_image_model,
        is_4k_image_model=is_4k_image_model,
    )


def apply_thinking_config(
    gen_config_dict: dict,
    base_model_name: str,
    *,
    is_nothinking_model: bool = False,
    is_max_thinking_model: bool = False,
) -> dict:
    """Apply thinking_config defaults and -nothinking/-max overrides."""
    if not isinstance(gen_config_dict.get("thinking_config"), dict):
        gen_config_dict["thinking_config"] = {}

    if "gemini-2.5-flash" in base_model_name or "gemini-2.5-pro" in base_model_name:
        gen_config_dict["thinking_config"]["include_thoughts"] = True

    if "gemini-2.5-flash-lite" in base_model_name:
        gen_config_dict["thinking_config"]["include_thoughts"] = False

    gen_config_dict["thinking_config"]["include_thoughts"] = True

    if "gemini-2.5-flash-lite" in base_model_name and is_max_thinking_model:
        gen_config_dict["thinking_config"]["include_thoughts"] = True
    elif "gemini-2.5-flash-lite" in base_model_name or "image" in base_model_name:
        gen_config_dict["thinking_config"]["include_thoughts"] = False
    else:
        gen_config_dict["thinking_config"]["include_thoughts"] = True

    if is_nothinking_model or is_max_thinking_model:
        if is_nothinking_model:
            budget = 128 if ("gemini-2.5-pro" in base_model_name or "gemini-3-pro" in base_model_name) else 0
        else:
            budget = 32768 if ("gemini-2.5-pro" in base_model_name or "gemini-3-pro" in base_model_name) else 24576
        gen_config_dict["thinking_config"]["thinking_budget"] = budget
        if budget == 0:
            gen_config_dict["thinking_config"]["include_thoughts"] = False

    return gen_config_dict


class GeminiClientError(Exception):
    """Raised when Gemini client construction fails."""

    def __init__(self, message: str, status_code: int = 500, error_type: str = "server_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


async def resolve_gemini_client(
    *,
    model: str,
    base_model_name: str,
    express_key_manager: Any,
) -> Optional[Any]:
    """
    Build a google.genai Client for Gemini using a Vertex Express API key.

    Raises GeminiClientError on auth/config failures.
    """
    client_to_use = None
    has_express = express_key_manager.get_total_keys() > 0

    if not has_express:
        raise GeminiClientError(
            f"Model '{model}' requires an Express API key, but none are configured.",
            status_code=401,
            error_type="authentication_error",
        )

    logger.info(f"Attempting Vertex Express Mode for model request: {model} (base: {base_model_name})")
    total_keys = express_key_manager.get_total_keys()
    for attempt in range(total_keys):
        key_tuple = express_key_manager.get_express_api_key()
        if key_tuple:
            original_idx, key_val = key_tuple
            try:
                if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                    project_id = await discover_project_id(key_val)
                    base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                    client_to_use = genai.Client(
                        vertexai=True,
                        api_key=key_val,
                        http_options=types.HttpOptions(base_url=base_url),
                    )
                    client_to_use._api_client._http_options.api_version = None
                    logger.info(f"Attempt {attempt + 1}/{total_keys} - Using Vertex Express Mode with custom base URL "
                        f"for model {model} (base: {base_model_name}) with API key (original index: {original_idx}).")
                else:
                    client_to_use = genai.Client(vertexai=True, api_key=key_val)
                    logger.info(f"Attempt {attempt + 1}/{total_keys} - Using Vertex Express Mode SDK "
                        f"for model {model} (base: {base_model_name}) with API key (original index: {original_idx}).")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{total_keys} - Vertex Express Mode client init failed "
                    f"for API key (original index: {original_idx}) for model {model}: {e}. Trying next key.")
                client_to_use = None
        else:
            logger.warning(f"Attempt {attempt + 1}/{total_keys} - get_express_api_key() returned None unexpectedly.")
            client_to_use = None

    if client_to_use is None:
        raise GeminiClientError(
            f"All {total_keys} configured Express API keys failed to initialize or were unavailable for model '{model}'.",
            status_code=500,
            error_type="server_error",
        )
    return client_to_use
