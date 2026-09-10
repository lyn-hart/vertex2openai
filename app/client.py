"""
Vertex Express client construction and upstream call helpers.

Merges the former gemini_client.py and project_id_discovery.py. All upstream
calls go through a google-genai client built from a Vertex Express API key;
when a call fails with 404/400 the client is rebuilt once against the
discovered numeric project id (needed for some models on the global endpoint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp
from google import genai
from google.genai import types

import config as app_config

logger = logging.getLogger(__name__)


# ── model feature parsing ───────────────────────────────────────────────────


@dataclass
class ModelFeatures:
    """Parsed model string features after suffix stripping."""

    original_model: str
    base_model_name: str
    is_grounded_search: bool = False
    is_2k_image_model: bool = False
    is_4k_image_model: bool = False


def parse_model_features(model: str) -> ModelFeatures:
    """Parse model name suffixes used by this adapter."""
    is_grounded_search = model.endswith("-search")
    is_2k_image_model = model.endswith("-2k")
    is_4k_image_model = model.endswith("-4k")

    base_model_name = model

    if is_grounded_search:
        base_model_name = base_model_name[: -len("-search")]
    elif is_2k_image_model:
        base_model_name = base_model_name[: -len("-2k")]
    elif is_4k_image_model:
        base_model_name = base_model_name[: -len("-4k")]

    return ModelFeatures(
        original_model=model,
        base_model_name=base_model_name,
        is_grounded_search=is_grounded_search,
        is_2k_image_model=is_2k_image_model,
        is_4k_image_model=is_4k_image_model,
    )


def apply_thinking_config(
    gen_config_dict: dict,
    base_model_name: str,
) -> dict:
    """Apply thinking_config defaults. Thinking budget/level come from
    request params only — no model-suffix overrides.

    A thinking_budget of 0 already set by request params (thinking off)
    is respected: include_thoughts stays False.
    """
    if not isinstance(gen_config_dict.get("thinking_config"), dict):
        gen_config_dict["thinking_config"] = {}

    tc = gen_config_dict["thinking_config"]

    # Thinking explicitly disabled via request params (budget == 0):
    # include_thoughts must stay False (Vertex rejects the combination).
    thinking_off = tc.get("thinking_budget") == 0
    if not thinking_off:
        if "gemini-2.5-flash" in base_model_name or "gemini-2.5-pro" in base_model_name:
            tc["include_thoughts"] = True
        if "gemini-2.5-flash-lite" in base_model_name:
            tc["include_thoughts"] = False
        if "gemini-2.5-flash-lite" in base_model_name or "image" in base_model_name:
            tc["include_thoughts"] = False
        else:
            tc["include_thoughts"] = True

    return gen_config_dict


class GeminiClientError(Exception):
    """Raised when Gemini client construction fails."""

    def __init__(self, message: str, status_code: int = 500, error_type: str = "server_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


# ── client construction ─────────────────────────────────────────────────────


def _get_http_options() -> Optional[types.HttpOptions]:
    """Proxy / TLS options applied to every google-genai client."""
    client_args: Dict[str, Any] = {}
    async_client_args: Dict[str, Any] = {}
    if app_config.PROXY_URL:
        client_args["proxy"] = app_config.PROXY_URL
        async_client_args["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        client_args["verify"] = app_config.SSL_CERT_FILE
        async_client_args["verify"] = app_config.SSL_CERT_FILE
    if not client_args:
        return None
    return types.HttpOptions(
        client_args=client_args,
        async_client_args=async_client_args,
    )


def _build_plain_client(api_key: str) -> Any:
    """Standard Vertex Express client (SDK-default base URL)."""
    http_options = _get_http_options()
    kwargs: Dict[str, Any] = {"vertexai": True, "api_key": api_key}
    if http_options:
        kwargs["http_options"] = http_options
    return genai.Client(**kwargs)


def _build_project_client(api_key: str, project_id: str) -> Any:
    """Express client pinned to the discovered project's global endpoint."""
    base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
    http_options = _get_http_options()
    if http_options is not None:
        http_options.base_url = base_url
    else:
        http_options = types.HttpOptions(base_url=base_url)
    client = genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
    client._api_client._http_options.api_version = None
    return client


async def resolve_gemini_client(
    *,
    model: str,
    base_model_name: str,
    express_key_manager: Any,
) -> Tuple[Any, str]:
    """
    Build a google.genai Client for Gemini using a Vertex Express API key.

    Returns (client, api_key). Raises GeminiClientError on auth/config failures.
    """
    total_keys = express_key_manager.get_total_keys()
    if total_keys == 0:
        raise GeminiClientError(
            f"Model '{model}' requires an Express API key, but none are configured.",
            status_code=401,
            error_type="authentication_error",
        )

    logger.info("Attempting Vertex Express Mode for model request: %s (base: %s)", model, base_model_name)
    for attempt in range(total_keys):
        key_tuple = express_key_manager.get_express_api_key()
        if not key_tuple:
            logger.warning("Attempt %d/%d - get_express_api_key() returned None unexpectedly.", attempt + 1, total_keys)
            continue
        original_idx, key_val = key_tuple
        try:
            client = _build_plain_client(key_val)
            logger.info(
                "Attempt %d/%d - Using Vertex Express Mode SDK for model %s (base: %s) with API key (original index: %d).",
                attempt + 1, total_keys, model, base_model_name, original_idx,
            )
            return client, key_val
        except Exception as e:
            logger.warning(
                "Attempt %d/%d - Vertex Express Mode client init failed for API key (original index: %d) for model %s: %s. Trying next key.",
                attempt + 1, total_keys, original_idx, model, e,
            )

    raise GeminiClientError(
        f"All {total_keys} configured Express API keys failed to initialize or were unavailable for model '{model}'.",
        status_code=500,
        error_type="server_error",
    )


# ── project id discovery ────────────────────────────────────────────────────


# Global cache for project IDs: {api_key: project_id}
PROJECT_ID_CACHE: Dict[str, str] = {}


async def discover_project_id(api_key: str) -> str:
    """
    Discover the numeric project ID behind an Express key by triggering an
    intentional error with a non-existent model and parsing the error message.

    Raises Exception if project ID discovery fails.
    """
    if api_key in PROJECT_ID_CACHE:
        logger.info("Using cached project ID: %s", PROJECT_ID_CACHE[api_key])
        return PROJECT_ID_CACHE[api_key]

    error_url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"gemini-2.7-pro-preview-05-06:streamGenerateContent?key={api_key}"
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": "test"}]}]}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                error_url,
                json=payload,
                proxy=app_config.PROXY_URL,
                ssl=app_config.SSL_CERT_FILE,
            ) as response:
                response_text = await response.text()
                try:
                    error_data = json.loads(response_text)
                    if isinstance(error_data, list) and len(error_data) > 0:
                        error_data = error_data[0]
                    if "error" in error_data:
                        error_message = error_data["error"].get("message", "")
                        # Pattern: "projects/39982734461/locations/..."
                        match = re.search(r"projects/(\d+)/locations/", error_message)
                        if match:
                            project_id = match.group(1)
                            PROJECT_ID_CACHE[api_key] = project_id
                            logger.info("Discovered project ID: %s", project_id)
                            return project_id
                except json.JSONDecodeError:
                    match = re.search(r"projects/(\d+)/locations/", response_text)
                    if match:
                        project_id = match.group(1)
                        PROJECT_ID_CACHE[api_key] = project_id
                        logger.info("Discovered project ID from raw response: %s", project_id)
                        return project_id

                raise RuntimeError(
                    f"Failed to discover project ID. Status: {response.status}, Response: {response_text[:500]}"
                )
        except Exception as e:
            logger.error("Failed to discover project ID: %s", e)
            raise


# ── upstream error classification ───────────────────────────────────────────


def _get_upstream_status_code(exc: Exception) -> Optional[int]:
    for attr_name in ("status_code", "code"):
        attr_value = getattr(exc, attr_name, None)
        if isinstance(attr_value, int):
            return attr_value
        if isinstance(attr_value, str) and attr_value.isdigit():
            return int(attr_value)

    response = getattr(exc, "response", None)
    for response_attr_name in ("status_code", "status"):
        response_status = getattr(response, response_attr_name, None)
        if isinstance(response_status, int):
            return response_status

    return None


def _is_upstream_429_error(exc: Exception) -> bool:
    status_code = _get_upstream_status_code(exc)
    if status_code == 429:
        return True

    exc_text = str(exc)
    return "429" in exc_text and (
        "Too Many Requests" in exc_text or "RESOURCE_EXHAUSTED" in exc_text
    )


def _is_not_found_error(exc: Exception) -> bool:
    """404/400 from upstream — conditions that trigger the project-id fallback."""
    return _get_upstream_status_code(exc) in (404, 400)


async def _sleep_before_429_retry(exc: Exception, retry_number: int, model_name: str) -> None:
    delay_ms = app_config.RETRY_INTERVAL_MS
    logger.warning(
        "Upstream 429 for Gemini model '%s'. Retrying %d/%d after fixed interval %dms.",
        model_name, retry_number, app_config.RETRY_COUNT, delay_ms,
    )
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000.0)


def _log_429_retry_recovered(model_name: str, retry_number: int) -> None:
    if retry_number <= 0:
        return
    logger.info(
        "Upstream 429 retry recovered for Gemini model '%s' after %d retry attempt(s).",
        model_name, retry_number,
    )


# ── upstream calls with fallback + 429 retries ──────────────────────────────


async def generate_gemini_content(
    express_key_manager: Any,
    model_for_api_call: str,
    contents: list,
    gen_config_dict: Dict[str, Any],
) -> Any:
    """
    Non-streaming generate_content. Builds an Express client, retries 429s,
    and rebuilds the client once against the discovered project id when the
    first attempt fails with 404/400.
    """
    client, api_key = await resolve_gemini_client(
        model=model_for_api_call,
        base_model_name=model_for_api_call,
        express_key_manager=express_key_manager,
    )
    retry_number = 0
    used_project_fallback = False
    while True:
        try:
            response = await client.aio.models.generate_content(
                model=model_for_api_call,
                contents=contents,
                config=gen_config_dict,
            )
            _log_429_retry_recovered(model_for_api_call, retry_number)
            return response
        except Exception as exc:
            if _is_not_found_error(exc) and not used_project_fallback:
                used_project_fallback = True
                logger.warning(
                    "Upstream %s for Gemini model '%s' on the default endpoint "
                    "(error: %s). Retrying once with a project-id base URL.",
                    _get_upstream_status_code(exc), model_for_api_call, str(exc)[:512],
                )
                project_id = await discover_project_id(api_key)
                client = _build_project_client(api_key, project_id)
                continue
            if _is_upstream_429_error(exc) and retry_number < app_config.RETRY_COUNT:
                retry_number += 1
                await _sleep_before_429_retry(exc, retry_number, model_for_api_call)
                continue
            raise


async def stream_gemini_content(
    express_key_manager: Any,
    model_for_api_call: str,
    contents: list,
    gen_config_dict: Dict[str, Any],
):
    """
    Async generator of raw Gemini stream chunks.

    Retries 429s before the first chunk; if the stream setup fails with
    404/400 before any chunk has been yielded, rebuilds the client once
    against the discovered project id and retries.
    """
    client, api_key = await resolve_gemini_client(
        model=model_for_api_call,
        base_model_name=model_for_api_call,
        express_key_manager=express_key_manager,
    )
    retry_number = 0
    used_project_fallback = False
    has_yielded_any_chunk = False
    while True:
        try:
            stream_gen_obj = await client.aio.models.generate_content_stream(
                model=model_for_api_call,
                contents=contents,
                config=gen_config_dict,
            )
            async for chunk_item in stream_gen_obj:
                if not has_yielded_any_chunk:
                    _log_429_retry_recovered(model_for_api_call, retry_number)
                has_yielded_any_chunk = True
                yield chunk_item
            return
        except Exception as e_stream:
            if (
                _is_not_found_error(e_stream)
                and not has_yielded_any_chunk
                and not used_project_fallback
            ):
                used_project_fallback = True
                logger.warning(
                    "Upstream %s for Gemini model '%s' on the default endpoint "
                    "(error: %s). Retrying stream once with a project-id base URL.",
                    _get_upstream_status_code(e_stream), model_for_api_call, str(e_stream)[:512],
                )
                project_id = await discover_project_id(api_key)
                client = _build_project_client(api_key, project_id)
                continue
            if (
                _is_upstream_429_error(e_stream)
                and not has_yielded_any_chunk
                and retry_number < app_config.RETRY_COUNT
            ):
                retry_number += 1
                await _sleep_before_429_retry(e_stream, retry_number, model_for_api_call)
                continue
            raise
