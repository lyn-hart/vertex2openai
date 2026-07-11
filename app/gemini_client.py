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


OPENAI_DIRECT_SUFFIX = "-openai"
OPENAI_SEARCH_SUFFIX = "-openaisearch"
EXPERIMENTAL_MARKER = "-exp-"
PAY_PREFIX = "[PAY]"
EXPRESS_PREFIX = "[EXPRESS] "  # trailing space for easier stripping


@dataclass
class ModelFeatures:
    """Parsed model string features after prefix/suffix stripping."""

    original_model: str
    base_model_name: str
    is_express_model_request: bool = False
    is_pay_model_request: bool = False
    is_openai_direct_model: bool = False
    is_openai_search_model: bool = False
    is_auto_model: bool = False
    is_grounded_search: bool = False
    is_encrypted_model: bool = False
    is_encrypted_full_model: bool = False
    is_nothinking_model: bool = False
    is_max_thinking_model: bool = False
    is_2k_image_model: bool = False
    is_4k_image_model: bool = False


def parse_model_features(model: str) -> ModelFeatures:
    """Parse model name prefixes/suffixes used by this adapter."""
    is_openai_direct_model = False
    is_openai_search_model = False
    if model.endswith(OPENAI_DIRECT_SUFFIX) or model.endswith(OPENAI_SEARCH_SUFFIX):
        is_openai_search_model = model.endswith(OPENAI_SEARCH_SUFFIX)
        suffix_to_remove = OPENAI_SEARCH_SUFFIX if is_openai_search_model else OPENAI_DIRECT_SUFFIX
        temp_name_for_marker_check = model[: -len(suffix_to_remove)]
        if (
            temp_name_for_marker_check.startswith(PAY_PREFIX)
            or temp_name_for_marker_check.startswith(EXPRESS_PREFIX)
            or EXPERIMENTAL_MARKER in temp_name_for_marker_check
        ):
            is_openai_direct_model = True

    is_auto_model = model.endswith("-auto")
    is_grounded_search = model.endswith("-search")
    is_encrypted_model = model.endswith("-encrypt")
    is_encrypted_full_model = model.endswith("-encrypt-full")
    is_nothinking_model = model.endswith("-nothinking")
    is_max_thinking_model = model.endswith("-max")
    is_2k_image_model = model.endswith("-2k")
    is_4k_image_model = model.endswith("-4k")

    base_model_name = model
    is_express_model_request = False
    is_pay_model_request = False
    # Legacy "[EXPRESS] " prefix still accepted for backward compatibility.
    if base_model_name.startswith(EXPRESS_PREFIX):
        is_express_model_request = True
        base_model_name = base_model_name[len(EXPRESS_PREFIX) :]

    if base_model_name.startswith(PAY_PREFIX):
        is_pay_model_request = True
        base_model_name = base_model_name[len(PAY_PREFIX) :]

    if is_openai_direct_model:
        suffix_to_remove = OPENAI_SEARCH_SUFFIX if is_openai_search_model else OPENAI_DIRECT_SUFFIX
        temp_base_for_openai = model[: -len(suffix_to_remove)]
        if temp_base_for_openai.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            temp_base_for_openai = temp_base_for_openai[len(EXPRESS_PREFIX) :]
        if temp_base_for_openai.startswith(PAY_PREFIX):
            is_pay_model_request = True
            temp_base_for_openai = temp_base_for_openai[len(PAY_PREFIX) :]
        base_model_name = temp_base_for_openai
    elif is_auto_model:
        base_model_name = base_model_name[: -len("-auto")]
    elif is_grounded_search:
        base_model_name = base_model_name[: -len("-search")]
    elif is_encrypted_full_model:
        base_model_name = base_model_name[: -len("-encrypt-full")]
    elif is_encrypted_model:
        base_model_name = base_model_name[: -len("-encrypt")]
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
        is_express_model_request=is_express_model_request,
        is_pay_model_request=is_pay_model_request,
        is_openai_direct_model=is_openai_direct_model,
        is_openai_search_model=is_openai_search_model,
        is_auto_model=is_auto_model,
        is_grounded_search=is_grounded_search,
        is_encrypted_model=is_encrypted_model,
        is_encrypted_full_model=is_encrypted_full_model,
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
    is_express_model_request: bool,
    credential_manager: Any,
    express_key_manager: Any,
    skip_if_openai_direct: bool = False,
    is_openai_direct_model: bool = False,
    is_pay_model_request: bool = False,
) -> Optional[Any]:
    """
    Build a google.genai Client for Gemini (Express or SA).

    Credential selection:
    - Explicit [PAY] → SA only
    - Explicit legacy [EXPRESS] → Express only
    - Bare model name → Express if keys are configured, else SA

    Returns None when skip_if_openai_direct and is_openai_direct_model are both True
    (OpenAI Direct path handles its own client).

    Raises GeminiClientError on auth/config failures.
    """
    if skip_if_openai_direct and is_openai_direct_model:
        return None

    client_to_use = None
    has_express = express_key_manager.get_total_keys() > 0
    # Explicit [PAY] → SA; explicit legacy [EXPRESS] → Express; bare name → Express if available.
    if is_pay_model_request and not is_express_model_request:
        use_express = False
    elif is_express_model_request:
        use_express = True
    else:
        use_express = has_express

    if use_express:
        if not has_express:
            raise GeminiClientError(
                f"Model '{model}' requires an Express API key, but none are configured.",
                status_code=401,
                error_type="authentication_error",
            )

        print(f"INFO: Attempting Vertex Express Mode for model request: {model} (base: {base_model_name})")
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
                        print(
                            f"INFO: Attempt {attempt + 1}/{total_keys} - Using Vertex Express Mode with custom base URL "
                            f"for model {model} (base: {base_model_name}) with API key (original index: {original_idx})."
                        )
                    else:
                        client_to_use = genai.Client(vertexai=True, api_key=key_val)
                        print(
                            f"INFO: Attempt {attempt + 1}/{total_keys} - Using Vertex Express Mode SDK "
                            f"for model {model} (base: {base_model_name}) with API key (original index: {original_idx})."
                        )
                    break
                except Exception as e:
                    print(
                        f"WARNING: Attempt {attempt + 1}/{total_keys} - Vertex Express Mode client init failed "
                        f"for API key (original index: {original_idx}) for model {model}: {e}. Trying next key."
                    )
                    client_to_use = None
            else:
                print(f"WARNING: Attempt {attempt + 1}/{total_keys} - get_express_api_key() returned None unexpectedly.")
                client_to_use = None

        if client_to_use is None:
            raise GeminiClientError(
                f"All {total_keys} configured Express API keys failed to initialize or were unavailable for model '{model}'.",
                status_code=500,
                error_type="server_error",
            )
        return client_to_use

    # SA credentials path
    print(f"INFO: Model '{model}' is an SA credential request for Gemini. Attempting SA credentials.")
    rotated_credentials, rotated_project_id = credential_manager.get_credentials()

    if rotated_credentials and rotated_project_id:
        try:
            client_to_use = genai.Client(
                vertexai=True,
                credentials=rotated_credentials,
                project=rotated_project_id,
                location="global",
            )
            print(f"INFO: Using SA credential for Gemini model {model} (project: {rotated_project_id})")
            return client_to_use
        except Exception as e:
            raise GeminiClientError(
                f"SA credential client initialization failed for Gemini model '{model}': {e}.",
                status_code=500,
                error_type="server_error",
            )

    raise GeminiClientError(
        f"Model '{model}' requires SA credentials for Gemini, but none are available or loaded.",
        status_code=401,
        error_type="authentication_error",
    )
