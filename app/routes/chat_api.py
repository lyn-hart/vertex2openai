from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from translate_openai import (
    create_gemini_prompt,
    create_generation_config,
    create_openai_error_response,
)
from streaming import execute_gemini_call
from client import (
    parse_model_features,
    apply_thinking_config,
)

import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        features = parse_model_features(request.model)
        base_model_name = features.base_model_name
        # Grounded search: -search model suffix OR request-body params
        # (search: true / web_search_options: {...}).
        is_grounded_search = (
            features.is_grounded_search
            or request.search is True
            or request.web_search_options is not None
        )
        is_nothinking_model = features.is_nothinking_model
        is_max_thinking_model = features.is_max_thinking_model

        # This will now be a dictionary
        gen_config_dict = create_generation_config(request)

        if "gemini-2.5-flash" in base_model_name or "gemini-2.5-pro" in base_model_name:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        if "gemini-2.5-flash-lite" in base_model_name:
            gen_config_dict["thinking_config"]["include_thoughts"] = False

        current_prompt_func = create_gemini_prompt

        if is_grounded_search:
            # Vertex does not support Google Search grounding together with
            # function calling: the upstream keeps the function tools and
            # silently drops the search tool (or rejects with 400). We still
            # send both, but warn so the behavior is diagnosable.
            has_function_tools = any(
                isinstance(t, types.Tool) and getattr(t, "function_declarations", None)
                for t in gen_config_dict.get("tools", [])
            )
            if has_function_tools:
                logger.warning(
                    "Search requested together with function tools for model '%s'; "
                    "the upstream endpoint ignores search when function tools are present.",
                    request.model,
                )
            # google_search must ride on the same Tool object as any function
            # declarations — separate tool entries are rejected by the SDK's
            # t_tools transformer.
            search_added = False
            for existing_tool in gen_config_dict.get("tools", []):
                if isinstance(existing_tool, types.Tool) and getattr(existing_tool, "function_declarations", None):
                    existing_tool.google_search = types.GoogleSearch()
                    search_added = True
                    break
            if not search_added:
                gen_config_dict.setdefault("tools", []).append(
                    types.Tool(google_search=types.GoogleSearch())
                )

        apply_thinking_config(
            gen_config_dict,
            base_model_name,
            is_nothinking_model=is_nothinking_model,
            is_max_thinking_model=is_max_thinking_model,
        )

        return await execute_gemini_call(express_key_manager_instance, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        logger.info(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
