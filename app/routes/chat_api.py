from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import create_gemini_prompt
from api_helpers import (
    create_generation_config, # Corrected import name
    create_openai_error_response,
    execute_gemini_call,
)
from gemini_client import (
    parse_model_features,
    resolve_gemini_client,
    apply_thinking_config,
    GeminiClientError,
)

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        express_key_manager_instance = fastapi_request.app.state.express_key_manager

        features = parse_model_features(request.model)
        base_model_name = features.base_model_name
        is_grounded_search = features.is_grounded_search
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

        client_to_use = None
        try:
            client_to_use = await resolve_gemini_client(
                model=request.model,
                base_model_name=base_model_name,
                express_key_manager=express_key_manager_instance,
            )
        except GeminiClientError as e:
            print(f"ERROR: {e.message}")
            return JSONResponse(
                status_code=e.status_code,
                content=create_openai_error_response(e.status_code, e.message, e.error_type),
            )

        # For Gemini models, client_to_use must be set, or an error returned above.
        if client_to_use is None:
            print(f"CRITICAL ERROR: Client for Gemini model '{request.model}' was not initialized, and no specific error was returned. This indicates a logic flaw.")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        current_prompt_func = create_gemini_prompt

        if is_grounded_search:
            search_tool = types.Tool(google_search=types.GoogleSearch())
            # Add or update the 'tools' key in the gen_config_dict
            if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                gen_config_dict["tools"].append(search_tool)
            else:
                gen_config_dict["tools"] = [search_tool]

        apply_thinking_config(
            gen_config_dict,
            base_model_name,
            is_nothinking_model=is_nothinking_model,
            is_max_thinking_model=is_max_thinking_model,
        )

        return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
