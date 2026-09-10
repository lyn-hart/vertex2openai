import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Google specific imports
from google.genai import types

# Local module imports
from models import OpenAIRequest
from auth import get_api_key
from message_processing import (
    create_gemini_prompt,
    create_encrypted_gemini_prompt,
    create_encrypted_full_gemini_prompt,
    ENCRYPTION_INSTRUCTIONS,
)
from api_helpers import (
    create_generation_config, # Corrected import name
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
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
        is_openai_direct_model = features.is_openai_direct_model
        is_openai_search_model = features.is_openai_search_model
        is_auto_model = features.is_auto_model
        is_grounded_search = features.is_grounded_search
        is_encrypted_model = features.is_encrypted_model
        is_encrypted_full_model = features.is_encrypted_full_model
        is_nothinking_model = features.is_nothinking_model
        is_max_thinking_model = features.is_max_thinking_model
        is_express_model_request = features.is_express_model_request
        is_pay_model_request = features.is_pay_model_request

        # This will now be a dictionary
        gen_config_dict = create_generation_config(request)

        if "gemini-2.5-flash" in base_model_name or "gemini-2.5-pro" in base_model_name:
            if "thinking_config" not in gen_config_dict:
                gen_config_dict["thinking_config"] = {}
            gen_config_dict["thinking_config"]["include_thoughts"] = True

        if "gemini-2.5-flash-lite" in base_model_name:
            gen_config_dict["thinking_config"]["include_thoughts"] = False

        client_to_use = None

        # This client initialization logic is for Gemini models (i.e., non-OpenAI Direct models).
        # If 'is_openai_direct_model' is true, this section will be skipped, and the
        # dedicated 'if is_openai_direct_model:' block later will handle it.
        if not is_openai_direct_model:
            try:
                client_to_use = await resolve_gemini_client(
                    model=request.model,
                    base_model_name=base_model_name,
                    is_express_model_request=is_express_model_request,
                    is_pay_model_request=is_pay_model_request,
                    express_key_manager=express_key_manager_instance,
                )
            except GeminiClientError as e:
                print(f"ERROR: {e.message}")
                return JSONResponse(
                    status_code=e.status_code,
                    content=create_openai_error_response(e.status_code, e.message, e.error_type),
                )

        # If we reach here and client_to_use is still None, it means it's an OpenAI Direct Model,
        # which handles its own client and responses.
        # For Gemini models (Express or SA), client_to_use must be set, or an error returned above.
        if not is_openai_direct_model and client_to_use is None:
             # This case should ideally not be reached if the logic above is correct,
             # as each path (Express/SA for Gemini) should either set client_to_use or return an error.
             # This is a safeguard.
            print(f"CRITICAL ERROR: Client for Gemini model '{request.model}' was not initialized, and no specific error was returned. This indicates a logic flaw.")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        if is_openai_direct_model:
            # Use the new OpenAI handler
            # Express is the only backend; always route OpenAI-direct through it.
            openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
            return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
        elif is_auto_model:
            print(f"Processing auto model: {request.model}")
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": ENCRYPTION_INSTRUCTIONS}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}
            ]
            last_err = None
            for attempt in attempts:
                print(f"Auto-mode attempting: '{attempt['name']}' for model {attempt['model']}")
                # Apply modifier to the dictionary. Ensure modifier returns a dict.
                current_gen_config_dict = attempt["config_modifier"](gen_config_dict.copy())
                try:
                    # Pass is_auto_attempt=True for auto-mode calls
                    result = await execute_gemini_call(client_to_use, attempt["model"], attempt["prompt_func"], current_gen_config_dict, request, is_auto_attempt=True)
                    return result
                except Exception as e_auto:
                    last_err = e_auto
                    print(f"Auto-attempt '{attempt['name']}' for model {attempt['model']} failed: {e_auto}")
                    await asyncio.sleep(1)

            print(f"All auto attempts failed. Last error: {last_err}")
            err_msg = f"All auto-mode attempts failed for model {request.model}. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream:
                # This is the final error handling for auto-mode if all attempts fail AND it was a streaming request
                async def final_auto_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    json_payload_final_auto_error = json.dumps(err_content)
                    # Log the final error being sent to client after all auto-retries failed
                    print(f"DEBUG: Auto-mode all attempts failed. Yielding final error JSON: {json_payload_final_auto_error}")
                    yield f"data: {json_payload_final_auto_error}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_auto_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed without specific error.", "server_error"))

        else: # Not an auto model
            current_prompt_func = create_gemini_prompt
            # Determine the actual model string to call the API with (e.g., "gemini-1.5-pro-search")

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                # Add or update the 'tools' key in the gen_config_dict
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]

            # For encrypted models, system instructions are handled by the prompt_func
            elif is_encrypted_model:
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                current_prompt_func = create_encrypted_full_gemini_prompt

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
