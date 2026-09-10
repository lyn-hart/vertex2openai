"""
Anthropic Messages API routes for Claude Code compatibility.

POST /v1/messages
POST /v1/messages/count_tokens
"""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from auth import get_api_key
from anthropic_models import AnthropicMessagesRequest
from translate_anthropic import (
    anthropic_error,
    count_tokens_for_request,
    create_anthropic_gemini_contents,
    create_anthropic_generation_config,
    gemini_response_to_anthropic,
    GeminiAnthropicStreamAssembler,
)
from client import (
    _is_upstream_429_error,
    generate_gemini_content,
    parse_model_features,
    stream_gemini_content,
)

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


def _anthropic_error_response(message: str, *, status: int = 500, err_type: str = "api_error"):
    return JSONResponse(
        status_code=status,
        content=anthropic_error(message, status=status, err_type=err_type),
    )


@router.post("/v1/messages")
async def create_message(
    fastapi_request: Request,
    request: AnthropicMessagesRequest,
    api_key: str = Depends(get_api_key),
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    anthropic_beta: str | None = Header(default=None, alias="anthropic-beta"),
):
    """Anthropic Messages create — Gemini-direct conversion for Claude Code."""
    _ = anthropic_version  # accepted for client compatibility
    _ = anthropic_beta
    _ = api_key

    if not request.messages:
        return _anthropic_error_response(
            "messages: Field required",
            status=400,
            err_type="invalid_request_error",
        )
    if not request.model:
        return _anthropic_error_response(
            "model: Field required",
            status=400,
            err_type="invalid_request_error",
        )

    try:
        features = parse_model_features(request.model)
        base_model_name = features.base_model_name
        express_key_manager = fastapi_request.app.state.express_key_manager

        contents = create_anthropic_gemini_contents(request.messages)
        gen_config = create_anthropic_generation_config(
            request,
            base_model_name=base_model_name,
            is_grounded_search=features.is_grounded_search,
            is_nothinking_model=features.is_nothinking_model,
            is_max_thinking_model=features.is_max_thinking_model,
        )

        logger.info(f"/v1/messages model='{request.model}' base='{base_model_name}' "
            f"stream={bool(request.stream)} tools={bool(request.tools)}")

        if request.stream:
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            estimated_input = count_tokens_for_request(request).get("input_tokens", 0)

            async def anthropic_sse_generator():
                assembler = GeminiAnthropicStreamAssembler(
                    message_id=message_id,
                    model=request.model,
                    input_tokens=estimated_input,
                )
                try:
                    async for chunk in stream_gemini_content(
                        express_key_manager, base_model_name, contents, gen_config
                    ):
                        for ev in assembler.process_chunk(chunk):
                            yield ev
                    for ev in assembler.finish():
                        yield ev
                except Exception as e:
                    logger.error(f"Anthropic stream failed for model '{base_model_name}': {e}")
                    err_type = "rate_limit_error" if _is_upstream_429_error(e) else "api_error"
                    for ev in assembler.error_close(str(e)[:1024], err_type=err_type):
                        yield ev

            return StreamingResponse(
                anthropic_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        try:
            response_obj = await generate_gemini_content(
                express_key_manager, base_model_name, contents, gen_config
            )
        except Exception as e:
            logger.error(f"Anthropic non-stream generate failed for '{base_model_name}': {e}")
            status = 429 if _is_upstream_429_error(e) else 500
            err_type = "rate_limit_error" if status == 429 else "api_error"
            return _anthropic_error_response(str(e)[:1024], status=status, err_type=err_type)

        if (
            hasattr(response_obj, "prompt_feedback")
            and hasattr(response_obj.prompt_feedback, "block_reason")
            and response_obj.prompt_feedback.block_reason
        ):
            block_msg = f"Blocked (Gemini): {response_obj.prompt_feedback.block_reason}"
            return _anthropic_error_response(block_msg, status=400, err_type="invalid_request_error")

        anthropic_body = gemini_response_to_anthropic(
            response_obj,
            model=request.model,
        )
        return JSONResponse(content=anthropic_body)

    except Exception as e:
        error_msg = f"Unexpected error in /v1/messages: {str(e)}"
        logger.info(error_msg)
        return _anthropic_error_response(error_msg, status=500)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: AnthropicMessagesRequest,
    api_key: str = Depends(get_api_key),
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
):
    """Local approximate token count (no upstream tokenizer)."""
    _ = anthropic_version
    _ = api_key
    if not request.messages and request.system is None:
        return _anthropic_error_response(
            "messages or system is required",
            status=400,
            err_type="invalid_request_error",
        )
    return JSONResponse(content=count_tokens_for_request(request))
