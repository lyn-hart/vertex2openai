import json
import time
import math
import asyncio
from typing import Any, Callable, Dict, List, Optional

from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types

from models import OpenAIRequest, OpenAIMessage
import config as app_config
from config import VERTEX_REASONING_TAG
from client import (
    GeminiBlockedError,
    GeminiClientError,
    generate_gemini_content,
    stream_gemini_content,
    _is_upstream_429_error,
)
from translate_openai import (
    chunk_has_finish_reason,
    convert_to_openai_format,
    convert_chunk_to_openai,
    create_final_chunk,
    create_openai_error_response,
    is_gemini_response_valid,
)

import logging
logger = logging.getLogger(__name__)

class StreamingReasoningProcessor:
    def __init__(self, tag_name: str = VERTEX_REASONING_TAG):
        self.tag_name = tag_name
        self.open_tag = f"<{tag_name}>"
        self.close_tag = f"</{tag_name}>"
        self.tag_buffer = ""
        self.inside_tag = False
        self.reasoning_buffer = ""
        self.partial_tag_buffer = "" 

    def process_chunk(self, content: str) -> tuple[str, str]:
        if self.partial_tag_buffer:
            content = self.partial_tag_buffer + content
            self.partial_tag_buffer = ""
        self.tag_buffer += content
        processed_content = ""
        current_reasoning = ""
        while self.tag_buffer:
            if not self.inside_tag:
                open_pos = self.tag_buffer.find(self.open_tag)
                if open_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.open_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.open_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                processed_content += self.tag_buffer[:-i]
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        processed_content += self.tag_buffer
                        self.tag_buffer = ""
                    break
                else:
                    processed_content += self.tag_buffer[:open_pos]
                    self.tag_buffer = self.tag_buffer[open_pos + len(self.open_tag):]
                    self.inside_tag = True
            else: 
                close_pos = self.tag_buffer.find(self.close_tag)
                if close_pos == -1:
                    partial_match = False
                    for i in range(1, min(len(self.close_tag), len(self.tag_buffer) + 1)):
                        if self.tag_buffer[-i:] == self.close_tag[:i]:
                            partial_match = True
                            if len(self.tag_buffer) > i:
                                new_reasoning = self.tag_buffer[:-i]
                                self.reasoning_buffer += new_reasoning
                                if new_reasoning: current_reasoning = new_reasoning
                                self.partial_tag_buffer = self.tag_buffer[-i:]
                            else: self.partial_tag_buffer = self.tag_buffer
                            self.tag_buffer = ""
                            break
                    if not partial_match:
                        if self.tag_buffer:
                            self.reasoning_buffer += self.tag_buffer
                            current_reasoning = self.tag_buffer
                            self.tag_buffer = ""
                    break
                else:
                    final_reasoning_chunk = self.tag_buffer[:close_pos]
                    self.reasoning_buffer += final_reasoning_chunk
                    if final_reasoning_chunk: current_reasoning = final_reasoning_chunk
                    self.reasoning_buffer = "" 
                    self.tag_buffer = self.tag_buffer[close_pos + len(self.close_tag):]
                    self.inside_tag = False
        return processed_content, current_reasoning
    
    def flush_remaining(self) -> tuple[str, str]:
        remaining_content, remaining_reasoning = "", ""
        if self.partial_tag_buffer:
            remaining_content += self.partial_tag_buffer
            self.partial_tag_buffer = ""
        if not self.inside_tag:
            if self.tag_buffer: remaining_content += self.tag_buffer
        else:
            if self.reasoning_buffer: remaining_reasoning = self.reasoning_buffer
            if self.tag_buffer: remaining_content += self.tag_buffer
            self.inside_tag = False
        self.tag_buffer, self.reasoning_buffer = "", ""
        return remaining_content, remaining_reasoning


async def _chunk_openai_response_dict_for_sse(
    openai_response_dict: Dict[str, Any],
    response_id_override: Optional[str] = None, 
    model_name_override: Optional[str] = None
):
    resp_id = response_id_override or openai_response_dict.get("id", f"chatcmpl-fakestream-{int(time.time())}")
    model_name = model_name_override or openai_response_dict.get("model", "unknown")
    created_time = openai_response_dict.get("created", int(time.time()))
    
    choices = openai_response_dict.get("choices", [])
    if not choices: 
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    for choice_idx, choice in enumerate(choices): 
        message = choice.get("message", {})
        final_finish_reason = choice.get("finish_reason", "stop")

        if message.get("tool_calls"):
            tool_calls_list = message.get("tool_calls", [])
            for tc_item_idx, tool_call_item in enumerate(tool_calls_list):
                delta_tc_start = {
                    "tool_calls": [{
                        "index": tc_item_idx, 
                        "id": tool_call_item["id"],
                        "type": "function",
                        "function": {"name": tool_call_item["function"]["name"], "arguments": ""}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_start, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01) 

                delta_tc_args = {
                    "tool_calls": [{
                        "index": tc_item_idx,
                        "id": tool_call_item["id"], 
                        "function": {"arguments": tool_call_item["function"]["arguments"]}
                    }]
                }
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_tc_args, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.01)
        
        elif message.get("content") is not None or message.get("reasoning_content") is not None : 
            reasoning_content = message.get("reasoning_content", "")
            actual_content = message.get("content") 

            if reasoning_content:
                delta_reasoning = {"reasoning_content": reasoning_content}
                yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': delta_reasoning, 'finish_reason': None}]})}\n\n"
                if actual_content is not None: await asyncio.sleep(0.05)

            content_to_chunk = actual_content if actual_content is not None else ""
            if actual_content is not None:
                chunk_size = max(1, math.ceil(len(content_to_chunk) / 10)) if content_to_chunk else 1
                if not content_to_chunk and not reasoning_content : 
                    yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': ''}, 'finish_reason': None}]})}\n\n"
                else:
                    for i in range(0, len(content_to_chunk), chunk_size):
                        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {'content': content_to_chunk[i:i+chunk_size]}, 'finish_reason': None}]})}\n\n"
                        if len(content_to_chunk) > chunk_size: await asyncio.sleep(0.05)
        
        yield f"data: {json.dumps({'id': resp_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model_name, 'choices': [{'index': choice_idx, 'delta': {}, 'finish_reason': final_finish_reason}]})}\n\n"

    yield "data: [DONE]\n\n"


async def gemini_fake_stream_generator(
    express_key_manager: Any,
    model_for_api_call: str,
    prompt_for_api_call: List[types.Content],
    gen_config_dict_for_api_call: Dict[str, Any],
    request_obj: OpenAIRequest,
):
    logger.info(f"FAKE STREAMING (Gemini): Prep for '{request_obj.model}' (API model string: '{model_for_api_call}')")

    api_call_task = asyncio.create_task(
        generate_gemini_content(
            express_key_manager,
            model_for_api_call,
            prompt_for_api_call,
            gen_config_dict_for_api_call,
        )
    )

    outer_keep_alive_interval = app_config.FAKE_STREAMING_INTERVAL_SECONDS
    if outer_keep_alive_interval > 0:
        while not api_call_task.done():
            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(outer_keep_alive_interval)
    
    try:
        raw_gemini_response = await api_call_task 
        openai_response_dict = convert_to_openai_format(raw_gemini_response, request_obj.model)
        
        if hasattr(raw_gemini_response, 'prompt_feedback') and \
           hasattr(raw_gemini_response.prompt_feedback, 'block_reason') and \
           raw_gemini_response.prompt_feedback.block_reason:
            block_message = f"Response blocked by Gemini safety filter: {raw_gemini_response.prompt_feedback.block_reason}"
            if hasattr(raw_gemini_response.prompt_feedback, 'block_reason_message') and \
               raw_gemini_response.prompt_feedback.block_reason_message:
                block_message += f" (Message: {raw_gemini_response.prompt_feedback.block_reason_message})"
            raise ValueError(block_message)

        async for chunk_sse in _chunk_openai_response_dict_for_sse(
            openai_response_dict=openai_response_dict
        ):
            yield chunk_sse

    except Exception as e_outer_gemini:
        err_msg_detail = f"Error in gemini_fake_stream_generator (model: '{request_obj.model}'): {type(e_outer_gemini).__name__} - {str(e_outer_gemini)}"
        logger.error(f"{err_msg_detail}")
        sse_err_msg_display = str(e_outer_gemini)
        if len(sse_err_msg_display) > 512: sse_err_msg_display = sse_err_msg_display[:512] + "..."
        status_code = 429 if _is_upstream_429_error(e_outer_gemini) else 500
        error_type = "rate_limit_error" if status_code == 429 else "server_error"
        err_resp_sse = create_openai_error_response(status_code, sse_err_msg_display, error_type)
        json_payload_error = json.dumps(err_resp_sse)
        yield f"data: {json_payload_error}\n\n"
        yield create_final_chunk(request_obj.model, "chatcmpl-keepalive")
        yield "data: [DONE]\n\n"
        return


async def execute_gemini_call(
    express_key_manager: Any,
    model_to_call: str,
    prompt_func: Callable[[List[OpenAIMessage]], List[types.Content]],
    gen_config_dict: Dict[str, Any],
    request_obj: OpenAIRequest,
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    logger.info(f"execute_gemini_call for requested API model '{model_to_call}'. Original request model: '{request_obj.model}'")

    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                gemini_fake_stream_generator(
                    express_key_manager, model_to_call, actual_prompt_for_call,
                    gen_config_dict,
                    request_obj
                ), media_type="text/event-stream"
            )
        else: # True Streaming
            response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
            async def _gemini_real_stream_generator_inner():
                saw_finish_reason = False
                try:
                    async for chunk_item_call in stream_gemini_content(
                        express_key_manager, model_to_call, actual_prompt_for_call, gen_config_dict
                    ):
                        if chunk_has_finish_reason(chunk_item_call):
                            saw_finish_reason = True
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, 0)
                    # Gemini normally closes with a chunk carrying the finish
                    # reason, but a truncated upstream stream may not. Clients
                    # treat a stream that ends without one as interrupted, so
                    # always terminate with a terminal chunk.
                    if not saw_finish_reason:
                        yield create_final_chunk(request_obj.model, response_id_for_stream)
                    yield "data: [DONE]\n\n"
                    return
                except Exception as e_stream_call:
                    err_msg_detail_stream = f"Streaming Error (Gemini API, model string: '{model_to_call}'): {type(e_stream_call).__name__} - {str(e_stream_call)}"
                    logger.error(f"{err_msg_detail_stream}")
                    s_err = str(e_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                    status_code = 400 if isinstance(e_stream_call, GeminiBlockedError) else (429 if _is_upstream_429_error(e_stream_call) else 500)
                    error_type = "invalid_request_error" if status_code == 400 else ("rate_limit_error" if status_code == 429 else "server_error")
                    err_resp = create_openai_error_response(status_code, s_err, error_type)
                    j_err = json.dumps(err_resp)
                    yield f"data: {j_err}\n\n"
                    # Terminate the stream properly even on failure: without a
                    # chunk carrying a finish_reason the client just sees the
                    # connection end mid-response.
                    yield create_final_chunk(request_obj.model, response_id_for_stream)
                    yield "data: [DONE]\n\n"
                    return
            return StreamingResponse(_gemini_real_stream_generator_inner(), media_type="text/event-stream")
    else: # Non-streaming
        try:
            response_obj_call = await generate_gemini_content(
                express_key_manager,
                model_to_call,
                actual_prompt_for_call,
                gen_config_dict
            )
        except GeminiClientError as e_client:
            return JSONResponse(
                status_code=e_client.status_code,
                content=create_openai_error_response(e_client.status_code, e_client.message, e_client.error_type),
            )
        except Exception as e_non_stream_call:
            if _is_upstream_429_error(e_non_stream_call):
                s_err = str(e_non_stream_call); s_err = s_err[:1024]+"..." if len(s_err)>1024 else s_err
                return JSONResponse(
                    status_code=429,
                    content=create_openai_error_response(429, s_err, "rate_limit_error")
                )
            raise
        if hasattr(response_obj_call, 'prompt_feedback') and \
           hasattr(response_obj_call.prompt_feedback, 'block_reason') and \
           response_obj_call.prompt_feedback.block_reason:
            block_msg = f"Blocked (Gemini): {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback,'block_reason_message') and \
               response_obj_call.prompt_feedback.block_reason_message: 
                block_msg+=f" ({response_obj_call.prompt_feedback.block_reason_message})"
            raise ValueError(block_msg)
        
        if not is_gemini_response_valid(response_obj_call):
            error_details = f"Invalid non-streaming Gemini response for model string '{model_to_call}'. "
            if hasattr(response_obj_call, 'candidates'):
                error_details += f"Candidates: {len(response_obj_call.candidates) if response_obj_call.candidates else 0}. "
                if response_obj_call.candidates and len(response_obj_call.candidates) > 0:
                    candidate = response_obj_call.candidates if isinstance(response_obj_call.candidates, list) else response_obj_call.candidates
                    if hasattr(candidate, 'content'):
                        error_details += "Has content. "
                        if hasattr(candidate.content, 'parts'):
                            error_details += f"Parts: {len(candidate.content.parts) if candidate.content.parts else 0}. "
                            if candidate.content.parts and len(candidate.content.parts) > 0:
                                part = candidate.content.parts if isinstance(candidate.content.parts, list) else candidate.content.parts
                                if hasattr(part, 'text'):
                                    text_preview = str(getattr(part, 'text', ''))[:100]
                                    error_details += f"First part text: '{text_preview}'"
                                elif hasattr(part, 'function_call'):
                                    error_details += f"First part is function_call: {part.function_call.name}"
            else:
                error_details += f"Response type: {type(response_obj_call).__name__}"
            raise ValueError(error_details)
        
        openai_response_content = convert_to_openai_format(response_obj_call, request_obj.model)
        return JSONResponse(content=openai_response_content)
