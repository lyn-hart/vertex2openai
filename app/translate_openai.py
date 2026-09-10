import base64
import re
import json
import time
import random # For more unique tool_call_id
from typing import List, Dict, Any, Tuple

from google.genai import types
from models import OpenAIRequest, OpenAIMessage, ContentPartText, ContentPartImage

import logging
logger = logging.getLogger(__name__)

SUPPORTED_ROLES = ["user", "model", "function"] # Added "function" for Gemini

THOUGHT_SIGNATURE_TOOL_CALL_ID_MARKER = "__tsig__"

def extract_reasoning_by_tags(full_text: str, tag_name: str) -> Tuple[str, str]:
    if not tag_name or not isinstance(full_text, str):
        return "", full_text if isinstance(full_text, str) else ""
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    pattern = re.compile(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", re.DOTALL)
    reasoning_parts = pattern.findall(full_text)
    normal_text = pattern.sub('', full_text)
    reasoning_content = "".join(reasoning_parts)
    return reasoning_content.strip(), normal_text.strip()

# Normalize SDK values into raw bytes before serializing them into tool_call_id.
def _coerce_thought_signature_to_bytes(thought_signature: Any) -> bytes:
    if thought_signature is None:
        return b""
    if isinstance(thought_signature, bytes):
        return thought_signature
    if isinstance(thought_signature, bytearray):
        return bytes(thought_signature)
    if isinstance(thought_signature, memoryview):
        return thought_signature.tobytes()
    if isinstance(thought_signature, str):
        return thought_signature.encode("utf-8")
    return b""

# Encode Gemini thought_signature into the OpenAI-facing tool_call_id for round-trip transport.
def _encode_tool_call_id_with_thought_signature(tool_call_id: str, thought_signature: Any) -> str:
    if not tool_call_id:
        return tool_call_id

    base_tool_call_id, _ = _decode_tool_call_id_thought_signature(tool_call_id)
    thought_signature_bytes = _coerce_thought_signature_to_bytes(thought_signature)
    if not thought_signature_bytes:
        return base_tool_call_id

    encoded_signature = base64.urlsafe_b64encode(thought_signature_bytes).decode("ascii").rstrip("=")
    return f"{base_tool_call_id}{THOUGHT_SIGNATURE_TOOL_CALL_ID_MARKER}{encoded_signature}"

# Recover the original Gemini tool call id and optional thought_signature from tool_call_id.
def _decode_tool_call_id_thought_signature(tool_call_id: str) -> Tuple[str, bytes]:
    if not tool_call_id or not isinstance(tool_call_id, str):
        return tool_call_id, b""
    if THOUGHT_SIGNATURE_TOOL_CALL_ID_MARKER not in tool_call_id:
        return tool_call_id, b""

    raw_tool_call_id, encoded_signature = tool_call_id.rsplit(THOUGHT_SIGNATURE_TOOL_CALL_ID_MARKER, 1)
    if not raw_tool_call_id or not encoded_signature:
        return tool_call_id, b""

    padding = "=" * (-len(encoded_signature) % 4)
    try:
        thought_signature = base64.urlsafe_b64decode(encoded_signature + padding)
    except Exception as e:
        logger.info(f"Warning: Failed to decode thought_signature from tool_call_id '{tool_call_id}': {e}")
        return tool_call_id, b""

    return raw_tool_call_id, thought_signature

# Best-effort recovery of a function name from the synthetic tool_call_id generated in this adapter.
def _infer_function_name_from_tool_call_id(tool_call_id: str) -> str:
    raw_tool_call_id, _ = _decode_tool_call_id_thought_signature(tool_call_id)
    if not raw_tool_call_id or not isinstance(raw_tool_call_id, str):
        return ""

    match = re.match(r"^call_[^_]+_\d+_(.+)_\d+$", raw_tool_call_id)
    if not match:
        return ""

    return match.group(1)

# Build a Gemini function_call Part while preserving the original function call id and thought signature.
def _build_function_call_part(function_name: str, parsed_arguments: Dict[str, Any], tool_call_id: str = "", thought_signature: bytes = b"") -> types.Part:
    try:
        function_call = types.FunctionCall(name=function_name, args=parsed_arguments)
        if tool_call_id:
            function_call.id = tool_call_id
        part = types.Part(function_call=function_call)
    except Exception as e:
        logger.info(f"Warning: Failed to build Gemini function_call Part directly for {function_name}: {e}")
        part = types.Part.from_function_call(name=function_name, args=parsed_arguments)
        if tool_call_id and getattr(part, "function_call", None) is not None:
            try:
                part.function_call.id = tool_call_id
            except Exception as id_error:
                logger.info(f"Warning: Failed to set function_call.id for {function_name}: {id_error}")

    if thought_signature:
        try:
            part.thought_signature = thought_signature
        except Exception as signature_error:
            logger.info(f"Warning: Failed to set thought_signature for {function_name}: {signature_error}")

    return part

# Build a Gemini function_response Part while preserving the original function response id.
def _build_function_response_part(function_name: str, tool_output_data: Dict[str, Any], tool_call_id: str = "") -> types.Part:
    try:
        function_response = types.FunctionResponse(name=function_name, response=tool_output_data)
        if tool_call_id:
            function_response.id = tool_call_id
        return types.Part(function_response=function_response)
    except Exception as e:
        logger.info(f"Warning: Failed to build Gemini function_response Part directly for {function_name}: {e}")
        part = types.Part.from_function_response(name=function_name, response=tool_output_data)
        if tool_call_id and getattr(part, "function_response", None) is not None:
            try:
                part.function_response.id = tool_call_id
            except Exception as id_error:
                logger.info(f"Warning: Failed to set function_response.id for {function_name}: {id_error}")
        return part

def _extract_markdown_images_to_parts(text: str) -> Tuple[List[types.Part], str]:
    """
    Extract markdown images from text and convert them to Gemini Parts.
    Returns a tuple of (image_parts, text_without_images)
    """
    parts = []
    remaining_text = text
    
    # Pattern to match markdown images with data URLs
    # Matches: ![alt text](data:image/...;base64,data)
    # Only matches image MIME types to avoid extracting other base64 data
    pattern = r'!\[[^\]]*\]\(data:(image/[^;]+);base64,([^)]+)\)'
    
    matches = list(re.finditer(pattern, text))
    
    if matches:
        # Process matches in reverse order to maintain correct text positions
        for match in reversed(matches):
            mime_type = match.group(1)
            b64_data = match.group(2)
            
            # Validate that it's an image MIME type
            if not mime_type.startswith('image/'):
                continue
            
            try:
                # Convert base64 to bytes
                image_bytes = base64.b64decode(b64_data)
                # Create Gemini image part
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                
                # Remove the markdown image from text
                start, end = match.span()
                remaining_text = remaining_text[:start] + remaining_text[end:]
                
                logger.info(f"Extracted markdown image with mime type: {mime_type}")
            except Exception as e:
                logger.info(f"Error extracting markdown image: {e}")
        
        # Reverse parts list since we processed matches in reverse
        parts.reverse()
    
    # Clean up any extra whitespace that might be left
    remaining_text = re.sub(r'\s+', ' ', remaining_text).strip()
    
    return parts, remaining_text

def create_gemini_prompt(messages: List[OpenAIMessage]) -> List[types.Content]:
    logger.info("Converting OpenAI messages to Gemini format...")
    gemini_messages = []
    pending_function_response_parts = []

    def flush_pending_function_response_parts():
        nonlocal pending_function_response_parts
        if pending_function_response_parts:
            gemini_messages.append(types.Content(role="function", parts=pending_function_response_parts))
            pending_function_response_parts = []

    for idx, message in enumerate(messages):
        role = message.role
        parts = []
        current_gemini_role = "" 

        if role == "tool":
            function_name = message.name or _infer_function_name_from_tool_call_id(message.tool_call_id)
            if function_name and message.tool_call_id and message.content is not None:
                function_response_id, _ = _decode_tool_call_id_thought_signature(message.tool_call_id)
                tool_output_data = {}
                try:
                    if isinstance(message.content, str) and \
                       (message.content.strip().startswith("{") and message.content.strip().endswith("}")) or \
                       (message.content.strip().startswith("[") and message.content.strip().endswith("]")):
                        tool_output_data = json.loads(message.content)
                    else: 
                        tool_output_data = {"result": message.content}
                except json.JSONDecodeError:
                    tool_output_data = {"result": str(message.content)}

                parts.append(_build_function_response_part(
                    function_name=function_name,
                    tool_output_data=tool_output_data,
                    tool_call_id=function_response_id
                ))
                pending_function_response_parts.extend(parts)
                continue
            else:
                logger.info(f"Skipping tool message {idx} due to missing name, tool_call_id, or content.")
                continue
        elif role == "assistant" and message.tool_calls:
            flush_pending_function_response_parts()
            current_gemini_role = "model"
            for tool_call in message.tool_calls:
                function_call_data = tool_call.get("function", {})
                function_name = function_call_data.get("name")
                arguments_str = function_call_data.get("arguments", "{}")
                raw_tool_call_id, thought_signature = _decode_tool_call_id_thought_signature(tool_call.get("id", ""))
                try:
                    parsed_arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    logger.info(f"Warning: Could not parse tool call arguments for {function_name}: {arguments_str}")
                    parsed_arguments = {} 
                
                if function_name:
                    parts.append(_build_function_call_part(
                        function_name=function_name,
                        parsed_arguments=parsed_arguments,
                        tool_call_id=raw_tool_call_id,
                        thought_signature=thought_signature
                    ))
            
            if message.content:
                if isinstance(message.content, str):
                    # Check for markdown images in assistant content too
                    image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                    
                    if clean_text:
                        parts.append(types.Part(text=clean_text))
                    
                    parts.extend(image_parts)
                elif isinstance(message.content, list):
                     for part_item in message.content: 
                        if isinstance(part_item, dict):
                            if part_item.get('type') == 'text':
                                text_content = part_item.get('text', '\n')
                                # Check for markdown images in assistant's text parts
                                image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
                                if clean_text:
                                    parts.append(types.Part(text=clean_text))
                                parts.extend(image_parts)
                            elif part_item.get('type') == 'image_url':
                                image_url_data = part_item.get('image_url', {})
                                image_url = image_url_data.get('url', '')
                                if image_url.startswith('data:'):
                                    mime_match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                                    if mime_match:
                                        mime_type, b64_data = mime_match.groups()
                                        image_bytes = base64.b64decode(b64_data)
                                        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                        elif isinstance(part_item, ContentPartText):
                             parts.append(types.Part(text=part_item.text))
                        elif isinstance(part_item, ContentPartImage):
                            image_url = part_item.image_url.url
                            if image_url.startswith('data:'):
                                mime_match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                                if mime_match:
                                    mime_type, b64_data = mime_match.groups()
                                    image_bytes = base64.b64decode(b64_data)
                                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            if not parts: 
                logger.info(f"Skipping assistant message {idx} with empty/invalid tool_calls and no content.")
                continue
        else: 
            flush_pending_function_response_parts()
            if message.content is None:
                logger.info(f"Skipping message {idx} (Role: {role}) due to None content.")
                continue
            if not message.content and isinstance(message.content, (str, list)) and not len(message.content):
                 logger.info(f"Skipping message {idx} (Role: {role}) due to empty content string or list.")
                 continue

            current_gemini_role = role
            if current_gemini_role == "system": current_gemini_role = "user"
            elif current_gemini_role == "assistant": current_gemini_role = "model"
            
            if current_gemini_role not in SUPPORTED_ROLES:
                logger.info(f"Warning: Role '{current_gemini_role}' (from original '{role}') is not in SUPPORTED_ROLES {SUPPORTED_ROLES}. Mapping to 'user'.")
                current_gemini_role = "user"

            if isinstance(message.content, str):
                # Check for markdown images in the content
                image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                
                # Add text part if there's any remaining text
                if clean_text:
                    parts.append(types.Part(text=clean_text))
                
                # Add extracted image parts
                parts.extend(image_parts)
            elif isinstance(message.content, list):
                for part_item in message.content:
                    if isinstance(part_item, dict):
                        if part_item.get('type') == 'text':
                            text_content = part_item.get('text', '\n')
                            # Check for markdown images in text parts
                            image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
                            if clean_text:
                                parts.append(types.Part(text=clean_text))
                            parts.extend(image_parts)
                        elif part_item.get('type') == 'image_url':
                            image_url_data = part_item.get('image_url', {})
                            image_url = image_url_data.get('url', '')
                            if image_url.startswith('data:'):
                                mime_match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                                if mime_match:
                                    mime_type, b64_data = mime_match.groups()
                                    image_bytes = base64.b64decode(b64_data)
                                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                    elif isinstance(part_item, ContentPartText):
                        parts.append(types.Part(text=part_item.text))
                    elif isinstance(part_item, ContentPartImage):
                        image_url = part_item.image_url.url
                        if image_url.startswith('data:'):
                            mime_match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                            if mime_match:
                                mime_type, b64_data = mime_match.groups()
                                image_bytes = base64.b64decode(b64_data)
                                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            elif message.content is not None: 
                parts.append(types.Part(text=str(message.content)))
            
            if not parts:
                 logger.info(f"Skipping message {idx} (Role: {role}) as it resulted in no processable parts.")
                 continue

        if not current_gemini_role:
            logger.info(f"Error: current_gemini_role not set for message {idx}. Original role: {message.role}. Defaulting to 'user'.")
            current_gemini_role = "user"

        if not parts:
            logger.info(f"Skipping message {idx} (Original role: {message.role}, Mapped Gemini role: {current_gemini_role}) as it resulted in no parts after processing.")
            continue
            
        gemini_messages.append(types.Content(role=current_gemini_role, parts=parts))

    flush_pending_function_response_parts()

    logger.info(f"Converted to {len(gemini_messages)} Gemini messages")
    if not gemini_messages:
        logger.info("Warning: No messages were converted. Returning a dummy user prompt to prevent API errors.")
        return [types.Content(role="user", parts=[types.Part(text="Placeholder prompt: No valid input messages provided.")])]
    
    return gemini_messages

def _convert_image_to_markdown(image_data: bytes, mime_type: str) -> str:
    """Convert image data to markdown format with base64 encoding."""
    try:
        # Convert bytes to base64 string
        b64_data = base64.b64encode(image_data).decode('utf-8')
        # Create markdown image with data URL
        data_url = f"data:{mime_type};base64,{b64_data}"
        # Return markdown formatted image
        return f"![Image]({data_url})"
    except Exception as e:
        logger.info(f"Error converting image to markdown: {e}")
        return "[Image could not be displayed]"

def extract_grounding_citations(candidate: Any) -> str:
    """
    Extract Google Search grounding sources from a Gemini candidate.

    Returns a markdown "Sources:" block ("" when the candidate carries no
    grounding metadata). Handles both SDK objects and plain dicts.
    """
    metadata = getattr(candidate, "grounding_metadata", None)
    if metadata is None and isinstance(candidate, dict):
        metadata = candidate.get("grounding_metadata")
    if not metadata:
        return ""

    chunks = getattr(metadata, "grounding_chunks", None)
    if chunks is None and isinstance(metadata, dict):
        chunks = metadata.get("grounding_chunks")
    if not chunks:
        return ""

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    lines: List[str] = []
    seen_uris: set = set()
    for chunk in chunks:
        web = _get(chunk, "web")
        if not web:
            continue
        uri = _get(web, "uri")
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        title = _get(web, "title") or uri
        lines.append(f"- [{title}]({uri})")

    if not lines:
        return ""
    return "\n\n---\n**Sources:**\n" + "\n".join(lines)


def parse_gemini_response_for_reasoning_and_content(gemini_response_candidate: Any) -> Tuple[str, str]:
    reasoning_text_parts = []
    normal_text_parts = []
    candidate_part_text = ""
    if hasattr(gemini_response_candidate, 'text') and gemini_response_candidate.text is not None:
        candidate_part_text = str(gemini_response_candidate.text)

    gemini_candidate_content = None
    if hasattr(gemini_response_candidate, 'content'):
        gemini_candidate_content = gemini_response_candidate.content

    if gemini_candidate_content and hasattr(gemini_candidate_content, 'parts') and gemini_candidate_content.parts:
        for part_item in gemini_candidate_content.parts:
            if hasattr(part_item, 'function_call') and part_item.function_call is not None: # Kilo Code: Added 'is not None' check
                continue
            
            part_text = ""
            if hasattr(part_item, 'text') and part_item.text is not None:
                part_text = str(part_item.text)
            
            # Check for image parts
            elif hasattr(part_item, 'inline_data') and part_item.inline_data is not None:
                # Handle image data in response
                inline_data = part_item.inline_data
                if hasattr(inline_data, 'data') and hasattr(inline_data, 'mime_type'):
                    image_bytes = inline_data.data
                    mime_type = inline_data.mime_type
                    # Convert image to markdown format
                    part_text = _convert_image_to_markdown(image_bytes, mime_type)
            
            # Check for blob/file reference (for images stored in blob)
            elif hasattr(part_item, 'file_data') and part_item.file_data is not None:
                # Handle file reference (typically for images)
                file_data = part_item.file_data
                if hasattr(file_data, 'file_uri'):
                    # Create a markdown link to the file
                    file_uri = file_data.file_uri
                    mime_type = getattr(file_data, 'mime_type', 'image/png')
                    # For file URIs, we can't embed directly, so we'll create a link
                    part_text = f"![Image]({file_uri})"
                    logger.info(f"Image file reference found: {file_uri}")
            
            part_is_thought = hasattr(part_item, 'thought') and part_item.thought is True

            if part_is_thought:
                reasoning_text_parts.append(part_text)
            elif part_text: # Only add if it's not a function_call and has text or converted image
                normal_text_parts.append(part_text)
    elif candidate_part_text:
        normal_text_parts.append(candidate_part_text)
    elif gemini_candidate_content and hasattr(gemini_candidate_content, 'text') and gemini_candidate_content.text is not None:
        normal_text_parts.append(str(gemini_candidate_content.text))
    elif hasattr(gemini_response_candidate, 'text') and gemini_response_candidate.text is not None and not gemini_candidate_content: # Should be caught by candidate_part_text
        normal_text_parts.append(str(gemini_response_candidate.text))

    return "".join(reasoning_text_parts), "".join(normal_text_parts)

# This function will be the core for converting a full Gemini response.
# It will be called by the non-streaming path and the fake-streaming path.
def process_gemini_response_to_openai_dict(gemini_response_obj: Any, request_model_str: str) -> Dict[str, Any]:
    choices = []
    response_timestamp = int(time.time())
    base_id = f"chatcmpl-{response_timestamp}-{random.randint(1000,9999)}"

    if hasattr(gemini_response_obj, 'candidates') and gemini_response_obj.candidates:
        for i, candidate in enumerate(gemini_response_obj.candidates):
            message_payload = {"role": "assistant"}
            
            raw_finish_reason = getattr(candidate, 'finish_reason', None)
            openai_finish_reason = "stop" # Default
            if raw_finish_reason:
                if hasattr(raw_finish_reason, 'name'): raw_finish_reason_str = raw_finish_reason.name.upper()
                else: raw_finish_reason_str = str(raw_finish_reason).upper()

                if raw_finish_reason_str == "STOP": openai_finish_reason = "stop"
                elif raw_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
                elif raw_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
                elif raw_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"
                # Other reasons like RECITATION, OTHER map to "stop" or a more specific OpenAI reason if available.
            
            function_call_detected = False
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call is not None: # Kilo Code: Added 'is not None' check
                        fc = part.function_call
                        raw_tool_call_id = getattr(fc, 'id', None) or f"call_{base_id}_{i}_{fc.name.replace(' ', '_')}_{int(time.time()*10000 + random.randint(0,9999))}"
                        tool_call_id = _encode_tool_call_id_with_thought_signature(raw_tool_call_id, getattr(part, 'thought_signature', None))
                        
                        if "tool_calls" not in message_payload:
                            message_payload["tool_calls"] = []
                        
                        message_payload["tool_calls"].append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args or {})
                            }
                        })
                        message_payload["content"] = None 
                        openai_finish_reason = "tool_calls" # Override if a tool call is made
                        function_call_detected = True
            
            if not function_call_detected:
                reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
                citations = extract_grounding_citations(candidate)
                if citations:
                    normal_content_str = (normal_content_str or "") + citations
                message_payload["content"] = normal_content_str
                if reasoning_str:
                    message_payload['reasoning_content'] = reasoning_str
            
            choice_item = {"index": i, "message": message_payload, "finish_reason": openai_finish_reason}
            if hasattr(candidate, 'logprobs') and candidate.logprobs is not None:
                 choice_item["logprobs"] = candidate.logprobs
            choices.append(choice_item)
            
    elif hasattr(gemini_response_obj, 'text') and gemini_response_obj.text is not None:
         content_str = gemini_response_obj.text or ""
         choices.append({"index": 0, "message": {"role": "assistant", "content": content_str}, "finish_reason": "stop"})
    else: 
         choices.append({"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"})

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(gemini_response_obj, 'usage_metadata'):
        um = gemini_response_obj.usage_metadata
        if hasattr(um, 'prompt_token_count'): usage_data['prompt_tokens'] = um.prompt_token_count
        # Gemini SDK might use candidates_token_count or total_token_count for completion.
        # Prioritize candidates_token_count if available.
        if hasattr(um, 'candidates_token_count'):
            usage_data['completion_tokens'] = um.candidates_token_count
            if hasattr(um, 'total_token_count'): # Ensure total is sum if both available
                 usage_data['total_tokens'] = um.total_token_count
            else: # Estimate total if only prompt and completion are available
                 usage_data['total_tokens'] = usage_data['prompt_tokens'] + usage_data['completion_tokens']
        elif hasattr(um, 'total_token_count'): # Fallback if only total is available
             usage_data['total_tokens'] = um.total_token_count
             if usage_data['prompt_tokens'] > 0 and usage_data['total_tokens'] > usage_data['prompt_tokens']:
                 usage_data['completion_tokens'] = usage_data['total_tokens'] - usage_data['prompt_tokens']
        else: # If only prompt_token_count is available, completion and total might remain 0 or be estimated differently
            usage_data['total_tokens'] = usage_data['prompt_tokens'] # Simplistic fallback

    return {
        "id": base_id, "object": "chat.completion", "created": response_timestamp,
        "model": request_model_str, "choices": choices,
        "usage": usage_data
    }

# Keep convert_to_openai_format as a wrapper for now if other parts of the code call it directly.
def convert_to_openai_format(gemini_response: Any, model: str) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(gemini_response, model)


def convert_chunk_to_openai(chunk: Any, model_name: str, response_id: str, candidate_index: int = 0) -> str:
    delta_payload = {}
    openai_finish_reason = None

    if hasattr(chunk, 'candidates') and chunk.candidates:
        candidate = chunk.candidates[0] # Process first candidate for streaming
        raw_gemini_finish_reason = getattr(candidate, 'finish_reason', None)
        if raw_gemini_finish_reason:
            if hasattr(raw_gemini_finish_reason, 'name'): raw_gemini_finish_reason_str = raw_gemini_finish_reason.name.upper()
            else: raw_gemini_finish_reason_str = str(raw_gemini_finish_reason).upper()

            if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
            elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
            elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
            elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"
            # Not setting a default here; None means intermediate chunk unless reason is terminal.

        function_call_detected_in_chunk = False
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call is not None: # Kilo Code: Added 'is not None' check
                    fc = part.function_call
                    raw_tool_call_id = getattr(fc, 'id', None) or f"call_{response_id}_{candidate_index}_{fc.name.replace(' ', '_')}_{int(time.time()*10000 + random.randint(0,9999))}"
                    tool_call_id = _encode_tool_call_id_with_thought_signature(raw_tool_call_id, getattr(part, 'thought_signature', None))
                    
                    current_tool_call_delta = {
                        "index": 0, 
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": fc.name}
                    }
                    if fc.args is not None: # Gemini usually sends full args.
                        current_tool_call_delta["function"]["arguments"] = json.dumps(fc.args)
                    else: # If args could be streamed (rare for Gemini FunctionCall part)
                        current_tool_call_delta["function"]["arguments"] = "" 

                    if "tool_calls" not in delta_payload:
                        delta_payload["tool_calls"] = []
                    delta_payload["tool_calls"].append(current_tool_call_delta)
                    
                    delta_payload["content"] = None 
                    function_call_detected_in_chunk = True
                    # If this chunk also has the finish_reason for tool_calls, it will be set.
                    break 

        if not function_call_detected_in_chunk:
            reasoning_text, normal_text = parse_gemini_response_for_reasoning_and_content(candidate)

            # Grounding metadata arrives on the terminal chunk; emit sources once there.
            if openai_finish_reason is not None:
                citations = extract_grounding_citations(candidate)
                if citations:
                    normal_text = (normal_text or "") + citations

            if reasoning_text: delta_payload['reasoning_content'] = reasoning_text
            if normal_text: # Only add content if it's non-empty
                delta_payload['content'] = normal_text
            elif not reasoning_text and not delta_payload.get("tool_calls") and openai_finish_reason is None:
                # If no other content and not a terminal chunk, send empty content string
                delta_payload['content'] = ""
    
    if not delta_payload and openai_finish_reason is None:
        # This case ensures that even if a chunk is completely empty (e.g. keep-alive or error scenario not caught above)
        # and it's not a terminal chunk, we still send a delta with empty content.
        delta_payload['content'] = ""

    chunk_data = {
        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name,
        "choices": [{"index": candidate_index, "delta": delta_payload, "finish_reason": openai_finish_reason}]
    }
    # Logprobs are typically not in streaming deltas for OpenAI.
    return f"data: {json.dumps(chunk_data)}\n\n"

def create_final_chunk(model: str, response_id: str, candidate_count: int = 1) -> str:
    # This function might need adjustment if the finish reason isn't always "stop"
    # For now, it's kept as is, but tool_calls might require a different final chunk structure
    # if not handled by the last delta from convert_chunk_to_openai.
    # However, OpenAI expects the last content/tool_call delta to carry the finish_reason.
    # This function is more of a safety net or for specific scenarios.
    choices = [{"index": i, "delta": {}, "finish_reason": "stop"} for i in range(candidate_count)]
    final_chunk_data = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": choices}
    return f"data: {json.dumps(final_chunk_data)}\n\n"


def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": status_code, "param": None}}


# Effort labels shared with the Anthropic path (translate_anthropic.py).
# Values are capped at 24576: gemini-2.5-flash rejects budgets above that
# ("thinking_budget is out of range; supported values are integers from 1 to
# 24576"). Per-model ceilings vary, so max/xhigh map to 24576, not 32768.
_EFFORT_TO_THINKING_BUDGET: Dict[str, int] = {
    "none": 0,
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 24576,
    "ultrathink": 24576,
    "ultra": 24576,
}

# Upper bound accepted by all current Express models.
_THINKING_BUDGET_CEILING = 24576


def _apply_thinking_params(request: OpenAIRequest, config: Dict[str, Any]) -> None:
    """Map request-body thinking params (reasoning_effort / thinking_budget)
    into the Gemini thinking_config. Explicit numeric budgets win over labels
    and are clamped to the model-supported ceiling."""
    if request.thinking_budget is not None:
        budget = max(0, min(int(request.thinking_budget), _THINKING_BUDGET_CEILING))
    elif request.reasoning_effort is not None:
        budget = _EFFORT_TO_THINKING_BUDGET.get(str(request.reasoning_effort).strip().lower())
        if budget is None:
            logger.warning("Unknown reasoning_effort %r; ignoring.", request.reasoning_effort)
            return
    else:
        return
    config.setdefault("thinking_config", {})
    config["thinking_config"]["thinking_budget"] = budget
    config["thinking_config"]["include_thoughts"] = budget > 0


def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    # Request-body thinking params -> thinking_config
    _apply_thinking_params(request, config)

    # Check for -2k or -4k suffix to add image generation capabilities
    model_name = request.model
    if model_name.endswith('-2k'):
        # Add image generation config for 2k resolution
        config["responseModalities"] = ["TEXT", "IMAGE"]
        config["imageConfig"] = {"imageSize": "2k"}
        logger.info(f"Detected -2k suffix, adding image generation config with 2k resolution")
    elif model_name.endswith('-4k'):
        # Add image generation config for 4k resolution
        config["responseModalities"] = ["TEXT", "IMAGE"]
        config["imageConfig"] = {"imageSize": "4k"}
        logger.info(f"Detected -4k suffix, adding image generation config with 4k resolution")
    
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.n is not None: config["candidate_count"] = request.n
    
    safety_threshold = "BLOCK_NONE"
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_UNSPECIFIED", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HATE", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_HARASSMENT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", threshold=safety_threshold),
            types.SafetySetting(category="HARM_CATEGORY_JAILBREAK", threshold=safety_threshold)
    ]
    # config["thinking_config"] = {"include_thoughts": True}

    # 1. Add tools (function declarations)
    # Prefer parameters_json_schema so richer JSON Schema fields from clients
    # (propertyNames, exclusiveMinimum, etc.) do not fail typed Schema validation.
    function_declarations = []
    if request.tools:
        for tool in request.tools:
            if tool.get("type") == "function":
                func_def = tool.get("function")
                if func_def and func_def.get("name"):
                    kwargs = {"name": func_def.get("name")}
                    if func_def.get("description") is not None:
                        kwargs["description"] = func_def.get("description")
                    parameters = func_def.get("parameters")
                    if isinstance(parameters, dict):
                        cleaned = {k: v for k, v in parameters.items() if k not in ("$schema", "$id", "$comment")}
                        kwargs["parameters_json_schema"] = cleaned
                    elif parameters is not None:
                        kwargs["parameters_json_schema"] = parameters
                    function_declarations.append(types.FunctionDeclaration(**kwargs))

    if function_declarations:
        config["tools"] = [types.Tool(function_declarations=function_declarations)]

    # 2. Add tool_config (based on tool_choice)
    tool_config = None
    if request.tool_choice:
        choice = request.tool_choice
        mode = None
        allowed_functions = None
        if isinstance(choice, str):
            if choice == "none":
                mode = "NONE"
            elif choice == "auto":
                mode = "AUTO"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name")
            if func_name:
                mode = "ANY"  # 'ANY' mode is used to force a specific function call
                allowed_functions = [func_name]
        
        # If a valid mode was parsed, build the tool_config
        if mode:
            config_dict = {"mode": mode}
            if allowed_functions:
                config_dict["allowed_function_names"] = allowed_functions
            tool_config = {"function_calling_config": config_dict}
    
    if tool_config:
        config["tool_config"] = tool_config
        
    return config


def is_gemini_response_valid(response: Any) -> bool:
    if response is None: return False
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip(): return True
    if hasattr(response, 'candidates') and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, 'text') and isinstance(cand.text, str) and cand.text.strip(): return True
            if hasattr(cand, 'content') and hasattr(cand.content, 'parts') and cand.content.parts:
                for part in cand.content.parts:
                    if hasattr(part, 'function_call'): return True 
                    if hasattr(part, 'text') and isinstance(getattr(part, 'text', None), str) and getattr(part, 'text', '').strip(): return True
    return False
