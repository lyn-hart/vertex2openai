import json

from google.genai import types
from models import OpenAIMessage

MODEL = "gemini-2.5-flash"


def response_chunk(parts=(), finish=None):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)),
        finish_reason=finish,
    )])


def tool_part(name="lookup", ident="call_1", args=None, signature=None):
    return types.Part(
        function_call=types.FunctionCall(
            name=name, id=ident, args={"query": name} if args is None else args,
        ),
        thought_signature=signature,
    )


def tool_history(content):
    return [
        OpenAIMessage(role="assistant", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }]),
        OpenAIMessage(role="tool", tool_call_id="call_1", content=content),
    ]


def parse_sse(value):
    return json.loads(value.removeprefix("data: ").strip())
