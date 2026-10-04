import json
import unittest

from tests.helpers import MODEL, parse_sse, response_chunk, tool_history, tool_part
from google.genai import types
from models import OpenAIMessage, OpenAIRequest
from translate_openai import (
    OpenAIStreamState,
    _decode_tool_call_id_thought_signature,
    convert_chunk_to_openai,
    convert_to_openai_format,
    create_gemini_prompt,
    create_generation_config,
)
from translate_anthropic import create_anthropic_gemini_contents, gemini_response_to_anthropic


class OpenAITranslationTests(unittest.TestCase):
    def stream_choice(self, chunk, state=None):
        return parse_sse(convert_chunk_to_openai(chunk, MODEL, "response_1", state=state))["choices"][0]

    def test_plain_text_response(self):
        chunk = response_chunk([types.Part(text="hello")], "STOP")
        response = convert_to_openai_format(chunk, MODEL)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.stream_choice(chunk)["finish_reason"], "stop")

    def test_parallel_tools_in_one_chunk(self):
        chunk = response_chunk([tool_part(), tool_part("other", "call_2")], "STOP")
        choice = self.stream_choice(chunk)
        calls = choice["delta"]["tool_calls"]
        self.assertEqual([c["index"] for c in calls], [0, 1])
        self.assertEqual([c["id"] for c in calls], ["call_1", "call_2"])
        self.assertEqual([c["function"]["name"] for c in calls], ["lookup", "other"])
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"query": "other"})
        self.assertEqual(choice["finish_reason"], "tool_calls")
        full_choice = convert_to_openai_format(chunk, MODEL)["choices"][0]
        self.assertEqual(len(full_choice["message"]["tool_calls"]), 2)
        self.assertEqual(full_choice["finish_reason"], "tool_calls")

    def test_parallel_tools_in_separate_chunks(self):
        state = OpenAIStreamState()
        first = self.stream_choice(response_chunk([tool_part()]), state)
        second = self.stream_choice(response_chunk([tool_part("other", "call_2")]), state)
        terminal = self.stream_choice(response_chunk(finish="STOP"), state)
        self.assertEqual(first["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(second["delta"]["tool_calls"][0]["index"], 1)
        self.assertIsNone(first["finish_reason"])
        self.assertIsNone(second["finish_reason"])
        self.assertEqual(terminal["finish_reason"], "tool_calls")

    def test_tool_continuation_uses_stable_index_without_repeating_metadata(self):
        state = OpenAIStreamState()
        first_part = types.Part(function_call=types.FunctionCall(name="lookup", id="call_1"))
        first = self.stream_choice(response_chunk([first_part]), state)["delta"]["tool_calls"][0]
        continuation = self.stream_choice(response_chunk([tool_part()]), state)["delta"]["tool_calls"][0]
        self.assertEqual(first["index"], continuation["index"])
        self.assertNotIn("id", continuation)
        self.assertNotIn("name", continuation["function"])
        self.assertEqual(json.loads(first["function"]["arguments"] + continuation["function"]["arguments"]), {"query": "lookup"})

    def test_parallel_calls_without_upstream_ids_stay_distinct(self):
        choice = self.stream_choice(response_chunk([tool_part(ident=None), tool_part(ident=None)], "STOP"))
        calls = choice["delta"]["tool_calls"]
        self.assertEqual([c["index"] for c in calls], [0, 1])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_stream_state_does_not_leak_between_requests(self):
        first = self.stream_choice(response_chunk([tool_part()], "STOP"))
        second = self.stream_choice(response_chunk([types.Part(text="hello")], "STOP"))
        third = self.stream_choice(response_chunk([tool_part("other", "call_2")], "STOP"))
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertEqual(second["finish_reason"], "stop")
        self.assertEqual(third["delta"]["tool_calls"][0]["index"], 0)

    def test_tool_calls_do_not_override_truncation_or_safety_finish(self):
        for finish, expected in [("MAX_TOKENS", "length"), ("SAFETY", "content_filter")]:
            with self.subTest(finish=finish):
                state = OpenAIStreamState()
                self.stream_choice(response_chunk([tool_part()]), state)
                self.assertEqual(self.stream_choice(response_chunk(finish=finish), state)["finish_reason"], expected)

    def test_text_and_thinking_survive_alongside_tools(self):
        chunk = response_chunk([
            types.Part(text="Let me check.", thought=True),
            types.Part(text="I will look that up."),
            tool_part(),
        ], "STOP")
        message = convert_to_openai_format(chunk, MODEL)["choices"][0]["message"]
        delta = self.stream_choice(chunk)["delta"]
        for value in (message, delta):
            self.assertEqual(value["content"], "I will look that up.")
            self.assertEqual(value["reasoning_content"], "Let me check.")
            self.assertEqual(len(value["tool_calls"]), 1)

    def test_thought_signatures_survive_parallel_tool_round_trip(self):
        chunk = response_chunk([
            tool_part(signature=b"signature-one"),
            tool_part("other", "call_2", signature=b"signature-two"),
        ], "STOP")
        for calls in (
            self.stream_choice(chunk)["delta"]["tool_calls"],
            convert_to_openai_format(chunk, MODEL)["choices"][0]["message"]["tool_calls"],
        ):
            with self.subTest(calls=calls):
                self.assertEqual(_decode_tool_call_id_thought_signature(calls[0]["id"]), ("call_1", b"signature-one"))
                messages = [OpenAIMessage(role="assistant", tool_calls=calls)]
                messages.extend(OpenAIMessage(role="tool", tool_call_id=c["id"], content="[1, 2]") for c in calls)
                contents = create_gemini_prompt(messages)
                self.assertEqual([p.thought_signature for p in contents[0].parts], [b"signature-one", b"signature-two"])
                self.assertEqual([p.function_response.id for p in contents[1].parts], ["call_1", "call_2"])
                self.assertEqual([p.function_response.name for p in contents[1].parts], ["lookup", "other"])

    def test_tool_result_normalization(self):
        cases = [
            ('{"ok": true}', {"ok": True}),
            ("[1, 2]", {"result": [1, 2]}),
            ("42", {"result": 42}),
            ("false", {"result": False}),
            ("null", {"result": None}),
            ("plain text", {"result": "plain text"}),
            ("{invalid", {"result": "{invalid"}),
            ("", {"result": ""}),
            ([], {"result": []}),
            ([{"type": "text", "text": "result"}], {"result": "result"}),
            ([{"type": "text", "text": "[1, 2]"}], {"result": [1, 2]}),
            ([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}], {"result": "first\nsecond"}),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                contents = create_gemini_prompt(tool_history(content))
                result = contents[-1].parts[0].function_response
                self.assertEqual(result.response, expected)
                self.assertEqual(result.name, "lookup")
                self.assertEqual(result.id, "call_1")
                json.dumps(result.model_dump(mode="json"))

    def test_nontext_tool_blocks_remain_serializable(self):
        block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}
        contents = create_gemini_prompt(tool_history([block]))
        result = contents[-1].parts[0].function_response.response
        self.assertEqual(result["result"][0]["image_url"]["url"], block["image_url"]["url"])
        json.dumps(result)

    def test_system_and_developer_instructions_are_separate_from_dialogue(self):
        request = OpenAIRequest(model=MODEL, messages=[
            {"role": "system", "content": "Keep\n  indentation."},
            {"role": "user", "content": "Hello"},
            {"role": "developer", "content": [{"type": "text", "text": "Rule one"}, {"type": "text", "text": "Rule two"}]},
            {"role": "system", "content": None},
        ])
        before = request.model_dump()
        config = create_generation_config(request)
        contents = create_gemini_prompt(request.messages)
        self.assertEqual(config["system_instruction"], "Keep\n  indentation.\n\nRule one\nRule two")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Hello")
        self.assertEqual(request.model_dump(), before)

    def test_no_system_instruction_when_not_provided(self):
        request = OpenAIRequest(model=MODEL, messages=[{"role": "user", "content": "Hello"}])
        self.assertNotIn("system_instruction", create_generation_config(request))

    def test_anthropic_tool_result_and_mixed_response_regression(self):
        contents = create_anthropic_gemini_contents([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "[1, 2]"}]},
        ])
        self.assertEqual(contents[-1].parts[0].function_response.response, {"result": [1, 2]})
        response = gemini_response_to_anthropic(response_chunk([types.Part(text="Checking"), tool_part()], "STOP"), model=MODEL)
        self.assertEqual([b["type"] for b in response["content"]], ["text", "tool_use"])
        self.assertEqual(response["stop_reason"], "tool_use")
