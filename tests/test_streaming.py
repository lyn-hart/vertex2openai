import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from tests.helpers import MODEL, parse_sse, response_chunk, tool_part
from google.genai import types
from models import OpenAIRequest
import streaming
from translate_openai import convert_to_openai_format, create_gemini_prompt


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return OpenAIRequest(model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True)

    async def real_stream(self, chunks):
        async def upstream(*args, **kwargs):
            for chunk in chunks:
                yield chunk
        request = self.request()
        with patch("streaming.stream_gemini_content", new=upstream), patch.object(streaming.app_config, "FAKE_STREAMING_ENABLED", False):
            response = await streaming.execute_gemini_call(None, MODEL, create_gemini_prompt, {}, request)
            events = [event async for event in response.body_iterator]
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        return [parse_sse(event) for event in events[:-1]]

    async def test_real_stream_wires_request_state_into_all_chunks(self):
        events = await self.real_stream([
            response_chunk([tool_part()]),
            response_chunk([tool_part("other", "call_2")]),
            response_chunk(finish="STOP"),
        ])
        calls = [call for event in events for call in event["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual([c["index"] for c in calls], [0, 1])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        text_events = await self.real_stream([response_chunk([types.Part(text="hello")], "STOP")])
        self.assertEqual(text_events[-1]["choices"][0]["finish_reason"], "stop")

    async def test_synthetic_terminal_chunk_uses_tool_call_state(self):
        events = await self.real_stream([response_chunk([tool_part()])])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        events = await self.real_stream([response_chunk([types.Part(text="hello")])])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    async def test_fake_stream_preserves_text_thinking_and_tool_ids(self):
        upstream_response = response_chunk([
            types.Part(text="Thinking", thought=True),
            types.Part(text="Checking"),
            tool_part(signature=b"signature-one"),
            tool_part("other", "call_2", signature=b"signature-two"),
        ], "STOP")
        with patch("streaming.generate_gemini_content", new=AsyncMock(return_value=upstream_response)), patch.object(streaming.app_config, "FAKE_STREAMING_INTERVAL_SECONDS", 0):
            events = [event async for event in streaming.gemini_fake_stream_generator(None, MODEL, [], {}, self.request())]
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        choices = [parse_sse(event)["choices"][0] for event in events[:-1]]
        self.assertEqual("".join(c["delta"].get("content") or "" for c in choices), "Checking")
        self.assertEqual("".join(c["delta"].get("reasoning_content") or "" for c in choices), "Thinking")
        # OpenAI clients concatenate string fields across deltas, including IDs.
        assembled = {}
        for choice in choices:
            for call in choice["delta"].get("tool_calls", []):
                result = assembled.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                result["id"] += call.get("id", "")
                result["name"] += call["function"].get("name", "")
                result["arguments"] += call["function"].get("arguments", "")
        expected_calls = convert_to_openai_format(upstream_response, MODEL)["choices"][0]["message"]["tool_calls"]
        for index, expected in enumerate(expected_calls):
            self.assertEqual(assembled[index]["id"], expected["id"])
            self.assertEqual(assembled[index]["name"], expected["function"]["name"])
            self.assertEqual(json.loads(assembled[index]["arguments"]), json.loads(expected["function"]["arguments"]))
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")

    async def test_closing_fake_stream_cancels_and_awaits_upstream(self):
        started = asyncio.Event()
        finished = asyncio.Event()
        tasks = []

        async def upstream(*args, **kwargs):
            tasks.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        with patch("streaming.generate_gemini_content", new=upstream), patch.object(streaming.app_config, "FAKE_STREAMING_INTERVAL_SECONDS", 60):
            gen = streaming.gemini_fake_stream_generator(None, MODEL, [], {}, self.request())
            try:
                await anext(gen)
                await asyncio.wait_for(started.wait(), timeout=1)
                await gen.aclose()
                self.assertTrue(tasks[0].cancelled())
                self.assertTrue(finished.is_set())
            finally:
                await gen.aclose()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_cancelling_fake_stream_during_heartbeat_or_upstream_wait(self):
        for interval in (60, 0):
            with self.subTest(interval=interval):
                started = asyncio.Event()
                finished = asyncio.Event()
                tasks = []

                async def upstream(*args, **kwargs):
                    tasks.append(asyncio.current_task())
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        finished.set()

                async def consume():
                    async for _ in streaming.gemini_fake_stream_generator(None, MODEL, [], {}, self.request()):
                        pass

                with patch("streaming.generate_gemini_content", new=upstream), patch.object(streaming.app_config, "FAKE_STREAMING_INTERVAL_SECONDS", interval):
                    consumer = asyncio.create_task(consume())
                    try:
                        await asyncio.wait_for(started.wait(), timeout=1)
                        consumer.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await consumer
                        self.assertTrue(tasks[0].cancelled())
                        self.assertTrue(finished.is_set())
                    finally:
                        consumer.cancel()
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(consumer, *tasks, return_exceptions=True)

    async def test_fake_stream_upstream_error_is_reported_and_retrieved(self):
        with patch("streaming.generate_gemini_content", new=AsyncMock(side_effect=RuntimeError("upstream failed"))), patch.object(streaming.app_config, "FAKE_STREAMING_INTERVAL_SECONDS", 0):
            events = [event async for event in streaming.gemini_fake_stream_generator(None, MODEL, [], {}, self.request())]
        self.assertEqual(parse_sse(events[0])["error"]["message"], "upstream failed")
        self.assertEqual(events[-1], "data: [DONE]\n\n")
