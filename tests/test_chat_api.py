import unittest
from unittest.mock import AsyncMock, patch

from tests.helpers import MODEL, response_chunk, tool_history
from google.genai import types
from fastapi.testclient import TestClient
import main


class ChatAPITests(unittest.TestCase):
    def setUp(self):
        # Not entering the lifespan avoids the remote model-catalog warm-up.
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        auth_patch = patch("auth.PROXY_API_KEY", "offline-proxy-key")
        auth_patch.start()
        self.addCleanup(auth_patch.stop)
        self.headers = {"Authorization": "Bearer offline-proxy-key"}

    def test_valid_tool_results_reach_upstream_instead_of_returning_500(self):
        for content in ("[1, 2]", [{"type": "text", "text": "result"}]):
            with self.subTest(content=content):
                upstream = AsyncMock(return_value=response_chunk([types.Part(text="done")], "STOP"))
                with patch("streaming.generate_gemini_content", new=upstream):
                    response = self.client.post("/v1/chat/completions", headers=self.headers, json={
                        "model": MODEL, "messages": [m.model_dump() for m in tool_history(content)],
                    })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "done")
                upstream.assert_awaited_once()
                self.assertIsInstance(upstream.await_args.args[2][-1].parts[0].function_response.response, dict)

    def test_system_instruction_is_forwarded_in_config_only(self):
        upstream = AsyncMock(return_value=response_chunk([types.Part(text="done")], "STOP"))
        with patch("streaming.generate_gemini_content", new=upstream):
            response = self.client.post("/v1/chat/completions", headers=self.headers, json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "System policy"},
                    {"role": "developer", "content": "Application policy"},
                    {"role": "user", "content": "Hello"},
                ],
            })
        self.assertEqual(response.status_code, 200, response.text)
        contents, config = upstream.await_args.args[2:4]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].parts[0].text, "Hello")
        self.assertEqual(config["system_instruction"], "System policy\n\nApplication policy")

    def test_auth_rejects_missing_or_wrong_key_before_upstream_call(self):
        upstream = AsyncMock()
        with patch("streaming.generate_gemini_content", new=upstream):
            for headers in ({}, {"Authorization": "Bearer wrong-key"}):
                response = self.client.post("/v1/chat/completions", headers=headers, json={
                    "model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
                })
                self.assertEqual(response.status_code, 401)
        upstream.assert_not_awaited()
