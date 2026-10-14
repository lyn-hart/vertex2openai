"""Endpoint fallback behaviour for global-only models (e.g. gemini-4-argon).

The Express SDK sends requests to a regional endpoint (us-central1) when the
key carries no location, but models such as gemini-4-argon are only served
from the project-scoped global endpoint. The adapter therefore retries once
against that endpoint and then remembers the mismatch, so later calls do not
pay for a guaranteed 404 on every request.
"""

import unittest
from unittest.mock import AsyncMock, patch

import tests  # Load the application's import path.
import client
from google.genai import types

MODEL = "gemini-4-argon"
KEY = "express-test-key"


class UpstreamError(Exception):
    def __init__(self, code: int):
        super().__init__(f"{code} upstream error")
        self.status_code = code


class FakeKeyManager:
    def __init__(self, keys=(KEY,)):
        self.keys = list(keys)

    def get_total_keys(self):
        return len(self.keys)

    def get_express_api_key(self):
        if not self.keys:
            return None
        return (0, self.keys[0])


def _stream(chunks=(), error=None, error_after_chunks=False):
    """Stand-in for the SDK stream handle.

    `error` is raised before the first chunk unless `error_after_chunks`,
    which models an upstream stream that breaks mid-response.
    """

    class FakeStream:
        def __await__(self):
            async def _result():
                return self
            return _result().__await__()

        async def __aiter__(self):
            if error is not None and not error_after_chunks:
                raise error
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error

    return FakeStream()


class FakeModelCalls:
    def __init__(self, content_error=None, stream_error=None, chunks=(), error_after_chunks=False):
        self.content_error = content_error
        self.stream_error = stream_error
        self.chunks = chunks
        self.error_after_chunks = error_after_chunks

    async def generate_content(self, **kwargs):
        if self.content_error is not None:
            raise self.content_error
        return "ok"

    async def generate_content_stream(self, **kwargs):
        return _stream(
            chunks=self.chunks,
            error=self.stream_error,
            error_after_chunks=self.error_after_chunks,
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.aio = type("Aio", (), {"models": FakeModelCalls(**kwargs)})()


class EndpointMissClassificationTests(unittest.TestCase):
    def test_only_404_triggers_the_fallback(self):
        self.assertTrue(client._is_default_endpoint_miss(UpstreamError(404)))
        # A 400 is a request-body rejection; switching endpoints cannot fix it.
        self.assertFalse(client._is_default_endpoint_miss(UpstreamError(400)))
        self.assertFalse(client._is_default_endpoint_miss(UpstreamError(429)))

    def test_project_scope_memory(self):
        self.assertFalse(client.is_project_scoped_model(KEY, MODEL))
        client.mark_project_scoped_model(KEY, MODEL)
        self.assertTrue(client.is_project_scoped_model(KEY, MODEL))
        self.assertFalse(client.is_project_scoped_model(KEY, "gemini-2.5-flash"))
        self.assertFalse(client.is_project_scoped_model("other-key", MODEL))


class ProjectClientConstructionTests(unittest.TestCase):
    def test_base_url_carries_a_single_version_segment(self):
        built = {}
        internal_options = types.HttpOptions()

        def fake_genai_client(**kwargs):
            built["kwargs"] = kwargs
            fake = FakeClient()
            fake._api_client = type("Api", (), {"_http_options": internal_options})()
            return fake

        with patch("client.genai.Client", side_effect=fake_genai_client):
            client._build_project_client(KEY, "829475126197")

        http_options = built["kwargs"]["http_options"]
        self.assertEqual(
            http_options.base_url,
            "https://aiplatform.googleapis.com/v1/projects/829475126197/locations/global",
        )
        # The /v1 in the base URL must be the only version segment: the SDK
        # appends http_options.api_version to every path it builds.
        self.assertEqual(http_options.api_version, "")
        self.assertIsNone(http_options.api_version or None)
        # ...and the same holds for the options object the SDK keeps internally.
        self.assertIsNone(internal_options.api_version)
        self.assertEqual(built["kwargs"]["api_key"], KEY)


class ExpressClientSelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = patch.multiple(
            client,
            PROJECT_ID_CACHE={},
            _PROJECT_SCOPED_MODELS=set(),
        )
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_unknown_model_uses_the_default_endpoint(self):
        with patch("client._build_plain_client", return_value="plain") as plain, patch("client._build_project_client") as project:
            self.assertEqual(client._build_express_client(KEY, MODEL), "plain")
        plain.assert_called_once_with(KEY)
        project.assert_not_called()

    def test_known_model_uses_the_cached_project_endpoint(self):
        client.PROJECT_ID_CACHE[KEY] = "829475126197"
        client.mark_project_scoped_model(KEY, MODEL)
        with patch("client._build_plain_client") as plain, patch("client._build_project_client", return_value="project") as project:
            self.assertEqual(client._build_express_client(KEY, MODEL), "project")
        project.assert_called_once_with(KEY, "829475126197")
        plain.assert_not_called()

    def test_known_model_without_cached_project_falls_back_to_default(self):
        client.mark_project_scoped_model(KEY, MODEL)
        with patch("client._build_plain_client", return_value="plain"), patch("client._build_project_client") as project:
            self.assertEqual(client._build_express_client(KEY, MODEL), "plain")
        project.assert_not_called()


class UpstreamFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = patch.multiple(
            client,
            PROJECT_ID_CACHE={},
            _PROJECT_SCOPED_MODELS=set(),
        )
        self.state.start()
        self.addCleanup(self.state.stop)
        retry_patch = patch.object(client.app_config, "RETRY_COUNT", 0)
        retry_patch.start()
        self.addCleanup(retry_patch.stop)
        self.manager = FakeKeyManager()
        self.plain_calls = []
        self.project_calls = []

    def _patch_clients(self, plain_error, project_error=None):
        async def discover(api_key):
            client.PROJECT_ID_CACHE[api_key] = "829475126197"
            return "829475126197"

        def build_plain(api_key):
            self.plain_calls.append(api_key)
            return FakeClient(content_error=plain_error, stream_error=plain_error)

        def build_project(api_key, project_id):
            self.project_calls.append((api_key, project_id))
            return FakeClient(content_error=project_error, stream_error=project_error)

        return patch.multiple(
            client,
            _build_plain_client=build_plain,
            _build_project_client=build_project,
            discover_project_id=AsyncMock(side_effect=discover),
        )

    async def test_generate_falls_back_once_then_remembers(self):
        with self._patch_clients(UpstreamError(404)):
            self.assertEqual(
                await client.generate_gemini_content(self.manager, MODEL, [], {}), "ok"
            )
            self.assertEqual(self.plain_calls, [KEY])
            self.assertEqual(self.project_calls, [(KEY, "829475126197")])
            self.assertTrue(client.is_project_scoped_model(KEY, MODEL))

            # Second request skips the endpoint that already 404'd.
            self.assertEqual(
                await client.generate_gemini_content(self.manager, MODEL, [], {}), "ok"
            )
        self.assertEqual(self.plain_calls, [KEY])
        self.assertEqual(self.project_calls, [(KEY, "829475126197"), (KEY, "829475126197")])

    async def test_generate_does_not_fall_back_on_400(self):
        with self._patch_clients(UpstreamError(400)):
            with self.assertRaises(UpstreamError):
                await client.generate_gemini_content(self.manager, MODEL, [], {})
        self.assertEqual(self.plain_calls, [KEY])
        self.assertEqual(self.project_calls, [])
        self.assertFalse(client.is_project_scoped_model(KEY, MODEL))

    async def test_stream_falls_back_once_then_remembers(self):
        with self._patch_clients(UpstreamError(404)):
            chunks = [chunk async for chunk in client.stream_gemini_content(self.manager, MODEL, [], {})]
            self.assertEqual(chunks, [])
            self.assertEqual(self.plain_calls, [KEY])
            self.assertEqual(self.project_calls, [(KEY, "829475126197")])
            self.assertTrue(client.is_project_scoped_model(KEY, MODEL))

            chunks = [chunk async for chunk in client.stream_gemini_content(self.manager, MODEL, [], {})]
            self.assertEqual(chunks, [])
        self.assertEqual(self.plain_calls, [KEY])
        self.assertEqual(self.project_calls, [(KEY, "829475126197"), (KEY, "829475126197")])

    async def test_stream_does_not_fall_back_on_400(self):
        with self._patch_clients(UpstreamError(400)):
            with self.assertRaises(UpstreamError):
                async for _ in client.stream_gemini_content(self.manager, MODEL, [], {}):
                    pass
        self.assertEqual(self.plain_calls, [KEY])
        self.assertEqual(self.project_calls, [])
        self.assertFalse(client.is_project_scoped_model(KEY, MODEL))

    async def test_stream_after_first_chunk_is_not_retried(self):
        def build_plain(api_key):
            self.plain_calls.append(api_key)
            return FakeClient(stream_error=UpstreamError(404), chunks=("chunk",), error_after_chunks=True)

        with patch.multiple(
            client,
            _build_plain_client=build_plain,
            _build_project_client=lambda *args: self.fail("must not fall back after yielding"),
            discover_project_id=AsyncMock(return_value="829475126197"),
        ):
            received = []
            with self.assertRaises(UpstreamError):
                async for chunk in client.stream_gemini_content(self.manager, MODEL, [], {}):
                    received.append(chunk)
            self.assertEqual(received, ["chunk"])
            self.assertEqual(self.plain_calls, [KEY])
            self.assertFalse(client.is_project_scoped_model(KEY, MODEL))

    async def test_resolve_client_returns_project_client_for_known_model(self):
        client.PROJECT_ID_CACHE[KEY] = "829475126197"
        client.mark_project_scoped_model(KEY, MODEL)
        with patch("client._build_plain_client") as plain, patch("client._build_project_client", return_value="project") as project:
            resolved, api_key = await client.resolve_gemini_client(
                model=MODEL,
                base_model_name=MODEL,
                express_key_manager=self.manager,
            )
        self.assertEqual((resolved, api_key), ("project", KEY))
        project.assert_called_once_with(KEY, "829475126197")
        plain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
