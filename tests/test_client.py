import json
import ssl
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import tests  # Load the application's import path.
import aiohttp
import client


class ProjectDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache_patch = patch.object(client, "PROJECT_ID_CACHE", {})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    async def test_custom_ca_is_loaded_into_ssl_context(self):
        context = ssl.create_default_context()
        response = SimpleNamespace(status=404, text=AsyncMock(return_value=json.dumps({
            "error": {"message": "Missing projects/123456/locations/global/model"},
        })))
        session = MagicMock()
        session.__aenter__.return_value = session
        session.post.return_value.__aenter__.return_value = response
        with patch.object(client.app_config, "SSL_CERT_FILE", "/custom/ca.pem"), patch.object(client.app_config, "PROXY_URL", "http://proxy.invalid:8080"), patch("client.ssl.create_default_context", return_value=context) as factory, patch("client.aiohttp.ClientSession", return_value=session):
            self.assertEqual(await client.discover_project_id("offline-test-key"), "123456")
        factory.assert_called_once_with(cafile="/custom/ca.pem")
        self.assertIs(session.post.call_args.kwargs["ssl"], context)
        self.assertEqual(session.post.call_args.kwargs["proxy"], "http://proxy.invalid:8080")
        self.assertEqual(client.PROJECT_ID_CACHE["offline-test-key"], "123456")

    async def test_custom_ca_passes_real_aiohttp_argument_validation(self):
        class ConnectionAttempted(Exception):
            pass

        context = ssl.create_default_context()
        # aiohttp validates SSL options before consulting this connector; block
        # the connector so this test never makes an actual network connection.
        connector = AsyncMock(side_effect=ConnectionAttempted)
        with patch.object(client.app_config, "SSL_CERT_FILE", "/custom/ca.pem"), patch.object(client.app_config, "PROXY_URL", None), patch("client.ssl.create_default_context", return_value=context), patch.object(aiohttp.TCPConnector, "connect", new=connector):
            with self.assertRaises(ConnectionAttempted):
                await client.discover_project_id("offline-test-key")
        connector.assert_awaited_once()

    async def test_default_tls_keeps_verification_enabled(self):
        response = SimpleNamespace(status=404, text=AsyncMock(return_value='{"error":{"message":"projects/123456/locations/global"}}'))
        session = MagicMock()
        session.__aenter__.return_value = session
        session.post.return_value.__aenter__.return_value = response
        with patch.object(client.app_config, "SSL_CERT_FILE", None), patch("client.aiohttp.ClientSession", return_value=session):
            self.assertEqual(await client.discover_project_id("offline-test-key"), "123456")
        self.assertIs(session.post.call_args.kwargs["ssl"], True)

    async def test_cached_project_does_not_open_a_session(self):
        client.PROJECT_ID_CACHE["offline-test-key"] = "123456"
        with patch("client.aiohttp.ClientSession") as session:
            self.assertEqual(await client.discover_project_id("offline-test-key"), "123456")
        session.assert_not_called()
