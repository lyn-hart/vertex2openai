import unittest

import tests  # Load the application's import path.
import client
from translate_anthropic import create_anthropic_generation_config
from translate_openai import create_generation_config
from models import OpenAIRequest
from anthropic_models import AnthropicMessagesRequest


class ThinkingConfigRoutingTests(unittest.TestCase):
    """Gemini 3.5+ rejects thinking_budget ("Thinking budget is not supported
    for this model. Use thinking level instead."), so a numeric budget must be
    converted to a thinking_level for those models."""

    def test_budget_model_support_boundaries(self):
        for model in ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.4-flash"):
            with self.subTest(model=model):
                self.assertTrue(client.supports_thinking_budget(model))
        for model in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-4-argon", "gemini-4.0-flash"):
            with self.subTest(model=model):
                self.assertFalse(client.supports_thinking_budget(model))

    def test_numeric_budget_is_converted_to_level_on_3_5_plus(self):
        self.assertEqual(
            client.resolve_thinking_config("gemini-3.7-flash", budget=10000),
            {"thinking_level": "MEDIUM", "include_thoughts": True},
        )
        self.assertEqual(
            client.resolve_thinking_config("gemini-4-argon", budget=24576),
            {"thinking_level": "HIGH", "include_thoughts": True},
        )

    def test_numeric_budget_still_sent_on_2_5_and_3_0_3_4(self):
        self.assertEqual(
            client.resolve_thinking_config("gemini-2.5-flash", budget=10000),
            {"thinking_budget": 10000, "include_thoughts": True},
        )
        self.assertEqual(
            client.resolve_thinking_config("gemini-3.1-pro-preview", budget=10000),
            {"thinking_budget": 10000, "include_thoughts": True},
        )

    def test_zero_budget_keeps_thoughts_disabled_on_level_models(self):
        # gemini-3.7-flash has no MINIMAL, so the level snaps up to LOW while
        # include_thoughts stays False.
        self.assertEqual(
            client.resolve_thinking_config("gemini-3.7-flash", budget=0),
            {"thinking_level": "LOW", "include_thoughts": False},
        )
        self.assertEqual(
            client.resolve_thinking_config("gemini-3.5-flash", budget=0),
            {"thinking_level": "MINIMAL", "include_thoughts": False},
        )

    def test_budget_level_approximation_rounds_to_nearest_anchor(self):
        self.assertEqual(client._level_for_budget(1), "MINIMAL")
        self.assertEqual(client._level_for_budget(2048), "LOW")
        self.assertEqual(client._level_for_budget(9000), "MEDIUM")
        self.assertEqual(client._level_for_budget(999999), "HIGH")

    def test_unsupported_levels_snap_up_within_family(self):
        # gemini-3-pro-preview has no MEDIUM; MINIMAL snaps to its lowest (LOW).
        self.assertEqual(
            client.resolve_thinking_config("gemini-3-pro-preview", effort="medium"),
            {"thinking_level": "HIGH", "include_thoughts": True},
        )
        self.assertEqual(
            client.resolve_thinking_config("gemini-3.1-flash-lite-image", effort="none"),
            {"thinking_level": "MINIMAL", "include_thoughts": True},
        )

    def test_effort_labels_still_map_to_budgets_on_2_5(self):
        self.assertEqual(
            client.resolve_thinking_config("gemini-2.5-flash", effort="high"),
            {"thinking_budget": 16384, "include_thoughts": True},
        )
        self.assertEqual(
            client.resolve_thinking_config("gemini-2.5-flash", effort="none"),
            {"thinking_budget": 0, "include_thoughts": False},
        )

    def test_openai_request_budget_reaches_config_as_level(self):
        request = OpenAIRequest(
            model="gemini-3.7-flash",
            messages=[{"role": "user", "content": "Hello"}],
            thinking_budget=10000,
        )
        config = create_generation_config(request)
        self.assertEqual(config["thinking_config"]["thinking_level"], "MEDIUM")
        self.assertNotIn("thinking_budget", config["thinking_config"])

    def test_anthropic_explicit_budget_reaches_config_as_level(self):
        request = AnthropicMessagesRequest(
            model="gemini-3.7-flash",
            messages=[{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        config = create_anthropic_generation_config(request, base_model_name="gemini-3.7-flash")
        self.assertEqual(config["thinking_config"]["thinking_level"], "MEDIUM")
        self.assertNotIn("thinking_budget", config["thinking_config"])

    def test_anthropic_budget_kept_on_2_5(self):
        request = AnthropicMessagesRequest(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        config = create_anthropic_generation_config(request, base_model_name="gemini-2.5-flash")
        self.assertEqual(config["thinking_config"]["thinking_budget"], 10000)
        self.assertNotIn("thinking_level", config["thinking_config"])
