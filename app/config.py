import os

from logging_setup import mask_key


def _get_int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        print(f"WARNING: Invalid integer for {name}: {raw_value!r}. Using default {default}.")
        return default
    if minimum is not None and parsed_value < minimum:
        print(f"WARNING: {name} must be >= {minimum}. Using default {default}.")
        return default
    return parsed_value


def _get_float_env(name: str, default: float, minimum: float | None = None) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed_value = float(raw_value)
    except ValueError:
        print(f"WARNING: Invalid float for {name}: {raw_value!r}. Using default {default}.")
        return default
    if minimum is not None and parsed_value < minimum:
        print(f"WARNING: {name} must be >= {minimum}. Using default {default}.")
        return default
    return parsed_value


# API key clients must present (Authorization: Bearer or x-api-key). Required, no default.
PROXY_API_KEY = os.environ.get("PROXY_API_KEY")

# Comma-separated Vertex Express API keys. Required.
raw_express_keys = os.environ.get("EXPRESS_API_KEYS")
if raw_express_keys:
    EXPRESS_API_KEYS = [key.strip() for key in raw_express_keys.split(',') if key.strip()]
else:
    EXPRESS_API_KEYS = []

# Express key selection strategy: "random" (default) or "roundrobin".
KEY_ROTATION_RAW = os.environ.get("KEY_ROTATION", "random").strip().lower()
if KEY_ROTATION_RAW not in ("random", "roundrobin"):
    print(f"WARNING: Invalid KEY_ROTATION {KEY_ROTATION_RAW!r}. Using 'random'.")
    KEY_ROTATION = "random"
else:
    KEY_ROTATION = KEY_ROTATION_RAW

# Logging verbosity (DEBUG, INFO, WARNING, ERROR).
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = _get_float_env("FAKE_STREAMING_INTERVAL", 1.0, minimum=0.0)

# Upstream 429 retry settings. Retry count means retries after the first attempt.
RETRY_COUNT = _get_int_env("RETRY_COUNT", 3, minimum=0)
# Fixed interval between upstream 429 retries, in milliseconds.
RETRY_INTERVAL_MS = _get_int_env("RETRY_INTERVAL_MS", 1000, minimum=0)

# URL for the remote JSON file containing model lists
MODELS_CONFIG_URL = os.environ.get("MODELS_CONFIG_URL", "https://raw.githubusercontent.com/lyn-hart/vertex2openai/refs/heads/main/vertexModels.json")

# Constant for the Vertex reasoning tag
VERTEX_REASONING_TAG = "vertex_think_tag"

# Proxy settings
PROXY_URL = os.environ.get("PROXY_URL")
SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE")


def validate_config() -> None:
    """Fail fast at startup when required configuration is missing."""
    missing = []
    if not PROXY_API_KEY:
        missing.append("PROXY_API_KEY")
    if not EXPRESS_API_KEYS:
        missing.append("EXPRESS_API_KEYS")
    if missing:
        raise RuntimeError(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Set these environment variables before starting the server."
        )


def config_summary() -> str:
    """Human-readable startup summary; keys are masked."""
    masked_keys = ", ".join(mask_key(k) for k in EXPRESS_API_KEYS) or "<none>"
    return (
        f"express_keys={len(EXPRESS_API_KEYS)} ({masked_keys}), "
        f"key_rotation={KEY_ROTATION}, "
        f"retry_count={RETRY_COUNT}, retry_interval_ms={RETRY_INTERVAL_MS}"
    )
