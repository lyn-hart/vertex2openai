import os


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


# Default password if not set in environment
DEFAULT_PASSWORD = "123456"

# Get password from environment variable or use default
API_KEY = os.environ.get("API_KEY", DEFAULT_PASSWORD)

# HuggingFace Authentication Settings
HUGGINGFACE = os.environ.get("HUGGINGFACE", "false").lower() == "true"
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "") # Default to empty string, auth logic will verify if HF_MODE is true and this key is needed

# API Key for Vertex Express Mode
raw_vertex_keys = os.environ.get("VERTEX_EXPRESS_API_KEY")
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(',') if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

# Fake streaming settings for debugging/testing
FAKE_STREAMING_ENABLED = os.environ.get("FAKE_STREAMING", "false").lower() == "true"
FAKE_STREAMING_INTERVAL_SECONDS = float(os.environ.get("FAKE_STREAMING_INTERVAL", "1.0"))

# Upstream 429 retry settings. Retry count means retries after the first attempt.
RETRY_COUNT = _get_int_env("RETRY_COUNT", 3, minimum=0)
# Fixed interval between upstream 429 retries, in milliseconds.
RETRY_INTERVAL_MS = _get_int_env("RETRY_INTERVAL_MS", 1000, minimum=0)

# URL for the remote JSON file containing model lists
MODELS_CONFIG_URL = os.environ.get("MODELS_CONFIG_URL", "https://raw.githubusercontent.com/ldsx163/vertex2openai/refs/heads/main/vertexModels.json")

# Constant for the Vertex reasoning tag
VERTEX_REASONING_TAG = "vertex_think_tag"

# Round-robin credential selection strategy
ROUNDROBIN = os.environ.get("ROUNDROBIN", "false").lower() == "true"

# Safety score display setting
SAFETY_SCORE = os.environ.get("SAFETY_SCORE", "false").lower() == "true"
# Validation logic moved to app/auth.py

# Proxy settings
PROXY_URL = os.environ.get("PROXY_URL")
SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE")

