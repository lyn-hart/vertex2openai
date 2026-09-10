from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local module imports
from auth import get_api_key # Potentially for root endpoint
from express_key_manager import ExpressKeyManager
from model_loader import refresh_models_config_cache
import config as app_config

# Routers
from routes import models_api
from routes import chat_api
from routes import messages_api

app = FastAPI(title="OpenAI to Gemini Adapter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

express_key_manager = ExpressKeyManager()
app.state.express_key_manager = express_key_manager # Store express key manager on app state

# Include API routers
app.include_router(models_api.router)
app.include_router(chat_api.router)
app.include_router(messages_api.router)

@app.on_event("startup")
async def startup_event():
    # Check Express API keys availability
    express_keys_count = express_key_manager.get_total_keys()

    print(f"INFO: Express API keys loaded: {express_keys_count}")
    print(
        "INFO: Upstream 429 retry config: "
        f"count={app_config.RETRY_COUNT}, "
        f"fixed_interval_ms={app_config.RETRY_INTERVAL_MS}",
        flush=True
    )

    if express_keys_count > 0:
        print("INFO: Vertex Express authentication initialization completed successfully.")
    else:
        print("ERROR: No Express API keys configured. API calls will fail.")

    # Pre-warm the model configuration cache
    print("INFO: Attempting to pre-warm model configuration cache during startup...")
    models_loaded_successfully = await refresh_models_config_cache()
    if models_loaded_successfully:
        print("INFO: Model configuration cache pre-warmed successfully.")
    else:
        print("WARNING: Failed to pre-warm model configuration cache during startup. It will be loaded lazily on first request.")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "OpenAI to Gemini Adapter is running.",
        "endpoints": [
            "GET /v1/models",
            "POST /v1/chat/completions",
            "POST /v1/messages",
            "POST /v1/messages/count_tokens",
        ],
    }
