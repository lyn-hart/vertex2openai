from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from logging_setup import configure_logging
from config import config_summary, validate_config

# Local module imports
from auth import get_api_key # Potentially for root endpoint
from express_key_manager import ExpressKeyManager
from catalog import refresh_models_config_cache

# Routers
from routes import models_api
from routes import chat_api
from routes import messages_api

import logging
logger = logging.getLogger("main")

configure_logging()

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
    # Fail fast when required configuration is missing.
    validate_config()

    logger.info("Vertex Express configuration: %s", config_summary())

    # Pre-warm the model configuration cache
    logger.info("Attempting to pre-warm model configuration cache during startup...")
    models_loaded_successfully = await refresh_models_config_cache()
    if models_loaded_successfully:
        logger.info("Model configuration cache pre-warmed successfully.")
    else:
        logger.warning("Failed to pre-warm model configuration cache during startup. It will be loaded lazily on first request.")

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
