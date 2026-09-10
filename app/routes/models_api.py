import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_vertex_express_models, refresh_models_config_cache

router = APIRouter()

@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    await refresh_models_config_cache()

    express_key_manager_instance = fastapi_request.app.state.express_key_manager

    has_express_key = express_key_manager_instance.get_total_keys() > 0

    raw_express_models = await get_vertex_express_models()

    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()
    current_time = int(time.time())

    def add_model_and_variants(base_id: str):
        """Adds a model and its variants to the list if not already present."""

        # Define all possible suffixes for a given model
        suffixes = [""] # For the base model itself
        if not base_id.startswith("gemini-2.0"):
            suffixes.extend(["-search"])
        if ("gemini-2.5-flash" in base_id or "gemini-2.5-pro" == base_id or "gemini-2.5-pro-preview-06-05" == base_id or "gemini-3-pro" in base_id) and "gemini-2.5-flash-image" not in base_id:
            suffixes.extend(["-nothinking", "-max"])
        if ("gemini-3-pro-image") in base_id:
            suffixes.extend(["-2k", "-4k"])

        for suffix in suffixes:
            model_id_with_suffix = f"{base_id}{suffix}"

            final_id = model_id_with_suffix

            if final_id not in processed_ids:
                final_model_list.append({
                    "id": final_id,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "google",
                    "permission": [],
                    "root": base_id,
                    "parent": None
                })
                processed_ids.add(final_id)

    for model_id in raw_express_models:
        add_model_and_variants(model_id)

    return {"object": "list", "data": sorted(final_model_list, key=lambda x: x['id'])}
