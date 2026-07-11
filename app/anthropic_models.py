"""Pydantic models for Anthropic Messages API requests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class AnthropicMessagesRequest(BaseModel):
    """Subset of Anthropic Messages API create params (extra fields allowed)."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Any]
    max_tokens: int = 4096
    system: Any = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Any = None
    thinking: Any = None
    container: Any = None
