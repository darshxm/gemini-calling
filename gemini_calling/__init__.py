"""Lightweight Gemini API client (REST) with .env key management.

Features:
- Minimal dependency: uses Python stdlib (urllib) for HTTP
- Supports multiple API keys and round-robin selection
- Helper to create a .env file for keys

This initial module focuses on text generation via the Gemini REST API.
"""

from .keys import (
    create_env,
    load_keys,
    set_keys,
    load_limits_tier,
    set_limits_tier,
    configure_limits_interactive,
    load_named_keys,
)
from .client import GeminiClient, GenerateContentResult, GeminiAPIError, gemini_request
from .limits import Limits

__all__ = [
    "GeminiClient",
    "GenerateContentResult",
    "GeminiAPIError",
    "gemini_request",
    "Limits",
    "create_env",
    "load_keys",
    "load_named_keys",
    "set_keys",
    "load_limits_tier",
    "set_limits_tier",
    "configure_limits_interactive",
]
