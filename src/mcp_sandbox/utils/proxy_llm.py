"""Lightweight wrapper for calling the Pluralsight/other OpenAI-compatible proxy.

This module centralizes environment handling and the HTTP logic.  Other demos
can import :func:`call_llm` and avoid duplicating the proxy-specific parsing.

Usage::

    from proxy_llm import call_llm
    text = call_llm("Hello world")

Configuration comes from the usual .env file.  Supported variables::

    PROXY_API_KEY    - bearer token for the proxy
    PROXY_BASE_URL   - base URL (no trailing slash) to which "/chat/completions"
                       will be appended
    PROXY_MODEL      - optional model name; defaults to ``chatgpt-4o``
    PROXY_LOG_LEVEL  - logging level for this module; defaults to ``DEBUG``
                       (e.g. ``INFO``, ``WARNING``, ``ERROR``)

The module raises on HTTP errors or if the proxy returns an unexpected format.
"""

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp_sandbox.services.logger import LoggerFactory

# load early so environment variables are available at import time
load_dotenv()

logger = LoggerFactory(handler_type="Stream", verbose=True).create_module_logger(
    module_name=__name__,
    level=os.getenv("PROXY_LOG_LEVEL"),
    force_reconfigure=False,
)

API_KEY = os.getenv("PROXY_API_KEY")
BASE_URL = os.getenv("PROXY_BASE_URL", "").rstrip("/")
MODEL = os.getenv("PROXY_MODEL", "chatgpt-4o")


def call_llm(prompt: str) -> str:
    """Send ``prompt`` to the proxy and return the assistant text.

    This proxy only supports the legacy ``{"prompt": "..."}`` format;
    tool descriptions and conversation history are embedded in the prompt
    string by the caller (see ``ProxyChatModel._build_prompt``).

    Raises:
        RuntimeError: If credentials are missing.
        httpx.HTTPStatusError: On non-2xx HTTP responses.
    """
    if not (API_KEY and BASE_URL):
        raise RuntimeError(
            "PROXY_API_KEY and PROXY_BASE_URL must be set in the environment"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload: dict[str, Any] = {"prompt": prompt}

    logger.debug("calling proxy %s — prompt length %s chars", BASE_URL, len(prompt))
    with httpx.Client() as client:
        resp = client.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers
        )
        try:
            resp.raise_for_status()
        except Exception as exc:
            logger.error("request failed: %s", exc)
            raise
        data = resp.json()
    logger.debug("proxy responded with %s", data)

    if choices := data.get("choices"):
        text = choices[0]["message"]["content"]
        logger.debug("extracted text from choices")
        return text
    if "response" in data:
        logger.debug("extracted text from response field")
        return data["response"]
    logger.error("unexpected response format: %s", data)
    raise RuntimeError(f"unexpected response format from proxy: {data}")
