"""
CAMEL Annotation Pipeline - Model Adapter
==========================================
Single OpenAI-compatible adapter that works with any provider:
OpenRouter, direct OpenAI, Ollama, vLLM, Together AI, etc.

All providers expose the same chat completions interface —
only the base_url and API key differ.

SYSTEM/USER SPLIT (v2 — March 2026):
  call_model() and call_model_async() now accept an optional
  system_prompt parameter. When provided, the request is sent as:
    [{"role": "system", "content": system_prompt},
     {"role": "user",   "content": prompt}]

  This enables vLLM's automatic prefix caching: all requests that
  share the same system message reuse cached KV states, so only
  the user message (the annotated text) needs fresh computation.

  When system_prompt is None (default), behavior is unchanged —
  everything goes in a single user message for backward compat.
"""

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; fall back to system env vars

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Cache clients per provider to avoid re-creating connections
_sync_clients = {}
_async_clients = {}


def _resolve_api_key(provider_config):
    """Resolve API key from environment variable."""
    api_key_env = provider_config.get("api_key_env")
    if not api_key_env:
        return "not-needed"

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable '{api_key_env}' is not set. "
            f"Run: export {api_key_env}='your-key-here'"
        )
    return api_key


def get_client(provider_config):
    """Get or create a sync OpenAI-compatible client."""
    base_url = provider_config["base_url"]

    if base_url not in _sync_clients:
        _sync_clients[base_url] = OpenAI(
            base_url=base_url,
            api_key=_resolve_api_key(provider_config),
        )

    return _sync_clients[base_url]


def get_async_client(provider_config):
    """Get or create an async OpenAI-compatible client."""
    base_url = provider_config["base_url"]

    if base_url not in _async_clients:
        _async_clients[base_url] = AsyncOpenAI(
            base_url=base_url,
            api_key=_resolve_api_key(provider_config),
        )

    return _async_clients[base_url]


def _build_messages(prompt, system_prompt=None):
    """
    Build the messages list for the chat completions API.

    Args:
        prompt: The user message content (or full prompt if no split).
        system_prompt: Optional system message. When provided, enables
                       vLLM prefix caching on the system content.

    Returns:
        List of message dicts for the OpenAI chat API.
    """
    if system_prompt is not None:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        return [{"role": "user", "content": prompt}]


def call_model(provider_config, model_name, prompt, system_prompt=None,
               max_tokens=300, temperature=0.5, max_retries=3, retry_delay=2):
    """
    Send a prompt to a model and return the response text.

    Args:
        provider_config: Dict with base_url and api_key_env.
        model_name: Model identifier (e.g. "openai/gpt-4o-mini").
        prompt: The user message string (or full prompt if system_prompt=None).
        system_prompt: Optional system message for prefix caching.
                       When provided, the API call uses two messages:
                       system (static, cached) + user (variable per text).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        max_retries: Number of retries on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        Response text string, or an error string prefixed with "ERROR:".
    """
    if not OPENAI_AVAILABLE:
        return "ERROR: openai package is not installed. Run: pip install openai"

    client = get_client(provider_config)
    messages = _build_messages(prompt, system_prompt)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)  # exponential-ish backoff
            else:
                return f"ERROR: {str(e)}"


async def call_model_async(provider_config, model_name, prompt,
                           system_prompt=None, max_tokens=300,
                           temperature=0.5, max_retries=3, retry_delay=2):
    """
    Async version of call_model. Send a prompt and return response text.

    Uses AsyncOpenAI client for non-blocking concurrent requests.

    Args:
        provider_config: Dict with base_url and api_key_env.
        model_name: Model identifier.
        prompt: The user message string (or full prompt if system_prompt=None).
        system_prompt: Optional system message for prefix caching.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.
        max_retries: Number of retries on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        Response text string, or an error string prefixed with "ERROR:".
    """
    import asyncio

    if not OPENAI_AVAILABLE:
        return "ERROR: openai package is not installed. Run: pip install openai"

    client = get_async_client(provider_config)
    messages = _build_messages(prompt, system_prompt)

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * attempt)
            else:
                return f"ERROR: {str(e)}"
