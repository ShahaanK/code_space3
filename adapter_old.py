"""
CAMEL Annotation Pipeline - Model Adapter
==========================================
Single OpenAI-compatible adapter that works with any provider:
OpenRouter, direct OpenAI, Ollama, vLLM, Together AI, etc.

All providers expose the same chat completions interface —
only the base_url and API key differ.
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


def call_model(provider_config, model_name, prompt, max_tokens=300,
               temperature=0.5, max_retries=3, retry_delay=2):
    """
    Send a prompt to a model and return the response text.

    Args:
        provider_config: Dict with base_url and api_key_env.
        model_name: Model identifier (e.g. "openai/gpt-4o-mini").
        prompt: The full prompt string to send as user message.
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

    messages = [{"role": "user", "content": prompt}]

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


async def call_model_async(provider_config, model_name, prompt, max_tokens=300,
                           temperature=0.5, max_retries=3, retry_delay=2):
    """
    Async version of call_model. Send a prompt and return response text.

    Uses AsyncOpenAI client for non-blocking concurrent requests.
    """
    import asyncio

    if not OPENAI_AVAILABLE:
        return "ERROR: openai package is not installed. Run: pip install openai"

    client = get_async_client(provider_config)
    messages = [{"role": "user", "content": prompt}]

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
