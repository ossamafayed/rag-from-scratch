"""The generator half of RAG: a local LLM served by Ollama.

Ollama runs the model as a background HTTP service on your machine.
`OllamaLLM` is just a LangChain-shaped client for it, so every chain in this
project would work identically against a cloud model - only this file changes.
"""

import urllib.error
import urllib.request
import json

from langchain_ollama import OllamaLLM

from rag import config


def check_ollama() -> list[str]:
    """Confirm the Ollama service is up, and return the models it has pulled.

    Raises with an actionable message rather than letting a confusing
    connection error surface five layers deep inside a chain.
    """
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5
        ) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL} ({exc}).\n"
            "Start it with:  ollama serve"
        ) from exc

    models = [m["name"] for m in payload.get("models", [])]
    if config.LLM_MODEL not in models:
        raise RuntimeError(
            f"Model '{config.LLM_MODEL}' is not pulled.\n"
            f"Available: {models or '(none)'}\n"
            f"Pull it with:  ollama pull {config.LLM_MODEL}"
        )
    return models


def get_llm() -> OllamaLLM:
    """Build the LLM client using the settings from config.py."""
    return OllamaLLM(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_predict=config.NUM_PREDICT,
        num_ctx=config.NUM_CTX,
        reasoning=config.THINKING,
    )
