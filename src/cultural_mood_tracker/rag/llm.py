from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from cultural_mood_tracker.core import load_project_environment

from .prompting import PromptResult


DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2

LOGGER = logging.getLogger(__name__)
_CLIENT_CACHE: dict[str, "GroqLLM"] = {}


class GroqLLM:
    def __init__(self, *, api_key: str | None = None) -> None:
        load_project_environment(Path.cwd())
        resolved_api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not resolved_api_key:
            raise RuntimeError("Missing required environment variable: GROQ_API_KEY")

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install groq before running chatbot generation.") from exc

        self.client = Groq(api_key=resolved_api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq returned an empty response.")
        return content.strip()


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt.messages must be a non-empty list")

    validated: list[dict[str, str]] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"prompt.messages[{index}] must be a dict")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"prompt.messages[{index}] has invalid role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt.messages[{index}] must have non-empty string content")
        validated.append({"role": role, "content": content})
    return validated


def _load_client(model_name: str) -> GroqLLM:
    if model_name not in _CLIENT_CACHE:
        started_at = time.perf_counter()
        _CLIENT_CACHE[model_name] = GroqLLM()
        LOGGER.info("Initialized Groq LLM client for %s in %.2fs", model_name, time.perf_counter() - started_at)
    return _CLIENT_CACHE[model_name]


def generate_answer(
    prompt: PromptResult,
    model_name: str = DEFAULT_MODEL,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    if not isinstance(prompt, PromptResult):
        raise TypeError("prompt must be a PromptResult")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    if model_name.strip() != DEFAULT_MODEL:
        LOGGER.warning("Ignoring unsupported LLM model %s; using Groq model %s", model_name, DEFAULT_MODEL)
    model_name = DEFAULT_MODEL

    messages = _validate_messages(prompt.messages)
    client = _load_client(model_name)

    started_at = time.perf_counter()
    try:
        answer = client.generate(
            messages,
            model_name=model_name,
            max_tokens=max_new_tokens,
            temperature=DEFAULT_TEMPERATURE,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq generation failed for model {model_name!r}: {exc}") from exc

    LOGGER.info(
        "Generation complete with Groq model %s in %.2fs; max_tokens=%s",
        model_name,
        time.perf_counter() - started_at,
        max_new_tokens,
    )
    return answer.rstrip()
