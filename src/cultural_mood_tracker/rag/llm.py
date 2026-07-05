from __future__ import annotations

import logging
import time
from typing import Any

from .prompting import PromptResult


DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_MAX_NEW_TOKENS = 120

LOGGER = logging.getLogger(__name__)
_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _load_model(model_name: str) -> tuple[Any, Any, str]:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    normalized_model_name = model_name.strip()
    if normalized_model_name in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_model_name]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install transformers and torch before running local LLM inference."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        load_started_at = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(normalized_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            normalized_model_name,
            dtype="auto",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load local LLM model {normalized_model_name!r}: {exc}") from exc

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = True
    model.to(device)
    model.eval()

    load_seconds = time.perf_counter() - load_started_at
    _MODEL_CACHE[normalized_model_name] = (tokenizer, model, device)
    LOGGER.info(
        "Initialized local LLM model %s on %s in %.2fs",
        normalized_model_name,
        device,
        load_seconds,
    )
    return tokenizer, model, device


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


def _format_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        LOGGER.warning("Tokenizer chat template unavailable; using simple chat fallback: %s", exc)
        system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
        formatted = f"System: {system}\nUser: {user}\nAssistant:"
    if not isinstance(formatted, str) or not formatted.strip():
        raise RuntimeError("Tokenizer produced an empty chat prompt.")
    return formatted


def generate_answer(prompt: PromptResult, model_name: str = DEFAULT_MODEL) -> str:
    if not isinstance(prompt, PromptResult):
        raise TypeError("prompt must be a PromptResult")

    messages = _validate_messages(prompt.messages)
    tokenizer, model, device = _load_model(model_name)
    formatted_prompt = _format_chat_prompt(tokenizer, messages)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install torch before running local LLM inference.") from exc

    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    input_token_count = int(encoded["input_ids"].shape[-1])
    LOGGER.info("Prompt token count: %s", input_token_count)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)

    try:
        with torch.inference_mode():
            generation_started_at = time.perf_counter()
            generated = model.generate(
                **encoded,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generation_seconds = time.perf_counter() - generation_started_at
    except Exception as exc:
        raise RuntimeError(f"Local LLM generation failed for model {model_name!r}: {exc}") from exc

    answer_tokens = generated[0][input_token_count:]
    output_token_count = int(answer_tokens.shape[-1])
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    LOGGER.info(
        "Generation complete with model %s on %s in %.2fs; output tokens=%s",
        model_name,
        device,
        generation_seconds,
        output_token_count,
    )
    return answer
