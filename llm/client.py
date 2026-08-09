from openai import OpenAI, OpenAIError
import json
from typing import Any

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    validate_llm_config,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific research assistant. "
    "Answer questions accurately and concisely."
)


def create_llm_client() -> OpenAI:
    validate_llm_config()

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def chat(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = create_llm_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response.choices[0].message.content or ""

    except OpenAIError as exc:
        raise RuntimeError(
            f"LLM request failed: {exc}"
        ) from exc


def chat_json(
    prompt: str,
    system_prompt: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = create_llm_client()

    last_error = None

    for attempt in range(1, max_retries + 2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
                max_tokens=500,
            )

        except OpenAIError as exc:
            raise RuntimeError(
                f"LLM request failed: {exc}"
            ) from exc

        content = response.choices[0].message.content

        if not content or not content.strip():
            last_error = (
                "LLM returned empty JSON content."
            )

            if attempt <= max_retries:
                print(
                    f"[LLM] Empty JSON response. "
                    f"Retrying ({attempt}/{max_retries})..."
                )
                continue

            break

        try:
            data = json.loads(content)

        except json.JSONDecodeError as exc:
            last_error = (
                f"Failed to parse LLM JSON response: {content}"
            )

            if attempt <= max_retries:
                print(
                    f"[LLM] Invalid JSON response. "
                    f"Retrying ({attempt}/{max_retries})..."
                )
                continue

            raise RuntimeError(
                last_error
            ) from exc

        if not isinstance(data, dict):
            last_error = (
                "LLM JSON response must be an object."
            )

            if attempt <= max_retries:
                print(
                    f"[LLM] Unexpected JSON structure. "
                    f"Retrying ({attempt}/{max_retries})..."
                )
                continue

            break

        return data

    raise RuntimeError(
        last_error
        or "LLM failed to return valid JSON."
    )