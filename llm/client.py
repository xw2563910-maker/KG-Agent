from openai import OpenAI, OpenAIError

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