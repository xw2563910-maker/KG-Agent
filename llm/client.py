from openai import OpenAI, OpenAIError

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    validate_llm_config,
)


def create_llm_client() -> OpenAI:
    validate_llm_config()

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def chat(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = create_llm_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scientific research assistant. "
                        "Answer questions accurately and concisely."
                    ),
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