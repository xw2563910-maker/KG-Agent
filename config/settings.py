import os

from dotenv import load_dotenv


load_dotenv()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "",
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "",
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "",
)

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j",
)

def validate_llm_config() -> None:
    required_config = {
        "LLM_PROVIDER": LLM_PROVIDER,
        "LLM_API_KEY": LLM_API_KEY,
        "LLM_MODEL": LLM_MODEL,
        "LLM_BASE_URL": LLM_BASE_URL,

    }

    missing = [
        name
        for name, value in required_config.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing LLM configuration: {', '.join(missing)}"
        )


def validate_openalex_config() -> None:
    if not OPENALEX_API_KEY:
        raise ValueError(
            "OPENALEX_API_KEY is not configured. "
            "Please set it in the .env file."
        )


def validate_neo4j_config() -> None:
    required_config = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
    }

    missing = [
        name
        for name, value in required_config.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing Neo4j configuration: "
            f"{', '.join(missing)}"
        )