"""Model catalog and paths. Swap models by changing DEFAULT_CHAT_MODEL / DEFAULT_EMBED_MODEL."""

from dataclasses import dataclass
from pathlib import Path

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_KEY_ENV_VAR = "FIREWORKS_API_KEY"

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "financials.db"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"

MAX_AGENT_STEPS = 12


@dataclass(frozen=True)
class ChatModelConfig:
    id: str  # full Fireworks model id, e.g. "accounts/fireworks/models/gpt-oss-120b"
    base_url: str
    api_key_env: str
    input_price_per_million: float  # USD per 1M input tokens
    output_price_per_million: float  # USD per 1M output tokens

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_price_per_million
            + completion_tokens * self.output_price_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class EmbedModelConfig:
    id: str  # full Fireworks model id, e.g. "nomic-ai/nomic-embed-text-v1.5"
    base_url: str
    api_key_env: str
    dimensions: int
    price_per_million_tokens: float
    query_prefix: str = ""  # prepended to queries before embedding
    document_prefix: str = ""  # prepended to documents before embedding

    def cost(self, tokens: int) -> float:
        return tokens * self.price_per_million_tokens / 1_000_000

    def collection_name(self) -> str:
        """Chroma collection name, unique per model + dimensions so switching
        embedding models can't silently query an index built by a different one."""
        safe_id = self.id.replace("/", "__").replace("-", "_").replace(".", "_")
        return f"filings__{safe_id}__{self.dimensions}"


GPT_OSS_120B = "gpt-oss-120b"
NOMIC_EMBED_TEXT_V1_5 = "nomic-embed-text-v1.5"

CHAT_MODELS: dict[str, ChatModelConfig] = {
    GPT_OSS_120B: ChatModelConfig(
        id="accounts/fireworks/models/gpt-oss-120b",
        base_url=FIREWORKS_BASE_URL,
        api_key_env=FIREWORKS_API_KEY_ENV_VAR,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
}

EMBED_MODELS: dict[str, EmbedModelConfig] = {
    NOMIC_EMBED_TEXT_V1_5: EmbedModelConfig(
        id="nomic-ai/nomic-embed-text-v1.5",
        base_url=FIREWORKS_BASE_URL,
        api_key_env=FIREWORKS_API_KEY_ENV_VAR,
        dimensions=768,
        price_per_million_tokens=0.008,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    ),
}

DEFAULT_CHAT_MODEL = GPT_OSS_120B
DEFAULT_EMBED_MODEL = NOMIC_EMBED_TEXT_V1_5


def get_chat_model(name: str = DEFAULT_CHAT_MODEL) -> ChatModelConfig:
    if name not in CHAT_MODELS:
        raise ValueError(
            f"Unknown chat model '{name}'. Available: {sorted(CHAT_MODELS)}"
        )
    return CHAT_MODELS[name]


def get_embed_model(name: str = DEFAULT_EMBED_MODEL) -> EmbedModelConfig:
    if name not in EMBED_MODELS:
        raise ValueError(
            f"Unknown embed model '{name}'. Available: {sorted(EMBED_MODELS)}"
        )
    return EMBED_MODELS[name]
