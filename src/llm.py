"""Thin wrapper around the OpenAI-compatible client pointed at Fireworks.

Every call is timed and costed so latency/cost data is available for the
report and the eval harness without having to retrofit it later.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.config import (
    ChatModelConfig,
    EmbedModelConfig,
    get_chat_model,
    get_embed_model,
)

load_dotenv()


@dataclass
class ChatResult:
    message: Any  # openai.types.chat.ChatCompletionMessage
    latency_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    parsed: BaseModel | None = None


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    latency_s: float
    cost_usd: float
    tokens: int


class LLM:
    """Fireworks client for chat completions and embeddings."""

    def __init__(self):
        self._clients: dict[tuple[str, str], OpenAI] = {}

    def _client_for(self, base_url: str, api_key_env: str) -> OpenAI:
        key = (base_url, api_key_env)
        if key not in self._clients:
            self._clients[key] = OpenAI(
                api_key=os.environ.get(api_key_env), base_url=base_url
            )
        return self._clients[key]

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        response_model: type[BaseModel] | None = None,
        **kwargs,
    ) -> ChatResult:
        config: ChatModelConfig = get_chat_model(model) if model else get_chat_model()
        client = self._client_for(config.base_url, config.api_key_env)

        if response_model is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }

        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=config.id, messages=messages, temperature=temperature, **kwargs
        )
        latency_s = time.perf_counter() - start

        message = completion.choices[0].message
        usage = completion.usage
        return ChatResult(
            message=message,
            latency_s=latency_s,
            cost_usd=config.cost(usage.prompt_tokens, usage.completion_tokens),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            parsed=response_model.model_validate_json(message.content)
            if response_model
            else None,
        )

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        is_query: bool = False,
    ) -> EmbedResult:
        """Embed a batch of texts. Applies the model's required task prefix
        (nomic-embed-text-v1.5 needs "search_query: " / "search_document: ")."""
        config: EmbedModelConfig = (
            get_embed_model(model) if model else get_embed_model()
        )
        client = self._client_for(config.base_url, config.api_key_env)
        prefix = config.query_prefix if is_query else config.document_prefix
        prefixed = [prefix + t for t in texts]

        start = time.perf_counter()
        response = client.embeddings.create(model=config.id, input=prefixed)
        latency_s = time.perf_counter() - start

        tokens = response.usage.total_tokens if response.usage else 0
        return EmbedResult(
            vectors=[d.embedding for d in response.data],
            latency_s=latency_s,
            cost_usd=config.cost(tokens),
            tokens=tokens,
        )
