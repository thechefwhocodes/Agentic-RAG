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
        **kwargs,
    ) -> ChatResult:
        """Send a chat completion request. Pinned to temperature=0 by default
        so agent behavior and eval scores are reproducible run to run;
        override via kwargs if a caller ever needs otherwise. Extra kwargs
        (tools, tool_choice, ...) pass through."""
        config: ChatModelConfig = get_chat_model(model) if model else get_chat_model()
        client = self._client_for(config.base_url, config.api_key_env)

        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=config.id, messages=messages, temperature=temperature, **kwargs
        )
        latency_s = time.perf_counter() - start

        usage = completion.usage
        return ChatResult(
            message=completion.choices[0].message,
            latency_s=latency_s,
            cost_usd=config.cost(usage.prompt_tokens, usage.completion_tokens),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
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
