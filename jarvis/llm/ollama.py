from __future__ import annotations

import json

from openai import OpenAI

from jarvis.config import LLMConfig
from jarvis.llm.base import BaseLLM, ChatResponse, ToolCall


class OllamaLLM(BaseLLM):
    """
    LLM backend that talks to Ollama via its OpenAI-compatible API.

    Ollama exposes the same HTTP interface as OpenAI, so we can use the
    official openai library pointed at localhost instead of any OpenAI cloud.
    api_key is required by the library but ignored by Ollama — we pass a
    placeholder so the library does not raise a missing-credentials error.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = OpenAI(
            api_key="ollama",   # placeholder — Ollama does not check this
            base_url=config.base_url,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # LLM requested one or more tool calls
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
            return ChatResponse(content=None, tool_calls=tool_calls)

        # Plain text reply
        return ChatResponse(content=message.content or "", tool_calls=[])
