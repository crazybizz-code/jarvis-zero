from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool invocation requested by the LLM."""
    id: str            # opaque ID the API assigns; needed when we send the result back
    name: str          # matches a registered tool's name
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """
    The LLM's reply to a chat() call.

    Exactly one of content or tool_calls carries meaningful data per turn:
    - content is set (tool_calls is empty)  → LLM produced a text reply
    - tool_calls is non-empty (content is None) → LLM wants to call tool(s)
    """
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """
    Abstract interface for any LLM backend.

    Subclasses implement chat() for their specific API. The agent loop
    only ever calls methods defined here, so backends are swappable.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name of the active model (for display)."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """
        Send a conversation to the LLM and return its response.

        Args:
            messages: Full conversation history in OpenAI message format.
                      [{"role": "user"|"assistant"|"tool", "content": "..."}]
            tools:    Optional list of tool definitions in OpenAI function
                      calling schema. Pass None to disable tool use.

        Returns:
            ChatResponse with either a text reply or tool call requests.
        """
        ...
