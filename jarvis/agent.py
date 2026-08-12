from __future__ import annotations

import json
from typing import Callable

from jarvis.llm.base import BaseLLM, ToolCall
from jarvis.permissions.gate import PermissionGate
from jarvis.tools.base import ToolResult
from jarvis.tools.registry import ToolRegistry

_SYSTEM_PROMPT = (
    "You are JARVIS, a local AI assistant running entirely on the user's machine. "
    "You are helpful, precise, and privacy-respecting. "
    "Use available tools when they help answer the user's request. "
    "When a tool returns an error, explain what went wrong and try to help another way."
)

_DEFAULT_MAX_ITERATIONS = 10


class Agent:
    """
    Core agent loop — connects the LLM, tool registry, and permission gate.

    Each call to chat() represents one user turn. The loop runs until the LLM
    produces a plain text reply (no tool calls), or until max_iterations is
    reached. Conversation history is kept in memory for the lifetime of this
    object; call clear_history() to start a fresh conversation.
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        gate: PermissionGate,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        on_tool_call: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._gate = gate
        self._max_iterations = max_iterations
        self._on_tool_call = on_tool_call
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send one user message, run the agent loop, and return the final reply.
        Conversation history is updated in place.
        """
        self._history.append({"role": "user", "content": user_message})

        for iteration in range(self._max_iterations):
            response = self._llm.chat(
                messages=self._build_messages(),
                tools=self._registry.all_schemas() or None,
            )

            if not response.has_tool_calls:
                # LLM produced a final text reply — we're done
                reply = response.content or ""
                self._history.append({"role": "assistant", "content": reply})
                return reply

            # LLM requested one or more tool calls — handle each one
            self._record_tool_request(response.tool_calls)

            for tool_call in response.tool_calls:
                result = self._dispatch(tool_call)
                self._record_tool_result(tool_call.id, result)

        # Reached iteration limit without a text reply
        fallback = (
            "I wasn't able to complete this in a reasonable number of steps. "
            "Please try rephrasing your request."
        )
        self._history.append({"role": "assistant", "content": fallback})
        return fallback

    def clear_history(self) -> None:
        """Reset conversation history, starting a fresh session."""
        self._history = []

    @property
    def history(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Prepend system prompt to history for each LLM call."""
        return [{"role": "system", "content": _SYSTEM_PROMPT}] + self._history

    def _dispatch(self, tool_call: ToolCall) -> ToolResult:
        """
        Resolve a tool call through the registry and permission gate,
        then execute it. Returns a ToolResult in all cases — never raises.
        """
        tool = self._registry.get(tool_call.name)
        if tool is None:
            return ToolResult(
                content=f"Unknown tool '{tool_call.name}'. Available tools: {self._registry.names()}",
                error=True,
            )

        if self._on_tool_call:
            self._on_tool_call(tool_call.name, tool_call.arguments)

        denial = self._gate.check(tool, tool_call.arguments)
        if denial is not None:
            return denial

        try:
            return tool.execute(**tool_call.arguments)
        except TypeError as exc:
            return ToolResult(
                content=f"Tool '{tool_call.name}' received unexpected arguments: {exc}",
                error=True,
            )
        except Exception as exc:
            return ToolResult(
                content=f"Tool '{tool_call.name}' failed: {exc}",
                error=True,
            )

    def _record_tool_request(self, tool_calls: list[ToolCall]) -> None:
        """
        Add the assistant's tool request to history.
        This must appear before the corresponding tool results — the OpenAI
        message format requires the assistant message to precede tool messages.
        """
        self._history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        })

    def _record_tool_result(self, tool_call_id: str, result: ToolResult) -> None:
        """Add one tool result to history, keyed by the matching tool_call_id."""
        self._history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result.content,
        })
