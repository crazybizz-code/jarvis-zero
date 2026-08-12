from __future__ import annotations

from jarvis.tools.base import BaseTool


class ToolRegistry:
    """
    Holds all tools JARVIS knows about and serves them to two consumers:

    1. The LLM — via all_schemas(), which returns the list of tool
       definitions the model needs to know what it can call.

    2. The agent loop — via get(), which looks up a tool by name when the
       LLM returns a ToolCall so execute() can be called.

    Tools are registered once at startup (in main.py) and then treated as
    read-only during a session.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry. Raises if the name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        """
        Return all tool definitions in OpenAI function-calling format.
        Passed to BaseLLM.chat() so the model knows what tools exist.
        """
        return [tool.schema for tool in self._tools.values()]

    def names(self) -> list[str]:
        """Sorted list of registered tool names (useful for display)."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
