from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """
    The outcome of executing a tool.

    content is always a string — the LLM receives tool results as text,
    regardless of what the underlying operation returned.
    error signals that something went wrong; the agent loop includes the
    content (error message) in history so the LLM can reason about it.
    """
    content: str
    error: bool = False


class BaseTool(ABC):
    """
    Contract that every JARVIS tool must satisfy.

    Subclasses must implement:
        name        — snake_case identifier the LLM uses to call this tool
        description — plain English so the LLM knows *when* to use it
        execute()   — the actual logic; receives validated kwargs from the LLM

    Subclasses may override:
        parameters  — JSON Schema for inputs (default: no arguments)
        dangerous   — whether the permission gate should intercept (default: False)
    """

    # ------------------------------------------------------------------
    # Required — subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """snake_case tool name, e.g. 'get_datetime' or 'run_shell'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One or two sentences: what this tool does and when to use it."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Run the tool with arguments supplied by the LLM."""
        ...

    # ------------------------------------------------------------------
    # Optional — subclasses override these when needed
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict:
        """
        JSON Schema for this tool's inputs.
        Default is an empty schema (no arguments required).
        Override for tools that accept parameters.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def dangerous(self) -> bool:
        """
        If True, the permission gate will ask for confirmation before
        execute() is called. Set to True for tools that write files,
        run shell commands, or make external network requests.
        """
        return False

    # ------------------------------------------------------------------
    # Assembled by the base class — registry and LLM consume this
    # ------------------------------------------------------------------

    @property
    def schema(self) -> dict:
        """
        OpenAI-compatible tool definition dict.
        The registry passes a list of these to the LLM on every chat() call.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
