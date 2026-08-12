from __future__ import annotations

from datetime import datetime

from jarvis.tools.base import BaseTool, ToolResult


class DateTimeTool(BaseTool):
    """Returns the current local date and time. Read-only, no arguments."""

    @property
    def name(self) -> str:
        return "get_datetime"

    @property
    def description(self) -> str:
        return (
            "Returns the current local date and time. "
            "Use this whenever the user asks what time or date it is."
        )

    def execute(self, **kwargs) -> ToolResult:
        now = datetime.now()
        return ToolResult(content=now.strftime("%Y-%m-%d %H:%M:%S"))
