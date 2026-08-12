from __future__ import annotations

import json
from typing import Callable

from jarvis.config import PermissionsConfig
from jarvis.tools.base import BaseTool, ToolResult


class PermissionGate:
    """
    Guards dangerous tool calls with an explicit user confirmation prompt.

    Before execute() is called on any tool where dangerous=True, the gate
    prints the tool name and all arguments and asks the user to approve.
    The default answer is NO — pressing Enter alone denies the action.

    input_fn is injected so tests can simulate user responses without
    touching stdin (pass lambda _: 'y' to approve, lambda _: 'n' to deny).
    """

    def __init__(
        self,
        config: PermissionsConfig,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self._confirm_dangerous = config.confirm_dangerous
        self._input_fn = input_fn

    def check(self, tool: BaseTool, kwargs: dict) -> ToolResult | None:
        """
        Decide whether a tool call is allowed to proceed.

        Returns:
            None          — approved, caller should run tool.execute()
            ToolResult    — denied, caller must use this as the result
                            and must NOT call tool.execute()
        """
        if not tool.dangerous or not self._confirm_dangerous:
            return None  # safe tool or confirmations disabled — proceed

        self._show_request(tool, kwargs)

        try:
            answer = self._input_fn("  Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer == "y":
            return None  # approved

        return ToolResult(
            content=f"Action denied by user: '{tool.name}' was not executed.",
            error=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_request(self, tool: BaseTool, kwargs: dict) -> None:
        """Print a clear, readable summary of what the tool is about to do."""
        print()
        print("  ┌─ Permission required ──────────────────────────────")
        print(f"  │  Tool      : {tool.name}")
        if kwargs:
            args_str = json.dumps(kwargs, indent=2)
            for line in args_str.splitlines():
                print(f"  │  Arguments : {line}" if line == args_str.splitlines()[0] else f"  │             {line}")
        else:
            print("  │  Arguments : (none)")
        print("  └────────────────────────────────────────────────────")
