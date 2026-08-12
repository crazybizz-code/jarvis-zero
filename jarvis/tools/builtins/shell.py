from __future__ import annotations

import subprocess

from jarvis.tools.base import BaseTool, ToolResult

_TIMEOUT_SECONDS = 30


class ShellTool(BaseTool):
    """
    Runs a shell command and returns its output.

    Marked dangerous=True — the permission gate will always ask the user
    to confirm before this tool executes, showing the exact command.
    """

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Runs a shell command on the local machine and returns its output. "
            "Use for file operations, system info, running scripts, etc. "
            "Always requires user confirmation before executing."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run, e.g. 'ls -la' or 'echo hello'",
                }
            },
            "required": ["command"],
        }

    @property
    def dangerous(self) -> bool:
        return True

    def execute(self, **kwargs) -> ToolResult:
        command = kwargs.get("command", "")
        if not command.strip():
            return ToolResult(content="No command provided.", error=True)

        try:
            proc = subprocess.run(
                command,
                shell=True,          # supports pipes and redirects; gate provides safety
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
            )
            output = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0:
                detail = stderr or output or "(no output)"
                return ToolResult(content=f"Command failed (exit {proc.returncode}): {detail}", error=True)

            return ToolResult(content=output or "(command completed with no output)")

        except subprocess.TimeoutExpired:
            return ToolResult(content=f"Command timed out after {_TIMEOUT_SECONDS}s.", error=True)
        except Exception as exc:
            return ToolResult(content=f"Unexpected error: {exc}", error=True)
