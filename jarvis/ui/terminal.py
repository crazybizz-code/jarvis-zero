from __future__ import annotations

import io

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


class TerminalUI:
    """
    All display logic for the JARVIS terminal interface.

    Accepts an optional console parameter so tests can inject a
    Console(file=StringIO()) and capture output without printing to the
    terminal.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def show_welcome(self, model_name: str) -> None:
        self._console.print()
        self._console.print(Panel(
            f"[bold green]JARVIS ZERO[/bold green]  [dim]v0.1[/dim]\n"
            f"[dim]Model: {model_name}  •  Type [/dim][bold]exit[/bold][dim] or Ctrl+C to quit[/dim]",
            expand=False,
            border_style="green",
        ))
        self._console.print()

    def show_goodbye(self) -> None:
        self._console.print()
        self._console.print("[dim]Goodbye.[/dim]")

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def get_input(self) -> str:
        """Show the user prompt and return the stripped input."""
        return self._console.input("[bold cyan]You:[/bold cyan] ").strip()

    def show_reply(self, text: str) -> None:
        self._console.print()
        self._console.print("[bold green]JARVIS:[/bold green]", end=" ")
        # Render as Markdown so the model can use bullet points, code blocks, etc.
        self._console.print(Markdown(text))
        self._console.print()

    # ------------------------------------------------------------------
    # Tool activity (shown before gate prompt for dangerous tools)
    # ------------------------------------------------------------------

    def show_tool_call(self, name: str, arguments: dict) -> None:
        args_str = (
            "  ".join(f"[italic]{k}[/italic]={v!r}" for k, v in arguments.items())
            if arguments else ""
        )
        label = f"[dim yellow]⚙  {name}[/dim yellow]"
        if args_str:
            label += f"[dim]({args_str})[/dim]"
        self._console.print(label)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def show_error(self, message: str) -> None:
        self._console.print()
        self._console.print(f"[bold red]Error:[/bold red] {message}")
        self._console.print()

    def show_connection_error(self, base_url: str) -> None:
        self._console.print()
        self._console.print(Panel(
            f"[bold red]Could not reach the LLM at[/bold red] [yellow]{base_url}[/yellow]\n\n"
            "[dim]Make sure Ollama is running:[/dim]\n"
            "  [bold]ollama serve[/bold]\n\n"
            "[dim]And the model is available:[/dim]\n"
            "  [bold]ollama pull llama3.2[/bold]",
            title="Connection failed",
            border_style="red",
            expand=False,
        ))
        self._console.print()

    # ------------------------------------------------------------------
    # Gate integration — used as input_fn for PermissionGate
    # ------------------------------------------------------------------

    def prompt_confirm(self, prompt: str) -> str:
        """Styled replacement for built-in input() used by PermissionGate."""
        return self._console.input(f"[bold yellow]{prompt}[/bold yellow]").strip()
