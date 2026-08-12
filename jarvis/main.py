from __future__ import annotations

from jarvis.agent import Agent
from jarvis.config import load_config
from jarvis.llm.ollama import OllamaLLM
from jarvis.permissions.gate import PermissionGate
from jarvis.tools.builtins import CalculatorTool, DateTimeTool, ShellTool
from jarvis.tools.registry import ToolRegistry
from jarvis.ui.terminal import TerminalUI

_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}


def build_agent(ui: TerminalUI | None = None) -> tuple[Agent, TerminalUI]:
    """
    Construct and wire all components from config.
    Returns (agent, ui) so callers can interact with both.
    Separated from main() so tests can call it without starting the loop.
    """
    cfg = load_config()

    if ui is None:
        ui = TerminalUI()

    llm = OllamaLLM(cfg.llm)

    registry = ToolRegistry()
    registry.register(DateTimeTool())
    registry.register(CalculatorTool())
    registry.register(ShellTool())

    gate = PermissionGate(cfg.permissions, input_fn=ui.prompt_confirm)

    agent = Agent(
        llm=llm,
        registry=registry,
        gate=gate,
        on_tool_call=ui.show_tool_call,
    )

    return agent, ui


def run_loop(agent: Agent, ui: TerminalUI, llm_model: str) -> None:
    """
    Run the interactive conversation loop until the user exits.
    Separated from main() so it can be tested with injected dependencies.
    """
    ui.show_welcome(llm_model)

    while True:
        try:
            user_input = ui.get_input()
        except (EOFError, KeyboardInterrupt):
            ui.show_goodbye()
            break

        if not user_input:
            continue

        if user_input.lower() in _EXIT_COMMANDS:
            ui.show_goodbye()
            break

        try:
            reply = agent.chat(user_input)
            ui.show_reply(reply)
        except Exception as exc:
            # Catch connection errors, unexpected model errors, etc.
            msg = str(exc)
            if "connection" in msg.lower() or "refused" in msg.lower():
                from jarvis.config import load_config as _lc
                ui.show_connection_error(_lc().llm.base_url)
            else:
                ui.show_error(msg)


def main() -> None:
    agent, ui = build_agent()
    from jarvis.config import load_config as _lc
    run_loop(agent, ui, llm_model=_lc().llm.model)


if __name__ == "__main__":
    main()
