from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Config dataclasses — one per section in config.yaml.
# All fields carry sensible defaults so the program works even if config.yaml
# is missing or partially filled in.
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class PermissionsConfig:
    confirm_dangerous: bool = True


@dataclass
class UIConfig:
    show_tool_calls: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _merge(dataclass_instance, data: dict) -> None:
    """Overwrite dataclass fields with values from a dict, ignoring unknown keys."""
    for key, value in data.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, value)


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load config.yaml from the given path (or the repo root next to this
    package).  Missing file or missing keys fall back to defaults silently.
    """
    if path is None:
        # Walk up from jarvis/config.py → jarvis/ → repo root
        path = Path(__file__).parent.parent / "config.yaml"

    config = Config()

    if not path.exists():
        return config  # all defaults

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if "llm" in raw:
        _merge(config.llm, raw["llm"])
    if "permissions" in raw:
        _merge(config.permissions, raw["permissions"])
    if "ui" in raw:
        _merge(config.ui, raw["ui"])

    return config
