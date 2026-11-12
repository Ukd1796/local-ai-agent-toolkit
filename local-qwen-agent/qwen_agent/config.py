"""Environment-backed configuration for the local Qwen agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


AGENT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(AGENT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    ollama_model: str
    ollama_host: str
    timezone: str
    max_tool_iterations: int
    max_history_turns: int
    max_context_chars: int
    max_tool_result_chars: int
    verbose: bool


def load_settings() -> Settings:
    configured_root = Path(os.getenv("WORKSPACE_ROOT", "..")).expanduser()
    workspace_root = (AGENT_DIR / configured_root).resolve() if not configured_root.is_absolute() else configured_root.resolve()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise ValueError(f"WORKSPACE_ROOT must point to an existing directory: {workspace_root}")

    try:
        iterations = int(os.getenv("MAX_TOOL_ITERATIONS", os.getenv("MAX_RESEARCH_ITERATIONS", "5")))
    except ValueError as exc:
        raise ValueError("MAX_TOOL_ITERATIONS must be an integer") from exc
    if not 1 <= iterations <= 20:
        raise ValueError("MAX_TOOL_ITERATIONS must be between 1 and 20")

    try:
        history_turns = int(os.getenv("MAX_HISTORY_TURNS", "6"))
        context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "48000"))
        tool_result_chars = int(os.getenv("MAX_TOOL_RESULT_CHARS", "12000"))
    except ValueError as exc:
        raise ValueError("History and context limits must be integers") from exc
    if not 1 <= history_turns <= 50:
        raise ValueError("MAX_HISTORY_TURNS must be between 1 and 50")
    if not 4000 <= context_chars <= 200000:
        raise ValueError("MAX_CONTEXT_CHARS must be between 4000 and 200000")
    if not 500 <= tool_result_chars <= context_chars:
        raise ValueError("MAX_TOOL_RESULT_CHARS must be between 500 and MAX_CONTEXT_CHARS")

    timezone = os.getenv("USER_TIMEZONE", "Asia/Kolkata")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"USER_TIMEZONE must be a valid IANA timezone: {timezone}") from exc

    return Settings(
        workspace_root=workspace_root,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        timezone=timezone,
        max_tool_iterations=iterations,
        max_history_turns=history_turns,
        max_context_chars=context_chars,
        max_tool_result_chars=tool_result_chars,
        verbose=os.getenv("VERBOSE", "false").lower() in {"1", "true", "yes", "on"},
    )
