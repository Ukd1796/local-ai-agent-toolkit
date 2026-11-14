"""Command-line interface for the local Qwen agent."""

from __future__ import annotations

import asyncio
import logging

import httpx
from ollama import ResponseError

from .agent import Agent
from .config import Settings, load_settings
from .ollama_client import LocalOllama
from .tools import ToolDispatcher


def _terminal_event(state: dict[str, str | None], event: str, text: str) -> None:
    channel = state["channel"]
    if event in {"thinking", "content"}:
        if channel != event:
            if channel is not None:
                print()
            label = "Thinking" if event == "thinking" else "Qwen"
            print(f"\n{label}: ", end="", flush=True)
            state["channel"] = event
        print(text, end="", flush=True)
    elif event == "tool_start":
        if channel is not None:
            print()
        print(f"\n[tool] Calling {text}...", flush=True)
        state["channel"] = None
    elif event == "tool_end":
        print(f"[tool] {text} finished.", flush=True)
        state["channel"] = None


async def run_interactive(settings: Settings, *, workspace_access: bool = False) -> None:
    dispatcher = ToolDispatcher(settings, workspace_access=workspace_access)
    if workspace_access:
        inventory = await asyncio.to_thread(dispatcher.workspace_inventory)
        files = inventory.get("files", []) if inventory else []
        directories = inventory.get("directories", []) if inventory else []
        print(f"Workspace access granted: {settings.workspace_root}")
        print(f"Initial scan found {len(directories)} folders and {len(files)} files" + (" (list capped)." if inventory and inventory.get("truncated") else "."))
    else:
        print("Workspace access not granted; local file and command tools are disabled.")

    ollama = LocalOllama(settings)
    try:
        await ollama.check_model()
    except (ResponseError, httpx.HTTPError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not use Ollama at {settings.ollama_host}: {exc}. "
            "Make sure Ollama is running and the configured model is pulled."
        ) from exc

    agent = Agent(settings, dispatcher, ollama)
    print("Local Qwen Research Agent")
    print(f"Model: {settings.ollama_model}")
    if workspace_access:
        print(f"Workspace: {settings.workspace_root}")
    print("Search: Tavily API (direct)")
    print("Available tools:")
    for name in dispatcher.tools:
        print(f"  - {name}")
    print("Type /exit to quit.\n")
    print(f"Conversation context: up to {settings.max_history_turns} prior turns are retained.")
    print("Type /reset to clear it.\n")
    print("Type /context to show the latest Ollama context usage.\n")

    while True:
        try:
            prompt = await asyncio.to_thread(input, "You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if prompt.lower() == "/reset":
            agent.reset()
            print("Conversation context cleared.\n")
            continue
        if prompt.lower() == "/context":
            if not agent.show_last_context():
                print("No context has been sent to Ollama yet.\n")
            else:
                print()
            continue
        logging.info("[user] %s", prompt)
        stream_state: dict[str, str | None] = {"channel": None}
        try:
            await agent.ask(
                prompt,
                on_event=lambda event, text: _terminal_event(stream_state, event, text),
            )
        except RuntimeError as exc:
            if stream_state["channel"] is not None:
                print()
            print(f"Agent error: {exc}")
            continue
        print("\n")


def main() -> None:
    try:
        settings = load_settings()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    logging.basicConfig(
        level=logging.DEBUG if settings.verbose else logging.INFO,
        format="%(message)s",
    )
    print(f"Workspace access request: {settings.workspace_root}")
    print("If granted, the agent can list, read, search, create, and edit files in this workspace.")
    print("It will only make file changes when you ask it to change something.")
    print("It can also run limited read-only discovery commands (rg, grep, find, ls, cat, head, tail, wc, pwd).")
    try:
        response = input("Allow workspace access for this session? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = ""
    workspace_access = response in {"y", "yes"}
    try:
        asyncio.run(run_interactive(settings, workspace_access=workspace_access))
    except RuntimeError as exc:
        raise SystemExit(f"Startup error: {exc}") from exc
    except KeyboardInterrupt:
        print("\nGoodbye.")
