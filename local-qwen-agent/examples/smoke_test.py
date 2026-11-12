"""End-to-end check for Ollama, direct tools, and a final answer."""

from __future__ import annotations

import asyncio
import logging

import httpx
from ollama import ResponseError

from qwen_agent.agent import Agent
from qwen_agent.config import load_settings
from qwen_agent.ollama_client import LocalOllama
from qwen_agent.tools import ToolDispatcher


async def smoke_test() -> None:
    settings = load_settings()
    print("[smoke] Checking Ollama and configured model...")
    ollama = LocalOllama(settings)
    try:
        await ollama.check_model()
        basic = await ollama.chat([{"role": "user", "content": "Reply with exactly: Ollama is ready."}])
    except (ResponseError, httpx.HTTPError, RuntimeError) as exc:
        raise RuntimeError(f"Ollama check failed: {exc}") from exc
    if not basic.message.content:
        raise RuntimeError("Ollama responded but returned no text for a plain chat request.")
    print("[smoke] Ollama responds without tools.")

    dispatcher = ToolDispatcher(settings)
    print("[smoke] Registered tools: " + ", ".join(dispatcher.tools))
    if "web_search" not in dispatcher.tools:
        raise RuntimeError("The direct Tavily search tool is unavailable.")
    agent = Agent(settings, dispatcher, ollama)
    print("[smoke] Asking the agent to search Tavily and answer...")
    turn = await agent.ask(
        "Use the web_search tool now. Search for: Python programming language official website. "
        "After you receive the tool result, summarize one result in one sentence."
    )
    if "web_search" not in turn.called_tools:
        raise RuntimeError("Qwen did not select web_search for the smoke-test request.")
    if not turn.tool_results:
        raise RuntimeError("web_search was called but returned no tool result to the agent.")
    if not turn.answer:
        raise RuntimeError("The agent returned no answer.")
    print("[smoke] Final answer: " + turn.answer)
    print("[smoke] End-to-end check passed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(smoke_test())
    except (RuntimeError, ResponseError, httpx.HTTPError) as exc:
        raise SystemExit(f"[smoke] FAILED: {exc}") from exc


if __name__ == "__main__":
    main()
