"""Run the generic research scenarios from the project architecture brief."""

from __future__ import annotations

import asyncio
import logging
import argparse

import httpx
from ollama import ResponseError

from qwen_agent.agent import Agent
from qwen_agent.config import load_settings
from qwen_agent.ollama_client import LocalOllama
from qwen_agent.tools import ToolDispatcher


EXAMPLES = [
    ("Simple concept", "What is a mutex?"),
    ("Current information", "What are the latest AI developments today?"),
    ("Technical research", "How does Kafka replication work?"),
    ("Comparison", "Compare Kafka, Pulsar and Redpanda."),
    ("Product research", "What are the current options for running a 30B open-source model on a single GPU?"),
    ("Scientific research", "Explain the latest approaches to federated learning with differential privacy."),
    ("Specific documentation", "Find the official Python documentation for asyncio task cancellation and explain it."),
    ("Current time", "What time is it in Mumbai?"),
]


async def run_examples(limit: int | None = None, start: int = 1) -> None:
    settings = load_settings()
    ollama = LocalOllama(settings)
    await ollama.check_model()
    agent = Agent(settings, ToolDispatcher(settings), ollama)
    selected = EXAMPLES[max(0, start - 1):]
    if limit is not None:
        selected = selected[:limit]
    for label, prompt in selected:
        print(f"\n=== {label} ===\nQuestion: {prompt}", flush=True)
        agent.reset()
        result = await agent.ask(prompt)
        print(f"Tools: {', '.join(result.called_tools) or 'none'}", flush=True)
        print(f"Answer: {result.answer}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Run only the first N examples")
    parser.add_argument("--start", type=int, default=1, help="Start at example number N")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(run_examples(args.limit, args.start))
    except (ResponseError, httpx.HTTPError, RuntimeError) as exc:
        raise SystemExit(f"Research examples stopped: {exc}") from exc


if __name__ == "__main__":
    main()
