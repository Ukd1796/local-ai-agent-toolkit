"""Tests for model-directed tool dispatch in the local agent harness."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from qwen_agent.agent import Agent
from qwen_agent.config import AGENT_DIR, Settings
from qwen_agent.tools import ToolDispatcher


def response(content: str = "", tool_calls=(), thinking: str = "") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls), role="assistant", thinking=thinking)
    return SimpleNamespace(message=message, model="test-model")


def tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def settings() -> Settings:
    return Settings(
        workspace_root=AGENT_DIR,
        ollama_model="qwen3:8b",
        ollama_host="http://localhost:11434",
        timezone="Asia/Kolkata",
        max_tool_iterations=4,
        max_history_turns=6,
        max_context_chars=48000,
        max_tool_result_chars=12000,
        verbose=False,
    )


class FakeOllama:
    def __init__(self, answers: list[SimpleNamespace]) -> None:
        self.answers = iter(answers)
        self.calls = []

    async def stream_chat(self, messages, tools=None, *, think=True):
        self.calls.append({"messages": messages, "tools": tools, "think": think})
        answer = next(self.answers).message
        if answer.thinking:
            yield SimpleNamespace(message=SimpleNamespace(thinking=answer.thinking, content=None, tool_calls=[]))
        if answer.content:
            midpoint = max(1, len(answer.content) // 2)
            for piece in (answer.content[:midpoint], answer.content[midpoint:]):
                if piece:
                    yield SimpleNamespace(message=SimpleNamespace(thinking=None, content=piece, tool_calls=[]))
        if answer.tool_calls:
            yield SimpleNamespace(message=SimpleNamespace(thinking=None, content=None, tool_calls=answer.tool_calls))


class FakeDispatcher:
    tools = ("web_search", "calculator", "get_current_time")

    def __init__(self) -> None:
        self.calls = []

    def ollama_tool_definitions(self):
        return [
            {"type": "function", "function": {
                "name": "web_search", "description": "Search the web for useful information.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }},
            {"type": "function", "function": {
                "name": "calculator", "description": "Calculate arithmetic.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
            }},
        ]

    async def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"tool": name, "arguments": arguments, "result": "ok"})


class ToolDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_selects_and_dispatches_multiple_tools(self):
        ollama = FakeOllama([
            response(tool_calls=[
                tool_call("web_search", {"query": "Kafka replication"}),
                tool_call("calculator", {"expression": "20 + 22"}),
            ], thinking="I need current details and a calculation."),
            response("Kafka copies records between brokers; 20 + 22 = 42."),
        ])
        dispatcher = FakeDispatcher()
        agent = Agent(settings(), dispatcher, ollama)
        events = []

        result = await agent.ask(
            "Explain Kafka replication and calculate 20 + 22.",
            on_event=lambda event, text: events.append((event, text)),
        )

        self.assertEqual(result.called_tools, ["web_search", "calculator"])
        self.assertEqual([name for name, _ in dispatcher.calls], ["web_search", "calculator"])
        self.assertIn("Search the web", json.dumps(ollama.calls[0]["tools"]))
        self.assertTrue(any(
            message.get("role") == "tool" and message.get("tool_name") == "web_search"
            for message in ollama.calls[1]["messages"]
        ))
        self.assertIn("42", result.answer)
        self.assertIn(("thinking", "I need current details and a calculation."), events)
        self.assertTrue(any(event == "content" for event, _ in events))

    async def test_qwen_can_answer_without_calling_a_tool(self):
        ollama = FakeOllama([response("A mutex is a lock for shared state.")])
        dispatcher = FakeDispatcher()
        agent = Agent(settings(), dispatcher, ollama)

        result = await agent.ask("What is a mutex?")

        self.assertEqual(result.called_tools, [])
        self.assertEqual(dispatcher.calls, [])
        self.assertIn("mutex", result.answer.lower())

    async def test_custom_registered_tool_is_advertised_and_dispatched(self):
        dispatcher = ToolDispatcher(settings())

        async def lookup(arguments):
            return {"lookup": arguments["term"]}

        dispatcher.register_tool(
            name="lookup",
            description="Look up an item by its term.",
            parameters={"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
            handler=lookup,
        )

        definitions = dispatcher.ollama_tool_definitions()
        lookup_definition = next(item for item in definitions if item["function"]["name"] == "lookup")
        result = await dispatcher.dispatch("lookup", {"term": "sample"})

        self.assertEqual(lookup_definition["function"]["description"], "Look up an item by its term.")
        self.assertEqual(json.loads(result), {"lookup": "sample"})


if __name__ == "__main__":
    unittest.main()
