"""Tests for model-directed tool dispatch in the local agent harness."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

from qwen_agent.agent import Agent
from qwen_agent.config import AGENT_DIR, Settings
from qwen_agent.task_state import TASK_STATE_SCHEMA
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
    def __init__(self, answers: list[SimpleNamespace], task_updates: list[dict] | None = None) -> None:
        self.answers = iter(answers)
        self.task_updates = iter(task_updates or [])
        self.calls = []
        self.state_calls = []

    async def chat(self, messages, tools=None, *, think=True, format=None):
        self.state_calls.append({"messages": messages, "tools": tools, "think": think, "format": format})
        state_message = next(message for message in messages if message.get("content", "").startswith("Persistent Task State"))
        prior_state = json.loads(state_message["content"].split("\n", 1)[1])
        user_message = messages[-1]["content"]
        try:
            update = next(self.task_updates)
        except StopIteration:
            starting = prior_state["objective"] is None
            update = {
                "decision": "start" if starting else "continue",
                "objective_update": user_message if starting else "",
                "current_subtask": user_message,
                "constraints_added": [],
                "constraints_removed": [],
                "status": "active",
                "confidence": 0.95,
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(update)))

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
    @staticmethod
    def task_state_from_call(call):
        message = next(
            message for message in call["messages"]
            if message.get("content", "").startswith("Persistent Task State")
        )
        return json.loads(message["content"].split("\n", 1)[1])

    async def test_initial_task_state_is_separate_and_sent_to_ollama(self):
        ollama = FakeOllama([response("I will build it.")], task_updates=[{
            "decision": "start", "objective_update": "Build a parser.",
            "current_subtask": "Build a parser.", "constraints_added": ["Do not add dependencies."],
            "constraints_removed": [], "status": "active", "confidence": 0.95,
        }])
        agent = Agent(settings(), FakeDispatcher(), ollama)

        await agent.ask("Build a parser. Do not add dependencies.")

        state = self.task_state_from_call(ollama.calls[0])
        self.assertEqual(state["objective"], "Build a parser.")
        self.assertEqual(state["current_subtask"], state["objective"])
        self.assertEqual(state["status"], "active")
        self.assertEqual(state["constraints"], ["Do not add dependencies."])
        self.assertFalse(any("Persistent Task State" in str(message) for turn in agent.history for message in turn))

    async def test_follow_up_preserves_objective_and_updates_subtask(self):
        ollama = FakeOllama([response("Starting."), response("Validation added.")], task_updates=[
            {"decision": "start", "objective_update": "Build a parser.", "current_subtask": "Build a parser.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
            {"decision": "continue", "objective_update": "", "current_subtask": "Now add input validation.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
        ])
        agent = Agent(settings(), FakeDispatcher(), ollama)

        await agent.ask("Build a parser.")
        await agent.ask("Now add input validation.")

        state = self.task_state_from_call(ollama.calls[-1])
        self.assertEqual(state["objective"], "Build a parser.")
        self.assertEqual(state["current_subtask"], "Now add input validation.")

    async def test_task_state_survives_history_trimming(self):
        ollama = FakeOllama([response("One."), response("Two."), response("Three.")], task_updates=[
            {"decision": "start", "objective_update": "Build a parser.", "current_subtask": "Build a parser.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
            {"decision": "continue", "objective_update": "", "current_subtask": "Add validation.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
            {"decision": "continue", "objective_update": "", "current_subtask": "Add error messages.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
        ])
        agent = Agent(replace(settings(), max_history_turns=1), FakeDispatcher(), ollama)

        await agent.ask("Build a parser.")
        await agent.ask("Add validation.")
        await agent.ask("Add error messages.")

        self.assertEqual(len(agent.history), 1)
        state = self.task_state_from_call(ollama.calls[-1])
        self.assertEqual(state["objective"], "Build a parser.")
        self.assertEqual(state["current_subtask"], "Add error messages.")

    async def test_explicit_task_change_replaces_objective(self):
        ollama = FakeOllama([response("Starting."), response("CLI started.")], task_updates=[
            {"decision": "start", "objective_update": "Build a parser.", "current_subtask": "Build a parser.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
            {"decision": "change", "objective_update": "Build a CLI instead.", "current_subtask": "Build a CLI instead.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
        ])
        agent = Agent(settings(), FakeDispatcher(), ollama)

        await agent.ask("Build a parser.")
        await agent.ask("Forget that, let's build a CLI instead.")

        state = self.task_state_from_call(ollama.calls[-1])
        self.assertEqual(state["objective"], "Build a CLI instead.")
        self.assertEqual(state["current_subtask"], state["objective"])

    async def test_uncertain_update_preserves_existing_task_state(self):
        ollama = FakeOllama([response("Starting."), response("Continuing." )], task_updates=[
            {"decision": "start", "objective_update": "Build a parser.", "current_subtask": "Build a parser.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.95},
            {"decision": "uncertain", "objective_update": "Replace it with a CLI.", "current_subtask": "Unclear follow-up.", "constraints_added": [], "constraints_removed": [], "status": "active", "confidence": 0.45},
        ])
        agent = Agent(settings(), FakeDispatcher(), ollama)

        await agent.ask("Build a parser.")
        await agent.ask("Maybe the command-line thing?")

        state = self.task_state_from_call(ollama.calls[-1])
        self.assertEqual(state["objective"], "Build a parser.")
        self.assertEqual(state["current_subtask"], "Build a parser.")

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
        self.assertTrue(all(
            any(message.get("content", "").startswith("Persistent Task State") for message in call["messages"])
            for call in ollama.calls
        ))
        self.assertEqual(len(ollama.state_calls), 1)
        self.assertEqual(ollama.state_calls[0]["format"], TASK_STATE_SCHEMA)
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
