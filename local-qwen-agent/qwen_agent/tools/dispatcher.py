"""Local tools used by the Qwen agent, including direct Tavily Search API access."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
import operator
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..config import Settings
from .filesystem import WorkspaceFiles
from .tavily_api import RECENCY_TO_TIME_RANGE, search_tavily


MAX_SEARCH_RESULTS = 5
CALCULATOR_OPERATORS = {  
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

def _normalized_results(raw_results: list[Any], limit: int) -> list[dict[str, str | None]]:
    normalized = []
    seen: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = " ".join(str(item.get("title") or "").split())
        if not url or not title or url in seen:
            continue
        seen.add(url)
        published = item.get("published_date")
        if isinstance(published, str) and published:
            try:
                try:
                    published = datetime.fromisoformat(published.replace("Z", "+00:00")).isoformat(timespec="seconds")
                except ValueError:
                    published = parsedate_to_datetime(published).isoformat(timespec="seconds")
            except (TypeError, ValueError, OverflowError):
                published = None
        else:
            published = None
        normalized.append({
            "title": title,
            "url": url,
            "snippet": " ".join(str(item.get("content") or "").split())[:1200],
            "source": "tavily",
            "published_at": published,
        })
        if len(normalized) >= limit:
            break
    return normalized


def _search(query: str, max_results: int, domain: str | None, recency: str | None) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query cannot be empty")
    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")
    if recency is not None and recency not in RECENCY_TO_TIME_RANGE:
        raise ValueError("recency must be hour, day, week, month, or year")
    if domain:
        domain = domain.strip().lower().removeprefix("https://").removeprefix("http://").strip("/")
        if "." not in domain or not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]{1,5})?", domain):
            raise ValueError("domain must be a hostname, such as docs.example.com")

    warnings = []
    if recency == "hour":
        warnings.append("Tavily supports day-level publication filtering, so hour uses the day range.")
    try:
        payload = search_tavily(query, max_results=max_results, domain=domain, recency=recency)
        results = _normalized_results(payload.get("results", []), max_results)
        if not results and domain:
            # A planner-selected domain is a source preference, not a user-enforced allowlist.
            # Widen once so an overly narrow hostname does not end research with zero candidates.
            payload = search_tavily(query, max_results=max_results, recency=recency)
            results = _normalized_results(payload.get("results", []), max_results)
            if results:
                warnings.append(f"No results from {domain}; retried without the domain restriction.")
        if not results:
            warnings.append("Tavily returned no usable results.")
        return {
            "query": query,
            "requested_recency": recency,
            "applied_recency": "day" if recency == "hour" else recency,
            "results": results,
            "warnings": warnings,
        }
    except (RuntimeError, ValueError) as exc:
        return {"query": query, "results": [], "warnings": [str(exc)]}


def _calculate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > 24:
        raise ValueError("Expression is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _calculate(node.operand, depth + 1)
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp) and type(node.op) in CALCULATOR_OPERATORS:
        left, right = _calculate(node.left, depth + 1), _calculate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent magnitude must be 12 or less")
        value = CALCULATOR_OPERATORS[type(node.op)](left, right)
    else:
        raise ValueError("Only numeric arithmetic expressions are allowed")
    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("Calculation result is outside the supported range")
    return value


class ToolDispatcher:
    """Registers tool descriptions and routes model tool calls to their handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.filesystem = WorkspaceFiles(settings.workspace_root)
        self._tools: dict[str, dict[str, Any]] = {}
        self.register_tool(
            name="web_search",
            description=(
                "Search the web with Tavily for current, recent, niche, disputed, or source-requested "
                "information. Returns titles, snippets, URLs, and publication dates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search terms"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_RESULTS, "default": 5},
                    "recency": {"type": "string", "enum": ["hour", "day", "week", "month", "year"]},
                    "domain": {"type": "string", "description": "Optional domain restriction, such as docs.example.com"},
                },
                "required": ["query"],
            },
            handler=self._web_search,
        )
        self.register_tool(
            name="get_current_time",
            description="Get the current date and time. Use this for questions asking what time or date it is now.",
            parameters={"type": "object", "properties": {
                "timezone": {"type": "string", "description": "IANA timezone, such as Asia/Kolkata"},
            }},
            handler=self._get_current_time,
        )
        self.register_tool(
            name="calculator",
            description="Calculate an arithmetic expression instead of estimating. Supports parentheses, +, -, *, /, //, %, and **.",
            parameters={"type": "object", "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression"},
            }, "required": ["expression"]},
            handler=self._calculate,
        )
        self.register_tool(
            name="list_files",
            description=(
                "List files under the configured workspace root. Use this to discover available projects and their structure. "
                "Paths returned are relative to the workspace root. Supports a glob such as '**/*.py'."
            ),
            parameters={"type": "object", "properties": {
                "pattern": {"type": "string", "description": "Optional workspace-relative glob, default **/*"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
            }},
            handler=self.filesystem.list_files,
        )
        self.register_tool(
            name="read_file",
            description="Read a UTF-8 text file by workspace-relative path. Use before explaining or editing a file.",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Path relative to the configured workspace root; include the project folder when needed"},
            }, "required": ["path"]},
            handler=self.filesystem.read_file,
        )
        self.register_tool(
            name="search_files",
            description="Search literal text across files in the workspace. Use to locate definitions, references, or configuration before making changes.",
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "Case-insensitive literal text to find"},
                "pattern": {"type": "string", "description": "Optional relative glob, default **/*"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            }, "required": ["query"]},
            handler=self.filesystem.search_files,
        )
        self.register_tool(
            name="write_file",
            description=(
                "Create or replace a UTF-8 text file under the workspace root. Use only when the user asks to create or change files. "
                "Inspect related files first, preserve existing behavior, and report every modified path."
            ),
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Destination path relative to the workspace root"},
                "content": {"type": "string", "description": "Complete file contents"},
            }, "required": ["path", "content"]},
            handler=self.filesystem.write_file,
        )
        self.register_tool(
            name="edit_file",
            description=(
                "Replace one exact text block in an existing workspace file. The old_text must match exactly once; otherwise no change is made. "
                "Use this for focused changes after reading the file."
            ),
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "Existing file path relative to the workspace root"},
                "old_text": {"type": "string", "description": "Exact existing text to replace, including whitespace"},
                "new_text": {"type": "string", "description": "Replacement text"},
            }, "required": ["path", "old_text", "new_text"]},
            handler=self.filesystem.edit_file,
        )

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Register another tool so the model can select it by name and description."""
        if not name or name in self._tools:
            raise ValueError(f"Tool name is empty or already registered: {name!r}")
        self._tools[name] = {
            "definition": {"type": "function", "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }},
            "handler": handler,
        }

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def ollama_tool_definitions(self) -> list[dict[str, Any]]:
        return [tool["definition"] for tool in self._tools.values()]

    async def dispatch(self, name: str, arguments: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        if not isinstance(arguments, Mapping):
            return json.dumps({"error": "Tool arguments must be an object."})
        try:
            result = tool["handler"](dict(arguments))
            if inspect.isawaitable(result):
                result = await result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})

    async def _web_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(
            _search,
            str(arguments.get("query", "")),
            int(arguments.get("max_results", 5)),
            arguments.get("domain"),
            arguments.get("recency"),
        )

    async def _get_current_time(self, arguments: dict[str, Any]) -> dict[str, str]:
        timezone = str(arguments.get("timezone", self.settings.timezone))
        try:
            now = datetime.now(ZoneInfo(timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {timezone}") from exc
        return {
            "timezone": timezone,
            "timezone_name": now.tzname() or timezone,
            "datetime": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
        }

    @staticmethod
    async def _calculate(arguments: dict[str, Any]) -> dict[str, Any]:
        expression = str(arguments.get("expression", "")).strip()
        if not expression or len(expression) > 256:
            raise ValueError("Expression must contain 1 to 256 characters")
        try:
            result = _calculate(ast.parse(expression, mode="eval").body)
        except (SyntaxError, ArithmeticError, OverflowError) as exc:
            raise ValueError(f"Invalid arithmetic expression: {exc}") from exc
        return {"expression": expression, "result": result}
