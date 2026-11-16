# Local Qwen Research Agent

A local Ollama agent with a model-directed tool dispatcher. Qwen chooses tools from their descriptions, the harness invokes them, then streams the tool results back to the model. It includes Tavily search, current time, calculator, and project-scoped file tools.

```text
local-qwen-agent/
├── qwen_agent/       # agent, CLI, config, Ollama client, and tools/
├── examples/         # direct Tavily example, research examples, smoke check
├── tests/            # dispatcher and agent tests
├── .env              # local credentials and settings
└── pyproject.toml
```

## Requirements and setup

- macOS with Ollama installed and running
- Python 3.11 or newer
- `uv` (recommended) or Python's `venv` and `pip`
- A Tavily API key
- A Qwen model installed in Ollama with tool-calling support

From this directory:

```bash
uv venv
uv pip install --python .venv/bin/python -e .
```

On first setup only, create `.env` from `.env.example`, then add your Tavily API key:

```bash
cp .env.example .env
```

```dotenv
TAVILY_API_KEY=your-tavily-api-key
OLLAMA_MODEL=qwen3:8b
OLLAMA_HOST=http://localhost:11434
USER_TIMEZONE=Asia/Kolkata
WORKSPACE_ROOT=..
MAX_TOOL_ITERATIONS=5
```

`.env` and `.venv` are ignored by Git. The Tavily key is sent to Tavily over HTTPS and is not printed.

## Run the agent

```bash
ollama pull qwen3:8b
.venv/bin/python -m qwen_agent
```

Or use the installed command:

```bash
.venv/bin/qwen-agent
```

Type `/exit` to quit or `/reset` to clear conversation history. The terminal streams Qwen's `Thinking:` channel separately from `Qwen:` answer text when the model provides it.

## Tool structure

Built-in tools live in `qwen_agent/tools/dispatcher.py`. At startup, the CLI asks whether to grant this session access to `WORKSPACE_ROOT`. If allowed, it scans the workspace at startup and refreshes a bounded file inventory before every request. The model receives that inventory and can use generic filesystem tools (`list_files`, `read_file`, `search_files`, `write_file`, and `edit_file`) across multiple projects inside the workspace. The default is the parent of `local-qwen-agent`, so sibling folders are available. Paths are relative to that workspace root; include a project folder in the path when needed. Set `WORKSPACE_ROOT` in `.env` to choose a different workspace. Relative values are resolved from this folder.

The filesystem tools skip dependency/build folders, reject symlinks and paths outside the workspace root, limit file size, and do not expose `.env` secrets. For Python code questions involving a named function or method, the `find_function_or_method` tool locates its exact definition in the specified file, then the agent reads that file to inspect the implementation. This tool is read-only and does not search outside the requested file.

Qwen is instructed to inspect relevant files before making edits and to edit only when the user asks for a change. The dispatcher passes tool names, descriptions, and argument schemas to Ollama, then routes selected calls. To add a tool, register it with `ToolDispatcher.register_tool(...)`.

Search results go directly to Qwen for interpretation and citation. The agent has no separate evidence evaluator or page-fetch stage.

## Examples and checks

Call Tavily directly without starting Ollama or the agent:

```bash
.venv/bin/python -m examples.tavily_example "claude code functionality" --domain anthropic.com
```

Run research examples or the live smoke check:

```bash
.venv/bin/python -m examples.research_examples --limit 2
.venv/bin/python -m examples.smoke_test
```

Run the local tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The examples and smoke check make live Ollama and web requests.

## Troubleshooting

- **Cannot connect to Ollama:** start the Ollama app and check `OLLAMA_HOST`.
- **Model not installed:** run `ollama pull qwen3:8b`, or set `OLLAMA_MODEL` to a name shown by `ollama list`.
- **Tavily returns no results:** check `TAVILY_API_KEY`; domain-specific searches retry without the domain restriction if empty.
- **Qwen does not call a tool:** confirm the model supports tool calling and that the tool has a clear description in the dispatcher.
