# Local Web MCP

A small Model Context Protocol (MCP) server that exposes public web search, page reading, current time, and safe arithmetic as tools. It uses the stdio transport, which is how desktop MCP hosts commonly launch local servers.

To connect these tools to a local Ollama/Qwen model through an agent, see [local-qwen-agent](local-qwen-agent/README.md). That is a separate project and launches this server as an MCP subprocess.

## Requirements

- Python 3.10 or newer
- `uv` (recommended) or `pip`
- An MCP-capable host connected to your local Ollama model

The MCP server is the tool provider; it does not load or call `qwen3:8b` itself. Connect this server through a host that can use both Ollama and MCP tools (for example, an MCP-enabled chat UI).

## Install and run

From this directory:

```bash
uv sync
```

`uv sync` creates the local `.venv/` virtual environment and installs the dependencies listed in `pyproject.toml`. You do not need to send or copy that folder; it is local and ignored by Git. `uv run` uses this environment automatically.

Do not start the server by running `uv run python server.py` in a plain terminal. The server speaks MCP over stdin/stdout and needs an MCP client to send it requests. Running it directly in a terminal will now print a short explanation and exit.

### Test the MCP connection

```bash
uv run --with mcp[cli] mcp dev server.py
```

This starts the MCP Inspector for the server. Open its Tools view and call `get_current_time` with `timezone="Asia/Kolkata"`, then call `calculator` with `expression="(23 * 7) / 2"`. You can also test `web_search` and `fetch_url`. Alternatively, configure your MCP host to launch `uv run --directory /Users/ujjwalkumar/learningAI python server.py` as shown below.

## MCP host configuration

For hosts that read a `mcpServers` configuration, add the following entry and update the directory if you move this project:

```json
{
  "mcpServers": {
    "local-web": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/ujjwalkumar/learningAI", "python", "server.py"]
    }
  }
}
```

Restart the host after saving its configuration. You should then see `web_search`, `fetch_url`, `get_current_time`, and `calculator` as available tools (the legacy `read_webpage` name remains available too).

## Tools

- `web_search(query, max_results=5, recency=None, domain=None)` returns up to five Tavily candidate results with title, URL, snippet, source, and publication date when available. It requires `TAVILY_API_KEY`. `domain` is sent as a Tavily domain restriction; `recency` accepts `hour`, `day`, `week`, `month`, or `year` (`hour` uses Tavily's day-level date filter).
- `read_webpage(url)` fetches an HTML or plain-text page and returns its title and readable text, capped at 20,000 characters.
- `fetch_url(url, max_chars=12000)` is an alias for bounded page extraction.
- `get_current_time(timezone="Asia/Kolkata")` returns the current date/time in an IANA timezone.
- `calculator(expression)` evaluates basic arithmetic using a restricted AST parser, not Python `eval`.

`read_webpage` accepts only HTTP(S) URLs and rejects URLs resolving to non-public IP addresses. Redirects must be followed manually by passing the destination URL back to the tool.

## Notes

- Search uses Tavily's Search API and may be rate-limited or unavailable. Results are candidates, not verified evidence; use `fetch_url` to inspect pages. The API key is read from `TAVILY_API_KEY` and is never returned in tool output or logged.
- Network requests need outbound internet access.
- The MCP Python SDK v1 line is pinned because v2 currently has breaking API changes relative to this example.
