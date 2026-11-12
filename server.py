"""A small stdio MCP server that gives a local model web search and page reading."""

from __future__ import annotations

import ipaddress
import ast
import logging
import math
import os
import re
import operator
import socket
import sys
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("local-web")
MAX_PAGE_CHARS = 20_000
USER_AGENT = "local-web-mcp/0.1 (page text reader)"
MAX_SEARCH_RESULTS = 5
RECENCY_TO_TIME_RANGE = {"hour": "day", "day": "day", "week": "week", "month": "month", "year": "year"}
SEARCH_TIMEZONE = os.getenv("USER_TIMEZONE", "Asia/Kolkata")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
CALCULATOR_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _normalize_search_results(raw_results: list[dict], limit: int) -> list[dict[str, str | None]]:
    normalized = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = " ".join(str(item.get("title") or "").split())
        snippet = " ".join(str(item.get("content") or "").split())
        if not title or not url:
            continue
        parsed = urlparse(url)
        dedupe_url = f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}".casefold()
        dedupe_title = title.casefold()
        if dedupe_url in seen_urls or dedupe_title in seen_titles:
            continue
        seen_urls.add(dedupe_url)
        seen_titles.add(dedupe_title)
        published = _parse_published_at(item.get("published_date"))
        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet[:1200],
                "source": "tavily",
                "published_at": published,
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


def _parse_published_at(value: object) -> str | None:
    if not isinstance(value, (str, int)) or not value:
        return None
    try:
        if isinstance(value, int):
            return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = parsedate_to_datetime(value)
        return parsed.isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError, TypeError):
        return None


def _filter_recent(
    results: list[dict[str, str | None]], recency: str, timezone_name: str
) -> tuple[list[dict[str, str | None]], int]:
    days = {"hour": 1, "day": 1, "week": 7, "month": 31, "year": 366}[recency]
    zone = ZoneInfo(timezone_name)
    now = datetime.now(zone)
    cutoff = now - timedelta(days=days)
    latest_allowed = now + timedelta(hours=2)
    retained = []
    unverified = 0
    for result in results:
        published = result["published_at"]
        if not published:
            unverified += 1
            retained.append(result)
            continue
        try:
            parsed = datetime.fromisoformat(published)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=zone)
            local_published = parsed.astimezone(zone)
            if cutoff <= local_published <= latest_allowed:
                retained.append(result)
        except ValueError:
            # The normalization layer normally makes this impossible; unknown
            # formats remain visible but explicitly undated.
            result["published_at"] = None
            unverified += 1
            retained.append(result)
    return retained, unverified


def _validate_public_url(url: str) -> str:
    """Reject non-web URLs and destinations that resolve to local/private IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("Requests to local or non-public network addresses are not allowed")
    return url


@mcp.tool()
def web_search(
    query: str,
    max_results: int = 5,
    recency: str | None = None,
    domain: str | None = None,
) -> dict:
    """Search the web for candidate sources.

    Search results are candidates, not verified evidence. Use fetch_url to inspect pages.
    Optional domain restricts results to a host or its subdomains. Recency accepts hour, day,
    week, month, or year.
    """
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")
    if recency is not None and recency not in RECENCY_TO_TIME_RANGE:
        raise ValueError("recency must be one of: hour, day, week, month, year")
    if domain is not None:
        domain = domain.strip().lower().removeprefix("https://").removeprefix("http://").strip("/")
        if (
            not domain
            or "." not in domain
            or not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]{1,5})?", domain)
        ):
            raise ValueError("domain must be a fully qualified hostname, such as docs.example.com")

    warnings = []
    if recency == "hour":
        warnings.append("The search API supports day-level publish-date filtering, not hour-level filtering.")

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {
            "query": query,
            "requested_recency": recency,
            "applied_recency": "day" if recency == "hour" else recency,
            "results": [],
            "warnings": ["TAVILY_API_KEY is not set; add it to the agent environment and restart the MCP server."],
        }

    payload: dict[str, object] = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_published_date": True,
    }
    if domain:
        payload["include_domains"] = [domain]
        payload["include_domains_mode"] = "restrict"
    if recency:
        payload["time_range"] = RECENCY_TO_TIME_RANGE[recency]

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            response = client.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        if response.status_code == 429:
            return {
                "query": query,
                "requested_recency": recency,
                "applied_recency": "day" if recency == "hour" else recency,
                "results": [],
                "warnings": ["Tavily rate-limited the search request (HTTP 429). Retry later."],
            }
        if not response.is_success:
            logging.getLogger(__name__).warning("Tavily search failed with HTTP %d", response.status_code)
            return {
                "query": query,
                "requested_recency": recency,
                "applied_recency": "day" if recency == "hour" else recency,
                "results": [],
                "warnings": [f"Tavily search failed with HTTP {response.status_code}."],
            }
        try:
            data = response.json()
        except ValueError:
            return {
                "query": query,
                "requested_recency": recency,
                "applied_recency": "day" if recency == "hour" else recency,
                "results": [],
                "warnings": ["Tavily returned malformed JSON."],
            }
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return {
                "query": query,
                "requested_recency": recency,
                "applied_recency": "day" if recency == "hour" else recency,
                "results": [],
                "warnings": ["Tavily returned an unexpected response shape."],
            }
        results = _normalize_search_results(data["results"], max_results)
        unverified = 0
        if recency and results:
            results, unverified = _filter_recent(results, recency, SEARCH_TIMEZONE)
            if unverified:
                warnings.append(f"{unverified} result(s) have no verifiable publication date; freshness is unverified.")
        results = results[:max_results]
        if not results:
            warnings.append("Tavily returned no usable results.")
    except httpx.TimeoutException:
        logging.getLogger(__name__).warning("Tavily search timed out")
        results = []
        warnings.append("Tavily search timed out. Retry later.")
    except httpx.ConnectError:
        logging.getLogger(__name__).warning("Could not connect to Tavily Search API")
        results = []
        warnings.append("Could not connect to Tavily. Check internet connectivity and retry.")
    except httpx.RequestError as exc:
        logging.getLogger(__name__).warning("Tavily request failed: %s", type(exc).__name__)
        results = []
        warnings.append("The Tavily search request failed. Check connectivity and retry.")

    return {
        "query": query,
        "requested_recency": recency,
        "applied_recency": "day" if recency == "hour" else recency,
        "results": results,
        "warnings": warnings,
    }


@mcp.tool()
def read_webpage(url: str) -> dict[str, str]:
    """Fetch a specific public URL and extract readable page text.

    Use after web_search when detailed information from a particular result is needed.
    """
    safe_url = _validate_public_url(url)
    with httpx.Client(
        timeout=httpx.Timeout(15.0),
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = client.get(safe_url)
        if 300 <= response.status_code < 400:
            raise ValueError("The page redirected; call read_webpage with the destination URL")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return {"url": safe_url, "title": title, "text": text[:MAX_PAGE_CHARS]}


@mcp.tool()
def fetch_url(url: str, max_chars: int = 12_000) -> dict[str, str]:
    """Fetch and extract a specific public URL, usually after web_search.

    Use this when details from a particular page are needed. It does not search the web.
    """
    if not 1 <= max_chars <= MAX_PAGE_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_PAGE_CHARS}")
    page = read_webpage(url)
    page["text"] = page["text"][:max_chars]
    return page


@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> dict[str, str]:
    """Get the actual current date and time for a timezone.

    Use for current time, date, day, or timezone questions. Do not use web_search for these.
    The timezone must be an IANA timezone such as Asia/Kolkata or Europe/London.
    """
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone}") from exc
    now = datetime.now(zone)
    return {
        "timezone": timezone,
        "timezone_name": now.tzname() or timezone,
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
    }


def _calculate(node: ast.AST, depth: int = 0) -> int | float:
    if depth > 24:
        raise ValueError("Expression is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _calculate(node.operand, depth + 1)
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp) and type(node.op) in CALCULATOR_OPERATORS:
        left = _calculate(node.left, depth + 1)
        right = _calculate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent magnitude must be 12 or less")
        value = CALCULATOR_OPERATORS[type(node.op)](left, right)
    else:
        raise ValueError("Only numeric arithmetic expressions are allowed")

    if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("Calculation result is outside the supported range")
    return value


@mcp.tool()
def calculator(expression: str) -> dict[str, str | int | float]:
    """Calculate arithmetic safely, without running Python code.

    Use for arithmetic tasks instead of estimating or doing arithmetic mentally.
    Supports numbers, parentheses, +, -, *, /, //, %, and **.
    """
    expression = expression.strip()
    if not expression or len(expression) > 256:
        raise ValueError("Expression must contain 1 to 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _calculate(tree.body)
    except (SyntaxError, ArithmeticError, OverflowError) as exc:
        raise ValueError(f"Invalid arithmetic expression: {exc}") from exc
    return {"expression": expression, "result": result}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    if sys.stdin.isatty():
        print(
            "This is an MCP stdio server. Launch it from an MCP host or the MCP Inspector; "
            "running it directly in a terminal does not provide an MCP client.",
            file=sys.stderr,
        )
        return
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
