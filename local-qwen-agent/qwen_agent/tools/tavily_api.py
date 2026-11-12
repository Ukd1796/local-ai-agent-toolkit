"""Small direct client for Tavily's Search API."""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

from ..config import AGENT_DIR


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
RECENCY_TO_TIME_RANGE = {
    "hour": "day",  # Tavily's publication-date filter is day-granularity.
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}


def search_tavily(
    query: str,
    *,
    max_results: int = 5,
    domain: str | None = None,
    recency: str | None = None,
) -> dict:
    """Run one direct Tavily Search API request and return its JSON response."""
    load_dotenv(AGENT_DIR / ".env")
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing. Set it in local-qwen-agent/.env or your environment.")
    if not query.strip():
        raise ValueError("query cannot be empty")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20")
    if recency is not None and recency not in RECENCY_TO_TIME_RANGE:
        raise ValueError("recency must be hour, day, week, month, or year")

    payload: dict[str, object] = {
        "query": query.strip(),
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_published_date": True,
    }
    if domain:
        payload["include_domains"] = [domain.strip().lower()]
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
    except httpx.TimeoutException as exc:
        raise RuntimeError("Tavily request timed out.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not connect to Tavily. Check internet connectivity.") from exc

    if response.status_code == 429:
        raise RuntimeError("Tavily rate-limited this request (HTTP 429). Retry later.")
    if not response.is_success:
        raise RuntimeError(f"Tavily returned HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Tavily returned malformed JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise RuntimeError("Tavily returned an unexpected response format.")
    return data
