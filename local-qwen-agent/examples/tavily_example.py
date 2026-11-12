"""Call Tavily Search directly and print results without the agent or MCP server."""

from __future__ import annotations

import argparse

from qwen_agent.tools.tavily_api import RECENCY_TO_TIME_RANGE, search_tavily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="claude code functionality", help="Text to search for")
    parser.add_argument("--domain", help="Restrict results to a domain, e.g. anthropic.com")
    parser.add_argument("--recency", choices=RECENCY_TO_TIME_RANGE, help="Filter by publication recency")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    try:
        response = search_tavily(
            args.query,
            max_results=args.max_results,
            domain=args.domain,
            recency=args.recency,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Tavily search failed: {exc}\n")

    results = response["results"]
    print(f"Query: {response.get('query', args.query)}")
    print(f"Results: {len(results)}\n")
    if not results:
        print("Tavily returned no results.")
        return
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        print(f"{index}. {item.get('title') or '(untitled)'}")
        print(f"   URL: {item.get('url') or '(missing)'}")
        print(f"   Published: {item.get('published_date') or 'unknown'}")
        print(f"   Snippet: {item.get('content') or '(no snippet)'}\n")


if __name__ == "__main__":
    main()
