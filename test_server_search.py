"""Direct tests for Tavily-backed MCP web_search."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
import server


class FakeResponse:
    def __init__(self, status_code=200, data=None, malformed=False):
        self.status_code = status_code
        self.data = data
        self.malformed = malformed
        self.is_success = 200 <= status_code < 300

    def json(self):
        if self.malformed:
            raise ValueError("invalid json")
        return self.data


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse(data={"results": []})
        self.error = error
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, *, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.error:
            raise self.error
        return self.response


def tavily_result(query="test query", published_date="Wed, 11 Nov 2026 12:00:00 GMT"):
    return {
        "query": query,
        "results": [{
            "title": "Example result",
            "url": "https://example.org/article",
            "content": "A useful result snippet.",
            "published_date": published_date,
        }],
    }


class TavilySearchTests(unittest.TestCase):
    def search_with(self, query, response=None, *, domain=None, recency=None, error=None, key="test-key"):
        client = FakeClient(response=response, error=error)
        with patch.dict(os.environ, {"TAVILY_API_KEY": key}, clear=False), patch.object(
            server.httpx, "Client", return_value=client
        ):
            result = server.web_search(query, domain=domain, recency=recency)
        return result, client

    def test_claude_code_query_is_sent_unchanged(self):
        result, client = self.search_with("claude code", FakeResponse(data=tavily_result("claude code")))
        self.assertEqual(client.calls[0]["json"]["query"], "claude code")
        self.assertEqual(result["results"][0]["source"], "tavily")

    def test_focused_claude_query_is_sent_unchanged(self):
        _result, client = self.search_with(
            "claude code functionality", FakeResponse(data=tavily_result("claude code functionality"))
        )
        self.assertEqual(client.calls[0]["json"]["query"], "claude code functionality")

    def test_domain_uses_tavily_include_domains_parameters(self):
        _result, client = self.search_with(
            "claude code functionality",
            FakeResponse(data=tavily_result("claude code functionality")),
            domain="anthropic.com",
        )
        payload = client.calls[0]["json"]
        self.assertEqual(payload["query"], "claude code functionality")
        self.assertEqual(payload["include_domains"], ["anthropic.com"])
        self.assertEqual(payload["include_domains_mode"], "restrict")

    def test_kafka_query(self):
        result, client = self.search_with("kafka replication", FakeResponse(data=tavily_result("kafka replication")))
        self.assertEqual(client.calls[0]["json"]["query"], "kafka replication")
        self.assertEqual(result["results"][0]["snippet"], "A useful result snippet.")

    def test_current_information_maps_recency_and_published_date(self):
        result, client = self.search_with(
            "latest technology news", FakeResponse(data=tavily_result("latest technology news")), recency="day"
        )
        self.assertEqual(client.calls[0]["json"]["time_range"], "day")
        self.assertEqual(result["results"][0]["published_at"], "2026-11-11T12:00:00+00:00")

    def test_hour_recency_uses_day_level_tavily_filter_with_warning(self):
        result, client = self.search_with("recent news", FakeResponse(data=tavily_result()), recency="hour")
        self.assertEqual(client.calls[0]["json"]["time_range"], "day")
        self.assertTrue(any("day-level" in warning for warning in result["warnings"]))

    def test_missing_api_key_returns_actionable_warning_without_request(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(server.httpx, "Client") as client:
            result = server.web_search("claude code")
        client.assert_not_called()
        self.assertEqual(result["results"], [])
        self.assertIn("TAVILY_API_KEY", result["warnings"][0])

    def test_429_and_other_http_failures_return_warnings(self):
        for status in (429, 401, 500):
            with self.subTest(status=status):
                result, _client = self.search_with("test", FakeResponse(status_code=status, data={}))
                self.assertEqual(result["results"], [])
                self.assertIn(str(status), result["warnings"][0])

    def test_timeout_and_connection_errors_are_handled(self):
        for error, expected in (
            (httpx.ReadTimeout("timeout"), "timed out"),
            (httpx.ConnectError("offline"), "connect to Tavily"),
        ):
            with self.subTest(error=type(error).__name__):
                result, _client = self.search_with("test", error=error)
                self.assertEqual(result["results"], [])
                self.assertIn(expected, result["warnings"][0])

    def test_malformed_json_shape_and_empty_results_are_handled(self):
        malformed, _ = self.search_with("test", FakeResponse(malformed=True))
        bad_shape, _ = self.search_with("test", FakeResponse(data={"unexpected": []}))
        empty, _ = self.search_with("test", FakeResponse(data={"results": []}))
        self.assertIn("malformed JSON", malformed["warnings"][0])
        self.assertIn("unexpected response shape", bad_shape["warnings"][0])
        self.assertIn("no usable results", empty["warnings"][0])

    def test_malformed_result_entries_are_skipped(self):
        result, _ = self.search_with("test", FakeResponse(data={"results": ["bad", {"title": "No URL"}]}))
        self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
