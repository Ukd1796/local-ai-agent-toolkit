"""Small evidence ledger kept separate from task state and chat history."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResearchState:
    question: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    claims: list[dict[str, str]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    sufficient: bool = False


class ResearchStateManager:
    """Record only source metadata and snippets returned by completed searches."""

    def __init__(self) -> None:
        self.state = ResearchState()

    def reset(self) -> None:
        self.state = ResearchState()

    def record_search(self, question: str, result: str) -> None:
        if question != self.state.question:
            self.state.sources.clear()
            self.state.claims.clear()
        self.state.question = question
        try:
            payload: Any = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        results = payload.get("results", []) if isinstance(payload, dict) else []
        for item in results:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            source = {key: str(item.get(key) or "") for key in ("title", "url")}
            if source not in self.state.sources:
                self.state.sources.append(source)
            snippet = str(item.get("snippet") or "").strip()
            claim = {"text": snippet, "source": source["url"]}
            if snippet and claim not in self.state.claims:
                self.state.claims.append(claim)
        self.state.sufficient = bool(self.state.sources)
        self.state.gaps = [] if self.state.sufficient else ["No usable sources have been verified yet."]

    def state_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": "Research Evidence State (verified tool-returned evidence; separate from task state and history):\n"
            + json.dumps(asdict(self.state), ensure_ascii=False),
        }
