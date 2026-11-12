"""Small async wrapper around the official Ollama Python client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ollama import AsyncClient

from .config import Settings


class LocalOllama:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncClient(host=settings.ollama_host)

    async def check_model(self) -> None:
        """Check that Ollama responds and the configured model is present."""
        response = await self.client.list()
        models = getattr(response, "models", [])
        names = {
            getattr(model, "model", None) or getattr(model, "name", None)
            for model in models
        }
        if self.settings.ollama_model not in names:
            available = ", ".join(sorted(name for name in names if name)) or "none"
            raise RuntimeError(
                f"Model {self.settings.ollama_model!r} is not installed in Ollama. "
                f"Available models: {available}. Run: ollama pull {self.settings.ollama_model}"
            )

    async def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        think: bool = True,
        format: str | dict[str, Any] | None = None,
    ) -> Any:
        options: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            options["tools"] = tools
        options["think"] = think
        if format is not None:
            options["format"] = format
        return await self.client.chat(**options)

    async def stream_chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        *,
        think: bool = True,
    ) -> AsyncIterator[Any]:
        """Yield Ollama chat chunks, including separate thinking and answer text."""
        options: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": True,
            "think": think,
        }
        if tools:
            options["tools"] = tools
        stream = await self.client.chat(**options)
        async for chunk in stream:
            yield chunk
