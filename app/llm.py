from __future__ import annotations

from typing import Sequence
import time

try:
    from google import genai
    from google.genai import types
    from google.genai import errors
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    genai = None
    types = None
    errors = None

from .config import CHAT_MODEL, EMBEDDING_MODEL, GEMINI_API_KEY
from .prompts import SYSTEM_PROMPT


class GeminiClient:
    """Small provider wrapper so the rest of the agent is provider-agnostic."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key) if (self.api_key and genai is not None) else None

    @property
    def configured(self) -> bool:
        return self.client is not None

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.client:
            raise RuntimeError("Gemini API key is not configured")
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=list(texts),
        )
        return [list(item.values) for item in response.embeddings]

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def generate(self, user_prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Gemini API key is not configured")

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                return text
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status != 503 and "503" not in str(exc):
                    raise
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError("Gemini is temporarily unavailable after 3 attempts") from last_error
