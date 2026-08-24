from __future__ import annotations

import os
import time
import uuid
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
MAX_RETRIES = 3


class OpenRouterLLMClient:
    """OpenAI-compatible OpenRouter client using the normalized Agent interface."""

    def __init__(self, api_key: str | None = None, model: str = MODEL_NAME):
        from openai import OpenAI

        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        self.model = model

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(self._convert_messages(messages))

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": 0.1,
            "max_tokens": 900,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"}),
                    },
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = "auto"

        response = self._generate_with_retry(**kwargs)
        return self._normalize_response(response)

    def _generate_with_retry(self, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                text = str(exc).lower()
                transient = (
                    status in {429, 500, 502, 503, 504}
                    or any(code in text for code in ("429", "500", "502", "503", "504"))
                    or "rate limit" in text
                    or "temporarily unavailable" in text
                )
                if not transient or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        raise last_error

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            # Normalized tool blocks are converted to OpenAI-compatible messages.
            if role == "assistant":
                text_parts = []
                tool_calls = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        import json
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                converted.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) or None,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                })
            elif role == "user":
                tool_blocks = [b for b in content if b.get("type") == "tool_result"]
                if tool_blocks:
                    for block in tool_blocks:
                        converted.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        })
                else:
                    converted.append({"role": "user", "content": str(content)})
            else:
                converted.append({"role": role, "content": str(content)})
        return converted

    @staticmethod
    def _normalize_response(response) -> dict:
        message = response.choices[0].message
        content = []

        if message.content:
            content.append({"type": "text", "text": message.content})

        for call in (getattr(message, "tool_calls", None) or []):
            import json
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            content.append({
                "type": "tool_use",
                "id": call.id or f"openrouter-{uuid.uuid4().hex}",
                "name": call.function.name,
                "input": arguments,
            })

        return {
            "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn",
            "content": content,
        }


class ScriptedMockLLMClient:
    def __init__(self, policy: Callable[[str, list[dict], list[dict]], dict]):
        self.policy = policy
        self.calls: list[dict] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        return self.policy(system, messages, tools)
