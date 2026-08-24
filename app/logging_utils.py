from __future__ import annotations

from dataclasses import dataclass, field
import json


@dataclass
class Trace:
    session_id: str
    user_message: str
    history_snapshot: str = ""
    retrieved: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_response: str = ""
    handoff: bool = False

    def print_debug(self) -> None:
        print("--- DEBUG TRACE ---")
        print(json.dumps({
            "session_id": self.session_id,
            "user_message": self.user_message,
            "history": self.history_snapshot,
            "retrieved": self.retrieved,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "final_response": self.final_response,
            "handoff": self.handoff,
        }, indent=2, ensure_ascii=False))
        print("-------------------")
