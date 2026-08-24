from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    turns: list[tuple[str, str]] = field(default_factory=list)
    last_order_id: str | None = None
    last_order_status: str | None = None

    def history_as_text(self) -> str:
        if not self.turns:
            return "(no prior turns)"
        return "\n".join(
            f"{role}: {content}" for role, content in self.turns[-8:]
        )

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append((role, content))
        self.turns = self.turns[-8:]


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session(session_id))
