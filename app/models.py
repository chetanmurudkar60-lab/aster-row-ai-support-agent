from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    content: str
    filename: str
    heading: str
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    chunk: Chunk
    semantic_score: float
    authority_score: float
    final_score: float


@dataclass
class AgentResult:
    response: str
    handoff: bool = False
    trace: Any = None
