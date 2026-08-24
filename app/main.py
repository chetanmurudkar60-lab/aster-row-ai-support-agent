from fastapi import FastAPI
from pydantic import BaseModel
from .agent import SupportAgent

app = FastAPI(title="Aster & Row Reliable RAG Support Agent", version="1.0.0")
agent = SupportAgent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health():
    return {"status":"ok","model":"configured" if agent.client else "local-fallback"}

@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.respond(req.session_id, req.message)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "handoff": result.handoff,
        "tool_calls": result.tool_calls,
        "trace_id": result.trace_id,
    }
