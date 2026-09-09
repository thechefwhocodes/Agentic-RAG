"""HTTP API: POST /api/chat -> {"answer": ..., "sources": [...]}.

Run with: uvicorn src.api:app --port 8000
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.agent import Agent
from src.models import ChatRequest, ChatResponse

app = FastAPI()
agent = Agent()


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = agent.ask(request.question)
    return ChatResponse(
        answer=answer.text,
        sources=answer.evidence,
        latency_s=answer.latency_s,
        cost_usd=answer.cost_usd,
    )
