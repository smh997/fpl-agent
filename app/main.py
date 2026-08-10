"""FastAPI app exposing the FPL agent over HTTP.

POST /chat runs the agent loop (app.agent.ask_with_fallback) for a
natural-language question and returns its full result -- which tools were
called, their arguments and results per iteration, the final answer,
iteration count, and whether the answer came from a live call or a cached
demo fixture (demo_mode).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import agent

app = FastAPI(title="FPL Agent")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_iterations: int = Field(5, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    """Plain liveness check -- no LLM or tool calls, just confirms the process is up."""
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """Answer a natural-language FPL question via the tool-calling agent loop."""
    try:
        return agent.ask_with_fallback(request.question, request.max_iterations)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
