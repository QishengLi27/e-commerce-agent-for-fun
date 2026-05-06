import time
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.agent import run_agent_with_cache

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start = time.time()
    response_text = run_agent_with_cache(request.message)
    latency = int((time.time() - start) * 1000)
    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        cached=False,  # TODO: expose cache hit from agent
        latency_ms=latency,
    )


async def _stream_response(message: str) -> AsyncGenerator[dict, None]:
    """Run the agent and yield SSE events word-by-word."""
    response_text = run_agent_with_cache(message)

    # Stream word by word for a nice typing effect
    words = response_text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield {"event": "message", "data": chunk}
        await asyncio.sleep(0.03)  # Small delay for visual effect

    yield {"event": "done", "data": "[DONE]"}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return EventSourceResponse(_stream_response(request.message))
