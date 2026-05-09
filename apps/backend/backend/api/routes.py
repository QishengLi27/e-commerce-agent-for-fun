import time
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.agent import run_agent_with_cache, stream_agent_response

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


async def _stream_response(message: str) -> AsyncGenerator[str, None]:
    """Yield raw SSE frames with real LLM tokens as they arrive."""
    async for token in stream_agent_response(message):
        # Escape newlines in token so SSE format stays valid
        safe_token = token.replace("\n", "\\n").replace("\r", "")
        yield f"data: {safe_token}\n\n"
        # Small delay to force HTTP flush and create visible typing effect
        await asyncio.sleep(0.03)

    yield f"data: [DONE]\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _stream_response(request.message),
        media_type="text/event-stream",
    )
