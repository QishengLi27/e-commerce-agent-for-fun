import time
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.graph.agent_graph import agent_graph

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start = time.time()
    result = agent_graph.invoke({
        "user_input": request.message,
        "messages": [],
    })
    latency = int((time.time() - start) * 1000)
    return ChatResponse(
        session_id=request.session_id,
        response=result.get("final_answer", ""),
        cached=result.get("cached", False),
        latency_ms=latency,
    )


async def _stream_response(message: str) -> AsyncGenerator[str, None]:
    """Yield raw SSE frames with a smooth typing effect."""
    # Run graph and collect final answer
    result = agent_graph.invoke({
        "user_input": message,
        "messages": [],
    })
    answer = result.get("final_answer", "")

    # Stream word-by-word for visible typing effect
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        safe = chunk.replace("\n", "\\n").replace("\r", "")
        yield f"data: {safe}\n\n"
        await asyncio.sleep(0.03)

    yield "data: [DONE]\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _stream_response(request.message),
        media_type="text/event-stream",
    )
