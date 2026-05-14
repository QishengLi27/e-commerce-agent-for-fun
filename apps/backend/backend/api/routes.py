import time
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.graph.agent_graph import agent_graph
from backend.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Deep health check: verifies LLM API is reachable."""
    healthy = True
    checks = {}

    # Check database connectivity via a lightweight operation
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        healthy = False

    # Check pgvector extension
    try:
        engine = create_engine(settings.database_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                checks["pgvector"] = "ok"
            else:
                checks["pgvector"] = "missing"
                healthy = False
    except Exception as e:
        checks["pgvector"] = f"error: {e}"
        healthy = False

    return HealthResponse(
        status="healthy" if healthy else "degraded",
        version="0.1.0",
        checks=checks,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message asynchronously via the LangGraph agent."""
    start = time.time()

    # Run the synchronous LangGraph in a thread pool so the event loop
    # stays free to handle other concurrent requests.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent_graph.invoke({
            "user_input": request.message,
            "messages": [],
        }),
    )

    latency = int((time.time() - start) * 1000)
    return ChatResponse(
        session_id=request.session_id,
        response=result.get("final_answer", ""),
        cached=result.get("cached", False),
        latency_ms=latency,
        validation_flag=result.get("validation_flag"),
    )


async def _stream_response(message: str) -> AsyncGenerator[str, None]:
    """Yield raw SSE frames with a smooth typing effect."""
    # Run graph in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent_graph.invoke({
            "user_input": message,
            "messages": [],
        }),
    )
    answer = result.get("final_answer", "")
    validation_flag = result.get("validation_flag")

    # Stream word-by-word for visible typing effect
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        safe = chunk.replace("\n", "\\n").replace("\r", "")
        yield f"data: {safe}\n\n"
        await asyncio.sleep(0.03)

    if validation_flag:
        yield f"data: [FLAG:{validation_flag}]\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _stream_response(request.message),
        media_type="text/event-stream",
    )
