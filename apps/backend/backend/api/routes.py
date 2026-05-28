import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.config import settings
from backend.graph.agent_graph import get_agent_graph

logger = logging.getLogger(__name__)

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
    session_id = request.session_id or str(uuid.uuid4())

    # Run the synchronous LangGraph in a thread pool so the event loop
    # stays free to handle other concurrent requests.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: get_agent_graph().invoke(
            {
                "user_input": request.message,
                "messages": [HumanMessage(content=request.message)],
            },
            config={"configurable": {"thread_id": session_id}},
        ),
    )

    latency = int((time.time() - start) * 1000)
    return ChatResponse(
        session_id=session_id,
        response=result.get("final_answer", ""),
        cached=result.get("cached", False),
        latency_ms=latency,
        validation_flag=result.get("validation_flag"),
    )


async def _stream_response(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """Stream tokens in real time via get_agent_graph().astream_events().

    Intercepts:
      - on_chat_model_stream  → token-level LLM output from generate_reply
      - on_tool_start/end     → tool execution visibility for the frontend
      - on_chain_start/end    → retry detection (validation loop)
      - on_chain_end          → final metadata after update_memory

    Only tokens from inside the generate_reply node are forwarded to the user.
    Tokens from auxiliary LLM calls (city extraction, summarization, validation)
    are filtered out by tracking whether we're inside the generate_reply node.
    """
    initial_state = {
        "user_input": message,
        "messages": [HumanMessage(content=message)],
    }

    config = {"configurable": {"thread_id": session_id}}
    in_generate_reply = False
    token_count = 0

    async def event_generator():
        nonlocal in_generate_reply, token_count

        async for event in get_agent_graph().astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chain_start" and name == "generate_reply":
                in_generate_reply = True
                if token_count > 0:
                    # Retry: validation flagged the previous answer, generate_reply
                    # is running again with the strict prompt. Signal frontend to
                    # clear the previous partial content.
                    token_count = 0
                    yield f"data: {json.dumps({'type': 'retry'})}\n\n"

            elif kind == "on_chain_end" and name == "generate_reply":
                in_generate_reply = False

            elif kind == "on_chat_model_stream" and in_generate_reply:
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    token_count += len(chunk.content)
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            elif kind == "on_tool_start":
                tool_input = event["data"].get("input", {})
                # Redact potentially long inputs for display
                safe_input = {k: str(v)[:100] for k, v in tool_input.items()} if isinstance(tool_input, dict) else str(tool_input)[:100]
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': name, 'input': safe_input})}\n\n"

            elif kind == "on_tool_end":
                tool_output = str(event["data"].get("output", ""))[:200]
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': name, 'output': tool_output})}\n\n"

        # Graph completed — pull final state for metadata (use async version
        # since the checkpointer is an AsyncPostgresSaver).
        final_state = await get_agent_graph().aget_state(config=config)
        metadata = {"type": "done"}
        if final_state and final_state.values:
            metadata["cached"] = final_state.values.get("cached", False)
            metadata["validation_flag"] = final_state.values.get("validation_flag")
            metadata["intent"] = final_state.values.get("intent")
        yield f"data: {json.dumps(metadata)}\n\n"

    try:
        async for frame in event_generator():
            yield frame
    except Exception:
        logger.exception("[stream] astream_events failed")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming failed. Please retry.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_response(request.message, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
