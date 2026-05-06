# API Layer (Not Yet Implemented)

## Overview

This module will expose the e-commerce support agent as a RESTful (and potentially WebSocket) API so that frontend clients and third-party services can interact with it.

## Planned Tech Stack

- **Framework**: FastAPI (Python) — async-native, automatic OpenAPI docs, great DX
- **Real-time**: Socket.IO or native WebSocket for streaming agent responses
- **Validation**: Pydantic v2 for request/response schemas
- **Auth**: OAuth2 / JWT bearer tokens (future)

## Planned Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message and receive an agent response |
| `POST` | `/chat/stream` | Stream agent response tokens (SSE or WebSocket) |
| `GET`  | `/health` | Health check including DB and vector store connectivity |
| `POST` | `/feedback` | Submit thumbs-up/down feedback for a response |
| `GET`  | `/memory/{session_id}` | Retrieve conversation history for a session |
| `DELETE`| `/memory/{session_id}` | Clear conversation history (GDPR compliance) |
| `GET`  | `/metrics` | Basic metrics: cache hit rate, latency histogram, circuit breaker state |

## Request/Response Schema

### POST /chat

**Request:**
```json
{
  "session_id": "uuid-v4",
  "message": "What is the return policy?"
}
```

**Response:**
```json
{
  "session_id": "uuid-v4",
  "response": "You can return most items within 30 days...",
  "sources": [
    {"type": "policy", "id": "return_policy", "snippet": "..."}
  ],
  "cached": false,
  "latency_ms": 420
}
```

## Implementation Checklist

- [ ] Set up FastAPI app with lifespan context manager
- [ ] Define Pydantic models in `packages/shared/` (or duplicate temporarily)
- [ ] Create `/chat` endpoint wiring `run_agent_with_cache`
- [ ] Add session-based memory (replace global `memory_store.json`)
- [ ] Add structured logging middleware
- [ ] Add rate limiting per session/IP
- [ ] Write basic integration tests with `httpx` + `pytest`
- [ ] Dockerize the API service

## Notes

- Keep the API stateless where possible; session memory should eventually move to Redis or PostgreSQL.
- Consider adding an `/admin/reload-policies` endpoint for dynamic policy updates without restart.
