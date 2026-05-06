# Architecture Overview

## System Context

```
+-----------+      HTTP/WebSocket      +----------------+      SQL/pgvector      +------------+
|  Frontend |  <---------------------> |  Backend (API) |  <------------------>  |  Postgres  |
|  (Future) |                         |  FastAPI       |                        |  +pgvector |
+-----------+                         +----------------+                        +------------+
                                             |  ^
                                             |  | tools
                                             v  |
                                        +----------------+
                                        |  AI Agent      |
                                        |  - ReAct       |
                                        |  - Memory      |
                                        |  - RAG         |
                                        +----------------+
```

## Backend Modules

| Module | Status | Responsibility |
|--------|--------|----------------|
| `backend.agent` | **Implemented** | ReAct-style agent with tool calling, semantic cache, and resilience patterns |
| `backend.memory` | **Implemented** | JSON-based persistent conversation history |
| `backend.retrieval` | **Implemented** | Hybrid search: pgvector dense + BM25 sparse + LLM re-rank |
| `backend.resilience` | **Implemented** | Circuit breakers, retry with backoff, graceful fallbacks |
| `backend.db.setup` | **Implemented** | PostgreSQL orders table setup |
| `backend.db.vector_setup` | **Implemented** | pgvector policy embeddings setup |
| `backend.db.migrate_pgvector` | **Implemented** | One-time SQLite/Chroma -> PostgreSQL migration |
| `backend.api` | **Planned** | FastAPI REST and WebSocket endpoints |

## Data Flow

1. **User Query** -> API -> `run_agent_with_cache()`
2. **Typo Correction** -> LLM with circuit breaker
3. **Cache Check** -> pgvector semantic similarity search
4. **Memory Load** -> recent conversation from JSON store
5. **Tool Use** -> `order_status_tool` (PostgreSQL) or `policy_retriever_tool` (hybrid RAG)
6. **Response** -> cached, stored to memory, returned to user

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | GLM-4-Flash via OpenAI-compatible API |
| Embeddings | Zhipu AI embedding-2 |
| Vector DB | PostgreSQL + pgvector |
| Relational DB | PostgreSQL |
| Agent Framework | LangChain (ReAct) |
| API (planned) | FastAPI |
| Frontend (planned) | React + TypeScript + Tailwind |

## Roadmap

See `apps/backend/backend/api/README.md`, `apps/frontend/README.md`, and `infra/*/README.md` for detailed implementation plans of unbuilt modules.
