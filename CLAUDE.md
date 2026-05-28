# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (one-shot)
bash scripts/dev-setup.sh

# Manual setup
cd apps/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL + pgvector (required infrastructure)
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16

# Initialize DB tables and vector store
python -m backend.db.setup
python -m backend.db.vector_setup

# Run the CLI agent
python main.py

# Run the API server (FastAPI on port 8000)
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Backend Coding Standards

When modifying code in `apps/backend/`, follow the standards defined in `apps/backend/AGENTS.md`:

- **ruff** for linting and formatting (enforced by pre-commit)
- **mypy** for static type checking (zero errors policy)
- Google-style docstrings for all public functions
- Full type annotations (`list[str]`, not `List` or bare `dict`)
- Single-responsibility functions (< 50 lines, < 4 parameters)
- No `print()` in modules (use `logging`); no bare `except:`

Always run before committing:
```bash
cd apps/backend
ruff check backend/ main.py tests/ --fix
ruff format backend/ main.py tests/
mypy backend/ main.py tests/
```

## Architecture

This is an e-commerce AI support agent monorepo. Only `apps/backend` is implemented; everything else is planned scaffolding.

### Two agent implementations coexist

- **`backend/agent.py`** — Legacy agent using LangChain's `create_agent()` (ReAct). Contains `AgentManager`, which holds lazy-singleton references to the LLM, pgvector cache store, circuit breaker, and memory. This is what `main.py` calls.
- **`backend/graph/`** — Newer LangGraph state machine with explicit nodes (`sanitize_input` → `classify_intent` → route to tool node → `generate_reply` → `update_memory`). The FastAPI routes (`backend/api/routes.py`) invoke this graph, not the legacy agent.

Both use the same underlying tools, memory store, and LLM. The LangGraph version replaces the "black box" `create_agent()` with a deterministic DAG — intent classification is keyword-based rather than LLM-decided.

### Request flow (LangGraph path)

1. **`sanitize_input`** — typo correction + semantic cache check (skipped for weather queries)
2. **`classify_intent`** — keyword-based routing: `order` | `list_orders` | `policy` | `weather` | `unknown`
3. **Tool node** — calls the matched `@tool` function, stores result in state
4. **`generate_reply`** — LLM formats the tool result into a user-facing answer
5. **`update_memory`** — persists to JSON memory store + pgvector semantic cache

### Key modules

| Module | Purpose |
|--------|---------|
| `backend/agent.py` | Legacy `create_agent()` wrapper, cache, streaming |
| `backend/graph/agent_graph.py` | LangGraph `StateGraph` definition and compilation |
| `backend/graph/nodes.py` | All node functions and the `AgentState` TypedDict |
| `backend/memory.py` | JSON-file conversation history (sessionless, global) |
| `backend/retrieval.py` | Hybrid RAG: pgvector dense + BM25 sparse → RRF fusion → LLM re-rank |
| `backend/resilience.py` | Circuit breaker, retry with exponential backoff (tenacity), static fallbacks |
| `backend/tools/` | LangChain `@tool` functions: `order_status_tool`, `list_orders_tool`, `policy_retriever_tool`, `get_current_weather` |
| `backend/db/setup.py` | PostgreSQL orders table creation + mock data |
| `backend/db/vector_setup.py` | pgvector policy embedding setup |
| `backend/db/migrate_pgvector.py` | One-shot migration from SQLite/Chroma → PostgreSQL pgvector |
| `backend/api/` | FastAPI app with `/chat`, `/chat/stream`, `/health` endpoints |
| `backend/config.py` | pydantic-settings from `.env` (GLM-4-Flash via OpenAI-compatible API) |

### Stack

- **LLM**: GLM-4-Flash via `https://open.bigmodel.cn/api/paas/v4/` (OpenAI-compatible protocol)
- **Embeddings**: Zhipu AI `embedding-2`
- **Vector DB**: PostgreSQL + pgvector (collections: `store_policies`, `semantic_cache`)
- **Agent framework**: LangChain + LangGraph
- **API**: FastAPI with SSE streaming, CORS configured for Vite dev server
- **Prod server**: Gunicorn + Uvicorn workers (see `Dockerfile`)

### Configuration

All config lives in `backend/config.py` via `pydantic-settings`, loaded from `.env` (see `.env.example`). The app will not start without valid `OPENAI_API_KEY` and a running PostgreSQL instance.

### Semantic cache behavior

Both agent implementations cache LLM responses in pgvector (`semantic_cache` collection). Cache lookup uses cosine distance < 0.3. Weather queries are explicitly excluded from caching (real-time data, and embeddings for different cities are too similar).
