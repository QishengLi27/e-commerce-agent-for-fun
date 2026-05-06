# Smart E-Commerce Support Agent

A lightweight AI customer service agent for an online store using LangChain, PostgreSQL + pgvector, and semantic caching.

## Monorepo Structure

```
.
├── apps/
│   ├── backend/          # Python AI agent backend (implemented)
│   └── frontend/         # Customer-facing web UI (planned)
├── packages/
│   └── shared/           # Shared types & schemas (planned)
├── infra/
│   ├── docker/           # Docker & Compose configs (planned)
│   └── database/         # Schema migrations (planned)
├── docs/
│   └── architecture.md   # System architecture overview
├── scripts/
│   └── dev-setup.sh      # One-command local setup
└── experiments/          # Research & prototype notebooks
```

## Quick Start

### Option 1: One-command setup

```bash
bash scripts/dev-setup.sh
```

### Option 2: Manual setup with virtual environment

**1. Create and activate a virtual environment**

```bash
cd apps/backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Start PostgreSQL + pgvector**

```bash
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

**4. Set up the database**

```bash
python -m backend.db.setup
python -m backend.db.vector_setup
```

**5. Run the agent**

```bash
python main.py
```

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| Agent | **Implemented** | ReAct reasoning with tool calling |
| Memory | **Implemented** | Persistent multi-turn conversation history |
| Retrieval | **Implemented** | Hybrid search (dense + sparse + re-rank) |
| Resilience | **Implemented** | Circuit breakers, retry, fallbacks |
| REST API | **Planned** | FastAPI endpoints for frontend integration |
| Frontend | **Planned** | React/TypeScript chat interface |
| Shared Types | **Planned** | Common schemas between backend and frontend |
| Docker | **Planned** | Containerized dev/prod environment |

## Test Scenarios

1. **RAG/Vector DB**: "What is your policy on returning electronics?"
2. **Relational DB**: "What is the status of order 1001?"
3. **Cache**: Ask "What is the return policy?" then "How do I return items?"
4. **Multi-Step**: "Can I still return the item in order 1001?"

## Documentation

- [Architecture](docs/architecture.md)
- [Backend README](apps/backend/README.md) — includes detailed venv setup
- [Frontend Plan](apps/frontend/README.md)
- [API Plan](apps/backend/backend/api/README.md)
- [Shared Package Plan](packages/shared/README.md)
- [Docker Plan](infra/docker/README.md)
- [Database Plan](infra/database/README.md)
