# Backend

The AI-powered e-commerce support agent backend.

## Structure

```
backend/
├── backend/          # Python package
│   ├── agent.py      # ReAct agent with tools, cache, and memory
│   ├── memory.py     # Persistent conversation history
│   ├── retrieval.py  # Hybrid RAG (pgvector + BM25 + LLM re-rank)
│   ├── resilience.py # Circuit breakers, retries, fallbacks
│   ├── db/           # Database setup & migrations
│   │   ├── setup.py
│   │   ├── vector_setup.py
│   │   └── migrate_pgvector.py
│   └── api/          # REST API layer (planned — see api/README.md)
├── data/             # Local data assets & runtime files
├── tests/            # Test suite (planned — see tests/README.md)
├── main.py           # CLI entry point
└── requirements.txt
```

## Setup with Virtual Environment

All commands below assume you are inside `apps/backend/`.

### 1. Create a virtual environment

```bash
cd apps/backend
python -m venv venv
```

### 2. Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start PostgreSQL + pgvector

```bash
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

### 5. Set up the database

Make sure your virtual environment is activated, then:

```bash
python -m backend.db.setup
python -m backend.db.vector_setup
```

### 6. Run the agent

```bash
python main.py
```

## Using the root-level venv (alternative)

If you prefer to keep one virtual environment at the repository root:

```bash
# From repo root
source venv/bin/activate
cd apps/backend
python -m backend.db.setup
python -m backend.db.vector_setup
python main.py
```

## Environment Variables

- `OPENAI_API_KEY` / `OPENAI_API_BASE` — currently hardcoded; should be moved to env vars

## Running individual modules

Always run modules with the `-m` flag so Python resolves the `backend` package correctly:

```bash
python -m backend.db.setup
python -m backend.db.vector_setup
python -m backend.db.migrate_pgvector
python -m backend.agent
```

## Notes

- All data paths are relative to `apps/backend/` (e.g., `data/store_policies.txt`).
- The `PYTHONPATH` is automatically correct when you run from `apps/backend/`.
