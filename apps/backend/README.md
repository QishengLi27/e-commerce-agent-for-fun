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

## Architecture: Intent → Response Pipeline

```
                              USER INPUT: "Can I return my headphones?"

                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  1. sanitize_input                                                            │
│     • Clean typos / normalize                                                 │
│     • Check semantic cache (pgvector cosine < 0.3)                            │
│     • Skip cache for weather queries                                          │
│     ┌──────────────────────────────────────────┐                              │
│     │ CACHE HIT? → set final_answer → skip to  │                              │
│     │              generate_reply (no tools)    │                              │
│     └──────────────────────────────────────────┘                              │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  2. classify_intent  (switchable by CLASSIFICATION_MODE config)                │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │                    SHARED PRE-LAYER (all modes)                       │     │
│  │  extract_entities(query) →  KnowledgeGraph (PostgreSQL)               │     │
│  │  ┌────────────┐     ┌──────────────┐     ┌──────────────┐            │     │
│  │  │ products   │     │ categories   │     │ order_ids    │            │     │
│  │  │ table      │     │ table        │     │ regex 10\d+  │            │     │
│  │  └────────────┘     └──────────────┘     └──────────────┘            │     │
│  │  "headphones"  ✔    "audio"  ✗           order #1005   ✔             │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│                                   │                                            │
│          ┌────────────────────────┼────────────────────────┐                   │
│          ▼                        ▼                        ▼                   │
│  ┌───────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐      │
│  │ MODE: keyword     │ │ MODE: llm_hybrid     │ │ MODE: semantic       │      │
│  │ < 100 products    │ │ 100–10K products     │ │ 10K+ products        │      │
│  │                   │ │                      │ │                      │      │
│  │ 1. entity+keyword │ │ 1. Fast pre-filters  │ │ 1. Fast pre-filters  │      │
│  │    co-signals ────│ │    (weather, order,  │ │    (weather, order,  │      │
│  │    ┌───────────┐  │ │     list_orders)     │ │     list_orders)     │      │
│  │    │has product│  │ │                      │ │                      │      │
│  │    │+ "return" │──│ │ 2. LLM extracts      │ │ 2. Vector search     │      │
│  │    │→ policy   │  │ │    entities+intent   │ │    query → top-K     │      │
│  │    └───────────┘  │ │    in one call       │ │    products (pgvector)│     │
│  │    ┌───────────┐  │ │                      │ │                      │      │
│  │    │has product│  │ │ 3. KG validates:     │ │ 3. LLM classifies    │      │
│  │    │no signals │  │ │    ┌──────────────┐  │ │    with top-K context│     │
│  │    │→ knowledge│  │ │    │exact match? ✓│  │ │                      │      │
│  │    └───────────┘  │ │    │fuzzy match? ─│  │ │ 4. KG validates      │      │
│  │    ...             │ │    │no match?   ✗ │  │ │    LLM's picks       │      │
│  │                    │ │    └──────────────┘  │ │                      │      │
│  │ 2. LLM fallback    │ │                      │ │                      │      │
│  │    (ambiguous only)│ │ Source: llm+kg       │ │ Source: semantic     │      │
│  │                    │ │                      │ │                      │      │
│  │ Source: entity |   │ │                      │ │                      │      │
│  │ entity+keyword|llm │ │                      │ │                      │      │
│  └───────────────────┘ └──────────────────────┘ └──────────────────────┘      │
│                                                                                │
│  Output: { intent, confidence, source, entities, context, order_id }           │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  3. route_by_intent  (conditional edge)                                       │
│                                                                                │
│     CACHED? ──────► generate_reply (skip tools)                                │
│     order ────────► order_node         (SQL: orders table)                    │
│     list_orders ──► list_orders_node   (SQL: orders table)                    │
│     policy ───────► policy_node        (KG + vector RAG)                     │
│     weather ──────► weather_node       (LLM city extract + API)               │
│     knowledge ────► knowledge_node     (KG: product → category → policy)      │
│     unknown ──────► generate_reply     (no tool result)                       │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┬──────────────┐
                    ▼              ▼              ▼              ▼
             ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐
             │policy_node│  │order_node │  │weather   │  │knowledge   │
             │           │  │           │  │_node     │  │_node       │
             └─────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬─────┘
                   │              │             │               │
                   ▼              │             │               ▼
    ┌──────────────────────────┐ │             │    ┌──────────────────────┐
    │ ENRICH query with entity │ │             │    │ KG traversal:        │
    │ context from classifier: │ │             │    │ product → category   │
    │ "headphones" + "return"  │ │             │    │ → policy_rules      │
    │  → "headphones Audio     │ │             │    │ (3-hop JOIN)         │
    │  14-day return Can I..." │ │             │    └──────────────────────┘
    └──────────┬───────────────┘ │             │
               ▼                 │             │
    ┌──────────────────────────┐ │             │
    │ RETRIEVAL_MODE switch     │ │             │
    │ • vector: pgvector+BM25  │ │             │
    │   + RRF + LLM rerank     │ │             │
    │ • graph: SQL JOIN        │ │             │
    │   product→category       │ │             │
    │   →policy_rules          │ │             │
    │ • hybrid: graph first    │ │             │
    │   → vector fallback      │ │             │
    └──────────┬───────────────┘ │             │
               │                 │             │
               ▼                 ▼             ▼
          tool_result        tool_result    tool_result
               │                 │             │
               └─────────┬───────┴─────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  4. generate_reply                                                            │
│     • LLM formats tool_result into user-facing answer                        │
│     • Normal path: friendly, grounded reply                                   │
│     • Retry path:  strict prompt (exact facts only, no inference)             │
│     • Context compression: sliding window + token budget for long sessions    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  5. validate_reply  (LLM auditor — hallucination guard)                       │
│                                                                                │
│     ┌───────────────────────────────────────────────────────┐                 │
│     │ LLM checks: answer grounded in tool_result?           │                 │
│     │                                                       │                 │
│     │ valid ──────────────► proceed to update_memory         │                 │
│     │ unverified_claims ──► retry ≤ 2: loop back to          │                 │
│     │                       generate_reply (strict prompt)   │                 │
│     │ not_applicable ─────► no tool result, proceed anyway   │                 │
│     └───────────────────────────────────────────────────────┘                 │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  6. update_memory                                                             │
│     • Persist to checkpoint (PostgreSQL checkpointer)                         │
│     • Cache response in pgvector semantic_cache (with intent metadata)        │
│     • Skip cache for weather (real-time data)                                 │
│     • Intent metadata prevents stale cache on reclassification                │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
                            FINAL ANSWER to user
```

### Key design decisions

**Intent classification is 3-tier, scaled by catalog size:**
- `keyword` (<100 products) — entity extraction from KG → rule engine with co-signals → LLM only as fallback. Zero LLM cost for most traffic.
- `llm_hybrid` (100–10K) — LLM extracts entities + intent in one shot → KG validates every entity. If the LLM hallucinates a product, the KG catches it and returns fuzzy candidates.
- `semantic` (10K+) — vector search narrows catalog to top-5 candidates → LLM only sees those 5 → KG confirms final picks.

**The Knowledge Graph serves dual roles:**
1. **Intent time** — supplies known entity lists for extraction, validates LLM claims
2. **Retrieval time** — `GraphPolicyRetriever` does deterministic 3-hop traversal (product → category → policy_rules) via SQL JOINs, no embedding needed

**Two independent policy retrieval paths:**
- **Graph**: Deterministic SQL JOINs — exact, explainable, zero hallucination risk
- **Vector**: pgvector + BM25 → RRF fusion → LLM rerank — semantic, handles rephrased queries
- **Hybrid** (default): Graph first, vector fallback

**Self-correction loop**: `validate_reply` → `generate_reply` catches LLM hallucinations after generation and retries with a stricter prompt (up to 2 retries).

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
pip install -e ".[dev]"
```

> **Note:** `requirements.txt` is deprecated. All dependencies are now defined in `pyproject.toml`. The command above installs the package in editable mode with development extras (testing, linting, type checking). For production-only installs, use `pip install -e .`

### 4. Install git hooks (one-time)

Pre-commit was already installed in step 3 (via `[dev]` extras). Just activate the hook:

```bash
pre-commit install
```

### 5. Start PostgreSQL + pgvector

```bash
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

### 6. Set up the database

Make sure your virtual environment is activated, then:

```bash
python -m backend.db.setup
python -m backend.db.vector_setup
python -m backend.knowledge.schema
```

### 7. Run the API server (FastAPI)

```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or with gunicorn + uvicorn workers (production):

```bash
gunicorn backend.api.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4
```

### 8. Run the CLI agent (alternative)

```bash
python main.py
```

## Using the root-level venv (alternative)

If you prefer to keep one virtual environment at the repository root:

```bash
# From repo root
source venv/bin/activate
cd apps/backend
pip install -e ".[dev]"
python -m backend.db.setup
python -m backend.db.vector_setup
python main.py
```

## Code Quality

This project uses **ruff** (linting + formatting) and **mypy** (static type checking). All code is checked automatically on every commit via pre-commit hooks.

### Run checks manually

```bash
# Linting + auto-fix import order, unused vars, Python upgrade checks, etc.
ruff check backend/ main.py tests/ --fix

# Format code
ruff format backend/ main.py tests/

# Type check
mypy backend/ main.py tests/
```

**Recommended workflow** after making changes:

```bash
ruff check . --fix && ruff format . && mypy .
```

### Run all hooks without committing

```bash
pre-commit run --all-files
```

### VS Code integration

Install the **Ruff** extension (Astral Software) for real-time linting and format-on-save.

## Environment Variables

See `.env.example` for the full configuration. Key settings:

- `OPENAI_API_KEY` / `OPENAI_API_BASE` — LLM provider credentials
- `DATABASE_URL` — PostgreSQL connection string
- `RETRIEVAL_MODE` — `vector` | `graph` | `hybrid` (see `.env.example` for details)
- `CORS_ORIGINS` — allowed frontend origins for the API

## Running individual modules

Always run modules with the `-m` flag so Python resolves the `backend` package correctly:

```bash
python -m backend.db.setup
python -m backend.db.vector_setup
python -m backend.knowledge.schema
python -m backend.db.migrate_pgvector
python -m backend.agent
```

## Notes

- All data paths are relative to `apps/backend/` (e.g., `data/store_policies.txt`).
- The `PYTHONPATH` is automatically correct when you run from `apps/backend/`.
