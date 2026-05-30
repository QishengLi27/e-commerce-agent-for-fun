# Backend Coding Standards

> This file governs all code in `apps/backend/` and its subdirectories.
> It supplements the root `CLAUDE.md` with backend-specific conventions.

---

## 1. Toolchain (Non-negotiable)

Every commit must pass:

```bash
cd apps/backend
ruff check backend/ main.py tests/ --fix
ruff format backend/ main.py tests/
mypy backend/ main.py tests/
```

Pre-commit hooks are installed via `.pre-commit-config.yaml`. If a hook fails, fix before committing.

---

## 2. Google Python Style Guide (Condensed)

### 2.1 Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Module / package | `snake_case` | `intent_classifier.py` |
| Class | `PascalCase` | `HybridPolicyRetriever` |
| Function / method | `snake_case` | `retrieve_policies()` |
| Constant | `UPPER_SNAKE_CASE` | `CACHE_SIMILARITY_THRESHOLD = 0.3` |
| Private | `_leading_underscore` | `_tokenize()` |
| Strong internal | `__double_underscore` | `__post_init__()` |

### 2.2 Imports (PEP 8 order)

```python
"""Module docstring."""

# 1. Standard library
import logging
import re
from typing import Any

# 2. Third-party
from langchain_core.documents import Document
from pydantic import BaseModel

# 3. Local (alphabetical)
from backend.config import settings
from backend.resilience import CircuitBreaker
```

**Rules:**
- Never use `from module import *`.
- Use `typing` imports only for things not in `builtins` (e.g. `Any`, `Protocol`, `cast`). Prefer `list[str]` over `List[str]`.

### 2.3 Type Annotations

- **All public functions** must have parameter and return type annotations.
- **All class attributes** must be typed.
- Avoid bare `dict`, `list`, `tuple` — be specific: `dict[str, float]`.
- Use `X | None` (Python 3.10+) instead of `Optional[X]`.

```python
# ✅ Good
def dense_retrieve(
    self,
    query: str,
    k: int = 5,
) -> list[tuple[Document, float]]:
    ...

# ❌ Bad
def dense_retrieve(self, query, k=5):
    ...
```

### 2.4 Docstrings (Google style)

Every public module, class, and function gets a docstring.

```python
def retrieve(
    self,
    query: str,
    k: int = 5,
    rerank: bool = True,
) -> list[tuple[Document, float]]:
    """Run hybrid retrieval: dense + sparse → RRF → optional re-rank.

    Args:
        query: User's natural language query.
        k: Number of top documents to return.
        rerank: Whether to apply LLM-based re-ranking.

    Returns:
        List of (document, score) tuples sorted by relevance descending.

    Raises:
        RetrievalError: If both dense and sparse retrieval fail.
    """
```

### 2.5 Function Design

- **Single Responsibility**: one function = one job.
- **Max 50 lines** per function (preferably < 30).
- **Max 4 parameters**; use dataclasses/Pydantic models for more.
- **No side effects** in pure utility functions.

### 2.6 Error Handling

- Use **specific exceptions**, never bare `except:`.
- Define a hierarchy under `backend/exceptions.py`.

```python
# backend/exceptions.py
class AgentError(Exception):
    """Base agent exception."""
    pass

class RetrievalError(AgentError):
    """RAG retrieval failed."""
    pass
```

### 2.7 Constants & Magic Numbers

No literal numbers/strings in business logic. Define at module top:

```python
# ✅ Good
DEFAULT_TOP_K = 5
CACHE_SIMILARITY_THRESHOLD = 0.3

# ❌ Bad
if score < 0.3:   # what is 0.3?
```

---

## 3. Architecture Patterns

### 3.1 LangGraph Nodes

- Each node is a **pure function**: `AgentState -> AgentState`.
- Nodes do not import each other.
- Side effects (DB writes, API calls) only in dedicated tool functions.

### 3.2 Dependency Injection

Use FastAPI `Depends` for retrievers, LLMs, and stores. Avoid global singletons in tests.

```python
@app.post("/chat")
async def chat(
    retriever: Annotated[HybridPolicyRetriever, Depends(get_retriever)],
):
    ...
```

### 3.3 RAG Pipeline Changes

Any change to retrieval logic must:
1. Update type annotations.
2. Add/update unit test for deterministic components (RRF, filtering).
3. Run `tests/eval_rag.py` and confirm scores do not regress.

---

## 4. Review Checklist (Before Commit)

- [ ] `ruff check . --fix` passes
- [ ] `ruff format .` passes
- [ ] `mypy .` passes (zero errors)
- [ ] All new functions have type annotations
- [ ] All new functions have Google-style docstrings
- [ ] No `print()` in module code (use `logging`)
- [ ] No bare `except:`
- [ ] No magic numbers (define constants)
- [ ] Tests added/updated for new logic

---

## 5. Design Patterns for Data Access (Taxonomy & Knowledge Graph)

These patterns apply to all new code in `backend/knowledge/`. Existing code should be migrated to these patterns when touched.

### 5.1 Repository Pattern with Typed Returns

`KnowledgeStore` is a **repository** — it encapsulates all SQL, and callers never write queries. New methods must return **dataclass instances**, never raw `dict`.

```python
# ✅ Good — typed return value
from dataclasses import dataclass

@dataclass(frozen=True)
class ProductInfo:
    name: str
    price: float | None
    sku: str
    category_name: str
    category_description: str | None
    attributes: list[ProductAttribute]
    policies: list[PolicySummary]
    accessories: list[ProductRef]
    alternatives: list[ProductRef]

class KnowledgeStore:
    def get_product_info(self, query: str) -> ProductInfo | None:
        ...

# ❌ Bad — raw dict, unknown shape
class KnowledgeStore:
    def get_product_info(self, query: str) -> dict | None:
        ...
```

**Rule:** if a method returns a compound value, define a frozen dataclass for it. Use `@dataclass(frozen=True)` for immutability. Place dataclasses in the same file as the method that produces them, or extract to `backend/knowledge/models.py` if shared.

### 5.2 Query Object Pattern

Complex queries (multi-filter, attribute search) must be encapsulated in a dedicated query object — no SQL string building in repository methods.

```python
# ✅ Good — query object
@dataclass(frozen=True)
class AttributeFilter:
    attribute_name: str
    operator: str  # eq, gt, lt, gte, lte, contains, between
    value: str | int | float | bool

@dataclass(frozen=True)
class ProductQuery:
    keyword: str | None = None
    category: str | None = None
    filters: list[AttributeFilter] = field(default_factory=list)
    limit: int = 20

class KnowledgeStore:
    def search_products(self, query: ProductQuery) -> list[ProductRef]:
        ...

# ❌ Bad — positional args, implicit logic
class KnowledgeStore:
    def search_products(
        self, keyword: str, filters: list[dict] | None = None,
        category: str | None = None, limit: int = 20,
    ) -> list[dict]:
        ...
```

### 5.3 Interface Segregation for Retrievers

The `PolicyRetriever` ABC already enforces a single interface. Extend this pattern — new retrieval capabilities get their own protocol/ABC, not monolith methods on an existing class.

```python
# ✅ Good — focused interface
class SynonymResolver(Protocol):
    def resolve(self, term: str, entity_type: str | None = None) -> str | None: ...
    def expand_query(self, query: str) -> list[str]: ...

# ✅ Good — separate interface
class RelationLookup(Protocol):
    def get_accessories(self, product_name: str) -> list[ProductRef]: ...
    def get_alternatives(self, product_name: str) -> list[ProductRef]: ...
```

**Rule:** if a new capability adds 3+ methods to KnowledgeStore, extract it to a focused class/Protocol behind its own interface.

### 5.4 No Primitive Obsession

Domain concepts must be represented as types, not strings/dicts.

| Primitive | Replace with |
|-----------|-------------|
| `str` product name | `ProductRef(name: str, id: int)` |
| `str` category name | `CategoryRef(name: str, id: int)` |
| `dict` policy | `PolicySummary(name: str, summary: str, type: str)` |
| `tuple[str, float]` score | `ScoredDoc(doc: Document, score: float)` |

### 5.5 Dependency Injection (reinforced)

The singleton `get_knowledge_store()` is acceptable for CLI usage, but all new retrievers and tools must accept their dependencies through `__init__`:

```python
# ✅ Good — injectable
class GraphPolicyRetriever(PolicyRetriever):
    def __init__(self, store: KnowledgeStore | None = None):
        self._store = store or get_knowledge_store()

# ✅ Good — FastAPI DI
async def chat(
    retriever: Annotated[HybridPolicyRetriever, Depends(get_retriever)],
): ...
```

### 5.6 Testing Requirements for Data Access

- **Unit tests:** mock the DB connection, test query logic in isolation
- **Integration tests:** hit a real PostgreSQL (use a test database or the existing local pgvector container)
- **AAA pattern:** Arrange → Act → Assert, with explicit section comments
- **Parametrize:** use `@pytest.mark.parametrize` for synonym variants, filter combinations, edge cases

---

## 6. Reference

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8](https://peps.python.org/pep-0008/)
- [PEP 257](https://peps.python.org/pep-0257/)
- [PEP 484](https://peps.python.org/pep-0484/)
- [Repository Pattern (Fowler)](https://martinfowler.com/eaaCatalog/repository.html)
- [Query Object Pattern (Fowler)](https://martinfowler.com/eaaCatalog/queryObject.html)
