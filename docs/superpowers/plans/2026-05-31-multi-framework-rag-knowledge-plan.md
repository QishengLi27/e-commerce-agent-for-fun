# Multi-Framework RAG + Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Neo4j graph + LlamaIndex RAG into the existing LangGraph agent to enable product Q&A with metadata-filtered vector search and deterministic graph traversal.

**Architecture:** PostgreSQL remains source of truth for seed data. Neo4j is a read-only graph cache synced on setup. LlamaIndex handles product description ingestion and metadata-filtered RAG via pgvector. A single `product_qa_tool` (LangChain `@tool`) orchestrates Neo4j → LlamaIndex → LLM synthesis. LangGraph agent routes `product_qa` intent to this tool.

**Tech Stack:** LangGraph (agent DAG), Neo4j + Cypher (graph traversal), LlamaIndex + pgvector (RAG), PostgreSQL (source of truth)

**Spec:** `docs/superpowers/specs/2026-05-31-multi-framework-rag-knowledge-design.md`

**Design Patterns (from AGENTS.md §5):** Typed dataclasses, query objects, interface segregation, dependency injection, no primitive obsession, AAA + parametrize tests.

---

## Summary

### Phase Overview

| Phase | Tasks | What it builds | New Files | Est. Time |
|-------|-------|---------------|-----------|-----------|
| **1: Dependencies** | 1.1–1.3 | neo4j + llama-index packages, docker-compose (pgvector + Neo4j), config settings | 1 | 15 min |
| **2: Neo4j Setup** | 2.1–2.2 | Shared dataclass models (`ProductRef`, `ProductInfo`, etc.), PG → Neo4j sync script with all node types + relationships + indexes | 2 | 30 min |
| **3: Neo4j Queries** | 3.1 | `Neo4jStore` with typed Cypher methods (resolve_product, get_product_info, search_products, relations, category tree) + unit tests | 2 | 45 min |
| **4: Product Text + Ingestion** | 4.1–4.2 | `product_descriptions.txt` with `[product: X]` blocks + LlamaIndex `IngestionPipeline` (SentenceWindowNodeParser → embed → pgvector) + parser tests | 3 | 30 min |
| **5: Query Engine** | 5.1 | LlamaIndex `RetrieverQueryEngine` factory with metadata filtering (product_name, brand, category) + sentence-window context expansion | 1 | 15 min |
| **6: Product QA Tool** | 6.1 | `product_qa_tool` — LangChain `@tool` orchestrating Neo4j → LlamaIndex → LLM synthesis for 3 query patterns (single-product, comparison, category) + integration tests | 2 | 45 min |
| **7: Agent Wiring** | 7.1–7.3 | `product_qa` intent detection in all classifiers, `product_qa_node` in LangGraph DAG, tool export from `tools/__init__` | 3 | 30 min |
| **8: Verification** | 8.1 | Full setup sequence, smoke tests via curl, ruff+mypy+pytest, final commit | 0 | 15 min |

### Files Changed

| Action | File |
|--------|------|
| **CREATE** | `apps/backend/docker-compose.yml` |
| **CREATE** | `apps/backend/backend/knowledge/models.py` |
| **CREATE** | `apps/backend/backend/knowledge/neo4j_setup.py` |
| **CREATE** | `apps/backend/backend/knowledge/neo4j_store.py` |
| **CREATE** | `apps/backend/data/product_descriptions.txt` |
| **CREATE** | `apps/backend/backend/rag/__init__.py` |
| **CREATE** | `apps/backend/backend/rag/ingestion.py` |
| **CREATE** | `apps/backend/backend/rag/query_engine.py` |
| **CREATE** | `apps/backend/backend/tools/product_qa.py` |
| **CREATE** | `apps/backend/tests/test_neo4j_store.py` |
| **CREATE** | `apps/backend/tests/test_rag_ingestion.py` |
| **CREATE** | `apps/backend/tests/test_product_qa_tool.py` |
| **MODIFY** | `apps/backend/pyproject.toml` |
| **MODIFY** | `apps/backend/backend/config.py` |
| **MODIFY** | `apps/backend/backend/tools/__init__.py` |
| **MODIFY** | `apps/backend/backend/graph/nodes.py` |
| **MODIFY** | `apps/backend/backend/graph/agent_graph.py` |
| **MODIFY** | `apps/backend/backend/intent/keyword.py` |
| **MODIFY** | `apps/backend/backend/intent/llm_hybrid.py` |
| **MODIFY** | `apps/backend/backend/intent/semantic.py` |

### Test Coverage

| Test File | Type | Cases |
|-----------|------|-------|
| `test_neo4j_store.py` | Unit (requires Neo4j) | 9 — resolve_product (4), get_product_info (4), search_products (3) |
| `test_rag_ingestion.py` | Unit (no DB) | 2 — single product parse, multiple products parse |
| `test_product_qa_tool.py` | Integration (requires Neo4j + PG) | 4 — single-product, category, comparison, unknown graceful |

### Design Patterns Applied (per AGENTS.md §5)

- **Typed dataclasses**: `ProductRef`, `ProductInfo`, `ProductAttribute`, `PolicySummary`, `CategoryRef` — no raw `dict` returns
- **Query objects**: `search_products()` uses keyword arguments, not positional
- **Interface segregation**: `Neo4jStore` separated from old `KnowledgeStore`
- **Dependency injection**: `Neo4jStore.__init__` accepts connection params; `create_filtered_query_engine` accepts filters
- **No primitive obsession**: Product names are `ProductRef`, not bare `str`
- **AAA + parametrize**: Tests structured Arrange-Act-Assert with `parametrize` for synonym variants

### Execution Options

**Option 1: Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

**Option 2: Inline Execution** — Execute tasks in this session via `superpowers:executing-plans`, batch execution with checkpoints.

---

## File Structure Map

```
apps/backend/
├── pyproject.toml                   # MODIFY: add neo4j, llama-index deps
├── docker-compose.yml               # CREATE: pgvector + Neo4j services
├── data/
│   └── product_descriptions.txt     # CREATE: [product: X] blocks with metadata
├── backend/
│   ├── config.py                    # MODIFY: add neo4j_uri/user/password
│   ├── knowledge/
│   │   ├── neo4j_setup.py           # CREATE: sync PG seed → Neo4j
│   │   ├── neo4j_store.py           # CREATE: Neo4j driver + query methods
│   │   └── models.py                # CREATE: shared dataclasses (ProductRef, etc.)
│   ├── rag/
│   │   ├── __init__.py              # CREATE: empty init
│   │   ├── ingestion.py             # CREATE: LlamaIndex IngestionPipeline
│   │   └── query_engine.py          # CREATE: RetrieverQueryEngine factory
│   ├── tools/
│   │   ├── __init__.py              # MODIFY: export product_qa_tool
│   │   └── product_qa.py            # CREATE: product_qa_tool (@tool)
│   ├── graph/
│   │   ├── nodes.py                 # MODIFY: add product_qa_node, route
│   │   └── agent_graph.py           # MODIFY: add product_qa node to graph
│   └── intent/
│       └── keyword.py               # MODIFY: add product_qa signals
└── tests/
    ├── test_neo4j_store.py          # CREATE: unit tests for Neo4j queries
    ├── test_rag_ingestion.py        # CREATE: unit tests for ingestion pipeline
    ├── test_product_qa_tool.py      # CREATE: integration tests for tool
    └── conftest.py                  # CREATE: Neo4j + PG test fixtures
```

---

## Phase 1: Dependencies & Configuration

### Task 1.1: Add Neo4j + LlamaIndex dependencies

**Files:**
- Modify: `apps/backend/pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Add to the `dependencies` list after the existing `# Resilience` section:

```toml
    # Neo4j graph database
    "neo4j>=5.27.0",
    # LlamaIndex for RAG ingestion and querying
    "llama-index>=0.12.0",
    "llama-index-vector-stores-postgres>=0.4.0",
    "llama-index-embeddings-openai>=0.3.0",
```

- [ ] **Step 2: Install new dependencies**

Run: `cd apps/backend && pip install -e .`
Expected: packages install without version conflicts

- [ ] **Step 3: Commit**

```bash
git add apps/backend/pyproject.toml
git commit -m "build: add neo4j and llama-index dependencies"
```

---

### Task 1.2: Create docker-compose.yml

**Files:**
- Create: `apps/backend/docker-compose.yml`

- [ ] **Step 1: Write docker-compose.yml**

```yaml
services:
  pgvector:
    image: pgvector/pgvector:pg16
    container_name: pgvector
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: ecommerce
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:community
    container_name: neo4j
    ports:
      - "7474:7474"   # HTTP / Browser UI
      - "7687:7687"   # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p password 'RETURN 1'"]
      interval: 10s
      timeout: 10s
      retries: 5

volumes:
  pgvector_data:
  neo4j_data:
  neo4j_logs:
```

- [ ] **Step 2: Verify containers start**

Run: `cd apps/backend && docker compose up -d`
Expected: both containers healthy (`docker compose ps` shows "healthy")

- [ ] **Step 3: Verify Neo4j is reachable**

Run: `curl http://localhost:7474`
Expected: JSON response with Neo4j version info

- [ ] **Step 4: Commit**

```bash
git add apps/backend/docker-compose.yml
git commit -m "infra: add docker-compose with pgvector and neo4j services"
```

---

### Task 1.3: Add Neo4j config to Settings

**Files:**
- Modify: `apps/backend/backend/config.py`

- [ ] **Step 1: Add Neo4j settings to Settings class**

Add after the existing `# ─── Database ──` section and before `# ─── LLM / Embeddings ──`:

```python
    # ─── Neo4j Graph Database ──────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j password",
    )
```

- [ ] **Step 2: Verify config loads**

Run: `cd apps/backend && python -c "from backend.config import settings; print(settings.neo4j_uri)"`
Expected: `bolt://localhost:7687`

- [ ] **Step 3: Commit**

```bash
git add apps/backend/backend/config.py
git commit -m "feat: add Neo4j connection settings to config"
```

---

## Phase 2: Neo4j Graph Setup

### Task 2.1: Create shared dataclass models

**Files:**
- Create: `apps/backend/backend/knowledge/models.py`

- [ ] **Step 1: Write models.py**

```python
"""Shared dataclass models for knowledge graph entities.

These replace raw dict returns from KnowledgeStore and Neo4jStore,
providing type-safe, self-documenting return values.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProductRef:
    """Lightweight product reference (list/search results)."""
    name: str
    price: float | None
    sku: str | None = None
    category_name: str | None = None


@dataclass(frozen=True)
class CategoryRef:
    """Category node reference with hierarchy info."""
    name: str
    level: int
    description: str | None = None
    children: list["CategoryRef"] = field(default_factory=list)


@dataclass(frozen=True)
class ProductAttribute:
    """Typed attribute value for a product."""
    name: str
    display_name: str
    value: str | int | float | bool
    data_type: str
    unit: str | None = None


@dataclass(frozen=True)
class PolicySummary:
    """Policy reference returned from graph traversal."""
    name: str
    policy_type: str
    summary: str
    details: str


@dataclass(frozen=True)
class ProductInfo:
    """Full product detail including attributes, policies, and relations."""
    name: str
    price: float | None
    sku: str | None
    category_name: str | None
    category_path: list[str] = field(default_factory=list)
    attributes: list[ProductAttribute] = field(default_factory=list)
    policies: list[PolicySummary] = field(default_factory=list)
    accessories: list[ProductRef] = field(default_factory=list)
    alternatives: list[ProductRef] = field(default_factory=list)


@dataclass(frozen=True)
class QAResult:
    """Result from product Q&A: retrieved context + synthesized answer."""
    answer: str
    matched_products: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
```

- [ ] **Step 2: Verify module imports**

Run: `cd apps/backend && python -c "from backend.knowledge.models import ProductRef, ProductInfo; print(ProductRef(name='test', price=9.99))"`
Expected: `ProductRef(name='test', price=9.99, sku=None, category_name=None)`

- [ ] **Step 3: Commit**

```bash
git add apps/backend/backend/knowledge/models.py
git commit -m "feat: add shared dataclass models for knowledge graph entities"
```

---

### Task 2.2: Create Neo4j graph sync script

**Files:**
- Create: `apps/backend/backend/knowledge/neo4j_setup.py`

- [ ] **Step 1: Write neo4j_setup.py**

```python
"""
Sync PostgreSQL seed data to Neo4j graph database.

Run once after schema setup:
    python -m backend.knowledge.neo4j_setup

Reads all taxonomy data from PostgreSQL (source of truth) and recreates
the Neo4j graph (read-optimized cache). Idempotent — wipes existing
graph first.
"""

import psycopg2
from neo4j import GraphDatabase

from backend.config import settings

PG_CONN = settings.pg_connection_raw
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password


def _get_pg_conn():
    return psycopg2.connect(PG_CONN)


def setup_neo4j():
    """Main entry point: sync all PG data to Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        _clear_graph(session)
        _create_indexes(session)
        _sync_categories(session)
        _sync_products(session)
        _sync_brands(session)
        _sync_attributes(session)
        _sync_product_relations(session)
        _sync_policies(session)
        _sync_synonyms(session)

    driver.close()
    print("[neo4j] Graph sync complete.")


def _clear_graph(session):
    """Remove all nodes and relationships (idempotent)."""
    session.run("MATCH (n) DETACH DELETE n")
    print("[neo4j] Cleared existing graph.")


def _create_indexes(session):
    """Create constraints and indexes for fast lookups."""
    session.run("CREATE CONSTRAINT product_name IF NOT EXISTS "
                "FOR (p:Product) REQUIRE p.name IS UNIQUE")
    session.run("CREATE CONSTRAINT category_name IF NOT EXISTS "
                "FOR (c:Category) REQUIRE c.name IS UNIQUE")
    session.run("CREATE CONSTRAINT brand_name IF NOT EXISTS "
                "FOR (b:Brand) REQUIRE b.name IS UNIQUE")
    session.run("CREATE CONSTRAINT policy_name IF NOT EXISTS "
                "FOR (pol:Policy) REQUIRE pol.name IS UNIQUE")
    session.run("CREATE FULLTEXT INDEX product_search IF NOT EXISTS "
                "FOR (p:Product) ON EACH [p.name, p.search_terms]")
    session.run("CREATE INDEX product_price IF NOT EXISTS "
                "FOR (p:Product) ON (p.price)")
    print("[neo4j] Indexes and constraints created.")


def _sync_categories(session):
    """Sync hierarchical categories from PG.

    Creates (:Category) nodes with CHILD_OF relationships forming the tree.
    """
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, description, parent_id, level FROM categories ORDER BY level"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    cat_data = {row[0]: {"name": row[1], "description": row[2],
                          "parent_id": row[3], "level": row[4]}
                for row in rows}

    # Create all category nodes
    for cat_id, data in cat_data.items():
        session.run(
            """
            CREATE (c:Category {
                name: $name,
                description: $description,
                level: $level
            })
            """,
            name=data["name"],
            description=data["description"] or "",
            level=data["level"],
        )

    # Create CHILD_OF relationships
    for cat_id, data in cat_data.items():
        if data["parent_id"] and data["parent_id"] in cat_data:
            parent_name = cat_data[data["parent_id"]]["name"]
            session.run(
                """
                MATCH (child:Category {name: $child_name})
                MATCH (parent:Category {name: $parent_name})
                CREATE (child)-[:CHILD_OF]->(parent)
                """,
                child_name=data["name"],
                parent_name=parent_name,
            )

    print(f"[neo4j] Synced {len(rows)} categories.")


def _sync_products(session):
    """Sync products with category relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.name, p.price, p.sku, c.name AS category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for name, price, sku, category_name in rows:
        session.run(
            """
            MATCH (c:Category {name: $category_name})
            CREATE (p:Product {
                name: $name,
                price: $price,
                sku: $sku,
                search_terms: $name
            })
            CREATE (p)-[:IN_CATEGORY]->(c)
            """,
            name=name,
            price=float(price) if price else 0.0,
            sku=sku or "",
            category_name=category_name,
        )

    print(f"[neo4j] Synced {len(rows)} products.")


def _sync_brands(session):
    """Extract brands from product_attributes (attribute_name='brand')
    and create (:Brand) nodes with HAS_BRAND relationships.
    """
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT pa.value_text
        FROM product_attributes pa
        JOIN attribute_definitions ad ON pa.attribute_id = ad.id
        WHERE ad.name = 'brand' AND pa.value_text IS NOT NULL
        """
    )
    brands = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    for brand_name in brands:
        session.run(
            "CREATE (:Brand {name: $name})",
            name=brand_name,
        )

        # Link products to brands
        session.run(
            """
            MATCH (p:Product)
            WHERE p.name CONTAINS $brand OR p.search_terms CONTAINS $brand
            MATCH (b:Brand {name: $brand})
            CREATE (p)-[:HAS_BRAND]->(b)
            """,
            brand=brand_name,
        )

    print(f"[neo4j] Synced {len(brands)} brands.")


def _sync_attributes(session):
    """Create attribute values as nodes with HAS_ATTRIBUTE relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.name, ad.name AS attr_name, ad.display_name, ad.data_type,
               pa.value_text, pa.value_number, pa.value_boolean
        FROM product_attributes pa
        JOIN products p ON pa.product_id = p.id
        JOIN attribute_definitions ad ON pa.attribute_id = ad.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    count = 0
    for product_name, attr_name, display_name, data_type, v_text, v_num, v_bool in rows:
        # Determine the actual value based on data_type
        if data_type == "text":
            value = str(v_text) if v_text else ""
        elif data_type == "number":
            value = str(v_num) if v_num is not None else "0"
        elif data_type == "boolean":
            value = str(v_bool) if v_bool is not None else "false"
        else:
            value = str(v_text or "")

        session.run(
            """
            MATCH (p:Product {name: $product_name})
            CREATE (av:AttributeValue {value: $value})-[:OF_TYPE]->(:Attribute {
                name: $attr_name,
                display_name: $display_name,
                data_type: $data_type
            })
            CREATE (p)-[:HAS_ATTRIBUTE]->(av)
            """,
            product_name=product_name,
            attr_name=attr_name,
            display_name=display_name,
            data_type=data_type,
            value=value,
        )
        count += 1

    print(f"[neo4j] Synced {count} product attributes.")


def _sync_product_relations(session):
    """Create ACCESSORY_OF, ALTERNATIVE_TO, COMPATIBLE_WITH relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sp.name, tp.name, pr.relation_type, pr.strength
        FROM product_relations pr
        JOIN products sp ON pr.source_product_id = sp.id
        JOIN products tp ON pr.target_product_id = tp.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    rel_type_map = {
        "accessory": "ACCESSORY_OF",
        "alternative": "ALTERNATIVE_TO",
        "bundle": "BUNDLED_WITH",
        "compatible": "COMPATIBLE_WITH",
        "upgrade": "UPGRADE_OF",
    }

    for source_name, target_name, rel_type, strength in rows:
        neo4j_rel = rel_type_map.get(rel_type, "RELATED_TO")
        session.run(
            f"""
            MATCH (source:Product {{name: $source_name}})
            MATCH (target:Product {{name: $target_name}})
            CREATE (source)-[:{neo4j_rel} {{strength: $strength}}]->(target)
            """,
            source_name=source_name,
            target_name=target_name,
            strength=float(strength),
        )

    print(f"[neo4j] Synced {len(rows)} product relations.")


def _sync_policies(session):
    """Create (:Policy) nodes with HAS_POLICY relationships to categories."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pr.name, pr.policy_type, pr.summary, pr.details, c.name AS category_name
        FROM policy_rules pr
        JOIN policy_category_rules pcr ON pr.id = pcr.policy_rule_id
        JOIN categories c ON pcr.category_id = c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    for policy_name, policy_type, summary, details, category_name in rows:
        session.run(
            """
            MERGE (pol:Policy {name: $name})
            SET pol.policy_type = $policy_type,
                pol.summary = $summary,
                pol.details = $details
            WITH pol
            MATCH (c:Category {name: $category_name})
            MERGE (c)-[:HAS_POLICY]->(pol)
            """,
            name=policy_name,
            policy_type=policy_type,
            summary=summary,
            details=details,
            category_name=category_name,
        )

    print(f"[neo4j] Synced {len(rows)} policy assignments.")


def _sync_synonyms(session):
    """Attach search_terms to products from entity_synonyms table."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT canonical_name, synonym
        FROM entity_synonyms
        WHERE entity_type IN ('product', 'brand')
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    count = 0
    for canonical_name, synonym in rows:
        result = session.run(
            """
            MATCH (p:Product)
            WHERE p.name CONTAINS $canonical OR p.name = $canonical
               OR (p)-[:HAS_BRAND]->(:Brand {name: $canonical})
            SET p.search_terms = COALESCE(p.search_terms, '') + ', ' + $synonym
            RETURN p.name
            """,
            canonical=canonical_name,
            synonym=synonym,
        )
        if result.single():
            count += 1

    print(f"[neo4j] Attached synonyms to {count} products.")


if __name__ == "__main__":
    setup_neo4j()
```

- [ ] **Step 2: Run the sync against a fresh Neo4j instance**

Precondition: PostgreSQL is running with seed data (`python -m backend.knowledge.schema` already run).

Run: `python -m backend.knowledge.neo4j_setup`
Expected: output shows counts for categories, products, brands, attributes, relations, policies, synonyms. No errors.

- [ ] **Step 3: Verify in Neo4j Browser**

Open `http://localhost:7474`, run: `MATCH (n) RETURN labels(n), count(n)`
Expected: multiple node types with counts matching PG data

- [ ] **Step 4: Commit**

```bash
git add apps/backend/backend/knowledge/neo4j_setup.py
git commit -m "feat: add Neo4j graph sync script (PG → Neo4j)"
```

---

## Phase 3: Neo4j Query Store

### Task 3.1: Create Neo4jStore with typed query methods

**Files:**
- Create: `apps/backend/backend/knowledge/neo4j_store.py`

- [ ] **Step 1: Write neo4j_store.py**

```python
"""
Neo4j-backed knowledge graph store.

Provides typed graph traversal queries via Cypher.
Singleton pattern: get_neo4j_store() returns the shared instance.

As of the multi-framework architecture, this replaces KnowledgeStore
for graph operations. The old graph_store.py is retained for backward
compatibility but new code uses Neo4jStore.
"""

from neo4j import GraphDatabase

from backend.config import settings
from backend.knowledge.models import (
    CategoryRef,
    PolicySummary,
    ProductAttribute,
    ProductInfo,
    ProductRef,
)


class Neo4jStore:
    """Neo4j-backed knowledge graph with Cypher traversal.

    Supports:
      - Product lookup with synonym expansion
      - Category tree traversal (ancestors, descendants)
      - Attribute-filtered product search
      - Product relations (accessories, alternatives)
      - Policy inheritance via CHILD_OF* traversal
    """

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()

    # ── Product Lookup ──────────────────────────────────────────────────────

    def resolve_product(self, query: str) -> ProductRef | None:
        """Resolve a term to a product via name, fulltext index, or synonym.

        Args:
            query: Product name or synonym (e.g., "苹果手机" → "iPhone 15 Pro").

        Returns:
            ProductRef if found, None otherwise.
        """
        with self._driver.session() as session:
            # Try fulltext search first (covers name + search_terms)
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('product_search', $query)
                YIELD node, score
                WHERE score > 0.5
                RETURN node.name AS name, node.price AS price,
                       node.sku AS sku, score
                ORDER BY score DESC
                LIMIT 1
                """,
                query=query,
            )
            record = result.single()
            if record:
                category = self._get_product_category(record["name"])
                return ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=category,
                )

            # Fallback: ILIKE match
            result = session.run(
                """
                MATCH (p:Product)
                WHERE p.name CONTAINS $query OR p.search_terms CONTAINS $query
                RETURN p.name, p.price, p.sku
                LIMIT 1
                """,
                query=query,
            )
            record = result.single()
            if record:
                category = self._get_product_category(record["name"])
                return ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=category,
                )

        return None

    def _get_product_category(self, product_name: str) -> str | None:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                RETURN c.name
                """,
                name=product_name,
            )
            record = result.single()
            return record["c.name"] if record else None

    # ── Product Detail ──────────────────────────────────────────────────────

    def get_product_info(self, product_name: str) -> ProductInfo | None:
        """Get full product detail: category path, attributes, policies, relations.

        Args:
            product_name: Exact or resolved product name.

        Returns:
            ProductInfo with category_path, attributes, policies, accessories,
            alternatives. None if product not found.
        """
        with self._driver.session() as session:
            # Verify product exists
            exists = session.run(
                "MATCH (p:Product {name: $name}) RETURN p.price, p.sku",
                name=product_name,
            ).single()
            if not exists:
                return None

        return ProductInfo(
            name=product_name,
            price=exists["p.price"],
            sku=exists["p.sku"],
            category_name=self._get_product_category(product_name),
            category_path=self._get_category_path(product_name),
            attributes=self._get_product_attributes(product_name),
            policies=self._get_product_policies(product_name),
            accessories=self.get_accessories(product_name),
            alternatives=self.get_alternatives(product_name),
        )

    def _get_category_path(self, product_name: str) -> list[str]:
        """Get full ancestor chain from product to root category."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                      -[:CHILD_OF*0..4]->(ancestor:Category)
                RETURN ancestor.name, ancestor.level
                ORDER BY ancestor.level
                """,
                name=product_name,
            )
            return [record["ancestor.name"] for record in result]

    def _get_product_attributes(self, product_name: str) -> list[ProductAttribute]:
        """Get all attributes for a product."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:HAS_ATTRIBUTE]->(av:AttributeValue)
                      -[:OF_TYPE]->(a:Attribute)
                RETURN a.name AS name, a.display_name AS display_name,
                       av.value AS value, a.data_type AS data_type
                """,
                name=product_name,
            )
            return [
                ProductAttribute(
                    name=record["name"],
                    display_name=record["display_name"],
                    value=record["value"],
                    data_type=record["data_type"],
                )
                for record in result
            ]

    def _get_product_policies(self, product_name: str) -> list[PolicySummary]:
        """Get all policies for a product via category inheritance (CHILD_OF*)."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                      -[:CHILD_OF*0..4]->(ancestor:Category)-[:HAS_POLICY]->(pol:Policy)
                RETURN DISTINCT pol.name AS name, pol.policy_type AS policy_type,
                       pol.summary AS summary, pol.details AS details
                """,
                name=product_name,
            )
            return [
                PolicySummary(
                    name=record["name"],
                    policy_type=record["policy_type"],
                    summary=record["summary"],
                    details=record["details"],
                )
                for record in result
            ]

    # ── Product Relations ───────────────────────────────────────────────────

    def get_accessories(self, product_name: str) -> list[ProductRef]:
        """Get accessories for a product."""
        return self._get_related(product_name, "ACCESSORY_OF")

    def get_alternatives(self, product_name: str) -> list[ProductRef]:
        """Get alternative products."""
        return self._get_related(product_name, "ALTERNATIVE_TO")

    def _get_related(
        self, product_name: str, rel_type: str
    ) -> list[ProductRef]:
        """Get products related by a given relationship type.

        Checks both directions (inbound and outbound).
        """
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (p:Product {{name: $name}})-[:{rel_type}]->(related:Product)
                OPTIONAL MATCH (related)-[:IN_CATEGORY]->(cat:Category)
                RETURN related.name AS name, related.price AS price,
                       related.sku AS sku, cat.name AS category_name
                UNION
                MATCH (related:Product)-[:{rel_type}]->(p:Product {{name: $name}})
                OPTIONAL MATCH (related)-[:IN_CATEGORY]->(cat:Category)
                RETURN related.name AS name, related.price AS price,
                       related.sku AS sku, cat.name AS category_name
                """,
                name=product_name,
            )
            return [
                ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=record["category_name"],
                )
                for record in result
            ]

    # ── Filtered Product Search ─────────────────────────────────────────────

    def search_products(
        self,
        category: str | None = None,
        brand: str | None = None,
        max_price: float | None = None,
        attributes: dict[str, str] | None = None,
        limit: int = 10,
    ) -> list[ProductRef]:
        """Search products with optional filters.

        Args:
            category: Category name (includes sub-categories via CHILD_OF*).
            brand: Brand name.
            max_price: Maximum price filter.
            attributes: Dict of attribute_name → value to match.
            limit: Max results.

        Returns:
            List of matching ProductRef.
        """
        with self._driver.session() as session:
            # Build WHERE clauses
            wheres = []
            params: dict = {"limit": limit}

            if category:
                wheres.append(
                    "(c:Category)-[:CHILD_OF*0..4]->(:Category {name: $category})"
                )
                params["category"] = category

            if brand:
                wheres.append("(p)-[:HAS_BRAND]->(:Brand {name: $brand})")
                params["brand"] = brand

            if max_price is not None:
                wheres.append("p.price <= $max_price")
                params["max_price"] = max_price

            where_clause = " AND ".join(wheres) if wheres else "true"

            cypher = f"""
                MATCH (p:Product)-[:IN_CATEGORY]->(:Category)
                WHERE {where_clause}
                OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
                RETURN DISTINCT p.name AS name, p.price AS price,
                       p.sku AS sku, cat.name AS category_name
                ORDER BY p.price DESC
                LIMIT $limit
            """

            result = session.run(cypher, **params)
            products = [
                ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=record["category_name"],
                )
                for record in result
            ]

            # Post-filter by attributes if specified (Cypher attribute filtering
            # is more complex due to the EAV model, so do it client-side)
            if attributes and products:
                filtered = []
                for product in products:
                    attrs = self._get_product_attributes(product.name)
                    attr_map = {a.name: a.value for a in attrs}
                    if all(
                        attr_map.get(k) == v for k, v in attributes.items()
                    ):
                        filtered.append(product)
                return filtered

            return products

    # ── Category Tree ───────────────────────────────────────────────────────

    def get_category_tree(self, root_name: str | None = None) -> list[CategoryRef]:
        """Return the category hierarchy as a nested tree.

        Args:
            root_name: If given, return subtree rooted at this category.
                       If None, return all root categories.

        Returns:
            List of root CategoryRef with nested children.
        """
        with self._driver.session() as session:
            if root_name:
                result = session.run(
                    """
                    MATCH (root:Category {name: $name})
                    OPTIONAL MATCH (root)<-[:CHILD_OF]-(child:Category)
                    RETURN root.name AS name, root.description AS description,
                           root.level AS level, collect(child.name) AS children
                    """,
                    name=root_name,
                )
            else:
                result = session.run(
                    """
                    MATCH (root:Category)
                    WHERE NOT (root)-[:CHILD_OF]->(:Category)
                    OPTIONAL MATCH (root)<-[:CHILD_OF]-(child:Category)
                    RETURN root.name AS name, root.description AS description,
                           root.level AS level, collect(child.name) AS children
                    """
                )

            roots = []
            for record in result:
                cat = CategoryRef(
                    name=record["name"],
                    level=record["level"],
                    description=record["description"],
                )
                # Recursively build children
                if record["children"]:
                    for child_name in record["children"]:
                        child_tree = self.get_category_tree(child_name)
                        cat.children.extend(child_tree)
                roots.append(cat)

            return roots


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: Neo4jStore | None = None


def get_neo4j_store() -> Neo4jStore:
    """Return the shared Neo4jStore singleton."""
    global _store
    if _store is None:
        _store = Neo4jStore()
    return _store
```

- [ ] **Step 2: Verify module imports**

Run: `cd apps/backend && python -c "from backend.knowledge.neo4j_store import get_neo4j_store; store = get_neo4j_store(); print(store.resolve_product('iPhone 15 Pro'))"`
Expected: `ProductRef(name='iPhone 15 Pro', price=999.0, ...)`

- [ ] **Step 3: Write unit tests**

Create: `apps/backend/tests/test_neo4j_store.py`

```python
"""Unit tests for Neo4jStore — requires Neo4j to be running with synced data."""

import pytest
from backend.knowledge.neo4j_store import Neo4jStore


@pytest.fixture
def store():
    """Return a Neo4jStore connected to the test database.

    Precondition: Neo4j is running and neo4j_setup has been run.
    """
    return Neo4jStore()


class TestResolveProduct:
    def test_exact_name(self, store):
        """Exact product name returns correct ProductRef."""
        result = store.resolve_product("iPhone 15 Pro")
        assert result is not None
        assert result.name == "iPhone 15 Pro"
        assert result.price == 999.00

    def test_partial_name(self, store):
        """Partial name via fulltext search."""
        result = store.resolve_product("MacBook")
        assert result is not None
        assert "MacBook" in result.name

    def test_synonym(self, store):
        """Synonym search returns canonical product."""
        result = store.resolve_product("苹果手机")
        assert result is not None
        assert "iPhone" in result.name

    def test_no_match(self, store):
        """Nonexistent product returns None."""
        result = store.resolve_product("nonexistent_xyz_product")
        assert result is None


class TestGetProductInfo:
    def test_has_category_path(self, store):
        """Product info includes full ancestor chain."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        assert "Smartphones" in info.category_path
        assert "Electronics" in info.category_path

    def test_has_attributes(self, store):
        """Product info includes attributes."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        attr_names = [a.name for a in info.attributes]
        assert "brand" in attr_names
        assert "storage" in attr_names

    def test_has_inherited_policies(self, store):
        """Product info includes policies inherited from ancestor categories."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        policy_names = [p.name for p in info.policies]
        assert "electronics_return" in policy_names

    def test_has_accessories(self, store):
        """Product info includes accessories."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        acc_names = [a.name for a in info.accessories]
        assert "iPhone 15 Pro Leather Case" in acc_names


class TestSearchProducts:
    def test_by_category(self, store):
        """Filter products by category (includes sub-categories)."""
        results = store.search_products(category="Smartphones")
        names = [p.name for p in results]
        assert "iPhone 15 Pro" in names
        assert "Google Pixel 8" in names

    def test_by_brand(self, store):
        """Filter products by brand."""
        results = store.search_products(brand="Sony")
        assert len(results) > 0
        assert all(
            "Sony" in p.name for p in results
        )

    def test_by_max_price(self, store):
        """Filter products by max price."""
        results = store.search_products(category="Smartphones", max_price=700)
        assert all(p.price <= 700 for p in results)
```

- [ ] **Step 4: Run tests**

Run: `cd apps/backend && pytest tests/test_neo4j_store.py -v`
Expected: 9 tests pass (precondition: Neo4j running with synced data)

- [ ] **Step 5: Commit**

```bash
git add apps/backend/backend/knowledge/neo4j_store.py apps/backend/tests/test_neo4j_store.py
git commit -m "feat: add Neo4jStore with typed Cypher query methods and tests"
```

---

## Phase 4: Product Descriptions + LlamaIndex Ingestion

### Task 4.1: Create product_descriptions.txt

**Files:**
- Create: `apps/backend/data/product_descriptions.txt`

- [ ] **Step 1: Write product_descriptions.txt with 3 products**

```text
[product: iPhone 15 Pro]
category: Smartphones
brand: Apple
price: 999.00
color: Titanium
storage: 256GB
screen_size: 6.1
weight: 0.187
release_year: 2023
wireless: true
warranty_years: 1

iPhone 15 Pro features a 6.1-inch Super Retina XDR display with ProMotion
technology delivering adaptive refresh rates up to 120Hz. The display reaches
2000 nits peak brightness outdoors, making it clearly visible in direct sunlight.

The A17 Pro chip is the industry's first 3-nanometer chip, delivering console-level
gaming performance with hardware-accelerated ray tracing. The 6-core GPU is 20%
faster than the previous generation, while the 6-core CPU includes 2 performance
cores and 4 efficiency cores for all-day battery life.

The pro camera system features a 48MP main camera with a quad-pixel sensor that
defaults to 24MP for optimal detail and light capture. Multiple focal lengths —
24mm, 28mm, and 35mm — are available at the main camera level. The new 5x Telephoto
camera at 120mm provides exceptional reach. Night mode, Portrait mode, and Photonic
Engine computational photography are available across all cameras.

MagSafe wireless charging supports up to 15W with compatible accessories. The
titanium design is both stronger and lighter than previous models at 187 grams.
USB-C with USB 3 speeds up to 10Gb/s enables fast data transfer. The Action button
replaces the mute switch with customizable shortcuts. Battery life lasts up to
29 hours of video playback. Face ID is faster and more secure with the new
Secure Enclave. The phone is rated IP68 for water and dust resistance, surviving
up to 6 meters for 30 minutes.

[product: Google Pixel 8]
category: Smartphones
brand: Google
price: 699.00
color: Obsidian
storage: 128GB
screen_size: 6.2
release_year: 2023
wireless: true

Google Pixel 8 features a 6.2-inch Actua OLED display with a smooth 120Hz
adaptive refresh rate. The display reaches 2000 nits peak brightness for
excellent outdoor visibility. The Tensor G3 chip powers advanced AI features
including Magic Eraser, Best Take, and Audio Magic Eraser for photo and video
enhancements.

The camera system includes a 50MP main sensor with Octa PD autofocus and a
12MP ultrawide lens with autofocus for Macro Focus. Google's computational
photography delivers Real Tone, Night Sight, and Astrophotography modes.
Super Res Zoom provides up to 8x magnification with AI-enhanced clarity.

The Pixel 8 runs Android 14 with 7 years of OS, security, and Feature Drop
updates guaranteed. The Titan M2 security chip protects sensitive data. Battery
life lasts over 24 hours with Adaptive Battery, extending to 72 hours with
Extreme Battery Saver. Fast wireless charging is supported up to 18W with
Pixel Stand. The phone supports Wi-Fi 7, Bluetooth 5.3, and 5G connectivity.
Face Unlock and an under-display fingerprint sensor provide biometric security.

[product: MacBook Pro 16]
category: Laptops
brand: Apple
price: 2499.00
color: Space Gray
storage: 512GB
screen_size: 16.2
weight: 2.15
release_year: 2023
wireless: true

MacBook Pro 16 features a 16.2-inch Liquid Retina XDR display with mini-LED
technology delivering 1000 nits sustained brightness and 1600 nits peak for
HDR content. ProMotion adaptive refresh rates up to 120Hz ensure fluid scrolling
and responsiveness. The display supports the P3 wide color gamut and True Tone.

Powered by the M3 Pro chip with a 12-core CPU (6 performance, 6 efficiency)
and 18-core GPU, the MacBook Pro 16 delivers exceptional performance for
professional workflows. The 16-core Neural Engine accelerates machine learning
tasks by up to 40% compared to M1 Pro. Unified memory of 18GB ensures smooth
multitasking across demanding applications.

Battery life delivers up to 22 hours of video playback, the longest in any
Mac. MagSafe 3 charging provides a secure magnetic connection with fast
charging capability. The six-speaker sound system with force-cancelling woofers
delivers room-filling audio with support for Spatial Audio and Dolby Atmos.

Connectivity includes three Thunderbolt 4 ports, HDMI, SDXC card slot,
MagSafe 3, and a 3.5mm headphone jack with high-impedance support. Wi-Fi 6E
and Bluetooth 5.3 provide cutting-edge wireless. The 1080p FaceTime HD camera
with advanced image signal processor ensures sharp video calls. The backlit
Magic Keyboard with full-height function keys and Touch ID completes the
professional package.
```

- [ ] **Step 2: Commit**

```bash
git add apps/backend/data/product_descriptions.txt
git commit -m "data: add product descriptions for iPhone 15 Pro, Pixel 8, MacBook Pro 16"
```

---

### Task 4.2: Create LlamaIndex ingestion pipeline

**Files:**
- Create: `apps/backend/backend/rag/__init__.py`
- Create: `apps/backend/backend/rag/ingestion.py`

- [ ] **Step 1: Write __init__.py**

```python
"""LlamaIndex RAG module for product description ingestion and querying."""
```

- [ ] **Step 2: Write ingestion.py**

```python
"""
LlamaIndex ingestion pipeline for product descriptions.

Parses product_descriptions.txt, chunks with SentenceWindowNodeParser,
embeds with Zhipu embedding-2, and stores in pgvector product_chunks collection.

Usage:
    python -m backend.rag.ingestion
"""

import re
import logging

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PRODUCT_DESCRIPTIONS_PATH = "data/product_descriptions.txt"
COLLECTION_NAME = "product_chunks"
SENTENCE_WINDOW_SIZE = 5
SENTENCE_WINDOW_OVERLAP = 1
CHUNK_SIZE = 64  # batch size for embedding API

# ── Parser ────────────────────────────────────────────────────────────────────

_BLOCK_HEADER_RE = re.compile(r"^\[product:\s*(.+?)\]$")
_METADATA_RE = re.compile(r"^(\w[\w\s]*?):\s*(.+)$")


def parse_product_descriptions(filepath: str) -> list[Document]:
    """Parse [product: Name] blocks with metadata headers and body text.

    Format:
        [product: iPhone 15 Pro]
        category: Smartphones
        brand: Apple
        price: 999.00

        The product description body text follows the metadata lines...

    Args:
        filepath: Path to product_descriptions.txt.

    Returns:
        List of LlamaIndex Document objects with metadata attached.
    """
    documents: list[Document] = []
    current_meta: dict[str, str] = {}
    current_name: str | None = None
    current_body: list[str] = []

    def _flush():
        """Save accumulated product block as a Document."""
        nonlocal current_meta, current_name, current_body
        if current_name and current_body:
            text = "\n".join(current_body).strip()
            if text:
                num_meta = {
                    "product_name": current_name,
                    "brand": current_meta.get("brand", ""),
                    "category": current_meta.get("category", ""),
                    "color": current_meta.get("color", ""),
                    "storage": current_meta.get("storage", ""),
                }
                # Convert numeric metadata
                if "price" in current_meta:
                    try:
                        num_meta["price"] = float(current_meta["price"])
                    except ValueError:
                        pass
                if "screen_size" in current_meta:
                    try:
                        num_meta["screen_size"] = float(
                            current_meta["screen_size"]
                        )
                    except ValueError:
                        pass
                if "release_year" in current_meta:
                    try:
                        num_meta["release_year"] = int(
                            current_meta["release_year"]
                        )
                    except ValueError:
                        pass
                if "wireless" in current_meta:
                    num_meta["wireless"] = (
                        current_meta["wireless"].lower() == "true"
                    )

                documents.append(
                    Document(text=text, metadata=num_meta)
                )
                logger.debug(
                    "Parsed product: %s (%d chars, %d metadata fields)",
                    current_name,
                    len(text),
                    len(num_meta),
                )

        current_meta = {}
        current_name = None
        current_body = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # Empty line — could be separator between metadata and body
            if not line:
                if current_name and current_meta and not current_body:
                    continue  # skip blank lines between metadata and body
                if current_body:
                    current_body.append("")  # preserve paragraph breaks
                continue

            # New product block header
            header_match = _BLOCK_HEADER_RE.match(line)
            if header_match:
                _flush()
                current_name = header_match.group(1)
                continue

            # Metadata line (key: value) — only before body starts
            meta_match = _METADATA_RE.match(line)
            if meta_match and not current_body:
                key = meta_match.group(1).strip().lower().replace(" ", "_")
                value = meta_match.group(2).strip()
                current_meta[key] = value
                continue

            # Body text
            if current_name:
                current_body.append(line)

    _flush()  # Don't forget the last block
    logger.info(
        "Parsed %d product documents from %s", len(documents), filepath
    )
    return documents


# ── Ingestion Pipeline ─────────────────────────────────────────────────────────

def build_product_index():
    """Run the full ingestion pipeline: parse → chunk → embed → store.

    Creates or replaces the 'product_chunks' collection in pgvector.
    """
    logger.info("Starting product description ingestion...")

    # Parse
    documents = parse_product_descriptions(PRODUCT_DESCRIPTIONS_PATH)
    if not documents:
        logger.warning("No product documents found in %s", PRODUCT_DESCRIPTIONS_PATH)
        return

    # Create embedding model (Zhipu via OpenAI-compatible API)
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        embed_batch_size=CHUNK_SIZE,
    )

    # Create vector store (pgvector)
    vector_store = PGVectorStore.from_params(
        database=settings.database_url.split("/")[-1],
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        table_name=COLLECTION_NAME,
        embed_dim=1024,
        perform_setup=False,  # table created by pgvector setup
    )

    # Build pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceWindowNodeParser(
                window_size=SENTENCE_WINDOW_SIZE,
                window_overlap=SENTENCE_WINDOW_OVERLAP,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Run
    nodes = pipeline.run(documents=documents)
    logger.info(
        "Ingestion complete: %d documents → %d nodes stored in %s",
        len(documents),
        len(nodes),
        COLLECTION_NAME,
    )

    return nodes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_product_index()
```

- [ ] **Step 3: Run ingestion**

Precondition: PostgreSQL is running with pgvector extension.

Run: `cd apps/backend && python -m backend.rag.ingestion`
Expected: log output showing parsed documents, nodes created, no errors.

- [ ] **Step 4: Write unit test for parser**

Create: `apps/backend/tests/test_rag_ingestion.py`

```python
"""Unit tests for product description parsing and ingestion."""

import tempfile
from pathlib import Path

from backend.rag.ingestion import parse_product_descriptions


SAMPLE_CONTENT = """[product: Test Phone]
category: Smartphones
brand: TestCorp
price: 599.00
color: Red
storage: 128GB

Test Phone features a vibrant display and long battery life.
The camera system captures stunning photos in any light.

Wireless charging is supported with Qi-compatible accessories.
Battery lasts up to 20 hours of typical usage.
"""


class TestParseProductDescriptions:
    def test_parses_single_product(self):
        """Single product block produces one Document with metadata."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(SAMPLE_CONTENT)
            tmp_path = f.name

        try:
            docs = parse_product_descriptions(tmp_path)
            assert len(docs) == 1
            doc = docs[0]

            assert doc.metadata["product_name"] == "Test Phone"
            assert doc.metadata["brand"] == "TestCorp"
            assert doc.metadata["category"] == "Smartphones"
            assert doc.metadata["price"] == 599.00
            assert doc.metadata["color"] == "Red"

            assert "vibrant display" in doc.text
            assert "Wireless charging" in doc.text
        finally:
            Path(tmp_path).unlink()

    def test_parses_multiple_products(self):
        """Multiple product blocks produce multiple Documents."""
        content = (
            SAMPLE_CONTENT
            + "\n[product: Test Tablet]\ncategory: Tablets\nbrand: TestCorp\n"
            "price: 399.00\n\nA lightweight tablet for everyday use.\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            docs = parse_product_descriptions(tmp_path)
            assert len(docs) == 2
            assert docs[0].metadata["product_name"] == "Test Phone"
            assert docs[1].metadata["product_name"] == "Test Tablet"
        finally:
            Path(tmp_path).unlink()
```

- [ ] **Step 5: Run tests**

Run: `cd apps/backend && pytest tests/test_rag_ingestion.py -v`
Expected: 2 tests pass

- [ ] **Step 6: Commit**

```bash
git add apps/backend/backend/rag/ apps/backend/tests/test_rag_ingestion.py
git commit -m "feat: add LlamaIndex ingestion pipeline for product descriptions"
```

---

## Phase 5: LlamaIndex Query Engine

### Task 5.1: Create query engine factory

**Files:**
- Create: `apps/backend/backend/rag/query_engine.py`

- [ ] **Step 1: Write query_engine.py**

```python
"""
LlamaIndex query engine for product description RAG.

Provides metadata-filtered vector search over product_chunks
with sentence-window context expansion.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from backend.config import settings

COLLECTION_NAME = "product_chunks"
DEFAULT_TOP_K = 5


def _build_vector_store() -> PGVectorStore:
    """Create a PGVectorStore connected to the product_chunks collection."""
    return PGVectorStore.from_params(
        database=settings.database_url.split("/")[-1],
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        table_name=COLLECTION_NAME,
        embed_dim=1024,
        perform_setup=False,
    )


def create_product_query_engine() -> RetrieverQueryEngine:
    """Factory: build a metadata-filtered query engine for product Q&A.

    Returns a RetrieverQueryEngine configured with:
      - Vector search over product_chunks in pgvector
      - Sentence-window context expansion (MetadataReplacementPostProcessor)
      - Metadata filtering support (filter by product_name, brand, category)
    """
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        embed_batch_size=64,
    )

    vector_store = _build_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
    )


def create_filtered_query_engine(
    product_names: list[str] | None = None,
    brand: str | None = None,
    category: str | None = None,
) -> RetrieverQueryEngine:
    """Factory: build a query engine with metadata filters applied.

    Args:
        product_names: If provided, limit search to these products' chunks.
        brand: If provided, filter by brand.
        category: If provided, filter by category.

    Returns:
        RetrieverQueryEngine with metadata filters pre-applied.
    """
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        embed_batch_size=64,
    )

    vector_store = _build_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # Build metadata filters
    filters_list = []
    if product_names:
        filters_list.append(
            MetadataFilter(
                key="product_name",
                value=product_names,
                operator=FilterOperator.IN,
            )
        )
    if brand:
        filters_list.append(
            MetadataFilter(
                key="brand",
                value=brand,
                operator=FilterOperator.EQ,
            )
        )
    if category:
        filters_list.append(
            MetadataFilter(
                key="category",
                value=category,
                operator=FilterOperator.EQ,
            )
        )

    filters = MetadataFilters(filters=filters_list) if filters_list else None

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=DEFAULT_TOP_K,
        filters=filters,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
        ],
    )
```

- [ ] **Step 2: Verify query engine creates without errors**

Run:
```bash
cd apps/backend && python -c "
from backend.rag.query_engine import create_product_query_engine
engine = create_product_query_engine()
print('Query engine created successfully')
"
```
Expected: `Query engine created successfully`

- [ ] **Step 3: Commit**

```bash
git add apps/backend/backend/rag/query_engine.py
git commit -m "feat: add LlamaIndex product query engine factory with metadata filtering"
```

---

## Phase 6: Product QA Tool

### Task 6.1: Create product_qa_tool

**Files:**
- Create: `apps/backend/backend/tools/product_qa.py`

- [ ] **Step 1: Write product_qa.py**

```python
"""
Product QA tool — orchestrates Neo4j graph traversal + LlamaIndex RAG.

Handles three query patterns:
  1. Single-product: "Does iPhone 15 Pro have MagSafe?"
     → Neo4j resolve → LlamaIndex RAG (filtered to product) → synthesize
  2. Cross-product: "Which phone under $800 has the best camera?"
     → Neo4j candidate search → LlamaIndex RAG (per candidate) → compare
  3. Category/definition: "What category is iPhone 15 Pro in?"
     → Neo4j category path → synthesize with attributes

The tool is the single integration point between Neo4j, LlamaIndex, and the LLM.
"""

import logging

from langchain.tools import tool

from backend.agent import llm
from backend.knowledge.neo4j_store import get_neo4j_store
from backend.rag.query_engine import create_filtered_query_engine

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORY_PATH_SIGNALS = {
    "category", "categories", "what kind", "what type",
    "classified", "belong to", "product line",
}

CROSS_PRODUCT_SIGNALS = {
    "which", "compare", "vs", "versus", "best", "cheapest",
    "under", "within budget", "recommend", "alternative",
}


@tool
def product_qa_tool(query: str) -> str:
    """Answer product questions using the knowledge graph and product descriptions.

    Use this tool when the user asks about:
    - Product specifications, features, or capabilities
      ("Does iPhone 15 Pro have MagSafe?", "How much does MacBook weigh?")
    - Product comparisons
      ("Which phone has the best camera under $800?", "Compare MacBook vs Dell XPS")
    - Category questions
      ("What category is iPhone in?", "Why is this product in Flagship Phones?")
    - Recommendations
      ("What's a good Android phone?", "What accessories for iPhone 15 Pro?")

    Args:
        query: The user's question about a product.

    Returns:
        Formatted answer synthesized from graph data and product descriptions.
    """
    store = get_neo4j_store()
    lowered = query.lower()

    # ── Pattern 1: Category/path queries ──────────────────────────────
    if any(signal in lowered for signal in CATEGORY_PATH_SIGNALS):
        return _answer_category_query(query, store)

    # ── Pattern 2: Cross-product comparison ───────────────────────────
    if any(signal in lowered for signal in CROSS_PRODUCT_SIGNALS):
        return _answer_comparison_query(query, store)

    # ── Pattern 3: Cross-product: "recommend a phone" ─────────────────
    if any(w in lowered for w in ("recommend", "suggestion", "suggest")):
        return _answer_recommendation_query(query, store)

    # ── Default: Single-product detail query ──────────────────────────
    return _answer_single_product_query(query, store)


# ── Query Handlers ────────────────────────────────────────────────────────────


def _answer_category_query(query: str, store) -> str:
    """Answer 'what category is X in?' with reasoning from attributes."""
    # Extract product name: resolve the query against known products
    product = _resolve_product_from_query(query, store)
    if not product:
        return f"I couldn't identify a specific product in your question: '{query}'. Could you specify which product you're asking about?"

    info = store.get_product_info(product.name)
    if not info or not info.category_path:
        return f"I found {product.name}, but couldn't determine its category path."

    # Format category path with explanations derived from attributes
    path_str = " → ".join(info.category_path)
    attr_lines = ""
    if info.attributes:
        attr_lines = "\n\nProduct attributes:\n"
        for attr in info.attributes:
            if attr.name in ("brand", "price", "storage", "screen_size", "color"):
                attr_lines += f"  - {attr.display_name}: {attr.value}"
                if attr.unit:
                    attr_lines += f" {attr.unit}"
                attr_lines += "\n"

    prompt = (
        f"The product '{product.name}' belongs to this category path: {path_str}."
        f"{attr_lines}\n\n"
        f"User question: {query}\n\n"
        f"Explain which category this product belongs to and why, "
        f"based on its position in the category hierarchy and its attributes. "
        f"Be concise and helpful."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_single_product_query(query: str, store) -> str:
    """Answer a question about a specific product using RAG over its description."""
    product = _resolve_product_from_query(query, store)
    if not product:
        return _fallback_search(query, store)

    # Get graph context (attributes, category, policies)
    info = store.get_product_info(product.name)

    # Get RAG context from LlamaIndex (filtered to this product)
    rag_context = _retrieve_product_chunks(query, product_names=[product.name])

    # Synthesize
    graph_context = ""
    if info:
        graph_context = f"Product: {info.name}\n"
        if info.category_path:
            graph_context += f"Category: {' → '.join(info.category_path)}\n"
        if info.attributes:
            graph_context += "Attributes:\n"
            for attr in info.attributes:
                graph_context += f"  - {attr.display_name}: {attr.value}"
                if attr.unit:
                    graph_context += f" {attr.unit}"
                graph_context += "\n"

    prompt = (
        f"You are a product expert. Answer the user's question about "
        f"'{product.name}' using the information below.\n\n"
        f"Structured product data:\n{graph_context}\n"
        f"Product description excerpts:\n{rag_context}\n\n"
        f"User question: {query}\n\n"
        f"Answer the question accurately. If the information is insufficient, "
        f"say so — don't guess. If relevant, mention where the product sits "
        f"in the category hierarchy."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_comparison_query(query: str, store) -> str:
    """Handle cross-product comparison queries.

    Strategy:
      1. Extract constraints from the query (category, brand, max_price)
      2. Neo4j search → candidate products
      3. LlamaIndex RAG on each candidate for the relevant feature
      4. LLM compares and recommends
    """
    # Extract category hint from query
    category_hint = _extract_category_hint(query)

    # Search for candidates
    candidates = store.search_products(
        category=category_hint,
        limit=5,
    )

    if not candidates:
        return f"I couldn't find any products matching your criteria in '{query}'."

    if len(candidates) == 1:
        return _answer_single_product_query(
            f"Tell me about {candidates[0].name}", store
        )

    # Get RAG context for each candidate
    context_parts = []
    for candidate in candidates:
        info = store.get_product_info(candidate.name)
        price_str = f"${candidate.price:.2f}" if candidate.price else "N/A"
        context_parts.append(
            f"\n--- {candidate.name} ({price_str}) ---\n"
            f"Category: {candidate.category_name or 'N/A'}\n"
        )
        # Get relevant description chunks
        rag_chunks = _retrieve_product_chunks(
            query, product_names=[candidate.name]
        )
        context_parts.append(rag_chunks or "No detailed description available.")

    combined_context = "\n".join(context_parts)

    prompt = (
        f"You are a product comparison expert. Compare these products "
        f"based on the user's question and recommend the best option.\n\n"
        f"Products:\n{combined_context}\n\n"
        f"User question: {query}\n\n"
        f"Compare the relevant features, explain trade-offs, and give a clear "
        f"recommendation with reasoning. Be fair and honest about each product's "
        f"strengths and weaknesses."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def _answer_recommendation_query(query: str, store) -> str:
    """Handle 'recommend me a phone' type queries — same as comparison
    but without the side-by-side comparison framing."""
    return _answer_comparison_query(query, store)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_product_from_query(query: str, store) -> object | None:
    """Try to identify which product the user is asking about.

    Strategy: try each word/phrase as a product name lookup.
    Uses the Neo4j fulltext index which includes search_terms (synonyms).
    """
    # Try the full query first (for queries that are just a product name)
    result = store.resolve_product(query)
    if result:
        return result

    # Try progressively shorter substrings
    words = query.split()
    for n in range(len(words), 0, -1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i : i + n])
            # Skip very short candidates (likely noise)
            if len(candidate) < 3:
                continue
            result = store.resolve_product(candidate)
            if result:
                return result

    return None


def _extract_category_hint(query: str) -> str | None:
    """Extract a likely category from the query for filtering.

    Simple keyword mapping — the LLM-based route classifier
    should handle this, but this serves as a fast deterministic fallback.
    """
    category_keywords = {
        "phone": "Smartphones",
        "smartphone": "Smartphones",
        "laptop": "Laptops",
        "notebook": "Laptops",
        "tablet": "Tablets",
        "headphone": "Headphones",
        "earbud": "Wireless Earbuds",
        "speaker": "Speakers",
        "keyboard": "Peripherals",
        "mouse": "Peripherals",
        "monitor": "Peripherals",
        "camera": "Smartphones",  # default to phone cameras
        "sneaker": "Sneakers",
        "shoe": "Sneakers",
        "coffee": "Coffee Makers",
        "blender": "Blenders",
    }
    lowered = query.lower()
    for keyword, category in category_keywords.items():
        if keyword in lowered:
            return category
    return None


def _retrieve_product_chunks(
    query: str, product_names: list[str] | None = None
) -> str:
    """Retrieve relevant chunks from product descriptions via LlamaIndex.

    Args:
        query: The user's question.
        product_names: If provided, filter to only these products' chunks.

    Returns:
        Formatted string of retrieved chunks, or empty string on failure.
    """
    try:
        engine = create_filtered_query_engine(product_names=product_names)
        response = engine.query(query)
        return str(response) if response else ""
    except Exception as e:
        logger.warning("[product_qa] RAG retrieval failed: %s", e)
        return ""


def _fallback_search(query: str, store) -> str:
    """Fallback: do a general product search when no specific product matched."""
    products = store.search_products(limit=5)
    if not products:
        return f"I couldn't find any products matching '{query}'. Could you be more specific about which product you're asking about?"

    names = [p.name for p in products]
    return _answer_comparison_query(query, store)
```

- [ ] **Step 2: Verify the tool imports and registers correctly**

Run:
```bash
cd apps/backend && python -c "
from backend.tools.product_qa import product_qa_tool
print(f'Tool: {product_qa_tool.name}')
print(f'Description: {product_qa_tool.description[:100]}...')
"
```
Expected: Tool name and description printed

- [ ] **Step 3: Write integration tests**

Create: `apps/backend/tests/test_product_qa_tool.py`

```python
"""Integration tests for product_qa_tool.

Requires: Neo4j running with synced data, PostgreSQL running with
product_chunks ingested.
"""

import pytest
from backend.tools.product_qa import product_qa_tool


class TestProductQATool:
    def test_single_product_feature_query(self):
        """Direct product feature question returns relevant answer."""
        result = product_qa_tool.invoke(
            {"query": "Does iPhone 15 Pro have MagSafe?"}
        )
        assert result is not None
        assert len(result) > 20
        # Should mention MagSafe or wireless charging
        assert "MagSafe" in result or "wireless" in result.lower()

    def test_category_query(self):
        """Category question returns category information."""
        result = product_qa_tool.invoke(
            {"query": "What category is iPhone 15 Pro in?"}
        )
        assert result is not None
        assert len(result) > 20
        assert "Smartphones" in result or "Flagship" in result

    def test_comparison_query(self):
        """Comparison question returns multi-product analysis."""
        result = product_qa_tool.invoke(
            {"query": "Which phone is better, iPhone or Pixel?"}
        )
        assert result is not None
        assert len(result) > 30

    def test_unknown_product_graceful(self):
        """Unknown product returns helpful message, not error."""
        result = product_qa_tool.invoke(
            {"query": "Tell me about the FooBar X9000"}
        )
        assert result is not None
        assert len(result) > 10
        # Should not be an error traceback
        assert "Traceback" not in result
```

- [ ] **Step 4: Run integration tests**

Run: `cd apps/backend && pytest tests/test_product_qa_tool.py -v`
Expected: 4 tests pass (precondition: Neo4j + PostgreSQL + product_chunks all available)

- [ ] **Step 5: Commit**

```bash
git add apps/backend/backend/tools/product_qa.py apps/backend/tests/test_product_qa_tool.py
git commit -m "feat: add product_qa_tool orchestrating Neo4j + LlamaIndex + LLM"
```

---

## Phase 7: Agent Wiring

### Task 7.1: Add product_qa intent to keyword classifier

**Files:**
- Modify: `apps/backend/backend/intent/keyword.py`

- [ ] **Step 1: Add PRODUCT_QA_SIGNALS and update _classify_with_entities**

Add the signal set after the existing `_ORDER_ACTION_SIGNALS` (line 34) and insert a product_qa detection rule in `_classify_with_entities` after the weather check (line 54) and before the policy check (line 60).

The `_classify_with_entities` function already receives `entities` dict with `products`, `categories`, `order_ids` from `extract_entities()`. No extra KG import needed.

```python
# After line 34, add:
_PRODUCT_QA_SIGNALS = {
    "does the", "do the", "can the", "how much", "how many",
    "feature", "spec", "specification", "battery", "camera",
    "weight", "size", "screen", "storage", "color", "magsafe",
    "compare", "vs", "versus", "better", "best", "recommend",
    "difference between",
}
```

In `_classify_with_entities`, after the weather check block (line 54) and before the `# List all orders` check, add:

```python
    # Product QA signals + product or generic product term
    # Placed BEFORE policy signals so "does iPhone have MagSafe?" → product_qa, not policy
    qa_hits = [w for w in _PRODUCT_QA_SIGNALS if w in lowered]
    generic_terms = {"phone", "laptop", "tablet", "earbud", "headphone",
                     "speaker", "sneaker", "shirt", "watch"}
    has_generic_product = bool(generic_terms & set(lowered.split()))
    if qa_hits and (has_product or has_generic_product):
        return {
            "intent": "product_qa",
            "confidence": "high",
            "source": "entity+keyword",
            "entities": {"products": entities["products"],
                         "categories": entities["categories"]},
        }
```

- [ ] **Step 2: Update LLM fallback prompt to include product_qa**

Modify `_INTENT_PROMPT` (line 110) to add the new intent option:

```python
_INTENT_PROMPT = """You are an intent classifier for an e-commerce support chatbot.
Classify the user's query into exactly one of these categories:

- order: user asks about a specific order by ID (e.g., "where is order 1001?")
- list_orders: user wants to see all their orders (e.g., "show my orders")
- policy: user asks about store policies — returns, refunds, shipping, warranty
- product_qa: user asks about a product's features, specs, comparisons, or category
- weather: user asks about weather in a city
- knowledge: user asks about product listings or category info (not specs)
- unknown: greeting, small talk, or anything else

{entity_context}
Respond with ONLY a JSON object (no markdown, no explanation):
{{"intent": "<category>", "confidence": "high|medium|low", "reason": "one-line reason"}}

User: {query}
"""
```

- [ ] **Step 3: Add product_qa to llm_hybrid.py and semantic.py**

For `llm_hybrid.py`: add the same `_PRODUCT_QA_SIGNALS` and detection rule in its `_classify_with_entities` equivalent, plus update its LLM prompt.

For `semantic.py`: update its `_SEMANTIC_PROMPT` to include `product_qa` as a valid intent (alongside `knowledge`). The semantic path already returns `knowledge` for product queries — `product_qa` is a narrower subset for spec/feature/comparison questions.

- [ ] **Step 5: Run existing intent tests to verify no regression**

Run: `cd apps/backend && pytest tests/ -k "intent" -v`
Expected: existing intent tests still pass

- [ ] **Step 6: Commit**

```bash
git add apps/backend/backend/intent/keyword.py apps/backend/backend/intent/llm_hybrid.py apps/backend/backend/intent/semantic.py
git commit -m "feat: add product_qa intent detection to all classifiers"
```

---

### Task 7.2: Add product_qa node to LangGraph agent

**Files:**
- Modify: `apps/backend/backend/graph/nodes.py`
- Modify: `apps/backend/backend/graph/agent_graph.py`

- [ ] **Step 1: Add product_qa_node function in nodes.py**

Add after `knowledge_node` (after line 249). No new imports needed — `product_qa_tool` is imported inline to avoid circular imports:

```python
# ─── Node: product_qa ────────────────────────────────────────────────────────

def product_qa_node(state: AgentState) -> AgentState:
    """Answer product questions using Neo4j graph + LlamaIndex RAG.

    Delegates to the product_qa_tool which orchestrates:
      - Neo4j for structured graph queries (category, attributes, relations)
      - LlamaIndex for semantic search over product descriptions
      - LLM for answer synthesis
    """
    from backend.tools.product_qa import product_qa_tool  # lazy import

    query = state.get("user_input", "")
    result = product_qa_tool.invoke({"query": query})
    state["tool_result"] = result
    logger.info("[graph] Product QA result: %s", result[:80])
    return state
```

- [ ] **Step 2: Update route_by_intent in nodes.py (line 118-132)**

Modify the return type `Literal` and the conditional:

```python
def route_by_intent(state: AgentState) -> Literal[
    "order", "list_orders", "policy", "weather", "knowledge",
    "product_qa", "generate_reply"
]:
    """Conditional edge: decide next node based on intent."""
    if state.get("cached"):
        logger.info("[graph] Route: cached -> generate_reply")
        return "generate_reply"

    intent = state.get("intent", "unknown")
    if intent in ("order", "list_orders", "policy", "weather",
                   "knowledge", "product_qa"):
        logger.info("[graph] Route: %s", intent)
        return cast(Literal[
            "order", "list_orders", "policy", "weather", "knowledge",
            "product_qa", "generate_reply"
        ], intent)

    logger.info("[graph] Route: unknown -> generate_reply")
    return "generate_reply"
```

- [ ] **Step 3: Register product_qa_node in agent_graph.py `_build_graph` (line 60)**

Three changes in the builder function:

Add node registration after `knowledge_node` (after line 71):
```python
    builder.add_node("product_qa_node", product_qa_node)
```

Add import of `product_qa_node` in the import block at line 18:
```python
from backend.graph.nodes import (
    AgentState,
    classify_intent,
    generate_reply,
    knowledge_node,
    list_orders_node,
    order_node,
    policy_node,
    product_qa_node,
    route_after_validation,
    route_by_intent,
    sanitize_input,
    update_memory,
    validate_reply,
    weather_node,
)
```

Add route in the conditional edges (line 83-94, add `"product_qa": "product_qa_node"`):
```python
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "order": "order_node",
            "list_orders": "list_orders_node",
            "policy": "policy_node",
            "weather": "weather_node",
            "knowledge": "knowledge_node",
            "product_qa": "product_qa_node",
            "generate_reply": "generate_reply",
        },
    )
```

Add edge to generate_reply (after line 101):
```python
    builder.add_edge("product_qa_node", "generate_reply")
```

- [ ] **Step 5: Verify the graph compiles**

Run:
```bash
cd apps/backend && python -c "
from backend.graph.agent_graph import build_agent_graph
graph = build_agent_graph()
print('Graph compiled successfully')
print('Nodes:', list(graph.nodes.keys()))
"
```
Expected: `product_qa` appears in the node list

- [ ] **Step 6: Commit**

```bash
git add apps/backend/backend/graph/nodes.py apps/backend/backend/graph/agent_graph.py
git commit -m "feat: add product_qa node to LangGraph agent DAG"
```

---

### Task 7.3: Export product_qa_tool from tools/__init__.py

**Files:**
- Modify: `apps/backend/backend/tools/__init__.py`

- [ ] **Step 1: Add import**

```python
from backend.tools.product_qa import product_qa_tool

__all__ = [
    ...
    "product_qa_tool",
]
```

- [ ] **Step 2: Commit**

```bash
git add apps/backend/backend/tools/__init__.py
git commit -m "feat: export product_qa_tool from tools package"
```

---

## Phase 8: End-to-End Verification

### Task 8.1: Full setup and smoke test

- [ ] **Step 1: Start infrastructure**

```bash
cd apps/backend && docker compose up -d
```

Expected: both `pgvector` and `neo4j` containers healthy

- [ ] **Step 2: Run full setup sequence**

```bash
cd apps/backend
python -m backend.knowledge.schema      # PG tables + seed
python -m backend.knowledge.neo4j_setup  # PG → Neo4j sync
python -m backend.rag.ingestion         # product descriptions → pgvector
```

Expected: all three commands complete without errors

- [ ] **Step 3: Start the API server**

```bash
cd apps/backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000 &
sleep 3
```

- [ ] **Step 4: Test via curl**

```bash
# Test 1: Product feature question
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Does iPhone 15 Pro have MagSafe?"}' | python -m json.tool

# Test 2: Category question
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What category is MacBook Pro 16 in?"}' | python -m json.tool

# Test 3: Comparison question
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which has a better camera, iPhone or Pixel?"}' | python -m json.tool
```

Expected: all three return coherent, factual answers (no "I don't know" unless the data is genuinely missing, no errors)

- [ ] **Step 5: Run full test suite**

```bash
cd apps/backend
ruff check backend/ tests/ --fix
ruff format backend/ tests/
mypy backend/
pytest tests/ -v
```

Expected: ruff clean, mypy zero errors, all tests pass

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "feat: complete multi-framework RAG + knowledge graph integration

- Neo4j graph sync from PostgreSQL seed data
- Neo4jStore with typed Cypher query methods
- LlamaIndex ingestion pipeline (SentenceWindowNodeParser)
- Metadata-filtered vector search over product descriptions
- product_qa_tool orchestrating Neo4j + LlamaIndex + LLM
- product_qa intent detection in all classifiers
- product_qa node in LangGraph agent DAG"
```
