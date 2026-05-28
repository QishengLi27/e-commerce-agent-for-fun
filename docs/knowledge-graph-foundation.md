# Knowledge Graphs — Foundations, How They Work, and When to Use Them

Taught from your actual implementation at `backend/knowledge/`.

---

## Part 1: What a Knowledge Graph Is

### The core idea

A knowledge graph stores **facts as relationships between things**, not as text to search.

```
Vector DB:         "return policy for electronics is 14 days" → [0.023, -0.451, ...]
Knowledge Graph:   Headphones ──belongs_to──→ Audio ──governed_by──→ 14-day return
```

The difference is what you can **do** with the data:

| Operation | Vector DB | Knowledge Graph |
|-----------|-----------|-----------------|
| Find policies about returns | Semantic search: "return policy" → similar vectors | Not applicable — text search isn't the graph's job |
| What policy applies to headphones? | Hope "headphones" and "return" are close in vector space | Traverse: headphones → Audio → 14-day return. **Exact.** |
| Are laptops and headphones covered by the same policy? | Impossible — no relationship data | Traverse both paths, compare results. **Deterministic.** |
| What happens if I add a new product? | Nothing — must re-embed the policy text | Add a row to `products`. Policies come from its category. **Automatic.** |

### The three building blocks

Every knowledge graph has only three concepts:

**1. Nodes — the things**

In your implementation, nodes are rows in three PostgreSQL tables:

```
categories table:          products table:             policy_rules table:
┌────┬──────────────┐     ┌────┬──────────┬──────┐   ┌────┬─────────────────┬──────┐
│ id │ name         │     │ id │ name     │ cat  │   │ id │ name            │ type │
├────┼──────────────┤     ├────┼──────────┼──────┤   ├────┼─────────────────┼──────┤
│ 1  │ Electronics  │     │ 2  │ Laptop   │ 1    │   │ 2  │ electronics_ret │return│
│ 2  │ Audio        │     │ 1  │ Headphon │ 2    │   │ 1  │ standard_return │return│
│ 3  │ Accessories  │     │ 5  │ Phone Cs │ 3    │   │ 4  │ manufacturer_w  │warr. │
│ 4  │ General      │     │ 6  │ T-Shirt  │ 4    │   │ 3  │ free_shipping   │ship. │
└────┴──────────────┘     └────┴──────────┴──────┘   └────┴─────────────────┴──────┘
```

**2. Edges — the relationships**

Edges are foreign keys and junction table rows:

```sql
-- Foreign key: product belongs to category
products.category_id → categories.id

-- Junction table: policy applies to category
policy_category_rules(policy_rule_id, category_id)
```

Your graph visually:

```
Headphones ──→ Audio ──→ electronics_return (14-day, for electronics & audio)
                    ──→ manufacturer_warranty (1 year)
                    ──→ free_shipping (over $50)

T-Shirt    ──→ General ──→ standard_return (30-day)
                       ──→ free_shipping (over $50)
```

**3. Properties — the attributes on nodes and edges**

```
Product(name="Headphones", price=79.99, sku="SKU-H001")
PolicyRule(name="electronics_return", policy_type="return", summary="14-day window", details="...")
```

### The three kinds of graph traversals

Every graph query is one of these patterns:

**1-hop: from one node to its neighbors**
```
Category "Electronics" → what policies apply?
→ electronics_return, manufacturer_warranty, free_shipping
```
Your implementation: `query_category_policies("electronics")`

**2-hop: through an intermediate node**
```
Product "Headphones" → what category? → what policies apply to that category?
→ Audio → electronics_return, manufacturer_warranty, free_shipping
```
Your implementation: `query_product_policies("headphones")`

**3-hop: same as 2-hop, just more intermediate nodes**
```
Order ORD-1001 → what products? → what categories? → what policies apply?
```

Your graph doesn't have order-product links yet, but the pattern is the same — just one more JOIN.

---

## Part 2: How Your Implementation Works

### The SQL engine IS the graph engine

You didn't install Neo4j. Your graph traversal is plain SQL:

```sql
-- 2-hop traversal: product → category → policy
SELECT pr.name, pr.summary, pr.policy_type, pr.details,
       p.name AS product_name, c.name AS category_name
FROM products p                                    -- start at products
JOIN categories c ON p.category_id = c.id         -- hop 1: product → category
JOIN policy_category_rules pcr ON c.id = pcr.category_id  -- hop 2: category → junction
JOIN policy_rules pr ON pcr.policy_rule_id = pr.id       -- hop 2: junction → policy
WHERE p.name ILIKE '%headphones%';                 -- filter by product name
```

This is a **join-based graph traversal**. Each JOIN is one step along an edge. The query plan looks exactly like walking the graph:

```
products ──[category_id]──→ categories ──[id]──→ policy_category_rules ──[policy_rule_id]──→ policy_rules
```

### Why this works without a graph database

Graph databases (Neo4j, Neptune) are optimized for **deep traversals** — 5, 10, 20 hops. They use index-free adjacency: each node stores pointers directly to its neighbors, so traversal cost is O(1) per hop regardless of total graph size.

Your graph is **3 hops max and 6 products**. PostgreSQL with foreign key indexes handles this with sub-millisecond latency. A dedicated graph database would add operational complexity with zero performance benefit at this scale.

### The entity extraction layer

The bridge between natural language and structured queries:

```python
# In graph_store.py
def _extract_product_name(self, query: str) -> str | None:
    """Find which product the user is asking about."""
    query_lower = query.lower()
    # Longest match first: "phone case" before "phone"
    for name in sorted(self._product_names, key=len, reverse=True):
        if name in query_lower:
            return name
    return None
```

| User query | Extracted | SQL becomes |
|-----------|-----------|-------------|
| "can I return headphones after 10 days?" | `headphones` | `ILIKE '%headphones%'` |
| "tell me about laptop warranty" | `laptop` | `ILIKE '%laptop%'` |
| "what's the price of a phone case?" | `phone case` | `ILIKE '%phone case%'` |
| "how long does standard shipping take?" | `None` | no product match → empty |

This is **keyword-based entity extraction**. It's simple, fast, and works for a known product catalog. In production with thousands of products, you'd replace this with an LLM call (`extract_entities(query) → ["headphones", "return"]`) or a proper NER model.

### The switchable retrieval architecture

```python
# In retrievers.py
class GraphPolicyRetriever:
    def retrieve(self, query: str) -> str:
        policies = store.query_product_policies(query)   # Try product → category → policy
        if policies:
            return format_policies(policies)
        policies = store.query_category_policies(query)   # Try category → policy
        if policies:
            return format_policies(policies)
        return ""  # No structured match — caller decides what to do

class HybridRetriever:
    def retrieve(self, query: str) -> str:
        result = self.graph.retrieve(query)   # Try graph first
        if result:
            return result
        return self.vector.retrieve(query)    # Fall back to vector RAG
```

The key insight: **the graph retriever returns empty for queries it can't handle.** It doesn't guess. If there's no product or category match, it returns `""` and the hybrid retriever falls back to vector search. This is the single-responsibility principle applied to retrieval.

---

## Part 3: Vector DB vs Knowledge Graph — The Decision Framework

### When to use a vector database

| Signal | Example |
|--------|---------|
| Your data is **unstructured text** | Policy documents, FAQs, support articles, product descriptions |
| Queries are **rephrasings** of concepts | "send it back" ≈ "return policy" ≈ "how to refund" |
| Exact matches would **miss the intent** | "Can I get my money back?" has no shared words with "return policy" |
| You need **semantic similarity**, not exact facts | "Is this laptop good for gaming?" → reviews mentioning gaming performance |

### When to use a knowledge graph

| Signal | Example |
|--------|---------|
| Your data has **explicit relationships** | Products belong to categories, categories have policies |
| Queries need **multi-hop reasoning** | "What policy applies to this specific product?" |
| You need **deterministic answers** | The answer must be correct, not probably correct |
| The data is **structured and queryable** | Prices, categories, SKUs, policy types |
| You need **explainable answers** | "Headphones are in category Audio, which is governed by electronics_return (14 days)" |

### The overlapping zone — both can work

| Query | Vector can answer? | Graph can answer? |
|-------|-------------------|-------------------|
| "What's the return policy?" | Yes — semantic search finds the return chunk | No — no product or category mentioned |
| "Can I return headphones?" | Sometimes — depends on embedding quality (we proved it fails!) | Yes — exact traversal: headphones → 14 days |
| "How long to get a refund?" | Yes — "refund" ≈ "return policy" | No — no structured entity |
| "Is there a warranty on laptops?" | Sometimes | Yes — exact traversal: laptop → warranty |

### The decision tree

```
Is the answer derivable from RELATIONSHIPS between known entities?
    │
    ├── YES → Use knowledge graph
    │         Example: "What policy applies to headphones?"
    │         Headphones → Audio → 14-day return (deterministic)
    │
    └── NO ──→ Is the answer buried in UNSTRUCTURED TEXT?
                  │
                  ├── YES → Use vector search
                  │         Example: "How do I initiate a return?"
                  │         (this detail is in the policy text, not a relationship)
                  │
                  └── NO ──→ This is probably not a retrieval problem
                             (use a tool, API call, or direct DB query)
```

### Why hybrid is usually the right answer

Your `HybridRetriever` implements the pattern that most production systems converge to:

1. **Try graph first** — for product/category queries, it's faster and more accurate
2. **Fall back to vector** — for free-text queries, semantic search handles the rephrasing
3. **Both paths return the same format** — the LLM doesn't know or care which path was used

This is the same pattern used by Microsoft's GraphRAG, Amazon Neptune's vector search, and Neo4j's vector index — just implemented with PostgreSQL JOINs instead of a graph database.

---

## Part 4: Interview Talking Points

### "What is a knowledge graph?"

> "A knowledge graph stores entities as nodes and relationships as edges, making multi-hop queries deterministic. Instead of searching for 'similar text,' you traverse explicit connections: product → category → policy. This gives exact, explainable answers for structured questions."

### "Why implement it in PostgreSQL instead of Neo4j?"

> "At this scale — 6 products, 4 categories, 3-hop max traversal — PostgreSQL with foreign keys and JOINs handles graph queries with sub-millisecond latency. Adding Neo4j would be operational overhead with no performance benefit. The traversal is just a 3-table JOIN. If the graph grew to hundreds of thousands of nodes with 10+ hop queries, that's when index-free adjacency in a graph database becomes necessary."

### "How does the switchable retriever work?"

> "Both retrievers implement the same `PolicyRetriever` interface with a single `retrieve(query) → str` method. The `GraphPolicyRetriever` does entity extraction then SQL JOIN traversal. The `VectorPolicyRetriever` delegates to the existing hybrid RAG pipeline. The `HybridRetriever` tries graph first — if it returns empty (no product/category match), it falls back to vector. The factory function `create_policy_retriever(mode)` returns the right implementation based on the `RETRIEVAL_MODE` config."

### "When would you use a vector DB vs a knowledge graph?"

> "Vector DBs excel at semantic search over unstructured text — 'how do I send this back?' finding 'return policy.' Knowledge graphs excel at deterministic traversal over structured relationships — 'what policy applies to headphones?' finding the exact rule via category traversal. In practice, most production systems need both: the graph for structured lookups, vector search for everything else, with a hybrid layer that routes queries to the right path."

### "How does entity extraction work without an LLM?"

> "For a known product catalog, keyword matching with longest-match-first handles common queries. 'Phone case' matches before 'phone,' preventing partial matches. When the catalog grows to thousands of products or users ask 'that blue wireless thing I bought last week,' you'd add an LLM-based entity extraction step. But for a bounded catalog, simple matching is faster, free, and deterministic — no API call needed."
