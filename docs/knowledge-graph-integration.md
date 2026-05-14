# Knowledge Graphs for E-Commerce AI Agents

## What a Knowledge Graph Is

A knowledge graph stores **entities**, their **attributes**, and **relationships** as an explicit, queryable graph structure — unlike a vector DB which stores meaning as opaque floating-point arrays.

```
Vector DB:  "return policy" → [0.023, -0.451, 0.891, ...]   ← meaning encoded as numbers
Knowledge Graph:  (Product:Headphones) -[has_policy]-> (Policy:Electronics_Return)  ← explicit fact
```

The fundamental difference: **you can query a knowledge graph and get an exact, verifiable answer.** A vector search gives you "something semantically close" — you hope it's right. A graph query gives you the connected fact — you know it's right.

---

## Vector DB vs Knowledge Graph: When to Use Which

| | Vector DB (your current system) | Knowledge Graph |
|---|---|---|
| **Best for** | Unstructured text: policies, FAQs, docs | Structured data: products, orders, categories |
| **Query type** | "What's the return policy?" | "What's in order 1001? What category is that product? What policy applies to that category?" |
| **Answer** | Retrieved text chunk → LLM paraphrases | Exact fact → LLM formats |
| **Hallucination risk** | Medium (LLM might misread context) | Low (facts are exact) |
| **Multi-hop** | No (each chunk is independent) | Yes (traverse relationships) |
| **Updates** | Re-embed chunks | Add/remove nodes and edges |
| **Example failure** | "Can I return headphones?" → retrieves shipping chunk | Would follow: headphones → category:electronics → policy:14-day-return |

---

## What a Knowledge Graph Looks Like for Your Project

### Your current data, modeled as a graph

```
                    ┌──────────────┐
                    │  Customer    │
                    │  id: user_42 │
                    └──────┬───────┘
                           │ placed
                           ▼
    ┌──────────────────────────────────────────────┐
    │              Order ORD-1001                   │
    │  status: processing, date: 2026-05-10        │
    └──────┬──────────────┬───────────────┬────────┘
           │ contains     │ contains      │ contains
           ▼              ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Headphone│   │ Laptop   │   │  Mouse   │
    │  SKU: H1 │   │ SKU: L2  │   │ SKU: M3  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         │ belongs_to   │ belongs_to   │ belongs_to
         ▼              ▼              ▼
    ┌──────────┐   ┌──────────────────────────┐
    │ Audio    │   │    Electronics           │
    └────┬─────┘   └────────────┬─────────────┘
         │                      │
         │ governed_by          │ governed_by
         ▼                      ▼
    ┌──────────────────────────────────────────┐
    │  Standard Return Policy                  │
    │  window: 30 days, refund: full           │
    └──────────────────────────────────────────┘
                              │
                              │ has_exception
                              ▼
                    ┌──────────────────────┐
                    │ Electronics Return   │
                    │ window: 14 days      │
                    │ condition: original  │
                    └──────────────────────┘
```

### The same data as triples (RDF-style)

```
(Order:1001, status, "processing")
(Order:1001, contains, Product:H1)
(Product:H1, name, "Headphones")
(Product:H1, belongs_to, Category:Electronics)
(Category:Electronics, governed_by, Policy:Electronics_Return)
(Policy:Electronics_Return, window, "14 days")
(Policy:Electronics_Return, condition, "original packaging")
```

Now you can answer "Can I return headphones after 10 days?" by traversing the graph instead of hoping the right text chunk ranks highest:

```
Query: Product:H1 → Category:Electronics → Policy:Electronics_Return
Result: {window: "14 days"} → YES, within policy
```

---

## How to Integrate a Knowledge Graph Into Your Current System

### Approach 1: Graph as an Additional Tool (simplest)

Add a knowledge graph tool alongside your existing tools. The agent decides whether to query the vector DB (for policies) or the graph (for structured data).

```python
@tool
def knowledge_graph_tool(query: str) -> str:
    """Query the knowledge graph for product info, order contents,
    category relationships, and policy rules that apply to specific products."""
    # Translate natural language to Cypher
    cypher = text_to_cypher(query)
    results = graph_db.run(cypher)
    return format_graph_results(results)
```

**Your updated graph:**
```
sanitize_input → classify_intent → {
    order_node     (queries PostgreSQL orders table)
    policy_node    (queries vector DB + graph for policy rules)
    product_node   (NEW: queries knowledge graph for product/category info)
    weather_node
}
```

**When the agent benefits:**
- "Can I return these headphones?" → graph finds product → category → applicable policy
- "What's in my order?" → graph returns items with their categories and prices
- "Is a laptop covered by warranty?" → graph traverses product → category → warranty terms

### Approach 2: Graph-Enhanced Retrieval (Graph RAG)

Before retrieving text chunks, query the graph to **enrich the query** with structured context:

```python
def graph_enhanced_retrieve(query: str) -> list[Document]:
    # 1. Extract entities from the query
    entities = extract_entities(query)  # → ["headphones", "return"]

    # 2. Query the graph for related structured facts
    graph_context = graph_db.query("""
        MATCH (p:Product)-[:belongs_to]->(c:Category)-[:governed_by]->(pol:Policy)
        WHERE p.name CONTAINS "headphones"
        RETURN pol.window, pol.condition, c.name
    """)
    # → {"window": "14 days", "condition": "original packaging", "category": "Electronics"}

    # 3. Use graph results to expand the retrieval query
    enhanced_query = f"{query} {graph_context['category']} {graph_context['window']}"
    # → "Can I return headphones after 10 days? Electronics 14 days"

    # 4. Hybrid retrieval with the enhanced query
    return hybrid_retriever.retrieve(enhanced_query)
```

**Why this helps:** The original query "Can I return headphones after 10 days?" doesn't contain "electronics" or "14 days". The graph adds those terms, making both dense and sparse retrieval more likely to find the right chunk.

### Approach 3: Graph as Ground Truth for Validation

Your `validate_reply` node checks if the LLM's answer is supported by tool results. A knowledge graph makes this **deterministic** for structured facts:

```python
def validate_with_graph(answer: str, graph_context: dict) -> str:
    """Check answer claims against graph facts."""
    claims = extract_claims(answer)
    # claims = ["return window is 14 days", "headphones are electronics"]

    for claim in claims:
        # Verify against graph
        if "return window" in claim:
            actual = graph_context.get("policy_window")
            if "14" in claim and actual != "14 days":
                return "unverified_claims"

    return "valid"
```

This is **rule-based validation** instead of LLM-based validation — faster, cheaper, and zero hallucination risk.

### Approach 4: Unified Graph + Vector Store (most powerful)

Store everything in a graph database that also supports vector search (Neo4j, Amazon Neptune, or pgvector on node properties):

```
Each node has:
  - Structured properties (name, price, category)
  - A text description
  - A vector embedding of that description

Query flow:
  1. Vector search on node embeddings → find relevant nodes
  2. Traverse relationships from those nodes → find connected facts
  3. Return both the retrieved text AND the graph context
```

```python
def unified_retrieve(query: str) -> dict:
    # Step 1: Vector search finds relevant nodes
    query_embedding = embed(query)
    nodes = graph_db.vector_search(query_embedding, k=5)
    # → [Policy:Electronics_Return, Product:Headphones, Policy:Standard_Return]

    # Step 2: Traverse relationships for context
    for node in nodes:
        if node.type == "Product":
            node.context = graph_db.traverse(node, depth=2)
            # → Category:Electronics → Policy:Electronics_Return

    # Step 3: Return enriched context
    return {
        "text_chunks": [n.description for n in nodes],
        "structured_facts": graph_db.format_context(nodes),
    }
```

---

## Concrete Implementation: Adding a Graph to Your Project

### Technology choice

For a prototype, **Neo4j Community Edition** is the standard. For your existing PostgreSQL setup, Apache AGE (a PostgreSQL graph extension) avoids adding a new database. Or you can model graphs relationally using recursive CTEs on your existing `orders` table.

Simplest approach for your project: **add a graph layer using networkx in-memory** for the prototype, then graduate to Neo4j if the pattern proves valuable.

### Step-by-step integration

```
1. Define the graph schema
   ├── Nodes: Product, Category, Order, Customer, Policy
   └── Edges: contains, belongs_to, governed_by, placed, purchased

2. Populate from existing data
   ├── Products from your orders table (extract product names)
   ├── Categories inferred from product names
   └── Policies from store_policies.txt (manually annotated)

3. Add graph_query_tool to tools/__init__.py
   └── Uses Cypher (Neo4j) or networkx traversals

4. Add product_node to graph/nodes.py
   └── Routes "what is product X?" and "category of Y?" queries

5. Wire into classify_intent
   └── Add "product" as a recognized intent

6. Optionally enhance policy_node
   └── Query graph for product → category → policy before vector search
```

### What changes in your agent graph

```python
# New node
def product_node(state: AgentState) -> AgentState:
    """Query product info and related policies from the knowledge graph."""
    query = state.get("user_input", "")
    product_name = extract_product_name(query)  # "headphones"

    # Query graph
    result = graph_db.query("""
        MATCH (p:Product {name: $name})-[:belongs_to]->(c:Category)
              -[:governed_by]->(pol:Policy)
        RETURN p, c.name AS category, pol.window, pol.condition
    """, name=product_name)

    state["tool_result"] = format_graph_result(result)
    return state

# Updated router
def route_by_intent(state):
    if "product" in state["intent"]:
        return "product_node"  # ← new branch
    # ... existing routes
```

---

## When This Matters for Your Agent

| Scenario | Current (Vector only) | With Knowledge Graph |
|----------|----------------------|---------------------|
| "Can I return headphones after 10 days?" | Retrieval might get shipping chunk (wrong) | Graph: headphones → electronics → 14 days → YES |
| "What's in order 1001?" | DB query returns raw rows | Graph returns items with categories and applicable policies |
| "Is this laptop covered?" | Must find warranty chunk via vector search | Graph: laptop → electronics → manufacturer warranty: yes |
| "Compare return policies for headphones vs furniture" | 2 separate retrievals, LLM compares | 2 graph traversals, structured comparison |

---

## Tradeoffs

| Factor | Vector DB Only | + Knowledge Graph |
|--------|---------------|-------------------|
| Setup complexity | Low (pgvector is one table) | Medium (schema design, ETL, Cypher) |
| Accuracy for structured facts | "Usually right" | "Always right" (deterministic traversal) |
| Multi-hop reasoning | No | Yes (graph traversal) |
| Handling rephrasing | Good (semantic search) | Needs entity extraction first |
| Maintenance | Just re-embed | Keep graph in sync with source data |
| Query latency | ~5ms (cosine distance) | ~5-50ms (traversal depth-dependent) |

---

## Interview Talking Points

### "How would you add a knowledge graph to this RAG system?"

> "I'd model products, categories, orders, and policies as nodes with explicit relationships. The agent would query the graph for structured facts — like which policy applies to which product category — then use vector search for the unstructured policy text. This gives me deterministic answers for structured questions and semantic search for everything else."

### "When does a vector database fail that a graph would solve?"

> "Vector search fails when the query and the target document share no semantic overlap. My system had this problem: 'Can I return headphones after 10 days?' retrieved the shipping chunk because the embedding model couldn't distinguish it from the return policy. A knowledge graph would follow the exact relationship: headphones → electronics → 14-day return policy."

### "How do you keep the graph in sync?"

> "The graph is built from source-of-truth data — the product catalog, the order system, and the policy documents. When a policy changes, you update the relevant policy node's properties. The graph relationships don't drift — they're explicit. This is easier to maintain than vector embeddings, which require re-embedding when content changes."

### "What graph database would you use?"

> "For a prototype, I'd start with the existing PostgreSQL database using recursive CTEs to model relationships, or networkx in-memory. For production, Neo4j is the standard for graph-native storage with Cypher querying. If I want to stay on PostgreSQL, the pgvector extension coexists with graph-style queries using CTEs, avoiding a second database."
