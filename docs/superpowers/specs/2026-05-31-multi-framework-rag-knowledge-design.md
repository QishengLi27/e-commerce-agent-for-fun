# Multi-Framework Architecture: LlamaIndex (RAG) + Neo4j (Graph) + LangChain (Agents)

**Status:** Draft  
**Date:** 2026-05-31  
**Scope:** Production-ready architecture for product Q&A combining structured knowledge graph queries with unstructured semantic search, orchestrated by a deterministic agent DAG.

**Implementation Plan:** `docs/superpowers/plans/2026-05-31-multi-framework-rag-knowledge-plan.md`

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture Overview](#architecture-overview)
3. [Data Architecture](#data-architecture)
4. [Query Routing](#query-routing)
5. [Neo4j Graph Design](#neo4j-graph-design)
6. [LlamaIndex RAG Design](#llamaindex-rag-design)
7. [Ingestion Pipeline](#ingestion-pipeline)
8. [Component Map](#component-map)
9. [New Files](#new-files)
10. [Risks & Mitigations](#risks--mitigations)

---

## Motivation

### Why not one framework?

| Framework | Strengths | Weaknesses |
|-----------|-----------|------------|
| **LangChain/LangGraph** | Explicit agent DAG, circuit breaker, streaming, SSE | Weak ingestion pipeline, no native graph support |
| **LlamaIndex** | Best-in-class ingestion (`SentenceWindowNodeParser`), `SubQuestionQueryEngine`, metadata-filtered RAG | Less mature agent orchestration, no streaming out of the box |
| **Neo4j** | Native graph traversal (Cypher), built-in graph algorithms, visual explorer | No text embeddings, no RAG |
| **PostgreSQL/pgvector** | Transactional integrity, existing orders data, pgvector HNSW index | Graph queries via recursive CTEs are verbose and hard to maintain at scale |

### The thesis

Each framework owns the layer it's best at. No framework does everything. Combined, they form a system that handles structured traversal, semantic search, and agent orchestration — each through the right tool.

---

## Architecture Overview

### System Diagram

```
                              ┌──────────────────────────────────────────┐
                              │              USER QUERY                  │
                              │  "Which Android phone under $800         │
                              │   has the best camera?"                  │
                              └────────────────────┬─────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH AGENT (orchestration)                        │
│                                                                                  │
│   ┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────────┐ │
│   │ sanitize     │───▶│ classify       │───▶│ route to     │───▶│ generate    │ │
│   │ input        │    │ intent         │    │ tool node    │    │ reply       │ │
│   │              │    │                │    │              │    │             │ │
│   │ typo fix     │    │ order          │    │ order_status │    │ LLM formats │ │
│   │ cache check  │    │ list_orders    │    │ list_orders  │    │ tool output │ │
│   └──────────────┘    │ policy         │    │ policy_retr  │    │ as user     │ │
│                        │ weather        │    │ weather      │    │ response    │ │
│                        │ product_qa ◀──│───▶│ product_qa  │    └─────────────┘ │
│                        │ knowledge      │    │ knowledge    │                    │
│                        │ unknown        │    │ category     │                    │
│                        └────────────────┘    └──────┬───────┘                    │
│                                                     │                            │
│   Resilience: Circuit breaker, retry, fallback       │                            │
│   Memory: JSON file + pgvector semantic cache        │                            │
└─────────────────────────────────────────────────────┼────────────────────────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────┐
                    │                                 ▼                          │
                    │              TOOL LAYER (LangChain @tool)                  │
                    │                                                             │
                    │  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
                    │  │ product_info  │  │ policy_retr   │  │ product_qa     │  │
                    │  │ tool          │  │ tool          │  │ tool ⬅ NEW     │  │
                    │  │               │  │               │  │                │  │
                    │  │ Neo4j graph   │  │ Neo4j graph   │  │ Orchestrates:  │  │
                    │  │ traversal     │  │ + fallback     │  │ 1. Neo4j query │  │
                    │  └───────┬───────┘  └───────┬───────┘  │ 2. LlamaIndex  │  │
                    │          │                  │          │    RAG         │  │
                    │          │                  │          │ 3. LLM synth   │  │
                    │          │                  │          └───────┬────────┘  │
                    └──────────┼──────────────────┼──────────────────┼───────────┘
                               │                  │                  │
              ┌────────────────┼──────────────────┼──────────────────┼───────────┐
              │                ▼                  ▼                  ▼           │
              │                    DATA & RETRIEVAL LAYER                         │
              │                                                                   │
              │  ┌─────────────────────┐    ┌──────────────────────────────┐      │
              │  │       Neo4j         │    │   PostgreSQL + pgvector       │      │
              │  │  (graph cache)      │    │   (source of truth)           │      │
              │  │                     │    │                               │      │
              │  │  Nodes:             │    │  Tables:                      │      │
              │  │   Product           │◀───│   products, categories,       │      │
              │  │   Category          │sync│   policy_rules, attributes,   │      │
              │  │   AttributeValue    │    │   synonyms, relations,        │      │
              │  │   Brand, Policy     │    │   orders                      │      │
              │  │                     │    │                               │      │
              │  │  Relationships:     │    │  pgvector collections:         │      │
              │  │   IN_CATEGORY       │    │   store_policies              │      │
              │  │   CHILD_OF          │    │   semantic_cache              │      │
              │  │   HAS_ATTRIBUTE     │    │   product_chunks ⬅ NEW        │      │
              │  │   HAS_BRAND         │    │                               │      │
              │  │   ACCESSORY_OF      │    │  LlamaIndex reads/writes      │      │
              │  │   ALTERNATIVE_TO    │    │  product_chunks via           │      │
              │  │   COMPATIBLE_WITH   │    │  PGVectorStore                │      │
              │  │   HAS_POLICY        │    │                               │      │
              │  └─────────────────────┘    └──────────────────────────────┘      │
              │                                                                   │
              │  ▸ Neo4j: graph traversal, path queries, attribute filtering      │
              │  ▸ pgvector: ANN search (HNSW), metadata-filtered vector search   │
              │  ▸ PostgreSQL: transactional data, seed data source               │
              └───────────────────────────────────────────────────────────────────┘
```

### Data Flow by Query Type

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                     QUERY ROUTING MATRIX                             │
 │                                                                      │
 │  "what category is X?"                                               │
 │    ──▶ Neo4j (graph traversal) ──▶ LLM explains why                  │
 │                                                                      │
 │  "does X have feature Y?"                                            │
 │    ──▶ Neo4j (resolve product) ──▶ LlamaIndex (single-product RAG)  │
 │                                                                      │
 │  "which product has best X under $Y?"                                │
 │    ──▶ Neo4j (candidates by attributes) ──▶ LlamaIndex (compare)    │
 │    ──▶ LLM synthesis (rank + recommend)                              │
 │                                                                      │
 │  "what accessories for X?"                                           │
 │    ──▶ Neo4j (graph traversal) ──▶ return results                    │
 │                                                                      │
 │  "compare X and Y on Z"                                              │
 │    ──▶ Neo4j (product lookup) ──▶ LlamaIndex (both products' RAG)   │
 │    ──▶ LLM synthesis (side-by-side comparison)                       │
 │                                                                      │
 │  "what's the return policy for X?"                                   │
 │    ──▶ Neo4j (product → category → policy, inherited)               │
 │    ──▶ LlamaIndex (fallback if no graph match)                       │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Data Architecture

### PostgreSQL: Source of Truth

```
┌──────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL + PGVECTOR                          │
│                                                                   │
│  ┌─ CORE TABLES (existing) ──────────────────────────────────┐   │
│  │  categories         │ hierarchical, parent_id/level/path   │   │
│  │  products           │ name, category_id, price, sku        │   │
│  │  policy_rules       │ return/shipping/warranty policies    │   │
│  │  policy_category_rules │ junction: policy ↔ category       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ TAXONOMY TABLES (existing, schema.py) ────────────────────┐   │
│  │  attribute_definitions │ EAV schema registry               │   │
│  │  product_attributes    │ typed values (text/num/bool)      │   │
│  │  entity_synonyms       │ bilingual search expansion        │   │
│  │  entity_tags           │ disambiguation hints              │   │
│  │  product_relations     │ accessory/alternative/compatible  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ OPERATIONAL (existing) ──────────────────────────────────┐   │
│  │  orders              │ customer orders                     │   │
│  │  order_items         │ line items per order                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ PGVECTOR COLLECTIONS ────────────────────────────────────┐   │
│  │  store_policies      │ policy text embeddings             │   │
│  │  semantic_cache       │ LLM response cache                 │   │
│  │  product_chunks ⬅ NEW │ product description embeddings    │   │
│  │                       │ with metadata JSONB for filtering │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Neo4j: Graph Cache (Read-Optimized)

```
┌──────────────────────────────────────────────────────────────────┐
│                         NEO4J GRAPH                               │
│                                                                   │
│  LEGEND:                                                          │
│    (Node)  ── square brackets for entities                        │
│    -[REL]→ ── arrows for directed relationships                   │
│    {props} ── curly braces for properties                         │
│                                                                   │
│                                                                   │
│  ┌─ CATEGORY TREE ────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  (Electronics)                                              │   │
│  │   ├─[:CHILD_OF]─ (Mobile Devices)                           │   │
│  │   │   ├─[:CHILD_OF]─ (Smartphones)                          │   │
│  │   │   │   └─[:CHILD_OF]─ (Flagship Phones)                 │   │
│  │   │   │       ├─[:IN_CATEGORY]─ (iPhone 15 Pro)             │   │
│  │   │   │       │   {price: 999.00, sku: "SKU-IPH15P"}       │   │
│  │   │   │       └─[:IN_CATEGORY]─ (...)                       │   │
│  │   │   └─[:CHILD_OF]─ (Tablets)                              │   │
│  │   │       ├─[:IN_CATEGORY]─ (iPad Air)                      │   │
│  │   │       └─[:IN_CATEGORY]─ (Samsung Galaxy Tab S9)         │   │
│  │   ├─[:CHILD_OF]─ (Audio)                                    │   │
│  │   │   ├─[:CHILD_OF]─ (Headphones)                           │   │
│  │   │   ├─[:CHILD_OF]─ (Speakers)                             │   │
│  │   │   └─[:CHILD_OF]─ (Wireless Earbuds)                     │   │
│  │   ├─[:CHILD_OF]─ (Computing)                                │   │
│  │   │   ├─[:CHILD_OF]─ (Laptops)                              │   │
│  │   │   │   ├─[:CHILD_OF]─ (Gaming Laptops)                   │   │
│  │   │   │   └─[:IN_CATEGORY]─ (MacBook Pro 16)               │   │
│  │   │   ├─[:CHILD_OF]─ (Desktops)                             │   │
│  │   │   └─[:CHILD_OF]─ (Peripherals)                          │   │
│  │   └─[:CHILD_OF]─ (Accessories)                              │   │
│  │                                                             │   │
│  │  (Home & Kitchen)                                           │   │
│  │   ├─[:CHILD_OF]─ (Kitchen Appliances)                       │   │
│  │   │   ├─[:CHILD_OF]─ (Coffee Makers)                        │   │
│  │   │   └─[:CHILD_OF]─ (Blenders)                             │   │
│  │   └─[:CHILD_OF]─ (Home Decor)                               │   │
│  │       └─[:CHILD_OF]─ (Lighting)                             │   │
│  │                                                             │   │
│  │  (Fashion)                                                  │   │
│  │   ├─[:CHILD_OF]─ (Clothing)                                 │   │
│  │   │   └─[:CHILD_OF]─ (T-Shirts)                             │   │
│  │   └─[:CHILD_OF]─ (Footwear)                                 │   │
│  │       └─[:CHILD_OF]─ (Sneakers)                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ PRODUCT RELATIONS ────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  (iPhone 15 Pro)                                            │   │
│  │   ├─[:ACCESSORY_OF {strength:0.95}]─ (Leather Case)        │   │
│  │   ├─[:ACCESSORY_OF {strength:0.90}]─ (AirPods Pro 2)       │   │
│  │   ├─[:ACCESSORY_OF {strength:0.85}]─ (20W USB-C Charger)   │   │
│  │   ├─[:ALTERNATIVE_TO {strength:0.85}]─ (Samsung Galaxy S24) │   │
│  │   └─[:ALTERNATIVE_TO {strength:0.80}]─ (Google Pixel 8)    │   │
│  │                                                             │   │
│  │  (MacBook Pro 16)                                           │   │
│  │   ├─[:ACCESSORY_OF {strength:0.85}]─ (Apple Magic Keyboard) │   │
│  │   ├─[:ALTERNATIVE_TO {strength:0.80}]─ (Dell XPS 15)       │   │
│  │   └─[:ALTERNATIVE_TO {strength:0.60}]─ (ASUS ROG Strix G16) │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ ATTRIBUTE GRAPH ──────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  (iPhone 15 Pro)                                            │   │
│  │   ├─[:HAS_BRAND]──▶ (Brand {name:"Apple"})                 │   │
│  │   ├─[:HAS_ATTRIBUTE]──▶ (AttrVal {value:"256GB"})          │   │
│  │   │   └─[:OF_TYPE]──▶ (Attribute {name:"storage"})         │   │
│  │   ├─[:HAS_ATTRIBUTE]──▶ (AttrVal {value:"Titanium"})       │   │
│  │   │   └─[:OF_TYPE]──▶ (Attribute {name:"color"})           │   │
│  │   ├─[:HAS_ATTRIBUTE]──▶ (AttrVal {value:6.1})              │   │
│  │   │   └─[:OF_TYPE]──▶ (Attribute {name:"screen_size"})     │   │
│  │   └─[:HAS_ATTRIBUTE]──▶ (AttrVal {value:true})             │   │
│  │       └─[:OF_TYPE]──▶ (Attribute {name:"wireless"})        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ POLICY ASSIGNMENT (inheritance via CHILD_OF*) ────────────┐   │
│  │                                                             │   │
│  │  (Flagship Phones)─[:CHILD_OF*]─▶(Smartphones)              │   │
│  │       └─[:HAS_POLICY]──▶ (Policy {name:"electronics_return"}) │  │
│  │                                                            │   │
│  │  Policy lookup walks UP the tree:                           │   │
│  │  Flagship Phones → Smartphones → Mobile Devices → Electronics│  │
│  │  Returns all policies from all ancestors (deduplicated)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─ INDEXES ──────────────────────────────────────────────────┐   │
│  │  FULLTEXT INDEX on Product.name, Category.name             │   │
│  │  BTREE INDEX on Product.price                               │   │
│  │  UNIQUE CONSTRAINT on Product.name, Category.name, Brand    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Sync Strategy

```
┌──────────────────┐     python -m backend.knowledge.neo4j_setup     ┌──────────────────┐
│   PostgreSQL     │ ──────────────────────────────────────────────▶ │     Neo4j        │
│  (source of      │         one-shot, rebuilds entire graph         │  (graph cache)   │
│   truth)         │                                                 │                  │
│                  │     Read from PG seed tables:                   │  Read-only at    │
│  ┌────────────┐  │       categories, products, attributes,          │  query time      │
│  │ categories │──┤       synonyms, relations, policies              │                  │
│  │ products   │──┤                                                 │  Neo4j has no    │
│  │ attributes │──┤     Write to Neo4j via Cypher:                  │  writes except   │
│  │ synonyms   │──┤       CREATE/MATCH nodes + relationships        │  during setup    │
│  │ relations  │──┤                                                 │                  │
│  │ policies   │──┤     Re-run whenever seed data changes            │                  │
│  └────────────┘  │                                                 │                  │
└──────────────────┘                                                 └──────────────────┘
```

**Why no runtime dual-write:** The taxonomy data is seed data — it changes when the catalog changes (offline), not during user queries. Rebuilding Neo4j from PostgreSQL on `neo4j_setup` eliminates sync drift entirely. No runtime consistency problems.

---

## Query Routing

### Intent Classification (LangGraph node)

The existing `classify_intent` node in `backend/graph/nodes.py` gains a new intent:

```python
# New intent detection signals
PRODUCT_QA_SIGNALS = {
    "what is", "tell me about", "does the", "do the", "can the",
    "how much", "how many", "which", "compare", "vs", "versus",
    "feature", "spec", "specification", "battery", "camera",
    "weight", "size", "screen", "storage", "color",
}
```

The routing logic:

```
classify_intent(query)
    │
    ├── order ID detected?          → intent: order
    ├── list orders phrases?        → intent: list_orders
    ├── weather signals?            → intent: weather
    ├── policy signals?             → intent: policy
    ├── category + product signals? → intent: knowledge
    ├── product QA signals?         → intent: product_qa  ⬅ NEW
    └── fallback                    → intent: unknown
```

### Tool Dispatch (product_qa_tool)

The `product_qa_tool` is the single integration point where Neo4j, LlamaIndex, and the LLM meet:

```
product_qa_tool(query: str)
    │
    │  Step 1: QUERY ANALYSIS
    │  ┌─────────────────────────────────────────┐
    │  │ LLM classifies the query:               │
    │  │   type: single_product | cross_product  │
    │  │   target: product names + attributes    │
    │  │   topic: what the user wants to know    │
    │  └──────────────┬──────────────────────────┘
    │                 │
    │     ┌───────────┴───────────┐
    │     ▼                       ▼
    │  SINGLE PRODUCT         CROSS PRODUCT
    │  "Does iPhone have      "Which phone under $800
    │   MagSafe?"              has best camera?"
    │     │                       │
    │     ▼                       ▼
    │  Step 2a:                Step 2b:
    │  Neo4j resolve           Neo4j candidate search
    │  product name            ┌─────────────────────┐
    │  (synonym expanded)      │ MATCH (p:Product)   │
    │                          │  -[:IN_CATEGORY]->  │
    │     │                    │  (:Category)        │
    │     ▼                    │  -[:CHILD_OF*]->    │
    │  Step 3a:                │  (:Category         │
    │  LlamaIndex RAG          │   {name:'Smartphones'})
    │  metadata filter:        │ WHERE p.price < 800 │
    │  product_name =          │ AND NOT (p)         │
    │  'iPhone 15 Pro'         │  -[:HAS_BRAND]->    │
    │  query: user question    │  (:Brand            │
    │     │                    │   {name:'Apple'})   │
    │     ▼                    │ RETURN p.name,      │
    │  Step 4:                 │   p.price           │
    │  LLM synthesis           └─────────┬───────────┘
    │  from top chunks         │
    │                          ▼
    │                    Step 3b:
    │                    LlamaIndex RAG
    │                    metadata filter:
    │                    product_name IN
    │                    candidates
    │                    query: topic
    │                          │
    │                          ▼
    │                    Step 4b:
    │                    LLM compare + rank
    │                    from per-product chunks
    │
    ▼
  RETURN: formatted answer
```

### Cypher Query Examples

```cypher
// ── Resolve product with synonym expansion ──
// "苹果手机" → "iPhone 15 Pro"
MATCH (p:Product)
WHERE '苹果手机' IN p.search_terms OR p.name CONTAINS '苹果手机'
RETURN p.name, p.price, p.sku;

// ── Full category path for a product ──
// "What category is iPhone 15 Pro in, and why?"
MATCH path = (p:Product {name: 'iPhone 15 Pro'})
            -[:IN_CATEGORY]->(c:Category)-[:CHILD_OF*0..4]->(ancestor:Category)
WITH p, [node IN nodes(path) | node.name] AS category_path,
     [node IN nodes(path) WHERE node:Category | node.description] AS descriptions
RETURN p.name, p.price, category_path, descriptions
ORDER BY length(path) DESC;
// Returns: ["iPhone 15 Pro", "Flagship Phones", "Smartphones",
//           "Mobile Devices", "Electronics"]

// ── Find smartphones under $800, not Apple, 256GB storage ──
MATCH (p:Product)-[:IN_CATEGORY]->(:Category)-[:CHILD_OF*0..]->(:Category {name: 'Smartphones'})
WHERE p.price < 800
  AND NOT (p)-[:HAS_BRAND]->(:Brand {name: 'Apple'})
  AND EXISTS {
    MATCH (p)-[:HAS_ATTRIBUTE]->(av:AttributeValue {value: '256GB'})
           -[:OF_TYPE]->(:Attribute {name: 'storage'})
  }
RETURN p.name, p.price
ORDER BY p.price DESC;

// ── What accessories exist for iPhone 15 Pro? ──
MATCH (p:Product {name: 'iPhone 15 Pro'})<-[:ACCESSORY_OF]-(accessory:Product)
RETURN accessory.name, accessory.price;

// ── Inherited policies via category tree ──
MATCH (p:Product {name: 'iPhone 15 Pro'})-[:IN_CATEGORY]->(c:Category)
      -[:CHILD_OF*0..4]->(ancestor:Category)-[:HAS_POLICY]->(pol:Policy)
RETURN DISTINCT pol.name, pol.summary, pol.policy_type;
// Returns: electronics_return, free_shipping, manufacturer_warranty
// (inherited from Smartphones → Mobile Devices → Electronics)

// ── Products similar to a given product ──
// (same category, different brand, overlapping attributes)
MATCH (p:Product {name: 'iPhone 15 Pro'})-[:IN_CATEGORY]->(c:Category)
      <-[:IN_CATEGORY]-(similar:Product)
WHERE similar <> p
  AND NOT (similar)-[:HAS_BRAND]->(:Brand {name: 'Apple'})
OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(av:AttributeValue)
OPTIONAL MATCH (similar)-[:HAS_ATTRIBUTE]->(av2:AttributeValue)
WHERE av.value = av2.value
RETURN similar.name, similar.price, COUNT(DISTINCT av2) AS shared_attrs
ORDER BY shared_attrs DESC
LIMIT 5;
```

---

## LlamaIndex RAG Design

### Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                               │
│                                                                      │
│  data/product_descriptions.txt                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ [product: iPhone 15 Pro]                                     │   │
│  │ category: Smartphones                                        │   │
│  │ category_path: Electronics > Mobile Devices > Smartphones    │   │
│  │              > Flagship Phones                               │   │
│  │ brand: Apple                                                 │   │
│  │ price: 999.00                                                │   │
│  │ color: Titanium                                              │   │
│  │ storage: 256GB                                               │   │
│  │ screen_size: 6.1                                             │   │
│  │                                                              │   │
│  │ iPhone 15 Pro features a 6.1-inch Super Retina XDR display   │   │
│  │ with ProMotion technology. The A17 Pro chip delivers...      │   │
│  │ [long description continues]                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼ SimpleDirectoryReader                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Document objects, each with:                                  │   │
│  │   text: full description body                                 │   │
│  │   metadata: {product_name, brand, price, category, ...}       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼ SentenceWindowNodeParser                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ window_size=5, window_overlap=1                               │   │
│  │                                                               │   │
│  │ Each chunk = 5 sentences with 1-sentence overlap              │   │
│  │ Metadata preserved: {product_name, brand, price, ...}         │   │
│  │                                                               │   │
│  │ Window context attached to each node:                         │   │
│  │   node.metadata['window'] = surrounding sentences text        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼ OpenAIEmbedding (Zhipu embedding-2)                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Embed each chunk → 1024-dim vector                            │   │
│  │ Metadata serialized to JSONB cmetadata column                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼ PGVectorStore                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Collection: product_chunks                                    │   │
│  │   embedding: vector(1024)                                     │   │
│  │   cmetadata: {                                                │   │
│  │     product_name: "iPhone 15 Pro",                            │   │
│  │     brand: "Apple",                                           │   │
│  │     price: 999.00,                                            │   │
│  │     category: "Smartphones",                                  │   │
│  │     chunk_index: 3,                                           │   │
│  │     window: "(surrounding context text)"                      │   │
│  │   }                                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Query Engine

```python
# backend/rag/query_engine.py

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.postgres import PGVectorStore

def create_product_query_engine() -> RetrieverQueryEngine:
    """Factory: builds a metadata-filtered query engine for product Q&A.

    Returns a RetrieverQueryEngine that:
      1. ANN search over product_chunks in pgvector
      2. Metadata filtering via cmetadata JSONB column
      3. Sentence-window context expansion per retrieved chunk
    """
    vector_store = PGVectorStore.from_params(
        database=settings.pg_database,
        host=settings.pg_host,
        port=settings.pg_port,
        user=settings.pg_user,
        password=settings.pg_password,
        table_name="product_chunks",
        embed_dim=1024,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[
            MetadataReplacementPostProcessor(
                target_metadata_key="window"
            ),
        ],
    )
```

### Why SentenceWindowNodeParser Matters

```
WITHOUT WINDOW CONTEXT (naive chunking):
─────────────────────────────────────────────
Query: "iPhone 15 Pro battery life"

Retrieved chunk [500 chars]:
"...supports up to 15W wireless charging. Battery life lasts up to
29 hours of video playback. The new Action button replaces..."
                    ↑ answer incomplete, no fast-charging context


WITH SENTENCE WINDOW (window_size=5):
─────────────────────────────────────────────
Query: "iPhone 15 Pro battery life"

Retrieved chunk [matched sentences + window]:
  [window-before] "MagSafe wireless charging supports up to 15W
   with compatible accessories."

  [MATCH] "Battery life lasts up to 29 hours of video playback."

  [window-after] "Fast charging delivers 50% charge in 30 minutes
   with a 20W USB-C adapter."

  [window-after] "The A17 Pro chip's efficiency cores extend battery
   life during casual tasks like web browsing and messaging."

  [window-after] "Battery health management learns your charging
   routine to reduce battery aging."
                          ↑ LLM sees 5× more context, gives complete answer
```

---

## Component Map

### Setup Order

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SETUP SEQUENCE                                │
│                                                                      │
│  Step 1: PostgreSQL infrastructure                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ docker run -d --name pgvector -p 5432:5432 pgvector/pgvector │   │
│  │ docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \       │   │
│  │   neo4j:community                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼                                                                │
│  Step 2: PostgreSQL seed data                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ python -m backend.knowledge.schema                            │   │
│  │   → Creates tables + inserts all seed data                    │   │
│  │   → Categories, products, attributes, synonyms, relations,   │   │
│  │     policies, policy mappings                                 │   │
│  │   → Same as today, no changes needed                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼                                                                │
│  Step 3: Neo4j graph sync (NEW)                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ python -m backend.knowledge.neo4j_setup                       │   │
│  │   → Reads all seed data from PostgreSQL                       │   │
│  │   → Creates Neo4j nodes, relationships, indexes               │   │
│  │   → Wipes existing graph first (idempotent)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│     │                                                                │
│     ▼                                                                │
│  Step 4: Vector embeddings                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ python -m backend.db.vector_setup                             │   │
│  │   → store_policies collection (existing)                      │   │
│  │   → product_chunks collection (NEW)                           │   │
│  │   → Uses LlamaIndex IngestionPipeline for product chunks      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Runtime Flow

```
                    ┌───────────────────────────────────┐
                    │            main.py / FastAPI       │
                    │  uvicorn backend.api.main:app      │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │         LangGraph Agent            │
                    │  backend/graph/agent_graph.py      │
                    │                                    │
                    │  Node sequence:                    │
                    │  sanitize_input                   │
                    │  → classify_intent                │
                    │  → route_to_tool                  │
                    │  → generate_reply                 │
                    │  → update_memory                  │
                    └──────────────┬────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
    │ order_status    │ │ policy_retriever│ │ product_qa_tool  │
    │ list_orders     │ │ product_info    │ │      ⬅ NEW       │
    │ weather         │ │ category_info   │ │                  │
    │                 │ │                 │ │ ┌──────────────┐ │
    │ (unchanged)     │ │ (updated to     │ │ │ Neo4j query  │ │
    │                 │ │  use Neo4j)     │ │ │ (candidates/ │ │
    │                 │ │                 │ │ │  relations)  │ │
    │                 │ │                 │ │ ├──────────────┤ │
    │                 │ │                 │ │ │ LlamaIndex   │ │
    │                 │ │                 │ │ │ RAG (product │ │
    │                 │ │                 │ │ │ descriptions)│ │
    │                 │ │                 │ │ ├──────────────┤ │
    │                 │ │                 │ │ │ LLM synthesis│ │
    │                 │ │                 │ │ └──────────────┘ │
    └────────┬────────┘ └────────┬────────┘ └────────┬─────────┘
             │                   │                    │
             ▼                   ▼                    ▼
    ┌────────────────────────────────────────────────────────┐
    │              DATA LAYER                                 │
    │                                                        │
    │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
    │  │ Neo4j    │  │ PG+      │  │ Zhipu LLM          │   │
    │  │ Cypher   │  │ pgvector │  │ (GLM-4-Flash)      │   │
    │  │ graph    │  │ ANN      │  │ synthesis +        │   │
    │  │ traversal│  │ search   │  │ classification     │   │
    │  └──────────┘  └──────────┘  └────────────────────┘   │
    └────────────────────────────────────────────────────────┘
```

---

## New Files

```
apps/backend/
├── backend/
│   ├── knowledge/
│   │   ├── graph_store.py          # Existing — PostgreSQL graph queries
│   │   │                           #   May deprecate in favor of neo4j_store
│   │   ├── neo4j_store.py          # NEW — Neo4j driver + query methods
│   │   ├── neo4j_setup.py          # NEW — sync PG seed → Neo4j graph
│   │   ├── schema.py               # Existing — source of truth (unchanged)
│   │   └── setup.py                # Existing — entry point (unchanged)
│   │
│   ├── rag/                        # NEW — LlamaIndex module
│   │   ├── __init__.py
│   │   ├── ingestion.py            # IngestionPipeline: parse → chunk → embed → pgvector
│   │   └── query_engine.py         # RetrieverQueryEngine factory + metadata filters
│   │
│   ├── tools/
│   │   ├── knowledge.py            # Existing — updated to use Neo4j (or dual-backend)
│   │   ├── policy.py               # Existing — unchanged
│   │   ├── order.py                # Existing — unchanged
│   │   └── product_qa.py           # NEW — product_qa_tool (orchestrates Neo4j + LlamaIndex)
│   │
│   ├── graph/
│   │   ├── nodes.py                # Update — add product_qa intent routing
│   │   └── agent_graph.py          # Update — add product_qa node to StateGraph
│   │
│   ├── intent/
│   │   ├── keyword.py              # Update — add product QA signals
│   │   └── semantic.py             # Update — recognize product_qa intent
│   │
│   └── config.py                   # Update — add settings:
│                                   #   neo4j_uri, neo4j_user, neo4j_password
│
├── data/
│   ├── store_policies.txt          # Existing — policy text for RAG
│   └── product_descriptions.txt    # NEW — [product: X] blocks with metadata
│
├── requirements.txt                # Update — add:
│                                   #   neo4j, llama-index,
│                                   #   llama-index-vector-stores-postgres,
│                                   #   llama-index-embeddings-openai
│                                   #   llama-index-llms-openai-like
│
└── docker-compose.yml              # Update — add Neo4j service
```

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Dual DB sync drift** — Neo4j diverges from PostgreSQL | Low | Neo4j is rebuilt from scratch on `neo4j_setup`. No runtime writes to Neo4j. Read-only graph cache. |
| **Neo4j operational burden** — another Docker container, another port, another health check | Low | `docker-compose.yml` already has pgvector. Adding Neo4j is 5 lines. `docker compose up` starts everything. |
| **LlamaIndex + LangChain boundary** — two frameworks in one tool, confusing stack traces | Medium | `product_qa_tool` is the single integration point. All errors caught + logged with `[neo4j]` / `[llama-index]` prefix. Circuit breaker wraps the whole tool. |
| **Framework version conflicts** — LangChain and LlamaIndex share deps (openai, tiktoken) | Low | Both target the same OpenAI-compatible API. Pin versions. Test `pip install` during setup. |
| **Neo4j Community Edition limits** — single DB, no clustering, no hot backups | Low | This is a demo/learning project. Community Edition handles millions of nodes. Not a production concern. |
| **Seed data duplication** — same data in PostgreSQL seed tuples AND `product_descriptions.txt` | Medium | `product_descriptions.txt` is long-form text ONLY. Attributes, price, relations stay in `schema.py` seed tuples. PG is the single source of truth for structured data. |
| **Cypher injection** — user queries interpolated into Cypher | Medium | Use Neo4j driver's **parameterized queries** (`$param` syntax). Never string-concatenate user input into Cypher. Same as SQL injection prevention. |
| **Embedding cost at scale** — 29 products × ~15 chunks = 435 embeddings. At 10K products = 150K embeddings | Low (future) | Ingestion is offline (`vector_setup`). Embedding cost is one-time per catalog update. `chunk_size=64` batches control API rate. |

---

## Appendix: docker-compose.yml Addition

```yaml
# Add to existing docker-compose.yml
services:
  pgvector:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: postgres
    # ... existing config

  neo4j:                                    # ⬅ NEW
    image: neo4j:community
    ports:
      - "7474:7474"   # HTTP / Browser UI
      - "7687:7687"   # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

volumes:
  neo4j_data:                               # ⬅ NEW
  neo4j_logs:                               # ⬅ NEW
```

---

## Appendix: product_descriptions.txt Format

```
[product: iPhone 15 Pro]
category: Flagship Phones
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

[product: Google Pixel 8]
category: Smartphones
brand: Google
price: 699.00
color: Obsidian
storage: 128GB
screen_size: 6.2
release_year: 2023
wireless: true

Google Pixel 8 features a 6.2-inch Actua display with a smooth 120Hz refresh rate...
[more description text follows the same metadata + body pattern]
```
