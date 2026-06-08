# Multi-Framework RAG + Knowledge Graph Architecture

## Overview

This diagram shows the complete architecture after integrating **Neo4j** (knowledge graph), **LlamaIndex** (RAG), and **LangGraph** (agent orchestration) into the existing e-commerce support backend.

---

## 1. End-to-End Request Flow

```mermaid
flowchart TB
    subgraph Client["Frontend Client"]
        User[User Query]
    end

    subgraph API["FastAPI Layer"]
        Chat[/POST /chat\]
        Stream[/POST /chat/stream\]
    end

    subgraph Agent["LangGraph Agent DAG"]
        direction TB
        Sanitize[sanitize_input]
        Classify[classify_intent]
        Router{route_by_intent}

        subgraph ToolNodes["Tool Execution Nodes"]
            direction LR
            ON[order_node]
            LON[list_orders_node]
            PN[policy_node]
            WN[weather_node]
            KN[knowledge_node]
            PQN["product_qa_node ⭐"]
        end

        Generate[generate_reply]
        Validate[validate_reply]
        Update[update_memory]
    end

    subgraph DataLayer["Data & Retrieval Layer"]
        direction TB
        PG[(PostgreSQL<br/>Source of Truth)]
        Neo4j[(Neo4j<br/>Graph Cache)]
        PGVec[(pgvector<br/>Product Chunks)]
        LLM[(LLM<br/>GLM-4-Flash)]
    end

    User --> Chat
    Chat --> Sanitize
    Sanitize --> Classify --> Router

    Router -->|order| ON
    Router -->|list_orders| LON
    Router -->|policy| PN
    Router -->|weather| WN
    Router -->|knowledge| KN
    Router -->|product_qa| PQN
    Router -->|unknown| Generate

    ON & LON & PN & WN & KN & PQN --> Generate
    Generate --> Validate
    Validate -->|unverified_claims<br/>retry < 2| Generate
    Validate -->|valid / not_applicable| Update
    Update --> Chat

    PG -.->|neo4j_setup.py| Neo4j
    PG -.->|ingestion.py| PGVec
    Neo4j & PGVec & LLM -.->|used by| PQN
```

---

## 2. LangGraph StateGraph Detail

```mermaid
flowchart LR
    Start([__start__]) --> Sanitize[sanitize_input]
    Sanitize --> Classify[classify_intent]
    Classify --> Router{route_by_intent}

    Router -->|order| ON[order_node]
    Router -->|list_orders| LON[list_orders_node]
    Router -->|policy| PN[policy_node]
    Router -->|weather| WN[weather_node]
    Router -->|knowledge| KN[knowledge_node]
    Router -->|product_qa| PQN["product_qa_node ⭐"]
    Router -->|cached / unknown| GR[generate_reply]

    ON --> GR
    LON --> GR
    PN --> GR
    WN --> GR
    KN --> GR
    PQN --> GR

    GR --> VR[validate_reply]
    VR -->|retry| GR
    VR -->|pass| UM[update_memory]
    UM --> End([END])
```

---

## 3. `product_qa_tool` Internal Orchestration

```mermaid
flowchart TB
    subgraph Input["User Query"]
        Q1["Does iPhone 15 Pro have MagSafe?"]
        Q2["Which phone under $800 has the best camera?"]
        Q3["What category is MacBook Pro 16 in?"]
    end

    subgraph Tool["product_qa_tool"]
        Pattern{Query Pattern}

        subgraph Pattern1["Single-Product Query"]
            Resolve1[Neo4jStore.resolve_product]
            Info1[Neo4jStore.get_product_info]
            RAG1["LlamaIndex RAG<br/>filtered to product"]
            Synth1[LLM Synthesis]
        end

        subgraph Pattern2["Cross-Product Comparison"]
            Search2[Neo4jStore.search_products]
            RAG2["LlamaIndex RAG<br/>per candidate"]
            Synth2[LLM Comparison]
        end

        subgraph Pattern3["Category / Recommendation"]
            Resolve3[Neo4jStore.resolve_product]
            Path3[Neo4jStore.get_category_tree]
            Synth3[LLM Explanation]
        end
    end

    Q1 --> Pattern -->|single-product| Pattern1
    Q2 --> Pattern -->|comparison| Pattern2
    Q3 --> Pattern -->|category| Pattern3

    Resolve1 --> Info1 --> RAG1 --> Synth1
    Search2 --> RAG2 --> Synth2
    Resolve3 --> Path3 --> Synth3
```

---

## 4. Data Sync Pipelines

```mermaid
flowchart LR
    subgraph Seed["PostgreSQL Seed Data"]
        Categories[(categories)]
        Products[(products)]
        Attributes[(product_attributes)]
        Relations[(product_relations)]
        Policies[(policy_rules)]
        Synonyms[(entity_synonyms)]
    end

    subgraph GraphSync["Graph Sync Pipeline"]
        Setup[neo4j_setup.py]
        Batch["UNWIND batches<br/>size=500"]
    end

    subgraph VectorSync["Vector Sync Pipeline"]
        Parse[parse_product_descriptions]
        Chunk["SentenceWindowNodeParser<br/>window=5"]
        Embed["OpenAIEmbedding<br/>embedding-2"]
    end

    Categories & Products & Attributes & Relations & Policies & Synonyms --> Setup --> Batch --> Neo4j[(Neo4j Graph)]

    Products --> Parse --> Chunk --> Embed --> PGVec[(pgvector<br/>product_chunks)]
```

### Neo4j Graph Schema

```mermaid
graph LR
    subgraph ProductGraph["Product-Centric Graph"]
        P[(:Product)] -->|IN_CATEGORY| C[(:Category)]
        P -->|HAS_BRAND| B[(:Brand)]
        P -->|HAS_ATTRIBUTE| AV[(:AttributeValue)]
        AV -->|OF_TYPE| A[(:Attribute)]
        P -->|ACCESSORY_OF| P2[(:Product)]
        P -->|ALTERNATIVE_TO| P3[(:Product)]
        C -->|CHILD_OF| C2[(:Category)]
        C2 -->|HAS_POLICY| POL[(:Policy)]
    end
```

---

## 5. Component Responsibilities

| Component | Framework | Role | Data Source |
|-----------|-----------|------|-------------|
| `product_qa_node` | LangGraph | Agent node that routes `product_qa` intent | — |
| `product_qa_tool` | LangChain `@tool` | Orchestrates graph + RAG + LLM | Neo4j + pgvector + LLM |
| `Neo4jStore` | Neo4j driver | Typed Cypher queries (resolve, search, relations) | Neo4j |
| `neo4j_setup.py` | Neo4j driver | Batch-sync PG seed → Neo4j graph | PostgreSQL → Neo4j |
| `ingestion.py` | LlamaIndex | Parse descriptions → chunk → embed → store | `data/product_descriptions.txt` → pgvector |
| `query_engine.py` | LlamaIndex | Metadata-filtered vector retriever with sentence-window expansion | pgvector |
| `classify_intent` | Keyword + LLM | Detects `product_qa` intent from user query | — |

---

## 6. Vector Store Decision: pgvector vs. Milvus

| Dimension | pgvector (chosen) | Milvus |
|-----------|-------------------|--------|
| **Infrastructure** | Zero new services — reuse existing PostgreSQL container | Dedicated vector database (another container/service to ops) |
| **Scale ceiling** | Comfortable to ~1M vectors with HNSW index | Designed for 10M–1B+ vectors, distributed sharding |
| **Latency @ 30 products** | ~5–15ms | ~2–5ms (difference is noise at this scale) |
| **Operational overhead** | Same backup/restore/monitoring as PG | Separate backup strategy, SDK, auth, clustering |
| **Multi-modal / hybrid** | Basic; requires manual joins | Native hybrid search (vector + scalar + full-text) |
| **Team familiarity** | Team already runs PG + pgvector for policy embeddings | New toolchain to onboard |

**Why pgvector was the right call for this project:**

1. **Catalog size is tiny.** 30 products × ~20 sentence-window chunks = ~600 vectors. pgvector handles this in single-digit milliseconds. Milvus would be over-engineering by two orders of magnitude.
2. **Infrastructure consolidation.** The project already runs PostgreSQL for transactional data, LangGraph checkpointer, and the existing `store_policies` semantic cache (via `PGVector`). Adding Milvus means a second vector store to monitor, backup, and secure.
3. **No multi-modal need.** Milvus shines when you need to co-locate image embeddings, audio fingerprints, and text vectors with complex filtering. This project only has text product descriptions.
4. **When we *would* switch to Milvus:** If the catalog grows beyond ~10K products, or if we add visual search (image-based product lookup), or if latency becomes a bottleneck under high concurrency. At that threshold, the migration path is straightforward: re-run the LlamaIndex `ingestion.py` pipeline with a `MilvusVectorStore` instead of `PGVectorStore`.

**Bottom line for your boss:** Milvus is a great tool, but for a 30-product catalog inside an already-PostgreSQL-heavy stack, it adds operational complexity with no measurable latency or accuracy benefit. pgvector is the pragmatic choice until scale justifies the switch.

---

## 7. Three Query Patterns in Detail

### Pattern A: Single-Product Feature Query
```mermaid
sequenceDiagram
    actor User
    participant Agent as LangGraph Agent
    participant Tool as product_qa_tool
    participant Neo4j as Neo4jStore
    participant RAG as LlamaIndex RAG
    participant LLM as GLM-4-Flash

    User->>Agent: "Does iPhone 15 Pro have MagSafe?"
    Agent->>Agent: classify_intent → product_qa
    Agent->>Tool: invoke(query)
    Tool->>Neo4j: resolve_product("Does iPhone 15 Pro have MagSafe?")
    Neo4j-->>Tool: ProductRef(name="iPhone 15 Pro")
    Tool->>Neo4j: get_product_info("iPhone 15 Pro")
    Neo4j-->>Tool: category_path, attributes, policies
    Tool->>RAG: create_filtered_query_engine(product_names=["iPhone 15 Pro"])
    RAG-->>Tool: "MagSafe wireless charging supports up to 15W..."
    Tool->>LLM: Synthesize graph context + RAG context
    LLM-->>Tool: "Yes, iPhone 15 Pro supports MagSafe..."
    Tool-->>Agent: Answer string
    Agent->>User: Final reply
```

### Pattern B: Cross-Product Comparison
```mermaid
sequenceDiagram
    actor User
    participant Agent as LangGraph Agent
    participant Tool as product_qa_tool
    participant Neo4j as Neo4jStore
    participant RAG as LlamaIndex RAG
    participant LLM as GLM-4-Flash

    User->>Agent: "Which phone has the best camera, iPhone or Pixel?"
    Agent->>Agent: classify_intent → product_qa
    Agent->>Tool: invoke(query)
    Tool->>Neo4j: search_products(category="Smartphones", limit=5)
    Neo4j-->>Tool: [iPhone 15 Pro, Google Pixel 8, ...]
    loop For each candidate
        Tool->>RAG: retrieve chunks filtered to candidate
        RAG-->>Tool: product-specific description excerpts
    end
    Tool->>LLM: Compare all candidates with context
    LLM-->>Tool: "iPhone 15 Pro has a 48MP main camera with 5x telephoto..."
    Tool-->>Agent: Comparison answer
    Agent->>User: Final reply
```

### Pattern C: Category / Recommendation Query
```mermaid
sequenceDiagram
    actor User
    participant Agent as LangGraph Agent
    participant Tool as product_qa_tool
    participant Neo4j as Neo4jStore
    participant LLM as GLM-4-Flash

    User->>Agent: "What category is MacBook Pro 16 in?"
    Agent->>Agent: classify_intent → product_qa
    Agent->>Tool: invoke(query)
    Tool->>Neo4j: resolve_product("What category is MacBook Pro 16 in?")
    Neo4j-->>Tool: ProductRef(name="MacBook Pro 16")
    Tool->>Neo4j: get_product_info("MacBook Pro 16")
    Neo4j-->>Tool: category_path: [Electronics, Computing, Laptops]
    Tool->>LLM: Explain category path + attributes
    LLM-->>Tool: "MacBook Pro 16 is in the Laptops category..."
    Tool-->>Agent: Category explanation
    Agent->>User: Final reply
```
