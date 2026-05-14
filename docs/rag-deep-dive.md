# RAG — Retrieval-Augmented Generation

## The problem RAG solves

LLMs only know what was in their training data. They hallucinate on private or recent information. Your store's return policy? Your actual order status? The LLM can't know those.

RAG fixes this by **retrieving relevant documents before generating a response** and injecting them into the LLM's prompt as context. The LLM reads the retrieved text and answers from it, instead of from memory.

```
User: "What's your return policy?"
        │
        ▼
   [Retrieve relevant policy chunks from vector DB]
        │
        ▼
   [Inject retrieved text into LLM prompt]
        │
        ▼
   LLM: "Our store offers a 30-day return window..."
              (grounded in actual policy text)
```

The pipeline has **4 stages**:

```
Ingestion  →  Retrieval  →  Fusion  →  Answer Generation
(vector_setup.py)  (retrieval.py)  (retrieval.py)  (graph/nodes.py)
```

---

# Stage 1: Ingestion — turning text into searchable vectors

Ingestion is in `backend/db/vector_setup.py`. There are 3 steps.

### Step 1: Load the source documents

```python
# vector_setup.py:23
loader = TextLoader("data/store_policies.txt")
documents = loader.load()
```

The source is a 3-section text file (return policy, shipping, warranty). `TextLoader` reads it as a single `Document` object with `page_content` = the raw text.

### Step 2: Chunk it

```python
# vector_setup.py:27
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
```

**Why chunk?** Embedding models have a context limit (usually 512 tokens). You can't embed a 10-page policy as one blob. You split it into overlapping pieces.

**Why overlap (50 chars)?** Imagine your policy says "...return within 30 days. Electronics have a 14-day return period." If the split point falls between those sentences, one chunk gets "return within 30 days." and the next gets "Electronics have a 14-day return period." Overlap ensures that at least one chunk contains the complete thought.

`RecursiveCharacterTextSplitter` tries to split on natural boundaries first: `\n\n` → `\n` → ` ` → `""`. So it breaks on paragraph boundaries before sentence boundaries, before character boundaries. This keeps chunks semantically coherent.

### Step 3: Embed and store

```python
# vector_setup.py:31-45
embeddings = OpenAIEmbeddings(model="embedding-2", ...)
vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="store_policies",
)
```

Each chunk passes through an **embedding model** (Zhipu `embedding-2`) which outputs a vector — a list of ~1024 floats. Vectors that are "close" (low cosine distance) represent semantically similar text. "Refund policy" and "return window" will have similar vectors even though they share no keywords.

These vectors are stored in PostgreSQL with the pgvector extension:

```
┌─────────────────────────────────────────────────────────┐
│                 store_policies table                     │
├──────────┬──────────────────┬────────────────────────────┤
│    id    │   page_content   │         embedding          │
├──────────┼──────────────────┼────────────────────────────┤
│    1     │ "Return Policy:  │ [0.023, -0.451, 0.891, ...│
│          │  Our store..."   │                            │
│    2     │ "Shipping and    │ [0.112, -0.332, 0.761, ...│
│          │  Delivery: We..."│                            │
│    3     │ "Warranty Info:  │ [0.091, -0.221, 0.651, ...│
│          │  All electronic."│                            │
└──────────┴──────────────────┴────────────────────────────┘
```

---

# Stage 2: Retrieval — finding relevant chunks

Retrieval is in `backend/retrieval.py`. At query time, two **independent searches** run and their results are fused.

## Dense retrieval (semantic search)

```python
# retrieval.py:118-133
def _dense_retrieve(self, query: str, k: int = 5):
    docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
    return [(doc, -score) for doc, score in docs_and_scores]
```

**How it works:**
1. Convert the user query into a vector (same embedding model)
2. pgvector computes cosine distance between the query vector and every stored chunk vector
3. Return the k closest ones

**What it's good at:** Finding "return window" when the query says "how long to send back." Semantic similarity catches the intent even when keywords don't match.

**What it's bad at:** "What is order ORD-003?" — embedding models aren't trained on your order IDs. The vector for "ORD-003" is just a random point in vector space.

## Sparse retrieval (keyword search)

```python
# retrieval.py:135-140
def _sparse_retrieve(self, query: str, k: int = 5):
    tokens = self._tokenize(query)          # split into words
    scores = self.bm25.get_scores(tokens)   # score every chunk
    top_indices = sorted(...)[:k]           # keep top k
    return [(self.chunks[i], scores[i]) for i in top_indices]
```

**How BM25 works:**

BM25 scores a document against a query using two factors:

```
BM25(q, d) = Σ IDF(term) × TF_term × (k1 + 1) / (TF_term + k1 × doc_length_ratio)
```

| Term | Meaning | Intuition |
|------|---------|-----------|
| **TF** (Term Frequency) | How often the word appears in this doc | More mentions = higher score (but capped — after ~3 times, extra mentions don't help) |
| **IDF** (Inverse Document Frequency) | log(N / df), where N = total docs, df = docs containing this word | Rare words ("warranty") score higher than common words ("the", "our") |
| **k1** | Saturation parameter (typically 1.2) | Controls how fast TF bonus saturates |
| **doc_length_ratio** | This doc's length / average doc length | Prevents long docs from dominating just because they contain more words |

Example: for query "electronics return" against the 3 policy chunks:

```
"electronics" appears in: warranty chunk (1 time) → high IDF (rare word)
"return" appears in: return chunk (once) → medium IDF

Score(return_chunk)    = IDF("return") × ... → medium
Score(shipping_chunk)  = 0 (neither word appears)
Score(warranty_chunk)  = IDF("electronics") × ... → high for "electronics", 0 for "return"
```

**What it's good at:** Exact IDs, product names, numbers — anything where keyword matching is the right tool.

**What it's bad at:** "how do I send stuff back" won't match "return policy" at all.

### Why you need both

| Query | Dense | Sparse |
|-------|-------|--------|
| "return window" | Finds return policy (semantic match) | Finds return policy (keyword match) |
| "how do I send this back" | Finds return policy (understands "send back" ≈ "return") | May miss (no shared keywords) |
| "track ORD-003" | Fails (no semantic meaning in order IDs) | Finds it (exact match) |
| "warranty for laptop" | Finds warranty chunk | Finds warranty chunk |

Dense catches rephrasing. Sparse catches exact IDs. Together they're more robust than either alone.

---

# Why RRF is smarter than simple score merging

The naive approach: sort all dense results by score, all sparse results by score, interleave them. Problem: the scales are incomparable. A cosine similarity of 0.85 is not "3x better" than a BM25 score of 15. You can't just add them.

The solution is **Reciprocal Rank Fusion (RRF)**:

```python
# retrieval.py:142-166
def _rrf_fuse(list_a, list_b, k=60):
    for rank, (doc, _) in enumerate(list_a, start=1):
        doc_scores[key] += 1.0 / (k + rank)  # rank=1 → 1/61, rank=2 → 1/62, ...

    for rank, (doc, _) in enumerate(list_b, start=1):
        doc_scores[key] += 1.0 / (k + rank)
```

**Key insight:** RRF doesn't care about the original scores. It only cares about **rank order**. A document ranked #1 in dense and #3 in sparse gets:

```
RRF score = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.0164 + 0.0159 = 0.0323
```

A document ranked #2 in both:
```
RRF score = 1/(60+2) + 1/(60+2) = 1/62 + 1/62 ≈ 0.0323
```

**Why k=60?** A large k makes 1/(k+rank) close to linear — it reduces the "winner-take-all" effect where rank #1 gets an overwhelming bonus. Lower k (like k=5) makes rank #1 dramatically more valuable than rank #2.

| k | rank=1 | rank=2 | ratio |
|---|--------|--------|-------|
| 5 | 0.167 | 0.143 | 1.17x |
| 60 | 0.0164 | 0.0161 | 1.02x |

k=60 means "being #1 vs #2 barely matters, but being in the top-5 at all matters a lot vs not being in the list." This is generally desired for RAG — you want consensus across both methods.

---

# Stage 3: Re-ranking — the precision filter

After RRF fusion, there are ~5 candidates. But RRF only knows about ranks — it doesn't understand which chunks genuinely answer the question. That's what re-ranking is for.

The current approach uses the LLM as a re-ranker:

```python
# retrieval.py:174-218
def _llm_rerank(self, query, docs, top_n=3):
    prompt = (
        "Rate how relevant each document is to answering the user's question. "
        "For each document, respond with its number and a score from 1 to 10..."
        f"User question: {query}\n\n"
        + "\n\n".join(doc_blocks)
    )
```

This uses **semantic understanding** to judge relevance. RRF might rank the shipping chunk high because it shares a few terms, but the LLM can read and say "this chunk is about delivery times, not returns" → score 2.

### The alternatives for re-ranking

| Method | Latency | Cost | Quality |
|--------|---------|------|---------|
| **LLM (current approach)** | ~200ms | ~$0.001/call | Best (reads and judges) |
| **Cross-encoder** (`ms-marco-MiniLM-L-6-v2`) | ~5ms | Free (local) | Very good |
| **No re-rank** (just RRF top-N) | 0ms | Free | Good enough for small doc sets |

A cross-encoder works by passing the (query, document) pair through a BERT-style model that outputs a single relevance score. Unlike embedding vectors (which encode query and doc independently), a cross-encoder sees them together and can model their interaction.

---

# Stage 4: Answer Generation — grounded response

The retrieved text enters the LLM prompt as context. In the graph, this happens in `generate_reply`:

```python
# graph/nodes.py — generate_reply node
prompt = f"""Using the following information, answer the user's question.
Information: {state['tool_result']}
Question: {state['cleaned_input']}
Answer:"""
```

The tool result (from `policy_retriever_tool`) is already the retrieved + filtered policy text. The LLM reads it and formats a natural language answer.

This is the "augmented generation" part — the LLM generates text, but **augmented** by the retrieved context. It's not answering from memory; it's reading the provided text and paraphrasing it.

---

# The full pipeline, traced step by step

Here's what happens when a user asks *"can I return headphones after 2 weeks?"*

```
1. INGESTION (done once, at setup)
   store_policies.txt → TextLoader → RecursiveCharacterTextSplitter(500, 50)
   → 3 chunks × embedding-2 → pgvector store_policies table

2. RETRIEVAL (at query time)
   query: "can I return headphones after 2 weeks?"

   DENSE: query → embedding-2 → pgvector cosine search
          → [return_chunk: 0.89, shipping_chunk: 0.45, warranty_chunk: 0.32]

   SPARSE: query → tokenize → BM25 scores
          → [return_chunk: 2.1, warranty_chunk: 1.8, shipping_chunk: 0.0]

   RRF FUSE:
          return_chunk:   1/(60+1) + 1/(60+1) = 0.0328  (rank 1 in both)
          shipping_chunk: 1/(60+2) + 1/(60+3) = 0.0320
          warranty_chunk: 1/(60+3) + 1/(60+2) = 0.0320
          → [return_chunk, shipping_chunk, warranty_chunk]

3. RE-RANK (LLM batch call)
   LLM reads all 3 chunks, scores each against query:
          return_chunk: 9   ← "30-day return window" matches query
          shipping_chunk: 2 ← about delivery, not returns
          warranty_chunk: 1 ← about defects, not returns

4. FILTER (score >= 7, in policy.py)
          Only return_chunk passes → used as context

5. GENERATE (in generate_reply node)
   LLM prompt: "Using: [return_chunk]... Answer: can i return headphones after 2 weeks?"
   LLM: "Yes! Our return policy allows returns within 30 days of delivery.
         Headphones fall under the standard return window, so you can absolutely
         return them after 2 weeks for a full refund."
```

---

# Key design decisions in this system

### 1. `pre_delete_collection=True` in vector_setup.py

Every time `vector_setup.py` runs, it wipes the `store_policies` collection and rebuilds it. This means you can edit `store_policies.txt` and re-run — no stale chunks, no duplicates. For a development workflow, this is correct. In production, you'd want incremental updates.

### 2. Score threshold of 7 in policy.py

```python
# tools/policy.py:10-11
filtered = [(doc, score) for doc, score in docs_and_scores if score >= 7]
```

This is a **precision filter.** If even the "best" chunk only scores 5 (barely relevant), you'd rather say "I don't know" than feed the LLM irrelevant text it might hallucinate from. The fallback to top-1 ensures you still answer when nothing is great — better a weak answer than no answer.

### 3. Truncating to 400 chars in re-rank prompt

```python
# retrieval.py:189
doc_blocks.append(f"[{i}] {doc.page_content[:400]}")
```

The re-ranker doesn't need the full chunk to judge relevance. The first 400 characters of a 500-char chunk captures the semantic topic. This saves prompt tokens.

### 4. Chunk overlap = 10% (50/500)

A 10% overlap means ~90% of each chunk is unique content. The overlap is just enough to prevent semantic breaks at chunk boundaries. Too much overlap (50%) means redundant storage and redundant retrieval; too little (0%) means you'll sometimes lose context at the split point.

---

# Where this RAG system can improve

| Issue | Current | Better |
|-------|---------|--------|
| **Chunk count** | 3 chunks for 3 policy sections | Add metadata (section headers), use smaller chunks, or add a hierarchy |
| **Re-ranker cost** | 1 LLM call per query | Cross-encoder (free, local, 5ms) for this small doc set |
| **No query rewriting** | Raw user query goes to retrieval | LLM expands "send it back" → "return policy refund" before retrieval |
| **Single data source** | One text file | Could add FAQ, product specs, past tickets |
| **Embedding model** | Zhipu `embedding-2` | Benchmark vs `text-embedding-3-small` (OpenAI) or `bge-large-en` (local) |
| **No evaluation** | Manual testing | RAGAS framework: faithfulness, answer relevancy, context precision |
