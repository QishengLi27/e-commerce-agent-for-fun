# RAG Evaluation Pipeline

## Why evaluate RAG?

A RAG pipeline has multiple failure modes, and they compound:

```
Bad retrieval → wrong context → hallucinated answer
Good retrieval → LLM ignores context → hallucinated answer
Good retrieval → LLM uses context → correct answer = only 1 of 3 paths is good
```

Without evaluation, you don't know which stage is failing. You just know "the answers feel wrong sometimes." Evaluation decomposes the problem so you know exactly where to fix it.

## The evaluation framework: RAGAS

RAGAS (RAG Assessment) is the standard open-source framework. It defines metrics that isolate each stage of the pipeline:

```
                    ┌──────────────┐
     question ─────►│   RETRIEVAL  │────► context ─────┐
                    └──────────────┘                   │
                           ▲                           ▼
                           │                   ┌──────────────┐
                    Context Precision          │  GENERATION  │────► answer
                    Context Recall             └──────────────┘
                                                       ▲
                                                       │
                                                Faithfulness
                                                Answer Relevancy
```

Each metric pinpoints a different failure:

| Metric | Measures | Bad score means |
|--------|----------|-----------------|
| **Context Precision** | Are retrieved chunks relevant? | Retrieval is noisy — pulling in unrelated chunks |
| **Context Recall** | Did we retrieve ALL relevant chunks? | Retrieval is missing information — gaps in coverage |
| **Faithfulness** | Does the answer contain ONLY facts from context? | LLM is hallucinating — making up claims not in the source |
| **Answer Relevancy** | Does the answer address the question? | LLM is rambling or answering a different question |
| **Answer Correctness** | Is the answer factually correct? (needs ground truth) | Overall failure — either retrieval or generation is wrong |

---

## Metric 1: Faithfulness

**What it asks:** "Of all the claims in the answer, how many can be found in the retrieved context?"

**How it's computed:**

```
Step 1: LLM extracts atomic claims from the answer.
        Answer: "You can return headphones within 30 days for a full refund."
        Claims: ["return window is 30 days", "headphones are eligible", "refund is full"]

Step 2: For each claim, LLM checks: "Is this claim supported by the context?"
        Context: "Our store offers a 30-day return window for most items."
        Claim 1 → YES
        Claim 2 → NO (context doesn't mention headphones)
        Claim 3 → NO (context doesn't mention refund amount)

Step 3: Faithfulness = supported_claims / total_claims = 1/3 = 0.33
```

**Why it matters in your system:** Your `validate_reply` node already does a crude version of this — it asks the LLM "does this answer match the tool result?" Faithfulness formalizes it into a measurable score you can track over time and across prompt variations.

**Threshold:** Faithfulness < 0.8 means your LLM is adding claims not in the context. Fix the prompt (add "only use the provided information") or improve retrieval so the context is more complete.

---

## Metric 2: Answer Relevancy

**What it asks:** "Does the answer actually address the user's question?"

**How it's computed:**

```
Step 1: LLM generates reverse questions from the answer.
        Answer: "Our return policy allows returns within 30 days..."
        Generated questions: ["What is the return window?", "How do I return items?"]

Step 2: Compute cosine similarity between each generated question and the original question.
        Original: "Can I return headphones after 2 weeks?"
        Generated Q1: "What is the return window?" → cosine similarity 0.72
        Generated Q2: "How do I return items?" → cosine similarity 0.45

Step 3: Answer Relevancy = mean similarity = (0.72 + 0.45) / 2 = 0.585
```

**Why this works:** If the answer is about shipping when the user asked about returns, the reverse-generated questions won't match the original question → low score.

**Why it matters in your system:** Your `classify_intent` node routes to tools, but the LLM in `generate_reply` might still go off-topic. Answer relevancy catches this.

**Threshold:** Answer Relevancy < 0.5 means the answer is off-topic. Check if the right tool was called, or if the LLM is ignoring the context.

---

## Metric 3: Context Precision

**What it asks:** "Of the chunks we retrieved, how many are actually relevant?"

**How it's computed:**

```
Step 1: For each retrieved chunk, LLM judges: "Is this chunk relevant to the question?"

        Question: "Can I return headphones after 2 weeks?"
        Chunk 1 (return policy):  YES, relevant
        Chunk 2 (shipping):       NO, not relevant
        Chunk 3 (warranty):       NO, not relevant

Step 2: Context Precision = Σ(relevant_at_rank_k / k) / total_relevant

        rank 1: 1/1 = 1.0    (relevant)
        rank 2: 0/2 = 0.0    (not relevant)
        rank 3: 0/3 = 0.0    (not relevant)

        Context Precision = (1.0 + 0.0 + 0.0) / 1 = 1.0
```

Wait — score is 1.0? That seems wrong when 2 of 3 chunks are irrelevant. This exposes a subtlety: Context Precision is **rank-weighted.** It heavily rewards getting the first chunk right and penalizes irrelevant chunks less if they're ranked lower.

The formula is actually:

```
Context Precision@K = Σ(P@k × rel_k) / total_relevant

where P@k is precision at rank k, and rel_k is 1 if chunk k is relevant.
```

This means: the metric prioritizes "is the best stuff at the top?" over "is everything relevant?" It's asking whether the user would see the right answer in the first few results.

**Why it matters in your system:** Your RRF fusion + re-rank pipeline is designed to push relevant chunks to the top. Context Precision validates that this is working.

**Threshold:** Context Precision < 0.5 means your retrieval is noisy — you're pulling in irrelevant chunks and potentially confusing the LLM. Check your re-ranker.

---

## Metric 4: Context Recall

**What it asks:** "Of ALL the relevant chunks that exist, how many did we retrieve?"

**How it's computed:**

```
Step 1: LLM examines each retrieved chunk and extracts the sentences that helped
        answer the question (the "attributed" content).

Step 2: Context Recall = |attributed sentences| / |total sentences that could answer the question|

        Question: "What is the return window?"
        Retrieved chunks: [return_policy_chunk, shipping_chunk]
        Attributed content: "Our store offers a 30-day return window for most items."

        Ground truth (from all policy docs): The return policy contains 3 relevant
        sentences about the return window. We attributed 1 of them.

        Context Recall = 1/3 = 0.33
```

**The problem with this metric:** It requires knowing what a "complete" answer looks like, which means you either need ground truth labels or an LLM to estimate what should have been retrieved. In practice, you often skip this metric unless you have a labeled dataset.

**Why it matters in your system:** You have only 3 policy chunks. If your recall is low, it means your chunking strategy is losing information — relevant text is split across chunks and the right chunk isn't being retrieved.

**Threshold:** Context Recall < 0.6 means you're missing important context. Try different chunk sizes or add query rewriting to capture more relevant chunks.

---

## Metric 5: Answer Correctness

**What it asks:** "Is the answer factually correct against ground truth?"

**How it's computed:**

```
Step 1: LLM extracts factual statements from both the generated answer and the ground truth.

Step 2: For each generated statement, check if it appears in (or is semantically equivalent
        to) a ground truth statement → True Positive.
        For each generated statement not in ground truth → False Positive.
        For each ground truth statement not in the answer → False Negative.

Step 3: F1 score = 2 × (Precision × Recall) / (Precision + Recall)

        Precision = TP / (TP + FP)  — how many of our claims are correct?
        Recall    = TP / (TP + FN)  — how many correct claims did we include?

Step 4: Answer Correctness = weighted combination of factual F1 and semantic similarity
        between the full answer and ground truth.
```

**Why it matters:** This is the "final exam" — it tells you whether the whole pipeline works end-to-end. But it requires ground truth answers, which are expensive to create.

---

## Full metrics summary

| Metric | Needs | What it catches | Quick fix if low |
|--------|-------|-----------------|------------------|
| Faithfulness | Context + Answer | Hallucination | Tighten prompt: "only use provided info" |
| Answer Relevancy | Question + Answer | Off-topic answers | Check intent routing, re-rank results |
| Context Precision | Question + Context | Noisy retrieval | Tune score threshold, improve re-ranker |
| Context Recall | Question + Context (or ground truth) | Missing context | Smaller chunks, query rewriting |
| Answer Correctness | Question + Answer + Ground Truth | Overall failure | Depends — check the component metrics first |

---

## Building an evaluation script for this project

Here's what a RAGAS evaluation pipeline looks like for this specific codebase:

```python
# tests/eval_rag.py
"""
RAG evaluation using RAGAS.
pip install ragas pandas
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from backend.retrieval import get_policy_retriever
from backend.agent import llm as eval_llm  # reuse your existing LLM

# --- Test cases ---
# Each is a question + what we expect the pipeline to return.
# For correctness, you'd also add ground_truth.
test_cases = [
    {
        "question": "What is the return policy for electronics?",
        "ground_truth": "Electronics have a 14-day return period.",
    },
    {
        "question": "How long does shipping take?",
        "ground_truth": "Standard delivery takes 5-7 business days.",
    },
    {
        "question": "Do you offer extended warranties?",
        "ground_truth": "We do not offer extended warranties, but third-party options may be available.",
    },
    {
        "question": "What's the weather in Tokyo?",
        "ground_truth": None,  # Not a policy question — tests retrieval boundary
    },
]

# --- Run the pipeline and collect results ---
retriever = get_policy_retriever()
records = []

for case in test_cases:
    # Simulate retrieval
    docs_and_scores = retriever.retrieve(case["question"], k=3, rerank=True)
    contexts = [doc.page_content for doc, _ in docs_and_scores]

    # Simulate generation (simplified — in production, call your graph)
    from langchain_core.messages import HumanMessage
    prompt = (
        "Using the following store policy information, answer the user's question. "
        "Only use information from the provided context.\n\n"
        f"Context:\n{chr(10).join(contexts)}\n\n"
        f"Question: {case['question']}\n\n"
        "Answer:"
    )
    response = eval_llm.invoke(prompt)
    answer = response.content.strip()

    records.append({
        "question": case["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": case["ground_truth"],
    })

# --- Evaluate with RAGAS ---
dataset = Dataset.from_list(records)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=eval_llm,             # RAGAS uses your LLM for scoring
    embeddings=retriever.embeddings,
)

# --- Report ---
df = result.to_pandas()
print("\n=== Per-Question Scores ===")
print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string())
print(f"\n=== Averages ===")
print(f"Faithfulness:        {df['faithfulness'].mean():.2f}")
print(f"Answer Relevancy:    {df['answer_relevancy'].mean():.2f}")
print(f"Context Precision:   {df['context_precision'].mean():.2f}")

# --- Diagnose failures ---
print("\n=== Diagnoses ===")
for _, row in df.iterrows():
    issues = []
    if row["faithfulness"] < 0.8:
        issues.append("HALLUCINATION — answer contains claims not in context")
    if row["answer_relevancy"] < 0.5:
        issues.append("OFF-TOPIC — answer doesn't address the question")
    if row["context_precision"] < 0.5:
        issues.append("NOISY RETRIEVAL — irrelevant chunks in top results")
    if issues:
        print(f"Q: {row['question'][:60]}... → {', '.join(issues)}")
    else:
        print(f"Q: {row['question'][:60]}... → OK")
```

This gives you:

1. **Per-question scores** — which specific questions fail
2. **Average scores** — overall pipeline health
3. **Diagnoses** — what to fix for each failure

---

## How evaluation drives iteration

Once you have metrics, the improvement cycle becomes:

```
1. Measure baseline scores across all metrics
2. Identify the worst metric → that's your bottleneck
3. Make ONE change (chunk size, prompt, re-ranker)
4. Re-measure — did the target metric improve? Did others regress?
5. Repeat
```

Example iteration:

```
Baseline:  Faithfulness=0.65, Context Precision=0.80, Answer Relevancy=0.70
           → Bottleneck is Faithfulness (hallucination)

Change:    Add "Only use information from the provided context" to generate_reply prompt
Result:    Faithfulness=0.85, Context Precision=0.80, Answer Relevancy=0.72
           → Now bottleneck is Answer Relevancy

Change:    Tune re-rank score threshold from 7 → 8
Result:    Faithfulness=0.88, Context Precision=0.87, Answer Relevancy=0.78
           → Continue...
```

Without metrics, you're just guessing whether a change helped or hurt. With metrics, you can prove it.

---

## Metrics to skip for this project (for now)

| Metric | Why skip |
|--------|----------|
| **Context Recall** | Requires ground truth of "all relevant chunks" — overkill for a 3-chunk policy store |
| **Answer Correctness** | Requires ground truth answers — worth adding once you have 20+ test cases |
| **Context Relevancy** | Redundant with Context Precision for small doc sets |

Start with **Faithfulness**, **Answer Relevancy**, and **Context Precision**. These three catch 90% of RAG failures with zero ground truth labels — RAGAS uses your LLM to score all of them.

---

## Beyond RAGAS: what to monitor in production

RAGAS is for offline evaluation (before you ship). In production, track these:

| Signal | What it means | How to track |
|--------|---------------|--------------|
| **Thumbs up/down ratio** | Overall user satisfaction | Add 👍/👎 to your chat UI |
| **"I don't know" rate** | Retrieval confidence below threshold | Count how often `policy_retriever_tool` returns empty or score<7 |
| **Cache hit rate** | How often semantic cache serves the response | You already log `Cache hit!` |
| **Retrieval latency** | Is the RAG pipeline slowing things down? | Time `retriever.retrieve()` calls |
| **LLM re-rank failure rate** | Circuit breaker trips on re-ranker | Your circuit breaker already tracks this |
