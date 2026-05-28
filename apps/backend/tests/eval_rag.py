"""
RAG evaluation script using RAGAS.

Measures Faithfulness, Answer Relevancy, and Context Precision
across your agent's different intent types.

Prerequisites:
    - Running PostgreSQL with pgvector (docker run pgvector/pgvector)
    - store_policies collection populated (python -m backend.db.vector_setup)
    - Valid .env with API keys

Usage:
    cd apps/backend && source venv/bin/activate
    python tests/eval_rag.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure backend/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from backend.agent import clean_query
from backend.agent import llm as gen_llm
from backend.config import settings
from backend.retrieval import get_policy_retriever
from backend.tools import list_orders_tool, order_status_tool

# ─── Setup LLM for RAGAS scoring ────────────────────────────────────────────────

ragas_client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_api_base,
    timeout=30,
)

_ragas_llm = llm_factory(
    model=settings.openai_model,
    client=ragas_client,
    temperature=0.0,
)

# Reuse the same OpenAI client for embeddings (Zhipu API serves both)
_ragas_embeddings = RagasOpenAIEmbeddings(
    client=ragas_client,
    model=settings.embedding_model,
)

# Retriever for the pipeline simulation
_retriever = get_policy_retriever()


# ─── Test Cases ─────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Policy queries (retrieval + generation) ──
    {
        "question": "What is the return policy?",
        "ground_truth": "Returns are accepted within 30 days for most items in original condition with all packaging. Refunds are processed within 5-7 business days.",
    },
    {
        "question": "Can I return electronics after 10 days?",
        "ground_truth": "Electronics may have a 14-day return period, so returning after 10 days is within the policy.",
    },
    {
        "question": "How much does shipping cost?",
        "ground_truth": "Free standard shipping on orders over $50. Standard delivery takes 5-7 business days. Expedited options are available for an additional fee.",
    },
    {
        "question": "Do you provide extended warranties?",
        "ground_truth": "We do not offer extended warranties, but third-party options may be available.",
    },
    {
        "question": "How long until I get my refund after returning something?",
        "ground_truth": "Refunds are processed within 5-7 business days after receiving the returned item.",
    },

    # ── Order queries (no retrieval — tool calls directly) ──
    {
        "question": "What's the status of order 1001?",
        "ground_truth": None,  # Depends on mock data — we skip correctness for these
    },
    {
        "question": "Show me all my orders",
        "ground_truth": None,
    },

    # ── Edge cases ──
    {
        "question": "Do you sell laptops?",
        "ground_truth": "The policy documents do not mention specific products for sale.",
    },
    {
        "question": "I want to talk to a human",
        "ground_truth": None,  # unknown intent — tests how the system handles it
    },
]


# ─── Pipeline simulation ────────────────────────────────────────────────────────

def run_agent_for_eval(user_input: str) -> dict:
    """
    Simulate the agent pipeline for a single question.
    Returns dict with keys: question, answer, contexts, ground_truth.
    """
    cleaned = clean_query(user_input)

    # Determine intent (simplified keyword matching, mirrors classify_intent)
    text = cleaned.lower()
    if any(w in text for w in ["weather", "temperature", "rain", "sunny"]):
        intent = "weather"
    elif any(w in text for w in ["all orders", "show me orders", "list orders"]):
        intent = "list_orders"
    elif any(w in text for w in ["order", "status of", "track"]):
        intent = "order"
    elif any(w in text for w in ["policy", "return", "refund", "shipping", "warranty", "退货", "退款"]):
        intent = "policy"
    else:
        intent = "unknown"

    # Execute tool
    tool_result = ""
    contexts = []

    if intent == "policy":
        docs_and_scores = _retriever.retrieve(cleaned, k=3, rerank=True)
        filtered = [(d, s) for d, s in docs_and_scores if s >= 7]
        if not filtered:
            filtered = docs_and_scores[:1]
        contexts = [doc.page_content for doc, _ in filtered]
        tool_result = "\n\n".join(contexts)
    elif intent == "order":
        import re
        match = re.search(r"\b(10\d{2,})\b", cleaned)
        oid = match.group(1) if match else "1001"
        tool_result = order_status_tool.invoke({"order_id": oid})
    elif intent == "list_orders":
        tool_result = list_orders_tool.invoke({})
    # weather and unknown: no tool result

    # Generate answer
    prompt = (
        "You are a helpful e-commerce support agent. "
        "Respond to the user's question based on the information below. "
        "Be concise, friendly, and honest. Only use information from the provided context. "
        "If the information is insufficient, say so.\n\n"
        f"User question: {cleaned}\n"
        f"Relevant information: {tool_result or 'No additional information available.'}\n\n"
        "Your reply:"
    )
    response = gen_llm.invoke(prompt)
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()

    return {
        "question": user_input,
        "answer": answer,
        "contexts": contexts,
    }


# ─── Prerequisite checks ────────────────────────────────────────────────────────

def _check_prerequisites():
    """Verify Postgres is reachable and the policy collection exists."""
    import psycopg2
    from sqlalchemy import make_url

    ok = True
    try:
        # settings.database_url is a SQLAlchemy URL (postgresql+psycopg2://...).
        # psycopg2.connect() needs a plain libpq DSN, so parse out the components.
        url = make_url(settings.database_url)
        conn = psycopg2.connect(
            dbname=url.database,
            user=url.username,
            password=url.password,
            host=url.host,
            port=url.port or 5432,
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT count(*) FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = 'store_policies'
        """)
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        if count == 0:
            print("  WARNING: store_policies collection is empty — run: python -m backend.db.vector_setup")
            ok = False
        else:
            print(f"  store_policies: {count} chunks")
    except Exception as e:
        print("  ERROR: Cannot connect to PostgreSQL — is the pgvector container running?")
        print("    docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16")
        print(f"    {e}")
        ok = False

    if not ok:
        print()
    return ok


# ─── Main Evaluation ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("RAG Evaluation — Faithfulness • Answer Relevancy • Context Precision")
    print("=" * 72)

    print("\n[0] Checking prerequisites...")
    if not _check_prerequisites():
        print("  Fix the issues above and re-run.")
        return

    # Run pipeline for all test cases
    print("\n[1/3] Running agent pipeline for each test case...")
    records = []
    for i, case in enumerate(TEST_CASES):
        case_dict: dict[str, str] = case  # type: ignore[assignment]
        q = case_dict["question"]
        print(f"  [{i+1}/{len(TEST_CASES)}] {q}")
        result: dict[str, Any] = run_agent_for_eval(q)
        result["ground_truth"] = case_dict.get("ground_truth", "")
        records.append(result)

    # Convert to Dataset
    dataset = Dataset.from_list(records)

    # Run RAGAS evaluation
    print("\n[2/3] Scoring with RAGAS (LLM-as-judge)...")

    # Old-style metrics are singletons — inject llm/embeddings into them
    faithfulness.llm = _ragas_llm
    answer_relevancy.llm = _ragas_llm
    answer_relevancy.embeddings = _ragas_embeddings
    context_precision.llm = _ragas_llm

    start = time.time()
    eval_result: Any = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    # Report
    print("\n[3/3] Results\n")
    df = eval_result.to_pandas()

    # Per-question scores
    for _, row in df.iterrows():
        f = row.get("faithfulness", 0) or 0
        ar = row.get("answer_relevancy", 0) or 0
        cp = row.get("context_precision", 0) or 0
        q = str(row.get("user_input", ""))[:68]

        issues = []
        if f < 0.8:
            issues.append("HALLUCINATION")
        if ar < 0.5:
            issues.append("OFF-TOPIC")
        if cp is not None and cp < 0.5:
            issues.append("NOISY-RETRIEVAL")

        status = " | ".join(issues) if issues else "OK"
        cp_str = f"{cp:.2f}" if cp else "N/A"
        print(f"  F={f:.2f}  AR={ar:.2f}  CP={cp_str}  [{status}]")
        print(f"    Q: {q}")
        if issues:
            a = str(row.get("response", ""))[:100]
            if a:
                print(f"    A: {a}")
        print()

    # Averages
    f_mean = df["faithfulness"].mean() or 0
    ar_mean = df["answer_relevancy"].mean() or 0
    cp_mean = df["context_precision"].mean() or 0
    print("-" * 72)
    print(f"  AVERAGE  F={f_mean:.2f}  AR={ar_mean:.2f}  CP={cp_mean:.2f}")
    print("-" * 72)

    # Diagnosis
    print("\n  Bottleneck:", end=" ")
    if f_mean < ar_mean and f_mean < cp_mean:
        print("FAITHFULNESS — LLM is adding claims not in context. Tighten the prompt.")
    elif ar_mean < f_mean and ar_mean < cp_mean:
        print("ANSWER RELEVANCY — Answers off-topic. Check intent routing or tool results.")
    elif cp_mean < f_mean and cp_mean < ar_mean:
        print("CONTEXT PRECISION — Retrieval is noisy. Tune re-ranker or score threshold.")
    else:
        print("All metrics balanced — pipeline is healthy.")

    # Export detailed results
    out_path = Path(__file__).parent / "eval_results.json"
    export = []
    for _, row in df.iterrows():
        export.append({
            "question": row.get("user_input", ""),
            "answer": row.get("response", ""),
            "contexts": row.get("retrieved_contexts", []),
            "faithfulness": row.get("faithfulness", None),
            "answer_relevancy": row.get("answer_relevancy", None),
            "context_precision": row.get("context_precision", None),
        })
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Detailed results: {out_path}")


if __name__ == "__main__":
    main()
