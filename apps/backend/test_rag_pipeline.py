"""
Quick RAG pipeline test after expanding the policy corpus.

Usage:
    cd apps/backend
    source venv/bin/activate
    PYTHONPATH=. python test_rag_pipeline.py
"""

import time
from backend.retrieval import get_policy_retriever

retriever = get_policy_retriever()

queries = [
    "Can I return headphones after 10 days?",
    "What is the shipping cost for furniture?",
    "Do cosmetics have a warranty?",
    "What is the return policy for food?",
    "How long is the warranty for electronics?",
    "Can I cancel my order for a customized item?",
    "What happens if my sports equipment arrives damaged?",
    "Do you offer price matching for jewelry?",
    "What payment methods do you accept?",
    "How do I contact customer support for a kitchen appliance?",
]

print("=" * 80)
print("RAG RETRIEVAL TEST — Expanded Policy Corpus")
print("=" * 80)

for q in queries:
    print(f"\n🔍 Query: {q}")
    print("-" * 60)
    
    start = time.time()
    
    # Dense only
    t0 = time.time()
    dense = retriever._dense_retrieve(q, k=5)
    t_dense = time.time() - t0
    
    # Sparse only
    t0 = time.time()
    sparse = retriever._sparse_retrieve(q, k=5)
    t_sparse = time.time() - t0
    
    # RRF fusion (no rerank)
    t0 = time.time()
    fused = retriever._rrf_fuse(dense, sparse)
    t_fuse = time.time() - t0
    
    # Full pipeline with LLM rerank
    t0 = time.time()
    final = retriever.retrieve(q, k=3, rerank=True)
    t_rerank = time.time() - t0
    
    total = time.time() - start
    
    print(f"   Dense ({len(dense)} docs): {t_dense:.2f}s")
    print(f"   Sparse ({len(sparse)} docs): {t_sparse:.2f}s")
    print(f"   RRF Fusion ({len(fused)} docs): {t_fuse:.3f}s")
    print(f"   LLM Rerank ({len(final)} docs): {t_rerank:.2f}s")
    print(f"   Total: {total:.2f}s")
    
    print(f"\n   📄 Top results:")
    for i, (doc, score) in enumerate(final[:3], 1):
        title = doc.page_content.split('\n')[0][:80]
        print(f"      [{i}] score={score:.2f} | {title}...")

print("\n" + "=" * 80)
print("Test complete.")
