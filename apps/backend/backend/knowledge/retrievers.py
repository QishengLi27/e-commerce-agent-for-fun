"""
Switchable policy retrievers: vector, graph, and hybrid modes.

Each retriever implements the same PolicyRetriever interface:
    retrieve(query: str, k: int = 3) -> str

Usage:
    from backend.config import settings, RetrievalMode
    from backend.knowledge.retrievers import create_policy_retriever

    retriever = create_policy_retriever(settings.retrieval_mode)
    result = retriever.retrieve("Can I return headphones after 10 days?")
"""

from abc import ABC, abstractmethod

from backend.knowledge.graph_store import get_knowledge_store

# ── Interface ──────────────────────────────────────────────────────────────────

class PolicyRetriever(ABC):
    """Unified interface for policy retrieval — vector, graph, or hybrid."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> str:
        """Return formatted policy text relevant to the query."""
        ...


# ── Vector Retriever (existing hybrid RAG, unchanged) ──────────────────────────

class VectorPolicyRetriever(PolicyRetriever):
    """Delegates to the existing hybrid RAG pipeline (pgvector + BM25 + RRF + re-rank)."""

    def retrieve(self, query: str, k: int = 3) -> str:
        from backend.retrieval import get_policy_retriever

        retriever = get_policy_retriever()
        docs_and_scores = retriever.retrieve(query, k=k, rerank=True)
        filtered = [(doc, score) for doc, score in docs_and_scores if score >= 7]
        if not filtered:
            filtered = docs_and_scores[:1]
        return "\n\n".join([doc.page_content for doc, _ in filtered])


# ── Graph Retriever (PostgreSQL knowledge graph, deterministic) ────────────────

class GraphPolicyRetriever(PolicyRetriever):
    """Queries the PostgreSQL knowledge graph for exact policy matches.

    Traverses: product → category → policy_rules via SQL JOINs.
    No embedding or semantic search — deterministic lookups only.
    """

    def retrieve(self, query: str, k: int = 3) -> str:
        store = get_knowledge_store()

        # Try product-level match first (product → category → policy)
        policies = store.query_product_policies(query)
        if policies:
            return self._format_policies(policies)

        # Fall back to category-level match
        policies = store.query_category_policies(query)
        if policies:
            return self._format_policies(policies)

        return ""

    @staticmethod
    def _format_policies(policies: list[dict]) -> str:
        """Format policy results as readable text for the LLM."""
        parts = []
        for p in policies:
            product_or_cat = p.get("product_name") or p.get("category_name", "")
            parts.append(
                f"[{p['policy_type'].upper()}] {p['summary']}\n"
                f"Applies to: {product_or_cat}\n"
                f"Details: {p['details']}"
            )
        return "\n\n".join(parts)


# ── Hybrid Retriever (graph first, vector fallback) ────────────────────────────

class HybridRetriever(PolicyRetriever):
    """Try graph first for deterministic matches, fall back to vector RAG."""

    def __init__(self):
        self._graph = GraphPolicyRetriever()
        self._vector = VectorPolicyRetriever()

    def retrieve(self, query: str, k: int = 3) -> str:
        result = self._graph.retrieve(query, k=k)
        if result:
            return result
        return self._vector.retrieve(query, k=k)


# ── Factory ────────────────────────────────────────────────────────────────────

def create_policy_retriever(mode) -> PolicyRetriever:
    """Factory: returns the right retriever based on RetrievalMode.

    Args:
        mode: RetrievalMode enum value or string ("vector", "graph", "hybrid")

    Returns:
        PolicyRetriever instance
    """
    # Accept string or enum
    mode_str = mode.value if hasattr(mode, "value") else str(mode)

    if mode_str == "vector":
        return VectorPolicyRetriever()
    elif mode_str == "graph":
        return GraphPolicyRetriever()
    elif mode_str == "hybrid":
        return HybridRetriever()
    else:
        raise ValueError(
            f"Unknown retrieval mode: {mode_str}. "
            f"Expected one of: vector, graph, hybrid"
        )
