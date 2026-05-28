"""
Knowledge graph module for e-commerce product/category/policy relationships.

Provides:
    - KnowledgeStore: PostgreSQL-backed graph store
    - Switchable retrievers: VectorPolicyRetriever, GraphPolicyRetriever, HybridRetriever
    - Factory: create_policy_retriever(mode)
    - DB setup: python -m backend.knowledge.schema
"""

from backend.knowledge.graph_store import KnowledgeStore, get_knowledge_store
from backend.knowledge.retrievers import (
    GraphPolicyRetriever,
    HybridRetriever,
    PolicyRetriever,
    VectorPolicyRetriever,
    create_policy_retriever,
)

__all__ = [
    "KnowledgeStore",
    "get_knowledge_store",
    "PolicyRetriever",
    "VectorPolicyRetriever",
    "GraphPolicyRetriever",
    "HybridRetriever",
    "create_policy_retriever",
]
