"""
Tests for the knowledge graph module.

Requires PostgreSQL with knowledge tables set up:
    python -m backend.knowledge.schema
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from backend.config import RetrievalMode
from backend.knowledge.graph_store import get_knowledge_store
from backend.knowledge.retrievers import (
    GraphPolicyRetriever,
    HybridRetriever,
    VectorPolicyRetriever,
    create_policy_retriever,
)

# ── KnowledgeStore tests ───────────────────────────────────────────────────


class TestKnowledgeStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.store = get_knowledge_store()

    def test_query_product_policies_headphones(self):
        policies = self.store.query_product_policies("headphones")
        assert len(policies) >= 1
        policy_types = {p["policy_type"] for p in policies}
        assert "return" in policy_types
        # Headphones are Audio → Electronics Return (14 days)
        return_policy = next(p for p in policies if p["policy_type"] == "return")
        assert "14-day" in return_policy["summary"]
        assert "Headphones" in return_policy["product_name"]
        assert return_policy["category_name"] == "Audio"

    def test_query_product_policies_laptop(self):
        policies = self.store.query_product_policies("laptop")
        policy_types = {p["policy_type"] for p in policies}
        assert "return" in policy_types
        assert "warranty" in policy_types
        # Laptop is Electronics → 14-day return
        return_policy = next(p for p in policies if p["policy_type"] == "return")
        assert "14-day" in return_policy["summary"]

    def test_query_product_policies_tshirt(self):
        policies = self.store.query_product_policies("t-shirt")
        {p["policy_type"] for p in policies}
        # T-Shirt is General → standard return (30 days)
        return_policy = next(p for p in policies if p["policy_type"] == "return")
        assert "30-day" in return_policy["summary"]

    def test_product_not_found_returns_empty(self):
        policies = self.store.query_product_policies("nonexistent_product_xyz")
        assert policies == []

    def test_get_product_info(self):
        info = self.store.get_product_info("headphones")
        assert info is not None
        assert info["category_name"] == "Audio"
        assert info["price"] == 79.99
        assert len(info["policies"]) >= 1

    def test_get_product_info_not_found(self):
        info = self.store.get_product_info("unicorn_dust")
        assert info is None

    def test_search_products(self):
        results = self.store.search_products("phone")
        names = {r["name"] for r in results}
        assert "Headphones" in names
        assert "Phone Case" in names
        assert len(results) >= 2

    def test_get_all_categories(self):
        cats = self.store.get_all_categories()
        cat_names = {c["name"] for c in cats}
        assert "Electronics" in cat_names
        assert "Audio" in cat_names
        assert "General" in cat_names


# ── Retriever tests ────────────────────────────────────────────────────────


class TestRetrievers:
    def test_factory_creates_vector(self):
        r = create_policy_retriever(RetrievalMode.VECTOR)
        assert isinstance(r, VectorPolicyRetriever)

    def test_factory_creates_graph(self):
        r = create_policy_retriever(RetrievalMode.GRAPH)
        assert isinstance(r, GraphPolicyRetriever)

    def test_factory_creates_hybrid(self):
        r = create_policy_retriever(RetrievalMode.HYBRID)
        assert isinstance(r, HybridRetriever)

    def test_factory_accepts_string(self):
        r = create_policy_retriever("graph")
        assert isinstance(r, GraphPolicyRetriever)

    def test_graph_retriever_finds_product_policy(self):
        r = GraphPolicyRetriever()
        result = r.retrieve("can I return headphones after 10 days?")
        assert "14-day" in result
        assert "Headphones" in result

    def test_graph_retriever_returns_empty_for_no_match(self):
        r = GraphPolicyRetriever()
        result = r.retrieve("how long does standard shipping take?")
        # "standard shipping" isn't a product or category, should be empty
        assert result == ""

    def test_hybrid_falls_back_to_vector(self):
        r = HybridRetriever()
        # This query has a product match → graph should return
        result = r.retrieve("can I return headphones after 10 days?")
        assert "14-day" in result
        assert "Headphones" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
