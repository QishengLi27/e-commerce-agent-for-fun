"""Integration tests for product_qa_tool.

Requires: Neo4j running with synced data, PostgreSQL running with
product_chunks ingested.
"""

from backend.tools.product_qa import product_qa_tool


class TestProductQATool:
    def test_single_product_feature_query(self):
        """Direct product feature question returns relevant answer."""
        result = product_qa_tool.invoke({"query": "Does iPhone 15 Pro have MagSafe?"})
        assert result is not None
        assert len(result) > 20
        # Should mention MagSafe or wireless charging
        assert "MagSafe" in result or "wireless" in result.lower()

    def test_category_query(self):
        """Category question returns category information."""
        result = product_qa_tool.invoke({"query": "What category is iPhone 15 Pro in?"})
        assert result is not None
        assert len(result) > 20
        assert "Smartphones" in result or "Flagship" in result

    def test_comparison_query(self):
        """Comparison question returns multi-product analysis."""
        result = product_qa_tool.invoke({"query": "Which phone is better, iPhone or Pixel?"})
        assert result is not None
        assert len(result) > 30

    def test_unknown_product_graceful(self):
        """Unknown product returns helpful message, not error."""
        result = product_qa_tool.invoke({"query": "Tell me about the FooBar X9000"})
        assert result is not None
        assert len(result) > 10
        # Should not be an error traceback
        assert "Traceback" not in result
