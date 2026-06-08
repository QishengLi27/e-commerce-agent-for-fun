"""Unit tests for Neo4jStore — requires Neo4j to be running with synced data."""

import pytest

from backend.knowledge.neo4j_store import Neo4jStore


@pytest.fixture
def store():
    """Return a Neo4jStore connected to the test database.

    Precondition: Neo4j is running and neo4j_setup has been run.
    """
    return Neo4jStore()


class TestResolveProduct:
    def test_exact_name(self, store):
        """Exact product name returns correct ProductRef."""
        result = store.resolve_product("iPhone 15 Pro")
        assert result is not None
        assert result.name == "iPhone 15 Pro"
        assert result.price == 999.00

    def test_partial_name(self, store):
        """Partial name via fulltext search."""
        result = store.resolve_product("MacBook")
        assert result is not None
        assert "MacBook" in result.name

    def test_synonym(self, store):
        """Synonym search returns canonical product."""
        result = store.resolve_product("苹果手机")
        assert result is not None
        assert "iPhone" in result.name

    def test_no_match(self, store):
        """Nonexistent product returns None."""
        result = store.resolve_product("nonexistent_xyz_product")
        assert result is None


class TestGetProductInfo:
    def test_has_category_path(self, store):
        """Product info includes full ancestor chain."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        assert "Smartphones" in info.category_path
        assert "Electronics" in info.category_path

    def test_has_attributes(self, store):
        """Product info includes attributes."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        attr_names = [a.name for a in info.attributes]
        assert "brand" in attr_names
        assert "storage" in attr_names

    def test_has_inherited_policies(self, store):
        """Product info includes policies inherited from ancestor categories."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        policy_names = [p.name for p in info.policies]
        assert "electronics_return" in policy_names

    def test_has_accessories(self, store):
        """Product info includes accessories."""
        info = store.get_product_info("iPhone 15 Pro")
        assert info is not None
        acc_names = [a.name for a in info.accessories]
        assert "iPhone 15 Pro Leather Case" in acc_names


class TestSearchProducts:
    def test_by_category(self, store):
        """Filter products by category (includes sub-categories)."""
        results = store.search_products(category="Smartphones")
        names = [p.name for p in results]
        assert "iPhone 15 Pro" in names
        assert "Google Pixel 8" in names

    def test_by_brand(self, store):
        """Filter products by brand."""
        results = store.search_products(brand="Sony")
        assert len(results) > 0
        assert all("Sony" in p.name for p in results)

    def test_by_max_price(self, store):
        """Filter products by max price."""
        results = store.search_products(category="Smartphones", max_price=700)
        assert all(p.price <= 700 for p in results)
