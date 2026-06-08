"""Shared test fixtures for the backend test suite."""

import pytest


@pytest.fixture(scope="session")
def neo4j_store():
    """Return a Neo4jStore connected to the test database.

    Skips tests if Neo4j is not available.
    """
    try:
        from backend.knowledge.neo4j_store import Neo4jStore

        store = Neo4jStore()
        # Quick health check
        store.resolve_product("iPhone 15 Pro")
        return store
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
