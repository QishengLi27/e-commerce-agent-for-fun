"""
PostgreSQL-backed knowledge graph store.

Provides graph traversal via SQL JOINs — no graph database needed.
Singleton pattern: get_knowledge_store() returns the shared instance.
"""

import psycopg2

from backend.config import settings

_CONNECTION_STRING = settings.pg_connection_raw


class KnowledgeStore:
    """PostgreSQL-backed knowledge graph with 3-hop traversal capability.

    Graph structure:
        products → categories → policy_category_rules ← policy_rules

    Queries use JOINs to traverse relationships — equivalent to graph edges.
    """

    def __init__(self, connection_string: str = _CONNECTION_STRING):
        self._conn_string = connection_string

    def _get_conn(self):
        return psycopg2.connect(self._conn_string)

    # ── Product queries ──────────────────────────────────────────────────────

    def query_product_policies(self, query: str) -> list[dict]:
        """3-hop traversal: product → category → policy_rules.

        Accepts either a product name or a full natural language query.
        Extracts product name via keyword matching, then traverses the graph.

        Returns list of policy dicts with keys:
            name, summary, policy_type, details, product_name, category_name
        """
        # Accept either a product name directly or extract from a query
        if self._product_names is None:
            self._load_names()
        product_name = query if query.lower() in (self._product_names or []) else self._extract_product_name(query)
        if not product_name:
            return []

        conn = self._get_conn()
        cur = conn.cursor()
        sql = """
            SELECT pr.name, pr.summary, pr.policy_type, pr.details,
                   p.name AS product_name, c.name AS category_name
            FROM products p
            JOIN categories c ON p.category_id = c.id
            JOIN policy_category_rules pcr ON c.id = pcr.category_id
            JOIN policy_rules pr ON pcr.policy_rule_id = pr.id
            WHERE p.name ILIKE %s
            ORDER BY pr.policy_type, pr.name
        """
        cur.execute(sql, (f"%{product_name}%",))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "name": r[0],
                "summary": r[1],
                "policy_type": r[2],
                "details": r[3],
                "product_name": r[4],
                "category_name": r[5],
            }
            for r in rows
        ]

    def query_category_policies(self, query: str) -> list[dict]:
        """1-hop traversal: category → policy_rules.

        Accepts either a category name or a full natural language query.
        Extracts category name, then finds applicable policies.

        Returns list of policy dicts with keys:
            name, summary, policy_type, details, category_name
        """
        if self._category_names is None:
            self._load_names()
        category_name = query if query.lower() in (self._category_names or []) else self._extract_category_name(query)
        if not category_name:
            return []

        conn = self._get_conn()
        cur = conn.cursor()
        sql = """
            SELECT pr.name, pr.summary, pr.policy_type, pr.details, c.name AS category_name
            FROM categories c
            JOIN policy_category_rules pcr ON c.id = pcr.category_id
            JOIN policy_rules pr ON pcr.policy_rule_id = pr.id
            WHERE c.name ILIKE %s
            ORDER BY pr.policy_type, pr.name
        """
        cur.execute(sql, (f"%{category_name}%",))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "name": r[0],
                "summary": r[1],
                "policy_type": r[2],
                "details": r[3],
                "category_name": r[4],
            }
            for r in rows
        ]

    def search_products(self, keyword: str) -> list[dict]:
        """Search products by name or category."""
        conn = self._get_conn()
        cur = conn.cursor()
        sql = """
            SELECT p.name, p.price, p.sku, c.name AS category_name
            FROM products p
            JOIN categories c ON p.category_id = c.id
            WHERE p.name ILIKE %s OR c.name ILIKE %s
            ORDER BY p.name
        """
        cur.execute(sql, (f"%{keyword}%", f"%{keyword}%"))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {"name": r[0], "price": float(r[1]), "sku": r[2], "category_name": r[3]}
            for r in rows
        ]

    def get_product_info(self, query: str) -> dict | None:
        """Get a single product with category and all applicable policies.

        Accepts either a product name or a full natural language query.
        """
        # Extract product name from query first
        if self._product_names is None:
            self._load_names()
        product_name = query if query.lower() in (self._product_names or []) else self._extract_product_name(query)
        if not product_name:
            return None

        conn = self._get_conn()
        cur = conn.cursor()
        sql = """
            SELECT p.name, p.price, p.sku, c.name AS category_name, c.description
            FROM products p
            JOIN categories c ON p.category_id = c.id
            WHERE p.name ILIKE %s
        """
        cur.execute(sql, (f"%{product_name}%",))
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None

        product = {
            "name": row[0],
            "price": float(row[1]) if row[1] else None,
            "sku": row[2],
            "category_name": row[3],
            "category_description": row[4],
        }

        # Also fetch applicable policies
        policies = self.query_product_policies(product_name)
        product["policies"] = [
            {"name": p["name"], "summary": p["summary"], "type": p["policy_type"]}
            for p in policies
        ]

        cur.close()
        conn.close()
        return product

    def get_all_categories(self) -> list[dict]:
        """List all categories."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT name, description FROM categories ORDER BY name")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"name": r[0], "description": r[1]} for r in rows]

    # ── Entity extraction ─────────────────────────────────────────────────────

    # Known product names for keyword matching (loaded lazily from DB)
    _product_names: list[str] | None = None
    _category_names: list[str] | None = None

    def _load_names(self):
        if self._product_names is None:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT name FROM products")
            self._product_names = [r[0].lower() for r in cur.fetchall()]
            cur.execute("SELECT name FROM categories")
            self._category_names = [r[0].lower() for r in cur.fetchall()]
            cur.close()
            conn.close()

    def _extract_product_name(self, query: str) -> str | None:
        """Extract product name from query via keyword matching against DB."""
        self._load_names()
        assert self._product_names is not None
        query_lower = query.lower()
        # Longest match first to avoid "phone" matching "phone case"
        for name in sorted(self._product_names, key=len, reverse=True):
            if name in query_lower:
                return name
        return None

    def _extract_category_name(self, query: str) -> str | None:
        """Extract category name from query via keyword matching."""
        self._load_names()
        assert self._category_names is not None
        query_lower = query.lower()
        for name in sorted(self._category_names, key=len, reverse=True):
            if name in query_lower:
                return name
        return None


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
