"""
Neo4j-backed knowledge graph store.

Provides typed graph traversal queries via Cypher.
Singleton pattern: get_neo4j_store() returns the shared instance.

As of the multi-framework architecture, this replaces KnowledgeStore
for graph operations. The old graph_store.py is retained for backward
compatibility but new code uses Neo4jStore.
"""

from neo4j import GraphDatabase

from backend.config import settings
from backend.knowledge.models import (
    CategoryRef,
    PolicySummary,
    ProductAttribute,
    ProductInfo,
    ProductRef,
)

# Valid relationship types for _get_related — prevents Cypher injection
_VALID_REL_TYPES = frozenset(
    {
        "ACCESSORY_OF",
        "ALTERNATIVE_TO",
        "BUNDLED_WITH",
        "COMPATIBLE_WITH",
        "UPGRADE_OF",
        "RELATED_TO",
    }
)


class Neo4jStore:
    """Neo4j-backed knowledge graph with Cypher traversal.

    Supports:
      - Product lookup with synonym expansion
      - Category tree traversal (ancestors, descendants)
      - Attribute-filtered product search
      - Product relations (accessories, alternatives)
      - Policy inheritance via CHILD_OF* traversal
    """

    def __init__(
        self,
        uri: str = settings.neo4j_uri,
        user: str = settings.neo4j_user,
        password: str = settings.neo4j_password,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        self._driver.close()

    # ── Product Lookup ──────────────────────────────────────────────────────

    def resolve_product(self, query: str) -> ProductRef | None:
        """Resolve a term to a product via name, fulltext index, or synonym.

        Args:
            query: Product name or synonym (e.g., "苹果手机" → "iPhone 15 Pro").

        Returns:
            ProductRef if found, None otherwise.
        """
        with self._driver.session() as session:
            # 1. Exact name match first — prevents "iPhone 15 Pro" resolving to "iPhone 15"
            result = session.run(
                "MATCH (p:Product {name: $search_term}) RETURN p.name, p.price, p.sku",
                search_term=query,
            )
            record = result.single()
            if record:
                category = self._get_product_category(record["p.name"])
                return ProductRef(
                    name=record["p.name"],
                    price=record["p.price"],
                    sku=record["p.sku"],
                    category_name=category,
                )

            # 2. Fulltext search (covers name + search_terms + synonyms)
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('product_search', $search_term)
                YIELD node, score
                WHERE score > 0.5
                RETURN node.name AS name, node.price AS price,
                       node.sku AS sku, score
                ORDER BY score DESC
                LIMIT 1
                """,
                search_term=query,
            )
            record = result.single()
            if record:
                category = self._get_product_category(record["name"])
                return ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=category,
                )

            # 3. Fallback: substring match
            result = session.run(
                """
                MATCH (p:Product)
                WHERE p.name CONTAINS $search_term OR p.search_terms CONTAINS $search_term
                RETURN p.name, p.price, p.sku
                LIMIT 1
                """,
                search_term=query,
            )
            record = result.single()
            if record:
                category = self._get_product_category(record["p.name"])
                return ProductRef(
                    name=record["p.name"],
                    price=record["p.price"],
                    sku=record["p.sku"],
                    category_name=category,
                )

        return None

    def _get_product_category(self, product_name: str) -> str | None:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                RETURN c.name
                """,
                name=product_name,
            )
            record = result.single()
            return record["c.name"] if record else None

    # ── Product Detail ──────────────────────────────────────────────────────

    def get_product_info(self, product_name: str) -> ProductInfo | None:
        """Get full product detail: category path, attributes, policies, relations.

        Args:
            product_name: Exact or resolved product name.

        Returns:
            ProductInfo with category_path, attributes, policies, accessories,
            alternatives. None if product not found.
        """
        with self._driver.session() as session:
            # Verify product exists
            exists = session.run(
                "MATCH (p:Product {name: $name}) RETURN p.price, p.sku",
                name=product_name,
            ).single()
            if not exists:
                return None

        return ProductInfo(
            name=product_name,
            price=exists["p.price"],
            sku=exists["p.sku"],
            category_name=self._get_product_category(product_name),
            category_path=self._get_category_path(product_name),
            attributes=self._get_product_attributes(product_name),
            policies=self._get_product_policies(product_name),
            accessories=self.get_accessories(product_name),
            alternatives=self.get_alternatives(product_name),
        )

    def _get_category_path(self, product_name: str) -> list[str]:
        """Get full ancestor chain from product to root category."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                      -[:CHILD_OF*0..4]->(ancestor:Category)
                RETURN ancestor.name, ancestor.level
                ORDER BY ancestor.level
                """,
                name=product_name,
            )
            return [record["ancestor.name"] for record in result]

    def _get_product_attributes(self, product_name: str) -> list[ProductAttribute]:
        """Get all attributes for a product."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:HAS_ATTRIBUTE]->(av:AttributeValue)
                      -[:OF_TYPE]->(a:Attribute)
                RETURN a.name AS name, a.display_name AS display_name,
                       av.value AS value, a.data_type AS data_type
                """,
                name=product_name,
            )
            return [
                ProductAttribute(
                    name=record["name"],
                    display_name=record["display_name"],
                    value=record["value"],
                    data_type=record["data_type"],
                )
                for record in result
            ]

    def _get_product_policies(self, product_name: str) -> list[PolicySummary]:
        """Get all policies for a product via category inheritance (CHILD_OF*)."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product {name: $name})-[:IN_CATEGORY]->(c:Category)
                      -[:CHILD_OF*0..4]->(ancestor:Category)-[:HAS_POLICY]->(pol:Policy)
                RETURN DISTINCT pol.name AS name, pol.policy_type AS policy_type,
                       pol.summary AS summary, pol.details AS details
                """,
                name=product_name,
            )
            return [
                PolicySummary(
                    name=record["name"],
                    policy_type=record["policy_type"],
                    summary=record["summary"],
                    details=record["details"],
                )
                for record in result
            ]

    # ── Product Relations ───────────────────────────────────────────────────

    def get_accessories(self, product_name: str) -> list[ProductRef]:
        """Get accessories for a product."""
        return self._get_related(product_name, "ACCESSORY_OF")

    def get_alternatives(self, product_name: str) -> list[ProductRef]:
        """Get alternative products."""
        return self._get_related(product_name, "ALTERNATIVE_TO")

    def _get_related(self, product_name: str, rel_type: str) -> list[ProductRef]:
        """Get products related by a given relationship type.

        Checks both directions (inbound and outbound).
        """
        if rel_type not in _VALID_REL_TYPES:
            raise ValueError(f"Invalid relationship type: {rel_type}")

        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (p:Product {{name: $name}})-[:{rel_type}]->(related:Product)
                OPTIONAL MATCH (related)-[:IN_CATEGORY]->(cat:Category)
                RETURN related.name AS name, related.price AS price,
                       related.sku AS sku, cat.name AS category_name
                UNION
                MATCH (related:Product)-[:{rel_type}]->(p:Product {{name: $name}})
                OPTIONAL MATCH (related)-[:IN_CATEGORY]->(cat:Category)
                RETURN related.name AS name, related.price AS price,
                       related.sku AS sku, cat.name AS category_name
                """,
                name=product_name,
            )
            return [
                ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=record["category_name"],
                )
                for record in result
            ]

    # ── Filtered Product Search ─────────────────────────────────────────────

    def search_products(
        self,
        category: str | None = None,
        brand: str | None = None,
        max_price: float | None = None,
        attributes: dict[str, str] | None = None,
        limit: int = 10,
    ) -> list[ProductRef]:
        """Search products with optional filters.

        Args:
            category: Category name (includes sub-categories via CHILD_OF*).
            brand: Brand name.
            max_price: Maximum price filter.
            attributes: Dict of attribute_name → value to match.
            limit: Max results.

        Returns:
            List of matching ProductRef.
        """
        with self._driver.session() as session:
            # Build WHERE clauses
            wheres = []
            params: dict = {"limit": limit}

            if category:
                wheres.append("(c)-[:CHILD_OF*0..4]->(:Category {name: $category})")
                params["category"] = category

            if brand:
                wheres.append("(p)-[:HAS_BRAND]->(:Brand {name: $brand})")
                params["brand"] = brand

            if max_price is not None:
                wheres.append("p.price <= $max_price")
                params["max_price"] = max_price

            where_clause = " AND ".join(wheres) if wheres else "true"

            cypher = f"""
                MATCH (p:Product)-[:IN_CATEGORY]->(c:Category)
                WHERE {where_clause}
                OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
                RETURN DISTINCT p.name AS name, p.price AS price,
                       p.sku AS sku, cat.name AS category_name
                ORDER BY p.price DESC
                LIMIT $limit
            """

            result = session.run(cypher, **params)
            products = [
                ProductRef(
                    name=record["name"],
                    price=record["price"],
                    sku=record["sku"],
                    category_name=record["category_name"],
                )
                for record in result
            ]

            # Post-filter by attributes if specified (Cypher attribute filtering
            # is more complex due to the EAV model, so do it client-side)
            if attributes and products:
                filtered = []
                for product in products:
                    attrs = self._get_product_attributes(product.name)
                    attr_map = {a.name: a.value for a in attrs}
                    if all(attr_map.get(k) == v for k, v in attributes.items()):
                        filtered.append(product)
                return filtered

            return products

    # ── Category Tree ───────────────────────────────────────────────────────

    def get_category_tree(self, root_name: str | None = None) -> list[CategoryRef]:
        """Return the category hierarchy as a nested tree.

        Args:
            root_name: If given, return subtree rooted at this category.
                       If None, return all root categories.

        Returns:
            List of root CategoryRef with nested children.
        """
        with self._driver.session() as session:
            if root_name:
                result = session.run(
                    """
                    MATCH (root:Category {name: $name})
                    OPTIONAL MATCH (root)<-[:CHILD_OF]-(child:Category)
                    RETURN root.name AS name, root.description AS description,
                           root.level AS level, collect(child.name) AS children
                    """,
                    name=root_name,
                )
            else:
                result = session.run(
                    """
                    MATCH (root:Category)
                    WHERE NOT (root)-[:CHILD_OF]->(:Category)
                    OPTIONAL MATCH (root)<-[:CHILD_OF]-(child:Category)
                    RETURN root.name AS name, root.description AS description,
                           root.level AS level, collect(child.name) AS children
                    """
                )

            roots = []
            for record in result:
                cat = CategoryRef(
                    name=record["name"],
                    level=record["level"],
                    description=record["description"],
                )
                # Recursively build children
                if record["children"]:
                    for child_name in record["children"]:
                        child_tree = self.get_category_tree(child_name)
                        cat.children.extend(child_tree)
                roots.append(cat)

            return roots


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: Neo4jStore | None = None


def get_neo4j_store() -> Neo4jStore:
    """Return the shared Neo4jStore singleton."""
    global _store
    if _store is None:
        _store = Neo4jStore()
    return _store
