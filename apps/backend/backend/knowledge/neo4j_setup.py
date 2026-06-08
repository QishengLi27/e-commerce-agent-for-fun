"""
Sync PostgreSQL seed data to Neo4j graph database.

Run once after schema setup:
    python -m backend.knowledge.neo4j_setup

Reads all taxonomy data from PostgreSQL (source of truth) and recreates
the Neo4j graph (read-optimized cache). Idempotent — wipes existing
graph first.
"""

import psycopg2
from neo4j import GraphDatabase

from backend.config import settings

PG_CONN = settings.pg_connection_raw
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password

BATCH_SIZE = 500


def _get_pg_conn():
    return psycopg2.connect(PG_CONN)


def setup_neo4j():
    """Main entry point: sync all PG data to Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        _clear_graph(session)
        _create_indexes(session)
        _sync_categories(session)
        _sync_products(session)
        _sync_brands(session)
        _sync_attributes(session)
        _sync_product_relations(session)
        _sync_policies(session)
        _sync_synonyms(session)

    driver.close()
    print("[neo4j] Graph sync complete.")


def _clear_graph(session):
    """Remove all nodes and relationships (idempotent)."""
    session.run("MATCH (n) DETACH DELETE n")
    print("[neo4j] Cleared existing graph.")


def _create_indexes(session):
    """Create constraints and indexes for fast lookups."""
    session.run(
        "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
    )
    session.run("CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
    session.run(
        "CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (pol:Policy) REQUIRE pol.name IS UNIQUE"
    )
    session.run(
        "CREATE FULLTEXT INDEX product_search IF NOT EXISTS "
        "FOR (p:Product) ON EACH [p.name, p.search_terms]"
    )
    session.run("CREATE INDEX product_price IF NOT EXISTS FOR (p:Product) ON (p.price)")
    print("[neo4j] Indexes and constraints created.")


def _sync_categories(session):
    """Sync hierarchical categories from PG."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description, parent_id, level FROM categories ORDER BY level")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    cat_data = {
        row[0]: {
            "name": row[1],
            "description": row[2],
            "parent_id": row[3],
            "level": row[4],
        }
        for row in rows
    }

    # Batch create all category nodes
    batch = [
        {
            "name": data["name"],
            "description": data["description"] or "",
            "level": data["level"],
        }
        for data in cat_data.values()
    ]
    session.run(
        """
        UNWIND $batch AS row
        CREATE (c:Category {
            name: row.name,
            description: row.description,
            level: row.level
        })
        """,
        batch=batch,
    )

    # Batch create CHILD_OF relationships
    rels = [
        {
            "child_name": cat_data[cat_id]["name"],
            "parent_name": cat_data[cat_data[cat_id]["parent_id"]]["name"],
        }
        for cat_id, data in cat_data.items()
        if data["parent_id"] and data["parent_id"] in cat_data
    ]
    if rels:
        session.run(
            """
            UNWIND $rels AS row
            MATCH (child:Category {name: row.child_name})
            MATCH (parent:Category {name: row.parent_name})
            CREATE (child)-[:CHILD_OF]->(parent)
            """,
            rels=rels,
        )

    print(f"[neo4j] Synced {len(rows)} categories.")


def _sync_products(session):
    """Sync products with category relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.name, p.price, p.sku, c.name AS category_name
        FROM products p
        JOIN categories c ON p.category_id = c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    batch = [
        {
            "name": name,
            "price": float(price) if price else 0.0,
            "sku": sku or "",
            "category_name": category_name,
        }
        for name, price, sku, category_name in rows
    ]

    session.run(
        """
        UNWIND $batch AS row
        MATCH (c:Category {name: row.category_name})
        CREATE (p:Product {
            name: row.name,
            price: row.price,
            sku: row.sku,
            search_terms: row.name
        })
        CREATE (p)-[:IN_CATEGORY]->(c)
        """,
        batch=batch,
    )

    print(f"[neo4j] Synced {len(rows)} products.")


def _sync_brands(session):
    """Extract brands from product_attributes and create (:Brand) nodes."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT pa.value_text
        FROM product_attributes pa
        JOIN attribute_definitions ad ON pa.attribute_id = ad.id
        WHERE ad.name = 'brand' AND pa.value_text IS NOT NULL
        """
    )
    brands = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    if brands:
        session.run(
            "UNWIND $brands AS name CREATE (:Brand {name: name})",
            brands=brands,
        )

    # Link products to brands via product_attributes (exact match)
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.name, pa.value_text
        FROM product_attributes pa
        JOIN attribute_definitions ad ON pa.attribute_id = ad.id
        JOIN products p ON pa.product_id = p.id
        WHERE ad.name = 'brand' AND pa.value_text IS NOT NULL
        """
    )
    brand_links = [{"product_name": row[0], "brand_name": row[1]} for row in cur.fetchall()]
    cur.close()
    conn.close()

    for chunk in _chunks(brand_links, BATCH_SIZE):
        session.run(
            """
            UNWIND $chunk AS row
            MATCH (p:Product {name: row.product_name})
            MATCH (b:Brand {name: row.brand_name})
            CREATE (p)-[:HAS_BRAND]->(b)
            """,
            chunk=chunk,
        )

    print(f"[neo4j] Synced {len(brands)} brands.")


def _sync_attributes(session):
    """Create attribute values as nodes with HAS_ATTRIBUTE relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.name, ad.name AS attr_name, ad.display_name, ad.data_type,
               pa.value_text, pa.value_number, pa.value_boolean
        FROM product_attributes pa
        JOIN products p ON pa.product_id = p.id
        JOIN attribute_definitions ad ON pa.attribute_id = ad.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    batch = []
    for product_name, attr_name, display_name, data_type, v_text, v_num, v_bool in rows:
        if data_type == "text":
            value = str(v_text) if v_text else ""
        elif data_type == "number":
            value = str(v_num) if v_num is not None else "0"
        elif data_type == "boolean":
            value = str(v_bool) if v_bool is not None else "false"
        else:
            value = str(v_text or "")

        batch.append(
            {
                "product_name": product_name,
                "attr_name": attr_name,
                "display_name": display_name,
                "data_type": data_type,
                "value": value,
            }
        )

    for chunk in _chunks(batch, BATCH_SIZE):
        session.run(
            """
            UNWIND $chunk AS row
            MATCH (p:Product {name: row.product_name})
            MERGE (a:Attribute {name: row.attr_name})
              ON CREATE SET a.display_name = row.display_name,
                            a.data_type = row.data_type
            CREATE (av:AttributeValue {value: row.value})-[:OF_TYPE]->(a)
            CREATE (p)-[:HAS_ATTRIBUTE]->(av)
            """,
            chunk=chunk,
        )

    print(f"[neo4j] Synced {len(batch)} product attributes.")


def _sync_product_relations(session):
    """Create ACCESSORY_OF, ALTERNATIVE_TO, COMPATIBLE_WITH relationships."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sp.name, tp.name, pr.relation_type, pr.strength
        FROM product_relations pr
        JOIN products sp ON pr.source_product_id = sp.id
        JOIN products tp ON pr.target_product_id = tp.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    rel_type_map = {
        "accessory": "ACCESSORY_OF",
        "alternative": "ALTERNATIVE_TO",
        "bundle": "BUNDLED_WITH",
        "compatible": "COMPATIBLE_WITH",
        "upgrade": "UPGRADE_OF",
    }

    batch = []
    for source_name, target_name, rel_type, strength in rows:
        batch.append(
            {
                "source_name": source_name,
                "target_name": target_name,
                "rel_type": rel_type_map.get(rel_type, "RELATED_TO"),
                "strength": float(strength),
            }
        )

    for chunk in _chunks(batch, BATCH_SIZE):
        session.run(
            """
            UNWIND $chunk AS row
            MATCH (source:Product {name: row.source_name})
            MATCH (target:Product {name: row.target_name})
            CALL apoc.create.relationship(source, row.rel_type, {strength: row.strength}, target)
            YIELD rel
            RETURN count(rel)
            """,
            chunk=chunk,
        )

    print(f"[neo4j] Synced {len(rows)} product relations.")


def _sync_policies(session):
    """Create (:Policy) nodes with HAS_POLICY relationships to categories."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pr.name, pr.policy_type, pr.summary, pr.details, c.name AS category_name
        FROM policy_rules pr
        JOIN policy_category_rules pcr ON pr.id = pcr.policy_rule_id
        JOIN categories c ON pcr.category_id = c.id
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    batch = [
        {
            "name": name,
            "policy_type": policy_type,
            "summary": summary,
            "details": details,
            "category_name": category_name,
        }
        for name, policy_type, summary, details, category_name in rows
    ]

    for chunk in _chunks(batch, BATCH_SIZE):
        session.run(
            """
            UNWIND $chunk AS row
            MERGE (pol:Policy {name: row.name})
            SET pol.policy_type = row.policy_type,
                pol.summary = row.summary,
                pol.details = row.details
            WITH pol, row
            MATCH (c:Category {name: row.category_name})
            MERGE (c)-[:HAS_POLICY]->(pol)
            """,
            chunk=chunk,
        )

    print(f"[neo4j] Synced {len(rows)} policy assignments.")


def _sync_synonyms(session):
    """Attach search_terms to products from entity_synonyms table."""
    conn = _get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT canonical_name, synonym
        FROM entity_synonyms
        WHERE entity_type IN ('product', 'brand')
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    batch = [{"canonical": canonical_name, "synonym": synonym} for canonical_name, synonym in rows]

    count = 0
    for chunk in _chunks(batch, BATCH_SIZE):
        result = session.run(
            """
            UNWIND $chunk AS row
            MATCH (p:Product)
            WHERE p.name CONTAINS row.canonical OR p.name = row.canonical
               OR (p)-[:HAS_BRAND]->(:Brand {name: row.canonical})
            SET p.search_terms = COALESCE(p.search_terms, p.name) + ', ' + row.synonym
            RETURN count(DISTINCT p) AS updated
            """,
            chunk=chunk,
        )
        record = result.single()
        if record:
            count += record["updated"]

    print(f"[neo4j] Attached synonyms to {count} products.")


def _chunks(data: list, size: int):
    """Yield successive chunks of a given size."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


if __name__ == "__main__":
    setup_neo4j()
