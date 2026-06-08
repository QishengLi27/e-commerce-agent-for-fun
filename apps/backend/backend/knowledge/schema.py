"""
Knowledge graph schema for PostgreSQL with taxonomy support.

Supports:
- Hierarchical categories (up to 5 levels)
- Product attributes (brand, color, storage, etc.)
- Entity synonyms for search expansion
- Entity tags for disambiguation
- Product relations (accessories, alternatives)

Graph structure:
    products ──→ categories (hierarchical, recursive CTE)
    products ──→ product_attributes ──→ attribute_definitions
    entity_synonyms ──→ canonical entities
    entity_tags ──→ disambiguation
    product_relations ──→ related products

Run setup via: python -m backend.knowledge.schema
"""

import json
from pathlib import Path

import psycopg2

from backend.config import settings

CONNECTION_STRING = settings.pg_connection_raw

# ─── Seed data paths ─────────────────────────────────────────────────────────

_SEED_DIR = Path(__file__).parent.parent.parent / "data" / "seed"


def _load_seed(filename: str):
    """Load a JSON seed file from data/seed/."""
    path = _SEED_DIR / filename
    with open(path) as f:
        return json.load(f)


# Lazy-loaded seed data (loaded on first access to keep module import fast)
_SEED_DATA: dict | None = None


def _get_seed() -> dict:
    """Return all seed data, loading from JSON files on first call."""
    global _SEED_DATA
    if _SEED_DATA is None:
        _SEED_DATA = {
            "categories": _load_seed("categories.json"),
            "products": _load_seed("products.json"),
            "attributes": _load_seed("attributes.json"),
            "relations": _load_seed("relations.json"),
            "synonyms": _load_seed("synonyms.json"),
            "tags": _load_seed("tags.json"),
            "policies": _load_seed("policies.json"),
        }
    return _SEED_DATA


# ─── DDL: Core tables (existing) ─────────────────────────────────────────────

DDL_CORE = [
    """
    CREATE TABLE IF NOT EXISTS categories (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        parent_id INTEGER REFERENCES categories(id),
        level INTEGER DEFAULT 0,
        path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS products (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        category_id INTEGER REFERENCES categories(id),
        price DECIMAL(10, 2),
        sku TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policy_rules (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        policy_type TEXT NOT NULL CHECK (policy_type IN ('return', 'shipping', 'warranty')),
        summary TEXT NOT NULL,
        details TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policy_category_rules (
        policy_rule_id INTEGER REFERENCES policy_rules(id),
        category_id INTEGER REFERENCES categories(id),
        PRIMARY KEY (policy_rule_id, category_id)
    )
    """,
]

# ─── DDL: Taxonomy extension tables ──────────────────────────────────────────

DDL_TAXONOMY = [
    """
    CREATE TABLE IF NOT EXISTS attribute_definitions (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        data_type TEXT NOT NULL CHECK (data_type IN ('text', 'number', 'boolean', 'enum', 'date')),
        is_filterable BOOLEAN DEFAULT true,
        is_facetable BOOLEAN DEFAULT true,
        allowed_values JSONB,
        unit TEXT,
        priority INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS product_attributes (
        id SERIAL PRIMARY KEY,
        product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
        attribute_id INTEGER REFERENCES attribute_definitions(id) ON DELETE CASCADE,
        value_text TEXT,
        value_number DECIMAL(15, 4),
        value_boolean BOOLEAN,
        UNIQUE(product_id, attribute_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_synonyms (
        id SERIAL PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        synonym TEXT NOT NULL,
        entity_type TEXT NOT NULL CHECK (entity_type IN ('brand', 'product', 'category', 'attribute_value')),
        language TEXT DEFAULT 'en',
        confidence FLOAT DEFAULT 1.0,
        UNIQUE(synonym, entity_type, language)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_tags (
        id SERIAL PRIMARY KEY,
        entity_name TEXT NOT NULL,
        entity_type TEXT NOT NULL CHECK (entity_type IN ('brand', 'product', 'category', 'attribute', 'generic')),
        domain TEXT,
        disambiguation_hint TEXT,
        UNIQUE(entity_name, entity_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS product_relations (
        id SERIAL PRIMARY KEY,
        source_product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
        target_product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL CHECK (relation_type IN ('accessory', 'alternative', 'bundle', 'compatible', 'upgrade')),
        strength FLOAT DEFAULT 1.0,
        UNIQUE(source_product_id, target_product_id, relation_type)
    )
    """,
]


# ─── Setup functions ─────────────────────────────────────────────────────────


def _get_connection():
    return psycopg2.connect(CONNECTION_STRING)


def _ensure_columns(conn):
    """Add taxonomy columns to existing tables (idempotent)."""
    cur = conn.cursor()
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='categories' AND column_name='parent_id') THEN
                ALTER TABLE categories ADD COLUMN parent_id INTEGER REFERENCES categories(id);
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='categories' AND column_name='level') THEN
                ALTER TABLE categories ADD COLUMN level INTEGER DEFAULT 0;
            END IF;
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_name='categories' AND column_name='path') THEN
                ALTER TABLE categories ADD COLUMN path TEXT;
            END IF;
        END $$;
    """)
    conn.commit()
    cur.close()


def _seed_categories(cur):
    """Seed hierarchical categories and compute paths."""
    seed = _get_seed()["categories"]
    for cat in seed:
        cur.execute(
            "INSERT INTO categories (name, description, parent_id, level) "
            "VALUES (%s, %s, NULL, %s) "
            "ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description, "
            "level = EXCLUDED.level, parent_id = NULL",
            (cat["name"], cat["description"], cat["level"]),
        )

    # Resolve parent_ids
    cur.execute("SELECT id, name FROM categories")
    cat_map = {row[1]: row[0] for row in cur.fetchall()}

    for cat in seed:
        if cat.get("parent_name"):
            parent_id = cat_map.get(cat["parent_name"])
            if parent_id:
                cur.execute(
                    "UPDATE categories SET parent_id = %s WHERE name = %s",
                    (parent_id, cat["name"]),
                )

    # Compute materialized paths using recursive CTE
    cur.execute("""
        WITH RECURSIVE path_cte AS (
            SELECT id, name, parent_id, level, name::text AS computed_path
            FROM categories WHERE parent_id IS NULL
            UNION ALL
            SELECT c.id, c.name, c.parent_id, c.level,
                   p.computed_path || '.' || c.id::text
            FROM categories c
            JOIN path_cte p ON c.parent_id = p.id
        )
        UPDATE categories SET path = path_cte.computed_path
        FROM path_cte WHERE categories.id = path_cte.id;
    """)

    return cat_map


def _seed_products(cur, cat_map):
    """Seed products with category mapping."""
    seed = _get_seed()["products"]
    for product in seed:
        cat_id = cat_map.get(product["category_name"])
        if cat_id:
            cur.execute(
                "INSERT INTO products (name, category_id, price, sku) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (name) DO UPDATE SET category_id = EXCLUDED.category_id, "
                "price = EXCLUDED.price, sku = EXCLUDED.sku",
                (product["name"], cat_id, product["price"], product["sku"]),
            )

    cur.execute("SELECT id, name FROM products")
    return {row[1]: row[0] for row in cur.fetchall()}


def _seed_attribute_definitions(cur):
    """Seed attribute definitions."""
    seed = _get_seed()["attributes"]
    for attr in seed["definitions"]:
        cur.execute(
            "INSERT INTO attribute_definitions (name, display_name, data_type, is_filterable, is_facetable, allowed_values, unit, priority) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET display_name = EXCLUDED.display_name, "
            "data_type = EXCLUDED.data_type, is_filterable = EXCLUDED.is_filterable, "
            "is_facetable = EXCLUDED.is_facetable, allowed_values = EXCLUDED.allowed_values, "
            "unit = EXCLUDED.unit, priority = EXCLUDED.priority",
            (
                attr["name"],
                attr["display_name"],
                attr["data_type"],
                attr["is_filterable"],
                attr["is_facetable"],
                attr["allowed_values"],
                attr["unit"],
                attr["priority"],
            ),
        )

    cur.execute("SELECT id, name FROM attribute_definitions")
    return {row[1]: row[0] for row in cur.fetchall()}


def _seed_product_attributes(cur, product_map, attr_map):
    """Seed product-attribute values."""
    seed = _get_seed()["attributes"]
    for pa in seed["product_attributes"]:
        product_id = product_map.get(pa["product_name"])
        attr_id = attr_map.get(pa["attribute_name"])
        if product_id and attr_id:
            cur.execute(
                "INSERT INTO product_attributes (product_id, attribute_id, value_text, value_number, value_boolean) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (product_id, attribute_id) DO UPDATE SET "
                "value_text = EXCLUDED.value_text, value_number = EXCLUDED.value_number, "
                "value_boolean = EXCLUDED.value_boolean",
                (product_id, attr_id, pa["value_text"], pa["value_number"], pa["value_boolean"]),
            )


def _seed_synonyms(cur):
    """Seed entity synonyms."""
    seed = _get_seed()["synonyms"]
    for syn in seed:
        cur.execute(
            "INSERT INTO entity_synonyms (canonical_name, synonym, entity_type, language, confidence) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (synonym, entity_type, language) DO UPDATE SET "
            "canonical_name = EXCLUDED.canonical_name, confidence = EXCLUDED.confidence",
            (
                syn["canonical_name"],
                syn["synonym"],
                syn["entity_type"],
                syn["language"],
                syn["confidence"],
            ),
        )


def _seed_entity_tags(cur):
    """Seed entity disambiguation tags."""
    seed = _get_seed()["tags"]
    for tag in seed:
        cur.execute(
            "INSERT INTO entity_tags (entity_name, entity_type, domain, disambiguation_hint) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (entity_name, entity_type) DO UPDATE SET "
            "domain = EXCLUDED.domain, disambiguation_hint = EXCLUDED.disambiguation_hint",
            (tag["entity_name"], tag["entity_type"], tag["domain"], tag["disambiguation_hint"]),
        )


def _seed_product_relations(cur, product_map):
    """Seed product relations."""
    seed = _get_seed()["relations"]
    for rel in seed:
        source_id = product_map.get(rel["source_product_name"])
        target_id = product_map.get(rel["target_product_name"])
        if source_id and target_id:
            cur.execute(
                "INSERT INTO product_relations (source_product_id, target_product_id, relation_type, strength) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (source_product_id, target_product_id, relation_type) DO UPDATE SET "
                "strength = EXCLUDED.strength",
                (source_id, target_id, rel["relation_type"], rel["strength"]),
            )


def _seed_policies(cur, cat_map):
    """Seed policy rules and category mappings."""
    seed = _get_seed()["policies"]
    for rule in seed["rules"]:
        cur.execute(
            "INSERT INTO policy_rules (name, policy_type, summary, details) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET policy_type = EXCLUDED.policy_type, "
            "summary = EXCLUDED.summary, details = EXCLUDED.details",
            (rule["name"], rule["policy_type"], rule["summary"], rule["details"]),
        )

    cur.execute("SELECT id, name FROM policy_rules")
    policy_map = {row[1]: row[0] for row in cur.fetchall()}

    for policy_name, category_names in seed["category_mapping"].items():
        policy_id = policy_map.get(policy_name)
        if not policy_id:
            continue
        for cat_name in category_names:
            cat_id = cat_map.get(cat_name)
            if cat_id:
                cur.execute(
                    "INSERT INTO policy_category_rules (policy_rule_id, category_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (policy_id, cat_id),
                )

    return policy_map


def setup_knowledge_db():
    """Create tables, add taxonomy columns, and insert all seed data."""
    conn = _get_connection()
    cur = conn.cursor()

    print("[taxonomy] Creating core tables...")
    for ddl in DDL_CORE:
        cur.execute(ddl)

    print("[taxonomy] Creating taxonomy tables...")
    for ddl in DDL_TAXONOMY:
        cur.execute(ddl)

    conn.commit()

    print("[taxonomy] Ensuring backward-compatible columns...")
    _ensure_columns(conn)

    print("[taxonomy] Seeding categories...")
    cat_map = _seed_categories(cur)

    print("[taxonomy] Seeding products...")
    product_map = _seed_products(cur, cat_map)

    print("[taxonomy] Seeding attributes...")
    attr_map = _seed_attribute_definitions(cur)
    _seed_product_attributes(cur, product_map, attr_map)

    print("[taxonomy] Seeding synonyms...")
    _seed_synonyms(cur)

    print("[taxonomy] Seeding entity tags...")
    _seed_entity_tags(cur)

    print("[taxonomy] Seeding product relations...")
    _seed_product_relations(cur, product_map)

    print("[taxonomy] Seeding policy rules...")
    _seed_policies(cur, cat_map)

    conn.commit()
    cur.close()
    conn.close()

    seed = _get_seed()
    print(
        f"[taxonomy] Setup complete: {len(seed['categories'])} categories, "
        f"{len(seed['products'])} products, {len(seed['attributes']['definitions'])} attribute definitions, "
        f"{len(seed['synonyms'])} synonyms, {len(seed['tags'])} entity tags, "
        f"{len(seed['relations'])} product relations, "
        f"{len(seed['policies']['rules'])} policy rules"
    )


if __name__ == "__main__":
    setup_knowledge_db()
