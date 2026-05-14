"""
Knowledge graph schema for PostgreSQL.

Defines the SQL DDL for categories, products, policy_rules,
and the junction table policy_category_rules.

Graph structure:
    products ──→ categories ──→ policy_category_rules ←── policy_rules

Run setup via: python -m backend.knowledge.schema
"""

import psycopg2
from backend.config import settings

CONNECTION_STRING = settings.pg_connection_raw

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS categories (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        description TEXT
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

SEED_CATEGORIES = [
    ("Electronics", "Electronic devices including computers, phones, and accessories"),
    ("Audio", "Audio equipment including headphones, speakers, and earbuds"),
    ("Accessories", "Device accessories including cases, chargers, and cables"),
    ("General", "General merchandise with standard store policies"),
]

SEED_PRODUCTS = [
    ("Headphones", "Audio", 79.99, "SKU-H001"),
    ("Laptop", "Electronics", 999.99, "SKU-L002"),
    ("Mouse", "Electronics", 29.99, "SKU-M003"),
    ("Keyboard", "Electronics", 89.99, "SKU-K004"),
    ("Phone Case", "Accessories", 19.99, "SKU-P005"),
    ("T-Shirt", "General", 24.99, "SKU-T006"),
]

SEED_POLICY_RULES = [
    (
        "standard_return",
        "return",
        "30-day return window for most items",
        "Our store offers a 30-day return window for most items. "
        "You can return products within 30 days of delivery for a full refund or exchange, "
        "provided the item is in its original condition with all packaging and accessories. "
        "Refunds are processed within 5-7 business days after we receive the returned item.",
    ),
    (
        "electronics_return",
        "return",
        "14-day return window for electronics and audio products",
        "Electronics and certain high-value items may have a 14-day return period. "
        "To initiate a return, please contact our support team or use the online return portal. "
        "Item must be in original condition with all packaging and accessories.",
    ),
    (
        "free_shipping",
        "shipping",
        "Free standard shipping on orders over $50",
        "We offer free standard shipping on orders over $50. "
        "Standard delivery typically takes 5-7 business days, while expedited options "
        "are available for an additional fee. You can track your order status using "
        "the tracking number provided in your confirmation email.",
    ),
    (
        "manufacturer_warranty",
        "warranty",
        "1-year manufacturer warranty on electronics",
        "All electronics come with a manufacturer's warranty that covers defects "
        "in materials and workmanship. The warranty period varies by product but is "
        "typically 1 year from the date of purchase. We do not offer extended warranties, "
        "but third-party options may be available.",
    ),
]

# Maps policy rule names → category names (by name, resolved to IDs during seeding)
POLICY_CATEGORY_MAPPING = {
    "standard_return": ["General", "Accessories"],
    "electronics_return": ["Electronics", "Audio"],
    "free_shipping": ["General", "Accessories", "Electronics", "Audio"],
    "manufacturer_warranty": ["Electronics", "Audio"],
}


def _get_connection():
    return psycopg2.connect(CONNECTION_STRING)


def setup_knowledge_db():
    """Create tables and insert seed data."""
    conn = _get_connection()
    cur = conn.cursor()

    # Create tables
    for ddl in DDL_STATEMENTS:
        cur.execute(ddl)

    # Seed categories
    cur.executemany(
        "INSERT INTO categories (name, description) VALUES (%s, %s) "
        "ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description",
        SEED_CATEGORIES,
    )

    # Seed products (need category IDs)
    cur.execute("SELECT id, name FROM categories")
    cat_map = {row[1]: row[0] for row in cur.fetchall()}

    for name, cat_name, price, sku in SEED_PRODUCTS:
        cur.execute(
            "INSERT INTO products (name, category_id, price, sku) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET category_id = EXCLUDED.category_id, "
            "price = EXCLUDED.price, sku = EXCLUDED.sku",
            (name, cat_map[cat_name], price, sku),
        )

    # Seed policy rules
    cur.executemany(
        "INSERT INTO policy_rules (name, policy_type, summary, details) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (name) DO UPDATE SET policy_type = EXCLUDED.policy_type, "
        "summary = EXCLUDED.summary, details = EXCLUDED.details",
        SEED_POLICY_RULES,
    )

    # Seed junction table
    cur.execute("SELECT id, name FROM policy_rules")
    policy_map = {row[1]: row[0] for row in cur.fetchall()}

    for policy_name, category_names in POLICY_CATEGORY_MAPPING.items():
        policy_id = policy_map[policy_name]
        for cat_name in category_names:
            cat_id = cat_map[cat_name]
            cur.execute(
                "INSERT INTO policy_category_rules (policy_rule_id, category_id) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (policy_id, cat_id),
            )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Knowledge DB setup complete: {len(SEED_CATEGORIES)} categories, "
          f"{len(SEED_PRODUCTS)} products, {len(SEED_POLICY_RULES)} policy rules")


if __name__ == "__main__":
    setup_knowledge_db()
