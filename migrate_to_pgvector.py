"""
One-time migration script: Chroma/SQLite → PostgreSQL + pgvector

Run this once after starting the pgvector Docker container:
    python migrate_to_pgvector.py

This will:
1. Enable the pgvector extension
2. Migrate policy chunks from store_policies.txt → pgvector
3. Migrate order data from SQLite → PostgreSQL
4. Clear and prepare the semantic cache collection in pgvector
"""

import os
import sqlite3

from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ─── Config ───────────────────────────────────────────────────────────────────

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/ecommerce"

EMBEDDINGS = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key="51bfecd9b55a448c927dd69288bfaeee.a2u6YiMOoo8S7WbU",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


def init_pgvector_extension():
    """Enable the pgvector extension in PostgreSQL."""
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    print("[migrate] pgvector extension enabled")


def migrate_policies():
    """Load store_policies.txt, chunk, embed, and store in pgvector."""
    print("[migrate] Loading store_policies.txt...")
    loader = TextLoader("store_policies.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[migrate] Created {len(chunks)} chunks")

    print("[migrate] Inserting into pgvector (store_policies collection)...")
    PGVector.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        collection_name="store_policies",
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    print(f"[migrate] ✅ Migrated {len(chunks)} policy chunks to pgvector")


def migrate_orders():
    """Copy orders from SQLite to PostgreSQL."""
    print("[migrate] Migrating orders from SQLite → PostgreSQL...")

    # Read from SQLite
    sqlite_conn = sqlite3.connect("ecommerce.db")
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("SELECT order_id, customer_name, status, estimated_delivery FROM orders")
    orders = sqlite_cursor.fetchall()
    sqlite_conn.close()

    if not orders:
        print("[migrate] No orders found in SQLite")
        return

    # Insert into PostgreSQL
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT,
                status TEXT,
                estimated_delivery TEXT
            )
        """))
        conn.execute(text("TRUNCATE TABLE orders"))
        conn.commit()

        for order in orders:
            conn.execute(
                text("INSERT INTO orders (order_id, customer_name, status, estimated_delivery) VALUES (:oid, :name, :status, :delivery)"),
                {"oid": order[0], "name": order[1], "status": order[2], "delivery": order[3]},
            )
        conn.commit()

    print(f"[migrate] ✅ Migrated {len(orders)} orders to PostgreSQL")


def prepare_semantic_cache():
    """Prepare an empty semantic cache collection in pgvector."""
    print("[migrate] Preparing semantic_cache collection in pgvector...")
    # PGVector will auto-create this on first use, but we clear any old data
    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=EMBEDDINGS,
        collection_name="semantic_cache",
    )
    # Try to delete old cache entries if any
    try:
        store.delete_collection()
    except Exception:
        pass
    print("[migrate] ✅ Semantic cache collection ready")


def main():
    print("=" * 60)
    print("Chroma/SQLite → PostgreSQL + pgvector Migration")
    print("=" * 60)

    init_pgvector_extension()
    migrate_policies()
    migrate_orders()
    prepare_semantic_cache()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print("Next steps:")
    print("  1. Update your .env or config to use PostgreSQL")
    print("  2. Test: python -c \"from agent import run_agent_with_cache; ...\"")


if __name__ == "__main__":
    main()
