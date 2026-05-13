"""
One-time migration script: Chroma/SQLite -> PostgreSQL + pgvector

Run this once after starting the pgvector Docker container:
    python -m backend.db.migrate_pgvector

This will:
1. Enable the pgvector extension
2. Migrate policy chunks from store_policies.txt -> pgvector
3. Migrate order data from SQLite -> PostgreSQL
4. Clear and prepare the semantic cache collection in pgvector
"""

import os
import sqlite3

from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from backend.config import settings

# -- Config -------------------------------------------------------------------

CONNECTION_STRING = settings.database_url

EMBEDDINGS = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
    openai_api_base=settings.openai_api_base,
)


def init_pgvector_extension():
    """Enable the pgvector extension in PostgreSQL."""
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    print("[migrate] pgvector extension enabled")


def migrate_policies_to_pgvector():
    """Load store_policies.txt, chunk, and store in pgvector."""
    loader = TextLoader("data/store_policies.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    PGVector.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        collection_name="store_policies",
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
        distance_strategy="cosine",
    )
    print(f"[migrate] Migrated {len(chunks)} policy chunks to pgvector")


def migrate_sqlite_to_postgres():
    """Copy order data from SQLite to PostgreSQL."""
    sqlite_conn = sqlite3.connect("data/ecommerce.db")
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("SELECT order_id, customer_name, status, estimated_delivery FROM orders")
    rows = sqlite_cursor.fetchall()
    sqlite_conn.close()

    pg_engine = create_engine(CONNECTION_STRING)
    with pg_engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT,
                status TEXT,
                estimated_delivery TEXT
            )
        """))
        for row in rows:
            conn.execute(text("""
                INSERT INTO orders (order_id, customer_name, status, estimated_delivery)
                VALUES (:oid, :name, :status, :delivery)
                ON CONFLICT (order_id) DO UPDATE SET
                    customer_name = EXCLUDED.customer_name,
                    status = EXCLUDED.status,
                    estimated_delivery = EXCLUDED.estimated_delivery
            """), {"oid": row[0], "name": row[1], "status": row[2], "delivery": row[3]})
        conn.commit()
    print(f"[migrate] Migrated {len(rows)} orders from SQLite to PostgreSQL")


def init_semantic_cache():
    """Prepare the semantic cache collection in pgvector."""
    PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=EMBEDDINGS,
        collection_name="semantic_cache",
        distance_strategy="cosine",
    )
    print("[migrate] Semantic cache collection ready")


if __name__ == "__main__":
    init_pgvector_extension()
    migrate_policies_to_pgvector()
    migrate_sqlite_to_postgres()
    init_semantic_cache()
    print("[migrate] All migrations complete!")
