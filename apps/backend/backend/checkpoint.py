"""
LangGraph checkpointer factory.

Provides a Postgres-backed checkpoint saver for per-session conversation
persistence, or an in-memory fallback for local development.

Usage:
    from backend.checkpoint import get_checkpointer
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
"""

import logging

from backend.config import settings

logger = logging.getLogger(__name__)

# Lazy singleton — created on first call so FastAPI import graphs don't
# trigger a DB connection at module-load time.
_checkpointer = None


def get_checkpointer():
    """Return a compiled checkpointer instance (Postgres or in-memory)."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    if settings.checkpoint_type == "postgres":
        _checkpointer = _create_postgres_checkpointer()
    else:
        _checkpointer = _create_memory_checkpointer()

    return _checkpointer


def _create_postgres_checkpointer():
    """Create a PostgresSaver backed by a connection pool."""
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    # psycopg v3 expects postgresql:// (not postgresql+psycopg2://)
    conninfo = settings.pg_connection_raw

    logger.info("[checkpoint] Creating Postgres connection pool for checkpointer")
    pool = ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": None, "autocommit": True},
    )
    pool.open(wait=True)

    saver = PostgresSaver(conn=pool)
    saver.setup()
    logger.info("[checkpoint] Postgres checkpointer ready (tables created if needed)")
    return saver


def _create_memory_checkpointer():
    """Create an in-memory checkpointer (sessions lost on restart)."""
    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("[checkpoint] Using in-memory checkpointer (not persistent)")
    return InMemorySaver()
