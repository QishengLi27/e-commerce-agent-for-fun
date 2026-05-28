"""
LangGraph checkpointer factory.

Provides a Postgres-backed checkpoint saver for per-session conversation
persistence, or an in-memory fallback for local development.

Usage:
    from backend.checkpoint import get_checkpointer
    checkpointer = get_checkpointer()          # sync (memory only)
    checkpointer = await aget_checkpointer()   # async (Postgres or memory)
"""

import logging

from backend.config import settings

logger = logging.getLogger(__name__)

_checkpointer = None


def get_checkpointer():
    """Return a previously-initialized checkpointer.

    Call aget_checkpointer() first to initialize. This sync getter exists
    for code paths that can't be async (e.g., graph invoked via run_in_executor).
    """
    global _checkpointer
    if _checkpointer is None:
        # Fall back to memory if not initialized yet (tests, scripts, CLI)
        _checkpointer = _create_memory_checkpointer()
    return _checkpointer


async def aget_checkpointer():
    """Async checkpointer factory. Call once during app startup.

    Creates an AsyncPostgresSaver (persistent) or InMemorySaver (dev),
    both of which support async operations required by astream_events().
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    if settings.checkpoint_type == "postgres":
        _checkpointer = await _create_postgres_checkpointer()
    else:
        _checkpointer = _create_memory_checkpointer()

    return _checkpointer


async def _create_postgres_checkpointer():
    """Create an AsyncPostgresSaver backed by an async connection pool."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    conninfo = settings.pg_connection_raw

    logger.info("[checkpoint] Creating async Postgres checkpointer")
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()
    saver = AsyncPostgresSaver(conn=pool)
    await saver.setup()
    logger.info("[checkpoint] Async Postgres checkpointer ready (tables created if needed)")
    return saver


def _create_memory_checkpointer():
    """Create an in-memory checkpointer (sessions lost on restart)."""
    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("[checkpoint] Using in-memory checkpointer (not persistent)")
    return InMemorySaver()
