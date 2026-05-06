# Database Migrations & Schema (Not Yet Implemented)

## Overview

This directory will manage the relational database schema evolution, seed data, and migration scripts as the application grows beyond the current prototype.

## Current State

- **PostgreSQL 16** with `pgvector` extension
- Tables created imperatively in `apps/backend/backend/db/setup.py`
- Vector collections managed by LangChain's `PGVector`

## Planned Migration Tooling

| Option | Pros | Cons |
|--------|------|------|
| **Alembic** (SQLAlchemy) | Native SQLAlchemy integration, mature | Python-only |
| **Atlas** | Multi-engine, declarative HCL | Newer, smaller community |
| **Flyway** | Battle-tested, language-agnostic | More verbose |

**Recommendation**: Alembic, since we already use SQLAlchemy in `migrate_pgvector.py`.

## Planned Schema Additions

1. **conversations** table
   - `session_id UUID PRIMARY KEY`
   - `created_at TIMESTAMP`
   - `last_message_at TIMESTAMP`
   - `metadata JSONB`

2. **messages** table
   - `id SERIAL PRIMARY KEY`
   - `session_id UUID REFERENCES conversations`
   - `role TEXT` (`user` | `assistant` | `system`)
   - `content TEXT`
   - `created_at TIMESTAMP`
   - `latency_ms INT`
   - `cached BOOLEAN`

3. **feedback** table
   - `id SERIAL PRIMARY KEY`
   - `message_id INT REFERENCES messages`
   - `rating INT` (-1, 0, +1)
   - `comment TEXT`
   - `created_at TIMESTAMP`

4. **customers** table (future)
   - `id UUID PRIMARY KEY`
   - `email TEXT UNIQUE`
   - `name TEXT`
   - `preferences JSONB`

## Seed Data

- `seeds/orders.sql` — mock orders for local development
- `seeds/policies.sql` — policy chunks (if moving away from file-based ingestion)

## Running Migrations (Future)

```bash
cd infra/database
alembic upgrade head
```

## Notes

- Keep vector collections (pgvector) separate from Alembic-managed tables where possible to avoid conflicts with LangChain's internal schema.
- Back up embeddings before any major pgvector upgrade.
