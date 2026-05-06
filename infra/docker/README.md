# Docker Infrastructure (Not Yet Implemented)

## Overview

This directory will contain Docker configurations for local development and production deployment of the full application stack.

## Planned Services

### `docker-compose.yml`

| Service | Image | Purpose |
|---------|-------|---------|
| `postgres` | `pgvector/pgvector:pg16` | Primary database + vector store |
| `backend` | Build from `apps/backend/Dockerfile` | FastAPI application |
| `frontend` | Build from `apps/frontend/Dockerfile` | Static site or SSR server |
| `redis` | `redis:7-alpine` | Session cache, rate limiting |

### `docker-compose.override.yml` (dev)

- Volume mounts for hot-reload
- Exposed ports for direct debugging
- Local `.env` file support

## Planned Files

```
infra/docker/
├── docker-compose.yml
├── docker-compose.override.yml
├── docker-compose.prod.yml
├── backend/
│   └── Dockerfile
├── frontend/
│   └── Dockerfile
└── README.md
```

## Backend Dockerfile (Draft)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY apps/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/backend/ .
ENV PYTHONPATH=/app

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Notes

- Use multi-stage builds for frontend to keep final image small.
- Consider `docker-compose.watch` (Docker Compose 2.23+) for even faster dev loops.
- Production should use an external managed PostgreSQL rather than containerized DB.
