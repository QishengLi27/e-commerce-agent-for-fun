# Shared Package (Not Yet Implemented)

## Overview

This package will contain types, schemas, and utilities shared between the **backend** and **frontend** (and any future services).

In a monorepo, keeping contracts in one place prevents drift between what the API promises and what the client expects.

## Planned Contents

### 1. TypeScript Type Definitions

```ts
// types/chat.ts
export interface ChatRequest {
  session_id: string;
  message: string;
}

export interface ChatResponse {
  session_id: string;
  response: string;
  sources: Source[];
  cached: boolean;
  latency_ms: number;
}

export interface Source {
  type: "policy" | "order";
  id: string;
  snippet: string;
}
```

### 2. Pydantic Models (Python)

```python
# schemas/chat.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[Source]
    cached: bool
    latency_ms: int
```

> **Note**: In the future these could be generated from a single source of truth (e.g., OpenAPI spec or protobuf) rather than manually maintained in two languages.

### 3. Shared Constants

- API version prefix (`/v1`)
- Default pagination limits
- Feature flags (for gradual rollouts)

## Integration

### Backend

```python
from shared.schemas.chat import ChatRequest, ChatResponse
```

### Frontend

```typescript
import type { ChatRequest, ChatResponse } from "@ecommerce/shared";
```

## Build / Publish

Initially this can be a local package referenced via workspace/monorepo tooling (e.g., pnpm workspaces, npm workspaces, or Python `PYTHONPATH`). No need to publish to npm/PyPI until there are external consumers.

## Files to Create

```
packages/shared/
├── ts/
│   ├── types/
│   │   ├── chat.ts
│   │   ├── memory.ts
│   │   └── metrics.ts
│   └── index.ts
├── py/
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   └── memory.py
│   │   └── constants.py
│   └── pyproject.toml
└── README.md
```
