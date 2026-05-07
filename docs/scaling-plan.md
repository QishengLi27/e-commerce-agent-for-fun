# Production Scaling Plan

## Current State (Demo / Prototype)

| Component | Current | Limitation |
|-----------|---------|------------|
| API Server | Single Uvicorn process, 1 worker | Blocks on LLM calls; ~10-20 concurrent users max |
| Agent | Synchronous `run_agent_with_cache()` | 1-3s latency per request; no request pipelining |
| Memory | JSON file (`memory_store.json`) | Race conditions under concurrency; no TTL; scales to ~100 sessions |
| Cache | pgvector semantic cache | DB-dependent; slow for simple KV lookups |
| Database | Single Postgres container | No failover; no read replicas |
| Frontend | Vite dev server | Not optimized for production; no CDN |
| Observability | Console logs only | Blind to errors, latency spikes, cache hit rates |
| Security | No rate limiting | Vulnerable to abuse and runaway costs |

---

## Phase 1: Quick Wins (Week 1-2)

**Goal: Go from ~10 concurrent users to ~100.**

### 1.1 Async Agent Execution

**Problem:** `run_agent_with_cache()` is fully synchronous. While waiting for the LLM (1-3s), the entire worker thread is blocked.

**Fix:**
- Make the FastAPI endpoints `async` (already done).
- Run the synchronous LangChain agent in a thread pool:

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

_executor = ThreadPoolExecutor(max_workers=4)

async def run_agent_async(user_input: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, run_agent_with_cache, user_input)
```

- Update `/chat` and `/chat/stream` to use `run_agent_async()`.

**Impact:** 4x concurrency improvement per worker process.

### 1.2 Multiple Uvicorn Workers

**Problem:** Single process can't use multiple CPU cores.

**Fix:**
```bash
# Use Gunicorn with Uvicorn workers
pip install gunicorn

gunicorn backend.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60
```

**Impact:** Scales across CPU cores. 4 workers × 4 threaded agents = ~16 concurrent LLM requests.

### 1.3 PostgreSQL Connection Pooling

**Problem:** Each request opens a new DB connection.

**Fix:**
- Add `pgbouncer` (connection pooler) or use SQLAlchemy's built-in pooling.
- LangChain's `PGVector` uses SQLAlchemy under the hood. Configure `pool_size=10`, `max_overflow=20`.

### 1.4 Redis for Session Memory

**Problem:** JSON file is not safe for concurrent writes.

**Fix:**
```bash
pip install redis
```

Replace `MemoryStore` with Redis-backed session store:
- Key: `session:{session_id}`
- Value: JSON list of messages
- TTL: 24 hours

**Impact:** Eliminates file I/O bottleneck; enables horizontal scaling across multiple backend instances.

### 1.5 Rate Limiting

**Problem:** No protection against abuse.

**Fix:** Use `slowapi` (Redis-backed rate limiter for FastAPI):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/chat")
@limiter.limit("10/minute")
def chat(request: ChatRequest):
    ...
```

**Impact:** Prevents runaway API costs and ensures fair usage.

---

## Phase 2: Caching & Latency (Week 3-4)

**Goal: Sub-100ms for repeat questions; reduce LLM costs by 60-80%.**

### 2.1 Two-Tier Cache Architecture

```
User Query
    ├──> L1: Redis Exact Match (O(1)) ──> Return instantly
    └──> L2: pgvector Semantic Cache ──> Return if similarity < 0.2
        └──> L3: LLM Inference ──> Store in L1 + L2
```

**L1: Redis Exact Match Cache**
- Hash the normalized query string.
- TTL: 1 hour for dynamic answers (order status), 24 hours for static answers (policies).

**L2: pgvector Semantic Cache (existing)**
- Keep as-is but add TTL cleanup job.

### 2.2 BM25 Index Warmup

**Problem:** First request after server restart rebuilds the BM25 index (~2s delay).

**Fix:** Build the BM25 index at startup and cache it in Redis. Serve from memory on every request.

### 2.3 Streaming by Default

**Problem:** Users wait 1-3s before seeing anything.

**Fix:** Make `/chat/stream` the default frontend path. Users see words appear immediately (perceived latency drops to ~50ms).

---

## Phase 3: Horizontal Scaling (Week 5-8)

**Goal: 1,000+ concurrent users.**

### 3.1 Containerization

```
infra/docker/
├── docker-compose.yml          # Local dev
├── docker-compose.prod.yml     # Production
├── backend/Dockerfile
├── frontend/Dockerfile
└── nginx/nginx.conf
```

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "backend.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 3.2 Load Balancer (Nginx)

```nginx
upstream backend {
    least_conn;
    server backend-1:8000;
    server backend-2:8000;
    server backend-3:8000;
}

server {
    location /api {
        proxy_pass http://backend;
        proxy_buffering off;  # Required for SSE streaming
    }
}
```

**Key setting:** `proxy_buffering off;` — essential for SSE to reach the client immediately.

### 3.3 Managed PostgreSQL

Move from Docker Postgres to a managed service:
- **AWS RDS PostgreSQL + pgvector** (or **Supabase**, **Neon**)
- Enable read replicas for vector search offloading
- Automated backups

### 3.4 Managed Redis

- **AWS ElastiCache**, **Upstash**, or **Redis Cloud**
- Use for: session memory, rate limiting, exact-match cache

### 3.5 Auto-Scaling

```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## Phase 4: LLM Cost & Reliability (Week 6-10)

**Goal: 99.9% uptime; cut LLM costs by 50%.**

### 4.1 LLM Fallback Chain

```python
LLM_PROVIDERS = [
    ("primary", ChatOpenAI(model="glm-4-flash", ...)),      # Fast, cheap
    ("fallback", ChatOpenAI(model="glm-4", ...)),            # Better quality
    ("emergency", ChatOpenAI(model="gpt-3.5-turbo", ...)),   # Different provider
]
```

If primary fails (circuit breaker opens), automatically fall back to the next tier.

### 4.2 Prompt Caching

Many questions are similar. Cache the final rendered prompt + LLM response in Redis with a 5-minute TTL.

### 4.3 Batching (Future)

If traffic spikes, batch multiple similar queries into a single LLM call using prompt templating:

```
Answer the following questions:
1. {query_1}
2. {query_2}
3. {query_3}
```

**Note:** This requires significant architectural changes. Consider in Phase 5.

### 4.4 Response Pre-computation

For the top 50 most common questions, pre-generate answers nightly and store them in Redis. Serve instantly.

---

## Phase 5: Frontend at Scale (Week 4-6)

### 5.1 Static Hosting + CDN

Build the React app and host on:
- **Vercel** (easiest)
- **AWS CloudFront + S3**
- **Cloudflare Pages**

```bash
cd apps/frontend
npm run build
# Upload dist/ folder to CDN
```

### 5.2 API Route Separation

Separate read and write traffic:
- `GET /api/health` — edge cached
- `GET /api/memory/{session_id}` — serve from Redis directly
- `POST /api/chat` — hit the LLM backend

### 5.3 WebSocket for High-Frequency Users

For users who send many messages, upgrade to WebSocket to avoid HTTP handshake overhead on every message.

---

## Phase 6: Observability (Week 2-ongoing)

### 6.1 Structured Logging

Replace `print()` with JSON logs:
```python
logger.info("agent_response", extra={
    "session_id": session_id,
    "latency_ms": latency,
    "cached": cached,
    "tool_used": tool_name,
})
```

### 6.2 Metrics to Track

| Metric | Target | Alert If |
|--------|--------|----------|
| p99 Latency | < 2s | > 3s |
| Cache Hit Rate | > 60% | < 40% |
| LLM Error Rate | < 1% | > 5% |
| DB Connection Pool | < 80% | > 90% |
| Active Sessions | — | Spike > 3x avg |

### 6.3 Tools

- **Prometheus + Grafana** — metrics dashboards
- **Sentry** — error tracking
- **LangSmith** — LLM tracing and prompt debugging

---

## Estimated Infrastructure Cost (1,000 DAU)

| Component | Service | Est. Monthly Cost |
|-----------|---------|-------------------|
| Backend | 2-4 AWS ECS Fargate tasks | $80-150 |
| Database | AWS RDS db.t3.medium + pgvector | $60-100 |
| Cache | Upstash Redis (1GB) | $20 |
| LLM API | GLM-4-Flash (~$0.001/1K tokens) | $50-200 |
| CDN | CloudFront / Vercel | $0-20 |
| Monitoring | Grafana Cloud + Sentry | $30 |
| **Total** | | **~$240-520/month** |

---

## Recommended Execution Order

**Week 1:**
1. Async agent + thread pool
2. Gunicorn with 4 workers
3. Rate limiting

**Week 2:**
4. Redis for session memory
5. Two-tier cache (Redis exact match)
6. Structured logging

**Week 3:**
7. Docker + docker-compose.prod.yml
8. Nginx load balancer with SSE support

**Week 4:**
9. Frontend CDN deployment
10. Managed Postgres + Redis

**Week 5+:**
11. Kubernetes + HPA
12. LLM fallback chain
13. LangSmith integration
