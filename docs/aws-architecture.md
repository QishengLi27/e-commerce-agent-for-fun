# Production AWS Architecture — E-Commerce AI Support Agent

Designed for an AWS Solutions Architect interview: every component choice has a reason, and every reason maps to a pillar of the Well-Architected Framework.

---

## Architecture Diagram

```
                              ┌──────────────────────────────┐
                              │       Amazon CloudFront       │
                              │    (CDN + WAF at edge)        │
                              └──────────────┬───────────────┘
                                             │
                              ┌──────────────┴───────────────┐
                              │   Application Load Balancer   │
                              │   (HTTPS termination,         │
                              │    health checks, routing)    │
                              └──────────────┬───────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
            ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
            │   Public AZ    │        │   Public AZ    │        │   Public AZ    │
            │                │        │                │        │                │
            │ ┌────────────┐ │        │ ┌────────────┐ │        │ ┌────────────┐ │
            │ │   NAT GW   │ │        │ │   NAT GW   │ │        │ │   NAT GW   │ │
            │ └────────────┘ │        │ └────────────┘ │        │ └────────────┘ │
            │                │        │                │        │                │
            │ ┌────────────┐ │        │ ┌────────────┐ │        │ ┌────────────┐ │
            │ │ ECS Fargate│ │        │ │ ECS Fargate│ │        │ │ ECS Fargate│ │
            │ │ ┌────────┐ │ │        │ │ ┌────────┐ │ │        │ │ ┌────────┐ │ │
            │ │ │FastAPI  │ │ │        │ │ │FastAPI  │ │ │        │ │ │FastAPI  │ │ │
            │ │ │32 tasks│ │ │        │ │ │32 tasks│ │ │        │ │ │32 tasks│ │ │
            │ │ └────────┘ │ │        │ │ └────────┘ │ │        │ │ └────────┘ │ │
            │ └────────────┘ │        │ └────────────┘ │        │ └────────────┘ │
            └───────┬────────┘        └───────┬────────┘        └───────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                 Private Subnets                 │
                    │                        │                        │
            ┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
            │  Aurora PG   ││        │  ElastiCache  ││        │   Bedrock     │
            │  (pgvector)  ││        │    Redis      ││        │  (Claude)     │
            │  writer+     ││        │  (semantic    ││        │  or SageMaker │
            │  reader x2   ││        │   cache +     ││        │  (self-hosted)│
            │              ││        │   session)    ││        │               │
            └──────────────┘│        └──────────────┘│        └──────────────┘
                            │                        │
                            │  ┌──────────────┐      │
                            │  │   Secrets    │      │
                            │  │   Manager    │      │
                            │  │  (API keys)  │      │
                            │  └──────────────┘      │
                            └────────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   CloudWatch    │
                                    │  Logs, Metrics, │
                                    │  Alarms, X-Ray  │
                                    └─────────────────┘
```

---

## Component Choices — Why Each One

### Compute: ECS Fargate (not EC2, not Lambda)

| Option | Why rejected |
|--------|-------------|
| **EC2** | Managing AMIs, patching, scaling policies for a stateless API is undifferentiated heavy lifting |
| **Lambda** | LangGraph graph execution takes 1-5 seconds. LLM calls take 200ms-3s. Lambda's 15-minute timeout is fine, but cold starts add 500ms-2s on every first request. WebSocket streaming complicates Lambda. |
| **ECS Fargate** ✓ | Serverless containers. No host management. Scales on CPU/memory or request count. Graceful shutdown for in-flight requests. gunicorn + 4 uvicorn workers per task handles the async LLM workloads naturally. |

**Talking point:**
> "I chose ECS Fargate over Lambda because the agent's graph execution involves long-running LLM calls and WebSocket streaming for real-time token delivery. Fargate gives me serverless operation without cold-start penalties, and each task can handle multiple concurrent streaming connections via uvicorn's async workers."

### Database: Aurora PostgreSQL with pgvector (not RDS, not standalone)

| Option | Why rejected |
|--------|-------------|
| **RDS PostgreSQL** | Works, but Aurora gives 3x throughput on the same instance size for vector workloads, plus auto-healing storage |
| **Standalone EC2 + pgvector** | No managed backups, no Multi-AZ failover, manual patching |
| **Aurora PostgreSQL** ✓ | pgvector extension on managed Aurora. Multi-AZ with <30s failover. Storage auto-scales to 128TB. Reader instances for read-heavy retrieval workloads. |

**Key detail:** The knowledge graph (4 tables) and vector store (2 tables for langchain_pg_embedding) coexist in the same Aurora cluster. For the current scale, this is fine. At high scale, you'd separate OLTP (orders, sessions) from vector search onto different read replicas.

**Talking point:**
> "I run both the operational data and vector embeddings in Aurora PostgreSQL with the pgvector extension. For the current workload, a single cluster with read replicas handles both. At scale, I'd use separate reader endpoints — one for the transaction workload, one tuned for vector similarity search with higher `maintenance_work_mem`."

### Caching: ElastiCache Redis (two use cases)

```
Use case 1 — Semantic cache:  query_embedding → cached_response
                               (replaces the current pgvector semantic_cache table)
                               TTL: 1 hour for policies, 5 min for order status
                               Eliminates ~40% of LLM calls. ~1ms lookup vs ~5ms pgvector.

Use case 2 — Session store:   session_id → conversation_history
                               (replaces the JSON file / PG conversations table)
                               TTL: 24 hours. Auto-expiry. No cleanup job needed.
```

**Talking point:**
> "I use ElastiCache Redis for two caching layers. First, a semantic cache keyed on query embedding — exact queries or near-duplicates hit Redis in <1ms instead of making an LLM call. Second, session state for resumable conversations with TTL-based expiry. I chose Redis over DAX because I need data structures like sorted sets for conversation ordering, not just key-value."

### AI/ML: Bedrock (primary) or SageMaker (for custom models)

```
Option A — Amazon Bedrock (recommended):
  ┌──────────────────────────────────────┐
  │  Bedrock                              │
  │  ├── Claude 3.5 Haiku  (intent, validation, re-rank)  │
  │  ├── Claude 3.5 Sonnet (complex multi-step)           │
  │  └── Titan Embeddings v2 (product embeddings)         │
  │                                      │
  │  Why: No infrastructure. Private —  │
  │  data doesn't leave AWS. Pay-per-    │
  │  token. HIPAA eligible.              │
  └──────────────────────────────────────┘

Option B — SageMaker (for fine-tuned models):
  ┌──────────────────────────────────────┐
  │  SageMaker endpoint                   │
  │  ├── Fine-tuned intent classifier    │
  │  │   (distilled from Claude on your  │
  │  │    labeled query dataset)         │
  │  └── Deployed on Inferentia2 for     │
  │      cost optimization ($/token)     │
  └──────────────────────────────────────┘
```

**Talking point:**
> "I'd use Bedrock with Claude as the primary LLM because it keeps data within AWS (no third-party API), provides model access via IAM (no API key rotation), and supports the OpenAI-compatible protocol via the Converse API. For the intent classifier specifically, if query volume justified it, I'd fine-tune a smaller model on SageMaker — a distilled classifier handling 80% of queries at 10% of the cost per token."

### Networking & Security

```
┌─────────────────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                                  │
│                                                     │
│  Public subnets:     ALB + NAT Gateways             │
│  Private subnets:    ECS tasks + Aurora + Redis     │
│                                                     │
│  Security layers:                                    │
│  ├── WAF on CloudFront: rate limiting, SQLi, XSS    │
│  ├── ALB: HTTPS only, TLS 1.2+, security headers   │
│  ├── Security Groups: ECS→Aurora:5432 only          │
│  ├── Secrets Manager: API keys, DB credentials      │
│  ├── IAM: ECS task role → Bedrock invoke,           │
│  │         S3 read (policies, prompts)              │
│  └── VPC Endpoints: S3, Bedrock, Secrets Manager    │
│       (traffic never leaves AWS backbone)           │
└─────────────────────────────────────────────────────┘
```

**Talking point:**
> "ECS tasks run in private subnets with no direct internet access. They reach Bedrock and S3 through VPC endpoints — traffic stays on the AWS backbone. The only public-facing component is the ALB, which terminates TLS and is protected by WAF rules for rate limiting and prompt injection patterns."

### Observability Stack

```
CloudWatch Logs      ← structured JSON logs from structlog
CloudWatch Metrics   ← request count, p50/p99 latency, LLM call count, cache hit rate
CloudWatch Alarms    ← p99 > 3s, error rate > 1%, LLM cost/day > $50
X-Ray                ← trace across ECS→Aurora→Bedrock→Redis
                       (each LangGraph node = an X-Ray subsegment)
```

**Talking point:**
> "I use X-Ray for distributed tracing — each LangGraph node becomes an X-Ray subsegment, so I can see the exact latency breakdown: sanitize_input (34ms) → classify_intent (0.2ms) → policy_retrieve (210ms) → generate_reply (890ms). CloudWatch alarms fire on p99 latency exceeding 3 seconds or per-day LLM spend exceeding the budget."

### CI/CD Pipeline

```
GitHub → CodePipeline
           ├── Source stage: GitHub webhook
           ├── Build stage: CodeBuild (Docker build + pytest + eval suite)
           │                 Tests run against a test Aurora cluster
           │                 RAGAS evaluation gates the deployment
           ├── Approve stage: Manual approval (for prod)
           └── Deploy stage: CodeDeploy (rolling update, min 100% healthy)
                              ECS service update with circuit breaker
```

**Talking point:**
> "CI/CD runs the full RAGAS evaluation suite as a gate — if faithfulness drops below 0.8, the deployment is blocked. CodeDeploy does rolling updates on ECS, keeping at least one task healthy during the deploy. If the new version's health checks fail, the ECS circuit breaker rolls back automatically."

---

## Cost Optimization Strategy

| Tier | Compute | Database | LLM | Monthly (est.) |
|------|---------|----------|-----|---------------|
| Dev (current) | 1 Fargate task (0.5 vCPU) | db.t4g.medium | GLM-4-Flash (~$0.001/call) | ~$80 |
| Prod (light) | 3 tasks × 1 vCPU | db.r6g.large + reader | Bedrock Claude Haiku | ~$600 |
| Prod (scale) | 6 tasks × 2 vCPU Graviton | db.r6g.xlarge + 2 readers | Bedrock Haiku + fine-tuned SageMaker | ~$1,800 |

**Cost levers to discuss in an interview:**
- **Graviton processors** for ECS tasks — 20% cheaper, 30% better perf/watt for Python
- **Savings Plans** for ECS Fargate — 30-50% off for 1-year commitment
- **Reserved Instances** for Aurora — 40% off for 1-year
- **Bedrock model selection** — Claude Haiku for classification/generation, Claude Sonnet only for complex multi-step
- **Semantic cache** — eliminating 40% of identical/near-identical LLM calls
- **Fine-tuned intent classifier** — small model handles 80% of queries at fraction of cost

---

## Scaling Levers (When Load Increases)

| Signal | What to scale | How |
|--------|--------------|-----|
| API latency increasing | ECS tasks | Service auto-scaling on CPU (target 70%) |
| Vector search slowing | Aurora reader | Add read replicas, route vector queries to dedicated reader |
| LLM calls growing | ElastiCache | Increase semantic cache TTL, broaden similarity threshold |
| Session storage growing | Redis | Increase node size or shard by session_id |
| Bedrock throttling | Bedrock | Request higher throughput quota, consider SageMaker endpoint |
| Policy data growing | Aurora storage | Already auto-scales to 128TB, add pgvector HNSW index |

---

## Migration Path from Current Prototype

```
Current:                                    Phase 1 (week 1-2):
  MacBook local                               AWS dev account
  ├── FastAPI (localhost:8000)                ├── ECS Fargate (1 task, 0.5 vCPU)
  ├── Docker pgvector (localhost:5432)  ──→   ├── Aurora PostgreSQL (db.t4g.medium)
  ├── Zhipu GLM-4 API                         ├── Bedrock Claude Haiku
  └── JSON file memory store                  └── ElastiCache Redis (cache.t4g.micro)

Phase 2 (week 3-4):                       Phase 3 (week 5-6):
  Add WAF + CloudFront                      Multi-AZ Aurora with reader
  Add VPC endpoints                         Auto-scaling ECS (min 2, max 6)
  Add X-Ray tracing                          CI/CD pipeline with eval gates
  Add Secrets Manager                        CloudWatch dashboards + alarms
```

---

## Interview Questions This Architecture Handles

**"Walk me through how a request flows."**
> "Request hits CloudFront → WAF inspects it → ALB terminates TLS → ECS Fargate task in private subnet processes it. The LangGraph agent runs: it checks Redis for a semantic cache hit, queries Aurora pgvector for policy retrieval, calls Bedrock for LLM generation, and stores conversation state back in Redis with a session TTL. Every step emits X-Ray subsegments so we can trace latency end-to-end."

**"How do you handle a region failure?"**
> "Aurora has cross-Region read replicas for disaster recovery. ECS task definitions and container images are in ECR which supports cross-Region replication. ElastiCache can use Global Datastore. Route 53 failover routing switches to the DR region. RPO for Aurora cross-Region is typically under 1 second with PostgreSQL logical replication."

**"How do you secure API keys and credentials?"**
> "Nothing is stored in code or environment variables on the task. API keys for Bedrock are managed via IAM roles on the ECS task — no key at all. Database credentials are in Secrets Manager with automatic rotation. The Zhipu API key (if still needed for embeddings) is in Secrets Manager, accessed via the AWS SDK at runtime."

**"What's your most cost-effective optimization?"**
> "The semantic cache in Redis. It eliminates ~40% of LLM calls for repeated or near-identical queries. At $0.00025 per 1K tokens for Claude Haiku, saving 4,000 LLM calls per day is $1/day in LLM costs for a $0.03/day Redis instance. That's a 30:1 ROI. The fine-tuned intent classifier on SageMaker is the second — handling 80% of classifications at 10% of the per-token cost."

**"Why not just use API Gateway + Lambda?"**
> "Lambda cold starts add 500ms-2s on first request, and WebSocket streaming for real-time LLM token delivery is complex to manage. The agent's graph execution runs 1-5 seconds — Lambda's 15-minute timeout is technically fine, but the cold start penalty on every new user session degrades the experience. Fargate gives consistent latency with zero cold starts."

---

## AWS Services Map

| Your component | AWS service | Why this one |
|---------------|-------------|-------------|
| FastAPI | ECS Fargate | Serverless containers, no cold starts |
| PostgreSQL + pgvector | Aurora PostgreSQL | Managed, Multi-AZ, auto-scaling, pgvector extension |
| Semantic cache + Session store | ElastiCache Redis | Sub-ms latency, data structures, TTL auto-expiry |
| LLM (ChatOpenAI) | Bedrock (Claude) | Private API, IAM auth, no key management |
| Embeddings | Bedrock (Titan) | Same as above |
| store_policies.txt | S3 | Cheap, durable, versioned |
| Prompts | S3 + AppConfig | Version-controlled, canary deployments for A/B |
| Memory store JSON | ElastiCache Redis | Session-scoped, TTL-based, concurrent-safe |
| Docker | ECR | Private image registry, vulnerability scanning |
| .env secrets | Secrets Manager | Auto-rotation, IAM-restricted access, audit trail |
| Health checks | ALB target group health checks | Automatic task replacement on failure |
| Logs | CloudWatch Logs | Structured JSON, metric filters, subscription |
| Tracing | X-Ray | Per-node subsegments, service map |
| CI/CD | CodePipeline + CodeBuild + CodeDeploy | Managed, ECS rolling updates with circuit breaker |
