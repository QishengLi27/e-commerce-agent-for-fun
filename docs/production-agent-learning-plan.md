# Production-Ready AI Agent — Learning & Implementation Plan

This plan turns your prototype into a production-quality agent. Each module is a self-contained learning project that teaches a specific production pattern. Implementation order is designed so each step builds on the previous ones.

---

## Learning Path Overview

```
Week 1-2:  Streaming       — Real token-level SSE streaming
Week 3-4:  Observability   — Tracing, structured logging, metrics
Week 5-6:  Session Memory  — Per-user conversation isolation
Week 7-8:  Multi-Agent     — Specialized sub-agents with handoffs
Week 9+:   Polish          — Prompt mgmt, cost tracking, security
```

---

## Module 1: Real Token-Level Streaming

**Why this matters for interviews:** 90% of candidates claim streaming works. <5% can explain how `astream_events()` actually traverses their graph. Implementing real streaming demonstrates you understand the event protocol, not just the API surface.

### What's broken now

```python
# Current fake streaming in routes.py:
result = await loop.run_in_executor(None, agent_graph.invoke, state)
for word in result["final_answer"].split(" "):
    yield word + " "
    await asyncio.sleep(0.03)  # Artificial delay after the fact
```

The user waits for the full graph to complete, then sees tokens replayed with artificial pauses.

### What you'll implement

1. **Graph-level streaming** with `agent_graph.astream()` instead of `invoke()`
2. **Token-level streaming** from the LLM inside `generate_reply` via `astream_events()`
3. **SSE endpoint** that yields tokens as they arrive from the LLM

### Implementation steps

**Step 1: Switch the streaming endpoint to use `astream_events()`**

```python
# routes.py — new /chat/stream endpoint
async def stream_chat(request: ChatRequest):
    initial_state = {"user_input": request.query}

    async def event_generator():
        async for event in agent_graph.astream_events(
            initial_state,
            config={"configurable": {"thread_id": request.session_id}},
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                yield f"data: {json.dumps({'token': token})}\n\n"
            elif kind == "on_chain_end" and event["name"] == "update_memory":
                final = event["data"]["output"].get("final_answer", "")
                yield f"data: {json.dumps({'done': True, 'final_answer': final})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Step 2: Add tool call events to the stream**

```python
if kind == "on_tool_start":
    yield f"data: {json.dumps({'tool_call': event['name'], 'input': event['data']['input']})}\n\n"
elif kind == "on_tool_end":
    yield f"data: {json.dumps({'tool_result': event['name'], 'output': str(event['data']['output'])})}\n\n"
```

This gives the frontend real-time visibility into what the agent is doing — which tool it's calling and what the result is. This is a differentiator in interviews.

**Step 3: Add the `messages` stream mode to LangGraph**

```python
# Use LangGraph's built-in message streaming
async for namespace, mode, data in agent_graph.astream(
    initial_state, config, stream_mode=["messages", "values"]
):
    if mode == "messages":
        token, metadata = data
        yield f"data: {json.dumps({'token': token.content, 'step': metadata['langgraph_step']})}\n\n"
    elif mode == "values":
        yield f"data: {json.dumps({'state_update': data})}\n\n"
```

**Interview talking point:**
> "I replaced the fake streaming with `agent_graph.astream_events()`, which intercepts the LLM's token stream and tool execution events in real time. The frontend sees tokens as the LLM generates them and knows which tool the agent is calling before the result comes back. This is critical for UX — a 3-second wait with a spinner is a bad experience; streaming each token keeps the user engaged."

---

## Module 2: Observability

**Why this matters for interviews:** Production agents fail silently. Without tracing, you can't debug. Without metrics, you can't alert. This module teaches the three pillars: tracing, structured logging, and metrics.

### What's broken now

- `httpx` INFO logs are the only observability signal
- No request IDs — can't correlate log lines to a specific user request
- No latency breakdown — can't tell if the bottleneck is the LLM, the DB, or retrieval
- No error tracking — exceptions are logged but not aggregated

### Implementation steps

**Step 1: Structured logging with `structlog`**

```bash
pip install structlog
```

```python
# backend/observability/logging.py
import structlog

logger = structlog.get_logger()

# Replace every logger.info(...) in nodes.py with:
logger.info("node_executed", node="classify_intent", intent=state["intent"], latency_ms=12.3)

# Structured output:
# {"event": "node_executed", "node": "classify_intent", "intent": "policy", "latency_ms": 12.3, "request_id": "abc-123"}
```

This makes logs queryable: `jq 'select(.node == "classify_intent")'` or `grep '"latency_ms": [5-9][0-9][0-9]'` to find slow nodes.

**Step 2: Request ID propagation**

```python
# backend/observability/middleware.py
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id")

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

Every log line and LLM call gets tagged with the request ID. When a user reports a bad answer, you find their request ID and trace the entire execution.

**Step 3: Node-level latency tracking**

```python
# In every node function, wrap with timing:
import time

def generate_reply(state: AgentState) -> AgentState:
    start = time.time()
    # ... original logic ...
    elapsed = time.time() - start
    logger.info("node_completed", node="generate_reply", latency_ms=round(elapsed * 1000, 1))
    return state

# Or better: a decorator
@trace_node("generate_reply")
def generate_reply(state: AgentState) -> AgentState:
    ...
```

Now you can answer "where is latency spent?" with data, not guesses.

**Step 4: OpenTelemetry spans (optional, high-signal)**

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
```

This gives you a trace view: `POST /chat → sanitize_input (34ms) → classify_intent (0.2ms) → policy_node (210ms, LLM call inside) → generate_reply (890ms) → validate_reply (450ms)`.

**Interview talking point:**
> "I added structured logging with `structlog`, request ID propagation, and per-node latency tracking. Every log line is a JSON object with `request_id`, `node`, and `latency_ms` fields. When a support query takes 3 seconds instead of the expected 800ms, I can grep for that request ID and immediately see which node is the bottleneck — was it the LLM, the retrieval, or a DB query?"

---

## Module 3: Per-Session Conversation Memory

**Why this matters for interviews:** A single global JSON file for all conversations is the #1 red flag for "prototype, not production." Implementing proper session isolation shows you understand multi-tenant systems.

### What's broken now

```python
# Current: one global file for all users
self.memory_store = MemoryStore(filepath="data/memory_store.json")
```

Two users talking simultaneously share the same memory. The `max_history=8` limit means the 9th message drops the 1st — across ALL users.

### Implementation steps

**Step 1: Replace JSON file with PostgreSQL `conversations` table**

```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_conversations_session ON conversations(session_id, created_at);
```

**Step 2: New `SessionMemoryStore` class**

```python
# backend/memory.py — new implementation
class SessionMemoryStore:
    def __init__(self, connection_string: str):
        self.conn_string = connection_string

    def add_message(self, session_id: str, role: str, content: str):
        with psycopg2.connect(self.conn_string) as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content)
            )

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        with psycopg2.connect(self.conn_string) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations "
                "WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                (session_id, limit)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
```

**Step 3: Pass `session_id` through the graph state**

```python
class AgentState(TypedDict):
    session_id: str          # NEW
    messages: Annotated[list, add]
    user_input: str
    # ...

# In update_memory node:
def update_memory(state: AgentState) -> AgentState:
    sid = state.get("session_id", "default")
    session_memory.add_message(sid, "user", state["user_input"])
    session_memory.add_message(sid, "agent", state["final_answer"])
    return state
```

**Step 4: Load history on graph start**

```python
# New node: load_history — runs before sanitize_input
def load_history(state: AgentState) -> AgentState:
    sid = state.get("session_id", "default")
    history = session_memory.get_recent_messages(sid)
    state["messages"] = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in history
    ]
    return state
```

**Interview talking point:**
> "I moved from a global JSON file to per-session PostgreSQL storage. Each conversation is isolated by `session_id`, indexed for efficient recent-message queries, and never cross-contaminates between users. The API accepts an optional `session_id` in the request — if provided, the agent loads that conversation's history and appends to it. If omitted, a new session is created."

---

## Module 4: Multi-Agent with Specialized Sub-Agents

**Why this matters for interviews:** Single-agent systems hit a complexity ceiling. Multi-agent architectures are the cutting edge of agent engineering. Implementing a supervisor-worker pattern demonstrates system design skills beyond "I called `create_agent()`."

### Architecture

```
                         ┌──────────────┐
                         │  Supervisor  │
                         │    Agent     │
                         └──────┬───────┘
                                │ routes to...
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
        │ Order Agent  │ │Policy Agent│ │Product Agent│
        │ (track, list)│ │(RAG, rules)│ │(KG queries) │
        └──────────────┘ └────────────┘ └─────────────┘
```

### Implementation steps

**Step 1: Extract each tool set into its own compiled sub-graph**

```python
# backend/graph/order_graph.py
order_graph = StateGraph(OrderState)
order_graph.add_node("lookup_order", order_node)
order_graph.add_node("list_orders", list_orders_node)
# ... compile as a self-contained graph

# backend/graph/policy_graph.py
policy_graph = StateGraph(PolicyState)
policy_graph.add_node("retrieve_policy", policy_node)
# ... compile
```

**Step 2: Create a supervisor graph that routes to sub-graphs**

```python
# backend/graph/supervisor_graph.py
class SupervisorState(TypedDict):
    messages: Annotated[list, add]
    next_agent: str  # "order_agent" | "policy_agent" | "product_agent" | "FINISH"

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """LLM decides which sub-agent to call next."""
    prompt = (
        "You are a routing supervisor. Given the conversation, decide which agent "
        "should handle the next step: order_agent (order tracking, listing), "
        "policy_agent (returns, shipping, warranty), product_agent (product info), "
        "or FINISH if the task is complete. Respond with just the agent name."
    )
    response = llm.invoke(prompt + f"\n\nConversation:\n{format_messages(state['messages'])}")
    state["next_agent"] = response.content.strip()
    return state

supervisor = StateGraph(SupervisorState)
supervisor.add_node("supervisor", supervisor_node)
supervisor.add_node("order_agent", order_graph.compile())    # sub-graph as a node
supervisor.add_node("policy_agent", policy_graph.compile())
supervisor.add_node("product_agent", knowledge_graph.compile())
# ... conditional routing based on next_agent
```

**Step 3: Implement the handoff pattern**

When the supervisor routes to a sub-agent, the sub-agent runs its own graph and returns results. The supervisor then decides: hand off to another agent, or finish.

```
User: "What's in my order 1001, and can I return those items?"
  → Supervisor: route to order_agent
  → Order Agent: "Order 1001 contains Headphones and a Laptop"
  → Supervisor: route to product_agent (to check return eligibility)
  → Product Agent: "Headphones (Audio): 14-day return. Laptop (Electronics): 14-day return"
  → Supervisor: FINISH
  → Final answer: "Your order has Headphones and a Laptop. Both are eligible for 14-day returns."
```

**Interview talking point:**
> "I implemented a supervisor-worker multi-agent architecture. The supervisor is an LLM-based router that decides which specialized sub-agent handles each turn. Each sub-agent is a self-contained LangGraph — the order agent has its own tools and graph, the policy agent has its own retrieval pipeline. The supervisor orchestrates handoffs between them for multi-step queries like 'what's in my order and can I return those items?'"

---

## Module 5: Prompt Management & A/B Testing

### What's broken now

All prompts are hardcoded Python strings. Changing a prompt requires a code deploy.

### Implementation

```python
# backend/prompts/registry.py
import json
from pathlib import Path

class PromptRegistry:
    def __init__(self, prompts_dir: str = "prompts"):
        self.dir = Path(prompts_dir)
        self._cache = {}

    def get(self, name: str, version: str | None = None) -> str:
        version = version or self._active_version(name)
        key = f"{name}:{version}"
        if key not in self._cache:
            path = self.dir / f"{name}__{version}.txt"
            self._cache[key] = path.read_text()
        return self._cache[key]

    def _active_version(self, name: str) -> str:
        # Read from a versions.json file
        ...

registry = PromptRegistry()
```

```
prompts/
├── generate_reply__v1.txt
├── generate_reply__v2.txt      # tested variant with "be concise" instruction
├── validate_reply__v1.txt
├── supervisor_route__v1.txt
└── versions.json               # {"generate_reply": "v2", "validate_reply": "v1"}
```

---

## Module 6: Cost Tracking & Guardrails

### What to implement

```python
# backend/observability/cost.py
from langchain_community.callbacks import get_openai_callback

class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.calls = []

    def track(self, llm_call):
        with get_openai_callback() as cb:
            result = llm_call()
            self.calls.append({
                "tokens": cb.total_tokens,
                "cost": cb.total_cost,
                "model": cb.succesful_requests,
            })
            self.total_cost += cb.total_cost
            return result

    def daily_budget_check(self, limit: float = 50.0):
        if self.total_cost > limit:
            raise BudgetExceededError(f"Daily budget of ${limit} exceeded")
```

Add a `/metrics` endpoint exposing: total tokens today, total cost today, average cost per request, cache hit rate.

---

## Recommended Implementation Order

```
Priority 1 (week 1-2):  Module 1 — Real Streaming
                         → Biggest differentiator. Fake streaming is the #1 "prototype" signal.

Priority 2 (week 3-4):  Module 2 — Observability
                         → Structured logging + request IDs + latency tracking.
                         → Prerequisite for debugging everything else.

Priority 3 (week 5-6):  Module 3 — Session Memory
                         → Per-user isolation. Demonstrates multi-tenant thinking.

Priority 4 (week 7-8):  Module 4 — Multi-Agent
                         → Supervisor-worker. Cutting-edge topic. Differentiator.

Priority 5 (week 9+):   Modules 5 & 6 — Prompts, Cost
                         → Polish. Shows engineering discipline.
```

---

## For Each Module, the Interview-Ready Summary

| Module | What to say |
|--------|------------|
| Streaming | "I use `agent_graph.astream_events()` to intercept LLM token streaming and tool execution events. Tokens arrive at the frontend as the LLM generates them — no artificial delays." |
| Observability | "Every request gets a UUID propagated through structured logs. I track per-node latency so I can pinpoint bottlenecks — is it the LLM, retrieval, or a DB query?" |
| Session Memory | "Conversations are isolated by `session_id` in PostgreSQL. No cross-contamination, no global state, and history persists across server restarts." |
| Multi-Agent | "I use a supervisor-worker pattern. The supervisor routes to specialized sub-agents — each is a self-contained LangGraph with its own tools and retrieval pipeline." |
| Prompt Mgmt | "Prompts are versioned in a registry. A/B testing a new prompt is a config change, not a deploy. I log which prompt version each response used." |
| Cost | "I track token usage and cost per request via LangChain callbacks. A daily budget guardrail prevents runaway costs." |

---

## What NOT to add (anti-patterns for this stage)

- **LangSmith/LangFuse** — great tools, but wiring them up teaches you nothing about how tracing works. Implement tracing yourself first, then graduate to a managed service.
- **Redis** — you already have PostgreSQL. Adding a second data store before you need it is premature optimization. PG can handle session memory at this scale.
- **Kubernetes** — a single Docker container with gunicorn workers is fine for a portfolio project. K8s signals "I can configure YAML," not "I understand agent architecture."
- **More tools** — 4 tools is the right number. Adding a 5th tool doesn't teach anything new. Multi-agent routing teaches more than "another @tool function."
